

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Optional, List, Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from core.util.refiner_util import sanitize_tensor


@dataclass
class TAFASConfig:
    # Output-GCM calibration block (variable-wise residual) under blackbox contract.
    hidden_dim: int = 128
    gating_init: float = 0.01
    gcm_var_wise: bool = True
    max_chunk_size: int = 100

    # Optimizer
    lr: float = 5e-3
    weight_decay: float = 1e-4

    # Replay buffer + warmup control (aligned with LinearConfig spirit).
    collect_train_windows: Optional[int] = None
    collect_val_windows: Optional[int] = None
    max_epochs: int = 100
    batch_size: int = 256
    early_stop_patience: int = 20

    # Online retrain trigger flag.
    online_training: bool = False
    update_rule: str = "ring_quartet"


class GCM(nn.Module):
    """Gated Calibration Module with zero-init residual.

    When gating is small and W/bias start at zero, GCM(x) approximately equals x,
    so the refiner starts as an identity on top of the blackbox base forecast.
    """

    def __init__(
        self,
        window_len: int,
        n_var: int = 1,
        gating_init: float = 0.01,
        var_wise: bool = True,
    ) -> None:
        super().__init__()
        self.window_len = int(window_len)
        self.n_var = int(n_var)
        self.var_wise = bool(var_wise)

        if self.var_wise:
            self.weight = nn.Parameter(torch.zeros(self.window_len, self.window_len, self.n_var))
        else:
            self.weight = nn.Parameter(torch.zeros(self.window_len, self.window_len))

        self.gating = nn.Parameter(float(gating_init) * torch.ones(self.n_var))
        self.bias = nn.Parameter(torch.zeros(self.window_len, self.n_var))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.var_wise:
            delta = torch.einsum("biv,iov->bov", x, self.weight) + self.bias
        else:
            delta = torch.einsum("biv,io->bov", x, self.weight) + self.bias
        return x + torch.tanh(self.gating).view(1, 1, -1) * delta


class OutputCalibrator(nn.Module):
    """Chunked output GCM that keeps memory bounded for high-channel datasets."""

    def __init__(
        self,
        *,
        pred_len: int,
        target_dim: int,
        hidden_dim: int,
        gating_init: float,
        var_wise: bool,
        max_chunk_size: int,
    ) -> None:
        super().__init__()
        _ = int(hidden_dim)
        self.pred_len = int(pred_len)
        self.target_dim = int(target_dim)
        self.chunk_slices: List[Tuple[int, int]] = self._build_chunk_slices(
            target_dim=int(target_dim), max_chunk_size=int(max_chunk_size)
        )
        self.chunks = nn.ModuleList(
            [
                GCM(
                    window_len=int(pred_len),
                    n_var=int(e - s),
                    gating_init=float(gating_init),
                    var_wise=bool(var_wise),
                )
                for s, e in self.chunk_slices
            ]
        )

    @staticmethod
    def _build_chunk_slices(*, target_dim: int, max_chunk_size: int) -> List[Tuple[int, int]]:
        d = int(target_dim)
        c = int(max(1, max_chunk_size))
        return [(s, min(d, s + c)) for s in range(0, d, c)]

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        out_chunks: List[torch.Tensor] = []
        for (s, e), gcm in zip(self.chunk_slices, self.chunks):
            out_chunks.append(gcm(y[:, :, s:e]))
        return torch.cat(out_chunks, dim=-1)


class OnlineRefinerTAFAS(nn.Module):
    """TAFAS-style calibration refiner adapted to the blackbox online framework.

    Compared to the original TAFAS paper, this refiner keeps the output Gated
    Calibration Module (GCM) as the only adaptation mechanism, because under the
    blackbox contract we cannot:
      - backpropagate through the frozen base forecaster (drops input GCM);
      - re-run the base forecaster with recalibrated inputs (drops adjust_pred).

    Training signal is the fully resolved GT window delivered by the online
    wrapper's ring pending queue, which is strictly aligned with the base
    prediction. The refiner therefore acts as a residual post-hoc calibrator
    fitted with the same warmup + optional online retrain schedule as Linear.
    """

    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int = 128,
        lr: Optional[float] = None,
        device: Optional[torch.device] = None,
        **kwargs,
    ) -> None:
        super().__init__()
        self.device = device or torch.device("cpu")
        self.target_dim = int(feature_dim)

        self.baseline_router = bool(kwargs.get("baseline_router", False))
        self.ema_error_momentum = float(kwargs.get("ema_error_momentum", 0.2))
        self.routing_temperature = float(kwargs.get("routing_temperature", 0.1))
        self.e_base_moving: Optional[torch.Tensor] = None
        self.e_ref_moving: Optional[torch.Tensor] = None
        self._routing_enabled = False
        self.router_raw_output: Optional[torch.Tensor] = None

        self.config = TAFASConfig(hidden_dim=int(hidden_dim))
        if lr is not None:
            self.config.lr = float(lr)
        if "collect_train_windows" not in kwargs or "collect_val_windows" not in kwargs:
            raise ValueError("TAFAS requires explicitly provided window counts.")
        self.config.collect_train_windows = max(1, int(kwargs["collect_train_windows"]))
        self.config.collect_val_windows = max(1, int(kwargs["collect_val_windows"]))

        # Apply external configurations.
        for k, v in kwargs.items():
            if hasattr(self.config, k):
                setattr(self.config, k, v)
        self.config.lr = max(1e-12, float(self.config.lr))
        self.config.weight_decay = max(0.0, float(self.config.weight_decay))

        self.is_initialized = False
        self.model: Optional[OutputCalibrator] = None
        self.optimizer: Optional[torch.optim.AdamW] = None

        self.expected_H: Optional[int] = None
        self.expected_L: Optional[int] = None
        self.replay_buffer: List[Dict] = []

        self.is_warmed_up: bool = False
        self.loss_history: List[List[float]] = []
        self.val_loss_history: List[List[float]] = []
        self._snapshot_idx: int = 0

    def reset_state(self, *, clear_loss_history: bool = True) -> None:
        self.expected_H = None
        self.expected_L = None
        self.replay_buffer.clear()
        self.is_warmed_up = False
        self._snapshot_idx = 0
        self.e_base_moving = None
        self.e_ref_moving = None
        self._routing_enabled = False
        self.router_raw_output = None
        if clear_loss_history:
            self.loss_history = []
            self.val_loss_history = []

    def _ensure_optimizer(self) -> None:
        if self.model is None:
            raise RuntimeError("Model must be initialized before optimizer creation.")
        if self.optimizer is None:
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=float(self.config.lr),
                weight_decay=float(self.config.weight_decay),
            )

    @staticmethod
    def _median_samples(y: torch.Tensor) -> torch.Tensor:
        if y.ndim == 2:
            return y.unsqueeze(0)
        if y.ndim == 3:
            if y.shape[0] == 1:
                return y
            return torch.median(y, dim=0, keepdim=True)[0]
        raise ValueError(f"Unsupported tensor shape for median reduction: {tuple(y.shape)}")

    def _prepare_batch(self, batch: List[Dict]) -> Tuple[torch.Tensor, torch.Tensor]:
        padded_Y_median: List[torch.Tensor] = []
        padded_Y_GT: List[torch.Tensor] = []
        for item in batch:
            padded_Y_median.append(item["Y_base_median"])
            padded_Y_GT.append(item["Y_GT"])
        Y_median = torch.cat(padded_Y_median, dim=0)
        Y_GT = torch.cat(padded_Y_GT, dim=0)
        return Y_median, Y_GT

    def update(
        self,
        X: torch.Tensor,
        Y_base: torch.Tensor,
        Y_ref_past: torch.Tensor,
        Y_GT_full: torch.Tensor,
        *,
        physical_stride: Optional[int] = None,
        step: int = 10,
        e_past_override: Optional[torch.Tensor] = None,
    ) -> List[float]:
        _ = physical_stride
        _ = step
        _ = e_past_override

        if Y_GT_full.shape[1] == 0:
            return []

        X = sanitize_tensor(X).detach().to(self.device)
        Y_base = sanitize_tensor(Y_base).detach().to(self.device)
        Y_ref_past = sanitize_tensor(Y_ref_past).detach().to(self.device)
        Y_GT_full = sanitize_tensor(Y_GT_full).detach().to(self.device)

        if Y_GT_full.ndim == 3 and Y_GT_full.shape[0] > 1:
            Y_GT_full = Y_GT_full[0:1]

        if self.expected_H is None and Y_GT_full.shape[1] > 0:
            self.expected_H = int(Y_GT_full.shape[1])
        if self.expected_L is None and X.ndim >= 2:
            self.expected_L = int(X.shape[1] if X.ndim == 3 else X.shape[0])

        Y_base_median = self._median_samples(Y_base)
        if self.baseline_router:
            y_ref_past_median = self._median_samples(Y_ref_past)
            if y_ref_past_median.shape != Y_base_median.shape:
                y_ref_past_median = Y_base_median
            err_base = torch.mean(torch.abs(Y_GT_full - Y_base_median), dim=(0, 1))
            err_ref = torch.mean(torch.abs(Y_GT_full - y_ref_past_median), dim=(0, 1))
            alpha = float(self.ema_error_momentum)
            if self.e_base_moving is None or self.e_ref_moving is None:
                self.e_base_moving = err_base.detach()
                self.e_ref_moving = err_ref.detach()
            else:
                self.e_base_moving = alpha * err_base + (1.0 - alpha) * self.e_base_moving
                self.e_ref_moving = alpha * err_ref + (1.0 - alpha) * self.e_ref_moving
            self._routing_enabled = True

        current_snap: Dict[str, torch.Tensor] = {
            "Y_base_median": sanitize_tensor(Y_base_median.clone()),
            "Y_GT": sanitize_tensor(Y_GT_full.clone()),
            "t_idx": int(self._snapshot_idx),
        }
        self._snapshot_idx += 1
        self.replay_buffer.append(current_snap)

        gap_windows = int(self.expected_H or 0)
        if self.config.online_training:
            target_collection_size = int(self.config.collect_train_windows)
        else:
            target_collection_size = int(self.config.collect_train_windows) + gap_windows + int(self.config.collect_val_windows)

        if (not self.is_warmed_up or self.config.online_training) and target_collection_size > 0:
            collected = int(len(self.replay_buffer))
            if collected % max(1, target_collection_size // 20) == 0 or collected == target_collection_size:
                print(f"[Refined][TAFAS] Snapshot collection: {collected}/{target_collection_size}", flush=True)

        should_train_now = (
            target_collection_size > 0
            and len(self.replay_buffer) >= target_collection_size
            and (not self.is_warmed_up or self.config.online_training)
        )
        if not should_train_now:
            return []

        print("\n[Refined][TAFAS] ====== BUFFER FULL. INITIATING SNAPSHOT TRAINING ======", flush=True)

        if not self.is_initialized:
            self.model = OutputCalibrator(
                pred_len=int(self.expected_H or 1),
                target_dim=int(self.target_dim),
                hidden_dim=int(self.config.hidden_dim),
                gating_init=float(self.config.gating_init),
                var_wise=bool(self.config.gcm_var_wise),
                max_chunk_size=int(self.config.max_chunk_size),
            ).to(self.device)
            self.is_initialized = True

        self._ensure_optimizer()

        if self.config.online_training:
            online_total = int(len(self.replay_buffer))
            val_size = max(1, int(round(0.1 * float(online_total))))
            if val_size >= online_total:
                val_size = max(1, online_total - 1)
            train_data = self.replay_buffer[: max(1, online_total - val_size)]
            val_data = self.replay_buffer[max(1, online_total - val_size) : online_total]
        else:
            train_end = int(self.config.collect_train_windows)
            val_start = int(train_end + gap_windows)
            val_end = int(val_start + int(self.config.collect_val_windows))
            train_data = self.replay_buffer[:train_end]
            val_data = self.replay_buffer[val_start:val_end]

        if not val_data:
            val_data = train_data[-1:]

        best_val_loss = float("inf")
        best_model_state: Optional[Dict[str, torch.Tensor]] = None
        patience_counter = 0

        for param in self.model.parameters():
            param.requires_grad = True

        for epoch in range(int(self.config.max_epochs)):
            self.model.train()
            epoch_train_data = list(train_data)
            epoch_train_loss = 0.0
            num_batches = 0

            for i in range(0, len(epoch_train_data), int(self.config.batch_size)):
                batch = epoch_train_data[i : i + int(self.config.batch_size)]
                Y_median_b, Y_GT_b = self._prepare_batch(batch)
                y_pred = sanitize_tensor(self.model(Y_median_b))
                loss = F.mse_loss(y_pred, Y_GT_b)
                loss = torch.nan_to_num(loss, nan=1e6, posinf=1e6, neginf=1e6)

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

                epoch_train_loss += float(loss.item())
                num_batches += 1

            avg_train_loss = epoch_train_loss / max(1, num_batches)
            self.loss_history.append([float(avg_train_loss)])

            self.model.eval()
            epoch_val_loss = 0.0
            num_val_batches = 0
            with torch.no_grad():
                for i in range(0, len(val_data), int(self.config.batch_size)):
                    batch = val_data[i : i + int(self.config.batch_size)]
                    Y_median_b, Y_GT_b = self._prepare_batch(batch)
                    y_pred = sanitize_tensor(self.model(Y_median_b))
                    v_loss = F.mse_loss(y_pred, Y_GT_b)
                    epoch_val_loss += float(v_loss.item())
                    num_val_batches += 1

            avg_val_loss = epoch_val_loss / max(1, num_val_batches)
            self.val_loss_history.append([float(avg_val_loss)])

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_model_state = copy.deepcopy(self.model.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= int(self.config.early_stop_patience):
                print(
                    f"          [Early Stop] Triggered at Epoch {epoch}. Restoring best weights (Val Loss: {best_val_loss:.6f})",
                    flush=True,
                )
                break

        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)

        self.is_warmed_up = True
        for param in self.model.parameters():
            param.requires_grad = False

        self.replay_buffer.clear()
        print("[Refined][TAFAS] ====== TRAINING COMPLETE. MODEL FROZEN. ======\n", flush=True)
        return []

    @torch.no_grad()
    def predict(self, y_pred_current: torch.Tensor, model_input_seq: Optional[torch.Tensor] = None) -> torch.Tensor:
        _ = model_input_seq
        Original_Samples = sanitize_tensor(y_pred_current).to(self.device)

        if self.expected_H is None and Original_Samples.ndim >= 2:
            self.expected_H = int(Original_Samples.shape[1])

        if not self.is_warmed_up or self.model is None:
            if self.baseline_router:
                self.router_raw_output = Original_Samples.detach()
            else:
                self.router_raw_output = None
            return Original_Samples

        self.model.eval()
        Y_median_original = self._median_samples(Original_Samples)
        Y_refined = sanitize_tensor(self.model(Y_median_original))
        raw_shift = Y_refined - Y_median_original
        raw_samples = sanitize_tensor(Original_Samples + raw_shift)
        if self.baseline_router:
            self.router_raw_output = raw_samples.detach()
            if self._routing_enabled and self.e_base_moving is not None and self.e_ref_moving is not None:
                tau = max(1e-6, float(self.routing_temperature))
                exp_base = torch.exp(-self.e_base_moving / tau)
                exp_ref = torch.exp(-self.e_ref_moving / tau)
                c_t = exp_ref / (exp_base + exp_ref + 1e-8)
                c_t = c_t.view(1, 1, self.target_dim)
            else:
                c_t = torch.ones(1, 1, self.target_dim, device=self.device)
            routed_shift = raw_shift * c_t
            final_samples = sanitize_tensor(Original_Samples + routed_shift)
            return final_samples if final_samples.ndim == 3 else final_samples.squeeze(0)
        self.router_raw_output = None
        return raw_samples if raw_samples.ndim == 3 else raw_samples.squeeze(0)
