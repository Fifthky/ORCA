

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Optional, List, Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from core.util.refiner_util import sanitize_tensor


@dataclass
class SOLIDConfig:
    # Lightweight affine head operating on the base prediction window.
    # Follows the SOLID principle of adapting only the last forecasting layer.
    lr: float = 1e-3
    weight_decay: float = 0.0

    # Warmup-time global head training (mirrors LinearConfig schedule).
    collect_train_windows: Optional[int] = None
    collect_val_windows: Optional[int] = None
    max_epochs: int = 100
    batch_size: int = 256
    early_stop_patience: int = 20

    online_training: bool = False
    update_rule: str = "ring_quartet"

    # Sample-level contextualized adaptation hyper-parameters.
    period: int = 24
    period_n: int = 1
    test_train_num: int = 1000      # history lookback pool capacity
    selected_data_num: int = 10     # top-k neighbors for local adaptation
    lambda_period: float = 0.25     # phase filter threshold ratio
    local_adapt_steps: int = 3      # local SGD iterations on a cloned head
    local_adapt_lr: float = 1e-2


class _AffineHead(nn.Module):
    """Per-horizon scale+bias head, initialized as identity."""

    def __init__(self, horizon: int, target_dim: int) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.ones(1, int(horizon), int(target_dim)))
        self.bias = nn.Parameter(torch.zeros(1, int(horizon), int(target_dim)))

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        return y * self.scale + self.bias


class OnlineRefinerSOLID(nn.Module):
    """SOLID-style sample-level contextualized calibration under the blackbox contract.

    Inspired by the KDD'24 SOLID paper, this refiner keeps the core recipe:
      1) maintain a lookback pool of recent (context, base, gt) triplets;
      2) for each test sample, filter candidates by phase proximity;
      3) keep top-k nearest neighbors by L2 similarity on the lookback window;
      4) adapt a lightweight head via a few Adam steps on those neighbors, then
         restore the global head to avoid persistent drift.

    Because the blackbox contract forbids fine-tuning the base forecaster, the
    adaptation target is a per-horizon AffineHead sitting on top of the frozen
    base prediction. The global head is trained in a warmup phase using the
    delayed GT windows delivered by the ring pending queue (same schedule as
    Linear / TAFAS).
    """

    def __init__(
        self,
        feature_dim: int,
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

        self.config = SOLIDConfig()
        if lr is not None:
            self.config.lr = float(lr)
        if "collect_train_windows" not in kwargs or "collect_val_windows" not in kwargs:
            raise ValueError("SOLID requires explicitly provided window counts.")
        self.config.collect_train_windows = max(1, int(kwargs["collect_train_windows"]))
        self.config.collect_val_windows = max(1, int(kwargs["collect_val_windows"]))

        for k, v in kwargs.items():
            if hasattr(self.config, k):
                setattr(self.config, k, v)
        self.config.lr = max(1e-12, float(self.config.lr))
        self.config.weight_decay = max(0.0, float(self.config.weight_decay))
        self.config.period = max(1, int(self.config.period))
        self.config.period_n = max(1, int(self.config.period_n))
        self.config.test_train_num = max(1, int(self.config.test_train_num))
        self.config.selected_data_num = max(1, int(self.config.selected_data_num))
        self.config.lambda_period = float(max(0.0, min(1.0, float(self.config.lambda_period))))
        self.config.local_adapt_steps = max(1, int(self.config.local_adapt_steps))
        self.config.local_adapt_lr = max(1e-12, float(self.config.local_adapt_lr))

        self.is_initialized = False
        self.head: Optional[_AffineHead] = None
        self.optimizer: Optional[torch.optim.AdamW] = None

        self.expected_H: Optional[int] = None
        self.expected_L: Optional[int] = None

        self.replay_buffer: List[Dict] = []
        self.is_warmed_up: bool = False
        self.loss_history: List[List[float]] = []
        self.val_loss_history: List[List[float]] = []
        self._snapshot_idx: int = 0

        # History pool for sample-level contextualized adaptation at predict time.
        self._history: List[Dict] = []
        self._sample_counter: int = 0

    def reset_state(self, *, clear_loss_history: bool = True) -> None:
        self.expected_H = None
        self.expected_L = None
        self.replay_buffer.clear()
        self.is_warmed_up = False
        self._snapshot_idx = 0
        self._history = []
        self._sample_counter = 0
        self.e_base_moving = None
        self.e_ref_moving = None
        self._routing_enabled = False
        self.router_raw_output = None
        if clear_loss_history:
            self.loss_history = []
            self.val_loss_history = []

    def _ensure_head_and_optimizer(self) -> None:
        if self.head is None:
            if self.expected_H is None:
                raise RuntimeError("SOLID head requires known horizon before creation.")
            self.head = _AffineHead(
                horizon=int(self.expected_H),
                target_dim=int(self.target_dim),
            ).to(self.device)
            self.is_initialized = True
        if self.optimizer is None and self.head is not None:
            self.optimizer = torch.optim.AdamW(
                self.head.parameters(),
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

    @staticmethod
    def _to_time_channel(x: torch.Tensor) -> torch.Tensor:
        xx = x
        if xx.ndim == 3:
            xx = xx[0]
        if xx.ndim != 2:
            raise ValueError(f"Expected input with shape [L, D] or [1, L, D], got {tuple(x.shape)}")
        return xx

    @staticmethod
    def _phase_dist_ratio(a: int, b: int, period: int) -> float:
        if int(period) <= 1:
            return 0.0
        da = int(a) % int(period)
        db = int(b) % int(period)
        diff = abs(da - db) / float(period)
        return float(min(diff, 1.0 - diff))

    def _prepare_batch(self, batch: List[Dict]) -> Tuple[torch.Tensor, torch.Tensor]:
        padded_Y_median: List[torch.Tensor] = []
        padded_Y_GT: List[torch.Tensor] = []
        for item in batch:
            padded_Y_median.append(item["Y_base_median"])
            padded_Y_GT.append(item["Y_GT"])
        Y_median = torch.cat(padded_Y_median, dim=0)
        Y_GT = torch.cat(padded_Y_GT, dim=0)
        return Y_median, Y_GT

    def _append_history(
        self,
        *,
        x_ctx: torch.Tensor,
        y_base: torch.Tensor,
        y_gt: torch.Tensor,
    ) -> None:
        self._history.append(
            {
                "idx": int(self._sample_counter),
                "x": x_ctx.detach(),
                "y_base": y_base.detach(),
                "y_gt": y_gt.detach(),
            }
        )
        if len(self._history) > int(self.config.test_train_num):
            self._history = self._history[-int(self.config.test_train_num) :]
        self._sample_counter += 1

    def _select_context_indices(self, x_cur: torch.Tensor) -> List[int]:
        n = len(self._history)
        if n <= 0:
            return []

        cur_idx = int(self._sample_counter)
        candidate_ids = list(range(0, n))

        period = int(max(1, int(self.config.period) * int(self.config.period_n)))
        if period > 1 and float(self.config.lambda_period) > 0.0:
            filtered: List[int] = []
            tol = float(period) * float(self.config.lambda_period)
            for hid in candidate_ids:
                hidx = int(self._history[hid]["idx"])
                dist_to_cur = max(0, int(cur_idx - hidx))
                rem = float(dist_to_cur % int(period))
                if rem <= tol or rem >= (float(period) - tol):
                    filtered.append(hid)
            if filtered:
                candidate_ids = filtered
        if not candidate_ids:
            return []

        cur_tc = self._to_time_channel(x_cur)
        cur_len = int(cur_tc.shape[0])
        dist_pairs: List[Tuple[int, float]] = []

        # Group candidates by lookback length for batched cdist.
        groups: Dict[int, List[int]] = {}
        for hid in candidate_ids:
            cand_x = self._history[hid]["x"]
            cand_len = int(cand_x.shape[1] if cand_x.ndim == 3 else cand_x.shape[0])
            groups.setdefault(cand_len, []).append(hid)

        for cand_len, ids in groups.items():
            l = min(int(cur_len), int(cand_len))
            if l <= 0:
                continue
            cur_vec = cur_tc[-l:, :].reshape(1, -1)
            cand_mat = torch.stack(
                [self._to_time_channel(self._history[hid]["x"])[-l:, :].reshape(-1) for hid in ids],
                dim=0,
            )
            dists = torch.cdist(cur_vec, cand_mat, p=2).squeeze(0)
            for i, hid in enumerate(ids):
                dist_pairs.append((hid, float(dists[i].item())))

        if not dist_pairs:
            return []
        dist_pairs.sort(key=lambda z: z[1])
        topk = int(self.config.selected_data_num)
        return [hid for hid, _ in dist_pairs[:topk]]

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

        # Append to the sample-level history pool for contextualized adaptation.
        self._append_history(x_ctx=X, y_base=Y_base_median, y_gt=Y_GT_full)

        # Collect snapshots for warmup head training.
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
                print(f"[Refined][SOLID] Snapshot collection: {collected}/{target_collection_size}", flush=True)

        should_train_now = (
            target_collection_size > 0
            and len(self.replay_buffer) >= target_collection_size
            and (not self.is_warmed_up or self.config.online_training)
        )
        if not should_train_now:
            return []

        print("\n[Refined][SOLID] ====== BUFFER FULL. INITIATING SNAPSHOT TRAINING ======", flush=True)

        self._ensure_head_and_optimizer()
        if self.head is None or self.optimizer is None:
            return []

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
        best_head_state: Optional[Dict[str, torch.Tensor]] = None
        patience_counter = 0

        for param in self.head.parameters():
            param.requires_grad = True

        for epoch in range(int(self.config.max_epochs)):
            self.head.train()
            epoch_train_data = list(train_data)
            epoch_train_loss = 0.0
            num_batches = 0

            for i in range(0, len(epoch_train_data), int(self.config.batch_size)):
                batch = epoch_train_data[i : i + int(self.config.batch_size)]
                Y_median_b, Y_GT_b = self._prepare_batch(batch)
                y_pred = sanitize_tensor(self.head(Y_median_b))
                loss = F.mse_loss(y_pred, Y_GT_b)
                loss = torch.nan_to_num(loss, nan=1e6, posinf=1e6, neginf=1e6)

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.head.parameters(), max_norm=1.0)
                self.optimizer.step()

                epoch_train_loss += float(loss.item())
                num_batches += 1

            avg_train_loss = epoch_train_loss / max(1, num_batches)
            self.loss_history.append([float(avg_train_loss)])

            self.head.eval()
            epoch_val_loss = 0.0
            num_val_batches = 0
            with torch.no_grad():
                for i in range(0, len(val_data), int(self.config.batch_size)):
                    batch = val_data[i : i + int(self.config.batch_size)]
                    Y_median_b, Y_GT_b = self._prepare_batch(batch)
                    y_pred = sanitize_tensor(self.head(Y_median_b))
                    v_loss = F.mse_loss(y_pred, Y_GT_b)
                    epoch_val_loss += float(v_loss.item())
                    num_val_batches += 1

            avg_val_loss = epoch_val_loss / max(1, num_val_batches)
            self.val_loss_history.append([float(avg_val_loss)])

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_head_state = copy.deepcopy(self.head.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= int(self.config.early_stop_patience):
                print(
                    f"          [Early Stop] Triggered at Epoch {epoch}. Restoring best weights (Val Loss: {best_val_loss:.6f})",
                    flush=True,
                )
                break

        if best_head_state is not None:
            self.head.load_state_dict(best_head_state)

        self.is_warmed_up = True
        for param in self.head.parameters():
            param.requires_grad = False

        self.replay_buffer.clear()
        print("[Refined][SOLID] ====== TRAINING COMPLETE. HEAD FROZEN. ======\n", flush=True)
        return []

    def _locally_adapted_head_forward(
        self,
        *,
        y_base_current: torch.Tensor,
        selected_ids: List[int],
    ) -> torch.Tensor:
        """Clone the global head, run a few Adam steps on top-k neighbors, predict, discard."""
        if self.head is None:
            return y_base_current

        local_head = copy.deepcopy(self.head).to(self.device)
        for p in local_head.parameters():
            p.requires_grad = True
        local_optim = torch.optim.Adam(
            local_head.parameters(),
            lr=float(self.config.local_adapt_lr),
            weight_decay=float(self.config.weight_decay),
        )

        # Stack neighbors into a single batch for efficient SGD.
        base_list: List[torch.Tensor] = []
        gt_list: List[torch.Tensor] = []
        for hid in selected_ids:
            b = self._history[hid]["y_base"]
            g = self._history[hid]["y_gt"]
            if b.ndim == 2:
                b = b.unsqueeze(0)
            if g.ndim == 2:
                g = g.unsqueeze(0)
            base_list.append(b)
            gt_list.append(g)
        if not base_list:
            del local_head, local_optim
            return y_base_current
        y_base_batch = torch.cat(base_list, dim=0).to(self.device)
        y_gt_batch = torch.cat(gt_list, dim=0).to(self.device)

        local_head.train()
        for _ in range(int(self.config.local_adapt_steps)):
            y_hat = local_head(y_base_batch)
            loss = F.mse_loss(y_hat, y_gt_batch)
            loss = torch.nan_to_num(loss, nan=1e6, posinf=1e6, neginf=1e6)
            local_optim.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(local_head.parameters(), max_norm=1.0)
            local_optim.step()

        local_head.eval()
        with torch.no_grad():
            y_out = local_head(y_base_current)

        # Discard the local copy; global head remains untouched (SOLID principle).
        del local_head, local_optim
        return y_out

    def predict(self, y_pred_current: torch.Tensor, model_input_seq: Optional[torch.Tensor] = None) -> torch.Tensor:
        Original_Samples = sanitize_tensor(y_pred_current).to(self.device)

        if self.expected_H is None and Original_Samples.ndim >= 2:
            self.expected_H = int(Original_Samples.shape[1])

        if not self.is_warmed_up or self.head is None:
            if self.baseline_router:
                self.router_raw_output = Original_Samples.detach()
            else:
                self.router_raw_output = None
            return Original_Samples

        self.head.eval()
        Y_median_original = self._median_samples(Original_Samples)

        selected_ids: List[int] = []
        if model_input_seq is not None and len(self._history) > 0:
            x_cur = sanitize_tensor(model_input_seq).to(self.device)
            selected_ids = self._select_context_indices(x_cur)

        if len(selected_ids) >= int(self.config.selected_data_num):
            Y_refined = self._locally_adapted_head_forward(
                y_base_current=Y_median_original,
                selected_ids=selected_ids,
            )
        else:
            with torch.no_grad():
                Y_refined = self.head(Y_median_original)

        Y_refined = sanitize_tensor(Y_refined)
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
