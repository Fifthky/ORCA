from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from core.util.refiner_util import sanitize_tensor


@dataclass
class AdaYConfig:
    hidden_dim: int = 512
    lr: float = 1e-4

    # Bounded correction magnitude in Delta-Adapter.
    delta: float = 0.1

    grad_clip: float = 1e-2

    # Reuse batch-style split controls when provided by evaluator.
    collect_train_windows: Optional[int] = None
    collect_val_windows: Optional[int] = None
    online_training: bool = False

    # Offline warmup replay size (stride-1 windows), aligned with linear buffer default.
    warmup_buffer_windows: int = 5000
    warmup_batch_size: int = 256
    warmup_epochs: int = 10
    warmup_val_ratio: float = 0.1
    warmup_patience: int = 3

    # Online phase settings after warmup.
    online_lr: float = 1e-6
    online_grad_clip: float = 1e-4


class AdaYPostProcessingNet(nn.Module):
    """Readonly-style output adapter used for additive AdaY correction."""

    def __init__(self, state_dim: int, hidden_dim: int, action_dim: int, delta: float = 0.1) -> None:
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.bn1 = nn.InstanceNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.InstanceNorm1d(hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)
        self.delta = float(delta)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = self.bn1(x)
        x = F.relu(self.fc2(x))
        x = self.bn2(x)
        x = self.fc3(x)
        return torch.tanh(x) * self.delta


class OnlineRefinerAdaY(nn.Module):
    """
    Delta-Adapter style output refiner aligned to the Sequence refiner I/O contract.

    update(X, Y_base, Y_ref_past, Y_GT_full, physical_stride=..., step=...)
    predict(y_pred_current, model_input_seq=...)
    """

    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int = 512,
        lr: float = 1e-4,
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

        delta = float(kwargs.get("delta", 0.1))
        grad_clip = float(kwargs.get("grad_clip", 1e-2))
        warmup_buffer_windows = int(kwargs.get("warmup_buffer_windows", 5000))
        warmup_batch_size = int(kwargs.get("warmup_batch_size", 256))
        warmup_epochs = int(kwargs.get("warmup_epochs", 10))
        warmup_val_ratio = float(kwargs.get("warmup_val_ratio", 0.1))
        warmup_patience = int(kwargs.get("warmup_patience", 3))
        online_lr = float(kwargs.get("online_lr", 1e-6))
        online_grad_clip = float(kwargs.get("online_grad_clip", 1e-4))
        collect_train_windows = kwargs.get("collect_train_windows", None)
        collect_val_windows = kwargs.get("collect_val_windows", None)
        online_training = bool(kwargs.get("online_training", False))
        self.config = AdaYConfig(
            hidden_dim=int(hidden_dim),
            lr=float(lr),
            delta=delta,
            grad_clip=grad_clip,
            collect_train_windows=(int(collect_train_windows) if collect_train_windows is not None else None),
            collect_val_windows=(int(collect_val_windows) if collect_val_windows is not None else None),
            online_training=online_training,
            warmup_buffer_windows=max(1, warmup_buffer_windows),
            warmup_batch_size=max(1, warmup_batch_size),
            warmup_epochs=max(1, warmup_epochs),
            warmup_val_ratio=min(max(0.0, warmup_val_ratio), 0.5),
            warmup_patience=max(1, warmup_patience),
            online_lr=max(1e-12, online_lr),
            online_grad_clip=max(0.0, online_grad_clip),
        )

        self.is_initialized = False
        self.is_warmed_up = False

        self.expected_H: Optional[int] = None
        self.state_dim: Optional[int] = None

        self.model: Optional[AdaYPostProcessingNet] = None
        self.optimizer: Optional[torch.optim.Optimizer] = None
        self.scheduler: Optional[torch.optim.lr_scheduler.OneCycleLR] = None
        self.loss_history: list[list[float]] = []
        self._warmup_done: bool = False
        self._warmup_replay: Deque[tuple[torch.Tensor, torch.Tensor]] = deque()

    def reset_state(self, *, clear_loss_history: bool = True) -> None:
        self.expected_H = None
        self.state_dim = None

        self.is_warmed_up = False

        if clear_loss_history:
            self.loss_history = []
        self._warmup_done = False
        self._warmup_replay.clear()
        self.e_base_moving = None
        self.e_ref_moving = None
        self._routing_enabled = False
        self.router_raw_output = None

    def _update_router_ema(
        self,
        y_base_median: torch.Tensor,
        y_ref_past_median: torch.Tensor,
        y_gt_aligned: torch.Tensor,
    ) -> None:
        if not self.baseline_router:
            return
        err_base = torch.mean(torch.abs(y_gt_aligned - y_base_median), dim=(0, 1))
        err_ref = torch.mean(torch.abs(y_gt_aligned - y_ref_past_median), dim=(0, 1))
        alpha = float(self.ema_error_momentum)
        if self.e_base_moving is None or self.e_ref_moving is None:
            self.e_base_moving = err_base.detach()
            self.e_ref_moving = err_ref.detach()
        else:
            self.e_base_moving = alpha * err_base + (1.0 - alpha) * self.e_base_moving
            self.e_ref_moving = alpha * err_ref + (1.0 - alpha) * self.e_ref_moving
        self._routing_enabled = True

    @staticmethod
    def _median_samples(y: torch.Tensor) -> torch.Tensor:
        # Accept [S, H, D] or [1, H, D] or [H, D].
        if y.ndim == 2:
            return y.unsqueeze(0)
        if y.ndim == 3:
            if y.shape[0] == 1:
                return y
            return torch.median(y, dim=0, keepdim=True)[0]
        raise ValueError(f"Unsupported tensor shape for median reduction: {tuple(y.shape)}")

    @staticmethod
    def _align_gt_to_base(y_gt: torch.Tensor, y_base: torch.Tensor) -> torch.Tensor:
        y = y_gt
        if y.ndim == 2:
            y = y.unsqueeze(0)
        b = y_base
        if b.ndim == 2:
            b = b.unsqueeze(0)

        if y.ndim != 3 or b.ndim != 3:
            raise ValueError(f"Expected 3D tensors, got gt={tuple(y.shape)}, base={tuple(b.shape)}")

        if y.shape[1] == b.shape[2] and y.shape[2] == b.shape[1]:
            y = y.transpose(1, 2)
        if y.shape[0] > 1 and b.shape[0] == 1:
            y = y[0:1]
        if y.shape[0] == 1 and b.shape[0] > 1:
            y = y.expand(b.shape[0], -1, -1)
        if y.shape != b.shape:
            raise RuntimeError(f"Shape mismatch gt/base: gt={tuple(y.shape)}, base={tuple(b.shape)}")
        return y

    def _build_or_resize_if_needed(self) -> None:
        if self.expected_H is None or self.state_dim is None:
            return

        if not self.is_initialized:
            self.model = AdaYPostProcessingNet(
                state_dim=int(self.state_dim),
                hidden_dim=self.config.hidden_dim,
                action_dim=int(self.state_dim),
                delta=self.config.delta,
            ).to(self.device)
            self.optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=self.config.lr,
            )
            self.is_initialized = True

    def _train_step(self, y_base_median: torch.Tensor, y_gt: torch.Tensor, *, grad_clip: Optional[float] = None) -> float:
        if self.model is None or self.optimizer is None:
            return 0.0

        y_state = y_base_median.reshape(y_base_median.shape[0], -1)
        action = self.model(y_state)
        y_refined = y_base_median + action.view_as(y_base_median)
        loss = F.mse_loss(y_refined, y_gt)

        self.optimizer.zero_grad(set_to_none=True)
        if loss.requires_grad:
            loss.backward()
            clip_val = float(self.config.grad_clip if grad_clip is None else grad_clip)
            if clip_val > 0.0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=clip_val)
            self.optimizer.step()
            if self.scheduler is not None:
                self.scheduler.step()

        return float(loss.item())

    def _run_offline_warmup(self) -> list[float]:
        if self.model is None or self.optimizer is None:
            return []

        target_windows = self._target_collection_size()
        if len(self._warmup_replay) < int(target_windows):
            return []

        losses: list[float] = []
        replay = list(self._warmup_replay)
        batch_size = int(self.config.warmup_batch_size)
        train_data, val_data = self._split_warmup_replay(replay)
        best_val = float("inf")
        best_state: Optional[dict[str, torch.Tensor]] = None
        patience = 0

        self.model.train()
        with torch.enable_grad():
            for _ in range(int(self.config.warmup_epochs)):
                for i in range(0, len(train_data), batch_size):
                    chunk = train_data[i : i + batch_size]
                    y_base_batch = torch.cat([x[0] for x in chunk], dim=0)
                    y_gt_batch = torch.cat([x[1] for x in chunk], dim=0)
                    losses.append(self._train_step(y_base_batch, y_gt_batch, grad_clip=self.config.grad_clip))

                if val_data:
                    self.model.eval()
                    val_loss_acc = 0.0
                    val_n = 0
                    with torch.no_grad():
                        for i in range(0, len(val_data), batch_size):
                            chunk = val_data[i : i + batch_size]
                            y_base_batch = torch.cat([x[0] for x in chunk], dim=0)
                            y_gt_batch = torch.cat([x[1] for x in chunk], dim=0)
                            y_state = y_base_batch.reshape(y_base_batch.shape[0], -1)
                            action = self.model(y_state)
                            y_refined = y_base_batch + action.view_as(y_base_batch)
                            val_loss_acc += float(F.mse_loss(y_refined, y_gt_batch).item())
                            val_n += 1
                    avg_val = val_loss_acc / max(1, val_n)
                    if avg_val < best_val:
                        best_val = avg_val
                        best_state = {k: v.detach().clone() for k, v in self.model.state_dict().items()}
                        patience = 0
                    else:
                        patience += 1
                        if patience >= int(self.config.warmup_patience):
                            break
                    self.model.train()

        if best_state is not None:
            self.model.load_state_dict(best_state)

        for group in self.optimizer.param_groups:
            group["lr"] = float(self.config.online_lr)
        self._warmup_done = True
        self.is_warmed_up = True
        return losses

    def _target_collection_size(self) -> int:
        if bool(self.config.online_training):
            return int(max(1, self.config.warmup_buffer_windows))

        train_n = self.config.collect_train_windows
        val_n = self.config.collect_val_windows
        if train_n is not None and val_n is not None and self.expected_H is not None:
            return int(max(1, int(train_n) + int(self.expected_H) + int(val_n)))

        return int(max(1, self.config.warmup_buffer_windows))

    def _split_warmup_replay(
        self, replay: list[tuple[torch.Tensor, torch.Tensor]]
    ) -> tuple[list[tuple[torch.Tensor, torch.Tensor]], list[tuple[torch.Tensor, torch.Tensor]]]:
        if bool(self.config.online_training):
            total = len(replay)
            val_size = int(round(float(total) * float(self.config.warmup_val_ratio)))
            if total > 1:
                val_size = min(max(1, val_size), total - 1)
            else:
                val_size = 0
            train_data = replay[: total - val_size] if val_size > 0 else replay
            val_data = replay[total - val_size :] if val_size > 0 else []
            return train_data, val_data

        train_n = int(self.config.collect_train_windows or 0)
        val_n = int(self.config.collect_val_windows or 0)
        gap_n = int(self.expected_H or 0)
        train_end = min(len(replay), max(1, train_n))
        val_start = min(len(replay), max(train_end, train_n + gap_n))
        val_end = min(len(replay), max(val_start, val_start + max(1, val_n)))
        train_data = replay[:train_end]
        val_data = replay[val_start:val_end]
        if not val_data:
            val_data = train_data[-1:]
        return train_data, val_data

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
    ) -> list[float]:
        _ = step
        # Temporal alignment is owned by the online wrapper; this refiner only
        # consumes the aligned quartet and ignores delayed-error payload.
        _ = e_past_override

        if Y_GT_full.shape[1] == 0:
            return []

        Y_base = sanitize_tensor(Y_base).detach().to(self.device)
        Y_GT_full = sanitize_tensor(Y_GT_full).detach().to(self.device)

        if Y_GT_full.ndim == 3 and Y_GT_full.shape[0] > 1:
            Y_GT_full = Y_GT_full[0:1]

        if self.expected_H is None and Y_GT_full.shape[1] > 0:
            self.expected_H = int(Y_GT_full.shape[1])
        if self.state_dim is None and self.expected_H is not None:
            self.state_dim = int(self.expected_H) * int(self.target_dim)

        y_base_median = self._median_samples(Y_base)
        Y_GT_full = self._align_gt_to_base(Y_GT_full, y_base_median)
        if self.baseline_router:
            y_ref_past_median = self._median_samples(sanitize_tensor(Y_ref_past).detach().to(self.device))
            y_ref_past_median = self._align_gt_to_base(y_ref_past_median, y_base_median)
            self._update_router_ema(y_base_median, y_ref_past_median, Y_GT_full)
        if physical_stride is not None and int(physical_stride) <= 0:
            raise ValueError(f"physical_stride must be >= 1, got {physical_stride}")

        step_losses: list[float] = []
        self._build_or_resize_if_needed()
        if self.model is None or self.optimizer is None:
            return step_losses

        # Use complete window as training state — same distribution as predict().
        y_base_window = y_base_median
        y_gt_window = Y_GT_full

        if not self._warmup_done:
            self._warmup_replay.append((y_base_window.detach(), y_gt_window.detach()))
            while len(self._warmup_replay) > int(self._target_collection_size()):
                self._warmup_replay.popleft()
            warmup_losses = self._run_offline_warmup()
            if warmup_losses:
                step_losses.extend(warmup_losses)
                self.loss_history.append(warmup_losses)
            return step_losses

        self.model.train()
        with torch.enable_grad():
            step_losses.append(self._train_step(y_base_window, y_gt_window, grad_clip=self.config.online_grad_clip))

        self.is_warmed_up = True
        if step_losses:
            self.loss_history.append(step_losses)
        return step_losses

    @torch.no_grad()
    def predict(self, y_pred_current: torch.Tensor, model_input_seq: Optional[torch.Tensor] = None) -> torch.Tensor:
        original_samples = sanitize_tensor(y_pred_current).to(self.device)
        _ = model_input_seq

        if self.expected_H is None and original_samples.shape[1] > 0:
            self.expected_H = int(original_samples.shape[1])

        if not self.is_warmed_up or self.model is None:
            if self.baseline_router:
                self.router_raw_output = original_samples.detach()
            else:
                self.router_raw_output = None
            return original_samples

        y_median = self._median_samples(original_samples)

        self.model.eval()
        y_state = y_median.reshape(y_median.shape[0], -1)
        action = self.model(y_state)
        y_refined = y_median + action.view_as(y_median)
        raw_shift = y_refined - y_median
        raw_samples = original_samples + raw_shift
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
            final_samples = original_samples + routed_shift
            return final_samples if final_samples.ndim == 3 else final_samples.squeeze(0)
        self.router_raw_output = None
        return raw_samples if raw_samples.ndim == 3 else raw_samples.squeeze(0)

