from __future__ import annotations

import copy
from collections import deque
from dataclasses import dataclass
from typing import Deque, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from core.util.refiner_util import sanitize_tensor


@dataclass
class DSOFConfig:
    hidden_dim: int = 128
    mlp_depth: int = 3
    dropout: float = 0.0

    # Slow stream (ER)
    replay_buffer_size: int = 300
    batch_replay_size: int = 32
    num_er_epochs: int = 1
    freq_er_update: int = 1

    # Fast stream (TD)
    td_enabled: bool = True
    td_k: int = 1
    discounted: float = 0.9

    # Optimizer settings for student stream
    warmup_lr: float = 1e-3
    online_batch_lr: float = 1e-4
    online_td_lr: float = 3e-4
    grad_clip: float = 1e-2

    # Batch-style split controls from evaluator
    collect_train_windows: Optional[int] = None
    collect_val_windows: Optional[int] = None
    online_training: bool = False
    warmup_epochs: int = 10
    warmup_patience: int = 3
    warmup_val_ratio: float = 0.1
    warmup_buffer_windows: int = 3000


class ResidualStudentMLP(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int, depth: int, dropout: float) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        in_dim = int(state_dim)
        for _ in range(max(1, int(depth) - 1)):
            layers.append(nn.Linear(in_dim, int(hidden_dim)))
            layers.append(nn.ReLU())
            if float(dropout) > 0.0:
                layers.append(nn.Dropout(float(dropout)))
            in_dim = int(hidden_dim)
        layers.append(nn.Linear(in_dim, int(action_dim)))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class OnlineRefinerDSOF(nn.Module):
    """
    Blackbox-compatible DSOF-style refiner.

    - Keeps teacher/base predictor frozen (no teacher gradients).
    - Learns a residual student on top of teacher outputs.
    - Slow stream: experience replay on fully closed windows.
    - Fast stream: TD-style update using previous student state and current teacher pseudo target.
    """

    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int = 128,
        num_blocks: int = 3,
        lr: float = 1e-3,
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

        self.config = DSOFConfig(
            hidden_dim=int(hidden_dim),
            mlp_depth=int(max(2, int(num_blocks))),
            dropout=float(kwargs.get("dropout", 0.0)),
            replay_buffer_size=max(1, int(kwargs.get("replay_buffer_size", 300))),
            batch_replay_size=max(1, int(kwargs.get("batch_replay_size", 32))),
            num_er_epochs=max(1, int(kwargs.get("num_er_epochs", 1))),
            freq_er_update=max(1, int(kwargs.get("freq_er_update", 1))),
            td_enabled=bool(kwargs.get("td_enabled", True)),
            td_k=max(1, int(kwargs.get("td_k", 1))),
            discounted=float(kwargs.get("discounted", 0.9)),
            warmup_lr=max(1e-12, float(kwargs.get("warmup_lr", lr))),
            online_batch_lr=max(1e-12, float(kwargs.get("online_batch_lr", 1e-4))),
            online_td_lr=max(1e-12, float(kwargs.get("online_td_lr", 3e-4))),
            grad_clip=max(0.0, float(kwargs.get("grad_clip", 1e-2))),
            collect_train_windows=(
                int(kwargs.get("collect_train_windows"))
                if kwargs.get("collect_train_windows", None) is not None
                else None
            ),
            collect_val_windows=(
                int(kwargs.get("collect_val_windows"))
                if kwargs.get("collect_val_windows", None) is not None
                else None
            ),
            online_training=bool(kwargs.get("online_training", False)),
            warmup_epochs=max(1, int(kwargs.get("warmup_epochs", 10))),
            warmup_patience=max(1, int(kwargs.get("warmup_patience", 3))),
            warmup_val_ratio=min(max(0.0, float(kwargs.get("warmup_val_ratio", 0.1))), 0.5),
            warmup_buffer_windows=max(1, int(kwargs.get("warmup_buffer_windows", 3000))),
        )

        self.expected_H: Optional[int] = None
        self.expected_L: Optional[int] = int(kwargs.get("seq_len_hint", 0)) if int(kwargs.get("seq_len_hint", 0)) > 0 else None
        self.state_dim: Optional[int] = None

        self.model: Optional[ResidualStudentMLP] = None
        self.opt_batch: Optional[torch.optim.Optimizer] = None
        self.opt_td: Optional[torch.optim.Optimizer] = None

        self.is_initialized = False
        self.is_warmed_up = False
        self._warmup_done = False
        self._er_step_count = 0

        self.loss_history: list[list[float]] = []
        self.val_loss_history: list[list[float]] = []

        self._warmup_replay: Deque[dict] = deque()
        self._replay_buffer: Deque[dict] = deque(maxlen=int(self.config.replay_buffer_size))
        self._prev_snapshot: Optional[dict] = None

    def reset_state(self, *, clear_loss_history: bool = True) -> None:
        self.expected_H = None
        if self.expected_L is None or self.expected_L <= 0:
            self.expected_L = None
        self.state_dim = None

        self.is_warmed_up = False
        self._warmup_done = False
        self._er_step_count = 0

        self._warmup_replay.clear()
        self._replay_buffer.clear()
        self._prev_snapshot = None
        self.e_base_moving = None
        self.e_ref_moving = None
        self._routing_enabled = False
        self.router_raw_output = None

        if clear_loss_history:
            self.loss_history = []
            self.val_loss_history = []

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
        if self.expected_H is None or self.expected_L is None:
            return

        state_dim = int(self.target_dim) * int(self.expected_H + self.expected_L)
        if self.is_initialized and self.state_dim == state_dim:
            return

        self.state_dim = int(state_dim)
        self.model = ResidualStudentMLP(
            state_dim=int(self.state_dim),
            action_dim=int(self.expected_H) * int(self.target_dim),
            hidden_dim=int(self.config.hidden_dim),
            depth=int(self.config.mlp_depth),
            dropout=float(self.config.dropout),
        ).to(self.device)
        self.opt_batch = torch.optim.AdamW(self.model.parameters(), lr=float(self.config.warmup_lr), amsgrad=True)
        self.opt_td = torch.optim.AdamW(self.model.parameters(), lr=float(self.config.online_td_lr), amsgrad=True)
        self.is_initialized = True

    def _build_state(self, x_seq: torch.Tensor, y_base_center: torch.Tensor) -> torch.Tensor:
        x = x_seq
        if x.ndim == 2:
            x = x.unsqueeze(0)
        if x.shape[0] != y_base_center.shape[0]:
            x = x.expand(y_base_center.shape[0], -1, -1)

        if self.expected_L is not None:
            L = int(self.expected_L)
            if x.shape[1] < L:
                x = F.pad(x, (0, 0, L - x.shape[1], 0))
            elif x.shape[1] > L:
                x = x[:, -L:, :]

        return torch.cat(
            [x.reshape(x.shape[0], -1), y_base_center.reshape(y_base_center.shape[0], -1)],
            dim=1,
        )

    def _forward_student(self, state: torch.Tensor, y_base_center: torch.Tensor) -> torch.Tensor:
        if self.model is None or self.expected_H is None:
            return y_base_center
        residual = self.model(state).view_as(y_base_center)
        return y_base_center + residual

    def _apply_step(self, optimizer: torch.optim.Optimizer, loss: torch.Tensor) -> float:
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if float(self.config.grad_clip) > 0.0 and self.model is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=float(self.config.grad_clip))
        optimizer.step()
        return float(loss.item())

    def _target_collection_size(self) -> int:
        if bool(self.config.online_training):
            return int(max(1, self.config.warmup_buffer_windows))

        train_n = self.config.collect_train_windows
        val_n = self.config.collect_val_windows
        if train_n is not None and val_n is not None and self.expected_H is not None:
            return int(max(1, int(train_n) + int(self.expected_H) + int(val_n)))

        return int(max(1, self.config.warmup_buffer_windows))

    def _split_warmup_replay(self, replay: list[dict]) -> tuple[list[dict], list[dict]]:
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

    def _run_warmup(self) -> list[float]:
        if self.model is None or self.opt_batch is None:
            return []
        if len(self._warmup_replay) < int(self._target_collection_size()):
            return []

        replay = list(self._warmup_replay)
        train_data, val_data = self._split_warmup_replay(replay)
        if not train_data:
            return []

        batch_size = int(self.config.batch_replay_size)
        losses: list[float] = []
        best_val = float("inf")
        best_state: Optional[dict[str, torch.Tensor]] = None
        patience = 0

        self.model.train()
        for _ in range(int(self.config.warmup_epochs)):
            for i in range(0, len(train_data), batch_size):
                chunk = train_data[i : i + batch_size]
                s = torch.cat([item["state"] for item in chunk], dim=0)
                yb = torch.cat([item["y_base"] for item in chunk], dim=0)
                yg = torch.cat([item["y_gt"] for item in chunk], dim=0)
                pred = self._forward_student(s, yb)
                loss = F.mse_loss(pred, yg)
                losses.append(self._apply_step(self.opt_batch, loss))

            if val_data:
                self.model.eval()
                with torch.no_grad():
                    val_acc = 0.0
                    val_n = 0
                    for i in range(0, len(val_data), batch_size):
                        chunk = val_data[i : i + batch_size]
                        s = torch.cat([item["state"] for item in chunk], dim=0)
                        yb = torch.cat([item["y_base"] for item in chunk], dim=0)
                        yg = torch.cat([item["y_gt"] for item in chunk], dim=0)
                        pred = self._forward_student(s, yb)
                        val_acc += float(F.mse_loss(pred, yg).item())
                        val_n += 1
                    avg_val = val_acc / max(1, val_n)
                self.val_loss_history.append([float(avg_val)])
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

        if self.opt_batch is not None:
            for pg in self.opt_batch.param_groups:
                pg["lr"] = float(self.config.online_batch_lr)
        self._warmup_done = True
        self.is_warmed_up = True

        self._replay_buffer.clear()
        for item in train_data:
            self._replay_buffer.append(item)

        return losses

    def _er_update(self) -> list[float]:
        if self.model is None or self.opt_batch is None:
            return []
        if len(self._replay_buffer) < int(self.config.batch_replay_size):
            return []
        if (self._er_step_count % int(self.config.freq_er_update)) != 0:
            return []

        losses: list[float] = []
        n = len(self._replay_buffer)
        bs = min(int(self.config.batch_replay_size), n)
        for _ in range(int(self.config.num_er_epochs)):
            idx = torch.randperm(n, device=self.device)[:bs].tolist()
            chunk = [self._replay_buffer[i] for i in idx]
            s = torch.cat([item["state"] for item in chunk], dim=0)
            yb = torch.cat([item["y_base"] for item in chunk], dim=0)
            yg = torch.cat([item["y_gt"] for item in chunk], dim=0)
            pred = self._forward_student(s, yb)
            loss = F.mse_loss(pred, yg)
            losses.append(self._apply_step(self.opt_batch, loss))
        return losses

    def _td_update(self, curr: dict) -> list[float]:
        if not bool(self.config.td_enabled):
            return []
        if self.model is None or self.opt_td is None:
            return []
        if self._prev_snapshot is None:
            return []

        prev = self._prev_snapshot
        k = int(max(1, min(int(self.config.td_k), int(self.expected_H or 1))))

        td_truth = torch.cat([prev["y_gt"][:, :k, :], curr["y_base"][:, :-k, :]], dim=1)

        if td_truth.shape != prev["y_base"].shape:
            return []

        pred_prev = self._forward_student(prev["state"], prev["y_base"])

        H = int(td_truth.shape[1])
        tail_len = int(max(0, H - k))
        if tail_len > 0:
            tail_w = torch.tensor(
                [float(self.config.discounted) ** i for i in range(tail_len)],
                device=self.device,
                dtype=td_truth.dtype,
            )
            weights = torch.cat([torch.ones(k, device=self.device, dtype=td_truth.dtype), tail_w], dim=0)
        else:
            weights = torch.ones(H, device=self.device, dtype=td_truth.dtype)
        weights = weights.view(1, H, 1)

        td_loss = (F.mse_loss(pred_prev, td_truth, reduction="none") * weights).mean()
        return [self._apply_step(self.opt_td, td_loss)]

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
        _ = e_past_override

        if Y_GT_full.shape[1] == 0:
            return []

        x = sanitize_tensor(X).detach().to(self.device)
        yb_all = sanitize_tensor(Y_base).detach().to(self.device)
        ygt = sanitize_tensor(Y_GT_full).detach().to(self.device)
        y_ref_past = sanitize_tensor(Y_ref_past).detach().to(self.device)

        yb = self._median_samples(yb_all)
        ygt = self._align_gt_to_base(ygt, yb)
        if self.baseline_router:
            y_ref_past_median = self._median_samples(y_ref_past)
            if y_ref_past_median.shape != yb.shape:
                y_ref_past_median = yb
            err_base = torch.mean(torch.abs(ygt - yb), dim=(0, 1))
            err_ref = torch.mean(torch.abs(ygt - y_ref_past_median), dim=(0, 1))
            alpha = float(self.ema_error_momentum)
            if self.e_base_moving is None or self.e_ref_moving is None:
                self.e_base_moving = err_base.detach()
                self.e_ref_moving = err_ref.detach()
            else:
                self.e_base_moving = alpha * err_base + (1.0 - alpha) * self.e_base_moving
                self.e_ref_moving = alpha * err_ref + (1.0 - alpha) * self.e_ref_moving
            self._routing_enabled = True

        if self.expected_H is None and ygt.shape[1] > 0:
            self.expected_H = int(ygt.shape[1])
        if self.expected_L is None and x.ndim == 3 and int(x.shape[1]) > 0:
            self.expected_L = int(x.shape[1])
        if self.expected_L is None and self.expected_H is not None:
            self.expected_L = int(self.expected_H)

        if physical_stride is not None and int(physical_stride) <= 0:
            raise ValueError(f"physical_stride must be >= 1, got {physical_stride}")

        self._build_or_resize_if_needed()
        if self.model is None:
            return []

        state = self._build_state(x, yb)
        snap = {
            "state": state.detach(),
            "y_base": yb.detach(),
            "y_gt": ygt.detach(),
        }

        step_losses: list[float] = []

        if not self._warmup_done:
            self._warmup_replay.append(snap)
            while len(self._warmup_replay) > int(self._target_collection_size()):
                self._warmup_replay.popleft()
            warmup_losses = self._run_warmup()
            if warmup_losses:
                step_losses.extend(warmup_losses)
                self.loss_history.append(warmup_losses)
            self._prev_snapshot = snap
            return step_losses

        self._replay_buffer.append(snap)
        td_losses = self._td_update(snap)
        er_losses = self._er_update()
        step_losses.extend(td_losses)
        step_losses.extend(er_losses)
        self._er_step_count += 1

        self._prev_snapshot = snap
        if step_losses:
            self.loss_history.append(step_losses)
        self.is_warmed_up = True
        return step_losses

    @torch.no_grad()
    def predict(self, y_pred_current: torch.Tensor, model_input_seq: Optional[torch.Tensor] = None) -> torch.Tensor:
        original_samples = sanitize_tensor(y_pred_current).to(self.device)

        if self.expected_H is None and original_samples.shape[1] > 0:
            self.expected_H = int(original_samples.shape[1])

        if model_input_seq is None:
            if self.expected_L is None:
                self.expected_L = int(self.expected_H or 1)
            x = torch.zeros(1, int(self.expected_L), int(self.target_dim), device=self.device)
        else:
            x = sanitize_tensor(model_input_seq).to(self.device)
            if x.ndim == 2:
                x = x.unsqueeze(0)
            if self.expected_L is None and x.shape[1] > 0:
                self.expected_L = int(x.shape[1])

        if not self.is_warmed_up or self.model is None or self.expected_H is None:
            if self.baseline_router:
                self.router_raw_output = original_samples.detach()
            else:
                self.router_raw_output = None
            return original_samples

        yb = self._median_samples(original_samples)
        state = self._build_state(x, yb)
        y_ref_center = self._forward_student(state, yb)
        raw_shift = y_ref_center - yb
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
