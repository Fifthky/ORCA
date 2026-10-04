from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

from core.util.refiner_util import sanitize_tensor


@dataclass
class ETSConfig:
    # Level smoothing coefficient.
    alpha: float = 0.3
    # Trend smoothing coefficient (weakened default for stability)
    beta: float = 0.03
    # Damped trend factor; set to 1.0 for undamped Holt's linear trend.
    damping_factor: float = 0.98
    # Keep only recent residual states for lightweight caching.
    max_history_steps: int = 12000
    # Gain sensitivity for residual-based gating (higher -> more suppression for noisy channels)
    gamma_gain: float = 1.0
    # Multiplier of median-abs to determine clipping threshold for updates
    residual_clip_scale: float = 1.0
    # Minimum history length required before applying full gating
    min_history_for_gain: int = 5
    # Warmup steps during which gating ramps from 0->1
    warmup_steps: int = 10


class OnlineRefinerETS(nn.Module):
    """Residual ETS baseline with parallel channel-wise Holt-style smoothing."""

    def __init__(
        self,
        feature_dim: int,
        device: Optional[torch.device] = None,
        **kwargs,
    ) -> None:
        super().__init__()
        self.device = device or torch.device("cpu")
        self.target_dim = int(feature_dim)

        alpha = float(kwargs.get("alpha", 0.3))
        beta = float(kwargs.get("beta", 0.03))
        damping_factor = float(kwargs.get("damping_factor", 0.98))
        max_history_steps = int(kwargs.get("max_history_steps", 12000))
        gamma_gain = float(kwargs.get("gamma_gain", 1.0))
        residual_clip_scale = float(kwargs.get("residual_clip_scale", 3.0))
        min_history_for_gain = int(kwargs.get("min_history_for_gain", 5))
        warmup_steps = int(kwargs.get("warmup_steps", 3))

        self.config = ETSConfig(
            alpha=min(max(0.0, alpha), 1.0),
            beta=min(max(0.0, beta), 1.0),
            damping_factor=min(max(0.0, damping_factor), 1.0),
            max_history_steps=max(32, max_history_steps),
            gamma_gain=max(0.0, gamma_gain),
            residual_clip_scale=max(0.0, residual_clip_scale),
            min_history_for_gain=max(1, min_history_for_gain),
            warmup_steps=max(0, warmup_steps),
        )

        self.collect_train_windows = 1
        self.expected_H: Optional[int] = None
        self.step_counter: int = 0
        self.training_cycle_count: int = 0

        self.residual_history: Optional[torch.Tensor] = None
        self._cached_residual_tail: Optional[torch.Tensor] = None
        self._level: Optional[torch.Tensor] = None
        self._trend: Optional[torch.Tensor] = None
        self._state_initialized: bool = False
        self.loss_history: list[list[float]] = []

    def reset_state(self, *, clear_loss_history: bool = True) -> None:
        self.expected_H = None
        self.step_counter = 0
        self.training_cycle_count = 0
        self.residual_history = None
        self._cached_residual_tail = None
        self._level = None
        self._trend = None
        self._state_initialized = False
        if clear_loss_history:
            self.loss_history = []

    @staticmethod
    def _median_samples(y: torch.Tensor) -> torch.Tensor:
        if y.ndim == 2:
            return y.unsqueeze(0)
        if y.ndim == 3:
            if y.shape[0] == 1:
                return y
            return torch.median(y, dim=0, keepdim=True)[0]
        raise ValueError(f"Unsupported tensor shape: {tuple(y.shape)}")

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

    def _append_residual_history(self, residual: torch.Tensor) -> None:
        r = residual.detach().to(self.device)
        if r.ndim != 3:
            raise ValueError(f"Expected residual with shape [B, H, D], got {tuple(r.shape)}")
        if r.shape[0] > 1:
            r = r[0:1]
        r2 = r.squeeze(0)

        self._cached_residual_tail = r2[-1:, :].detach().clone()
        if self.residual_history is None:
            self.residual_history = r2
        else:
            self.residual_history = torch.cat([self.residual_history, r2], dim=0)

        max_keep = int(self.config.max_history_steps)
        if self.residual_history.shape[0] > max_keep:
            self.residual_history = self.residual_history[-max_keep:, :]
            self._cached_residual_tail = self.residual_history[-1:, :].detach().clone()

    def _ensure_state_initialized(self, residual_seq: torch.Tensor) -> None:
        if self._state_initialized:
            return
        if residual_seq.ndim != 2 or residual_seq.shape[0] <= 0:
            return
        self._level = residual_seq[0].detach().clone()
        if residual_seq.shape[0] > 1:
            self._trend = (residual_seq[1] - residual_seq[0]).detach().clone()
        else:
            self._trend = torch.zeros_like(self._level)
        self._state_initialized = True

    def _update_state_with_residual_sequence(self, residual_seq: torch.Tensor) -> float:
        seq = residual_seq.detach().to(self.device)
        if seq.ndim != 2 or seq.shape[0] <= 0:
            return float("nan")

        self._ensure_state_initialized(seq)
        if self._level is None or self._trend is None:
            return float("nan")
        # Robustify updates by clipping large residual impulses
        eps = 1e-8
        mad = torch.mean(torch.abs(seq), dim=0) + eps
        clip_thr = mad * float(self.config.residual_clip_scale)
        # elementwise clip while preserving sign
        seq_clipped = torch.sign(seq) * torch.min(torch.abs(seq), clip_thr.unsqueeze(0))

        alpha = float(self.config.alpha)
        beta = float(self.config.beta)
        phi = float(self.config.damping_factor)
        level = self._level
        trend = self._trend

        errors = []
        start_idx = 1 if seq_clipped.shape[0] > 0 else 0
        for idx in range(start_idx, int(seq_clipped.shape[0])):
            obs = seq_clipped[idx]
            forecast = level + (phi * trend)
            err = obs - forecast
            errors.append(err)
            new_level = alpha * obs + (1.0 - alpha) * forecast
            new_trend = beta * (new_level - level) + (1.0 - beta) * (phi * trend)
            level = new_level
            trend = new_trend

        self._level = level.detach().clone()
        self._trend = trend.detach().clone()
        self.training_cycle_count += 1

        if not errors:
            return float("nan")

        err_tensor = torch.stack(errors, dim=0)
        mse = torch.mean(err_tensor ** 2)
        return float(mse.item())

    def _forecast_residual_horizon(self, horizon: int) -> torch.Tensor:
        if self._level is None or self._trend is None:
            return torch.zeros((1, horizon, self.target_dim), device=self.device)

        level = self._level
        trend = self._trend
        phi = float(self.config.damping_factor)
        trend_acc = torch.zeros_like(level)
        out = torch.zeros((horizon, self.target_dim), device=self.device, dtype=level.dtype)

        for step in range(horizon):
            trend_acc = (phi * trend_acc) + (phi * trend)
            out[step, :] = level + trend_acc

        return out.unsqueeze(0)

    def update(
        self,
        x: torch.Tensor,
        y_base: torch.Tensor,
        y_ref_past: torch.Tensor,
        y_gt_full: torch.Tensor,
        *,
        physical_stride: Optional[int] = None,
        step: int = 10,
        e_past_override: Optional[torch.Tensor] = None,
        time_index: Optional[int] = None,
    ) -> list[float]:
        _ = x
        _ = y_ref_past
        _ = physical_stride
        _ = step
        _ = e_past_override
        _ = time_index

        y_base = sanitize_tensor(y_base).detach().to(self.device)
        y_gt_full = sanitize_tensor(y_gt_full).detach().to(self.device)

        y_base_median = self._median_samples(y_base)
        y_gt_aligned = self._align_gt_to_base(y_gt_full, y_base_median)
        residual = y_gt_aligned - y_base_median

        if self.expected_H is None:
            self.expected_H = int(y_base_median.shape[1])

        self._append_residual_history(residual)
        residual_seq = residual.squeeze(0)
        if residual_seq.ndim == 1:
            residual_seq = residual_seq.unsqueeze(0)
        mse = self._update_state_with_residual_sequence(residual_seq)
        self.step_counter += 1

        if math.isfinite(mse):
            self.loss_history.append([float(mse)])

        return []

    @torch.no_grad()
    def predict(self, y_pred_current: torch.Tensor, model_input_seq: Optional[torch.Tensor] = None) -> torch.Tensor:
        _ = model_input_seq
        samples = sanitize_tensor(y_pred_current).to(self.device)
        if samples.ndim == 2:
            samples = samples.unsqueeze(0)

        horizon = int(samples.shape[1])
        if self._level is None or self._trend is None:
            return samples if y_pred_current.ndim == 3 else samples.squeeze(0)

        residual_forecast = self._forecast_residual_horizon(horizon)
        # Compute adaptive gain per channel based on recent residual stability
        gain = 1.0
        if self.residual_history is None:
            gain = 0.0
        else:
            hist_len = int(self.residual_history.shape[0])
            if hist_len < int(self.config.min_history_for_gain):
                # not enough history => don't apply correction yet
                gain = 0.0
            else:
                tail_len = min(hist_len, max(1, int(self.config.min_history_for_gain)))
                tail = self.residual_history[-tail_len:, :]
                # relative std: std / (mean_abs + eps)
                eps = 1e-8
                std = torch.std(tail, dim=0)
                mean_abs = torch.mean(torch.abs(tail), dim=0) + eps
                rstd = std / mean_abs
                gamma = float(self.config.gamma_gain)
                channel_gain = torch.exp(-gamma * rstd)
                # warmup ramp
                if int(self.training_cycle_count) < int(self.config.warmup_steps) and int(self.config.warmup_steps) > 0:
                    ramp = float(self.training_cycle_count) / float(self.config.warmup_steps)
                    channel_gain = channel_gain * ramp
                gain = channel_gain.view(1, 1, -1).to(self.device)

        # If gain is a scalar 0/1, allow broadcasting; else channel-wise
        if isinstance(gain, (int, float)) and (gain == 0.0 or gain == 1.0):
            if gain == 0.0:
                refined = samples
            else:
                refined = samples + residual_forecast
        else:
            refined = samples + (gain * residual_forecast)
        return refined if y_pred_current.ndim == 3 else refined.squeeze(0)
