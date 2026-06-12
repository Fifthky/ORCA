from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from core.util.refiner_util import sanitize_tensor


@dataclass
class RidgeConfig:
    # Number of snapshots required before first fit.
    collect_train_windows: int = 512
    # L2 regularization coefficient.
    ridge_lambda: float = 1e-3
    # Re-fit once every horizon cycle after warmup.
    fit_every_cycles: int = 1
    # Robust gating and clipping params
    max_history_steps: int = 2048
    gamma_gain: float = 1.0
    residual_clip_scale: float = 3.0
    min_history_for_gain: int = 5
    warmup_steps: int = 10


class OnlineRefinerRidge(nn.Module):
    """XY-concatenated ridge baseline for fast residual correction."""

    def __init__(
        self,
        feature_dim: int,
        device: Optional[torch.device] = None,
        **kwargs,
    ) -> None:
        super().__init__()
        self.device = device or torch.device("cpu")
        self.target_dim = int(feature_dim)

        collect_train_windows = int(kwargs.get("collect_train_windows", 512))
        ridge_lambda = float(kwargs.get("ridge_lambda", 1e-3))
        fit_every_cycles = int(kwargs.get("fit_every_cycles", 1))
        max_history_steps = int(kwargs.get("max_history_steps", 2048))
        gamma_gain = float(kwargs.get("gamma_gain", 1.0))
        residual_clip_scale = float(kwargs.get("residual_clip_scale", 3.0))
        min_history_for_gain = int(kwargs.get("min_history_for_gain", 5))
        warmup_steps = int(kwargs.get("warmup_steps", 3))

        self.config = RidgeConfig(
            collect_train_windows=max(1, collect_train_windows),
            ridge_lambda=max(0.0, ridge_lambda),
            fit_every_cycles=max(1, fit_every_cycles),
            max_history_steps=max(1, max_history_steps),
            gamma_gain=max(0.0, gamma_gain),
            residual_clip_scale=max(0.0, residual_clip_scale),
            min_history_for_gain=max(1, min_history_for_gain),
            warmup_steps=max(0, warmup_steps),
        )

        self.collect_train_windows = int(self.config.collect_train_windows)
        self.expected_H: Optional[int] = None
        self.expected_L: Optional[int] = None
        self.step_counter: int = 0
        self.training_cycle_count: int = 0
        self.is_warmed_up: bool = False

        self._feature_buffer: Deque[torch.Tensor] = deque(maxlen=self.collect_train_windows)
        self._target_buffer: Deque[torch.Tensor] = deque(maxlen=self.collect_train_windows)

        self._weights: Optional[torch.Tensor] = None
        self._bias: Optional[torch.Tensor] = None
        self.residual_history: Optional[torch.Tensor] = None
        self._cached_residual_tail: Optional[torch.Tensor] = None
        self.loss_history: list[list[float]] = []

    def reset_state(self, *, clear_loss_history: bool = True) -> None:
        self.expected_H = None
        self.expected_L = None
        self.step_counter = 0
        self.training_cycle_count = 0
        self.is_warmed_up = False
        self._feature_buffer.clear()
        self._target_buffer.clear()
        self._weights = None
        self._bias = None
        self.residual_history = None
        self._cached_residual_tail = None
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

    def _prepare_xy_features(self, x: torch.Tensor, y_base_median: torch.Tensor) -> torch.Tensor:
        h = int(y_base_median.shape[1])
        if x.ndim == 2:
            x = x.unsqueeze(0)
        if x.ndim != 3:
            raise ValueError(f"Expected X tensor with 3D shape, got {tuple(x.shape)}")

        if x.shape[1] < h:
            x_h = F.pad(x, (0, 0, h - x.shape[1], 0))
        else:
            x_h = x[:, -h:, :]

        return torch.cat([x_h, y_base_median], dim=-1)

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

    def _fit_ridge(self) -> None:
        if len(self._feature_buffer) <= 0:
            return

        feat = torch.cat(list(self._feature_buffer), dim=0)
        tgt = torch.cat(list(self._target_buffer), dim=0)
        bsz, horizon, feat_dim = feat.shape

        x2 = feat.reshape(bsz * horizon, feat_dim)
        y2 = tgt.reshape(bsz * horizon, self.target_dim)

        ones = torch.ones((x2.shape[0], 1), device=self.device, dtype=x2.dtype)
        x_aug = torch.cat([x2, ones], dim=1)

        xtx = x_aug.T @ x_aug
        lam = float(self.config.ridge_lambda)
        if lam > 0.0:
            eye = torch.eye(xtx.shape[0], device=self.device, dtype=xtx.dtype)
            xtx = xtx + lam * eye
        xty = x_aug.T @ y2

        try:
            wb = torch.linalg.solve(xtx, xty)
        except Exception:
            wb = torch.linalg.pinv(xtx) @ xty

        self._weights = wb[:-1, :]
        self._bias = wb[-1:, :]

        with torch.no_grad():
            pred = x_aug @ wb
            mse = torch.mean((pred - y2) ** 2)
            self.loss_history.append([float(mse.item())])

        self.is_warmed_up = True
        self.training_cycle_count += 1

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
        _ = physical_stride
        _ = step
        _ = y_ref_past
        _ = e_past_override
        _ = time_index

        x = sanitize_tensor(x).detach().to(self.device)
        y_base = sanitize_tensor(y_base).detach().to(self.device)
        y_gt_full = sanitize_tensor(y_gt_full).detach().to(self.device)

        y_base_median = self._median_samples(y_base)
        y_gt_aligned = self._align_gt_to_base(y_gt_full, y_base_median)

        if self.expected_H is None:
            self.expected_H = int(y_base_median.shape[1])
        if self.expected_L is None:
            self.expected_L = int(x.shape[1]) if x.ndim == 3 else 1

        feat = self._prepare_xy_features(x, y_base_median)
        residual = y_gt_aligned - y_base_median

        # Robust update: clip residual impulses before storing/training
        eps = 1e-8
        # compute per-channel mean-abs as robust scale
        mad = torch.mean(torch.abs(residual.detach()), dim=(0, 1)) + eps
        clip_thr = mad * float(self.config.residual_clip_scale)
        # residual shape [B, H, D] -> elementwise clip preserving sign
        abs_res = torch.abs(residual)
        clip_thr_tensor = clip_thr.view(1, 1, -1).to(residual.device)
        residual_clipped = torch.sign(residual) * torch.min(abs_res, clip_thr_tensor)

        # append clipped residuals to target buffer and residual history
        self._feature_buffer.append(feat)
        self._target_buffer.append(residual_clipped)
        try:
            self._append_residual_history(residual_clipped)
        except Exception:
            # If history append fails, ignore to keep robustness
            pass

        self.step_counter += 1

        if len(self._feature_buffer) >= int(self.collect_train_windows):
            h = max(1, int(self.expected_H or 1))
            cycle_ok = (self.step_counter % h) == 0
            cycle_spacing = (self.training_cycle_count % int(self.config.fit_every_cycles)) == 0
            if cycle_ok and cycle_spacing:
                self._fit_ridge()

        return []

    @torch.no_grad()
    def predict(self, y_pred_current: torch.Tensor, model_input_seq: Optional[torch.Tensor] = None) -> torch.Tensor:
        samples = sanitize_tensor(y_pred_current).to(self.device)
        if samples.ndim == 2:
            samples = samples.unsqueeze(0)

        if self._weights is None or self._bias is None or model_input_seq is None:
            return samples if y_pred_current.ndim == 3 else samples.squeeze(0)

        y_base_median = self._median_samples(samples)
        x = sanitize_tensor(model_input_seq).to(self.device)
        feat = self._prepare_xy_features(x, y_base_median)

        modifier = feat.reshape(-1, feat.shape[-1]) @ self._weights + self._bias
        modifier = modifier.reshape(1, y_base_median.shape[1], self.target_dim)

        # Compute adaptive gain per channel based on recent residual stability
        gain = 1.0
        if self.residual_history is None:
            gain = 0.0
        else:
            hist_len = int(self.residual_history.shape[0])
            if hist_len < int(self.config.min_history_for_gain):
                gain = 0.0
            else:
                tail_len = min(hist_len, max(1, int(self.config.min_history_for_gain)))
                tail = self.residual_history[-tail_len:, :]
                eps = 1e-8
                std = torch.std(tail, dim=0)
                mean_abs = torch.mean(torch.abs(tail), dim=0) + eps
                rstd = std / mean_abs
                gamma = float(self.config.gamma_gain)
                channel_gain = torch.exp(-gamma * rstd)
                if int(self.training_cycle_count) < int(self.config.warmup_steps) and int(self.config.warmup_steps) > 0:
                    ramp = float(self.training_cycle_count) / float(self.config.warmup_steps)
                    channel_gain = channel_gain * ramp
                gain = channel_gain.view(1, 1, -1).to(self.device)

        if isinstance(gain, (int, float)) and (gain == 0.0 or gain == 1.0):
            if gain == 0.0:
                out = samples
            else:
                refined_median = y_base_median + modifier
                shift = refined_median - y_base_median
                out = samples + shift
        else:
            out = samples + (gain * modifier)

        return out if y_pred_current.ndim == 3 else out.squeeze(0)
