from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

from core.util.refiner_util import sanitize_tensor


@dataclass
class ArimaConfig:
    # Residual ARIMA order; this implementation uses fast ARIMA(p, d, 0).
    p: int = 3
    d: int = 1
    q: int = 0
    # Maximum stored residual timesteps.
    max_history_steps: int = 12000
    # Refit interval in update calls.
    refit_interval: int = 16
    # Ridge regularization for stable autoregressive fit.
    ridge_lambda: float = 1e-4


class OnlineRefinerARIMA(nn.Module):
    """Residual-only ARIMA baseline with lightweight autoregressive fitting."""

    def __init__(
        self,
        feature_dim: int,
        device: Optional[torch.device] = None,
        **kwargs,
    ) -> None:
        super().__init__()
        self.device = device or torch.device("cpu")
        self.target_dim = int(feature_dim)

        p = int(kwargs.get("arima_p", kwargs.get("p", 3)))
        d = int(kwargs.get("arima_d", kwargs.get("d", 1)))
        q = int(kwargs.get("arima_q", kwargs.get("q", 0)))
        max_history_steps = int(kwargs.get("max_history_steps", 12000))
        refit_interval = int(kwargs.get("refit_interval", 16))
        ridge_lambda = float(kwargs.get("ridge_lambda", 1e-4))

        self.config = ArimaConfig(
            p=max(1, p),
            d=max(0, min(1, d)),
            q=max(0, q),
            max_history_steps=max(32, max_history_steps),
            refit_interval=max(1, refit_interval),
            ridge_lambda=max(0.0, ridge_lambda),
        )

        self.collect_train_windows = 1
        self.expected_H: Optional[int] = None
        self._effective_refit_interval: Optional[int] = None
        self.step_counter: int = 0
        self.training_cycle_count: int = 0

        self.residual_history: Optional[torch.Tensor] = None
        self._cached_residual_tail: Optional[torch.Tensor] = None
        self._cached_diff_history: Optional[torch.Tensor] = None
        self._coef: Optional[torch.Tensor] = None  # [D, p]
        self._bias: Optional[torch.Tensor] = None  # [D]
        self._fitted: bool = False
        self.loss_history: list[list[float]] = []

    def reset_state(self, *, clear_loss_history: bool = True) -> None:
        self.expected_H = None
        self._effective_refit_interval = None
        self.step_counter = 0
        self.training_cycle_count = 0
        self.residual_history = None
        self._cached_residual_tail = None
        self._cached_diff_history = None
        self._coef = None
        self._bias = None
        self._fitted = False
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

        prev_tail = self._cached_residual_tail
        has_prev_tail = prev_tail is not None

        if self.residual_history is None:
            self.residual_history = r2
        else:
            self.residual_history = torch.cat([self.residual_history, r2], dim=0)

        self._cached_residual_tail = r2[-1:, :].detach().clone()

        if int(self.config.d) == 1:
            if r2.shape[0] > 0:
                local_diffs = r2[1:, :] - r2[:-1, :] if r2.shape[0] > 1 else None
                cross_diff = None
                if has_prev_tail:
                    cross_diff = r2[0:1, :] - prev_tail
                if cross_diff is not None and local_diffs is not None:
                    new_diffs = torch.cat([cross_diff, local_diffs], dim=0)
                elif cross_diff is not None:
                    new_diffs = cross_diff
                elif local_diffs is not None:
                    new_diffs = local_diffs
                else:
                    new_diffs = None

                if new_diffs is not None:
                    if self._cached_diff_history is None:
                        self._cached_diff_history = new_diffs.detach().clone()
                    else:
                        self._cached_diff_history = torch.cat([self._cached_diff_history, new_diffs.detach().clone()], dim=0)

        max_keep = int(self.config.max_history_steps)
        if self.residual_history.shape[0] > max_keep:
            self.residual_history = self.residual_history[-max_keep:, :]
            self._cached_residual_tail = self.residual_history[-1:, :].detach().clone()
            if self._cached_diff_history is not None:
                max_diff_keep = max(1, max_keep - 1)
                if self._cached_diff_history.shape[0] > max_diff_keep:
                    self._cached_diff_history = self._cached_diff_history[-max_diff_keep:, :]

    def _build_training_series(self) -> Optional[torch.Tensor]:
        if self.residual_history is None:
            return None

        hist = self.residual_history
        if int(self.config.d) == 1:
            if self._cached_diff_history is not None:
                return self._cached_diff_history
            if hist.shape[0] <= 1:
                return None
            return hist[1:, :] - hist[:-1, :]
        return hist

    def _fit_ar_component(self) -> None:
        series = self._build_training_series()
        if series is None:
            return

        p = int(self.config.p)
        lam = float(self.config.ridge_lambda)

        t_len = int(series.shape[0])
        if t_len <= p:
            return

        x_rows = []
        y_rows = []
        for t in range(p, t_len):
            x_rows.append(series[t - p:t, :].flip(0))
            y_rows.append(series[t, :])

        x_seq = torch.stack(x_rows, dim=0)  # [N, p, D]
        y_seq = torch.stack(y_rows, dim=0)  # [N, D]

        x_seq = x_seq.permute(2, 0, 1).contiguous()  # [D, N, p]
        y_seq = y_seq.permute(1, 0).contiguous()  # [D, N]

        ones = torch.ones((x_seq.shape[0], x_seq.shape[1], 1), device=self.device, dtype=x_seq.dtype)
        x_aug = torch.cat([x_seq, ones], dim=-1)  # [D, N, p+1]

        xtx = torch.matmul(x_aug.transpose(1, 2), x_aug)  # [D, p+1, p+1]
        if lam > 0.0:
            eye = torch.eye(xtx.shape[-1], device=self.device, dtype=xtx.dtype).unsqueeze(0)
            xtx = xtx + lam * eye
        xty = torch.matmul(x_aug.transpose(1, 2), y_seq.unsqueeze(-1))  # [D, p+1, 1]

        try:
            wb = torch.linalg.solve(xtx, xty).squeeze(-1)  # [D, p+1]
        except Exception:
            wb = torch.matmul(torch.linalg.pinv(xtx), xty).squeeze(-1)

        self._coef = wb[:, :-1]
        self._bias = wb[:, -1]

        with torch.no_grad():
            pred = torch.matmul(x_aug, wb.unsqueeze(-1)).squeeze(-1)
            mse = torch.mean((pred - y_seq) ** 2)
            self.loss_history.append([float(mse.item())])

        self._fitted = True
        self.training_cycle_count += 1

    def _forecast_residual_horizon(self, horizon: int) -> torch.Tensor:
        if self.residual_history is None or self._coef is None or self._bias is None:
            return torch.zeros((1, horizon, self.target_dim), device=self.device)

        p = int(self.config.p)

        levels = self.residual_history
        if int(self.config.d) == 1:
            if levels.shape[0] <= 1:
                return torch.zeros((1, horizon, self.target_dim), device=self.device)
            diff_hist = self._cached_diff_history
            if diff_hist is None:
                diff_hist = levels[1:, :] - levels[:-1, :]
            if diff_hist.shape[0] < p:
                return torch.zeros((1, horizon, self.target_dim), device=self.device)
            state = diff_hist[-p:, :].clone()  # [p, D]
            pred_diff = torch.zeros((horizon, self.target_dim), device=self.device, dtype=levels.dtype)
            for h in range(horizon):
                nxt = torch.sum(self._coef * state.flip(0).T, dim=1) + self._bias
                pred_diff[h, :] = nxt
                if p > 1:
                    state = torch.cat([state[1:, :], nxt.view(1, -1)], dim=0)
                else:
                    state[0, :] = nxt

            out = torch.zeros((horizon, self.target_dim), device=self.device, dtype=levels.dtype)
            prev = levels[-1, :].clone()
            for h in range(horizon):
                prev = prev + pred_diff[h, :]
                out[h, :] = prev
            return out.unsqueeze(0)

        if levels.shape[0] < p:
            return torch.zeros((1, horizon, self.target_dim), device=self.device)

        state = levels[-p:, :].clone()  # [p, D]
        out = torch.zeros((horizon, self.target_dim), device=self.device, dtype=levels.dtype)
        for h in range(horizon):
            nxt = torch.sum(self._coef * state.flip(0).T, dim=1) + self._bias
            out[h, :] = nxt
            if p > 1:
                state = torch.cat([state[1:, :], nxt.view(1, -1)], dim=0)
            else:
                state[0, :] = nxt

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
        self.step_counter += 1

        if self._effective_refit_interval is None:
            self._effective_refit_interval = max(1, int(self.expected_H or self.config.refit_interval))

        need_fit = (not self._fitted) or (self.step_counter % int(self._effective_refit_interval) == 0)
        if need_fit:
            self._fit_ar_component()

        return []

    @torch.no_grad()
    def predict(self, y_pred_current: torch.Tensor, model_input_seq: Optional[torch.Tensor] = None) -> torch.Tensor:
        _ = model_input_seq
        samples = sanitize_tensor(y_pred_current).to(self.device)
        if samples.ndim == 2:
            samples = samples.unsqueeze(0)

        horizon = int(samples.shape[1])
        modifier = self._forecast_residual_horizon(horizon)
        out = samples + modifier

        return out if y_pred_current.ndim == 3 else out.squeeze(0)
