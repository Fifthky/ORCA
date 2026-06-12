

from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from typing import Optional, List, Tuple, Dict
import collections

import torch
import torch.nn as nn
import torch.nn.functional as F

from core.util.refiner_util import (
    align_sequence_length,
    collapse_batch_median,
    prepare_aligned_sequence_batch,
    sanitize_tensor,
)


@dataclass
class ELFConfig:
    # Interface compatibility parameters
    hidden_dim: int = 128
    num_blocks: int = 2
    lr: float = 1e-3
    
    # True ELF hyperparameters
    M: int = 200            # Update interval measured in stride=1 time steps (paper setting)
    alpha: float = 0.9      # Frequency retention proportion
    lam: float = 20.0       # L2 Regularization parameter
    eta: float = 0.5        # Learning rate for the exponential weighter
    gamma: float = 0.5      # Slow weighter temperature
    B_len: int = 5          # Fast weighter history size
    warmup_steps: int = 5   # Number of block updates before ELF-Forecaster is trusted


class OnlineRefinerELF(nn.Module):
    """
    ELF solver with evaluator-managed causal closure.
    This refiner consumes aligned quartet snapshots and performs pure numerical updates.
    """

    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int = 128,  
        num_blocks: int = 2,    
        lr: float = 1e-3,       
        stride: int | None = None,
        device: Optional[torch.device] = None,
        **kwargs
    ) -> None:
        super().__init__()
        self.device = device or torch.device("cpu")
        self.feature_dim = feature_dim
        self.config = ELFConfig(hidden_dim=hidden_dim, num_blocks=num_blocks, lr=lr)

        self.baseline_router = bool(kwargs.get("baseline_router", False))
        self.ema_error_momentum = float(kwargs.get("ema_error_momentum", 0.2))
        self.routing_temperature = float(kwargs.get("routing_temperature", 0.1))
        self.e_base_moving: Optional[torch.Tensor] = None
        self.e_ref_moving: Optional[torch.Tensor] = None
        self._routing_enabled = False
        self.router_raw_output: Optional[torch.Tensor] = None
        
        # Override M if explicitly passed via kwargs
        if "M" in kwargs:
            self.config.M = int(kwargs["M"])
        if "gamma" in kwargs:
            self.config.gamma = float(kwargs["gamma"])
        # Keep chunking policy internal to ELF: no external CLI/kwargs override.
        self.channel_chunk_size = int(min(100, max(1, int(feature_dim))))

        self.update_stride = int(stride) if stride is not None else None
        self._effective_stride = max(1, int(self.update_stride) if self.update_stride is not None else 1)
        # Keep paper semantics: M is defined in stride=1 rolling steps.
        # In this pipeline, one online update sample advances approximately `stride` time steps,
        # so the block size in samples should be ceil(M / stride).
        self.block_update_size = max(1, int(ceil(float(self.config.M) / float(self._effective_stride))))
        self.expected_L: Optional[int] = None
        
        # Accumulation buffers for Woodbury block update (stores perfect H-length pairs)
        self.buf_X: List[torch.Tensor] = []
        self.buf_Y: List[torch.Tensor] = []
        self.buf_FM: List[torch.Tensor] = []
        # Internal pure-ELF history, intentionally kept inside refiner.
        self.buf_EF_past: collections.deque = collections.deque(maxlen=max(8, int(self.config.B_len) * 4))
        # Evaluator compatibility: non-CSV pipeline expects this field.
        self.loss_history: List[List[float]] = []
        
        # ---------------------------------------------------------------------
        # ELF-Forecaster & Weighter States
        # ---------------------------------------------------------------------
        self.FIRSTFIT = True
        self.NumSeen = 0
        self.update_count = 0
        
        self.welford_M2 = torch.zeros(feature_dim, device=self.device)
        self.welford_count = 0
        
        self.A_inv: Optional[torch.Tensor] = None  
        self.B_mat: Optional[torch.Tensor] = None  
        self.W: Optional[torch.Tensor] = None      
        
        self.sum_loss_fm = torch.zeros(feature_dim, device=self.device)
        self.sum_loss_ef = torch.zeros(feature_dim, device=self.device)
        self.sum_loss_f = torch.zeros(feature_dim, device=self.device)
        self.sum_loss_s = torch.zeros(feature_dim, device=self.device)
        
        self.fast_loss_fm_buffer: collections.deque = collections.deque(maxlen=self.config.B_len)
        self.fast_loss_ef_buffer: collections.deque = collections.deque(maxlen=self.config.B_len)
        
        self.w_tau = torch.full((feature_dim,), 1.0, device=self.device) 
        self.w_slow = torch.full((feature_dim,), 0.5, device=self.device)
        self.w_fast = torch.full((feature_dim,), 0.5, device=self.device)
        self.beta_merge = torch.full((feature_dim,), 0.5, device=self.device)
        self._eye_m_cache: Dict[Tuple[int, torch.dtype], torch.Tensor] = {}

    def _get_eye_m(self, m: int, dtype: torch.dtype) -> torch.Tensor:
        key = (int(m), dtype)
        cached = self._eye_m_cache.get(key)
        if cached is None or cached.device != self.device:
            cached = torch.eye(int(m), dtype=dtype, device=self.device)
            self._eye_m_cache[key] = cached
        return cached

    def _bmm_channel_chunks(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        d = int(left.shape[0])
        out_chunks: list[torch.Tensor] = []
        step = int(max(1, self.channel_chunk_size))
        for i in range(0, d, step):
            j = min(d, i + step)
            out_chunks.append(torch.bmm(left[i:j], right[i:j]))
        return torch.cat(out_chunks, dim=0)

    def _current_sigma(self, dtype: torch.dtype) -> torch.Tensor:
        if self.welford_count <= 0:
            return torch.ones((1, 1, self.feature_dim), device=self.device, dtype=dtype)
        sigma = torch.sqrt(self.welford_M2 / max(1, int(self.welford_count)))
        sigma = torch.clamp(sigma, min=1e-5)
        return sigma.view(1, 1, self.feature_dim).to(dtype=dtype)

    def reset_state(self, *, clear_loss_history: bool = True) -> None:
        self.expected_L = None
        self.FIRSTFIT = True
        self.NumSeen = 0
        self.update_count = 0
        
        self.welford_M2.zero_()
        self.welford_count = 0
        
        self.A_inv = None
        self.B_mat = None
        self.W = None
        
        self.sum_loss_fm.zero_()
        self.sum_loss_ef.zero_()
        self.sum_loss_f.zero_()
        self.sum_loss_s.zero_()
        
        self.fast_loss_fm_buffer.clear()
        self.fast_loss_ef_buffer.clear()
        
        self.w_tau.fill_(1.0)
        self.w_slow.fill_(0.5)
        self.w_fast.fill_(0.5)
        self.beta_merge.fill_(0.5)
        
        self.buf_X.clear()
        self.buf_Y.clear()
        self.buf_FM.clear()
        self.buf_EF_past.clear()
        if clear_loss_history:
            self.loss_history = []
        self.e_base_moving = None
        self.e_ref_moving = None
        self._routing_enabled = False
        self.router_raw_output = None

    def _prepare_input_batch(self, seq_list: List[torch.Tensor]) -> torch.Tensor:
        valid_lengths = [int(x.shape[1]) for x in seq_list if x.ndim == 3 and int(x.shape[1]) > 0]
        if self.expected_L is None:
            self.expected_L = max(valid_lengths) if valid_lengths else 1
        target_len = int(self.expected_L)
        return prepare_aligned_sequence_batch(seq_list, target_len)

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

    def _get_frequency_bounds(self, L: int, H: int) -> Tuple[int, int, int]:
        kept_L = int(self.config.alpha * L)
        k1_L = kept_L // 2
        k2_L = kept_L - k1_L
        kept_H = int(self.config.alpha * H / 2)
        return max(1, k1_L), max(1, k2_L), max(1, kept_H)

    def _predict_ef(self, X: torch.Tensor, H: int) -> torch.Tensor:
        """Generates H-length forecasts using the Fourier domain weights."""
        B, L, D = X.shape
        if self.W is None:
            return X[:, -1:, :].expand(-1, H, -1)

        k1_L, k2_L, kept_H = self._get_frequency_bounds(L, H)
        
        mu = X.mean(dim=1, keepdim=True)
        X_centered = X - mu
        X_fft = torch.fft.fft(X_centered, dim=1, norm='ortho')
        X_filtered = torch.cat([X_fft[:, :k1_L, :], X_fft[:, L-k2_L:, :]], dim=1) 
        
        X_mat = X_filtered.permute(2, 0, 1)
        U_hat = self._bmm_channel_chunks(X_mat, self.W)
        
        U_hat = U_hat.permute(1, 2, 0) 
        pad_len = (H // 2 + 1) - kept_H
        padded = F.pad(U_hat, (0, 0, 0, pad_len)) 
        
        Y_hat = torch.fft.irfft(padded, n=H, dim=1, norm='ortho') 
        Y_hat = Y_hat + mu
        return Y_hat

    def _fit_ef(self, X: torch.Tensor, Y: torch.Tensor) -> None:
        """Algorithm 2/3: Woodbury matrix update expecting strictly H-length sequences."""
        M_batch, L, D = X.shape
        H = Y.shape[1]
        k1_L, k2_L, kept_H = self._get_frequency_bounds(L, H)
        kept_L = k1_L + k2_L
        
        X_centered = X - X.mean(dim=1, keepdim=True)
        
        var_sum = (X_centered ** 2).sum(dim=(0, 1)) 
        self.welford_M2 += var_sum
        self.welford_count += (M_batch * L)
        sigma = self._current_sigma(dtype=X_centered.dtype)
        
        X_scaled = X_centered / sigma
        
        X_fft = torch.fft.fft(X_scaled, dim=1, norm='ortho')
        Y_rft = torch.fft.rfft(Y, dim=1, norm='ortho')
        
        X_filtered = torch.cat([X_fft[:, :k1_L, :], X_fft[:, L-k2_L:, :]], dim=1) 
        Y_filtered = Y_rft[:, :kept_H, :] 
        
        X_mat = X_filtered.permute(2, 0, 1) 
        Y_mat = Y_filtered.permute(2, 0, 1) 

        need_reinit = (
            self.A_inv is None
            or self.B_mat is None
            or int(self.A_inv.shape[-1]) != int(kept_L)
            or int(self.B_mat.shape[-2]) != int(kept_L)
            or int(self.B_mat.shape[-1]) != int(kept_H)
            or int(self.A_inv.shape[0]) != int(D)
        )

        if need_reinit:
            eye_l = torch.eye(kept_L, dtype=X_mat.dtype, device=self.device).unsqueeze(0).expand(D, -1, -1)
            self.A_inv = eye_l / float(self.config.lam)
            self.B_mat = torch.zeros((D, kept_L, kept_H), dtype=X_mat.dtype, device=self.device)
            self.FIRSTFIT = True

        eye_m = self._get_eye_m(M_batch, X_mat.dtype)

        if self.FIRSTFIT or self.NumSeen <= 0:
            # First fit from current block only, matching Algorithm 2 initialization.
            xtx = self._bmm_channel_chunks(X_mat.mH, X_mat) / float(max(1, M_batch))
            lam_eye = float(self.config.lam) * torch.eye(kept_L, dtype=X_mat.dtype, device=self.device).unsqueeze(0).expand(D, -1, -1)
            self.A_inv = torch.linalg.inv(xtx + lam_eye)
            self.B_mat = self._bmm_channel_chunks(X_mat.mH, Y_mat) / float(max(1, M_batch))
            self.FIRSTFIT = False
            self.NumSeen = int(M_batch)
        else:
            # Complex-valued Woodbury update in normalized (per-instance) form for stability.
            prev_seen = int(max(1, self.NumSeen))
            next_seen = int(prev_seen + M_batch)

            d_step = int(max(1, self.channel_chunk_size))
            for i in range(0, int(D), d_step):
                j = min(int(D), i + d_step)

                x_i = X_mat[i:j]
                a_scaled = self.A_inv[i:j] / float(prev_seen)
                eye_i = eye_m.unsqueeze(0).expand(j - i, -1, -1)

                xa = torch.bmm(x_i, a_scaled)
                inner = eye_i + torch.bmm(xa, x_i.mH)
                solved = torch.linalg.solve(inner, xa)
                updated_scaled = a_scaled - torch.bmm(torch.bmm(a_scaled, x_i.mH), solved)
                self.A_inv[i:j] = updated_scaled * float(next_seen)

            xhy = self._bmm_channel_chunks(X_mat.mH, Y_mat)
            self.B_mat = (float(prev_seen) * self.B_mat + xhy) / float(next_seen)
            self.NumSeen = next_seen

        # Channel-wise Fourier linear map.
        self.W = self._bmm_channel_chunks(self.A_inv, self.B_mat)

    def _update_weighter(self, Y_FM: torch.Tensor, Y_EF: torch.Tensor, Y_gt: torch.Tensor) -> None:
        """Algorithm 5: update slow/fast/merge weighters with MAE-based losses."""
        eps = 1e-8

        def _mae_loss(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
            mae = torch.mean(torch.abs(pred - gt), dim=(0, 1))
            return mae

        L_FM = _mae_loss(Y_FM, Y_gt)
        L_EF = _mae_loss(Y_EF, Y_gt)
        
        w_s_view = self.w_slow.view(1, 1, -1)
        w_f_view = self.w_fast.view(1, 1, -1)
        
        Y_s = w_s_view * Y_FM + (1.0 - w_s_view) * Y_EF
        Y_f = w_f_view * Y_FM + (1.0 - w_f_view) * Y_EF
        
        L_s = _mae_loss(Y_s, Y_gt)
        L_f = _mae_loss(Y_f, Y_gt)
        
        self.sum_loss_fm += L_FM
        self.sum_loss_ef += L_EF
        self.sum_loss_s += L_s
        self.sum_loss_f += L_f
        
        self.fast_loss_fm_buffer.append(L_FM)
        self.fast_loss_ef_buffer.append(L_EF)

        # Slow weighter: multiplicative-weights update with previous w_slow and current losses.
        slow_num = self.w_slow * torch.exp(-self.config.gamma * L_FM)
        slow_den = slow_num + (1.0 - self.w_slow) * torch.exp(-self.config.gamma * L_EF)
        self.w_slow = slow_num / slow_den.clamp(min=eps)

        # Fast weighter: multiplicative-weights update over rolling window B.
        stack_fast_fm = torch.stack(list(self.fast_loss_fm_buffer), dim=0)
        stack_fast_ef = torch.stack(list(self.fast_loss_ef_buffer), dim=0)
        sum_fast_fm = stack_fast_fm.sum(dim=0)
        sum_fast_ef = stack_fast_ef.sum(dim=0)
        fast_num = torch.exp(-self.config.eta * sum_fast_fm)
        fast_den = fast_num + torch.exp(-self.config.eta * sum_fast_ef)
        self.w_fast = fast_num / fast_den.clamp(min=eps)

        # Merge weighter: choose between fast and slow combined forecasts.
        merge_num = self.beta_merge * torch.exp(-self.config.eta * L_f)
        merge_den = merge_num + (1.0 - self.beta_merge) * torch.exp(-self.config.eta * L_s)
        self.beta_merge = merge_num / merge_den.clamp(min=eps)

        self.w_tau = self.beta_merge * self.w_fast + (1.0 - self.beta_merge) * self.w_slow
        self.w_tau = torch.clamp(self.w_tau, min=0.0, max=1.0)

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
        # ELF Algorithm 5 uses FM and pure EF losses only; evaluator-provided
        # refined past forecast is intentionally ignored to avoid loss pollution.
        # Temporal causality is owned by the online wrapper; delayed errors are
        # consumed upstream and are not needed by ELF internal updates.
        del physical_stride, step, e_past_override

        X = sanitize_tensor(X).detach().to(self.device)
        Y_base = sanitize_tensor(Y_base).detach().to(self.device)
        Y_ref_past = sanitize_tensor(Y_ref_past).detach().to(self.device)
        Y_GT_full = sanitize_tensor(Y_GT_full).detach().to(self.device)

        if Y_GT_full.ndim == 3 and Y_GT_full.shape[0] > 1:
            Y_GT_full = Y_GT_full[0:1]
        Y_base = collapse_batch_median(Y_base)
        X = collapse_batch_median(X)
        Y_GT_full = self._align_gt_to_base(Y_GT_full, Y_base)
        if self.baseline_router:
            y_ref_past_median = collapse_batch_median(Y_ref_past)
            if y_ref_past_median.shape != Y_base.shape:
                y_ref_past_median = Y_base
            err_base = torch.mean(torch.abs(Y_GT_full - Y_base), dim=(0, 1))
            err_ref = torch.mean(torch.abs(Y_GT_full - y_ref_past_median), dim=(0, 1))
            alpha = float(self.ema_error_momentum)
            if self.e_base_moving is None or self.e_ref_moving is None:
                self.e_base_moving = err_base.detach()
                self.e_ref_moving = err_ref.detach()
            else:
                self.e_base_moving = alpha * err_base + (1.0 - alpha) * self.e_base_moving
                self.e_ref_moving = alpha * err_ref + (1.0 - alpha) * self.e_ref_moving
            self._routing_enabled = True

        self.buf_X.append(X)
        self.buf_Y.append(Y_GT_full)
        self.buf_FM.append(Y_base)

        if len(self.buf_X) >= int(self.block_update_size):
            X_batch = self._prepare_input_batch(self.buf_X)
            Y_batch = torch.cat(self.buf_Y, dim=0)
            FM_batch = torch.cat(self.buf_FM, dim=0)
            expected_h = int(Y_batch.shape[1])

            # Must evaluate EF with old weights before any fitting update.
            with torch.no_grad():
                EF_batch = self._predict_ef(X_batch, expected_h)
            self.buf_EF_past.append(EF_batch.detach())

            self._update_weighter(FM_batch, EF_batch, Y_batch)

            # Track one block-level scalar loss for logging and compatibility.
            with torch.no_grad():
                w_view = self.w_tau.view(1, 1, -1)
                mix_batch = w_view * FM_batch + (1.0 - w_view) * EF_batch
                block_mae = torch.mean(torch.abs(mix_batch - Y_batch), dim=(0, 1))
                block_loss = block_mae.mean()
                self.loss_history.append([float(block_loss.detach().cpu().item())])

            self._fit_ef(X_batch, Y_batch)

            self.update_count += 1
            self.buf_X.clear()
            self.buf_Y.clear()
            self.buf_FM.clear()

        return []

    @torch.no_grad()
    def predict(self, y_pred_current: torch.Tensor, model_input_seq: Optional[torch.Tensor] = None) -> torch.Tensor:
        z0 = sanitize_tensor(y_pred_current).to(self.device)

        original_samples = z0
        
        # Consolidate probabilistic ensemble into a robust median spine
        base_median = z0
        if z0.ndim == 3 and z0.shape[0] > 1:
            base_median = z0.median(dim=0, keepdim=True).values
            
        if model_input_seq is None or model_input_seq.shape[1] == 0:
            if self.baseline_router:
                self.router_raw_output = z0.detach()
            else:
                self.router_raw_output = None
            return z0
            
        model_input_seq = sanitize_tensor(model_input_seq).to(self.device)

        if model_input_seq.ndim == 3 and model_input_seq.shape[0] > 1:
            model_input_seq = collapse_batch_median(model_input_seq)

        if model_input_seq.shape[0] == 1 and base_median.shape[0] > 1:
            model_input_seq = model_input_seq.expand(base_median.shape[0], -1, -1)

        if self.expected_L is not None:
            model_input_seq = align_sequence_length(model_input_seq, int(self.expected_L))
            
        if self.update_count < self.config.warmup_steps:
            if self.baseline_router:
                self.router_raw_output = z0.detach()
            else:
                self.router_raw_output = None
            return z0

        if self.W is None:
            if self.baseline_router:
                self.router_raw_output = z0.detach()
            else:
                self.router_raw_output = None
            return z0
            
        y_ef = self._predict_ef(model_input_seq, int(base_median.shape[1]))

        w_view = self.w_tau.view(1, 1, -1)
        final_forecast_median = w_view * base_median + (1.0 - w_view) * y_ef

        if self.baseline_router:
            if original_samples.ndim == 3 and original_samples.shape[0] > 1:
                ref_delta = y_ef - base_median
                raw_samples = original_samples + ref_delta.expand_as(original_samples)
            else:
                raw_samples = y_ef
            self.router_raw_output = raw_samples.detach()

            if self._routing_enabled and self.e_base_moving is not None and self.e_ref_moving is not None:
                tau = max(1e-6, float(self.routing_temperature))
                exp_base = torch.exp(-self.e_base_moving / tau)
                exp_ref = torch.exp(-self.e_ref_moving / tau)
                c_t = exp_ref / (exp_base + exp_ref + 1e-8)
                c_t = c_t.view(1, 1, self.feature_dim)
            else:
                c_t = torch.ones(1, 1, self.feature_dim, device=self.device)

            mixed_median = base_median + (y_ef - base_median) * c_t
            if original_samples.ndim == 3 and original_samples.shape[0] > 1:
                delta_curve = mixed_median - base_median
                routed_samples = original_samples + delta_curve.expand_as(original_samples)
            else:
                routed_samples = mixed_median
            return routed_samples

        if original_samples.ndim == 3 and original_samples.shape[0] > 1:
            delta_curve = final_forecast_median - base_median
            raw_samples = original_samples + delta_curve.expand_as(original_samples)
        else:
            raw_samples = final_forecast_median

        self.router_raw_output = None
        return raw_samples