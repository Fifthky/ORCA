

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Optional, List, Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from core.util.refiner_util import (
    advance_period_clock,
    sanitize_tensor,
)


@dataclass
class LinearConfig:
    # V11.0 Covariate-DLinear Configuration
    hidden_dim: int = 128                
    num_blocks: int = 2                  
    
    lr: float = 1e-4
    weight_decay: float = 1e-5 
    
    collect_train_windows: Optional[int] = None
    collect_val_windows: Optional[int] = None
    max_epochs: int = 100
    batch_size: int = 256
    early_stop_patience: int = 20        
    
    train_sample_ratio: float = 1.0      
    mixer_reg_lambda: float = 1e-3       
    
    ema_error_momentum: float = 0.2
    routing_temperature: float = 0.1
    refiner_input: str = "all"
    online_training: bool = False
    update_rule: str = "plain"
    short_pred_len_threshold: int = 30
    ma_kernel_short: int = 7
    ma_kernel_long: int = 25
    channel_mix: bool = True
    
    # NEW: Gate Lock Switch. If True, always fully trust the refiner (c_t = 1.0)
    force_gate_open: bool = False


class MovingAverage(nn.Module):
    def __init__(self, kernel_size: int, stride: int = 1):
        super().__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x_pad = torch.cat([front, x, end], dim=1)
        x_trend = self.avg(x_pad.permute(0, 2, 1)).permute(0, 2, 1)
        return x_trend


class SeriesDecomp(nn.Module):
    def __init__(self, kernel_size: int = 25):
        super().__init__()
        self.moving_avg = MovingAverage(kernel_size, stride=1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean  # season, trend


class ChannelMixerBlock(nn.Module):
    def __init__(self, channels: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(channels)
        self.fc1 = nn.Linear(channels, hidden_dim)
        self.act = nn.GELU()
        self.dropout1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, channels)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = x
        x = self.norm(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.dropout2(x)
        return res + x


class CovariateDLinearMixerCore(nn.Module):
    """
    V11.0 Backbone: E_past as Primary Variable, X and Y_base as Covariates.
    All components decomposed before concatenation.
    """
    def __init__(
        self, 
        seq_len: int, 
        pred_len: int, 
        feature_dim: int, 
        hidden_dim: int, 
        num_blocks: int, 
        refiner_input: str = "all",
        short_pred_len_threshold: int = 30,
        ma_kernel_short: int = 7,
        ma_kernel_long: int = 25,
        channel_mix: bool = True,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.feature_dim = feature_dim
        self.channel_mix = bool(channel_mix)
        self.refiner_input = str(refiner_input).strip().lower()
        if self.refiner_input not in {"all", "xy", "x", "y", "e_past"}:
            raise ValueError(f"Unsupported refiner_input={refiner_input!r}. Expected one of: all, xy, x, y, e_past")

        ma_kernel = self._resolve_ma_kernel(
            pred_len=int(pred_len),
            short_pred_len_threshold=int(short_pred_len_threshold),
            ma_kernel_short=int(ma_kernel_short),
            ma_kernel_long=int(ma_kernel_long),
        )
        self.decomp = SeriesDecomp(kernel_size=ma_kernel)
        
        if self.refiner_input == "all":
            in_context_len = seq_len + 2 * pred_len
        elif self.refiner_input == "xy":
            in_context_len = seq_len + pred_len
        elif self.refiner_input == "x":
            in_context_len = seq_len
        elif self.refiner_input == "y":
            in_context_len = pred_len
        else:
            in_context_len = pred_len
        
        # Linear mappers for the concatenated covariates -> future residual
        self.linear_trend = nn.Linear(in_context_len, pred_len)
        self.linear_season = nn.Linear(in_context_len, pred_len)
        
        self.channel_mixers = nn.ModuleList(
            [
                ChannelMixerBlock(channels=feature_dim, hidden_dim=hidden_dim)
                for _ in range(num_blocks)
            ]
            if self.channel_mix
            else []
        )
        
        self.gamma = nn.Parameter(torch.zeros(1, 1, 1))
        self.out_proj = nn.Linear(pred_len, pred_len)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)
        self._last_z_tr: torch.Tensor | None = None
        self._last_z_se: torch.Tensor | None = None

    @staticmethod
    def _resolve_ma_kernel(
        *,
        pred_len: int,
        short_pred_len_threshold: int,
        ma_kernel_short: int,
        ma_kernel_long: int,
    ) -> int:
        kernel = int(ma_kernel_short) if int(pred_len) <= int(short_pred_len_threshold) else int(ma_kernel_long)
        if kernel < 1:
            kernel = 1
        # Keep odd kernel size for symmetric left/right padding.
        if kernel % 2 == 0:
            kernel += 1
        return kernel

    def forward(self, e_past: torch.Tensor, x_norm: torch.Tensor, y_base_norm: torch.Tensor) -> torch.Tensor:
        # 1. Decompose ALL variables
        e_se, e_tr = self.decomp(e_past)
        x_se, x_tr = self.decomp(x_norm)
        y_se, y_tr = self.decomp(y_base_norm)
        
        # 2. Covariate In-Context Concatenation
        # Order: [Past Error, Past Context, Future Draft]
        if self.refiner_input == "all":
            z_tr = torch.cat([e_tr, x_tr, y_tr], dim=1)  # [B, L+2H, D]
            z_se = torch.cat([e_se, x_se, y_se], dim=1)  # [B, L+2H, D]
        elif self.refiner_input == "xy":
            z_tr = torch.cat([x_tr, y_tr], dim=1)  # [B, L+H, D]
            z_se = torch.cat([x_se, y_se], dim=1)  # [B, L+H, D]
        elif self.refiner_input == "x":
            z_tr = x_tr  # [B, L, D]
            z_se = x_se  # [B, L, D]
        elif self.refiner_input == "y":
            z_tr = y_tr  # [B, H, D]
            z_se = y_se  # [B, H, D]
        else:
            z_tr = e_tr  # [B, H, D]
            z_se = e_se  # [B, H, D]

        self._last_z_tr = z_tr.detach()
        self._last_z_se = z_se.detach()
        
        # 3. Channel Independent Mapping
        delta_y_tr = self.linear_trend(z_tr.transpose(1, 2)).transpose(1, 2)
        delta_y_se = self.linear_season(z_se.transpose(1, 2)).transpose(1, 2)
        
        delta_y_ci = delta_y_tr + delta_y_se   # [B, H, D]
        
        # 4. TSMixer Channel Communication
        if self.channel_mix and len(self.channel_mixers) > 0:
            cross_feat = delta_y_ci
            for mixer in self.channel_mixers:
                cross_feat = mixer(cross_feat)
            # 5. Zero-Init Fusion
            delta_y_final = delta_y_ci + self.gamma * cross_feat
        else:
            delta_y_final = delta_y_ci
        return self.out_proj(delta_y_final.transpose(1, 2)).transpose(1, 2)


class OnlineRefinerLinear(nn.Module):
    """
    V11.0 - Covariate Paradigm with Strict Anti-Leakage Delay Queue.
    """

    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int = 128,
        num_blocks: int = 2,
        lr: Optional[float] = None,
        device: Optional[torch.device] = None,
        **kwargs
    ) -> None:
        super().__init__()
        self.device = device or torch.device("cpu")
        self.target_dim = feature_dim
        
        self.config = LinearConfig(
            hidden_dim=hidden_dim,
            num_blocks=num_blocks,
        )
        if lr is not None:
            self.config.lr = float(lr)
        if "collect_train_windows" not in kwargs or "collect_val_windows" not in kwargs:
            raise ValueError("Linear requires explicitly provided window counts.")
        self.config.collect_train_windows = max(1, int(kwargs["collect_train_windows"]))
        self.config.collect_val_windows = max(1, int(kwargs["collect_val_windows"]))
        
        # Apply external configurations
        for k, v in kwargs.items():
            if hasattr(self.config, k):
                setattr(self.config, k, v)
        self.config.lr = max(1e-12, float(self.config.lr))
        self.config.weight_decay = max(0.0, float(self.config.weight_decay))
        self.config.update_rule = self._normalize_update_rule(self.config.update_rule)
        
        self.is_initialized = False
        self.model: Optional[CovariateDLinearMixerCore] = None
        self.opt_warmup: Optional[torch.optim.AdamW] = None
        
        self.expected_H: Optional[int] = None
        self.expected_L: Optional[int] = None
        self.replay_buffer: List[Dict] = []
        
        self.is_warmed_up: bool = False
        self.loss_history: list[list[float]] = []
        self.val_loss_history: list[list[float]] = []
        self._last_collection_report_windows: int = 0
        
        self.e_base_moving: Optional[torch.Tensor] = None
        self.e_ref_moving: Optional[torch.Tensor] = None
        self._last_pure_y_median: Optional[torch.Tensor] = None
        self._routing_enabled: bool = False
        self._routing_bootstrap_remaining: int = 0
        self.prev_model_state: Optional[Dict[str, torch.Tensor]] = None
        self.E_prior: Optional[torch.Tensor] = None
        self._prior_model: Optional[CovariateDLinearMixerCore] = None
        self._latest_e_past: Optional[torch.Tensor] = None
        
        # --- V11.0: Strict Anti-Leakage Delay Queue ---
        # Stores recent fully resolved E_past tensors. 
        # Crucial for stride=1 training to avoid future leakage.
        self.resolved_error_queue: List[torch.Tensor] = []
        self._snapshot_idx: int = 0

    @staticmethod
    def _normalize_update_rule(value: str) -> str:
        key = str(value).strip().lower()
        if key not in {"plain", "bayesian"}:
            raise ValueError(f"Unsupported update_rule={value!r}. Expected one of: plain, bayesian")
        return key

    def _ensure_optimizer(self) -> None:
        if self.model is None:
            raise RuntimeError("Model must be initialized before optimizer creation.")
        if self.opt_warmup is None:
            self.opt_warmup = torch.optim.AdamW(
                self.model.parameters(), lr=self.config.lr, weight_decay=self.config.weight_decay
            )

    def reset_state(self, *, clear_loss_history: bool = True) -> None:
        self.expected_H = None
        self.expected_L = None
        self.replay_buffer.clear()
        self.is_warmed_up = False
        self.e_base_moving = None
        self.e_ref_moving = None
        self._last_pure_y_median = None
        self._routing_enabled = False
        self._routing_bootstrap_remaining = 0
        self.E_prior = None
        self._prior_model = None
        self.prev_model_state = None
        self._latest_e_past = None
        self.resolved_error_queue.clear()
        self._snapshot_idx = 0
        if clear_loss_history:
            self.loss_history = []
            self.val_loss_history = []
        self._last_collection_report_windows = 0

    def _prepare_batch(self, batch: List[Dict]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
        padded_X, padded_Y_median, padded_Y_GT, padded_E_past, padded_Y_ref_teacher, padded_t_idx = [], [], [], [], [], []
        stride_val = 1
        for item in batch:
            x = item["X"]
            L = x.shape[1]
            if L < self.expected_L:
                x = F.pad(x, (0, 0, self.expected_L - L, 0))
            elif L > self.expected_L:
                x = x[:, -self.expected_L:, :]
                
            padded_X.append(x)
            padded_Y_median.append(item["Y_base_median"])
            padded_Y_GT.append(item["Y_GT"])
            padded_E_past.append(item["E_past"])
            padded_Y_ref_teacher.append(item["Y_ref_teacher"])
            padded_t_idx.append(int(item.get("t_idx", 0)))
            stride_val = int(item.get("stride", stride_val))
            
        X = torch.cat(padded_X, dim=0) 
        Y_median = torch.cat(padded_Y_median, dim=0) 
        Y_GT = torch.cat(padded_Y_GT, dim=0) 
        E_past = torch.cat(padded_E_past, dim=0)
        Y_ref_teacher = torch.cat(padded_Y_ref_teacher, dim=0)
        t_idx = torch.tensor(padded_t_idx, dtype=torch.long, device=self.device)
        return X, Y_median, Y_GT, E_past, Y_ref_teacher, t_idx, max(1, stride_val)

    def _compute_loss(
        self,
        E_past: torch.Tensor,
        X: torch.Tensor,
        Y_median: torch.Tensor,
        Y_GT: torch.Tensor,
        Y_ref_teacher: torch.Tensor,
        t_idx: torch.Tensor,
        stride: int,
        is_training: bool = False,
    ) -> torch.Tensor:
        modifier = sanitize_tensor(self.model(E_past, X, Y_median))
        Y_refined_median = sanitize_tensor(self._apply_modifier(Y_median, modifier))
        err_ref_sq = F.mse_loss(Y_refined_median, Y_GT, reduction="none").mean(dim=(1, 2))
        loss_obs = err_ref_sq.mean()

        loss_task = loss_obs
        if self.config.update_rule == "bayesian" and self.E_prior is not None:
            prior_target = self.E_prior
            if prior_target.shape != Y_refined_median.shape:
                prior_target = None
            if prior_target is None:
                return torch.nan_to_num(loss_obs, nan=1e6, posinf=1e6, neginf=1e6)
            if self.e_ref_moving is None or self.e_base_moving is None:
                c_t = 0.0
            else:
                tau = max(1e-6, float(self.config.routing_temperature))
                energy_ref = torch.exp(-self.e_ref_moving / tau)
                energy_base = torch.exp(-self.e_base_moving / tau)
                confidence = energy_ref / (energy_ref + energy_base + 1e-8)
                c_t = float(torch.clamp(torch.mean(confidence), 0.1, 1.0).item())
            loss_prior = F.mse_loss(Y_refined_median, prior_target)
            loss = loss_task + c_t * loss_prior
        else:
            loss = loss_task
        
        if is_training:
            l1_reg = 0.0
            for name, param in self.model.named_parameters():
                if 'channel_mixers' in name and 'weight' in name:
                    l1_reg += torch.sum(torch.abs(param))
            loss = loss + self.config.mixer_reg_lambda * l1_reg
        loss = torch.nan_to_num(loss, nan=1e6, posinf=1e6, neginf=1e6)
            
        return loss

    def _apply_modifier(self, Y_base: torch.Tensor, modifier: torch.Tensor) -> torch.Tensor:
        return Y_base + modifier

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
        _ = physical_stride

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
        if self.expected_L is None and X.shape[1] > 0:
            self.expected_L = int(X.shape[1])

        S_dim = 0 if Y_base.ndim == 3 else 1
        Y_base_median = torch.median(Y_base, dim=S_dim, keepdim=True)[0]
        if Y_base_median.ndim == 4:
            Y_base_median = Y_base_median.squeeze(0) 

        S_dim_ref = 0 if Y_ref_past.ndim == 3 else 1
        Y_ref_past_median = torch.median(Y_ref_past, dim=S_dim_ref, keepdim=True)[0]
        if Y_ref_past_median.ndim == 4:
            Y_ref_past_median = Y_ref_past_median.squeeze(0)

        # --- DATA MANAGEMENT STRICT RULES FOR TRAINING ---
        # 1. At update time, current window error is fully observable.
        current_error = sanitize_tensor(Y_GT_full - Y_base_median)

        # 2. E_past is externally assembled by the online pipeline when provided.
        if e_past_override is not None:
            safe_e_past = sanitize_tensor(e_past_override).detach().to(self.device)
            if safe_e_past.ndim == 2:
                safe_e_past = safe_e_past.unsqueeze(0)
            if safe_e_past.shape != current_error.shape:
                safe_e_past = torch.zeros_like(current_error)
        else:
            # Backward-compatible fallback when no external E_past is provided.
            if len(self.resolved_error_queue) >= int(self.expected_H):
                safe_e_past = self.resolved_error_queue[0].clone()
            else:
                safe_e_past = torch.zeros_like(current_error)
            self.resolved_error_queue.append(current_error.clone())
            if len(self.resolved_error_queue) > int(self.expected_H):
                self.resolved_error_queue.pop(0)

        # For next-step prediction, use the newest causally available resolved error
        # from this closure event, not the delayed training input E_past.
        self._latest_e_past = current_error.detach().clone()

        current_snap = {
            "X": X.clone(),
            "Y_base_median": sanitize_tensor(Y_base_median.clone()),
            "Y_GT": sanitize_tensor(Y_GT_full.clone()),
            "E_past": sanitize_tensor(safe_e_past),  # Safely delayed primary variable
            "Y_ref_teacher": sanitize_tensor(Y_ref_past_median.clone()),
            "t_idx": int(self._snapshot_idx),
            "stride": int(physical_stride) if physical_stride is not None else 1,
        }
        self._snapshot_idx += 1
        self.replay_buffer.append(current_snap)
        
        # Standard EMA routing updates (only after warmup).
        if self.is_warmed_up:
            alpha = self.config.ema_error_momentum
            err_base = torch.mean(torch.abs(Y_GT_full - Y_base_median), dim=(0, 1))
            # Use snapshot-aligned refiner output to avoid cross-window EMA drift.
            err_ref = torch.mean(torch.abs(Y_GT_full - Y_ref_past_median), dim=(0, 1))

            if self.e_base_moving is None:
                self.e_base_moving = err_base.clone()
                self.e_ref_moving = err_ref.clone()
            else:
                self.e_base_moving = alpha * err_base + (1 - alpha) * self.e_base_moving
                self.e_ref_moving = alpha * err_ref + (1 - alpha) * self.e_ref_moving

        gap_windows = int(self.expected_H)
        if self.config.online_training:
            target_collection_size = int(self.config.collect_train_windows)
        else:
            target_collection_size = int(self.config.collect_train_windows) + gap_windows + int(self.config.collect_val_windows)

        if (not self.is_warmed_up or self.config.online_training) and target_collection_size > 0:
            collected = int(len(self.replay_buffer))
            if collected % max(1, target_collection_size // 20) == 0 or collected == target_collection_size:
                print(f"[Refined][Linear] Snapshot collection: {collected}/{target_collection_size}", flush=True)

        should_train_now = target_collection_size > 0 and len(self.replay_buffer) >= target_collection_size and (not self.is_warmed_up or self.config.online_training)
        if should_train_now:
            print(f"\n[Refined][V11.0-CovariateDLinear] ====== BUFFER FULL. INITIATING SNAPSHOT TRAINING ======", flush=True)

            if not self.is_initialized:
                valid_Ls = [snap["X"].shape[1] for snap in self.replay_buffer if snap["X"].shape[1] > 0]
                self.expected_L = max(valid_Ls) if valid_Ls else 1
                
                self.model = CovariateDLinearMixerCore(
                    seq_len=self.expected_L, pred_len=self.expected_H, feature_dim=self.target_dim, 
                    hidden_dim=self.config.hidden_dim, num_blocks=self.config.num_blocks, 
                    refiner_input=str(self.config.refiner_input),
                    short_pred_len_threshold=int(self.config.short_pred_len_threshold),
                    ma_kernel_short=int(self.config.ma_kernel_short),
                    ma_kernel_long=int(self.config.ma_kernel_long),
                    channel_mix=bool(self.config.channel_mix),
                ).to(self.device)
                self.is_initialized = True

            # Keep optimizer state across periodic training cycles.
            self._ensure_optimizer()

            if self.config.update_rule == "bayesian" and self.prev_model_state is not None:
                self._prior_model = copy.deepcopy(self.model).to(self.device)
                self._prior_model.load_state_dict(self.prev_model_state)
                self._prior_model.eval()
            else:
                self._prior_model = None
            self.E_prior = None

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
            
            best_val_loss = float('inf')
            best_model_state = None
            patience_counter = 0
            
            for param in self.model.parameters():
                param.requires_grad = True
            
            for epoch in range(self.config.max_epochs):
                self.model.train()
                epoch_train_data = list(train_data)
                
                epoch_train_loss = 0.0
                num_batches = 0
                
                for i in range(0, len(epoch_train_data), self.config.batch_size):
                    batch = epoch_train_data[i : i + self.config.batch_size]
                    X_b, Y_median_b, Y_GT_b, E_past_b, Y_ref_teacher_b, t_idx_b, stride_b = self._prepare_batch(batch)

                    if self.config.update_rule == "bayesian" and self._prior_model is not None:
                        with torch.no_grad():
                            prior_modifier = sanitize_tensor(self._prior_model(E_past_b, X_b, Y_median_b))
                            self.E_prior = sanitize_tensor(self._apply_modifier(Y_median_b, prior_modifier)).detach()
                    else:
                        self.E_prior = None
                    
                    loss = self._compute_loss(
                        E_past_b,
                        X_b,
                        Y_median_b,
                        Y_GT_b,
                        Y_ref_teacher_b,
                        t_idx_b,
                        stride_b,
                        is_training=True,
                    )
                    
                    self.opt_warmup.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.opt_warmup.step()
                    
                    epoch_train_loss += loss.item()
                    num_batches += 1
                
                avg_train_loss = epoch_train_loss / max(1, num_batches)
                self.loss_history.append([float(avg_train_loss)])
                
                self.model.eval()
                epoch_val_loss = 0.0
                num_val_batches = 0
                
                with torch.no_grad():
                    for i in range(0, len(val_data), self.config.batch_size):
                        batch = val_data[i : i + self.config.batch_size]
                        X_b, Y_median_b, Y_GT_b, E_past_b, Y_ref_teacher_b, t_idx_b, stride_b = self._prepare_batch(batch)

                        if self.config.update_rule == "bayesian" and self._prior_model is not None:
                            prior_modifier = sanitize_tensor(self._prior_model(E_past_b, X_b, Y_median_b))
                            self.E_prior = sanitize_tensor(self._apply_modifier(Y_median_b, prior_modifier)).detach()
                        else:
                            self.E_prior = None
                        
                        v_loss = self._compute_loss(
                            E_past_b,
                            X_b,
                            Y_median_b,
                            Y_GT_b,
                            Y_ref_teacher_b,
                            t_idx_b,
                            stride_b,
                            is_training=False,
                        )
                        
                        epoch_val_loss += v_loss.item()
                        num_val_batches += 1
                        
                avg_val_loss = epoch_val_loss / max(1, num_val_batches)
                self.val_loss_history.append([float(avg_val_loss)])
                
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    best_model_state = copy.deepcopy(self.model.state_dict())
                    patience_counter = 0
                else:
                    patience_counter += 1
                    
                if patience_counter >= self.config.early_stop_patience:
                    print(f"          [Early Stop] Triggered at Epoch {epoch}. Restoring best weights (Val Loss: {best_val_loss:.6f})", flush=True)
                    break

            if best_model_state is not None:
                self.model.load_state_dict(best_model_state)
            if self.config.update_rule == "bayesian":
                self.prev_model_state = {
                    k: v.detach().clone() for k, v in self.model.state_dict().items()
                }
            self._prior_model = None
            self.E_prior = None
                
            self.is_warmed_up = True
            self._routing_enabled = True
            self._routing_bootstrap_remaining = 1
            
            for param in self.model.parameters():
                param.requires_grad = False
                
            self.replay_buffer.clear()
            print(f"[Refined][V11.0-CovariateDLinear] ====== TRAINING COMPLETE. MODEL FROZEN. ====== \n", flush=True)

        return []

    @torch.no_grad()
    def predict(self, y_pred_current: torch.Tensor, model_input_seq: Optional[torch.Tensor] = None) -> torch.Tensor:
        Original_Samples = sanitize_tensor(y_pred_current).to(self.device)
        
        if self.expected_H is None and Original_Samples.shape[1] > 0:
            self.expected_H = Original_Samples.shape[1]
            
        if model_input_seq is None:
            X_tensor = torch.zeros(1, 1, self.target_dim, device=self.device)
        else:
            X_tensor = sanitize_tensor(model_input_seq).to(self.device)

        if not self.is_warmed_up or self.model is None:
            return Original_Samples

        self.model.eval()
        
        L = X_tensor.shape[1]
        if L < self.expected_L:
            X_tensor = F.pad(X_tensor, (0, 0, self.expected_L - L, 0))
        elif L > self.expected_L:
            X_tensor = X_tensor[:, -self.expected_L:, :]
            
        X_tensor = X_tensor.unsqueeze(0) if X_tensor.ndim == 2 else X_tensor  
        
        S_dim = 0 if Original_Samples.ndim == 3 else 1
        Y_median_original = sanitize_tensor(torch.median(Original_Samples, dim=S_dim, keepdim=True)[0])
        if Y_median_original.ndim == 4:
            Y_median_original = Y_median_original.squeeze(0)

        # --- DATA MANAGEMENT STRICT RULES FOR TESTING ---
        # During inference (stride=H or dynamic), the safest E_past is the most recently 
        # computed error from the queue. If external pipeline strictly follows stride=H, 
        # queue[-1] is exactly the previous window's error.
        if self._latest_e_past is not None:
            safe_e_past = sanitize_tensor(self._latest_e_past).to(self.device)
        elif len(self.resolved_error_queue) > 0:
            safe_e_past = sanitize_tensor(self.resolved_error_queue[-1].clone())
        else:
            safe_e_past = torch.zeros(1, self.expected_H, self.target_dim, device=self.device)

        modifier = sanitize_tensor(self.model(safe_e_past, X_tensor, Y_median_original))
        
        Y_pure_median_norm = sanitize_tensor(self._apply_modifier(Y_median_original, modifier))
        self._last_pure_y_median = Y_pure_median_norm.detach()

        # Gate Logic with Lock Override
        if self.config.force_gate_open or not self._routing_enabled:
            c_t = torch.ones(1, 1, self.target_dim, device=self.device)
        elif self._routing_bootstrap_remaining > 0:
            c_t = torch.ones(1, 1, self.target_dim, device=self.device)
            self._routing_bootstrap_remaining -= 1
        elif self.e_base_moving is None or self.e_ref_moving is None:
            c_t = torch.ones(1, 1, self.target_dim, device=self.device)
        else:
            tau = self.config.routing_temperature
            exp_base = torch.exp(-self.e_base_moving / tau)
            exp_ref = torch.exp(-self.e_ref_moving / tau)
            c_t = exp_ref / (exp_base + exp_ref + 1e-8)
            c_t = c_t.view(1, 1, self.target_dim)
        c_t = sanitize_tensor(c_t)

        Y_refined_median = sanitize_tensor(Y_median_original + (c_t * modifier))
        
        Actual_Shift = sanitize_tensor(Y_refined_median - Y_median_original)
        Final_Samples = sanitize_tensor(Original_Samples + Actual_Shift)

        return Final_Samples if Final_Samples.ndim == 3 else Final_Samples.squeeze(0)




