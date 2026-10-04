from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn


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
    Decomposition and channel mixing for the selected residual context.
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
