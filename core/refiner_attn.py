from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from core.util.refiner_util import sanitize_tensor


@dataclass
class AttnConfig:
    # Attn V2: pure patch-token backbone + channel-cross attention.
    hidden_dim: int = 64
    num_blocks: int = 1

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

    # PatchTST-like patch settings (all configured here, no extra run args).
    patch_len: int = 24
    patch_stride: int = 12
    patch_embed_dim: int = 32
    attn_heads: int = 2
    patch_mlp_ratio: float = 1.5
    patch_dropout: float = 0.05
    attn_inner_dim: int = 16
    reduced_patch_tokens: int = 4
    max_patches: int = 32
    drop_path_rate: float = 0.05
    share_temporal_channel_attn: bool = True

    # Legacy fields retained for backward compatibility.
    attn_dim: int = 16
    attn_anchors: int = 8
    attn_recent_ratio: float = 0.5

    force_gate_open: bool = False


class PatchChannelMixerBlock(nn.Module):
    """Patch-first block: temporal patch attention, then iTransformer-style channel attention."""

    def __init__(
        self,
        embed_dim: int,
        attn_inner_dim: int,
        num_heads: int,
        mlp_ratio: float,
        dropout: float,
        drop_path: float,
        share_temporal_channel_attn: bool,
    ):
        super().__init__()
        self.embed_dim = int(embed_dim)
        self.attn_inner_dim = int(attn_inner_dim)

        self.temporal_norm = nn.LayerNorm(embed_dim)
        self.temporal_in = nn.Linear(self.embed_dim, self.attn_inner_dim)
        self.temporal_attn = nn.MultiheadAttention(
            embed_dim=self.attn_inner_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.temporal_out = nn.Linear(self.attn_inner_dim, self.embed_dim)
        self.temporal_dropout = nn.Dropout(dropout)

        self.channel_norm = nn.LayerNorm(embed_dim)
        self.channel_in = nn.Linear(self.embed_dim, self.attn_inner_dim)
        self.share_temporal_channel_attn = bool(share_temporal_channel_attn)
        if self.share_temporal_channel_attn:
            self.channel_attn = self.temporal_attn
        else:
            self.channel_attn = nn.MultiheadAttention(
                embed_dim=self.attn_inner_dim,
                num_heads=num_heads,
                dropout=dropout,
                batch_first=True,
            )
        self.channel_out = nn.Linear(self.attn_inner_dim, self.embed_dim)
        self.channel_dropout = nn.Dropout(dropout)
        self.gamma_channel = nn.Parameter(torch.zeros(1, 1, 1, self.embed_dim))
        self.channel_gate = nn.Sequential(
            nn.Linear(self.embed_dim * 2, max(8, self.embed_dim // 4)),
            nn.GELU(),
            nn.Linear(max(8, self.embed_dim // 4), 1),
        )

        hidden = max(embed_dim, int(round(float(mlp_ratio) * float(embed_dim))))
        self.ffn_norm = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, embed_dim),
            nn.Dropout(dropout),
        )
        self.ffn_dropout = nn.Dropout(dropout)
        self.drop_path = DropPath(float(drop_path))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, D, P, E]
        bsz, channels, num_patches, embed_dim = x.shape

        temporal_in = self.temporal_norm(x).reshape(bsz * channels, num_patches, embed_dim)
        temporal_in = self.temporal_in(temporal_in)
        temporal_out, _ = self.temporal_attn(
            temporal_in,
            temporal_in,
            temporal_in,
            need_weights=False,
        )
        temporal_out = self.temporal_out(temporal_out)
        temporal_out = temporal_out.reshape(bsz, channels, num_patches, embed_dim)
        x = x + self.drop_path(self.temporal_dropout(temporal_out))

        channel_in = self.channel_norm(x).permute(0, 2, 1, 3).reshape(bsz * num_patches, channels, embed_dim)
        channel_in = self.channel_in(channel_in)
        channel_out, _ = self.channel_attn(
            channel_in,
            channel_in,
            channel_in,
            need_weights=False,
        )
        channel_out = self.channel_out(channel_out)
        channel_out = channel_out.reshape(bsz, num_patches, channels, embed_dim).permute(0, 2, 1, 3)
        gate_input = torch.cat([x, channel_out], dim=-1)
        gate = torch.sigmoid(self.channel_gate(gate_input))
        gated_channel_out = gate * self.channel_dropout(channel_out)
        x = x + self.drop_path(self.gamma_channel * gated_channel_out)

        x = x + self.drop_path(self.ffn_dropout(self.ffn(self.ffn_norm(x))))
        return x


class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = float(max(0.0, drop_prob))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor = random_tensor.floor()
        return x.div(keep_prob) * random_tensor


class AttnCore(nn.Module):
    """Pure patch-token core with iTransformer-style channel cross-attention."""

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        feature_dim: int,
        hidden_dim: int,
        num_blocks: int,
        refiner_input: str = "all",
        channel_mix: bool = True,
        attn_dim: int = 16,
        attn_anchors: int = 8,
        attn_recent_ratio: float = 0.5,
        patch_len: int = 16,
        patch_stride: int = 8,
        patch_embed_dim: int = 64,
        attn_heads: int = 4,
        patch_mlp_ratio: float = 2.0,
        patch_dropout: float = 0.1,
        attn_inner_dim: int = 16,
        reduced_patch_tokens: int = 4,
        max_patches: int = 32,
        drop_path_rate: float = 0.05,
        share_temporal_channel_attn: bool = True,
    ):
        super().__init__()
        self.seq_len = int(seq_len)
        self.pred_len = int(pred_len)
        self.feature_dim = int(feature_dim)
        _ = hidden_dim
        self.channel_mix = bool(channel_mix)
        self.refiner_input = self._normalize_refiner_input(refiner_input)
        if self.refiner_input not in {"all", "xy", "x", "y", "e_past"}:
            raise ValueError(f"Unsupported refiner_input={refiner_input!r}. Expected one of: all, xy, x, y, e_past")

        _ = attn_anchors
        _ = attn_recent_ratio

        if self.refiner_input == "all":
            in_context_len = self.seq_len + 2 * self.pred_len
        elif self.refiner_input == "xy":
            in_context_len = self.seq_len + self.pred_len
        elif self.refiner_input == "x":
            in_context_len = self.seq_len
        elif self.refiner_input == "y":
            in_context_len = self.pred_len
        else:
            in_context_len = self.pred_len

        self.in_context_len = int(in_context_len)

        self.patch_len = max(2, int(patch_len))
        self.patch_stride = max(1, int(patch_stride))
        raw_embed_dim = int(max(8, patch_embed_dim if patch_embed_dim is not None else attn_dim))
        self.embed_dim = raw_embed_dim
        self.attn_inner_dim = int(max(8, min(attn_inner_dim, self.embed_dim)))
        self.attn_heads = self._resolve_attn_heads(self.attn_inner_dim, int(attn_heads))
        self.patch_mlp_ratio = float(max(1.0, patch_mlp_ratio))
        self.patch_dropout = float(max(0.0, patch_dropout))
        self.max_patches = max(1, int(max_patches))
        self.drop_path_rate = float(max(0.0, drop_path_rate))
        self.share_temporal_channel_attn = bool(share_temporal_channel_attn)

        self.patch_count = self._compute_patch_count(self.in_context_len, self.patch_len, self.patch_stride)
        self.effective_patch_count = min(self.patch_count, self.max_patches)
        self.reduced_patch_tokens = max(1, min(int(reduced_patch_tokens), self.effective_patch_count))
        self.patch_proj = nn.Linear(self.patch_len, self.embed_dim)
        self.patch_pos_embed = nn.Parameter(torch.zeros(1, 1, self.effective_patch_count, self.embed_dim))
        self.channel_embed = nn.Parameter(torch.zeros(1, self.feature_dim, 1, self.embed_dim))

        self.channel_mixers = nn.ModuleList(
            [
                PatchChannelMixerBlock(
                    embed_dim=self.embed_dim,
                    attn_inner_dim=self.attn_inner_dim,
                    num_heads=self.attn_heads,
                    mlp_ratio=self.patch_mlp_ratio,
                    dropout=self.patch_dropout,
                    drop_path=(0.0 if int(num_blocks) <= 1 else self.drop_path_rate * (idx / float(int(num_blocks) - 1))),
                    share_temporal_channel_attn=self.share_temporal_channel_attn,
                )
                for idx in range(max(1, int(num_blocks)))
            ]
            if self.channel_mix
            else []
        )

        self.final_norm = nn.LayerNorm(self.embed_dim)
        self.patch_reducer = nn.AdaptiveAvgPool1d(self.reduced_patch_tokens)
        self.readout = nn.Linear(self.reduced_patch_tokens * self.embed_dim, self.pred_len)
        self.out_proj = nn.Linear(self.pred_len, self.pred_len)
        nn.init.normal_(self.patch_pos_embed, mean=0.0, std=0.02)
        nn.init.normal_(self.channel_embed, mean=0.0, std=0.02)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    @staticmethod
    def _resolve_attn_heads(embed_dim: int, requested_heads: int) -> int:
        heads = max(1, int(requested_heads))
        heads = min(heads, max(1, int(embed_dim)))
        while heads > 1 and (int(embed_dim) % heads != 0):
            heads -= 1
        return heads

    @staticmethod
    def _compute_patch_count(length: int, patch_len: int, patch_stride: int) -> int:
        if int(length) <= int(patch_len):
            return 1
        span = int(length) - int(patch_len)
        return 1 + (span + int(patch_stride) - 1) // int(patch_stride)

    @staticmethod
    def _normalize_refiner_input(value: str) -> str:
        key = str(value).strip().lower()
        if key == "epast":
            key = "e_past"
        return key

    def _select_input(self, e_past: torch.Tensor, x_norm: torch.Tensor, y_base_norm: torch.Tensor) -> torch.Tensor:
        if self.refiner_input == "all":
            return torch.cat([e_past, x_norm, y_base_norm], dim=1)
        if self.refiner_input == "xy":
            return torch.cat([x_norm, y_base_norm], dim=1)
        if self.refiner_input == "x":
            return x_norm
        if self.refiner_input == "y":
            return y_base_norm
        return e_past

    def _pad_for_patch_unfold(self, z: torch.Tensor) -> torch.Tensor:
        t_len = int(z.shape[1])
        if t_len < self.patch_len:
            z = F.pad(z, (0, 0, 0, self.patch_len - t_len))
            t_len = self.patch_len
        remainder = (t_len - self.patch_len) % self.patch_stride
        if remainder != 0:
            z = F.pad(z, (0, 0, 0, self.patch_stride - remainder))
        return z

    def forward(self, e_past: torch.Tensor, x_norm: torch.Tensor, y_base_norm: torch.Tensor) -> torch.Tensor:
        z = self._select_input(e_past, x_norm, y_base_norm)
        z = self._pad_for_patch_unfold(z)

        # Patch-first tokenization without any temporal/channel mean pooling.
        patches = z.unfold(dimension=1, size=self.patch_len, step=self.patch_stride)
        patches = patches.permute(0, 2, 1, 3).contiguous()  # [B, D, P, patch_len]
        if int(patches.shape[2]) > self.effective_patch_count:
            patches = patches[:, :, -self.effective_patch_count :, :]

        patch_tokens = self.patch_proj(patches)
        patch_tokens = patch_tokens + self.channel_embed + self.patch_pos_embed[:, :, : patch_tokens.shape[2], :]

        if self.channel_mix and len(self.channel_mixers) > 0:
            for mixer in self.channel_mixers:
                patch_tokens = mixer(patch_tokens)

        patch_tokens = self.final_norm(patch_tokens)
        bsz, channels, num_patches, embed_dim = patch_tokens.shape
        patch_tokens = patch_tokens.permute(0, 1, 3, 2).reshape(bsz * channels, embed_dim, num_patches)
        patch_tokens = self.patch_reducer(patch_tokens)
        patch_tokens = patch_tokens.reshape(bsz, channels, embed_dim, self.reduced_patch_tokens).permute(0, 1, 3, 2).contiguous()
        bsz, channels, num_patches, embed_dim = patch_tokens.shape
        flat_tokens = patch_tokens.reshape(bsz, channels, num_patches * embed_dim)
        delta = self.readout(flat_tokens).transpose(1, 2)
        return self.out_proj(delta.transpose(1, 2)).transpose(1, 2)


class OnlineRefinerAttn(nn.Module):
    """Attn V2: patch-based online trainer with lightweight attention gate."""

    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int = 128,
        num_blocks: int = 2,
        lr: Optional[float] = None,
        device: Optional[torch.device] = None,
        **kwargs,
    ) -> None:
        super().__init__()
        self.device = device or torch.device("cpu")
        self.target_dim = int(feature_dim)

        self.config = AttnConfig(hidden_dim=hidden_dim, num_blocks=num_blocks)
        if lr is not None:
            self.config.lr = float(lr)
        if "collect_train_windows" not in kwargs or "collect_val_windows" not in kwargs:
            raise ValueError("Attn refiner requires explicitly provided window counts.")
        self.config.collect_train_windows = max(1, int(kwargs["collect_train_windows"]))
        self.config.collect_val_windows = max(1, int(kwargs["collect_val_windows"]))

        for k, v in kwargs.items():
            if hasattr(self.config, k):
                setattr(self.config, k, v)
        self.channel_mix = bool(kwargs.get("channel_mix", True))

        self.config.lr = max(1e-12, float(self.config.lr))
        self.config.weight_decay = max(0.0, float(self.config.weight_decay))
        self.config.update_rule = self._normalize_update_rule(self.config.update_rule)

        self.is_initialized = False
        self.model: Optional[AttnCore] = None
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
        self._prior_model: Optional[AttnCore] = None
        self._latest_e_past: Optional[torch.Tensor] = None

        # Strict delayed error queue copied from linear for anti-leakage.
        self.resolved_error_queue: List[torch.Tensor] = []
        self._snapshot_idx: int = 0

    @staticmethod
    def _normalize_update_rule(value: str) -> str:
        key = str(value).strip().lower()
        if key not in {"plain", "bayesian"}:
            raise ValueError(
                f"Attn supports update_rule in {{'plain', 'bayesian'}} only, got {value!r}."
            )
        return key

    def _resolve_effective_batch_size(self) -> int:
        base_batch = max(1, int(self.config.batch_size))
        channels = int(max(1, self.target_dim))
        if channels > 500:
            return max(1, base_batch // 4)
        if channels > 100:
            return max(1, base_batch // 2)
        return base_batch

    def _ensure_optimizer(self) -> None:
        if self.model is None:
            raise RuntimeError("Model must be initialized before optimizer creation.")
        if self.opt_warmup is None:
            self.opt_warmup = torch.optim.AdamW(
                self.model.parameters(),
                lr=float(self.config.lr),
                weight_decay=float(self.config.weight_decay),
            )

    @staticmethod
    def _is_oom_like_error(exc: Exception) -> bool:
        msg = str(exc).lower()
        return (
            "out of memory" in msg
            or "cuda out of memory" in msg
            or "cudacachingallocator" in msg
            or "nvml_success" in msg
        )

    @staticmethod
    def _clear_cuda_cache() -> None:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

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
        self.prev_model_state = None
        self.E_prior = None
        self._prior_model = None
        self._latest_e_past = None
        self.resolved_error_queue.clear()
        self._snapshot_idx = 0
        if clear_loss_history:
            self.loss_history = []
            self.val_loss_history = []
        self._last_collection_report_windows = 0

    def _prepare_batch(self, batch: List[Dict]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
        padded_X, padded_Y_median, padded_Y_GT, padded_E_past, padded_Y_ref_teacher, padded_t_idx = [], [], [], [], [], []
        stride_val = 1
        for item in batch:
            x = item["X"]
            l_cur = int(x.shape[1])
            if l_cur < int(self.expected_L):
                x = F.pad(x, (0, 0, int(self.expected_L) - l_cur, 0))
            elif l_cur > int(self.expected_L):
                x = x[:, -int(self.expected_L):, :]

            padded_X.append(x)
            padded_Y_median.append(item["Y_base_median"])
            padded_Y_GT.append(item["Y_GT"])
            padded_E_past.append(item["E_past"])
            padded_Y_ref_teacher.append(item["Y_ref_teacher"])
            padded_t_idx.append(int(item.get("t_idx", 0)))
            stride_val = int(item.get("stride", stride_val))

        x_out = torch.cat(padded_X, dim=0)
        y_median_out = torch.cat(padded_Y_median, dim=0)
        y_gt_out = torch.cat(padded_Y_GT, dim=0)
        e_past_out = torch.cat(padded_E_past, dim=0)
        y_ref_teacher_out = torch.cat(padded_Y_ref_teacher, dim=0)
        _ = torch.tensor(padded_t_idx, dtype=torch.long, device=self.device)
        return x_out, y_median_out, y_gt_out, e_past_out, y_ref_teacher_out, max(1, stride_val)

    def _apply_modifier(self, y_base: torch.Tensor, modifier: torch.Tensor) -> torch.Tensor:
        return y_base + modifier

    def _compute_loss(
        self,
        e_past: torch.Tensor,
        x: torch.Tensor,
        y_median: torch.Tensor,
        y_gt: torch.Tensor,
        y_ref_teacher: torch.Tensor,
        stride: int,
        is_training: bool = False,
    ) -> torch.Tensor:
        _ = y_ref_teacher
        _ = stride

        modifier = sanitize_tensor(self.model(e_past, x, y_median))
        y_refined = sanitize_tensor(self._apply_modifier(y_median, modifier))

        err_ref_sq = F.mse_loss(y_refined, y_gt, reduction="none").mean(dim=(1, 2))
        task_loss = err_ref_sq.mean()

        if self.config.update_rule == "bayesian" and self.E_prior is not None:
            prior_target = self.E_prior
            if prior_target.shape != y_refined.shape:
                prior_target = None
            if prior_target is None:
                return torch.nan_to_num(task_loss, nan=1e6, posinf=1e6, neginf=1e6)
            if self.e_ref_moving is None or self.e_base_moving is None:
                c_t = 0.0
            else:
                tau = max(1e-6, float(self.config.routing_temperature))
                energy_ref = torch.exp(-self.e_ref_moving / tau)
                energy_base = torch.exp(-self.e_base_moving / tau)
                confidence = energy_ref / (energy_ref + energy_base + 1e-8)
                c_t = float(torch.clamp(torch.mean(confidence), 0.1, 1.0).item())
            loss_prior = F.mse_loss(y_refined, prior_target, reduction="mean")
            task_loss = task_loss + c_t * loss_prior

        if is_training:
            l1_reg = 0.0
            for name, param in self.model.named_parameters():
                if "channel_mixers" in name and "weight" in name:
                    l1_reg += torch.sum(torch.abs(param))
            task_loss = task_loss + float(self.config.mixer_reg_lambda) * l1_reg

        loss = torch.nan_to_num(task_loss, nan=1e6, posinf=1e6, neginf=1e6)
        return loss

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

        if int(Y_GT_full.shape[1]) == 0:
            return []

        X = sanitize_tensor(X).detach().to(self.device)
        Y_base = sanitize_tensor(Y_base).detach().to(self.device)
        Y_ref_past = sanitize_tensor(Y_ref_past).detach().to(self.device)
        Y_GT_full = sanitize_tensor(Y_GT_full).detach().to(self.device)

        if Y_GT_full.ndim == 3 and int(Y_GT_full.shape[0]) > 1:
            Y_GT_full = Y_GT_full[0:1]

        if self.expected_H is None and int(Y_GT_full.shape[1]) > 0:
            self.expected_H = int(Y_GT_full.shape[1])
        if self.expected_L is None and int(X.shape[1]) > 0:
            self.expected_L = int(X.shape[1])

        s_dim = 0 if Y_base.ndim == 3 else 1
        y_base_median = torch.median(Y_base, dim=s_dim, keepdim=True)[0]
        if y_base_median.ndim == 4:
            y_base_median = y_base_median.squeeze(0)

        s_dim_ref = 0 if Y_ref_past.ndim == 3 else 1
        y_ref_past_median = torch.median(Y_ref_past, dim=s_dim_ref, keepdim=True)[0]
        if y_ref_past_median.ndim == 4:
            y_ref_past_median = y_ref_past_median.squeeze(0)

        current_error = sanitize_tensor(Y_GT_full - y_base_median)

        if e_past_override is not None:
            safe_e_past = sanitize_tensor(e_past_override).detach().to(self.device)
            if safe_e_past.ndim == 2:
                safe_e_past = safe_e_past.unsqueeze(0)
            if safe_e_past.shape != current_error.shape:
                safe_e_past = torch.zeros_like(current_error)
        else:
            # Backward-compatible fallback for non-online callers.
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
            "Y_base_median": sanitize_tensor(y_base_median.clone()),
            "Y_GT": sanitize_tensor(Y_GT_full.clone()),
            "E_past": sanitize_tensor(safe_e_past),
            "Y_ref_teacher": sanitize_tensor(y_ref_past_median.clone()),
            "t_idx": int(self._snapshot_idx),
            "stride": int(physical_stride) if physical_stride is not None else 1,
        }
        self._snapshot_idx += 1
        self.replay_buffer.append(current_snap)

        if self.is_warmed_up:
            alpha = float(self.config.ema_error_momentum)
            err_base = torch.mean(torch.abs(Y_GT_full - y_base_median), dim=(0, 1))

            # Use snapshot-aligned refiner output to avoid cross-window EMA drift.
            err_ref = torch.mean(torch.abs(Y_GT_full - y_ref_past_median), dim=(0, 1))

            if self.e_base_moving is None:
                self.e_base_moving = err_base.clone()
                self.e_ref_moving = err_ref.clone()
            else:
                self.e_base_moving = alpha * err_base + (1.0 - alpha) * self.e_base_moving
                self.e_ref_moving = alpha * err_ref + (1.0 - alpha) * self.e_ref_moving

        gap_windows = int(self.expected_H)
        if self.config.online_training:
            target_collection_size = int(self.config.collect_train_windows)
        else:
            target_collection_size = int(self.config.collect_train_windows) + gap_windows + int(self.config.collect_val_windows)

        if (not self.is_warmed_up or self.config.online_training) and target_collection_size > 0:
            collected = int(len(self.replay_buffer))
            if collected % max(1, target_collection_size // 20) == 0 or collected == target_collection_size:
                print(f"[Refined][Attn] Snapshot collection: {collected}/{target_collection_size}", flush=True)

        should_train_now = (
            target_collection_size > 0
            and len(self.replay_buffer) >= target_collection_size
            and (not self.is_warmed_up or self.config.online_training)
        )
        if should_train_now:
            print("\n[Refined][Attn] ====== BUFFER FULL. INITIATING SNAPSHOT TRAINING ======", flush=True)

            if not self.is_initialized:
                valid_lens = [snap["X"].shape[1] for snap in self.replay_buffer if int(snap["X"].shape[1]) > 0]
                self.expected_L = max(valid_lens) if valid_lens else 1

                self.model = AttnCore(
                    seq_len=int(self.expected_L),
                    pred_len=int(self.expected_H),
                    feature_dim=int(self.target_dim),
                    hidden_dim=int(self.config.hidden_dim),
                    num_blocks=int(self.config.num_blocks),
                    refiner_input=str(self.config.refiner_input),
                    channel_mix=bool(self.channel_mix),
                    attn_dim=int(self.config.attn_dim),
                    attn_anchors=int(self.config.attn_anchors),
                    attn_recent_ratio=float(self.config.attn_recent_ratio),
                    patch_len=int(self.config.patch_len),
                    patch_stride=int(self.config.patch_stride),
                    patch_embed_dim=int(self.config.patch_embed_dim),
                    attn_heads=int(self.config.attn_heads),
                    patch_mlp_ratio=float(self.config.patch_mlp_ratio),
                    patch_dropout=float(self.config.patch_dropout),
                    attn_inner_dim=int(self.config.attn_inner_dim),
                    reduced_patch_tokens=int(self.config.reduced_patch_tokens),
                    max_patches=int(self.config.max_patches),
                    drop_path_rate=float(self.config.drop_path_rate),
                    share_temporal_channel_attn=bool(self.config.share_temporal_channel_attn),
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
                val_data = self.replay_buffer[max(1, online_total - val_size): online_total]
            else:
                train_end = int(self.config.collect_train_windows)
                val_start = int(train_end + gap_windows)
                val_end = int(val_start + int(self.config.collect_val_windows))
                train_data = self.replay_buffer[:train_end]
                val_data = self.replay_buffer[val_start:val_end]

            if not val_data:
                val_data = train_data[-1:]

            best_val_loss = float("inf")
            best_model_state = None
            patience_counter = 0

            for param in self.model.parameters():
                param.requires_grad = True

            effective_batch_size = self._resolve_effective_batch_size()
            print(
                f"[Refined][Attn] Adaptive batch size: base={int(self.config.batch_size)} "
                f"| target_dim={int(self.target_dim)} | effective={effective_batch_size}",
                flush=True,
            )

            for epoch in range(int(self.config.max_epochs)):
                self.model.train()
                epoch_train_data = list(train_data)

                while True:
                    try:
                        epoch_train_loss = 0.0
                        num_batches = 0

                        for i in range(0, len(epoch_train_data), effective_batch_size):
                            batch = epoch_train_data[i: i + effective_batch_size]
                            x_b, y_median_b, y_gt_b, e_past_b, y_ref_teacher_b, stride_b = self._prepare_batch(batch)

                            if self.config.update_rule == "bayesian" and self._prior_model is not None:
                                with torch.no_grad():
                                    prior_modifier = sanitize_tensor(self._prior_model(e_past_b, x_b, y_median_b))
                                    self.E_prior = sanitize_tensor(self._apply_modifier(y_median_b, prior_modifier)).detach()
                            else:
                                self.E_prior = None

                            loss = self._compute_loss(
                                e_past_b,
                                x_b,
                                y_median_b,
                                y_gt_b,
                                y_ref_teacher_b,
                                stride_b,
                                is_training=True,
                            )

                            self.opt_warmup.zero_grad(set_to_none=True)
                            loss.backward()
                            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                            self.opt_warmup.step()

                            epoch_train_loss += float(loss.item())
                            num_batches += 1
                        break
                    except RuntimeError as exc:
                        if self._is_oom_like_error(exc) and effective_batch_size > 1:
                            new_batch_size = max(1, effective_batch_size // 2)
                            print(
                                f"[Refined][Attn] OOM-like runtime detected during train; "
                                f"reduce batch size {effective_batch_size} -> {new_batch_size} and retry.",
                                flush=True,
                            )
                            effective_batch_size = new_batch_size
                            self.opt_warmup.zero_grad(set_to_none=True)
                            self._clear_cuda_cache()
                            continue
                        raise

                avg_train_loss = float(epoch_train_loss / max(1, num_batches))
                self.loss_history.append([avg_train_loss])

                self.model.eval()
                while True:
                    try:
                        epoch_val_loss = 0.0
                        num_val_batches = 0

                        with torch.no_grad():
                            for i in range(0, len(val_data), effective_batch_size):
                                batch = val_data[i: i + effective_batch_size]
                                x_b, y_median_b, y_gt_b, e_past_b, y_ref_teacher_b, stride_b = self._prepare_batch(batch)

                                if self.config.update_rule == "bayesian" and self._prior_model is not None:
                                    prior_modifier = sanitize_tensor(self._prior_model(e_past_b, x_b, y_median_b))
                                    self.E_prior = sanitize_tensor(self._apply_modifier(y_median_b, prior_modifier)).detach()
                                else:
                                    self.E_prior = None

                                v_loss = self._compute_loss(
                                    e_past_b,
                                    x_b,
                                    y_median_b,
                                    y_gt_b,
                                    y_ref_teacher_b,
                                    stride_b,
                                    is_training=False,
                                )

                                epoch_val_loss += float(v_loss.item())
                                num_val_batches += 1
                        break
                    except RuntimeError as exc:
                        if self._is_oom_like_error(exc) and effective_batch_size > 1:
                            new_batch_size = max(1, effective_batch_size // 2)
                            print(
                                f"[Refined][Attn] OOM-like runtime detected during val; "
                                f"reduce batch size {effective_batch_size} -> {new_batch_size} and retry.",
                                flush=True,
                            )
                            effective_batch_size = new_batch_size
                            self._clear_cuda_cache()
                            continue
                        raise

                avg_val_loss = float(epoch_val_loss / max(1, num_val_batches))
                self.val_loss_history.append([avg_val_loss])

                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    best_model_state = copy.deepcopy(self.model.state_dict())
                    patience_counter = 0
                else:
                    patience_counter += 1

                if patience_counter >= int(self.config.early_stop_patience):
                    print(
                        f"          [Early Stop] Triggered at Epoch {epoch}. Restoring best weights "
                        f"(Val Loss: {best_val_loss:.6f})",
                        flush=True,
                    )
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
            print("[Refined][Attn] ====== TRAINING COMPLETE. MODEL FROZEN. ======\n", flush=True)

        return []

    @torch.no_grad()
    def predict(self, y_pred_current: torch.Tensor, model_input_seq: Optional[torch.Tensor] = None) -> torch.Tensor:
        original_samples = sanitize_tensor(y_pred_current).to(self.device)

        if self.expected_H is None and int(original_samples.shape[1]) > 0:
            self.expected_H = int(original_samples.shape[1])

        if model_input_seq is None:
            x_tensor = torch.zeros(1, 1, self.target_dim, device=self.device)
        else:
            x_tensor = sanitize_tensor(model_input_seq).to(self.device)

        if not self.is_warmed_up or self.model is None:
            return original_samples

        self.model.eval()

        l_cur = int(x_tensor.shape[1])
        if l_cur < int(self.expected_L):
            x_tensor = F.pad(x_tensor, (0, 0, int(self.expected_L) - l_cur, 0))
        elif l_cur > int(self.expected_L):
            x_tensor = x_tensor[:, -int(self.expected_L):, :]

        x_tensor = x_tensor.unsqueeze(0) if x_tensor.ndim == 2 else x_tensor

        s_dim = 0 if original_samples.ndim == 3 else 1
        y_median_original = sanitize_tensor(torch.median(original_samples, dim=s_dim, keepdim=True)[0])
        if y_median_original.ndim == 4:
            y_median_original = y_median_original.squeeze(0)

        if self._latest_e_past is not None:
            safe_e_past = sanitize_tensor(self._latest_e_past).to(self.device)
        elif len(self.resolved_error_queue) > 0:
            safe_e_past = sanitize_tensor(self.resolved_error_queue[-1].clone())
        else:
            safe_e_past = torch.zeros(1, int(self.expected_H), self.target_dim, device=self.device)

        modifier = sanitize_tensor(self.model(safe_e_past, x_tensor, y_median_original))

        y_pure_median_norm = sanitize_tensor(self._apply_modifier(y_median_original, modifier))
        self._last_pure_y_median = y_pure_median_norm.detach()

        if self.config.force_gate_open or not self._routing_enabled:
            c_t = torch.ones(1, 1, self.target_dim, device=self.device)
        elif self._routing_bootstrap_remaining > 0:
            c_t = torch.ones(1, 1, self.target_dim, device=self.device)
            self._routing_bootstrap_remaining -= 1
        elif self.e_base_moving is None or self.e_ref_moving is None:
            c_t = torch.ones(1, 1, self.target_dim, device=self.device)
        else:
            tau = float(self.config.routing_temperature)
            exp_base = torch.exp(-self.e_base_moving / tau)
            exp_ref = torch.exp(-self.e_ref_moving / tau)
            c_t = exp_ref / (exp_base + exp_ref + 1e-8)
            c_t = c_t.view(1, 1, self.target_dim)
        c_t = sanitize_tensor(c_t)

        y_refined_median = sanitize_tensor(y_median_original + (c_t * modifier))
        actual_shift = sanitize_tensor(y_refined_median - y_median_original)
        final_samples = sanitize_tensor(original_samples + actual_shift)

        return final_samples if final_samples.ndim == 3 else final_samples.squeeze(0)
