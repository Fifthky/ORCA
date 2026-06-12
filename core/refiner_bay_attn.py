from __future__ import annotations

import copy
from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from core.refiner_attn import AttnCore
from core.util.refiner_util import sanitize_tensor


@dataclass
class BayesianAttnConfig:
    hidden_dim: int = 64
    num_blocks: int = 1

    lr: float = 1e-4
    weight_decay: float = 1e-5

    mixer_reg_lambda: float = 1e-3

    ema_error_momentum: float = 0.2
    routing_temperature: float = 0.1
    refiner_input: str = "all"
    update_rule: str = "plain"
    channel_mix: bool = True

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

    train_batch_size: int = 256
    warmup_epochs: int = 50
    update_steps_per_cycle: int = 10

    force_gate_open: bool = False


class OnlineRefinerBayAttn(nn.Module):
    """Streaming Bayesian posterior updater with Attn backbone and H-paced cycle training."""

    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int = 64,
        num_blocks: int = 1,
        lr: Optional[float] = None,
        device: Optional[torch.device] = None,
        **kwargs,
    ) -> None:
        super().__init__()
        self.device = device or torch.device("cpu")
        self.target_dim = int(feature_dim)

        self.config = BayesianAttnConfig(hidden_dim=hidden_dim, num_blocks=num_blocks)
        if lr is not None:
            self.config.lr = float(lr)
        if "collect_train_windows" not in kwargs:
            raise ValueError("Bay_Attn refiner requires explicitly provided collect_train_windows.")
        self.collect_train_windows = max(1, int(kwargs["collect_train_windows"]))
        for k, v in kwargs.items():
            if hasattr(self.config, k):
                setattr(self.config, k, v)
        self.config.lr = max(1e-12, float(self.config.lr))
        self.config.weight_decay = max(0.0, float(self.config.weight_decay))
        self.config.update_rule = self._normalize_update_rule(self.config.update_rule)
        self.config.refiner_input = self._normalize_refiner_input(self.config.refiner_input)

        self.is_initialized = False
        self.model: Optional[AttnCore] = None
        self.optimizer: Optional[torch.optim.AdamW] = None

        self.expected_H: Optional[int] = None
        self.expected_L: Optional[int] = None
        self.replay_buffer: deque[Dict] = deque(maxlen=int(self.collect_train_windows))
        self.step_counter: int = 0

        self.is_warmed_up: bool = False
        self.loss_history: list[list[float]] = []
        self.val_loss_history: list[list[float]] = []
        self.cycle_loss_history: list[list[float]] = []

        self.e_base_moving: Optional[torch.Tensor] = None
        self.e_ref_moving: Optional[torch.Tensor] = None
        self._last_pure_y_median: Optional[torch.Tensor] = None
        self._routing_enabled: bool = False
        self._prior_model: Optional[AttnCore] = None
        self.theta_prior: Optional[Dict[str, torch.Tensor]] = None
        self._latest_e_past: Optional[torch.Tensor] = None

        self.resolved_error_queue: List[torch.Tensor] = []
        self._snapshot_idx: int = 0

    @staticmethod
    def _normalize_refiner_input(value: str) -> str:
        key = str(value).strip().lower()
        if key == "epast":
            key = "e_past"
        if key not in {"all", "xy", "x", "y", "e_past"}:
            raise ValueError(f"Unsupported refiner input {value!r}. Expected one of: all, xy, x, y, e_past")
        return key

    @staticmethod
    def _normalize_update_rule(value: str) -> str:
        key = str(value).strip().lower()
        if key not in {"plain", "bayesian"}:
            raise ValueError(f"Unsupported update_rule={value!r}. Expected one of: plain, bayesian")
        return key

    def _routing_confidence_scalar(self) -> float:
        if self.e_ref_moving is None or self.e_base_moving is None:
            return 0.0
        tau = max(1e-6, float(self.config.routing_temperature))
        exp_base = torch.exp(-self.e_base_moving / tau)
        exp_ref = torch.exp(-self.e_ref_moving / tau)
        conf = exp_ref / (exp_base + exp_ref + 1e-8)
        return float(torch.clamp(torch.mean(conf), 0.0, 1.0).item())

    def _ensure_optimizer(self) -> None:
        if self.model is None:
            raise RuntimeError("Model must be initialized before optimizer creation.")
        if self.optimizer is None:
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=float(self.config.lr),
                weight_decay=float(self.config.weight_decay),
            )

    def reset_state(self, *, clear_loss_history: bool = True) -> None:
        self.expected_H = None
        self.expected_L = None
        self.replay_buffer = deque(maxlen=int(self.collect_train_windows))
        self.step_counter = 0
        self.is_warmed_up = False
        self.e_base_moving = None
        self.e_ref_moving = None
        self._last_pure_y_median = None
        self._routing_enabled = False
        self._prior_model = None
        self.theta_prior = None
        self._latest_e_past = None
        self.resolved_error_queue.clear()
        self._snapshot_idx = 0
        if clear_loss_history:
            self.loss_history = []
            self.val_loss_history = []
            self.cycle_loss_history = []

    def _prepare_batch(self, batch: List[Dict]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        padded_x, padded_y_median, padded_y_gt, padded_e_past = [], [], [], []
        for item in batch:
            x = item["X"]
            l_cur = int(x.shape[1])
            if l_cur < int(self.expected_L):
                x = F.pad(x, (0, 0, int(self.expected_L) - l_cur, 0))
            elif l_cur > int(self.expected_L):
                x = x[:, -int(self.expected_L):, :]

            padded_x.append(x)
            padded_y_median.append(item["Y_base_median"])
            padded_y_gt.append(item["Y_GT"])
            padded_e_past.append(item["E_past"])

        x_out = torch.cat(padded_x, dim=0)
        y_median_out = torch.cat(padded_y_median, dim=0)
        y_gt_out = torch.cat(padded_y_gt, dim=0)
        e_past_out = torch.cat(padded_e_past, dim=0)
        return x_out, y_median_out, y_gt_out, e_past_out

    def _apply_modifier(self, y_base: torch.Tensor, modifier: torch.Tensor) -> torch.Tensor:
        return y_base + modifier

    def _compute_loss(
        self,
        e_past: torch.Tensor,
        x: torch.Tensor,
        y_median: torch.Tensor,
        y_gt: torch.Tensor,
        *,
        is_training: bool,
    ) -> torch.Tensor:
        modifier = sanitize_tensor(self.model(e_past, x, y_median))
        y_refined = sanitize_tensor(self._apply_modifier(y_median, modifier))

        err_ref_sq = F.mse_loss(y_refined, y_gt, reduction="none").mean(dim=(1, 2))
        loss_obs = err_ref_sq.mean()

        loss = loss_obs

        if self.config.update_rule == "bayesian" and self._prior_model is not None:
            with torch.no_grad():
                prior_modifier = sanitize_tensor(self._prior_model(e_past, x, y_median))
                y_prior = sanitize_tensor(self._apply_modifier(y_median, prior_modifier))
            c_t = self._routing_confidence_scalar()
            loss_prior = F.mse_loss(y_refined, y_prior)
            loss = loss_obs + c_t * loss_prior

        if is_training:
            l1_reg = 0.0
            for name, param in self.model.named_parameters():
                if "channel_mixers" in name and "weight" in name:
                    l1_reg += torch.sum(torch.abs(param))
            loss = loss + float(self.config.mixer_reg_lambda) * l1_reg

        return torch.nan_to_num(loss, nan=1e6, posinf=1e6, neginf=1e6)

    def _sample_decay_batch(self, batch_size: int) -> List[Dict]:
        items = list(self.replay_buffer)
        n = len(items)
        if n == 0:
            return []

        decay_alpha = 2.0 / 4000.0
        ages = torch.arange(n - 1, -1, -1, device=self.device, dtype=torch.float32)
        weights = torch.exp(-decay_alpha * ages)
        probs = weights / torch.clamp(weights.sum(), min=1e-8)

        replacement = bool(n < int(batch_size))
        idx = torch.multinomial(probs, num_samples=int(batch_size), replacement=replacement)
        return [items[int(i)] for i in idx.tolist()]

    def _finalize_cycle(self, cycle_losses: list[float]) -> None:
        if cycle_losses:
            self.loss_history.append([float(sum(cycle_losses) / len(cycle_losses))])
        self.cycle_loss_history.append(cycle_losses)

        self.theta_prior = {k: v.detach().clone() for k, v in self.model.state_dict().items()}
        if self._prior_model is None:
            self._prior_model = copy.deepcopy(self.model).to(self.device)
        self._prior_model.load_state_dict(self.theta_prior)
        self._prior_model.eval()

        self.is_warmed_up = True
        self._routing_enabled = True
        self.model.eval()

    def _run_warmup_epochs(self, *, epochs: int) -> None:
        if self.model is None or self.optimizer is None:
            return

        self.model.train()
        cycle_losses: list[float] = []
        batch_size = int(self.config.train_batch_size)
        all_items = list(self.replay_buffer)

        for _ in range(int(epochs)):
            if not all_items:
                continue
            for i in range(0, len(all_items), batch_size):
                batch = all_items[i : i + batch_size]
                x_b, y_median_b, y_gt_b, e_past_b = self._prepare_batch(batch)

                loss = self._compute_loss(e_past_b, x_b, y_median_b, y_gt_b, is_training=True)

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

                cycle_losses.append(float(loss.item()))

        self._finalize_cycle(cycle_losses)

    def _run_update_steps(self, *, steps: int) -> None:
        if self.model is None or self.optimizer is None:
            return

        self.model.train()
        cycle_losses: list[float] = []
        batch_size = int(self.config.train_batch_size)

        for _ in range(int(steps)):
            batch = self._sample_decay_batch(batch_size=batch_size)
            if not batch:
                continue
            x_b, y_median_b, y_gt_b, e_past_b = self._prepare_batch(batch)

            loss = self._compute_loss(e_past_b, x_b, y_median_b, y_gt_b, is_training=True)

            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            cycle_losses.append(float(loss.item()))

        self._finalize_cycle(cycle_losses)

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
            if len(self.resolved_error_queue) >= int(self.expected_H):
                safe_e_past = self.resolved_error_queue[0].clone()
            else:
                safe_e_past = torch.zeros_like(current_error)
            self.resolved_error_queue.append(current_error.clone())
            if len(self.resolved_error_queue) > int(self.expected_H):
                self.resolved_error_queue.pop(0)

        self._latest_e_past = current_error.detach().clone()

        current_snap = {
            "X": X.clone(),
            "Y_base_median": sanitize_tensor(y_base_median.clone()),
            "Y_GT": sanitize_tensor(Y_GT_full.clone()),
            "E_past": sanitize_tensor(safe_e_past),
            "t_idx": int(self._snapshot_idx),
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

        self.step_counter += 1

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
                channel_mix=bool(self.config.channel_mix),
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
            self._ensure_optimizer()

        if self.expected_H is None or int(self.expected_H) <= 0:
            return []

        should_run_cycle = (
            (self.step_counter % int(self.expected_H) == 0)
            and (len(self.replay_buffer) >= int(self.collect_train_windows))
        )

        if should_run_cycle:
            if not self.is_warmed_up:
                self._run_warmup_epochs(epochs=int(self.config.warmup_epochs))
            else:
                self._run_update_steps(steps=int(self.config.update_steps_per_cycle))

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
