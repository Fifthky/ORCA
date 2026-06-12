

from __future__ import annotations

import copy
from collections import deque
from dataclasses import dataclass
from typing import Optional, List, Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from core.refiner_linear import CovariateDLinearMixerCore
from core.util.refiner_util import (
    sanitize_tensor,
)


@dataclass
class BayesianConfig:
    # Covariate-DLinear Bayesian online configuration.
    hidden_dim: int = 128
    num_blocks: int = 2

    # Single persistent optimizer configuration.
    lr: float = 1e-4
    weight_decay: float = 1e-5

    # Keep L1 mixer regularization active in cycle training.
    mixer_reg_lambda: float = 1e-3

    # Routing and backbone behavior.
    ema_error_momentum: float = 0.2
    routing_temperature: float = 0.1
    router: str = "boltzmann"
    refiner_input: str = "all"
    update_rule: str = "plain"
    loss_variant: str = "mse"
    huber_delta: float = 1.0
    short_pred_len_threshold: int = 30
    ma_kernel_short: int = 7
    ma_kernel_long: int = 25
    channel_mix: bool = True

    # Streaming Bayesian cycle defaults.
    train_batch_size: int = 256
    warmup_epochs: int = 50
    update_steps_per_cycle: int = 10

    # If True, always fully trust the refiner (c_t = 1.0).
    force_gate_open: bool = False


class OnlineRefinerBayesian(nn.Module):
    """Streaming Bayesian posterior updater with H-paced cycle training."""

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

        # Single source of truth: config owns optimizer hyper-parameters.
        self.config = BayesianConfig(
            hidden_dim=hidden_dim,
            num_blocks=num_blocks,
        )
        if lr is not None:
            self.config.lr = float(lr)

        if "collect_train_windows" not in kwargs:
            raise ValueError("Bay refiner requires explicitly provided collect_train_windows.")
        self.collect_train_windows = max(1, int(kwargs["collect_train_windows"]))

        # Apply external configurations
        for k, v in kwargs.items():
            if hasattr(self.config, k):
                setattr(self.config, k, v)

        self.config.lr = max(1e-12, float(self.config.lr))
        self.config.weight_decay = max(0.0, float(self.config.weight_decay))
        self.config.update_rule = self._normalize_update_rule(self.config.update_rule)
        self.config.loss_variant = self._normalize_loss_variant(self.config.loss_variant)
        self.config.router = self._normalize_router(self.config.router)
        self.config.huber_delta = max(1e-8, float(self.config.huber_delta))

        self.is_initialized = False
        self.model: Optional[CovariateDLinearMixerCore] = None
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
        self._prior_model: Optional[CovariateDLinearMixerCore] = None
        self.theta_prior: Optional[Dict[str, torch.Tensor]] = None
        self._latest_e_past: Optional[torch.Tensor] = None

        self.supports_gate_confidence = True
        self.last_gate_confidence: Optional[torch.Tensor] = None
        self.last_gate_time_index: Optional[int] = None

        # Stores recent fully resolved E_past tensors for anti-leakage fallback.
        self.resolved_error_queue: List[torch.Tensor] = []
        self._snapshot_idx: int = 0
        self.training_cycle_count: int = 0

    @staticmethod
    def _normalize_update_rule(value: str) -> str:
        key = str(value).strip().lower()
        if key not in {"plain", "bayesian", "semi_prior", "prior"}:
            raise ValueError(
                f"Unsupported update_rule={value!r}. Expected one of: plain, bayesian, semi_prior, prior"
            )
        return key

    @staticmethod
    def _normalize_loss_variant(value: str) -> str:
        key = str(value).strip().lower()
        if key not in {"mse", "mae", "huber"}:
            raise ValueError(f"Unsupported loss_variant={value!r}. Expected one of: mse, mae, huber")
        return key

    @staticmethod
    def _normalize_router(value: str) -> str:
        key = str(value).strip().lower()
        if key == "ema":
            key = "inema"
        if key not in {"boltzmann", "inema", "hard"}:
            raise ValueError(
                f"Unsupported router={value!r}. Expected one of: boltzmann, inema, hard"
            )
        return key

    def _compute_router_confidence(self, e_base: torch.Tensor, e_ref: torch.Tensor) -> torch.Tensor:
        e_base = torch.clamp(sanitize_tensor(e_base), min=0.0)
        e_ref = torch.clamp(sanitize_tensor(e_ref), min=0.0)
        eps = 1e-8

        router = str(self.config.router).strip().lower()
        if router == "boltzmann":
            tau = max(1e-6, float(self.config.routing_temperature))
            exp_base = torch.exp(-e_base / tau)
            exp_ref = torch.exp(-e_ref / tau)
            conf = exp_ref / (exp_base + exp_ref + eps)
            return torch.clamp(conf, 0.0, 1.0)

        if router == "inema":
            conf = e_base / (e_base + e_ref + eps)
            return torch.clamp(conf, 0.0, 1.0)

        conf = (e_ref <= e_base).to(e_base.dtype)
        return conf

    def _routing_confidence_scalar(self) -> float:
        if self.e_ref_moving is None or self.e_base_moving is None:
            return 0.0
        conf = self._compute_router_confidence(self.e_base_moving, self.e_ref_moving)
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
        self.last_gate_confidence = None
        self.last_gate_time_index = None
        self.training_cycle_count = 0
        if clear_loss_history:
            self.loss_history = []
            self.val_loss_history = []
            self.cycle_loss_history = []

    def _prepare_batch(self, batch: List[Dict]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        padded_X, padded_Y_median, padded_Y_GT, padded_E_past = [], [], [], []
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

        X = torch.cat(padded_X, dim=0) 
        Y_median = torch.cat(padded_Y_median, dim=0) 
        Y_GT = torch.cat(padded_Y_GT, dim=0) 
        E_past = torch.cat(padded_E_past, dim=0)
        return X, Y_median, Y_GT, E_past

    def _compute_loss(
        self,
        E_past: torch.Tensor,
        X: torch.Tensor,
        Y_median: torch.Tensor,
        Y_GT: torch.Tensor,
        *,
        is_training: bool,
    ) -> torch.Tensor:
        modifier = sanitize_tensor(self.model(E_past, X, Y_median))
        Y_refined_median = sanitize_tensor(self._apply_modifier(Y_median, modifier))

        if self.config.loss_variant == "mse":
            loss_obs_per_sample = F.mse_loss(Y_refined_median, Y_GT, reduction="none").mean(dim=(1, 2))
        elif self.config.loss_variant == "mae":
            loss_obs_per_sample = F.l1_loss(Y_refined_median, Y_GT, reduction="none").mean(dim=(1, 2))
        else:
            loss_obs_per_sample = F.huber_loss(
                Y_refined_median,
                Y_GT,
                reduction="none",
                delta=float(self.config.huber_delta),
            ).mean(dim=(1, 2))
        loss_obs = loss_obs_per_sample.mean()

        loss = loss_obs

        if self.config.update_rule in {"bayesian", "semi_prior", "prior"} and self._prior_model is not None:
            with torch.no_grad():
                prior_modifier = sanitize_tensor(self._prior_model(E_past, X, Y_median))
                Y_prior = sanitize_tensor(self._apply_modifier(Y_median, prior_modifier))
            if self.config.update_rule == "bayesian":
                c_t = self._routing_confidence_scalar()
            elif self.config.update_rule == "semi_prior":
                c_t = 0.5
            else:
                c_t = 1.0
            loss_prior = F.mse_loss(Y_refined_median, Y_prior)
            loss = loss_obs + c_t * loss_prior
        
        if is_training:
            l1_reg = 0.0
            for name, param in self.model.named_parameters():
                if 'channel_mixers' in name and 'weight' in name:
                    l1_reg += torch.sum(torch.abs(param))
            loss = loss + self.config.mixer_reg_lambda * l1_reg
        loss = torch.nan_to_num(loss, nan=1e6, posinf=1e6, neginf=1e6)

        return loss

    def _sample_decay_batch(self, batch_size: int) -> List[Dict]:
        items = list(self.replay_buffer)
        n = len(items)
        if n == 0:
            return []

        # Decay is decoupled from H: alpha = 2.0 / 4000.0 with k=0 as newest sample.
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
        self.training_cycle_count += 1

        # Hard snapshot anchoring after each completed cycle.
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
                X_b, Y_median_b, Y_GT_b, E_past_b = self._prepare_batch(batch)

                loss = self._compute_loss(
                    E_past_b,
                    X_b,
                    Y_median_b,
                    Y_GT_b,
                    is_training=True,
                )

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
            X_b, Y_median_b, Y_GT_b, E_past_b = self._prepare_batch(batch)

            loss = self._compute_loss(
                E_past_b,
                X_b,
                Y_median_b,
                Y_GT_b,
                is_training=True,
            )

            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            cycle_losses.append(float(loss.item()))

        self._finalize_cycle(cycle_losses)

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
        time_index: Optional[int] = None,
    ) -> list[float]:
        _ = physical_stride
        _ = step

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

        current_error = sanitize_tensor(Y_GT_full - Y_base_median)

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

        self._latest_e_past = current_error.detach().clone()

        current_snap = {
            "X": X.clone(),
            "Y_base_median": sanitize_tensor(Y_base_median.clone()),
            "Y_GT": sanitize_tensor(Y_GT_full.clone()),
            "E_past": sanitize_tensor(safe_e_past),
            "t_idx": int(self._snapshot_idx),
        }
        self._snapshot_idx += 1
        self.replay_buffer.append(current_snap)

        # Keep routing EMA synced with online closure stream.
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

            gate = self._compute_router_confidence(self.e_base_moving, self.e_ref_moving)
            self.last_gate_confidence = gate.detach().clone()
            if time_index is not None:
                self.last_gate_time_index = int(time_index)

        self.step_counter += 1

        if not self.is_initialized:
            valid_Ls = [snap["X"].shape[1] for snap in self.replay_buffer if int(snap["X"].shape[1]) > 0]
            self.expected_L = max(valid_Ls) if valid_Ls else 1
            self.model = CovariateDLinearMixerCore(
                seq_len=self.expected_L,
                pred_len=self.expected_H,
                feature_dim=self.target_dim,
                hidden_dim=self.config.hidden_dim,
                num_blocks=self.config.num_blocks,
                refiner_input=str(self.config.refiner_input),
                short_pred_len_threshold=int(self.config.short_pred_len_threshold),
                ma_kernel_short=int(self.config.ma_kernel_short),
                ma_kernel_long=int(self.config.ma_kernel_long),
                channel_mix=bool(self.config.channel_mix),
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
        Original_Samples = sanitize_tensor(y_pred_current).to(self.device)
        
        if self.expected_H is None and Original_Samples.shape[1] > 0:
            self.expected_H = Original_Samples.shape[1]
            
        if model_input_seq is None:
            X_tensor = torch.zeros(1, 1, self.target_dim, device=self.device)
        else:
            X_tensor = sanitize_tensor(model_input_seq).to(self.device)

        # Before first cycle training, output base forecast without refinement shift.
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

        if self._latest_e_past is not None:
            safe_e_past = sanitize_tensor(self._latest_e_past).to(self.device)
        elif len(self.resolved_error_queue) > 0:
            safe_e_past = sanitize_tensor(self.resolved_error_queue[-1].clone())
        else:
            safe_e_past = torch.zeros(1, self.expected_H, self.target_dim, device=self.device)

        modifier = sanitize_tensor(self.model(safe_e_past, X_tensor, Y_median_original))
        
        Y_pure_median_norm = sanitize_tensor(self._apply_modifier(Y_median_original, modifier))
        self._last_pure_y_median = Y_pure_median_norm.detach()

        if self.config.force_gate_open or not self._routing_enabled:
            c_t = torch.ones(1, 1, self.target_dim, device=self.device)
        elif self.e_base_moving is None or self.e_ref_moving is None:
            c_t = torch.ones(1, 1, self.target_dim, device=self.device)
        else:
            c_t = self._compute_router_confidence(self.e_base_moving, self.e_ref_moving)
            c_t = c_t.view(1, 1, self.target_dim)
        c_t = sanitize_tensor(c_t)

        Y_refined_median = sanitize_tensor(Y_median_original + (c_t * modifier))
        
        Actual_Shift = sanitize_tensor(Y_refined_median - Y_median_original)
        Final_Samples = sanitize_tensor(Original_Samples + Actual_Shift)

        return Final_Samples if Final_Samples.ndim == 3 else Final_Samples.squeeze(0)


 





