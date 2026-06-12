from __future__ import annotations

from collections import deque
import math
import time
from typing import Iterator, List

import numpy as np
import torch
import torch.nn.functional as F
from torch.profiler import ProfilerActivity, profile
from gluonts.dataset.common import Dataset
from gluonts.model.forecast import Forecast, QuantileForecast, SampleForecast

from core.util.refiner_util import sanitize_tensor, select_quantile_index
from data.data_provider import entry_series_length, is_contiguous_length_step


class OnlineRefinerPredictor:
    """Predictor wrapper with online refiner updates."""

    def __init__(
        self,
        base_predictor,
        refiner,
        device: torch.device,
        context_length: int | None = None,
        *,
        update_input_dataset: Dataset | None = None,
        update_stride: int | None = None,
        prefer_model_input_for_refiner: bool | None = None,
        dense_to_baseline_ratio: int | None = None,
        predict_batch_size: int | None = None,
        learning_batch_list: list[str] | None = None,
        leanrning_batch_list: list[str] | None = None,
        static_mean_scale: torch.Tensor | np.ndarray | None = None,
        buffered_update_records: list | None = None,
        speed_mode: bool | None = None,
    ) -> None:
        self.base_predictor = base_predictor
        self.refiner = refiner
        self.device = device
        self.prediction_length = getattr(base_predictor, "prediction_length", None)
        self.context_length = (
            int(context_length)
            if context_length is not None
            else getattr(base_predictor, "context_length", None)
        )
        self.past_length = getattr(base_predictor, "past_length", None)
        self.window_count = 0
        self.flow_steps = 0
        self._prefer_model_input_for_refiner = prefer_model_input_for_refiner

        _ = update_input_dataset
        _ = update_stride
        self.update_stride = 1
        _ = dense_to_baseline_ratio
        self.predict_batch_size = int(predict_batch_size) if predict_batch_size is not None else None
        if self.predict_batch_size is not None and self.predict_batch_size < 1:
            raise ValueError(f"predict_batch_size must be >= 1, got {self.predict_batch_size}")
        learning_keys = learning_batch_list if learning_batch_list is not None else leanrning_batch_list
        if learning_keys is None:
            learning_keys = ["linear"]
        self.learning_batch_list = tuple(str(x).lower() for x in learning_keys if str(x).strip())
        self.static_mean_scale = self._coerce_static_mean_scale(static_mean_scale)
        self.buffered_update_records = list(buffered_update_records) if buffered_update_records is not None else None
        self.speed_mode = bool(speed_mode)
        self.refiner_predict_time_total = 0.0
        self.refiner_predict_calls = 0
        self.refiner_predict_gpu_total_mb = 0.0
        self.refiner_predict_gpu_calls = 0
        self.refiner_predict_flops_total = 0.0
        self.refiner_predict_flops_calls = 0
        self.refiner_update_time_total = 0.0
        self.refiner_update_calls = 0
        self.refiner_update_gpu_total_mb = 0.0
        self.refiner_update_gpu_calls = 0
        self.refiner_update_flops_total = 0.0
        self.refiner_update_flops_calls = 0
        self.refiner_predict_flops_per_step = float("nan")
        self.refiner_update_flops_per_step = float("nan")
        self._refiner_flops_estimated = False
        self._speed_measure_limit = 10
        self._speed_collecting = False
        self._speed_warmup_completed = False
        self._speed_pending_predict_sample: dict[str, float] | None = None
        self._speed_infer_steps_recorded = 0
        self._speed_stop_requested = False
        self._speed_collect_start_step = 2 * int(getattr(self.refiner, "collect_train_windows", 3000))

        refiner_optim = getattr(self.refiner, "optimizer", None)
        self._refiner_init_state = {
            "module": {k: v.detach().cpu().clone() for k, v in self.refiner.state_dict().items()},
            "optim": refiner_optim.state_dict() if refiner_optim is not None else None,
        }

        self.global_t: int = 0
        self.hist_GT_buffer: torch.Tensor | None = None
        self.ring_pending_queue: list[dict] = []
        self.error_delay_queue: list[torch.Tensor] = []
        self._ring_first_failure: dict | None = None

    def _sync_device(self) -> None:
        if torch.cuda.is_available() and str(self.device).startswith("cuda"):
            try:
                torch.cuda.synchronize(self.device)
            except Exception:
                torch.cuda.synchronize()

    def _record_refiner_predict_time(self, elapsed: float) -> None:
        self.refiner_predict_time_total += float(elapsed)
        self.refiner_predict_calls += 1

    def _record_refiner_predict_gpu(self, gpu_mb: float) -> None:
        if math.isfinite(float(gpu_mb)):
            self.refiner_predict_gpu_total_mb += float(gpu_mb)
            self.refiner_predict_gpu_calls += 1

    def _record_refiner_predict_flops(self, flops: float) -> None:
        if math.isfinite(float(flops)) and float(flops) > 0.0:
            self.refiner_predict_flops_total += float(flops)
            self.refiner_predict_flops_calls += 1
            self.refiner_predict_flops_per_step = float(flops)

    def _record_refiner_update_time(self, elapsed: float) -> None:
        self.refiner_update_time_total += float(elapsed)
        self.refiner_update_calls += 1

    def _record_refiner_update_gpu(self, gpu_mb: float) -> None:
        if math.isfinite(float(gpu_mb)):
            self.refiner_update_gpu_total_mb += float(gpu_mb)
            self.refiner_update_gpu_calls += 1

    def _record_refiner_update_flops(self, flops: float) -> None:
        if math.isfinite(float(flops)) and float(flops) > 0.0:
            self.refiner_update_flops_total += float(flops)
            self.refiner_update_flops_calls += 1
            self.refiner_update_flops_per_step = float(flops)

    def _reset_speed_tracking(self) -> None:
        self.refiner_predict_time_total = 0.0
        self.refiner_predict_calls = 0
        self.refiner_predict_gpu_total_mb = 0.0
        self.refiner_predict_gpu_calls = 0
        self.refiner_predict_flops_total = 0.0
        self.refiner_predict_flops_calls = 0
        self.refiner_update_time_total = 0.0
        self.refiner_update_calls = 0
        self.refiner_update_gpu_total_mb = 0.0
        self.refiner_update_gpu_calls = 0
        self.refiner_update_flops_total = 0.0
        self.refiner_update_flops_calls = 0
        self.refiner_predict_flops_per_step = float("nan")
        self.refiner_update_flops_per_step = float("nan")
        self._refiner_flops_estimated = False
        self._speed_measure_limit = 10
        self._speed_collecting = False
        self._speed_warmup_completed = False
        self._speed_pending_predict_sample = None
        self._speed_infer_steps_recorded = 0
        self._speed_stop_requested = False
        self._speed_collect_start_step = 2 * int(getattr(self.refiner, "collect_train_windows", 3000))

    def _cuda_peak_memory_mb(self) -> float:
        if not (torch.cuda.is_available() and str(self.device).startswith("cuda")):
            return float("nan")
        try:
            return float(torch.cuda.max_memory_allocated(self.device) / (1024.0 * 1024.0))
        except Exception:
            try:
                return float(torch.cuda.max_memory_allocated() / (1024.0 * 1024.0))
            except Exception:
                return float("nan")

    def _reset_cuda_peak(self) -> None:
        if torch.cuda.is_available() and str(self.device).startswith("cuda"):
            try:
                torch.cuda.reset_peak_memory_stats(self.device)
            except Exception:
                try:
                    torch.cuda.reset_peak_memory_stats()
                except Exception:
                    pass

    def _finalize_pending_predict_sample(self) -> None:
        if self._speed_pending_predict_sample is None:
            return
        sample = self._speed_pending_predict_sample
        self._record_refiner_predict_time(float(sample.get("time", float("nan"))))
        self._record_refiner_predict_gpu(float(sample.get("gpu_mb", float("nan"))))
        self._record_refiner_predict_flops(float(sample.get("flops", float("nan"))))
        self._speed_infer_steps_recorded += 1
        self._speed_pending_predict_sample = None
        if self._speed_infer_steps_recorded >= int(self._speed_measure_limit):
            self._speed_stop_requested = True

    def _mean_or_nan(self, total: float, count: int) -> float:
        if count <= 0:
            return float("nan")
        value = float(total) / float(count)
        return value if math.isfinite(value) else float("nan")

    def get_speed_stats(self) -> dict[str, float]:
        return {
            "infer_time": self._mean_or_nan(self.refiner_predict_time_total, self.refiner_predict_calls),
            "infer_gpu": self._mean_or_nan(self.refiner_predict_gpu_total_mb, self.refiner_predict_gpu_calls),
            "infer_flops": self._mean_or_nan(self.refiner_predict_flops_total, self.refiner_predict_flops_calls),
            "train_time": self._mean_or_nan(self.refiner_update_time_total, self.refiner_update_calls),
            "train_gpu": self._mean_or_nan(self.refiner_update_gpu_total_mb, self.refiner_update_gpu_calls),
            "train_flops": self._mean_or_nan(self.refiner_update_flops_total, self.refiner_update_flops_calls),
        }

    def _refiner_is_predict_ready(self) -> bool:
        class_name = self.refiner.__class__.__name__.lower()
        if "bay" not in class_name:
            return False
        if bool(getattr(self.refiner, "is_warmed_up", True)) is False:
            return False
        return getattr(self.refiner, "model", None) is not None

    def _profile_refiner_predict(
        self,
        y_pred_current: torch.Tensor,
        model_input_seq: torch.Tensor,
    ) -> tuple[torch.Tensor, float]:
        if not self._refiner_is_predict_ready():
            out = self.refiner.predict(y_pred_current, model_input_seq=model_input_seq)
            return out, float("nan")

        out = None
        flops = float("nan")
        try:
            activities = [ProfilerActivity.CPU]
            if torch.cuda.is_available() and str(self.device).startswith("cuda"):
                activities.append(ProfilerActivity.CUDA)
            with profile(activities=activities, with_flops=True, profile_memory=False) as prof:
                out = self.refiner.predict(y_pred_current, model_input_seq=model_input_seq)
            flops = 0.0
            for evt in prof.key_averages():
                evt_flops = getattr(evt, "flops", None)
                if evt_flops is not None:
                    flops += float(evt_flops)
            if not math.isfinite(flops) or flops <= 0.0:
                flops = float("nan")
            return out, float(flops)
        except Exception:
            return out, float("nan")

    def _profile_refiner_update(self, update_callable):
        out = None
        flops = float("nan")
        try:
            activities = [ProfilerActivity.CPU]
            if torch.cuda.is_available() and str(self.device).startswith("cuda"):
                activities.append(ProfilerActivity.CUDA)
            with profile(activities=activities, with_flops=True, profile_memory=False) as prof:
                out = update_callable()
            flops = 0.0
            for evt in prof.key_averages():
                evt_flops = getattr(evt, "flops", None)
                if evt_flops is not None:
                    flops += float(evt_flops)
            if not math.isfinite(flops) or flops <= 0.0:
                flops = float("nan")
            return out, float(flops)
        except Exception:
            return out, float("nan")

    def _coerce_static_mean_scale(self, scale_obj) -> torch.Tensor | None:
        if scale_obj is None:
            return None
        try:
            if torch.is_tensor(scale_obj):
                x = scale_obj.detach().to(device=self.device, dtype=torch.float32)
            else:
                x = torch.as_tensor(np.asarray(scale_obj, dtype=np.float32), device=self.device)
            if x.ndim == 1:
                x = x.view(1, 1, -1)
            elif x.ndim == 2:
                x = x.unsqueeze(0)
            if x.ndim != 3:
                return None
            return x.clamp(min=1e-6)
        except Exception:
            return None

    @staticmethod
    def _coerce_label_target_window(label_entry) -> np.ndarray:
        arr = np.asarray(label_entry, dtype=np.float32)
        if arr.ndim == 1:
            return arr.reshape(-1, 1).astype(np.float32)
        if arr.ndim == 2:
            return arr.astype(np.float32)
        return arr.reshape(arr.shape[0], -1).astype(np.float32)

    @staticmethod
    def _entry_forecast_start(entry):
        main = entry[0] if isinstance(entry, tuple) else entry
        if isinstance(main, dict):
            if "forecast_start" in main:
                return main.get("forecast_start")
            if "start" in main and "target" in main:
                target = np.asarray(main["target"], dtype=np.float32)
                target_length = int(target.shape[0]) if target.ndim == 1 else int(target.shape[-1])
                return main["start"] + target_length
            return main.get("start")
        return None

    def _safe_reset_refiner(self, *, clear_loss_history: bool) -> None:
        try:
            self.refiner.reset_state(clear_loss_history=clear_loss_history)
        except TypeError:
            self.refiner.reset_state()

    def _reset_refiner_train_state(self) -> None:
        self.refiner.load_state_dict(self._refiner_init_state["module"])
        refiner_optim = getattr(self.refiner, "optimizer", None)
        if refiner_optim is not None and self._refiner_init_state["optim"] is not None:
            refiner_optim.load_state_dict(self._refiner_init_state["optim"])
            for state in refiner_optim.state.values():
                for key, value in list(state.items()):
                    if torch.is_tensor(value):
                        state[key] = value.to(self.device)
        self._safe_reset_refiner(clear_loss_history=False)
        self.global_t = 0
        self.hist_GT_buffer = None
        self.ring_pending_queue.clear()
        self.error_delay_queue.clear()
        self._ring_first_failure = None

    @staticmethod
    def _align_gt_to_base(y_gt: torch.Tensor, y_base: torch.Tensor) -> torch.Tensor:
        y = y_gt
        if y.ndim == 2:
            y = y.unsqueeze(0)
        if y.ndim != 3:
            raise ValueError(f"Expected GT tensor with 2D/3D shape, got {tuple(y_gt.shape)}")

        b = y_base
        if b.ndim == 2:
            b = b.unsqueeze(0)
        if b.ndim != 3:
            raise ValueError(f"Expected base tensor with 2D/3D shape, got {tuple(y_base.shape)}")

        if y.shape[1] == b.shape[2] and y.shape[2] == b.shape[1]:
            y = y.transpose(1, 2)
        if y.shape[0] == 1 and b.shape[0] > 1:
            y = y.expand(b.shape[0], -1, -1)
        if y.shape != b.shape:
            raise RuntimeError(f"Shape mismatch y_gt/base: y_gt={tuple(y.shape)}, base={tuple(b.shape)}")
        return y

    def _append_gt_increment(self, y_gt_increment: torch.Tensor, *, max_keep_steps: int = 2000) -> None:
        inc = y_gt_increment.detach().to(self.device)
        if inc.ndim == 2:
            inc = inc.unsqueeze(0)
        if inc.ndim != 3:
            raise ValueError(f"Expected GT increment with 2D/3D shape, got {tuple(y_gt_increment.shape)}")
        if inc.shape[0] > 1:
            inc = inc[0:1]

        if self.hist_GT_buffer is None:
            self.hist_GT_buffer = inc
        else:
            self.hist_GT_buffer = torch.cat([self.hist_GT_buffer, inc], dim=1)

        self.global_t += int(inc.shape[1])
        if self.hist_GT_buffer.shape[1] > int(max_keep_steps):
            self.hist_GT_buffer = self.hist_GT_buffer[:, -int(max_keep_steps):, :]

    def _collect_ready_ring_snapshots(
        self,
    ) -> list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None, int]]:
        if self.hist_GT_buffer is None or not self.ring_pending_queue:
            return []

        ready: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None, int]] = []
        surviving: list[dict] = []
        buffer_start_t = int(self.global_t - int(self.hist_GT_buffer.shape[1]))

        for snap in self.ring_pending_queue:
            t_start = int(snap.get("t_start", 0))
            expected_h = int(snap.get("expected_H", int(self.prediction_length or 0)))
            if expected_h <= 0:
                continue

            if t_start + expected_h <= self.global_t:
                start_idx = int(t_start - buffer_start_t)
                end_idx = int(start_idx + expected_h)
                self._assert_ring_closure(
                    t_start=t_start,
                    global_t=int(self.global_t),
                    closure_horizon=int(self.global_t - t_start),
                    expected_h=int(expected_h),
                    start_idx=int(start_idx),
                    end_idx=int(end_idx),
                    hist_len=int(self.hist_GT_buffer.shape[1]),
                )
                if start_idx < 0 or end_idx > int(self.hist_GT_buffer.shape[1]):
                    continue

                y_gt_full = self.hist_GT_buffer[:, start_idx:end_idx, :]
                x = snap["X"]
                y_base = snap["Y_base"]
                y_ref_past = snap["Y_ref_past"]
                y_ref_s = snap.get("Y_ref_S")

                if y_base.ndim == 2:
                    y_base = y_base.unsqueeze(0)
                if y_ref_past.ndim == 2:
                    y_ref_past = y_ref_past.unsqueeze(0)

                y_gt_aligned = self._align_gt_to_base(y_gt_full, y_base)
                time_index = int(t_start + expected_h)
                ready.append((x, y_base, y_ref_past, y_gt_aligned, y_ref_s, time_index))
            else:
                surviving.append(snap)

        self.ring_pending_queue = surviving
        return ready

    def _assert_ring_closure(
        self,
        *,
        t_start: int,
        global_t: int,
        closure_horizon: int,
        expected_h: int,
        start_idx: int,
        end_idx: int,
        hist_len: int,
    ) -> None:
        ok = True
        reason = ""
        if int(expected_h) <= 0:
            ok = False
            reason = "non_positive_expected_h"
        elif int(end_idx - start_idx) != int(expected_h):
            ok = False
            reason = "span_mismatch"
        elif int(closure_horizon) != int(expected_h):
            ok = False
            reason = "closure_horizon_mismatch"
        elif int(start_idx) < 0 or int(end_idx) > int(hist_len):
            ok = False
            reason = "hist_index_oob"

        if ok:
            return

        snapshot = {
            "reason": reason,
            "t_start": int(t_start),
            "global_t": int(global_t),
            "closure_horizon": int(closure_horizon),
            "expected_h": int(expected_h),
            "start_idx": int(start_idx),
            "end_idx": int(end_idx),
            "hist_len": int(hist_len),
        }
        if self._ring_first_failure is None:
            self._ring_first_failure = dict(snapshot)
        raise RuntimeError(f"Ring closure assertion failed: {snapshot}")

    def _dispatch_ring_quartet_update(
        self,
        x: torch.Tensor,
        y_base: torch.Tensor,
        y_ref_past: torch.Tensor,
        y_gt_full: torch.Tensor,
        e_past: torch.Tensor | None = None,
        y_ref_s: torch.Tensor | None = None,
        time_index: int | None = None,
    ) -> list[float]:
        _ = y_ref_s
        step = getattr(self.refiner, "step", 5)
        if not self.speed_mode:
            if bool(getattr(self.refiner, "supports_gate_confidence", False)):
                return self.refiner.update(
                    x,
                    y_base,
                    y_ref_past,
                    y_gt_full,
                    physical_stride=int(self.update_stride),
                    step=step,
                    e_past_override=e_past,
                    time_index=time_index,
                )
            return self.refiner.update(
                x,
                y_base,
                y_ref_past,
                y_gt_full,
                physical_stride=int(self.update_stride),
                step=step,
                e_past_override=e_past,
            )
        self._sync_device()
        prev_cycle_count = int(getattr(self.refiner, "training_cycle_count", 0))
        prev_step_counter = int(getattr(self.refiner, "step_counter", 0))
        self._reset_cuda_peak()
        t0 = time.perf_counter()
        def _call_update():
            if bool(getattr(self.refiner, "supports_gate_confidence", False)):
                return self.refiner.update(
                    x,
                    y_base,
                    y_ref_past,
                    y_gt_full,
                    physical_stride=int(self.update_stride),
                    step=step,
                    e_past_override=e_past,
                    time_index=time_index,
                )
            return self.refiner.update(
                x,
                y_base,
                y_ref_past,
                y_gt_full,
                physical_stride=int(self.update_stride),
                step=step,
                e_past_override=e_past,
            )

        profile_update_now = self._speed_collecting or (prev_step_counter + 1 >= int(self._speed_collect_start_step))
        if profile_update_now:
            out, train_flops = self._profile_refiner_update(_call_update)
        else:
            out = _call_update()
            train_flops = float("nan")
        self._sync_device()
        elapsed = time.perf_counter() - t0
        gpu_mb = self._cuda_peak_memory_mb()
        post_cycle_count = int(getattr(self.refiner, "training_cycle_count", 0))
        post_step_counter = int(getattr(self.refiner, "step_counter", 0))

        should_record_update = False
        training_completed = post_cycle_count > prev_cycle_count
        if training_completed and post_step_counter >= int(self._speed_collect_start_step):
            self._speed_collecting = True
            self._speed_pending_predict_sample = None
            should_record_update = True

        if should_record_update:
            self._record_refiner_update_time(elapsed)
            self._record_refiner_update_gpu(gpu_mb)
            self._record_refiner_update_flops(train_flops)

        return out

    @staticmethod
    def _normalize_model_input_seq_with_scale(
        model_input_seq: torch.Tensor,
        mean_scale: torch.Tensor | None,
    ) -> torch.Tensor:
        if mean_scale is None:
            return model_input_seq
        if model_input_seq.ndim != 3:
            return model_input_seq
        if int(mean_scale.shape[-1]) not in {1, int(model_input_seq.shape[-1])}:
            return model_input_seq
        return model_input_seq / mean_scale.to(device=model_input_seq.device, dtype=model_input_seq.dtype)

    @staticmethod
    def _build_mean_scale(y_gt: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        y = y_gt
        if y.ndim == 2:
            y = y.unsqueeze(0)
        if y.ndim != 3:
            raise ValueError(f"Expected y_gt with 2D/3D shape, got {tuple(y.shape)}")
        scale = y.mean(dim=(0, 1), keepdim=True).abs().clamp(min=eps)
        return scale

    @staticmethod
    def _normalize_with_scale(x: torch.Tensor, mean_scale: torch.Tensor | None) -> torch.Tensor:
        if mean_scale is None:
            return x
        if x.ndim == 2:
            x = x.unsqueeze(-1)
        return x / mean_scale.to(device=x.device, dtype=x.dtype)

    @staticmethod
    def _denormalize_with_scale(x: torch.Tensor, mean_scale: torch.Tensor | None) -> torch.Tensor:
        if mean_scale is None:
            return x
        if x.ndim == 2:
            x = x.unsqueeze(-1)
        return x * mean_scale.to(device=x.device, dtype=x.dtype)

    @staticmethod
    def _to_blc(arr: np.ndarray, device: torch.device) -> tuple[torch.Tensor, bool, int]:
        is_univariate = arr.ndim == 1
        if is_univariate:
            x_input = torch.from_numpy(arr).to(device).unsqueeze(0).unsqueeze(-1)
            length = arr.shape[0]
        else:
            x_input = torch.from_numpy(arr).to(device).T.unsqueeze(0)
            length = arr.shape[1]
        return x_input, is_univariate, length

    def _extract_model_input_seq(self, entry: dict, *, is_univariate: bool) -> torch.Tensor:
        if "past_target" in entry:
            arr_full = np.asarray(entry["past_target"], dtype=np.float32)
        else:
            arr_full = np.asarray(entry["target"], dtype=np.float32)

        if is_univariate:
            if self.context_length is not None and arr_full.shape[0] > int(self.context_length):
                arr_full = arr_full[-int(self.context_length):]
            model_input, _, _ = self._to_blc(arr_full, self.device)
            return model_input

        if arr_full.ndim < 2:
            raise ValueError(f"Expected multivariate past window to be 2D, got {tuple(arr_full.shape)}")
        if self.context_length is not None and arr_full.shape[1] > int(self.context_length):
            arr_full = arr_full[:, -int(self.context_length):]
        model_input, _, _ = self._to_blc(arr_full, self.device)
        return model_input

    def _set_refiner_state(self, state_obj: object | None) -> None:
        if hasattr(self.refiner, "momentum_state"):
            self.refiner.momentum_state = state_obj
            return
        if hasattr(self.refiner, "momentum_trend") and hasattr(self.refiner, "momentum_seasonal"):
            if state_obj is None:
                self.refiner.momentum_trend = None
                self.refiner.momentum_seasonal = None
            else:
                trend_state, seasonal_state = state_obj
                self.refiner.momentum_trend = trend_state
                self.refiner.momentum_seasonal = seasonal_state

    def _get_refiner_state(self) -> object | None:
        if hasattr(self.refiner, "momentum_state"):
            return self.refiner.momentum_state
        if hasattr(self.refiner, "momentum_trend") and hasattr(self.refiner, "momentum_seasonal"):
            if self.refiner.momentum_trend is None or self.refiner.momentum_seasonal is None:
                return None
            return (self.refiner.momentum_trend, self.refiner.momentum_seasonal)
        return None

    def _extract_prev_gt(self, entry: dict) -> tuple[torch.Tensor | None, bool]:
        if "past_target" in entry:
            past_arr_full = np.asarray(entry["past_target"], dtype=np.float32)
        else:
            past_arr_full = np.asarray(entry["target"], dtype=np.float32)
        is_univariate = past_arr_full.ndim == 1
        pred_len = int(self.prediction_length) if self.prediction_length is not None else None
        if pred_len is None:
            return None, is_univariate

        if "past_is_pad" in entry:
            pad = np.asarray(entry["past_is_pad"]).astype(bool)
            if pad.ndim == 1:
                if pad.shape[0] >= pred_len and bool(pad[-pred_len:].any()):
                    return None, is_univariate
            elif pad.ndim == 2:
                if pad.shape[-1] >= pred_len and bool(pad[..., -pred_len:].any()):
                    return None, is_univariate

        if is_univariate:
            if past_arr_full.shape[0] < pred_len:
                return None, is_univariate
            y_gt = past_arr_full[-pred_len:]
        else:
            if past_arr_full.shape[1] < pred_len:
                return None, is_univariate
            y_gt = past_arr_full[:, -pred_len:]
        y_gt_tensor, _, _ = self._to_blc(y_gt, self.device)
        return y_gt_tensor, is_univariate

    @staticmethod
    def _is_monotonic_forecast_start(prev_start, curr_start) -> bool:
        if prev_start is None or curr_start is None:
            return True
        try:
            return bool(curr_start > prev_start)
        except Exception:
            return True

    def _normalize_samples_to_sld(self, samples: torch.Tensor, *, is_univariate: bool) -> torch.Tensor:
        if is_univariate:
            if samples.ndim == 2:
                return samples.unsqueeze(-1)
            if samples.ndim == 3 and samples.shape[-1] == 1:
                return samples
            if samples.ndim == 3 and samples.shape[1] == 1:
                return samples.transpose(1, 2)
            raise ValueError(f"Unexpected univariate samples shape: {tuple(samples.shape)}")

        if samples.ndim != 3:
            raise ValueError(f"Expected multivariate samples to be 3D, got {tuple(samples.shape)}")

        pred_len = int(self.prediction_length) if self.prediction_length is not None else None
        if pred_len is None:
            return samples

        if samples.shape[1] == pred_len and samples.shape[0] != pred_len:
            return samples
        if samples.shape[0] == pred_len and samples.shape[1] != pred_len:
            return samples.permute(1, 0, 2)
        if samples.shape[2] == pred_len:
            return samples.transpose(1, 2)
        return samples

    @staticmethod
    def _forecast_to_sample_array(forecast: Forecast) -> np.ndarray:
        if hasattr(forecast, "samples"):
            return np.asarray(getattr(forecast, "samples"), dtype=np.float32)

        # Fast path for QuantileForecast-backed objects that already store arrays.
        for attr_name in ("forecast_arrays", "_forecast_arrays", "forecast_array"):
            if hasattr(forecast, attr_name):
                try:
                    arr = np.asarray(getattr(forecast, attr_name), dtype=np.float32)
                    if arr.size > 0:
                        return arr
                except Exception:
                    pass

        if hasattr(forecast, "forecast_array"):
            return np.asarray(getattr(forecast, "forecast_array"), dtype=np.float32)

        keys = getattr(forecast, "forecast_keys", None)
        if keys:
            quantile_arrays: list[np.ndarray] = []
            for key in keys:
                try:
                    q = np.asarray(forecast.quantile(str(key)), dtype=np.float32)
                except Exception:
                    continue
                quantile_arrays.append(q)
            if quantile_arrays:
                return np.stack(quantile_arrays, axis=0)

        raise TypeError(f"Unsupported forecast type for sample extraction: {type(forecast)!r}")

    def _select_point_from_distribution(self, dist: torch.Tensor, forecast: Forecast) -> torch.Tensor:
        keys = getattr(forecast, "forecast_keys", None)
        if dist.ndim == 3 and keys is not None and len(list(keys)) == int(dist.shape[0]):
            best_idx = select_quantile_index(list(map(str, keys)), int(dist.shape[0]), target_quantile=0.5)
            if best_idx is not None and 0 <= int(best_idx) < int(dist.shape[0]):
                return dist[int(best_idx) : int(best_idx) + 1]
        return dist.mean(dim=0, keepdim=True)

    @staticmethod
    def _align_samples_to_gt(samples: torch.Tensor, y_gt: torch.Tensor | None) -> torch.Tensor:
        if y_gt is None:
            return samples

        x = samples
        if x.ndim == 2:
            x = x.unsqueeze(-1)

        y = y_gt
        if y.ndim == 2:
            y = y.unsqueeze(0)

        if x.ndim != 3 or y.ndim != 3:
            return x

        target_l = int(y.shape[1])
        target_d = int(y.shape[2])
        if x.shape[1] == target_l and x.shape[2] == target_d:
            return x

        for perm in ((0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0)):
            xp = x.permute(*perm)
            if xp.shape[1] == target_l and xp.shape[2] == target_d:
                return xp.contiguous()

        return x

    def _entry_forecast_iter(self, dataset: Dataset, **kwargs):
        stream_desc = "Refined-EvalStream"
        has_len = hasattr(dataset, "__len__")
        total = int(len(dataset)) if has_len else None
        if total is not None and total <= 0:
            return

        progress_marks: set[int] = set()
        if total is not None and total > 0:
            for k in range(1, 11):
                progress_marks.add(max(1, int(round(float(total) * float(k) / 10.0))))

        if self.buffered_update_records is not None:
            n = min(len(self.buffered_update_records), int(total or len(self.buffered_update_records)))
            for i, entry in enumerate(dataset):
                if i >= n:
                    break
                rec = self.buffered_update_records[i]
                if isinstance(rec, dict):
                    kind = str(rec.get("kind", "sample")).lower()
                    sample_arr = np.asarray(rec.get("payload"), dtype=np.float32)
                    fkeys = rec.get("forecast_keys", None)
                else:
                    kind = "sample"
                    sample_arr = np.asarray(rec, dtype=np.float32)
                    fkeys = None
                if isinstance(entry, dict):
                    start_date = self._entry_forecast_start(entry)
                    item_id = entry.get("item_id", None)
                else:
                    start_date = self._entry_forecast_start(entry)
                    item_id = getattr(entry, "item_id", None)
                if kind == "quantile":
                    qkeys = list(map(str, fkeys)) if fkeys else [str(x / 10.0) for x in range(1, 10)]
                    forecast = QuantileForecast(
                        item_id=item_id,
                        forecast_arrays=sample_arr,
                        start_date=start_date,
                        forecast_keys=qkeys,
                    )
                else:
                    forecast = SampleForecast(
                        samples=sample_arr,
                        start_date=start_date,
                        item_id=item_id,
                    )
                done = i + 1
                if done in progress_marks:
                    pct = 100.0 * float(done) / float(max(1, n))
                    print(f"{stream_desc}-Infer: {done}/{n} ({pct:.1f}%)", flush=True)
                yield entry, forecast
            return

        entry_queue: deque = deque()

        source_iter = iter(dataset)

        def _buffering_iter():
            for e in source_iter:
                entry_queue.append(e)
                yield e

        predict_kwargs = dict(kwargs)
        if self.predict_batch_size is not None:
            predict_kwargs["batch_size"] = int(self.predict_batch_size)
        try:
            forecast_iter = self.base_predictor.predict(_buffering_iter(), **predict_kwargs)
        except TypeError as exc:
            msg = str(exc)
            if "unexpected keyword argument" in msg and "batch_size" in msg:
                predict_kwargs.pop("batch_size", None)
                forecast_iter = self.base_predictor.predict(_buffering_iter(), **predict_kwargs)
            else:
                raise
        processed = 0
        for forecast in forecast_iter:
            processed += 1
            if not entry_queue:
                break
            entry = entry_queue.popleft()

            if processed in progress_marks and total is not None:
                pct = 100.0 * float(processed) / float(max(1, total))
                print(f"{stream_desc}-Infer: {processed}/{total} ({pct:.1f}%)", flush=True)
            yield entry, forecast

    def _process_stream_entry(
        self,
        entry: dict,
        forecast: Forecast,
        *,
        last_state,
        momentum_state,
        prev_series_len: int | None,
        dataset_mean_scale: torch.Tensor | None,
        last_forecast_start,
    ) -> tuple[Forecast, torch.Tensor | None, object | None, object | None, int | None, object | None]:

        y_gt_past, is_univariate = self._extract_prev_gt(entry)
        if dataset_mean_scale is None and y_gt_past is not None:
            dataset_mean_scale = self._build_mean_scale(y_gt_past)
        forecast_start_entry = entry.get("forecast_start", None)
        forecast_start_forecast = getattr(forecast, "start_date", None)
        curr_forecast_start = forecast_start_forecast if forecast_start_forecast is not None else forecast_start_entry
        curr_series_len = entry_series_length(entry)

        self._set_refiner_state(momentum_state)
        temporal_aligned = True
        start_monotonic = self._is_monotonic_forecast_start(last_forecast_start, curr_forecast_start)
        if last_state is not None:
            temporal_aligned = is_contiguous_length_step(prev_series_len, curr_series_len, self.update_stride) and bool(start_monotonic)

        if last_state is not None and not temporal_aligned:
            last_state = None
            momentum_state = None
            self._set_refiner_state(None)
            self.ring_pending_queue.clear()
            self.hist_GT_buffer = None
            self.global_t = 0
            self.error_delay_queue.clear()

        pred_samples_np = self._forecast_to_sample_array(forecast)
        pred_tensor_raw = torch.from_numpy(pred_samples_np).to(self.device)
        pred_tensor_all = self._normalize_samples_to_sld(pred_tensor_raw, is_univariate=is_univariate)
        pred_tensor_all = self._align_samples_to_gt(pred_tensor_all, y_gt_past)
        pred_tensor_flow_all = self._normalize_with_scale(pred_tensor_all, dataset_mean_scale)
        # Keep quantile/sample distribution for refiner output shaping, but use
        # a single point estimate (q50/mean) for online supervision snapshots.
        pred_tensor_flow = self._select_point_from_distribution(pred_tensor_flow_all, forecast)
        model_input_seq = self._extract_model_input_seq(entry, is_univariate=is_univariate)
        model_input_seq_flow = self._normalize_model_input_seq_with_scale(model_input_seq, dataset_mean_scale)

        effective_len = min(int(self.update_stride), int(self.prediction_length or self.update_stride or 1))

        if last_state is not None and y_gt_past is not None and temporal_aligned and effective_len > 0:
            y_gt_increment = y_gt_past
            if y_gt_increment.ndim == 3 and y_gt_increment.shape[1] > effective_len:
                y_gt_increment = y_gt_increment[:, -effective_len:, :]
            y_gt_increment = self._normalize_with_scale(y_gt_increment, dataset_mean_scale)

            self._append_gt_increment(y_gt_increment, max_keep_steps=2000)
            ready_snapshots = self._collect_ready_ring_snapshots()

            for x_snap, y_base_snap, y_ref_snap, y_gt_snap, y_ref_s_snap, time_index in ready_snapshots:
                if y_base_snap.ndim == 3 and int(y_base_snap.shape[0]) > 1:
                    y_base_center = torch.median(y_base_snap, dim=0, keepdim=True)[0]
                elif y_base_snap.ndim == 2:
                    y_base_center = y_base_snap.unsqueeze(0)
                else:
                    y_base_center = y_base_snap

                current_error = sanitize_tensor(y_gt_snap - y_base_center)
                expected_h = int(y_gt_snap.shape[1]) if y_gt_snap.ndim == 3 else int(self.prediction_length or 0)
                if len(self.error_delay_queue) >= int(expected_h):
                    safe_e_past = self.error_delay_queue[0].clone()
                else:
                    safe_e_past = torch.zeros_like(current_error)
                self.error_delay_queue.append(current_error.clone())
                if len(self.error_delay_queue) > int(expected_h):
                    self.error_delay_queue.pop(0)
                self._dispatch_ring_quartet_update(
                    x_snap,
                    y_base_snap,
                    y_ref_snap,
                    y_gt_snap,
                    safe_e_past,
                    y_ref_s_snap,
                    time_index,
                )

            self.flow_steps += int(len(ready_snapshots))

        chosen_current_input = model_input_seq_flow

        if self.speed_mode:
            self._sync_device()
            t0 = time.perf_counter()
            if (not self._refiner_flops_estimated) and self._speed_collecting and self._refiner_is_predict_ready():
                refined_samples_flow_all, flops = self._profile_refiner_predict(
                    pred_tensor_flow_all,
                    chosen_current_input,
                )
                if math.isfinite(float(flops)) and float(flops) > 0.0:
                    self.refiner_predict_flops_per_step = float(flops)
                    self._refiner_flops_estimated = True
            else:
                refined_samples_flow_all = self.refiner.predict(
                    pred_tensor_flow_all,
                    model_input_seq=chosen_current_input,
                )
            self._sync_device()
            elapsed = time.perf_counter() - t0
            gpu_mb = self._cuda_peak_memory_mb()
            predict_sample = {
                "time": float(elapsed),
                "gpu_mb": float(gpu_mb),
                "flops": float(self.refiner_predict_flops_per_step),
            }
            if self._speed_collecting:
                self._record_refiner_predict_time(predict_sample["time"])
                self._record_refiner_predict_gpu(predict_sample["gpu_mb"])
                self._record_refiner_predict_flops(predict_sample["flops"])
                self._speed_infer_steps_recorded += 1
                if self._speed_infer_steps_recorded >= int(self._speed_measure_limit):
                    self._speed_stop_requested = True
            else:
                self._speed_pending_predict_sample = predict_sample
        else:
            refined_samples_flow_all = self.refiner.predict(pred_tensor_flow_all, model_input_seq=chosen_current_input)
        out_forecast_source = forecast
        refined_samples_all = self._denormalize_with_scale(refined_samples_flow_all, dataset_mean_scale)
        refined_train_target_flow: torch.Tensor | None = refined_samples_flow_all
        raw_router_output = getattr(self.refiner, "router_raw_output", None)
        if raw_router_output is not None:
            refined_train_target_flow = raw_router_output.detach()
        bad_mask = torch.isnan(refined_samples_all) | torch.isinf(refined_samples_all)
        if bad_mask.any():
            refined_samples_all = torch.where(bad_mask, pred_tensor_all, refined_samples_all)

        if is_univariate:
            refined_samples_all = refined_samples_all.squeeze(-1)

        refined_np = refined_samples_all.detach().cpu().numpy()
        has_samples = hasattr(out_forecast_source, "samples") and getattr(out_forecast_source, "samples", None) is not None
        if has_samples:
            out_forecast_source.samples = refined_np
            out_forecast = out_forecast_source
        else:
            raw_keys = getattr(out_forecast_source, "forecast_keys", None)
            if raw_keys:
                q_arr = np.asarray(refined_np, dtype=np.float32)
                if is_univariate and q_arr.ndim == 3 and q_arr.shape[-1] == 1:
                    q_arr = q_arr[..., 0]
                out_forecast = QuantileForecast(
                    item_id=getattr(out_forecast_source, "item_id", None),
                    forecast_arrays=q_arr,
                    start_date=getattr(out_forecast_source, "start_date", None),
                    forecast_keys=list(map(str, raw_keys)),
                )
            else:
                out_forecast = SampleForecast(
                    samples=refined_np,
                    start_date=getattr(out_forecast_source, "start_date", None),
                    item_id=getattr(out_forecast_source, "item_id", None),
                )

        refined_point_flow = self._select_point_from_distribution(
            refined_train_target_flow.detach() if refined_train_target_flow is not None else pred_tensor_flow_all.detach(),
            out_forecast,
        )

        if self.prediction_length is not None and int(self.prediction_length) > 0:
            self.ring_pending_queue.append(
                {
                    "t_start": int(self.global_t),
                    "expected_H": int(self.prediction_length),
                    "adapted_until": 0,
                    "X": chosen_current_input.detach(),
                    # Keep unified contract: pass full distribution samples/quantiles
                    # to every refiner and let each refiner decide its own reduction.
                    "Y_base": pred_tensor_flow_all.detach(),
                    "Y_ref_past": (refined_train_target_flow.detach() if refined_train_target_flow is not None else pred_tensor_flow_all.detach()),
                    "Y_ref_S": None,
                }
            )

        last_state = (
            pred_tensor_flow.detach(),
            refined_point_flow.detach(),
            (model_input_seq_flow.detach(), model_input_seq_flow.detach()),
        )
        refiner_state = self._get_refiner_state()
        if refiner_state is not None:
            if isinstance(refiner_state, tuple):
                momentum_state = (refiner_state[0].detach(), refiner_state[1].detach())
            else:
                momentum_state = refiner_state.detach()
        else:
            momentum_state = None

        prev_series_len = curr_series_len
        last_forecast_start = curr_forecast_start
        self.window_count += 1
        return out_forecast, dataset_mean_scale, last_state, momentum_state, prev_series_len, last_forecast_start

    def predict(self, dataset: Dataset, **kwargs) -> Iterator[Forecast]:
        self.window_count = 0
        self.flow_steps = 0
        if self.speed_mode:
            self._reset_speed_tracking()
        self._safe_reset_refiner(clear_loss_history=True)
        dataset_mean_scale: torch.Tensor | None = (
            self.static_mean_scale.clone() if self.static_mean_scale is not None else None
        )
        self.global_t = 0
        self.hist_GT_buffer = None
        self.ring_pending_queue.clear()
        self.error_delay_queue.clear()
        self._ring_first_failure = None

        last_state = None
        momentum_state = None
        prev_series_len: int | None = None
        last_forecast_start = None

        for entry, forecast in self._entry_forecast_iter(dataset, **kwargs):
            forecast, dataset_mean_scale, last_state, momentum_state, prev_series_len, last_forecast_start = self._process_stream_entry(
                entry,
                forecast,
                last_state=last_state,
                momentum_state=momentum_state,
                prev_series_len=prev_series_len,
                dataset_mean_scale=dataset_mean_scale,
                last_forecast_start=last_forecast_start,
            )
            yield forecast
            if self.speed_mode and self._speed_stop_requested:
                break
        return
