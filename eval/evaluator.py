from __future__ import annotations

import logging
import os
import time
import warnings
from concurrent.futures import ThreadPoolExecutor
import csv
from pathlib import Path
from typing import Dict

import numpy as np

import torch
from gluonts.ev.metrics import MAE, MSE
from gluonts.model.forecast import Forecast, QuantileForecast, SampleForecast

from core.refiner_linear import OnlineRefinerLinear
from core.refiner_attn import OnlineRefinerAttn
from core.refiner_bay import OnlineRefinerBayesian
from core.refiner_bay_attn import OnlineRefinerBayAttn
from core.refiner_AdaY import OnlineRefinerAdaY
from core.refiner_DSOF import OnlineRefinerDSOF
from core.refiner_TAFAS import OnlineRefinerTAFAS
from core.refiner_SOLID import OnlineRefinerSOLID
from core.refiner_ELF import OnlineRefinerELF
from core.refiner_ridge import OnlineRefinerRidge
from core.refiner_arima import OnlineRefinerARIMA
from core.refiner_ets import OnlineRefinerETS
from core.util.refiner_util import parse_quantile_key, select_quantile_index
from data.csv_dataset import CsvSeriesDataset
from data.data_provider import (
    compute_window_and_update_steps_for_test_data,
    entry_series_length,
    filter_test_data_by_context_length,
    split_window_counts,
    slice_filtered_test_data,
)
from eval.eval_util import (
    build_progress_line,
    first_value,
    format_duration_dhms,
    inspect_attention_maps,
    save_comparison_plots,
)
from eval.model_backends import create_base_predictor, model_supports_multivariate, resolve_model_ref
from eval.online_training import OnlineRefinerPredictor

# Match notebook behavior by silencing repeated QuantileForecast mean fallback warnings.
logging.getLogger("gluonts.model.forecast").setLevel(logging.ERROR)

# Suppress known third-party AMP deprecation spam from xlstm kernels.
warnings.filterwarnings(
    "ignore",
    message=r"`torch\.cuda\.amp\.custom_fwd\(args\.\.\.\)` is deprecated.*",
    category=FutureWarning,
)
warnings.filterwarnings(
    "ignore",
    message=r"`torch\.cuda\.amp\.custom_bwd\(args\.\.\.\)` is deprecated.*",
    category=FutureWarning,
)

PRIMARY_METRIC_LABEL_1 = "MAE"
PRIMARY_METRIC_LABEL_2 = "MSE"
PRIMARY_METRIC_KEY_1 = "MAE[mean]"
PRIMARY_METRIC_KEY_2 = "MSE[mean]"
PRIMARY_METRIC_KEY_1_RAW = "MAE_raw[mean]"
PRIMARY_METRIC_KEY_2_RAW = "MSE_raw[mean]"
CORE_METRIC_KEYS = (
    PRIMARY_METRIC_KEY_1,
    PRIMARY_METRIC_KEY_2,
    PRIMARY_METRIC_KEY_1_RAW,
    PRIMARY_METRIC_KEY_2_RAW,
)


_CSV_DATASET_CACHE: dict[tuple[str, int, str, int | None], CsvSeriesDataset] = {}
_BIG_DATASET_NAMES: set[str] = {"traffic", "electricity"}
_BIG_DATASET_PRED_LEN_THRESHOLD = 100


def _is_oom_exception(exc: Exception) -> bool:
    if isinstance(exc, torch.cuda.OutOfMemoryError):
        return True
    msg = str(exc).lower()
    return (
        ("out of memory" in msg)
        or ("cuda oom" in msg)
        or ("nvml_success == r internal assert failed" in msg)
        or ("cudacachingallocator.cpp" in msg)
        or ("cuda error: invalid configuration argument" in msg)
        or ("invalid configuration argument" in msg)
    )


def _dataset_cache_key(args) -> tuple[str, int, str, int | None]:
    csv_path = str(Path(str(getattr(args, "csv_path", ""))).expanduser().resolve())
    pred_len = int(getattr(args, "pred_len", 96))
    target_column = str(getattr(args, "target_column", "all"))
    windows = getattr(args, "windows", None)
    windows_val = int(windows) if windows is not None else None
    return (csv_path, pred_len, target_column, windows_val)


def _get_or_create_csv_dataset(args) -> CsvSeriesDataset:
    key = _dataset_cache_key(args)
    cached = _CSV_DATASET_CACHE.get(key)
    if cached is not None:
        print(f"[Refined-CSV] Dataset cache hit: {Path(key[0]).name}", flush=True)
        return cached

    ds = CsvSeriesDataset(
        csv_path=key[0],
        prediction_length=int(key[1]),
        target_column=key[2],
        windows=key[3],
    )
    _CSV_DATASET_CACHE[key] = ds
    print(f"[Refined-CSV] Dataset cache miss -> loaded: {Path(key[0]).name}", flush=True)
    return ds

def _coerce_args_int(val, default: int) -> int:
    """Convert an args attribute to int, tolerating list/tuple produced by argparse nargs."""
    if val is None:
        return default
    if isinstance(val, (list, tuple)):
        return int(val[0]) if val else default
    return int(val)


def _coerce_args_float(val, default: float) -> float:
    """Convert an args attribute to float, tolerating list/tuple produced by argparse nargs."""
    if val is None:
        return default
    if isinstance(val, (list, tuple)):
        return float(val[0]) if val else default
    return float(val)


def _resolve_attn_maps_enabled(args) -> bool:
    return bool(getattr(args, "attn_maps", False))


def _resolve_refiner(args) -> str:
    return str(getattr(args, "refiner", "linear")).lower()


def _resolve_learning_batch_keys(args) -> list[str]:
    refiner_key = _resolve_refiner(args)
    method = _resolve_training_method(args)
    if method == "batch" and refiner_key in {"linear", "attn", "bay", "bay_attn", "aday", "dsof", "tafas", "solid"}:
        return [refiner_key]
    return ["linear"]


def _resolve_training_method(args) -> str:
    raw = getattr(args, "training_method", "online")
    if isinstance(raw, (list, tuple)):
        value = str(raw[0]).strip().lower() if raw else "online"
    else:
        value = str(raw).strip().lower()

    # Be tolerant to accidentally stringified list values such as "['batch']".
    if value not in {"batch", "online"}:
        has_batch = "batch" in value
        has_online = "online" in value
        if has_batch and not has_online:
            value = "batch"
        elif has_online and not has_batch:
            value = "online"

    if value not in {"batch", "online"}:
        raise ValueError(f"Unsupported training_method={value!r}. Expected one of: batch, online")
    return value


def _resolve_refiner_input(args) -> str:
    raw = getattr(args, "refiner_input", "all")
    if isinstance(raw, (list, tuple)):
        value = str(raw[0]).strip().lower() if raw else "all"
    else:
        value = str(raw).strip().lower()

    # Be tolerant to accidentally stringified list values such as "['all']".
    if value not in {"all", "xy", "x", "y", "e_past", "epast"}:
        has_epast = "e_past" in value or "epast" in value
        if "all" in value:
            value = "all"
        elif "xy" in value:
            value = "xy"
        elif has_epast:
            value = "e_past"
        elif "x" in value and "y" not in value:
            value = "x"
        elif "y" in value and "x" not in value:
            value = "y"
    if value == "epast":
        value = "e_past"
    if value not in {"all", "xy", "x", "y", "e_past"}:
        raise ValueError(f"Unsupported refiner_input={value!r}. Expected one of: all, xy, x, y, e_past")
    return value


def _resolve_update_rule(args) -> str:
    raw = getattr(args, "update_rule", "plain")
    if isinstance(raw, (list, tuple)):
        value = str(raw[0]).strip().lower() if raw else "plain"
    else:
        value = str(raw).strip().lower()
    if value not in {"plain", "bayesian"}:
        raise ValueError(f"Unsupported update_rule={value!r}. Expected one of: plain, bayesian")
    return value


def _resolve_bay_update_rule(args) -> str:
    raw = getattr(args, "update_rule", "plain")
    if isinstance(raw, (list, tuple)):
        value = str(raw[0]).strip().lower() if raw else "plain"
    else:
        value = str(raw).strip().lower()
    if value not in {"plain", "bayesian", "semi_prior", "prior"}:
        raise ValueError(
            f"Unsupported Bay update_rule={value!r}. Expected one of: plain, bayesian, semi_prior, prior"
        )
    return value


def _resolve_online_buffer_windows(args) -> int:
    raw = getattr(args, "online_buffer_windows", None)
    if isinstance(raw, (list, tuple)):
        raw = raw[0] if raw else None
    if raw is None:
        raw = getattr(args, "online_buffer_meta_windows", 1024)
    return max(1, int(raw))


def _resolve_force_gate_open(args) -> bool:
    return bool(getattr(args, "force_gate_open", False))


def _resolve_channel_mix(args) -> bool:
    return bool(getattr(args, "channel_mix", True))


def _resolve_bay_loss(args) -> str:
    raw = getattr(args, "bay_loss", "mse")
    if isinstance(raw, (list, tuple)):
        value = str(raw[0]).strip().lower() if raw else "mse"
    else:
        value = str(raw).strip().lower()
    if value not in {"mse", "mae", "huber"}:
        if "huber" in value:
            value = "huber"
        elif "mae" in value:
            value = "mae"
        elif "mse" in value:
            value = "mse"
    if value not in {"mse", "mae", "huber"}:
        raise ValueError(f"Unsupported bay_loss={value!r}. Expected one of: mse, mae, huber")
    return value


def _resolve_routing_temperature(args) -> float:
    return _coerce_args_float(getattr(args, "routing_temperature", 0.1), 0.1)


def _resolve_ema_error_momentum(args) -> float:
    return _coerce_args_float(getattr(args, "ema_error_momentum", 0.2), 0.2)


def _resolve_bay_router(args) -> str:
    raw = getattr(args, "bay_router", "boltzmann")
    if isinstance(raw, (list, tuple)):
        value = str(raw[0]).strip().lower() if raw else "boltzmann"
    else:
        value = str(raw).strip().lower()
    if value == "ema":
        value = "inema"
    if value not in {"boltzmann", "inema", "hard"}:
        raise ValueError(
            f"Unsupported bay_router={value!r}. Expected one of: boltzmann, inema, hard"
        )
    return value


def _resolve_refiner_tag(refiner: str) -> str:
    key = str(refiner).lower()
    if key == "elf":
        return "ELF"
    if key == "linear":
        return "Linear"
    if key == "bay":
        return "Bay"
    if key == "attn":
        return "Attn"
    if key == "bay_attn":
        return "Bay_Attn"
    if key == "aday":
        return "AdaY"
    if key == "dsof":
        return "DSOF"
    if key == "tafas":
        return "TAFAS"
    if key == "solid":
        return "SOLID"
    if key == "ridge":
        return "Ridge"
    if key == "arima":
        return "ARIMA"
    if key == "ets":
        return "ETS"
    return "Linear"


def _save_bay_gate_confidence_csv(
    *,
    model_name: str,
    dataset_name: str,
    pred_len: int,
    time_index: int,
    gate_confidence: np.ndarray,
) -> Path:
    safe_model = str(model_name).replace("/", "_").replace(" ", "_")
    safe_dataset = str(dataset_name).replace("/", "_").replace(" ", "_")
    file_name = f"gate_confidence_bay_{safe_model}_{safe_dataset}_pred{int(pred_len)}.csv"
    out_dir = Path("results/details")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / file_name

    if torch.is_tensor(gate_confidence):
        arr = gate_confidence.detach().to(device="cpu", dtype=torch.float32).reshape(-1).numpy()
    else:
        arr = np.asarray(gate_confidence, dtype=np.float32).reshape(-1)
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["time_index", "channel", "confidence"])
        for idx, val in enumerate(arr):
            writer.writerow([int(time_index), int(idx), float(val)])

    return out_path


def _is_large_data_channel(*, dataset_name: str, pred_len: int) -> bool:
    return str(dataset_name).strip().lower() in _BIG_DATASET_NAMES and int(pred_len) > int(_BIG_DATASET_PRED_LEN_THRESHOLD)


def _resolve_aday_delta(args) -> float:
    """Paper-aligned safe default: smaller bounded step on ETT, standard elsewhere."""
    dataset_name = ""
    csv_path = getattr(args, "csv_path", None)
    if csv_path:
        try:
            dataset_name = Path(str(csv_path)).stem
        except Exception:
            dataset_name = ""
    if not dataset_name:
        dataset_name = str(getattr(args, "dataset", "") or "")

    key = str(dataset_name).strip().lower()
    if key.startswith("etth") or key.startswith("ettm"):
        return 0.01
    return 0.1


def _resolve_solid_period(args) -> int:
    dataset_name = ""
    csv_path = getattr(args, "csv_path", None)
    if csv_path:
        try:
            dataset_name = Path(str(csv_path)).stem
        except Exception:
            dataset_name = ""
    if not dataset_name:
        dataset_name = str(getattr(args, "dataset", "") or "")

    key = str(dataset_name).strip().lower()
    if key.startswith("etth") or key.startswith("wth"):
        return 24
    if key.startswith("ettm"):
        return 96
    if "electricity" in key:
        return 24
    if "traffic" in key:
        return 24
    if "illness" in key:
        return 52
    if "weather" in key:
        return 144
    if "exchange" in key:
        return 1
    return 24


def _extract_label_target(label_entry) -> np.ndarray:
    if isinstance(label_entry, tuple):
        label_entry = label_entry[0]
    if isinstance(label_entry, dict):
        if "target" in label_entry:
            arr = np.asarray(label_entry["target"], dtype=np.float32)
        elif "future_target" in label_entry:
            arr = np.asarray(label_entry["future_target"], dtype=np.float32)
        else:
            raise ValueError("Label entry does not contain target/future_target")
    else:
        arr = np.asarray(label_entry, dtype=np.float32)

    # Normalize to shape (horizon, dims). For this CSV pipeline, labels
    # generated from one_dim_target=False are channel-first (D, H).
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim == 1:
        return _sanitize_forecast_array(arr.reshape(-1, 1))
    if arr.ndim == 2:
        return _sanitize_forecast_array(arr.T)
    # Fallback for unexpected ranks.
    return _sanitize_forecast_array(arr.reshape(arr.shape[0], -1).T)


def _sanitize_forecast_array(arr: np.ndarray, *, fallback: float = 0.0) -> np.ndarray:
    x = np.asarray(arr, dtype=np.float32)
    if x.size == 0:
        return x
    finite_mask = np.isfinite(x)
    if bool(np.all(finite_mask)):
        return x
    if bool(np.any(finite_mask)):
        finite_vals = x[finite_mask]
        fill_value = float(np.median(finite_vals))
    else:
        fill_value = float(fallback)
    y = np.nan_to_num(x, nan=fill_value, posinf=fill_value, neginf=fill_value)
    return np.asarray(y, dtype=np.float32)


def _align_point_window_2d(
    x2d: np.ndarray,
    *,
    expected_pred_len: int | None = None,
    expected_target_dim: int | None = None,
) -> np.ndarray:
    x = np.asarray(x2d, dtype=np.float32)
    if x.ndim != 2:
        return np.asarray(x, dtype=np.float32)
    if expected_pred_len is not None and expected_target_dim is not None:
        if x.shape == (int(expected_pred_len), int(expected_target_dim)):
            return x
        if x.shape == (int(expected_target_dim), int(expected_pred_len)):
            return x.transpose(1, 0)
    if expected_pred_len is not None:
        if x.shape[0] == int(expected_pred_len):
            return x
        if x.shape[1] == int(expected_pred_len):
            return x.transpose(1, 0)
    if expected_target_dim is not None:
        if x.shape[1] == int(expected_target_dim):
            return x
        if x.shape[0] == int(expected_target_dim):
            return x.transpose(1, 0)
    return x


def _forecast_samples_to_mean_window(
    samples: np.ndarray,
    *,
    expected_pred_len: int | None = None,
    expected_target_dim: int | None = None,
    forecast_keys: list[str] | None = None,
) -> np.ndarray:
    pred = _sanitize_forecast_array(np.asarray(samples))
    if pred.ndim == 1:
        return pred.reshape(-1, 1).astype(np.float32)
    if pred.ndim == 2:
        q_idx = select_quantile_index(forecast_keys, int(pred.shape[0]), target_quantile=0.5)
        if q_idx is not None:
            return pred[int(q_idx)].reshape(-1, 1).astype(np.float32)
        return pred.mean(axis=0).reshape(-1, 1).astype(np.float32)
    if pred.ndim == 3:
        q_idx = select_quantile_index(forecast_keys, int(pred.shape[0]), target_quantile=0.5)
        if q_idx is not None:
            pred_point = np.asarray(pred[int(q_idx)], dtype=np.float32)
        else:
            pred_point = np.asarray(pred.mean(axis=0), dtype=np.float32)
        if pred_point.ndim == 1:
            return pred_point.reshape(-1, 1).astype(np.float32)
        if pred_point.ndim == 2:
            return _align_point_window_2d(
                pred_point,
                expected_pred_len=expected_pred_len,
                expected_target_dim=expected_target_dim,
            ).astype(np.float32)
        return np.asarray(pred_point, dtype=np.float32)

    pred_mean = pred.mean(axis=0)
    if pred_mean.ndim == 1:
        pred_mean = pred_mean.reshape(-1, 1)
    return np.asarray(pred_mean, dtype=np.float32)


def _forecast_to_sample_array(forecast) -> np.ndarray:
    if hasattr(forecast, "samples"):
        return _sanitize_forecast_array(np.asarray(getattr(forecast, "samples"), dtype=np.float32))

    # Fast path for QuantileForecast-backed objects that already store arrays.
    for attr_name in ("forecast_arrays", "_forecast_arrays", "forecast_array"):
        if hasattr(forecast, attr_name):
            try:
                arr = np.asarray(getattr(forecast, attr_name), dtype=np.float32)
                if arr.size > 0:
                    return _sanitize_forecast_array(arr)
            except Exception:
                pass

    if hasattr(forecast, "forecast_array"):
        return _sanitize_forecast_array(np.asarray(getattr(forecast, "forecast_array"), dtype=np.float32))

    keys = getattr(forecast, "forecast_keys", None)
    if keys:
        quantile_arrays: list[np.ndarray] = []
        for key in keys:
            try:
                q = np.asarray(forecast.quantile(str(key)), dtype=np.float32)
            except Exception:
                continue
            quantile_arrays.append(_sanitize_forecast_array(q))
        if quantile_arrays:
            return _sanitize_forecast_array(np.stack(quantile_arrays, axis=0))

    raise TypeError(f"Unsupported forecast type for plotting extraction: {type(forecast)!r}")


def _build_gt_windows_from_labels(labels_iterable) -> list[np.ndarray]:
    out: list[np.ndarray] = []
    for label_entry in labels_iterable:
        out.append(_extract_label_target(label_entry))
    return out


def _align_gt_pred_windows(gt_windows: list[np.ndarray], pred_windows: list[np.ndarray]) -> tuple[list[np.ndarray], list[np.ndarray]]:
    n = min(len(gt_windows), len(pred_windows))
    gt_out: list[np.ndarray] = []
    pred_out: list[np.ndarray] = []
    for i in range(n):
        gt = gt_windows[i]
        pred = pred_windows[i]
        horizon = min(int(gt.shape[0]), int(pred.shape[0]))
        gt_out.append(gt[:horizon])
        pred_out.append(pred[:horizon])
    return gt_out, pred_out


def _compute_point_primary_metrics(
    gt_windows: list[np.ndarray],
    pred_windows: list[np.ndarray],
    *,
    channel_mean_abs_scale: np.ndarray | None = None,
) -> dict[str, float]:
    gt_aligned, pred_aligned = _align_gt_pred_windows(gt_windows, pred_windows)
    if not gt_aligned or not pred_aligned:
        return {
            PRIMARY_METRIC_KEY_1: float("nan"),
            PRIMARY_METRIC_KEY_2: float("nan"),
        }

    # Use GluonTS metrics directly (forecast_type="mean").
    mae_metric = MAE(forecast_type="mean")(axis=None)
    mse_metric = MSE(forecast_type="mean")(axis=None)
    for gt_w, pred_w in zip(gt_aligned, pred_aligned):
        g = np.asarray(gt_w, dtype=np.float32)
        p = np.asarray(pred_w, dtype=np.float32)
        if g.ndim == 1:
            g = g.reshape(-1, 1)
        if p.ndim == 1:
            p = p.reshape(-1, 1)
        h = min(int(g.shape[0]), int(p.shape[0]))
        d = min(int(g.shape[1]), int(p.shape[1]))
        if h <= 0 or d <= 0:
            continue

        if channel_mean_abs_scale is not None:
            ch_scale = np.asarray(channel_mean_abs_scale, dtype=np.float32).reshape(-1)
            if int(ch_scale.shape[0]) == 1:
                s = float(max(1e-6, ch_scale[0]))
                g = g / s
                p = p / s
            elif int(ch_scale.shape[0]) >= d:
                s = np.asarray(ch_scale[:d], dtype=np.float32).reshape(1, d)
                s = np.clip(s, 1e-6, np.inf)
                g = g / s
                p = p / s

        payload = {
            "label": np.asarray(g[:h, :d], dtype=np.float32),
            "mean": np.asarray(p[:h, :d], dtype=np.float32),
        }
        mae_metric.update(payload)
        mse_metric.update(payload)

    mae_val = first_value(mae_metric.get())
    mse_val = first_value(mse_metric.get())
    if (not np.isfinite(float(mae_val))) or (not np.isfinite(float(mse_val))):
        return {
            PRIMARY_METRIC_KEY_1: float("nan"),
            PRIMARY_METRIC_KEY_2: float("nan"),
        }

    return {
        PRIMARY_METRIC_KEY_1: float(mae_val),
        PRIMARY_METRIC_KEY_2: float(mse_val),
    }


def _entry_to_channel_time(entry) -> np.ndarray:
    main = entry[0] if isinstance(entry, tuple) else entry
    if "past_target" in main:
        arr = np.asarray(main["past_target"], dtype=np.float32)
    else:
        arr = np.asarray(main["target"], dtype=np.float32)
    if arr.ndim == 1:
        return arr.reshape(1, -1)
    if arr.ndim == 2:
        return arr
    return arr.reshape(arr.shape[0], -1)


def _use_electricity_channelwise_norm(dataset_name: str) -> bool:
    return str(dataset_name).strip().lower() == "electricity"


def _compute_train_global_mean_abs_scale(entries) -> float | None:
    if entries is None or len(entries) == 0:
        return None
    total_abs = 0.0
    total_count = 0
    try:
        for entry in entries:
            arr = _entry_to_channel_time(entry)
            x = np.asarray(arr, dtype=np.float32).reshape(-1)
            if x.size <= 0:
                continue
            finite = np.isfinite(x)
            if not np.any(finite):
                continue
            xf = np.abs(x[finite])
            total_abs += float(np.sum(xf))
            total_count += int(xf.size)
        if total_count <= 0:
            return None
        return max(1e-6, float(total_abs / float(total_count)))
    except Exception:
        return None


def _compute_train_channel_mean_abs_scale(entries) -> np.ndarray | None:
    if entries is None or len(entries) == 0:
        return None
    sum_abs: np.ndarray | None = None
    count: np.ndarray | None = None
    try:
        for entry in entries:
            arr = _entry_to_channel_time(entry)
            x = np.asarray(arr, dtype=np.float32)
            if x.ndim != 2 or x.size <= 0:
                continue
            finite = np.isfinite(x)
            if not np.any(finite):
                continue
            abs_x = np.abs(x)
            valid_abs = np.where(finite, abs_x, 0.0)
            ch_sum = valid_abs.sum(axis=1, dtype=np.float64)
            ch_count = finite.sum(axis=1, dtype=np.int64).astype(np.float64)
            if sum_abs is None or count is None:
                sum_abs = ch_sum
                count = ch_count
            else:
                if int(ch_sum.shape[0]) != int(sum_abs.shape[0]):
                    continue
                sum_abs += ch_sum
                count += ch_count
        if sum_abs is None or count is None:
            return None
        out = np.full_like(sum_abs, fill_value=np.nan, dtype=np.float64)
        valid = count > 0.0
        if not np.any(valid):
            return None
        out[valid] = sum_abs[valid] / count[valid]
        out = np.clip(out, 1e-6, np.inf)
        if not np.any(np.isfinite(out)):
            return None
        return np.asarray(out, dtype=np.float32)
    except Exception:
        return None


def _to_static_mean_scale_tensor(
    global_mean_scale: float | None,
    *,
    channel_mean_scale: np.ndarray | None = None,
) -> np.ndarray | None:
    if channel_mean_scale is not None:
        ch = np.asarray(channel_mean_scale, dtype=np.float32).reshape(-1)
        finite = np.isfinite(ch)
        if np.any(finite):
            ch_valid = np.where(finite, ch, 1.0)
            ch_valid = np.clip(ch_valid, 1e-6, np.inf)
            return np.asarray(ch_valid.reshape(1, 1, -1), dtype=np.float32)

    if global_mean_scale is None or not np.isfinite(float(global_mean_scale)):
        return None
    s = max(1e-6, float(global_mean_scale))
    return np.asarray([[[s]]], dtype=np.float32)


def _inject_scaled_primary_metrics(agg_metrics: Dict | None, *, mean_abs_scale: float | None) -> Dict:
    out: Dict = dict(agg_metrics or {})
    raw_mae = first_value(out.get(PRIMARY_METRIC_KEY_1))
    raw_mse = first_value(out.get(PRIMARY_METRIC_KEY_2))
    out[PRIMARY_METRIC_KEY_1_RAW] = float(raw_mae)
    out[PRIMARY_METRIC_KEY_2_RAW] = float(raw_mse)
    if mean_abs_scale is None or not np.isfinite(float(mean_abs_scale)) or float(mean_abs_scale) <= 0.0:
        return out

    scale = float(mean_abs_scale)
    out[PRIMARY_METRIC_KEY_1] = float(raw_mae) / scale
    out[PRIMARY_METRIC_KEY_2] = float(raw_mse) / (scale * scale)
    return out


def _inject_scaled_primary_metrics_channelwise(
    agg_metrics: Dict | None,
    *,
    gt_windows: list[np.ndarray],
    pred_windows: list[np.ndarray],
    channel_mean_abs_scale: np.ndarray | None,
) -> Dict:
    out: Dict = dict(agg_metrics or {})
    raw_mae = first_value(out.get(PRIMARY_METRIC_KEY_1))
    raw_mse = first_value(out.get(PRIMARY_METRIC_KEY_2))
    out[PRIMARY_METRIC_KEY_1_RAW] = float(raw_mae)
    out[PRIMARY_METRIC_KEY_2_RAW] = float(raw_mse)
    if channel_mean_abs_scale is None:
        return out

    scaled_point_metrics = _compute_point_primary_metrics(
        gt_windows,
        pred_windows,
        channel_mean_abs_scale=channel_mean_abs_scale,
    )
    out[PRIMARY_METRIC_KEY_1] = float(first_value(scaled_point_metrics.get(PRIMARY_METRIC_KEY_1)))
    out[PRIMARY_METRIC_KEY_2] = float(first_value(scaled_point_metrics.get(PRIMARY_METRIC_KEY_2)))
    return out


def _project_core_metric_keys(agg_metrics: Dict | None) -> Dict:
    src = dict(agg_metrics or {})
    return {key: first_value(src.get(key)) for key in CORE_METRIC_KEYS}


def _resolve_infer_cache_enabled(args) -> bool:
    return bool(getattr(args, "cache", False) or getattr(args, "cahce", False))


def _build_infer_cache_path(args, dataset_name: str) -> Path:
    model_raw = str(getattr(args, "model", "unknown"))
    safe_model = model_raw.replace("/", "_").replace("-", "_")
    safe_ds = str(dataset_name).replace("/", "_")
    ctx = int(getattr(args, "context_length", 0) or 0)
    pred = int(getattr(args, "pred_len", 0) or 0)
    cache_dir = Path("data/model_infer_cache")
    version_suffix = "npyv3"
    return cache_dir / f"{safe_model}_{safe_ds}_ctx{ctx}_pred{pred}_s1_{version_suffix}"


def _infer_cache_component_paths(cache_path: Path) -> tuple[Path, Path, Path]:
    base = str(cache_path)
    payload_path = Path(base + ".payloads.npy")
    kinds_path = Path(base + ".kinds.npy")
    qkeys_path = Path(base + ".qkeys.npy")
    return payload_path, kinds_path, qkeys_path
def _load_infer_cache(cache_path: Path) -> dict | None:

    payload_path, kinds_path, qkeys_path = _infer_cache_component_paths(cache_path)
    if not (payload_path.exists() and kinds_path.exists() and qkeys_path.exists()):
        return None
    try:
        try:
            total_bytes = int(payload_path.stat().st_size) + int(kinds_path.stat().st_size) + int(qkeys_path.stat().st_size)
            cache_size_mb = float(total_bytes) / (1024.0 * 1024.0)
            print(f"[Refined-CSV] Cache load: opening {cache_path.name}* ({cache_size_mb:.1f} MB)", flush=True)
        except Exception:
            print(f"[Refined-CSV] Cache load: opening {cache_path.name}*", flush=True)

        # Prefer memory-mapped loading for fastest startup on contiguous ndarray payloads.
        try:
            payloads = np.load(payload_path, allow_pickle=True, mmap_mode="r")
        except Exception:
            payloads = np.load(payload_path, allow_pickle=True)
        try:
            kinds = np.load(kinds_path, allow_pickle=True, mmap_mode="r")
        except Exception:
            kinds = np.load(kinds_path, allow_pickle=True)
        try:
            qkeys = np.load(qkeys_path, allow_pickle=True, mmap_mode="r")
        except Exception:
            qkeys = np.load(qkeys_path, allow_pickle=True)

        record_count = int(len(payloads))
        if not (record_count == len(kinds) == len(qkeys)):
            return None

        def _decode_one(index: int) -> dict:
            p = payloads[index]
            k = kinds[index]
            qk = qkeys[index]
            return {
                "kind": str(k),
                # Keep payload in cache dtype (typically float16) to avoid heavy eager conversion.
                "payload": p,
                "forecast_keys": list(qk) if qk is not None else None,
            }

        cpu_count = int(os.cpu_count() or 1)
        use_parallel_decode = bool(record_count >= 2048 and cpu_count > 1)
        if use_parallel_decode:
            workers = max(2, min(32, cpu_count))
            print(
                f"[Refined-CSV] Cache load: parallel decode start | records={record_count} | workers={workers}",
                flush=True,
            )
            with ThreadPoolExecutor(max_workers=workers) as executor:
                dense_records = list(executor.map(_decode_one, range(record_count)))
        else:
            dense_records = [_decode_one(i) for i in range(record_count)]

        print(f"[Refined-CSV] Cache load: decoded dense records={len(dense_records)}", flush=True)
        return {
            "dense_records": dense_records,
        }
    except Exception:
        return None


def _samples_list_to_mean_windows(
    samples_list: list,
    *,
    prediction_length: int,
    target_dim: int,
) -> list[np.ndarray]:
    out: list[np.ndarray] = []
    for samples in samples_list:
        if isinstance(samples, dict):
            arr = _sanitize_forecast_array(np.asarray(samples.get("payload"), dtype=np.float32))
            raw_keys = samples.get("forecast_keys", None)
            q_keys = list(map(str, raw_keys)) if raw_keys is not None else None
        else:
            arr = _sanitize_forecast_array(np.asarray(samples, dtype=np.float32))
            q_keys = None
        out.append(
            _forecast_samples_to_mean_window(
                arr,
                expected_pred_len=prediction_length,
                expected_target_dim=target_dim,
                forecast_keys=q_keys,
            )
        )
    return out


def _forecast_to_cache_record(forecast) -> dict:
    arr = _sanitize_forecast_array(_forecast_to_sample_array(forecast))
    if hasattr(forecast, "samples") and getattr(forecast, "samples") is not None:
        return {
            "kind": "sample",
            "payload": np.asarray(arr, dtype=np.float32),
            "forecast_keys": None,
        }
    keys = getattr(forecast, "forecast_keys", None)
    return {
        "kind": "quantile",
        "payload": np.asarray(arr, dtype=np.float32),
        "forecast_keys": list(map(str, keys)) if keys is not None else None,
    }


def _parse_quantile_key(key: str) -> float | None:
    # Keep this thin wrapper for existing call sites.
    return parse_quantile_key(key)


def _audit_window_alignment(
    gt_windows: list[np.ndarray],
    pred_windows: list[np.ndarray],
    point_meta_windows: list[dict] | None,
) -> None:
    if point_meta_windows is not None and len(point_meta_windows) != len(pred_windows):
        fail = {
            "window_idx": 0,
            "forecast_start": (point_meta_windows[0].get("forecast_start") if point_meta_windows else None),
            "pred_shape": (tuple(np.asarray(pred_windows[0]).shape) if pred_windows else ()),
            "gt_shape": (tuple(np.asarray(gt_windows[0]).shape) if gt_windows else ()),
            "chosen_quantile": (point_meta_windows[0].get("chosen_quantile") if point_meta_windows else None),
        }
        raise RuntimeError(f"Window alignment audit failed: meta/pred length mismatch | snapshot={fail}")

    n = min(len(gt_windows), len(pred_windows))
    if point_meta_windows is not None:
        n = min(n, len(point_meta_windows))
    for i in range(n):
        gt_shape = tuple(np.asarray(gt_windows[i]).shape)
        pred_shape = tuple(np.asarray(pred_windows[i]).shape)
        if len(gt_shape) < 2 or len(pred_shape) < 2:
            fail = {
                "window_idx": int(i),
                "forecast_start": (None if point_meta_windows is None else point_meta_windows[i].get("forecast_start")),
                "pred_shape": pred_shape,
                "gt_shape": gt_shape,
                "chosen_quantile": (None if point_meta_windows is None else point_meta_windows[i].get("chosen_quantile")),
            }
            raise RuntimeError(f"Window alignment audit failed: invalid rank | snapshot={fail}")
        if int(gt_shape[0]) <= 0 or int(pred_shape[0]) <= 0:
            fail = {
                "window_idx": int(i),
                "forecast_start": (None if point_meta_windows is None else point_meta_windows[i].get("forecast_start")),
                "pred_shape": pred_shape,
                "gt_shape": gt_shape,
                "chosen_quantile": (None if point_meta_windows is None else point_meta_windows[i].get("chosen_quantile")),
            }
            raise RuntimeError(f"Window alignment audit failed: non-positive horizon | snapshot={fail}")

        if point_meta_windows is not None:
            meta = point_meta_windows[i]
            meta_pred_shape = tuple(meta.get("pred_shape", ()))
            if meta_pred_shape and meta_pred_shape != pred_shape:
                fail = {
                    "window_idx": int(meta.get("window_idx", i)),
                    "forecast_start": meta.get("forecast_start"),
                    "pred_shape": pred_shape,
                    "gt_shape": gt_shape,
                    "chosen_quantile": meta.get("chosen_quantile"),
                }
                raise RuntimeError(f"Window alignment audit failed: pred shape mismatch | snapshot={fail}")


def _compose_window_audit_rows(
    gt_windows: list[np.ndarray],
    pred_windows: list[np.ndarray],
    point_meta_windows: list[dict] | None,
) -> list[dict]:
    n = min(len(gt_windows), len(pred_windows))
    if point_meta_windows is not None:
        n = min(n, len(point_meta_windows))
    rows: list[dict] = []
    for i in range(n):
        meta = point_meta_windows[i] if point_meta_windows is not None else {}
        rows.append(
            {
                "window_idx": int(meta.get("window_idx", i)),
                "forecast_start": meta.get("forecast_start", None),
                "pred_shape": tuple(np.asarray(pred_windows[i]).shape),
                "gt_shape": tuple(np.asarray(gt_windows[i]).shape),
                "chosen_quantile": meta.get("chosen_quantile", None),
            }
        )
    return rows


def _sanitize_quantile_payload(arr: np.ndarray, forecast_keys: list[str] | None) -> tuple[np.ndarray, list[str]]:
    x = _sanitize_forecast_array(np.asarray(arr, dtype=np.float32))
    if x.ndim == 1:
        x = x.reshape(1, -1)

    keys = list(map(str, forecast_keys)) if forecast_keys else []
    kept_idx: list[int] = []
    kept_q: list[float] = []
    if keys and x.ndim >= 2 and len(keys) == int(x.shape[0]):
        for idx, key in enumerate(keys):
            q = _parse_quantile_key(key)
            if q is None:
                continue
            kept_idx.append(idx)
            kept_q.append(float(q))
    if kept_idx:
        x = x[kept_idx, ...]
        order = np.argsort(np.asarray(kept_q, dtype=np.float32))
        x = x[order, ...]
        sorted_q = [kept_q[int(i)] for i in order.tolist()]
        return _sanitize_forecast_array(x), [f"{float(q):g}" for q in sorted_q]

    q_count = int(max(1, x.shape[0]))
    default_q = np.linspace(0.1, 0.9, num=q_count, dtype=np.float32).tolist()
    return _sanitize_forecast_array(x), [f"{float(q):g}" for q in default_q]


def _sanitize_forecast_for_eval(forecast):
    item_id = getattr(forecast, "item_id", None)
    start_date = getattr(forecast, "start_date", None)

    if hasattr(forecast, "samples") and getattr(forecast, "samples", None) is not None:
        arr = _sanitize_forecast_array(np.asarray(getattr(forecast, "samples"), dtype=np.float32))
        return SampleForecast(
            samples=arr,
            start_date=start_date,
            item_id=item_id,
        )

    raw_keys = getattr(forecast, "forecast_keys", None)
    arr = _forecast_to_sample_array(forecast)
    q_arr, q_keys = _sanitize_quantile_payload(arr, list(map(str, raw_keys)) if raw_keys is not None else None)
    return QuantileForecast(
        item_id=item_id,
        forecast_arrays=q_arr,
        start_date=start_date,
        forecast_keys=q_keys,
    )


def _save_infer_cache(
    cache_path: Path,
    *,
    dense_records: list[dict],
    large_data_channel: bool = False,
) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    payload_path, kinds_path, qkeys_path = _infer_cache_component_paths(cache_path)

    f16_max = np.float32(np.finfo(np.float16).max)
    f16_min = np.float32(np.finfo(np.float16).min)
    record_count = int(len(dense_records))
    if record_count <= 0:
        return

    # Detect whether all payload arrays share one shape; if true we can stream-write
    # directly into a single contiguous .npy via memmap (much faster, low RAM).
    first_shape = None
    all_same_shape = True
    for rec in dense_records:
        shp = tuple(np.asarray(rec.get("payload")).shape)
        if first_shape is None:
            first_shape = shp
        elif shp != first_shape:
            all_same_shape = False
            break

    progress_step = max(1, int(record_count // 20))
    t0 = time.perf_counter()

    dense_kinds = np.array([str(rec.get("kind", "sample")) for rec in dense_records], dtype=object)
    dense_qkeys = np.array([rec.get("forecast_keys", None) for rec in dense_records], dtype=object)
    np.save(kinds_path, dense_kinds, allow_pickle=True)
    np.save(qkeys_path, dense_qkeys, allow_pickle=True)

    # Small/regular data path: preserve legacy behavior and write one contiguous array.
    if all_same_shape and first_shape is not None:
        est_bytes = int(record_count) * int(np.prod(np.asarray(first_shape, dtype=np.int64))) * 2
        if not bool(large_data_channel):
            print(
                f"[Refined-CSV] Cache save(v3) mode=legacy_stack_save | records={record_count} | est={est_bytes/(1024.0**3):.2f} GiB",
                flush=True,
            )
            payloads: list[np.ndarray] = []
            for i, rec in enumerate(dense_records):
                arr = _sanitize_forecast_array(np.asarray(rec.get("payload"), dtype=np.float32))
                arr = np.clip(arr, f16_min, f16_max).astype(np.float16, copy=False)
                payloads.append(arr)
                done = i + 1
                if done % progress_step == 0 or done == record_count:
                    elapsed = float(time.perf_counter() - t0)
                    print(
                        build_progress_line(
                            prefix="[Refined-CSV] Cache save(v3-stack)",
                            done=done,
                            total=record_count,
                            elapsed_seconds=elapsed,
                            unit="rec",
                        ),
                        flush=True,
                    )

            payload_array = np.stack(payloads, axis=0)
            np.save(payload_path, payload_array, allow_pickle=True)
            return

        print(
            f"[Refined-CSV] Cache save(v3) mode=large_data_memmap_stream | records={record_count} | est={est_bytes/(1024.0**3):.2f} GiB",
            flush=True,
        )
        payload_mm = np.lib.format.open_memmap(
            payload_path,
            mode="w+",
            dtype=np.float16,
            shape=(record_count, *tuple(first_shape)),
        )

        block_records = 512
        flush_every_blocks = 32
        block_count = (record_count + block_records - 1) // block_records
        for block_idx in range(block_count):
            s = int(block_idx * block_records)
            e = int(min(record_count, s + block_records))
            block_payloads: list[np.ndarray] = []
            for rec in dense_records[s:e]:
                arr = _sanitize_forecast_array(np.asarray(rec.get("payload"), dtype=np.float32))
                arr = np.clip(arr, f16_min, f16_max).astype(np.float16, copy=False)
                block_payloads.append(arr)
            payload_mm[s:e] = np.stack(block_payloads, axis=0)

            if ((block_idx + 1) % flush_every_blocks == 0) or (e == record_count):
                payload_mm.flush()

            done = e
            if done % progress_step == 0 or done == record_count:
                elapsed = float(time.perf_counter() - t0)
                print(
                    build_progress_line(
                        prefix="[Refined-CSV] Cache save(v3-memmap)",
                        done=done,
                        total=record_count,
                        elapsed_seconds=elapsed,
                        unit="rec",
                    ),
                    flush=True,
                )
        payload_mm.flush()
        del payload_mm
        return

    # Fallback for ragged payload shapes: keep exact v3 format via object array.
    print(
        f"[Refined-CSV] Cache save(v3-compat) fallback(object) start: records={record_count}",
        flush=True,
    )
    payloads: list[np.ndarray] = []
    for i, rec in enumerate(dense_records):
        arr = _sanitize_forecast_array(np.asarray(rec.get("payload"), dtype=np.float32))
        arr = np.clip(arr, f16_min, f16_max).astype(np.float16, copy=False)
        payloads.append(arr)
        done = i + 1
        if done % progress_step == 0 or done == record_count:
            elapsed = float(time.perf_counter() - t0)
            print(
                build_progress_line(
                    prefix="[Refined-CSV] Cache save(v3-compat)",
                    done=done,
                    total=record_count,
                    elapsed_seconds=elapsed,
                    unit="rec",
                ),
                flush=True,
            )
    payload_array = np.array(payloads, dtype=object)
    np.save(payload_path, payload_array, allow_pickle=True)


def _select_eval_samples_from_dense_stream(
    dense_input,
    dense_samples: list,
    eval_input,
) -> list:
    dense_n = min(int(len(dense_input)), int(len(dense_samples)))
    dense_lengths = [entry_series_length(dense_input[i]) for i in range(dense_n)]
    eval_lengths = [entry_series_length(entry) for entry in eval_input]

    out: list = []
    dense_idx = 0
    for target_len in eval_lengths:
        while dense_idx < dense_n and int(dense_lengths[dense_idx]) < int(target_len):
            dense_idx += 1
        if dense_idx >= dense_n:
            break
        if int(dense_lengths[dense_idx]) == int(target_len):
            out.append(dense_samples[dense_idx])
            dense_idx += 1
            continue

        # Fallback: keep best-effort monotonic pick if exact length is unavailable.
        out.append(dense_samples[dense_idx])
        dense_idx += 1

    if len(out) != len(eval_lengths):
        raise ValueError(
            f"Dense-to-eval alignment mismatch: dense_samples={dense_n}, eval_windows={len(eval_lengths)}, aligned={len(out)}"
        )
    return out


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


def _check_instance_order_monotonic(entries) -> dict[str, int]:
    prev_start = None
    comparable_pairs = 0
    non_increasing_pairs = 0
    for entry in entries:
        curr_start = _entry_forecast_start(entry)
        if prev_start is not None and curr_start is not None:
            try:
                comparable_pairs += 1
                if not bool(curr_start > prev_start):
                    non_increasing_pairs += 1
            except Exception:
                pass
        prev_start = curr_start
    return {
        "comparable_pairs": int(comparable_pairs),
        "non_increasing_pairs": int(non_increasing_pairs),
    }


class _CachedArrayPredictor:
    def __init__(self, samples_list: list) -> None:
        self.samples_list = list(samples_list)

    def predict(self, dataset, **kwargs):
        _ = kwargs
        for idx, entry in enumerate(dataset):
            if idx >= len(self.samples_list):
                break
            rec = self.samples_list[idx]
            if isinstance(rec, dict):
                kind = str(rec.get("kind", "sample")).lower()
                arr = _sanitize_forecast_array(np.asarray(rec.get("payload"), dtype=np.float32))
                fkeys = rec.get("forecast_keys", None)
            else:
                kind = "sample"
                arr = _sanitize_forecast_array(np.asarray(rec, dtype=np.float32))
                fkeys = None
            main = entry[0] if isinstance(entry, tuple) else entry
            item_id = main.get("item_id", None) if isinstance(main, dict) else None
            start_date = _entry_forecast_start(entry)
            if kind == "quantile":
                q_arr, qkeys = _sanitize_quantile_payload(
                    arr,
                    list(map(str, fkeys)) if fkeys else None,
                )
                yield QuantileForecast(
                    item_id=item_id,
                    forecast_arrays=q_arr,
                    start_date=start_date,
                    forecast_keys=qkeys,
                )
            else:
                yield SampleForecast(
                    samples=_sanitize_forecast_array(arr),
                    start_date=start_date,
                    item_id=item_id,
                )


class _RecordingPredictor:
    """Proxy predictor that records emitted forecasts during evaluation."""

    def __init__(self, inner_predictor, *, prediction_length: int | None = None, target_dim: int | None = None) -> None:
        self.inner_predictor = inner_predictor
        self.recorded_pred_windows: list[np.ndarray] = []
        self.prediction_length = prediction_length
        self.target_dim = target_dim
        self.raw_samples = []
        self.raw_records = []
        self.point_meta_windows: list[dict] = []

    def __getattr__(self, name):
        return getattr(self.inner_predictor, name)

    def reset_records(self) -> None:
        self.recorded_pred_windows = []
        self.raw_samples = []
        self.raw_records = []
        self.point_meta_windows = []

    def predict(self, dataset, **kwargs):
        self.reset_records()
        has_len = hasattr(dataset, "__len__")
        total = int(len(dataset)) if has_len else None
        progress_marks: set[int] = set()
        if total is not None and total > 0:
            for k in range(1, 11):
                progress_marks.add(max(1, int(round(float(total) * float(k) / 10.0))))

        wall_t0 = time.perf_counter()

        processed = 0
        try:
            forecast_iter = self.inner_predictor.predict(dataset, **kwargs)
        except TypeError as exc:
            msg = str(exc)
            if "unexpected keyword argument" in msg and "batch_size" in msg:
                kwargs_compat = dict(kwargs)
                kwargs_compat.pop("batch_size", None)
                forecast_iter = self.inner_predictor.predict(dataset, **kwargs_compat)
            else:
                raise

        for forecast in forecast_iter:
            safe_forecast = _sanitize_forecast_for_eval(forecast)
            raw_keys = getattr(safe_forecast, "forecast_keys", None)
            q_keys = list(map(str, raw_keys)) if raw_keys is not None else None
            processed += 1
            point_window = _forecast_samples_to_mean_window(
                _forecast_to_sample_array(safe_forecast),
                expected_pred_len=self.prediction_length,
                expected_target_dim=self.target_dim,
                forecast_keys=q_keys,
            )
            self.recorded_pred_windows.append(
                point_window
            )
            self.raw_samples.append(_forecast_to_sample_array(safe_forecast))
            self.raw_records.append(_forecast_to_cache_record(safe_forecast))
            selected_idx = select_quantile_index(q_keys, int(np.asarray(self.raw_samples[-1]).shape[0]), target_quantile=0.5)
            chosen_quantile = None
            if q_keys is not None and selected_idx is not None and 0 <= int(selected_idx) < len(q_keys):
                chosen_quantile = str(q_keys[int(selected_idx)])
            self.point_meta_windows.append(
                {
                    "window_idx": int(processed - 1),
                    "forecast_start": getattr(safe_forecast, "start_date", None),
                    "pred_shape": tuple(np.asarray(point_window).shape),
                    "chosen_quantile": chosen_quantile,
                }
            )

            if total is not None and processed in progress_marks:
                pct = 100.0 * float(processed) / float(max(1, total))
                if processed > 0:
                    try:
                        elapsed = float(time.perf_counter() - wall_t0)
                        print(
                            build_progress_line(
                                prefix="Model-Infer",
                                done=processed,
                                total=total,
                                elapsed_seconds=elapsed,
                                unit="it",
                            ),
                            flush=True,
                        )
                    except Exception:
                        print(f"Model-Infer: {processed}/{total} ({pct:.1f}%)", flush=True)
                else:
                    print(f"Model-Infer: {processed}/{total} ({pct:.1f}%)", flush=True)
            yield safe_forecast


class _SequentialChannelPackedPredictor:
    """Adapter: run single-channel backend per channel, then pack to multivariate forecast."""

    def __init__(self, base_predictor, target_dim: int) -> None:
        self.base_predictor = base_predictor
        self.prediction_length = int(getattr(base_predictor, "prediction_length", 1))
        self.context_length = getattr(base_predictor, "context_length", None)
        self.batch_size = int(getattr(base_predictor, "batch_size", 1))
        self.target_dim = int(max(1, target_dim))
        self._packed_batch_logged = False

    @staticmethod
    def _iter_entries(dataset) -> list[dict]:
        entries: list[dict] = []
        for entry in dataset:
            if isinstance(entry, tuple):
                entries.append(entry[0])
            else:
                entries.append(entry)
        return entries

    @staticmethod
    def _ensure_channel_time(arr: np.ndarray) -> np.ndarray:
        x = np.asarray(arr, dtype=np.float32)
        if x.ndim == 1:
            return x.reshape(1, -1)
        if x.ndim == 2:
            return x
        return x.reshape(x.shape[0], -1)

    @staticmethod
    def _forecast_start(entry: dict):
        start = entry.get("forecast_start", None)
        if start is not None:
            return start
        if "start" in entry and "target" in entry:
            target = np.asarray(entry["target"], dtype=np.float32)
            target_length = int(target.shape[0]) if target.ndim == 1 else int(target.shape[-1])
            return entry["start"] + target_length
        return entry.get("start", None)

    @staticmethod
    def _to_univariate_samples(arr: np.ndarray, *, expected_pred_len: int) -> np.ndarray:
        x = np.asarray(arr, dtype=np.float32)
        if x.ndim == 1:
            return x.reshape(1, -1)
        if x.ndim == 2:
            # Normalize ambiguous 2D samples to (S, L).
            # Some backends emit (L, S), especially when prediction length is known.
            l = int(expected_pred_len)
            if x.shape[1] == l and x.shape[0] != l:
                return x
            if x.shape[0] == l and x.shape[1] != l:
                return x.T
            return x
        if x.ndim == 3:
            l = int(expected_pred_len)
            # Accept common variants: (S, 1, L), (S, L, 1), (L, S, 1), (L, 1, S).
            if x.shape[1] == 1 and x.shape[2] == l:
                return x[:, 0, :]
            if x.shape[2] == 1 and x.shape[1] == l:
                return x[:, :, 0]
            if x.shape[0] == l and x.shape[2] == 1:
                return x[:, :, 0].T
            if x.shape[0] == l and x.shape[1] == 1:
                return x[:, 0, :].T

            if x.shape[-1] == l:
                return x.reshape(-1, l)
            if x.shape[0] == l:
                return x.reshape(l, -1).T
            return x.reshape(x.shape[0], -1)
        return x.reshape(1, -1)

    def _split_entry_per_channel(self, entry: dict) -> list[dict]:
        if "past_target" in entry:
            src = self._ensure_channel_time(entry["past_target"])
            field = "past_target"
        else:
            src = self._ensure_channel_time(entry["target"])
            field = "target"

        channel_entries: list[dict] = []
        for ch_idx in range(int(src.shape[0])):
            out = dict(entry)
            out[field] = np.asarray(src[ch_idx], dtype=np.float32)
            if "target" in out:
                tgt = self._ensure_channel_time(entry["target"])
                out["target"] = np.asarray(tgt[ch_idx], dtype=np.float32)
            out["item_id"] = f"{entry.get('item_id', 'item')}_ch{ch_idx}"
            channel_entries.append(out)
        return channel_entries

    def _pack_quantile_forecast(self, channel_forecasts: list[Forecast], entry: dict) -> QuantileForecast:
        raw_keys = list(map(str, getattr(channel_forecasts[0], "forecast_keys", [])))

        def _is_quantile_key(key: str) -> bool:
            k = str(key).strip().lower()
            if k.startswith("p") and len(k) > 1:
                try:
                    float(k[1:])
                    return True
                except Exception:
                    return False
            try:
                float(k)
                return True
            except Exception:
                return False

        q_keys = [k for k in raw_keys if _is_quantile_key(k)]
        if not q_keys:
            raise RuntimeError(
                f"No numeric quantile keys available for packing. raw_keys={raw_keys}"
            )

        channel_q_arrays: list[np.ndarray] = []
        pred_len: int | None = None
        for fcst in channel_forecasts:
            per_q: list[np.ndarray] = []
            for q in q_keys:
                q_arr = _sanitize_forecast_array(np.asarray(fcst.quantile(str(q)), dtype=np.float32))
                if q_arr.ndim == 2 and q_arr.shape[-1] == 1:
                    q_arr = q_arr[:, 0]
                if q_arr.ndim != 1:
                    q_arr = q_arr.reshape(-1)
                per_q.append(q_arr)
            q_stack = np.stack(per_q, axis=0)  # (Q, L)
            pred_len = q_stack.shape[1] if pred_len is None else min(pred_len, int(q_stack.shape[1]))
            channel_q_arrays.append(q_stack)

        if pred_len is None:
            pred_len = int(self.prediction_length)
        aligned = [arr[:, :pred_len] for arr in channel_q_arrays]
        packed = _sanitize_forecast_array(np.stack(aligned, axis=-1))  # (Q, L, D)
        return QuantileForecast(
            item_id=entry.get("item_id"),
            forecast_arrays=_sanitize_forecast_array(np.asarray(packed, dtype=np.float32)),
            start_date=self._forecast_start(entry),
            forecast_keys=q_keys,
        )

    def _pack_sample_forecast(self, channel_forecasts: list[Forecast], entry: dict) -> SampleForecast:
        channel_samples: list[np.ndarray] = []
        sample_count: int | None = None
        pred_len: int | None = None
        for fcst in channel_forecasts:
            arr = _sanitize_forecast_array(np.asarray(getattr(fcst, "samples"), dtype=np.float32))
            s = self._to_univariate_samples(arr, expected_pred_len=int(self.prediction_length))  # (S, L)
            sample_count = s.shape[0] if sample_count is None else min(sample_count, int(s.shape[0]))
            pred_len = s.shape[1] if pred_len is None else min(pred_len, int(s.shape[1]))
            channel_samples.append(s)

        if sample_count is None or pred_len is None:
            sample_count = 1
            pred_len = int(self.prediction_length)
        aligned = [arr[:sample_count, :pred_len] for arr in channel_samples]
        # Defensive fallback: if horizon got into axis-0 due backend shape oddities, transpose.
        if int(pred_len) != int(self.prediction_length) and int(sample_count) == int(self.prediction_length):
            aligned = [arr.T for arr in aligned]
            sample_count, pred_len = pred_len, sample_count
        # GluonTS multivariate SampleForecast expects (S, L, D).
        packed = _sanitize_forecast_array(np.stack(aligned, axis=-1))  # (S, L, D)
        return SampleForecast(
            samples=_sanitize_forecast_array(np.asarray(packed, dtype=np.float32)),
            start_date=self._forecast_start(entry),
            item_id=entry.get("item_id"),
        )

    def predict(self, dataset, **kwargs):
        entries = self._iter_entries(dataset)
        if not entries:
            return

        # Keep memory bounded but avoid one backend invocation per window.
        # Process windows in chunks and pack per-window forecasts from flattened channel outputs.
        request_batch = kwargs.get("batch_size", None)
        try:
            window_batch_size = max(1, int(request_batch)) if request_batch is not None else int(self.batch_size)
        except Exception:
            window_batch_size = int(self.batch_size)
        window_batch_size = max(1, int(window_batch_size))

        for start in range(0, len(entries), window_batch_size):
            window_chunk = entries[start : start + window_batch_size]
            flat_channel_entries: list[dict] = []
            channel_counts: list[int] = []
            for entry in window_chunk:
                ch_entries = self._split_entry_per_channel(entry)
                flat_channel_entries.extend(ch_entries)
                channel_counts.append(len(ch_entries))

            # Preserve window-level batching semantics for single-channel backends.
            # If a window has D channels, backend work scales with D flattened entries.
            backend_kwargs = dict(kwargs)
            channel_factor = max(channel_counts) if channel_counts else 1
            backend_kwargs["batch_size"] = max(1, int(window_batch_size) * int(channel_factor))
            if not bool(self._packed_batch_logged):
                print(
                    f"[Refined-CSV][InferBatchPacked] window_batch={window_batch_size} | "
                    f"channel_factor={channel_factor} | backend_batch={int(backend_kwargs['batch_size'])}",
                    flush=True,
                )
                self._packed_batch_logged = True

            try:
                flat_fcsts = list(self.base_predictor.predict(flat_channel_entries, **backend_kwargs))
            except TypeError as exc:
                msg = str(exc)
                if "unexpected keyword argument" in msg and "batch_size" in msg:
                    kwargs_compat = dict(backend_kwargs)
                    kwargs_compat.pop("batch_size", None)
                    flat_fcsts = list(self.base_predictor.predict(flat_channel_entries, **kwargs_compat))
                else:
                    raise

            if len(flat_fcsts) != len(flat_channel_entries):
                raise RuntimeError(
                    f"Forecast count mismatch in sequential channel packing chunk: forecasts={len(flat_fcsts)} entries={len(flat_channel_entries)}"
                )

            offset = 0
            for entry, ch_count in zip(window_chunk, channel_counts):
                group_fcsts = flat_fcsts[offset : offset + ch_count]
                offset += ch_count

                has_samples = all(hasattr(fcst, "samples") and getattr(fcst, "samples", None) is not None for fcst in group_fcsts)
                has_quantile = all(getattr(fcst, "forecast_keys", None) for fcst in group_fcsts)

                if has_samples:
                    yield self._pack_sample_forecast(group_fcsts, entry)
                elif has_quantile:
                    yield self._pack_quantile_forecast(group_fcsts, entry)
                else:
                    raise RuntimeError(
                        f"Unsupported forecast objects in sequential channel packing: {type(group_fcsts[0]).__name__}"
                    )


def _build_predictor(ds: CsvSeriesDataset, args, device: torch.device):
    model_name = str(getattr(args, "model", "moirai-2"))
    model_ref = resolve_model_ref(
        model_name=model_name,
        tsfm_local_path=getattr(args, "tsfm_local_path", None),
        download_online=bool(getattr(args, "download_online", False)),
    )
    supports_mv = bool(model_supports_multivariate(model_name))
    backend_target_dim = int(ds.target_dim) if supports_mv else 1
    predictor = create_base_predictor(
        model_name=model_name,
        model_ref=model_ref,
        prediction_length=int(ds.prediction_length),
        context_length=int(args.context_length),
        target_dim=int(backend_target_dim),
        batch_size=int(args.batch_size),
        device=device,
        chronos_predict_batches_jointly=bool(getattr(args, "chronos_predict_batches_jointly", False)),
    )
    if int(ds.target_dim) > 1 and not supports_mv:
        return _SequentialChannelPackedPredictor(predictor, target_dim=int(ds.target_dim))
    return predictor


def _measure_base_model_single_inference_seconds(
    predictor,
    input_entries,
    *,
    steps: int = 10,
    batch_size: int = 1,
) -> float:
    """Measure average single-step inference latency of the base model."""
    if input_entries is None:
        return float("nan")

    total_entries = int(len(input_entries))
    measure_steps = min(max(0, int(steps)), total_entries)
    if measure_steps <= 0:
        return float("nan")

    subset = list(input_entries[:measure_steps])
    start_t = time.perf_counter()
    try:
        forecast_iter = predictor.predict(subset, batch_size=int(batch_size))
    except TypeError as exc:
        msg = str(exc)
        if "unexpected keyword argument" in msg and "batch_size" in msg:
            forecast_iter = predictor.predict(subset)
        else:
            raise

    produced = 0
    for _ in forecast_iter:
        produced += 1
        if produced >= measure_steps:
            break

    elapsed = max(1e-12, float(time.perf_counter() - start_t))
    if produced <= 0:
        return float("nan")
    return float(elapsed / float(produced))


def _build_refiner(
    args,
    device: torch.device,
    *,
    target_dim: int,
    train_window_count: int,
    val_window_count: int,
):
    refiner = _resolve_refiner(args)
    if refiner == "linear":
        return OnlineRefinerLinear(
            feature_dim=int(target_dim),
            device=device,
            collect_train_windows=int(train_window_count),
            collect_val_windows=int(val_window_count),
            refiner_input=_resolve_refiner_input(args),
            online_training=(_resolve_training_method(args) == "online"),
            update_rule=_resolve_update_rule(args),
            force_gate_open=_resolve_force_gate_open(args),
            channel_mix=_resolve_channel_mix(args),
        )
    if refiner == "bay":
        return OnlineRefinerBayesian(
            feature_dim=int(target_dim),
            device=device,
            collect_train_windows=int(train_window_count),
            refiner_input=_resolve_refiner_input(args),
            update_rule=_resolve_bay_update_rule(args),
            loss_variant=_resolve_bay_loss(args),
            router=_resolve_bay_router(args),
            routing_temperature=_resolve_routing_temperature(args),
            ema_error_momentum=_resolve_ema_error_momentum(args),
            huber_delta=_coerce_args_float(getattr(args, "bay_huber_delta", 1.0), 1.0),
            force_gate_open=_resolve_force_gate_open(args),
            channel_mix=_resolve_channel_mix(args),
            train_batch_size=_coerce_args_int(getattr(args, "batch", 256), 256),
        )
    if refiner == "attn":
        return OnlineRefinerAttn(
            feature_dim=int(target_dim),
            device=device,
            collect_train_windows=int(train_window_count),
            collect_val_windows=int(val_window_count),
            refiner_input=_resolve_refiner_input(args),
            online_training=(_resolve_training_method(args) == "online"),
            update_rule=_resolve_update_rule(args),
            force_gate_open=_resolve_force_gate_open(args),
            channel_mix=_resolve_channel_mix(args),
        )
    if refiner == "bay_attn":
        return OnlineRefinerBayAttn(
            feature_dim=int(target_dim),
            device=device,
            collect_train_windows=int(train_window_count),
            refiner_input=_resolve_refiner_input(args),
            update_rule=_resolve_update_rule(args),
            force_gate_open=_resolve_force_gate_open(args),
            channel_mix=_resolve_channel_mix(args),
            train_batch_size=_coerce_args_int(getattr(args, "batch", 256), 256),
        )
    if refiner == "aday":
        # Keep AdaY close to paper/reference defaults while fitting this framework's online protocol.
        aday_lr = 1e-4
        aday_delta = _resolve_aday_delta(args)
        aday_grad_clip = 1e-2
        return OnlineRefinerAdaY(
            feature_dim=int(target_dim),
            lr=aday_lr,
            device=device,
            delta=aday_delta,
            grad_clip=aday_grad_clip,
            collect_train_windows=int(train_window_count),
            collect_val_windows=int(val_window_count),
            online_training=(_resolve_training_method(args) == "online"),
            baseline_router=bool(getattr(args, "baseline_router", False)),
            seq_len_hint=_coerce_args_int(getattr(args, "context_length", 0), 0),
            warmup_buffer_windows=_coerce_args_int(getattr(args, "online_buffer_windows", 3000), 3000),
            warmup_batch_size=_coerce_args_int(getattr(args, "batch", 256), 256),
            warmup_epochs=10,
            warmup_patience=3,
            warmup_val_ratio=0.1,
            online_lr=1e-6,
            online_grad_clip=1e-4,
        )
    if refiner == "dsof":
        return OnlineRefinerDSOF(
            feature_dim=int(target_dim),
            hidden_dim=128,
            num_blocks=3,
            lr=1e-3,
            device=device,
            collect_train_windows=int(train_window_count),
            collect_val_windows=int(val_window_count),
            online_training=(_resolve_training_method(args) == "online"),
            baseline_router=bool(getattr(args, "baseline_router", False)),
            seq_len_hint=_coerce_args_int(getattr(args, "context_length", 0), 0),
            replay_buffer_size=300,
            batch_replay_size=32,
            num_er_epochs=1,
            freq_er_update=1,
            td_enabled=True,
            td_k=1,
            discounted=0.9,
            warmup_buffer_windows=_coerce_args_int(getattr(args, "online_buffer_windows", 3000), 3000),
            warmup_batch_size=_coerce_args_int(getattr(args, "batch", 256), 256),
            warmup_epochs=10,
            warmup_patience=3,
            warmup_val_ratio=0.1,
            warmup_lr=1e-3,
            online_batch_lr=1e-4,
            online_td_lr=3e-4,
            grad_clip=1e-2,
        )
    if refiner == "tafas":
        return OnlineRefinerTAFAS(
            feature_dim=int(target_dim),
            device=device,
            collect_train_windows=int(train_window_count),
            collect_val_windows=int(val_window_count),
            online_training=(_resolve_training_method(args) == "online"),
            baseline_router=bool(getattr(args, "baseline_router", False)),
        )
    if refiner == "solid":
        solid_period = _resolve_solid_period(args)
        return OnlineRefinerSOLID(
            feature_dim=int(target_dim),
            device=device,
            collect_train_windows=int(train_window_count),
            collect_val_windows=int(val_window_count),
            online_training=(_resolve_training_method(args) == "online"),
            baseline_router=bool(getattr(args, "baseline_router", False)),
            period=int(solid_period),
        )
    if refiner == "elf":
        return OnlineRefinerELF(
            feature_dim=int(target_dim),
            stride=1,
            device=device,
            baseline_router=bool(getattr(args, "baseline_router", False)),
        )
    if refiner == "ridge":
        return OnlineRefinerRidge(
            feature_dim=int(target_dim),
            device=device,
            collect_train_windows=int(train_window_count),
        )
    if refiner == "arima":
        return OnlineRefinerARIMA(
            feature_dim=int(target_dim),
            device=device,
            max_history_steps=max(2048, int(train_window_count) * int(getattr(args, "pred_len", 96))),
        )
    if refiner == "ets":
        return OnlineRefinerETS(
            feature_dim=int(target_dim),
            device=device,
        )

    raise ValueError(f"Unsupported refiner={refiner!r}.")


def run_geoflow_csv_evaluation(args, device: torch.device) -> dict:
    ds = _get_or_create_csv_dataset(args)

    channel_names = list(getattr(ds, "selected_columns", list(ds.dataframe.columns)))
    if not channel_names:
        raise ValueError("No numeric channels found in CSV dataset")
    last_channel = channel_names[-1]
    attn_maps_enabled = _resolve_attn_maps_enabled(args)
    save_plots_enabled = True
    if attn_maps_enabled and save_plots_enabled:
        print("[Refined-CSV] Attention-map mode active: disable comparison plots for this run.")
        save_plots_enabled = False

    print(f"[Refined-CSV] CSV={Path(args.csv_path).name} | pred_len={ds.prediction_length} | channels={len(channel_names)}")
    print(
        "[Refined-CSV] Evaluation mode: multivariate windows (shared slicing), "
        "single-channel backends use sequential channel packing"
    )
    print(f"[Refined-CSV] comparison_plots_enabled={save_plots_enabled} | target_dim={ds.target_dim}")
    print(f"[Refined-CSV] Batch config: batch_size={int(args.batch_size)}")
    dataset_name = Path(args.csv_path).stem
    large_data_channel = _is_large_data_channel(dataset_name=dataset_name, pred_len=int(ds.prediction_length))
    if large_data_channel:
        print(
            "[Refined-CSV] Large-data channel ON: dataset in {Traffic, Electricity} and pred_len > 100.",
            flush=True,
        )

    if attn_maps_enabled:
        min_context_length = _coerce_args_int(getattr(args, "context_length", 0), 0)
        baseline_test_data_raw = ds.test_data
        baseline_test_data = filter_test_data_by_context_length(baseline_test_data_raw, min_context_length)
        if len(baseline_test_data.input) == 0:
            raise ValueError(
                f"No baseline windows left after context-length filter: context_length={min_context_length}"
            )

        total_baseline_windows = int(len(baseline_test_data.input))
        train_meta_window_count, val_meta_window_count, test_meta_window_count = split_window_counts(
            total_baseline_windows,
            train_ratio=0.7,
            val_ratio=0.1,
        )
        test_start = int(train_meta_window_count + val_meta_window_count)
        test_end = int(test_start + test_meta_window_count)

        baseline_eval_data = slice_filtered_test_data(baseline_test_data, start=test_start, end=test_end)
        if len(baseline_eval_data.input) == 0:
            raise ValueError("No evaluation window available after train/val/test split")
        train_meta_partition = slice_filtered_test_data(
            baseline_test_data,
            start=0,
            end=int(train_meta_window_count),
        )
        use_channelwise_norm = _use_electricity_channelwise_norm(dataset_name)
        train_global_mean_scale = _compute_train_global_mean_abs_scale(train_meta_partition.input)
        train_channel_mean_scale = (
            _compute_train_channel_mean_abs_scale(train_meta_partition.input) if use_channelwise_norm else None
        )

        base_predictor = _build_predictor(ds, args, device)
        dataset_name = Path(args.csv_path).stem
        model_name = str(getattr(args, "model", "")).strip().lower()

        attn_summary = inspect_attention_maps(
            model_name=model_name,
            predictor=base_predictor,
            eval_input=baseline_eval_data.input,
            dataset_name=dataset_name,
            max_windows=1,
        )
        print(
            f"[Refined-CSV] Attention maps saved: count={int(attn_summary.get('num_maps', 0))} | "
            f"dir={attn_summary.get('output_dir')}"
        )
        if attn_summary.get("warning"):
            print(
                f"[Refined-CSV] Attention-map warning: {attn_summary.get('warning')} | "
                f"q_proj_captured={attn_summary.get('q_proj_captured', 0)} | "
                f"k_proj_captured={attn_summary.get('k_proj_captured', 0)} | "
                f"weight_proxy_maps={attn_summary.get('weight_proxy_maps', 0)}"
            )

        base_recording_predictor = _RecordingPredictor(
            base_predictor,
            prediction_length=int(ds.prediction_length),
            target_dim=int(ds.target_dim),
        )
        for _ in base_recording_predictor.predict(baseline_eval_data.input, batch_size=args.batch_size):
            pass
        baseline_gt_windows_metric = _build_gt_windows_from_labels(baseline_eval_data.label)
        _, baseline_pred_windows_metric = _align_gt_pred_windows(
            baseline_gt_windows_metric,
            list(getattr(base_recording_predictor, "recorded_pred_windows", [])),
        )
        base_point_metrics = _compute_point_primary_metrics(
            baseline_gt_windows_metric,
            baseline_pred_windows_metric,
        )
        if use_channelwise_norm:
            agg_metrics_base = _inject_scaled_primary_metrics_channelwise(
                base_point_metrics,
                gt_windows=baseline_gt_windows_metric,
                pred_windows=baseline_pred_windows_metric,
                channel_mean_abs_scale=train_channel_mean_scale,
            )
        else:
            agg_metrics_base = _inject_scaled_primary_metrics(
                base_point_metrics,
                mean_abs_scale=train_global_mean_scale,
            )
        agg_metrics_flow = dict(agg_metrics_base)
        agg_metrics_base = _project_core_metric_keys(agg_metrics_base)
        agg_metrics_flow = _project_core_metric_keys(agg_metrics_flow)
        if agg_metrics_base is not None:
            print(
                f"[Refined-CSV][Baseline] {PRIMARY_METRIC_LABEL_1}={first_value(agg_metrics_base.get(PRIMARY_METRIC_KEY_1)):.4f} | "
                f"{PRIMARY_METRIC_LABEL_2}={first_value(agg_metrics_base.get(PRIMARY_METRIC_KEY_2)):.4f}"
            )

        refiner_tag = _resolve_refiner_tag(_resolve_refiner(args))
        return {
            "dataset_name": dataset_name,
            "model_short_name": str(getattr(args, "model", "unknown")),
            "refiner_tag": refiner_tag,
            "pred_len": int(ds.prediction_length),
            "agg_metrics_base": agg_metrics_base,
            "agg_metrics_flow": agg_metrics_flow,
            "window_count": 0,
            "flow_steps": 0,
            "eval_window_count": int(len(baseline_eval_data.input)),
            "meta_window_count": int(total_baseline_windows),
            "update_window_count": 0,
            "train_meta_window_count": int(train_meta_window_count),
            "val_meta_window_count": int(val_meta_window_count),
            "test_meta_window_count": int(test_meta_window_count),
            "train_update_window_count": 0,
            "val_update_window_count": 0,
            "test_update_window_count": 0,
            "loss_history": [],
            "val_loss_history": [],
            "attn_map_summary": attn_summary,
        }

    stride = 1
    flow_windows_req = int(ds.windows) * int(ds.prediction_length)
    flow_update_test_data_raw, flow_windows = ds.build_test_data(distance=int(stride), windows=flow_windows_req)

    min_context_length = _coerce_args_int(getattr(args, "context_length", 0), 0)
    flow_update_test_data = filter_test_data_by_context_length(flow_update_test_data_raw, min_context_length)

    if len(flow_update_test_data.input) == 0:
        raise ValueError(
            f"No flow-update windows left after context-length filter: context_length={min_context_length}"
        )
    total_update_windows = int(len(flow_update_test_data.input))

    order_diag = _check_instance_order_monotonic(flow_update_test_data.input)
    if int(order_diag["comparable_pairs"]) > 0 and int(order_diag["non_increasing_pairs"]) > 0:
        raise RuntimeError(
            "Detected non-monotonic GluonTS instance order in flow update stream. "
            "This can deterministically corrupt online closure/update alignment."
        )

    training_method = _resolve_training_method(args)
    online_buffer_windows = _resolve_online_buffer_windows(args)
    refiner_key = _resolve_refiner(args)
    eval_train_window_count, eval_val_window_count, eval_test_window_count = split_window_counts(
        total_update_windows,
        train_ratio=0.7,
        val_ratio=0.1,
    )
    refiner_train_window_count = int(eval_train_window_count)
    refiner_val_window_count = int(eval_val_window_count)
    if training_method == "online" and refiner_key in {"linear", "attn", "bay", "bay_attn", "dsof", "tafas", "solid"}:
        # Online retrain trigger uses stride-1 mini-window buffer and is independent of eval split.
        refiner_train_window_count = min(max(1, int(online_buffer_windows)), int(total_update_windows))
        refiner_val_window_count = 1
    test_start = int(eval_train_window_count + eval_val_window_count)
    test_end = int(test_start + eval_test_window_count)

    # Single-stream execution for updates; metrics are computed on test split only.
    baseline_eval_data = slice_filtered_test_data(flow_update_test_data, start=test_start, end=test_end)
    if len(baseline_eval_data.input) == 0:
        raise ValueError(
            "No evaluation window available in test split "
            f"(total={int(total_update_windows)}, train={int(eval_train_window_count)}, "
            f"val={int(eval_val_window_count)}, test={int(eval_test_window_count)}, "
            f"online_buffer_windows={int(online_buffer_windows)}, pred_len={int(ds.prediction_length)})"
        )

    total_baseline_windows = int(total_update_windows)
    train_meta_window_count = int(eval_train_window_count)
    val_meta_window_count = int(eval_val_window_count)
    test_meta_window_count = int(eval_test_window_count)
    train_scale_partition = slice_filtered_test_data(
        flow_update_test_data,
        start=0,
        end=int(eval_train_window_count),
    )
    scale_entries = train_scale_partition.input if len(train_scale_partition.input) > 0 else flow_update_test_data.input
    use_channelwise_norm = _use_electricity_channelwise_norm(dataset_name)
    train_global_mean_scale = _compute_train_global_mean_abs_scale(scale_entries)
    train_channel_mean_scale = _compute_train_channel_mean_abs_scale(scale_entries) if use_channelwise_norm else None

    window_count_est, flow_steps_est, stream_count = compute_window_and_update_steps_for_test_data(
        flow_update_test_data,
        pred_len=int(ds.prediction_length),
        stride=int(stride),
    )
    print(
        f"[Refined-CSV][{Path(args.csv_path).stem}] stream_count={stream_count} | baseline_windows={ds.windows} | flow_windows={flow_windows} | "
        f"window_count_est={window_count_est} | refiner_update_steps_est={flow_steps_est} | "
        f"context_filter={min_context_length} | baseline_eval_windows={len(baseline_eval_data.input)} | "
        f"flow_eval_windows={len(flow_update_test_data.input)} | target_dim={ds.target_dim}"
    )
    print(
        f"[Refined-CSV] Window split (meta windows, for evaluation): total={total_baseline_windows} | "
        f"train={train_meta_window_count} | val={val_meta_window_count} | test={test_meta_window_count} | eval_test_partition={len(baseline_eval_data.input)}"
    )
    print(
        f"[Refined-CSV] Window split (mini windows, for online updates): total={total_update_windows} | "
        f"train={eval_train_window_count} | val={eval_val_window_count} | test={eval_test_window_count}"
    )
    if training_method == "online" and refiner_key in {"linear", "attn", "bay", "bay_attn", "dsof", "tafas", "solid"}:
        print(
            f"[Refined-CSV] online_buffer_windows={int(online_buffer_windows)} (unit=stride-1 mini windows)",
            flush=True,
        )
        print(
            f"[Refined-CSV] Refiner retrain trigger windows: train_buffer={int(refiner_train_window_count)} | val_marker={int(refiner_val_window_count)}",
            flush=True,
        )
    print(
        "[Refined-CSV] Single-stream mode active: refiner runs on full stride=1 stream; metrics/logs use test split only.",
        flush=True,
    )

    static_mean_scale = _to_static_mean_scale_tensor(
        train_global_mean_scale,
        channel_mean_scale=(train_channel_mean_scale if use_channelwise_norm else None),
    )
    if static_mean_scale is not None:
        preview = ", ".join(f"{float(v):.4g}" for v in static_mean_scale.reshape(-1)[:5])
        scale_mode = "channelwise" if use_channelwise_norm else "global"
        print(
            f"[Refined-CSV] Refiner static scaler ({scale_mode} mean abs) ready: channels={int(static_mean_scale.shape[-1])} | preview=[{preview}]"
        )

    cache_requested = _resolve_infer_cache_enabled(args)
    refiner_for_eval = _resolve_refiner(args)
    cache_enabled = bool(cache_requested)
    cache_path = _build_infer_cache_path(args, Path(args.csv_path).stem) if cache_enabled else None
    cache_load_t0 = time.perf_counter()
    if cache_path is not None:
        print(f"[Refined-CSV] Cache load stage start: path={cache_path}", flush=True)
    cache_payload = _load_infer_cache(cache_path) if cache_path is not None else None
    cache_load_dt = max(1e-9, float(time.perf_counter() - cache_load_t0))
    if cache_path is not None:
        print(
            f"[Refined-CSV] Cache load stage done: path={cache_path} | elapsed={cache_load_dt:.2f}s",
            flush=True,
        )
    cached_dense_records = [] if cache_payload is None else list(cache_payload.get("dense_records", []))
    dense_needed = int(len(flow_update_test_data.input))
    dense_cache_hit = len(cached_dense_records) >= dense_needed
    use_dense_cache_pipeline = bool(cache_enabled)
    skip_refiner_for_cache_build = bool(cache_enabled and (not dense_cache_hit))
    if bool(getattr(args, "speed", False)):
        skip_refiner_for_cache_build = False
    if cache_enabled:
        if dense_cache_hit:
            print(f"[Refined-CSV] Inference cache hit: {cache_path}")
        else:
            print(f"[Refined-CSV] Inference cache miss: {cache_path}")

    base_predictor = _build_predictor(ds, args, device)
    base_model_single_infer_time = float("nan")
    if bool(getattr(args, "speed", False)):
        print("[Refined-CSV] Speed mode: base-model 10-step re-inference benchmark start.", flush=True)
        base_model_single_infer_time = _measure_base_model_single_inference_seconds(
            base_predictor,
            flow_update_test_data.input,
            steps=10,
            batch_size=1,
        )
        if np.isfinite(base_model_single_infer_time):
            print(
                f"[Refined-CSV] Speed mode: base-model avg single inference={base_model_single_infer_time * 1000.0:.4f} ms over 10 steps.",
                flush=True,
            )
        else:
            print("[Refined-CSV] Speed mode: base-model 10-step benchmark unavailable (NaN).", flush=True)
    dense_record_predictor = _RecordingPredictor(
        base_predictor,
        prediction_length=int(ds.prediction_length),
        target_dim=int(ds.target_dim),
    )

    dense_records_full: list[dict] = []
    if use_dense_cache_pipeline:
        if dense_cache_hit:
            dense_records_full = cached_dense_records[:dense_needed]
        else:
            print(
                f"[Refined-CSV] Building dense inference cache payload: total_dense_windows={dense_needed} (stride=1)",
                flush=True,
            )
            if large_data_channel:
                print(
                    "[Refined-CSV] Large-data inference path active: keep high-throughput batch predict + progress/ETA logging.",
                    flush=True,
                )
            dense_batch_size = max(1, int(args.batch_size))
            dense_dt = 0.0
            while True:
                try:
                    dense_t0 = time.perf_counter()
                    for _ in dense_record_predictor.predict(flow_update_test_data.input, batch_size=dense_batch_size):
                        pass
                    dense_dt = max(1e-9, float(time.perf_counter() - dense_t0))
                    dense_records_full = list(getattr(dense_record_predictor, "raw_records", []))
                    break
                except Exception as exc:
                    if not (skip_refiner_for_cache_build and _is_oom_exception(exc)):
                        raise
                    if dense_batch_size <= 1:
                        raise RuntimeError(
                            "[Refined-CSV] Cache build OOM fallback exhausted at batch_size=1."
                        ) from exc
                    new_batch_size = max(1, int(dense_batch_size) // 4)
                    print(
                        f"[Refined-CSV][InferBatch][cache-build] OOM fallback: batch_size {dense_batch_size} -> {new_batch_size}",
                        flush=True,
                    )
                    dense_batch_size = new_batch_size
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
            if len(dense_records_full) < dense_needed:
                raise ValueError(
                    f"Dense inference outputs are insufficient: expected={dense_needed}, got={len(dense_records_full)}"
                )
            print(
                f"[Refined-CSV] Dense inference finished: collected={len(dense_records_full)} | elapsed={format_duration_dhms(dense_dt)} | throughput={len(dense_records_full)/dense_dt:.2f}it/s",
                flush=True,
            )
            if cache_enabled and cache_path is not None:
                if large_data_channel:
                    print(
                        "[Refined-CSV] Large-data save path active: v3 memmap stream with large blocks.",
                        flush=True,
                    )
                _save_infer_cache(
                    cache_path,
                    dense_records=dense_records_full,
                    large_data_channel=bool(large_data_channel),
                )
                print(f"[Refined-CSV] Inference cache saved: {cache_path}")
                print(
                    "[Refined-CSV] Cache-build mode on miss: skip refiner for this run.",
                    flush=True,
                )

    if use_dense_cache_pipeline:
        baseline_eval_samples = list(dense_records_full[: int(len(flow_update_test_data.input))])
        if len(baseline_eval_samples) < int(len(flow_update_test_data.input)):
            raise ValueError(
                f"Dense inference outputs are insufficient for baseline evaluation: expected={len(flow_update_test_data.input)}, got={len(baseline_eval_samples)}"
            )
        baseline_predictor = _CachedArrayPredictor(baseline_eval_samples)
    else:
        baseline_predictor = base_predictor

    print(
        f"[Refined-CSV] Baseline evaluate stage start: full_windows={len(flow_update_test_data.input)} | test_windows={len(baseline_eval_data.input)}",
        flush=True,
    )
    baseline_eval_t0 = time.perf_counter()
    baseline_recording_predictor = _RecordingPredictor(
        baseline_predictor,
        prediction_length=int(ds.prediction_length),
        target_dim=int(ds.target_dim),
    )
    for _ in baseline_recording_predictor.predict(flow_update_test_data.input, batch_size=args.batch_size):
        pass
    baseline_eval_dt = max(1e-9, float(time.perf_counter() - baseline_eval_t0))
    print(
        f"[Refined-CSV] Baseline evaluate stage done: elapsed={baseline_eval_dt:.2f}s",
        flush=True,
    )

    baseline_gt_windows_metric = _build_gt_windows_from_labels(baseline_eval_data.label)
    baseline_recorded_windows_all = list(getattr(baseline_recording_predictor, "recorded_pred_windows", []))
    baseline_meta_all = list(getattr(baseline_recording_predictor, "point_meta_windows", []))
    baseline_recorded_windows = baseline_recorded_windows_all[int(test_start):int(test_end)]
    baseline_meta_windows = baseline_meta_all[int(test_start):int(test_end)]
    baseline_gt_windows_metric, baseline_pred_windows_metric = _align_gt_pred_windows(
        baseline_gt_windows_metric,
        baseline_recorded_windows,
    )
    _audit_window_alignment(
        baseline_gt_windows_metric,
        baseline_pred_windows_metric,
        baseline_meta_windows,
    )
    baseline_window_audit_rows = _compose_window_audit_rows(
        baseline_gt_windows_metric,
        baseline_pred_windows_metric,
        baseline_meta_windows,
    )
    base_point_metrics = _compute_point_primary_metrics(
        baseline_gt_windows_metric,
        baseline_pred_windows_metric,
    )

    baseline_gt_windows: list[np.ndarray] = []
    baseline_pred_windows: list[np.ndarray] = []
    flow_pred_windows: list[np.ndarray] = []

    if save_plots_enabled:
        baseline_gt_windows = [w.copy() for w in baseline_gt_windows_metric]
        baseline_pred_windows = [w.copy() for w in baseline_pred_windows_metric]

    flow_pred_windows_metric: list[np.ndarray] = []

    if isinstance(flow_steps_est, int) and flow_steps_est <= 0:
        if use_channelwise_norm:
            agg_metrics_base = _inject_scaled_primary_metrics_channelwise(
                base_point_metrics,
                gt_windows=baseline_gt_windows_metric,
                pred_windows=baseline_pred_windows_metric,
                channel_mean_abs_scale=train_channel_mean_scale,
            )
        else:
            agg_metrics_base = _inject_scaled_primary_metrics(
                base_point_metrics,
                mean_abs_scale=train_global_mean_scale,
            )
        agg_metrics_flow = dict(agg_metrics_base)
        final_window_count = int(window_count_est)
        final_flow_steps = 0
        loss_history = []
        val_loss_history = []
        flow_pred_windows_metric = [w.copy() for w in baseline_pred_windows_metric]
        if save_plots_enabled:
            flow_pred_windows = [w.copy() for w in baseline_pred_windows]
    elif skip_refiner_for_cache_build:
        if use_channelwise_norm:
            agg_metrics_base = _inject_scaled_primary_metrics_channelwise(
                base_point_metrics,
                gt_windows=baseline_gt_windows_metric,
                pred_windows=baseline_pred_windows_metric,
                channel_mean_abs_scale=train_channel_mean_scale,
            )
        else:
            agg_metrics_base = _inject_scaled_primary_metrics(
                base_point_metrics,
                mean_abs_scale=train_global_mean_scale,
            )
        agg_metrics_flow = dict(agg_metrics_base)
        final_window_count = int(window_count_est)
        final_flow_steps = 0
        loss_history = []
        val_loss_history = []
        flow_pred_windows_metric = [w.copy() for w in baseline_pred_windows_metric]
        if save_plots_enabled:
            flow_pred_windows = [w.copy() for w in baseline_pred_windows]
    else:
        refiner = _build_refiner(
            args,
            device,
            target_dim=int(ds.target_dim),
            train_window_count=int(refiner_train_window_count),
            val_window_count=int(refiner_val_window_count),
        )
        # Hard guarantee: each (model, dataset, pred_len) run starts with a fresh refiner state.
        if hasattr(refiner, "reset_state"):
            try:
                refiner.reset_state(clear_loss_history=True)
            except TypeError:
                refiner.reset_state()
        predictor_raw = OnlineRefinerPredictor(
            base_predictor=base_predictor,
            refiner=refiner,
            device=device,
            context_length=args.context_length,
            update_input_dataset=None,
            update_stride=1,
            dense_to_baseline_ratio=None,
            predict_batch_size=int(args.batch_size),
            learning_batch_list=_resolve_learning_batch_keys(args),
            static_mean_scale=static_mean_scale,
            buffered_update_records=(dense_records_full if use_dense_cache_pipeline else None),
            speed_mode=bool(getattr(args, "speed", False)),
        )
        predictor = _RecordingPredictor(
            predictor_raw,
            prediction_length=int(ds.prediction_length),
            target_dim=int(ds.target_dim),
        )
        for _ in predictor.predict(flow_update_test_data.input, batch_size=args.batch_size):
            pass

        if bool(getattr(args, "speed", False)):
            speed_stats = dict(getattr(predictor_raw, "get_speed_stats", lambda: {})())
            speed_stats["base_model_infer_time"] = float(base_model_single_infer_time)
            if refiner_key == "bay" and "refiner" in locals():
                last_gate = getattr(refiner, "last_gate_confidence", None)
                last_time = getattr(refiner, "last_gate_time_index", None)
                if last_gate is not None and last_time is not None:
                    gate_path = _save_bay_gate_confidence_csv(
                        model_name=str(getattr(args, "model", "unknown")),
                        dataset_name=str(dataset_name),
                        pred_len=int(ds.prediction_length),
                        time_index=int(last_time),
                        gate_confidence=last_gate,
                    )
                    print(f"[Refined-CSV] Bay gate confidence saved: {gate_path}")

            refiner_tag = _resolve_refiner_tag(refiner_key)
            print(
                f"[Refined-CSV] Speed benchmark finished: infer_steps={int(getattr(predictor_raw, '_speed_infer_steps_recorded', 0))} | "
                f"train_calls={int(getattr(predictor_raw, 'refiner_update_calls', 0))}"
            )
            return {
                "dataset_name": dataset_name,
                "model_short_name": str(getattr(args, "model", "unknown")),
                "refiner_tag": refiner_tag,
                "pred_len": int(ds.prediction_length),
                "agg_metrics_base": {},
                "agg_metrics_flow": {},
                "window_count": 0,
                "flow_steps": 0,
                "eval_window_count": 0,
                "meta_window_count": 0,
                "update_window_count": 0,
                "train_meta_window_count": 0,
                "val_meta_window_count": 0,
                "test_meta_window_count": 0,
                "train_update_window_count": 0,
                "val_update_window_count": 0,
                "test_update_window_count": 0,
                "loss_history": [],
                "val_loss_history": [],
                "speed_stats": speed_stats,
            }

        flow_recorded_windows_all = list(getattr(predictor, "recorded_pred_windows", []))
        flow_meta_all = list(getattr(predictor, "point_meta_windows", []))
        flow_recorded_windows = flow_recorded_windows_all[int(test_start):int(test_end)]
        flow_meta_windows = flow_meta_all[int(test_start):int(test_end)]
        _, flow_pred_windows_metric = _align_gt_pred_windows(
            baseline_gt_windows_metric,
            flow_recorded_windows,
        )
        _audit_window_alignment(
            baseline_gt_windows_metric,
            flow_pred_windows_metric,
            flow_meta_windows,
        )
        flow_window_audit_rows = _compose_window_audit_rows(
            baseline_gt_windows_metric,
            flow_pred_windows_metric,
            flow_meta_windows,
        )
        _ = baseline_window_audit_rows
        _ = flow_window_audit_rows
        flow_point_metrics = _compute_point_primary_metrics(
            baseline_gt_windows_metric,
            flow_pred_windows_metric,
        )

        if use_channelwise_norm:
            agg_metrics_base = _inject_scaled_primary_metrics_channelwise(
                base_point_metrics,
                gt_windows=baseline_gt_windows_metric,
                pred_windows=baseline_pred_windows_metric,
                channel_mean_abs_scale=train_channel_mean_scale,
            )
            agg_metrics_flow = _inject_scaled_primary_metrics_channelwise(
                flow_point_metrics,
                gt_windows=baseline_gt_windows_metric,
                pred_windows=flow_pred_windows_metric,
                channel_mean_abs_scale=train_channel_mean_scale,
            )
        else:
            agg_metrics_base = _inject_scaled_primary_metrics(
                base_point_metrics,
                mean_abs_scale=train_global_mean_scale,
            )
            agg_metrics_flow = _inject_scaled_primary_metrics(
                flow_point_metrics,
                mean_abs_scale=train_global_mean_scale,
            )
        final_window_count = int(predictor_raw.window_count)
        final_flow_steps = int(predictor_raw.flow_steps)
        loss_history = list(getattr(refiner, "loss_history", [])) if hasattr(refiner, "loss_history") else []
        val_loss_history = list(getattr(refiner, "val_loss_history", [])) if hasattr(refiner, "val_loss_history") else []
        if save_plots_enabled:
            flow_pred_windows = [w.copy() for w in flow_pred_windows_metric]
        speed_stats = dict(getattr(predictor_raw, "get_speed_stats", lambda: {})())
    if "predictor_raw" not in locals():
        speed_stats = {
            "base_model_infer_time": float(base_model_single_infer_time),
            "infer_time": float("nan"),
            "infer_gpu": float("nan"),
            "infer_flops": float("nan"),
            "train_time": float("nan"),
            "train_gpu": float("nan"),
            "train_flops": float("nan"),
        }
    else:
        speed_stats["base_model_infer_time"] = float(base_model_single_infer_time)

    agg_metrics_base = dict(agg_metrics_base or {})
    agg_metrics_flow = dict(agg_metrics_flow or {})
    agg_metrics_base = _project_core_metric_keys(agg_metrics_base)
    agg_metrics_flow = _project_core_metric_keys(agg_metrics_flow)
    if agg_metrics_base is not None:
        print(
            f"[Refined-CSV][Baseline] {PRIMARY_METRIC_LABEL_1}={first_value(agg_metrics_base.get(PRIMARY_METRIC_KEY_1)):.4f} | "
            f"{PRIMARY_METRIC_LABEL_2}={first_value(agg_metrics_base.get(PRIMARY_METRIC_KEY_2)):.4f}"
        )
    if agg_metrics_flow is not None:
        print(
            f"[Refined-CSV][Refined] {PRIMARY_METRIC_LABEL_1}={first_value(agg_metrics_flow.get(PRIMARY_METRIC_KEY_1)):.4f} | "
            f"{PRIMARY_METRIC_LABEL_2}={first_value(agg_metrics_flow.get(PRIMARY_METRIC_KEY_2)):.4f}"
        )
    if 'refiner' in locals() and hasattr(refiner, "consume_unit_diagnostics_once"):
        try:
            diag_line = refiner.consume_unit_diagnostics_once()
            if diag_line:
                print(diag_line)
        except Exception:
            pass

    total_update_window_count = int(final_window_count)
    total_flow_steps = int(final_flow_steps)
    total_eval_window_count = int(len(baseline_eval_data.input))
    collected_losses: list = list(loss_history)
    collected_val_losses: list = list(val_loss_history)

    dataset_name = Path(args.csv_path).stem
    if agg_metrics_base:
        print(
            f"[Refined-CSV][Baseline][All Channels Avg] {PRIMARY_METRIC_LABEL_1}={first_value(agg_metrics_base.get(PRIMARY_METRIC_KEY_1)):.4f} | "
            f"{PRIMARY_METRIC_LABEL_2}={first_value(agg_metrics_base.get(PRIMARY_METRIC_KEY_2)):.4f}"
        )
    if agg_metrics_flow:
        print(
            f"[Refined-CSV][Refined][All Channels Avg] {PRIMARY_METRIC_LABEL_1}={first_value(agg_metrics_flow.get(PRIMARY_METRIC_KEY_1)):.4f} | "
            f"{PRIMARY_METRIC_LABEL_2}={first_value(agg_metrics_flow.get(PRIMARY_METRIC_KEY_2)):.4f}"
        )

    if refiner_key == "bay" and "refiner" in locals():
        last_gate = getattr(refiner, "last_gate_confidence", None)
        last_time = getattr(refiner, "last_gate_time_index", None)
        if last_gate is not None and last_time is not None:
            gate_path = _save_bay_gate_confidence_csv(
                model_name=str(getattr(args, "model", "unknown")),
                dataset_name=str(dataset_name),
                pred_len=int(ds.prediction_length),
                time_index=int(last_time),
                gate_confidence=last_gate,
            )
            print(f"[Refined-CSV] Bay gate confidence saved: {gate_path}")

    refiner_tag = _resolve_refiner_tag(_resolve_refiner(args))

    if save_plots_enabled:
        variant_suffix = str(getattr(args, "refiner_variant_suffix", "") or "")
        global_plot_path, last_plot_path = save_comparison_plots(
            dataset_name=dataset_name,
            baseline_gt_windows=baseline_gt_windows,
            baseline_pred_windows=baseline_pred_windows,
            flow_pred_windows=flow_pred_windows,
            refiner=refiner_tag,
            pred_len=int(ds.prediction_length),
            variant_suffix=variant_suffix,
        )
        print(f"[Refined-CSV] Global comparison plot: {global_plot_path}")
        print(f"[Refined-CSV] Last-window comparison plot: {last_plot_path}")
        print(f"[Refined-CSV] Plotted channel (fixed last dim): index=-1, name={last_channel}")
    print(
        f"[Refined-CSV] Final counts: eval_windows={total_eval_window_count} | "
        f"update_windows={total_update_window_count} | flow_steps={total_flow_steps}"
    )

    return {
        "dataset_name": dataset_name,
        "model_short_name": str(getattr(args, "model", "unknown")),
        "refiner_tag": refiner_tag,
        "pred_len": int(ds.prediction_length),
        "agg_metrics_base": agg_metrics_base,
        "agg_metrics_flow": agg_metrics_flow,
        "window_count": int(total_update_window_count),
        "flow_steps": int(total_flow_steps),
        "eval_window_count": int(total_eval_window_count),
        "meta_window_count": int(total_baseline_windows),
        "update_window_count": int(total_update_windows),
        "train_meta_window_count": int(train_meta_window_count),
        "val_meta_window_count": int(val_meta_window_count),
        "test_meta_window_count": int(test_meta_window_count),
        "train_update_window_count": int(eval_train_window_count),
        "val_update_window_count": int(eval_val_window_count),
        "test_update_window_count": int(eval_test_window_count),
        "loss_history": collected_losses,
        "val_loss_history": collected_val_losses,
        "speed_stats": speed_stats,
    }
