from __future__ import annotations
from typing import Dict
import numpy as np
from gluonts.ev.metrics import MAE, MSE
from gluonts.model.forecast import QuantileForecast, SampleForecast
from core.util.refiner_util import parse_quantile_key, select_quantile_index
from eval.eval_util import first_value


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
