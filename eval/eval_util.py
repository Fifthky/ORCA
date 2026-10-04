from __future__ import annotations

import csv
import math
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from core.util.refiner_util import extract_loss_history_values


def format_duration_dhms(seconds: float | int) -> str:
    try:
        sec = int(max(0, round(float(seconds))))
    except Exception:
        sec = 0
    days, rem = divmod(sec, 24 * 3600)
    hours, rem = divmod(rem, 3600)
    minutes, secs = divmod(rem, 60)
    return f"{days}d {hours:02d}h {minutes:02d}m {secs:02d}s"


def build_progress_line(
    *,
    prefix: str,
    done: int,
    total: int,
    elapsed_seconds: float,
    unit: str = "it",
) -> str:
    done_i = int(max(0, done))
    total_i = int(max(1, total))
    elapsed = max(1e-9, float(elapsed_seconds))
    pct = 100.0 * float(done_i) / float(total_i)
    rate = float(done_i) / elapsed
    remain = max(0, total_i - done_i)
    eta = float(remain) / max(1e-9, rate)
    return (
        f"{prefix}: {done_i}/{total_i} ({pct:.1f}%) | "
        f"elapsed={format_duration_dhms(elapsed)} | "
        f"eta={format_duration_dhms(eta)} | {rate:.2f}{unit}/s"
    )


def append_refiner_update_log(
    log_dir: Path,
    *,
    dataset_name: str,
    ds_config: str,
    model_name: str,
    args,
    window_count: int,
    update_steps: int,
    loss_history: list[float],
    eval_window_count: int | None = None,
    meta_window_count: int | None = None,
    update_window_count: int | None = None,
    train_meta_window_count: int | None = None,
    val_meta_window_count: int | None = None,
    test_meta_window_count: int | None = None,
    train_update_window_count: int | None = None,
    val_update_window_count: int | None = None,
    test_update_window_count: int | None = None,
) -> None:
    """Append one dataset-level refiner update log line."""
    import json

    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "refiner_updates.json"
    timestamp = datetime.now().isoformat(timespec="seconds")
    all_losses = extract_loss_history_values(loss_history)
    num_updates = int(len(all_losses))
    loss_mean = float(np.mean(all_losses)) if all_losses else float("nan")
    loss_last = float(all_losses[-1]) if all_losses else float("nan")
    log_obj = {
        "timestamp": timestamp,
        "dataset_name": dataset_name,
        "ds_config": ds_config,
        "model_name": model_name,
        "context_length": getattr(args, "context_length", None),
        "batch_size": getattr(args, "batch_size", None),
        "window_count": int(window_count),
        "eval_window_count": (int(eval_window_count) if eval_window_count is not None else None),
        "meta_window_count": (int(meta_window_count) if meta_window_count is not None else None),
        "update_window_count": (
            int(update_window_count)
            if update_window_count is not None
            else int(window_count)
        ),
        "train_meta_window_count": (int(train_meta_window_count) if train_meta_window_count is not None else None),
        "val_meta_window_count": (int(val_meta_window_count) if val_meta_window_count is not None else None),
        "test_meta_window_count": (int(test_meta_window_count) if test_meta_window_count is not None else None),
        "train_update_window_count": (int(train_update_window_count) if train_update_window_count is not None else None),
        "val_update_window_count": (int(val_update_window_count) if val_update_window_count is not None else None),
        "test_update_window_count": (int(test_update_window_count) if test_update_window_count is not None else None),
        "update_steps": int(update_steps),
        "num_updates": num_updates,
        "loss_mean": loss_mean,
        "loss_last": loss_last,
        "loss_history": loss_history,
    }
    with open(log_path, "a") as f:
        f.write(json.dumps(log_obj, ensure_ascii=False) + "\n")


def first_value(v):
    if v is None:
        return float("nan")
    try:
        if isinstance(v, torch.Tensor):
            if int(v.numel()) <= 0:
                return float("nan")
            x = v.detach().float().reshape(-1)
            out = float(x[0].item())
            return out if np.isfinite(out) else float("nan")

        if isinstance(v, np.ndarray):
            if int(v.size) <= 0:
                return float("nan")
            out = float(np.asarray(v).reshape(-1)[0])
            return out if np.isfinite(out) else float("nan")

        if isinstance(v, (list, tuple)):
            if len(v) <= 0:
                return float("nan")
            return first_value(v[0])

        out = float(v)
        return out if np.isfinite(out) else float("nan")
    except Exception:
        return float("nan")


def compose_output_suffix(
    *,
    suffix: str = "",
    context_length: int | None = None,
    pred_len: int | None = None,
    pred_len_avg: bool = False,
) -> str:
    parts: list[str] = []
    if str(suffix).strip():
        parts.append(str(suffix).strip())
    if pred_len_avg:
        parts.append("pred_avg")
    elif pred_len is not None:
        parts.append(f"pred{int(pred_len)}")
    return f"_{'_'.join(parts)}" if parts else ""


def build_split_summary_csv_paths(
    *,
    refiner_tag: str,
    suffix: str = "",
    context_length: int | None = None,
    pred_len: int | None = None,
    pred_len_avg: bool = False,
) -> dict[str, Path]:
    suffix_str = compose_output_suffix(
        suffix=suffix,
        context_length=context_length,
        pred_len=pred_len,
        pred_len_avg=pred_len_avg,
    )
    file_stem = f"results_csv_{str(refiner_tag)}{suffix_str}_all_csv_dataset"
    return {
        "mae": Path("results/MAE_summary") / f"{file_stem}_mae.csv",
        "mse": Path("results/MSE_summary") / f"{file_stem}_mse.csv",
    }


def _summary_formal_model_name(model_short_name: str) -> str:
    mapping = {
        "moirai-2": "Moirai-2",
        "chronos-2": "Chronos-2",
        "sundial": "Sundial",
        "tirex": "TiRex",
        "timesfm-2.5": "TimesFM-2.5",
        "moirai-1-small": "Moirai-1-Small",
        "moirai-1-base": "Moirai-1-Base",
        "moirai-1-large": "Moirai-1-Large",
    }
    return mapping.get(str(model_short_name), str(model_short_name))


def _summary_to_float_nan(raw: str) -> float:
    try:
        return float(raw)
    except Exception:
        return float("nan")


def _summary_format_metric(value: float) -> str:
    if not math.isfinite(float(value)):
        return "nan"
    return f"{float(value):.4f}"


def _summary_format_change(value: float) -> str:
    if not math.isfinite(float(value)):
        return "nan"
    return f"{float(value):.1f}%"


def summary_pct_change(refiner_val: float, baseline_val: float) -> float:
    if (not math.isfinite(float(baseline_val))) or float(baseline_val) == 0.0:
        return float("nan")
    return (float(refiner_val) - float(baseline_val)) / float(baseline_val) * 100.0


def _summary_mean_or_nan(values: list[float]) -> float:
    valid = [float(v) for v in values if math.isfinite(float(v))]
    if not valid:
        return float("nan")
    return float(sum(valid) / len(valid))


def parse_split_summary_metric_csv(
    csv_path: Path,
) -> tuple[dict[tuple[str, str], dict[str, float]], list[str], list[str]]:
    if not csv_path.exists():
        return {}, [], []

    try:
        with open(csv_path, "r", newline="") as f:
            rows = list(csv.reader(f))
    except Exception:
        return {}, [], []

    if len(rows) < 1:
        return {}, [], []

    header_model = rows[0]
    model_order: list[str] = []
    model_cols: list[tuple[int, str]] = []
    formal_to_short: dict[str, str] = {
        _summary_formal_model_name(m): str(m)
        for m in [
            "moirai-2",
            "chronos-2",
            "sundial",
            "tirex",
            "timesfm-2.5",
            "moirai-1-small",
            "moirai-1-base",
            "moirai-1-large",
        ]
    }

    max_col = max(0, len(header_model) - 1)
    for c in range(2, max_col, 2):
        model_name = str(header_model[c]).strip()
        if not model_name:
            continue
        model_short = formal_to_short.get(model_name, model_name)
        model_order.append(model_short)
        model_cols.append((c, model_short))

    metrics_by_key: dict[tuple[str, str], dict[str, float]] = {}
    dataset_order: list[str] = []
    i = 1
    while i + 1 < len(rows):
        row_v = rows[i]
        if str(row_v[0]).strip() == "Models Avg.":
            break
        row_r = rows[i + 1]
        if len(row_v) < 2 or len(row_r) < 2:
            i += 1
            continue
        if str(row_v[1]).strip().lower() != "vanilla" or str(row_r[1]).strip().lower() != "refined":
            i += 1
            continue

        ds = str(row_v[0]).strip()
        if not ds:
            i += 2
            continue
        dataset_order.append(ds)
        for c, model_short in model_cols:
            base_v = _summary_to_float_nan(row_v[c] if c < len(row_v) else "nan")
            refined_v = _summary_to_float_nan(row_r[c] if c < len(row_r) else "nan")
            metrics_by_key[(ds, model_short)] = {
                "base": float(base_v),
                "refined": float(refined_v),
            }
        i += 2

    # Keep first-seen order stable.
    dedup_ds: list[str] = []
    seen_ds: set[str] = set()
    for ds in dataset_order:
        if ds in seen_ds:
            continue
        seen_ds.add(ds)
        dedup_ds.append(ds)

    dedup_models: list[str] = []
    seen_models: set[str] = set()
    for m in model_order:
        if m in seen_models:
            continue
        seen_models.add(m)
        dedup_models.append(m)

    return metrics_by_key, dedup_ds, dedup_models


def load_existing_split_summary_records(
    *,
    mae_csv_path: Path,
    mse_csv_path: Path,
    pred_len: int,
    refiner: str,
    refiner_tag: str,
    variant_suffix: str,
    training_method,
    refiner_input,
    update_rule,
    online_buffer_windows,
    mae_metric_key: str,
    mse_metric_key: str,
) -> tuple[list[dict], set[tuple[str, str]], list[str], list[str]]:
    mae_map, mae_ds_order, mae_model_order = parse_split_summary_metric_csv(mae_csv_path)
    mse_map, mse_ds_order, mse_model_order = parse_split_summary_metric_csv(mse_csv_path)

    key_set = set(mae_map.keys()) | set(mse_map.keys())
    existing_records: list[dict] = []
    completed_keys: set[tuple[str, str]] = set()

    for ds, model in sorted(key_set):
        mae_pair = mae_map.get((ds, model), {})
        mse_pair = mse_map.get((ds, model), {})
        base_mae = float(mae_pair.get("base", float("nan")))
        refined_mae = float(mae_pair.get("refined", float("nan")))
        base_mse = float(mse_pair.get("base", float("nan")))
        refined_mse = float(mse_pair.get("refined", float("nan")))

        rec = {
            "dataset_name": ds,
            "dataset_label": ds,
            "dataset_result_name": ds,
            "model_short_name": model,
            "refiner": refiner,
            "refiner_tag": refiner_tag,
            "variant_suffix": variant_suffix,
            "training_method": training_method,
            "refiner_input": refiner_input,
            "update_rule": update_rule,
            "online_buffer_windows": online_buffer_windows,
            "pred_len": int(pred_len),
            "agg_metrics_base": {
                str(mae_metric_key): base_mae,
                str(mse_metric_key): base_mse,
            },
            "agg_metrics_refined": {
                str(mae_metric_key): refined_mae,
                str(mse_metric_key): refined_mse,
            },
            "window_count": 0,
            "update_steps": 0,
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
        }
        existing_records.append(rec)

        if math.isfinite(refined_mae) and math.isfinite(refined_mse):
            completed_keys.add((ds, model))

    # Keep existing order first, then fill missing keys from the other file.
    dataset_order: list[str] = []
    seen_ds: set[str] = set()
    for ds in list(mae_ds_order) + list(mse_ds_order):
        if ds in seen_ds:
            continue
        seen_ds.add(ds)
        dataset_order.append(ds)
    for ds, _ in sorted(key_set):
        if ds not in seen_ds:
            seen_ds.add(ds)
            dataset_order.append(ds)

    model_order: list[str] = []
    seen_model: set[str] = set()
    for m in list(mae_model_order) + list(mse_model_order):
        if m in seen_model:
            continue
        seen_model.add(m)
        model_order.append(m)
    for _, m in sorted(key_set):
        if m not in seen_model:
            seen_model.add(m)
            model_order.append(m)

    return existing_records, completed_keys, dataset_order, model_order


def write_split_summary_metric_csv(
    *,
    csv_path: Path,
    dataset_order: list[str],
    model_order: list[str],
    records: list[dict],
    metric_key: str,
) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    by_key: dict[tuple[str, str], dict] = {}
    for rec in records:
        ds = str(rec.get("dataset_label"))
        model = str(rec.get("model_short_name"))
        if ds and model:
            by_key[(ds, model)] = rec

    header_model = ["Model", ""]
    for model_short_name in model_order:
        header_model.extend([_summary_formal_model_name(model_short_name), ""])
    header_model.append("Datasets Avg.")

    model_change_values: dict[str, list[float]] = {m: [] for m in model_order}

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header_model)

        for dataset_name in dataset_order:
            row_vanilla = [dataset_name, "Vanilla"]
            row_refined = ["", "Refined"]
            ds_changes: list[float] = []

            for model_short_name in model_order:
                rec = by_key.get((dataset_name, model_short_name))
                if rec is None:
                    row_vanilla.extend(["nan", "nan"])
                    row_refined.extend(["nan", ""])
                    continue

                base_metric = first_value((rec.get("agg_metrics_base") or {}).get(metric_key))
                refined_metric = first_value((rec.get("agg_metrics_refined") or {}).get(metric_key))
                change_metric = summary_pct_change(refined_metric, base_metric)

                row_vanilla.extend([
                    _summary_format_metric(base_metric),
                    _summary_format_change(change_metric),
                ])
                row_refined.extend([
                    _summary_format_metric(refined_metric),
                    "",
                ])

                if math.isfinite(change_metric):
                    ds_changes.append(float(change_metric))
                    model_change_values.setdefault(model_short_name, []).append(float(change_metric))

            row_vanilla.append(_summary_format_change(_summary_mean_or_nan(ds_changes)))
            row_refined.append("")
            writer.writerow(row_vanilla)
            writer.writerow(row_refined)

        models_avg_row = ["Models Avg.", "Change"]
        model_means: list[float] = []
        for model_short_name in model_order:
            avg_change = _summary_mean_or_nan(model_change_values.get(model_short_name, []))
            models_avg_row.extend([
                _summary_format_change(avg_change),
                "",
            ])
            if math.isfinite(avg_change):
                model_means.append(float(avg_change))
        models_avg_row.append(_summary_format_change(_summary_mean_or_nan(model_means)))
        writer.writerow(models_avg_row)


def write_split_summary_csv_map(
    *,
    csv_path_by_name: dict[str, Path],
    dataset_order: list[str],
    model_order: list[str],
    records: list[dict],
    metric_key_by_name: dict[str, str],
) -> None:
    for metric_name, metric_key in metric_key_by_name.items():
        csv_path = csv_path_by_name.get(str(metric_name))
        if csv_path is None:
            continue
        write_split_summary_metric_csv(
            csv_path=csv_path,
            dataset_order=dataset_order,
            model_order=model_order,
            records=records,
            metric_key=str(metric_key),
        )
