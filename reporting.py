from __future__ import annotations
from cli import resolve_refiner_tag
import csv
import math
from pathlib import Path
from typing import Dict, List
from core.util.refiner_util import build_time_id, save_refiner_loss_history_json
from eval.eval_util import append_refiner_update_log, build_split_summary_csv_paths, first_value, load_existing_split_summary_records, parse_split_summary_metric_csv, write_split_summary_csv_map
from model_registry import TSFM_MODEL_ORDER
from cli import _compose_output_suffix, _normalize_pred_len_values


SUMMARY_METRIC_COLUMNS: List[str] = ["MAE", "MSE"]


PRIMARY_METRIC_LABEL_1 = "MAE"


PRIMARY_METRIC_LABEL_2 = "MSE"


PRIMARY_METRIC_KEY_1 = "MAE[mean]"


PRIMARY_METRIC_KEY_2 = "MSE[mean]"


CORE_METRIC_KEYS: List[str] = [
    "MAE[mean]",
    "MSE[mean]",
    "MAE_raw[mean]",
    "MSE_raw[mean]",
]


def _safe_speed_token(value: str) -> str:
    return str(value).strip().replace("/", "_").replace(" ", "_")


def _write_speed_table(
    *,
    dataset_name: str,
    model_name: str,
    pred_len: int,
    refiner_tags: list[str],
    stats_by_refiner: dict[str, dict],
) -> Path:
    out_dir = Path("results/speed")
    out_dir.mkdir(parents=True, exist_ok=True)
    safe_model = _safe_speed_token(model_name)
    safe_dataset = _safe_speed_token(dataset_name)
    file_name = f"speed_pred{int(pred_len)}_{safe_model}_{safe_dataset}.csv"
    out_path = out_dir / file_name

    # display names: map 'orca' -> 'ORCA' (case-insensitive), keep others
    display_tags: list[str] = []
    for t in list(refiner_tags):
        if str(t).strip().lower() == "orca":
            display_tags.append("ORCA")
        else:
            display_tags.append(str(t))
    header = ["Adapter"] + display_tags
    rows = []
    row_labels = [
        "Base Model Single Inference (ms)",
        "Single Inference Time (ms)",
        "Single Inference FLOPs",
        "Inference GPU Usage (MB)",
        "Single Training Time (ms)",
        "Single Training FLOPs",
        "Training GPU Usage (MB)",
    ]
    metric_keys = (
        "base_model_infer_time",
        "infer_time",
        "infer_flops",
        "infer_gpu",
        "train_time",
        "train_flops",
        "train_gpu",
    )
    for metric_key, row_label in zip(metric_keys, row_labels):
        row = [row_label]
        for tag in refiner_tags:
            stats = stats_by_refiner.get(tag, {})
            is_orca = str(tag).strip().lower() == "orca"
            if metric_key.startswith("train_") and not is_orca:
                row.append("")
                continue

            val = stats.get(metric_key, float("nan"))
            if metric_key in {"base_model_infer_time", "infer_time", "train_time"}:
                try:
                    v = float(val)
                    if math.isfinite(v):
                        v = v * 1000.0
                except Exception:
                    v = float("nan")
                row.append(v)
            else:
                row.append(val)
        rows.append(row)

    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)
    return out_path


def _metric_dict_with_nan() -> Dict[str, float]:
    return {k: float("nan") for k in CORE_METRIC_KEYS}


def _strict_mean_or_nan(values: list[float]) -> float:
    if not values:
        return float("nan")
    parsed = [float(v) for v in values]
    if any(not math.isfinite(v) for v in parsed):
        return float("nan")
    return float(sum(parsed) / len(parsed))


def _average_metric_dicts_strict(metric_dicts: list[Dict | None]) -> Dict[str, float]:
    keys: set[str] = set(CORE_METRIC_KEYS)
    for m in metric_dicts:
        if isinstance(m, dict):
            keys.update(map(str, m.keys()))

    out: Dict[str, float] = {}
    for key in sorted(keys):
        vals = [_metric_or_nan(m, key) for m in metric_dicts]
        out[key] = _strict_mean_or_nan(vals)
    return out


def _build_failed_eval_record(
    *,
    dataset_label: str,
    model_short_name: str,
    refiner: str,
    refiner_tag: str,
    variant_suffix: str,
    training_method,
    refiner_input,
    update_rule,
    online_buffer_windows,
    router,
    routing_temperature,
    ema_error_momentum,
    pred_len: int,
) -> dict:
    nan_metrics = _metric_dict_with_nan()
    return {
        "dataset_name": dataset_label,
        "dataset_label": dataset_label,
        "dataset_result_name": dataset_label,
        "model_short_name": model_short_name,
        "refiner": refiner,
        "refiner_tag": refiner_tag,
        "variant_suffix": variant_suffix,
        "training_method": training_method,
        "refiner_input": refiner_input,
        "update_rule": update_rule,
        "online_buffer_windows": online_buffer_windows,
        "router": router,
        "routing_temperature": routing_temperature,
        "ema_error_momentum": ema_error_momentum,
        "pred_len": int(pred_len),
        "agg_metrics_base": dict(nan_metrics),
        "agg_metrics_refined": dict(nan_metrics),
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


def _build_pred_len_average_records(records: list[dict], pred_len_values: list[int]) -> list[dict]:
    pred_set = {int(v) for v in pred_len_values}
    grouped: dict[tuple, dict[int, dict]] = {}
    for rec in records:
        key = (
            str(rec.get("dataset_label")),
            str(rec.get("model_short_name")),
            str(rec.get("refiner")),
            str(rec.get("refiner_tag")),
            str(rec.get("variant_suffix", "")),
            str(rec.get("training_method")),
            str(rec.get("refiner_input")),
            str(rec.get("update_rule")),
        )
        grouped.setdefault(key, {})[int(rec.get("pred_len", 0))] = rec

    averaged: list[dict] = []
    for key, rec_map in grouped.items():
        metric_base_list: list[Dict | None] = []
        metric_refined_list: list[Dict | None] = []
        for pred_len in sorted(pred_set):
            rec = rec_map.get(pred_len)
            if rec is None:
                metric_base_list.append(_metric_dict_with_nan())
                metric_refined_list.append(_metric_dict_with_nan())
            else:
                metric_base_list.append(rec.get("agg_metrics_base"))
                metric_refined_list.append(rec.get("agg_metrics_refined"))

        avg_base = _average_metric_dicts_strict(metric_base_list)
        avg_refined = _average_metric_dicts_strict(metric_refined_list)

        base_rec = next(iter(rec_map.values()))
        merged = dict(base_rec)
        merged["pred_len"] = None
        merged["agg_metrics_base"] = avg_base
        merged["agg_metrics_refined"] = avg_refined
        averaged.append(merged)

    return averaged


def _merge_orders_preserve_existing(existing: list[str], current: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for seq in (existing, current):
        for raw in seq:
            key = str(raw)
            if key in seen:
                continue
            seen.add(key)
            out.append(key)
    return out


def _build_result_csv_path(
    dataset_name: str,
    refiner: str,
    model_short_name: str,
    *,
    suffix: str = "",
    context_length: int | None = None,
    pred_len: int | None = None,
    pred_len_avg: bool = False,
) -> Path:
    refiner_tag = resolve_refiner_tag(refiner)
    safe_model = str(model_short_name).replace("-", "_")
    suffix_str = _compose_output_suffix(
        suffix=suffix,
        context_length=context_length,
        pred_len=pred_len,
        pred_len_avg=pred_len_avg,
    )
    return Path("results/details/single_dataset") / str(dataset_name) / f"results_csv_{dataset_name}_{safe_model}_{refiner_tag}{suffix_str}.csv"


def _resume_load_existing_summary_records(
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
) -> tuple[list[dict], set[tuple[str, str]], list[str], list[str]]:
    return load_existing_split_summary_records(
        mae_csv_path=mae_csv_path,
        mse_csv_path=mse_csv_path,
        pred_len=int(pred_len),
        refiner=refiner,
        refiner_tag=refiner_tag,
        variant_suffix=variant_suffix,
        training_method=training_method,
        refiner_input=refiner_input,
        update_rule=update_rule,
        online_buffer_windows=online_buffer_windows,
        mae_metric_key=PRIMARY_METRIC_KEY_1,
        mse_metric_key=PRIMARY_METRIC_KEY_2,
    )


def _build_all_result_csv_paths(
    refiner: str,
    *,
    suffix: str = "",
    context_length: int | None = None,
    pred_len: int | None = None,
    pred_len_avg: bool = False,
) -> dict[str, Path]:
    refiner_tag = resolve_refiner_tag(refiner)
    return build_split_summary_csv_paths(
        refiner_tag=refiner_tag,
        suffix=suffix,
        context_length=context_length,
        pred_len=pred_len,
        pred_len_avg=pred_len_avg,
    )


def _to_float_or_nan(value) -> float:
    try:
        return float(value)
    except Exception:
        return float("nan")


def _metric_or_nan(metrics_dict: Dict | None, key: str) -> float:
    if not isinstance(metrics_dict, dict):
        return float("nan")
    return _to_float_or_nan(first_value(metrics_dict.get(key)))


def _write_single_dataset_csv(
    csv_path: Path,
    dataset_name: str,
    agg_metrics_base: Dict | None,
    agg_metrics_refined: Dict | None,
    *,
    pred_len: int | None = None,
) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    header = [
        "dataset",
        "pred_len",
        "model",
        "MAE",
        "MSE",
        "MAE_raw",
        "MSE_raw",
    ]
    original_keys = [
        "MAE[mean]",
        "MSE[mean]",
        "MAE_raw[mean]",
        "MSE_raw[mean]",
    ]

    write_header = not csv_path.exists()
    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(header)

        def _write_one_row(model_name: str, metrics_dict: Dict | None) -> None:
            source = metrics_dict if isinstance(metrics_dict, dict) else _metric_dict_with_nan()
            row = [dataset_name, (int(pred_len) if pred_len is not None else "avg"), model_name] + [first_value(source.get(key)) for key in original_keys]
            writer.writerow(row)

        _write_one_row("baseline", agg_metrics_base)
        _write_one_row("refined", agg_metrics_refined)


def _refresh_refiner_summary_csv(
    *,
    refiner_records: list[dict],
    refiner_value: str,
    dataset_order: list[str],
    model_order: list[str],
    context_length: int | None,
    pred_len: int | None,
    pred_len_avg: bool = False,
) -> dict[str, Path]:
    variant_suffix = str(refiner_records[0].get("variant_suffix", "")) if refiner_records else ""
    out_paths = _build_all_result_csv_paths(
        refiner_value,
        suffix=variant_suffix,
        context_length=context_length,
        pred_len=pred_len,
        pred_len_avg=pred_len_avg,
    )

    # Always merge with current on-disk content before rewrite to avoid losing already-finished units.
    loaded_records, _, existing_ds_order, existing_model_order = _resume_load_existing_summary_records(
        mae_csv_path=out_paths["mae"],
        mse_csv_path=out_paths["mse"],
        pred_len=int(pred_len) if pred_len is not None else -1,
        refiner=refiner_value,
        refiner_tag=resolve_refiner_tag(refiner_value),
        variant_suffix=variant_suffix,
        training_method=None,
        refiner_input=None,
        update_rule=None,
        online_buffer_windows=None,
    )

    by_key: dict[tuple[str, str], dict] = {}
    for rec in loaded_records:
        k = (str(rec.get("dataset_label")), str(rec.get("model_short_name")))
        by_key[k] = rec
    for rec in refiner_records:
        k = (str(rec.get("dataset_label")), str(rec.get("model_short_name")))
        by_key[k] = rec
    merged_records = list(by_key.values())

    # Keep existing ordering from current run preferences first, then append unseen keys from records.
    merged_dataset_order: list[str] = []
    seen_ds: set[str] = set()
    for ds in list(existing_ds_order) + list(dataset_order):
        d = str(ds)
        if d in seen_ds:
            continue
        seen_ds.add(d)
        merged_dataset_order.append(d)
    for rec in merged_records:
        d = str(rec.get("dataset_label"))
        if d and d not in seen_ds:
            seen_ds.add(d)
            merged_dataset_order.append(d)

    merged_model_order: list[str] = []
    seen_models: set[str] = set()
    for m in list(existing_model_order) + list(model_order):
        mm = str(m)
        if mm in seen_models:
            continue
        seen_models.add(mm)
        merged_model_order.append(mm)
    for rec in merged_records:
        mm = str(rec.get("model_short_name"))
        if mm and mm not in seen_models:
            seen_models.add(mm)
            merged_model_order.append(mm)

    write_split_summary_csv_map(
        csv_path_by_name={
            "mae": out_paths["mae"],
            "mse": out_paths["mse"],
        },
        dataset_order=merged_dataset_order,
        model_order=merged_model_order,
        records=merged_records,
        metric_key_by_name={
            "mae": PRIMARY_METRIC_KEY_1,
            "mse": PRIMARY_METRIC_KEY_2,
        },
    )
    return out_paths


def _persist_single_record(rec: dict, args) -> None:
    dataset_result_name = rec["dataset_result_name"]
    model_short_name = rec["model_short_name"]

    variant_suffix = str(rec.get("variant_suffix", ""))
    single_csv_path = _build_result_csv_path(
        dataset_result_name,
        rec["refiner"],
        model_short_name,
        suffix=variant_suffix,
        context_length=getattr(args, "context_length", None),
        pred_len=rec.get("pred_len"),
    )
    _write_single_dataset_csv(
        csv_path=single_csv_path,
        dataset_name=dataset_result_name,
        agg_metrics_base=rec.get("agg_metrics_base"),
        agg_metrics_refined=rec.get("agg_metrics_refined"),
        pred_len=rec.get("pred_len"),
    )

    logs_dir = Path("results/details/logs")
    append_refiner_update_log(
        logs_dir,
        dataset_name=dataset_result_name,
        ds_config=f"csv/{dataset_result_name}",
        model_name=model_short_name,
        args=args,
        window_count=int(rec.get("window_count", 0)),
        update_steps=int(rec.get("update_steps", 0)),
        loss_history=list(rec.get("loss_history", [])),
        eval_window_count=rec.get("eval_window_count"),
        meta_window_count=rec.get("meta_window_count"),
        update_window_count=rec.get("update_window_count"),
        train_meta_window_count=rec.get("train_meta_window_count"),
        val_meta_window_count=rec.get("val_meta_window_count"),
        test_meta_window_count=rec.get("test_meta_window_count"),
        train_update_window_count=rec.get("train_update_window_count"),
        val_update_window_count=rec.get("val_update_window_count"),
        test_update_window_count=rec.get("test_update_window_count"),
    )

    if str(rec.get("refiner_tag", "")).lower() in {"orca_no_decay", "orca"}:
        run_time_id = build_time_id()
        train_loss_history = list(rec.get("loss_history", []))
        val_loss_history = list(rec.get("val_loss_history", []))
        training_method = str(rec.get("training_method") or getattr(args, "training_method", "online")).strip().lower()
        base_logs_dir = Path("results/details/logs")
        if training_method == "online":
            suffix_tag = str(rec.get("variant_suffix", "default"))
            logs_dir = base_logs_dir / "online_training_runs" / f"{dataset_result_name}_{model_short_name}_{rec.get('refiner_tag')}_{suffix_tag}"
        else:
            logs_dir = base_logs_dir
        model_config = {
            "batch_size": getattr(args, "batch_size", None),
            "context_length": getattr(args, "context_length", None),
            "pred_len": getattr(args, "pred_len", None),
            "training_method": rec.get("training_method"),
            "refiner_input": rec.get("refiner_input"),
            "update_rule": rec.get("update_rule"),
            "router": rec.get("router", getattr(args, "router", None)),
            "routing_temperature": rec.get("routing_temperature", getattr(args, "routing_temperature", None)),
            "ema_error_momentum": rec.get("ema_error_momentum", getattr(args, "ema_error_momentum", None)),
            "online_buffer_windows": rec.get("online_buffer_windows", getattr(args, "online_buffer_windows", None)),
            "force_gate_open": bool(getattr(args, "force_gate_open", False)),
            "channel_mix": bool(getattr(args, "channel_mix", True)),
            "train_batch_size": int(getattr(args, "train_batch_size", 256)),
        }
        json_path = save_refiner_loss_history_json(
            logs_dir,
            dataset_name=dataset_result_name,
            model_name=model_short_name,
            refiner_name=f"{str(rec.get('refiner_tag', 'ORCA_NoDecay'))}_{variant_suffix}",
            loss_history=train_loss_history,
            val_loss_history=val_loss_history,
            command_line=getattr(args, "_command_line", None),
            model_config=model_config,
            time_id=run_time_id,
        )
        print(f"[ORCA] Refiner loss json saved: {json_path}")


def _persist_results_at_end(records: list[dict], args) -> None:
    if not records:
        print("[ORCA] No result records to save.")
        return

    pred_len_values = _normalize_pred_len_values(getattr(args, "pred_len_values", [getattr(args, "pred_len", 96)]))
    refiner_groups: dict[tuple[str, str], list[dict]] = {}
    for rec in records:
        group_key = (str(rec.get("refiner_tag")), str(rec.get("variant_suffix", "")))
        refiner_groups.setdefault(group_key, []).append(rec)

    for (_, _), recs in refiner_groups.items():
        model_order: list[str] = []
        seen_models: set[str] = set()
        for m in TSFM_MODEL_ORDER:
            if any(str(r.get("model_short_name")) == m for r in recs):
                if m not in seen_models:
                    seen_models.add(m)
                    model_order.append(m)
        for rec in recs:
            m = str(rec.get("model_short_name"))
            if m and m not in seen_models:
                seen_models.add(m)
                model_order.append(m)

        dataset_order: list[str] = []
        seen_ds: set[str] = set()
        for rec in recs:
            ds = str(rec.get("dataset_label"))
            if ds and ds not in seen_ds:
                seen_ds.add(ds)
                dataset_order.append(ds)
        variant_suffix = str(recs[0].get("variant_suffix", ""))
        refiner = str(recs[0].get("refiner"))

        by_pred_len: dict[int, list[dict]] = {}
        for rec in recs:
            p = int(rec.get("pred_len", 0) or 0)
            by_pred_len.setdefault(p, []).append(rec)

        for pred_len in pred_len_values:
            pred_records = by_pred_len.get(int(pred_len), [])
            out_paths = _build_all_result_csv_paths(
                refiner,
                suffix=variant_suffix,
                context_length=getattr(args, "context_length", None),
                pred_len=int(pred_len),
            )
            if bool(getattr(args, "resume_eval", False)):
                existing_order_ds: list[str] = []
                existing_order_models: list[str] = []
                if out_paths["mae"].exists():
                    _, existing_order_ds, existing_order_models = parse_split_summary_metric_csv(out_paths["mae"])
                elif out_paths["mse"].exists():
                    _, existing_order_ds, existing_order_models = parse_split_summary_metric_csv(out_paths["mse"])
                dataset_order_for_write = _merge_orders_preserve_existing(existing_order_ds, dataset_order)
                model_order_for_write = _merge_orders_preserve_existing(existing_order_models, model_order)
            else:
                dataset_order_for_write = list(dataset_order)
                model_order_for_write = list(model_order)
            write_split_summary_csv_map(
                csv_path_by_name={
                    "mae": out_paths["mae"],
                    "mse": out_paths["mse"],
                },
                dataset_order=dataset_order_for_write,
                model_order=model_order_for_write,
                records=pred_records,
                metric_key_by_name={
                    "mae": PRIMARY_METRIC_KEY_1,
                    "mse": PRIMARY_METRIC_KEY_2,
                },
            )
            print(f"[ORCA] Summary MAE written: {out_paths['mae']}")
            print(f"[ORCA] Summary MSE written: {out_paths['mse']}")

        if len(pred_len_values) > 1:
            avg_records = _build_pred_len_average_records(recs, pred_len_values)
            avg_out_paths = _build_all_result_csv_paths(
                refiner,
                suffix=variant_suffix,
                context_length=getattr(args, "context_length", None),
                pred_len_avg=True,
            )
            if bool(getattr(args, "resume_eval", False)):
                existing_order_ds: list[str] = []
                existing_order_models: list[str] = []
                if avg_out_paths["mae"].exists():
                    _, existing_order_ds, existing_order_models = parse_split_summary_metric_csv(avg_out_paths["mae"])
                elif avg_out_paths["mse"].exists():
                    _, existing_order_ds, existing_order_models = parse_split_summary_metric_csv(avg_out_paths["mse"])
                avg_dataset_order_for_write = _merge_orders_preserve_existing(existing_order_ds, dataset_order)
                avg_model_order_for_write = _merge_orders_preserve_existing(existing_order_models, model_order)
            else:
                avg_dataset_order_for_write = list(dataset_order)
                avg_model_order_for_write = list(model_order)
            write_split_summary_csv_map(
                csv_path_by_name={
                    "mae": avg_out_paths["mae"],
                    "mse": avg_out_paths["mse"],
                },
                dataset_order=avg_dataset_order_for_write,
                model_order=avg_model_order_for_write,
                records=avg_records,
                metric_key_by_name={
                    "mae": PRIMARY_METRIC_KEY_1,
                    "mse": PRIMARY_METRIC_KEY_2,
                },
            )
            print(f"[ORCA] Pred-len average MAE summary written: {avg_out_paths['mae']}")
            print(f"[ORCA] Pred-len average MSE summary written: {avg_out_paths['mse']}")
