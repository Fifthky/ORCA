from __future__ import annotations

import argparse
import csv
import gc
import math
import random
import shlex
import sys
import traceback
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

from data.csv_dataset import CSV_DATASET_SPECS
from data.download_CSV import DEFAULT_CACHE_DIR, ensure_dataset_csv, resolve_cached_csv_path
from core.util.refiner_util import (
    build_time_id,
    save_refiner_loss_history_json,
    save_refiner_loss_history_plot,
)
from eval.eval_util import (
    append_refiner_update_log,
    build_split_summary_csv_paths,
    first_value,
    load_existing_split_summary_records,
    parse_split_summary_metric_csv,
    write_split_summary_csv_map,
)
from eval.evaluator import run_geoflow_csv_evaluation
from model_registry import (
    BACKEND_COMPATIBLE_MODELS,
    TSFM_MODEL_ORDER,
    TSFM_MODEL_PATH_PREFIX,
    normalize_model_name,
    resolve_model_path,
)


ALL_CSV_DATASETS: List[str] = ["ETTh1", "ETTh2", "ETTm1", "ETTm2", "Exchange", "Weather", "Electricity", "Traffic"]
SUMMARY_METRIC_COLUMNS: List[str] = ["MAE", "MSE"]
PRIMARY_METRIC_LABEL_1 = "MAE"
PRIMARY_METRIC_LABEL_2 = "MSE"
PRIMARY_METRIC_KEY_1 = "MAE[mean]"
PRIMARY_METRIC_KEY_2 = "MSE[mean]"
REFINER_CHOICES: List[str] = [
    "Linear",
    "Bay",
    "Attn",
    "Bay_Attn",
    "AdaY",
    "DSOF",
    "TAFAS",
    "SOLID",
    "ELF",
    "Ridge",
    "ARIMA",
    "ETS",
    "linear",
    "bay",
    "attn",
    "bay_attn",
    "aday",
    "dsof",
    "tafas",
    "solid",
    "elf",
    "ridge",
    "arima",
    "ets",
]
CANONICAL_REFINERS: List[str] = ["Linear", "Bay", "Attn", "Bay_Attn", "AdaY", "DSOF", "TAFAS", "SOLID", "ELF", "Ridge", "ARIMA", "ETS"]
CORE_METRIC_KEYS: List[str] = [
    "MAE[mean]",
    "MSE[mean]",
    "MAE_raw[mean]",
    "MSE_raw[mean]",
]


def _normalize_pred_len_values(values) -> list[int]:
    out: list[int] = []
    seen: set[int] = set()
    for raw in list(values or []):
        v = int(raw)
        if v <= 0:
            raise ValueError(f"Unsupported pred_len {raw!r}. All pred_len values must be positive integers.")
        if v in seen:
            continue
        seen.add(v)
        out.append(v)
    if not out:
        out = [96]
    return out


def _normalize_online_buffer_values(values) -> list[int]:
    out: list[int] = []
    seen: set[int] = set()
    for raw in list(values or []):
        v = int(raw)
        if v <= 0:
            raise ValueError(
                f"Unsupported online buffer {raw!r}. All online buffer values must be positive integers."
            )
        if v in seen:
            continue
        seen.add(v)
        out.append(v)
    if not out:
        out = [3000]
    return out


def _normalize_float_values(values) -> list[float]:
    out: list[float] = []
    seen: set[float] = set()
    for raw in list(values or []):
        v = float(raw)
        if v in seen:
            continue
        seen.add(v)
        out.append(v)
    return out


def _online_buffer_tag(online_buffer_windows: int) -> str:
    return f"buf{int(online_buffer_windows)}"


def _compact_float_tag(value: float) -> str:
    return format(float(value), "g").replace(".", "")


def _pred_len_tag(pred_len: int) -> str:
    return f"pred{int(pred_len)}"


def _compose_output_suffix(
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
        parts.append(_pred_len_tag(int(pred_len)))
    return f"_{'_'.join(parts)}" if parts else ""


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

    # display names: map 'bay' -> 'ORCA' (case-insensitive), keep others
    display_tags: list[str] = []
    for t in list(refiner_tags):
        if str(t).strip().lower() == "bay":
            display_tags.append("ORCA")
        else:
            display_tags.append(str(t))
    header = ["Adapter"] + display_tags
    rows = []
    row_labels = [
        "Base Model Single Inference (ms)",
        "Single Inference Time (ms)",
        "Single Inference FLOPS",
        "Inference GPU Usage (MB)",
        "Single Training Time (ms)",
        "Single Training FLOPS",
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
            is_bay = str(tag).strip().lower() == "bay"
            if metric_key.startswith("train_") and not is_bay:
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
    bay_loss,
    bay_router,
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
        "bay_loss": bay_loss,
        "bay_router": bay_router,
        "routing_temperature": routing_temperature,
        "ema_error_momentum": ema_error_momentum,
        "pred_len": int(pred_len),
        "agg_metrics_base": dict(nan_metrics),
        "agg_metrics_flow": dict(nan_metrics),
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
        metric_flow_list: list[Dict | None] = []
        for pred_len in sorted(pred_set):
            rec = rec_map.get(pred_len)
            if rec is None:
                metric_base_list.append(_metric_dict_with_nan())
                metric_flow_list.append(_metric_dict_with_nan())
            else:
                metric_base_list.append(rec.get("agg_metrics_base"))
                metric_flow_list.append(rec.get("agg_metrics_flow"))

        avg_base = _average_metric_dicts_strict(metric_base_list)
        avg_flow = _average_metric_dicts_strict(metric_flow_list)

        base_rec = next(iter(rec_map.values()))
        merged = dict(base_rec)
        merged["pred_len"] = None
        merged["agg_metrics_base"] = avg_base
        merged["agg_metrics_flow"] = avg_flow
        averaged.append(merged)

    return averaged


def _normalize_csv_dataset_name(name: str) -> str:
    raw = str(name).strip()
    if raw in CSV_DATASET_SPECS:
        return raw
    lower_map = {k.lower(): k for k in CSV_DATASET_SPECS.keys()}
    key = lower_map.get(raw.lower())
    if key is None:
        raise ValueError(
            f"Unsupported CSV dataset {name!r}. Supported: {sorted(CSV_DATASET_SPECS.keys())}"
        )
    return key


def _resolve_refiner_tag(refiner: str) -> str:
    key = str(refiner).lower()
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
    if key == "elf":
        return "ELF"
    return "Linear"


def _normalize_training_method(value: str) -> str:
    key = str(value).strip().lower()
    if key not in {"batch", "online"}:
        raise ValueError(f"Unsupported training_method {value!r}. Supported: ['batch', 'online']")
    return key


def _normalize_refiner_input(value: str) -> str:
    key = str(value).strip().lower()
    if key == "epast":
        key = "e_past"
    if key not in {"all", "xy", "x", "y", "e_past"}:
        raise ValueError(f"Unsupported refiner_input {value!r}. Supported: ['all', 'xy', 'x', 'y', 'e_past']")
    return key


def _normalize_update_rule(value: str) -> str:
    key = str(value).strip().lower()
    if key not in {"plain", "bayesian", "semi_prior", "prior"}:
        raise ValueError(
            f"Unsupported update_rule {value!r}. Supported: ['plain', 'bayesian', 'semi_prior', 'prior']"
        )
    return key


def _normalize_bay_update_rule(value: str) -> str:
    key = str(value).strip().lower()
    if key not in {"plain", "bayesian", "semi_prior", "prior"}:
        raise ValueError(
            f"Unsupported Bay update_rule {value!r}. Supported: ['plain', 'bayesian', 'semi_prior', 'prior']"
        )
    return key


def _normalize_non_bay_update_rule(value: str) -> str:
    key = str(value).strip().lower()
    if key not in {"plain", "bayesian"}:
        raise ValueError(
            f"Unsupported non-Bay update_rule {value!r}. Supported: ['plain', 'bayesian']"
        )
    return key


def _normalize_bay_loss(value: str) -> str:
    key = str(value).strip().lower()
    if key not in {"mse", "mae", "huber"}:
        raise ValueError(f"Unsupported bay_loss {value!r}. Supported: ['mse', 'mae', 'huber']")
    return key


def _normalize_bay_router(value: str) -> str:
    key = str(value).strip().lower()
    if key == "ema":
        key = "inema"
    if key not in {"boltzmann", "inema", "hard"}:
        raise ValueError(
            f"Unsupported bay_router {value!r}. Supported: ['boltzmann', 'inema', 'hard']"
        )
    return key


def _flag_was_explicitly_set(args, *, positive_flag: str, negative_flag: str | None = None) -> bool:
    cmd = str(getattr(args, "_command_line", "") or "")
    if positive_flag and (positive_flag in cmd):
        return True
    if negative_flag and (negative_flag in cmd):
        return True
    return False


def _build_refiner_variant_suffix(
    refiner_tag: str,
    training_method: str,
    refiner_input: str | None,
    update_rule: str | None,
    online_buffer_windows: int | None = None,
    bay_loss: str | None = None,
    bay_router: str | None = None,
    routing_temperature: float | None = None,
    ema_error_momentum: float | None = None,
    *,
    args=None,
) -> str:
    def _append_seed_suffix(base: str) -> str:
        if args is None:
            return base
        seed_val = getattr(args, "random_seed", None)
        if seed_val is None:
            return base
        seed_tag = f"seed{int(seed_val)}"
        return f"{base}_{seed_tag}" if base else seed_tag

    if refiner_tag in {"Ridge", "ARIMA", "ETS"}:
        # Keep statistical baselines free of ablation tags, but still separate outputs by random seed.
        return _append_seed_suffix("")

    def _ablation_tags() -> list[str]:
        tags: list[str] = []
        if args is None:
            return tags
        gate_explicit = _flag_was_explicitly_set(
            args,
            positive_flag="--force_gate_open",
            negative_flag="--no-force_gate_open",
        )
        mix_explicit = _flag_was_explicitly_set(
            args,
            positive_flag="--channel_mix",
            negative_flag="--no-channel_mix",
        )
        if gate_explicit and bool(getattr(args, "force_gate_open", False)):
            tags.append("gate_open")
        if mix_explicit and (not bool(getattr(args, "channel_mix", True))):
            tags.append("ci")
        return tags

    if refiner_tag in {"Linear", "Bay", "Attn", "Bay_Attn"}:
        parts: list[str] = []
        if str(training_method).strip().lower() == "batch":
            parts.append("batch")
        if refiner_input is not None:
            parts.append(str(refiner_input))
        if update_rule is not None:
            parts.append(str(update_rule))
        if online_buffer_windows is not None:
            parts.append(_online_buffer_tag(int(online_buffer_windows)))
        if refiner_tag in {"Bay", "Bay_Attn"} and args is not None:
            parts.append(f"batch{int(getattr(args, 'batch', 256))}")
        if refiner_tag == "Bay":
            if routing_temperature is not None and not math.isclose(float(routing_temperature), 0.1, rel_tol=0.0, abs_tol=1e-12):
                parts.append(f"rt{_compact_float_tag(float(routing_temperature))}")
            if ema_error_momentum is not None and not math.isclose(float(ema_error_momentum), 0.2, rel_tol=0.0, abs_tol=1e-12):
                parts.append(f"ema{_compact_float_tag(float(ema_error_momentum))}")
        if refiner_tag == "Bay" and bay_loss in {"mae", "huber"}:
            parts.append(str(bay_loss))
        if refiner_tag == "Bay" and bay_router is not None:
            router_key = str(bay_router).strip().lower()
            if router_key == "inema":
                parts.append("ema")
            elif router_key == "hard":
                parts.append("hard")
        parts.extend(_ablation_tags())
        return _append_seed_suffix("_".join([p for p in parts if str(p).strip()]))
    if refiner_tag in {"AdaY", "DSOF", "TAFAS", "SOLID", "ELF"}:
        if args is not None and bool(getattr(args, "baseline_router", False)):
            return _append_seed_suffix("router")
    return _append_seed_suffix("")


def _expand_refiner_variants(base_refiner: str, args) -> list[dict]:
    refiner_tag = _resolve_refiner_tag(base_refiner)
    out: list[dict] = []
    if refiner_tag in {"Linear", "Bay", "Attn", "Bay_Attn"}:
        methods = [_normalize_training_method(x) for x in list(getattr(args, "training_method", ["online"]))]
        refiner_inputs = [_normalize_refiner_input(x) for x in list(getattr(args, "refiner_input", ["all"]))]
        if refiner_tag == "Bay":
            rules = [_normalize_bay_update_rule(x) for x in list(getattr(args, "update_rule", ["plain"]))]
        else:
            rules = [_normalize_non_bay_update_rule(x) for x in list(getattr(args, "update_rule", ["plain"]))]
        bay_losses = [_normalize_bay_loss(x) for x in list(getattr(args, "bay_loss", ["mse"]))]
        bay_routers = [_normalize_bay_router(x) for x in list(getattr(args, "bay_router", ["boltzmann"]))]
        routing_temperatures = _normalize_float_values(list(getattr(args, "routing_temperature", [0.1])))
        ema_error_momentums = _normalize_float_values(list(getattr(args, "ema_error_momentum", [0.2])))
        buffer_values = _normalize_online_buffer_values(
            list(getattr(args, "online_buffer_windows_values", getattr(args, "online_buffer_windows", [3000])))
        )
        for m in methods:
            for ri in refiner_inputs:
                for r in rules:
                    loss_variants = bay_losses if refiner_tag == "Bay" else [None]
                    router_variants = bay_routers if refiner_tag == "Bay" else [None]
                    temp_variants = routing_temperatures if refiner_tag == "Bay" else [None]
                    ema_variants = ema_error_momentums if refiner_tag == "Bay" else [None]
                    for bl in loss_variants:
                        for br in router_variants:
                            for rt in temp_variants:
                                for ema in ema_variants:
                                    for b in buffer_values:
                                        out.append(
                                            {
                                                "refiner": base_refiner,
                                                "refiner_tag": refiner_tag,
                                                "training_method": m,
                                                "refiner_input": ri,
                                                "update_rule": r,
                                                "online_buffer_windows": int(b),
                                                "bay_loss": bl,
                                                "bay_router": br,
                                                "routing_temperature": rt,
                                                "ema_error_momentum": ema,
                                                "variant_suffix": _build_refiner_variant_suffix(
                                                    refiner_tag,
                                                    m,
                                                    ri,
                                                    r,
                                                    online_buffer_windows=int(b),
                                                    bay_loss=bl,
                                                    bay_router=br,
                                                    routing_temperature=rt,
                                                    ema_error_momentum=ema,
                                                    args=args,
                                                ),
                                            }
                                        )
        return out

    out.append(
        {
            "refiner": base_refiner,
            "refiner_tag": refiner_tag,
            "training_method": None,
            "refiner_input": None,
            "update_rule": None,
            "online_buffer_windows": None,
            "bay_loss": None,
            "bay_router": None,
            "variant_suffix": _build_refiner_variant_suffix(
                refiner_tag,
                training_method="",
                refiner_input=None,
                update_rule=None,
                online_buffer_windows=None,
                bay_loss=None,
                bay_router=None,
                args=args,
            ),
        }
    )
    return out


def _unique_preserve_order(values: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for v in values:
        key = str(v)
        if key in seen:
            continue
        seen.add(key)
        out.append(key)
    return out


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


def _preflight_validate_and_print_plan(args, *, selected_refiners: list[str]) -> list[dict]:
    raw_methods = list(getattr(args, "training_method", ["online"]))
    raw_refiner_inputs = list(getattr(args, "refiner_input", ["all"]))
    raw_update_rules = list(getattr(args, "update_rule", ["plain"]))
    raw_bay_losses = list(getattr(args, "bay_loss", ["mse"]))
    raw_bay_routers = list(getattr(args, "bay_router", ["boltzmann"]))
    raw_routing_temperatures = list(getattr(args, "routing_temperature", [0.1]))
    raw_ema_error_momentums = list(getattr(args, "ema_error_momentum", [0.2]))

    normalized_methods = _unique_preserve_order([_normalize_training_method(x) for x in raw_methods])
    normalized_refiner_inputs = _unique_preserve_order([_normalize_refiner_input(x) for x in raw_refiner_inputs])
    normalized_update_rules = _unique_preserve_order([_normalize_update_rule(x) for x in raw_update_rules])
    normalized_bay_losses = _unique_preserve_order([_normalize_bay_loss(x) for x in raw_bay_losses])
    normalized_bay_routers = _unique_preserve_order([_normalize_bay_router(x) for x in raw_bay_routers])
    normalized_routing_temperatures = _unique_preserve_order(_normalize_float_values(raw_routing_temperatures))
    normalized_ema_error_momentums = _unique_preserve_order(_normalize_float_values(raw_ema_error_momentums))
    normalized_online_buffer_windows = _unique_preserve_order(
        _normalize_online_buffer_values(getattr(args, "online_buffer_windows", [3000]))
    )

    if not normalized_online_buffer_windows:
        raise ValueError("--online_buffer_windows must contain at least one positive integer")

    args.training_method = normalized_methods
    args.refiner_input = normalized_refiner_inputs
    args.update_rule = normalized_update_rules
    args.bay_loss = normalized_bay_losses
    args.bay_router = normalized_bay_routers
    args.routing_temperature = list(normalized_routing_temperatures)
    args.ema_error_momentum = list(normalized_ema_error_momentums)
    args.online_buffer_windows = list(normalized_online_buffer_windows)
    args.online_buffer_windows_values = list(normalized_online_buffer_windows)

    plan: list[dict] = []
    for refiner in selected_refiners:
        plan.extend(_expand_refiner_variants(refiner, args))

    print("\n[Refined-CSV] ====== Preflight Parameter Validation ======")
    print(f"[Refined-CSV] training_method={args.training_method}")
    print(f"[Refined-CSV] refiner_input={args.refiner_input}")
    print(f"[Refined-CSV] update_rule={args.update_rule}")
    print(f"[Refined-CSV] bay_loss={args.bay_loss}")
    print(f"[Refined-CSV] bay_router={args.bay_router}")
    print(f"[Refined-CSV] routing_temperature={args.routing_temperature}")
    print(f"[Refined-CSV] ema_error_momentum={args.ema_error_momentum}")
    print(f"[Refined-CSV] bay_huber_delta={float(getattr(args, 'bay_huber_delta', 1.0))}")
    print(f"[Refined-CSV] online_buffer_windows={args.online_buffer_windows_values} (stride=1 mini windows)")
    print(f"[Refined-CSV] force_gate_open={bool(getattr(args, 'force_gate_open', False))}")
    print(f"[Refined-CSV] channel_mix={bool(getattr(args, 'channel_mix', True))}")
    print(f"[Refined-CSV] bay_train_batch={int(getattr(args, 'batch', 256))}")
    print("[Refined-CSV] ====== Experiment Plan ======")
    for idx, variant in enumerate(plan, start=1):
        tag = str(variant.get("refiner_tag"))
        suffix = str(variant.get("variant_suffix", "")) or "default"
        tm = variant.get("training_method")
        ri = variant.get("refiner_input")
        ur = variant.get("update_rule")
        bl = variant.get("bay_loss")
        br = variant.get("bay_router")
        ob = variant.get("online_buffer_windows")
        print(
            f"[Refined-CSV][Plan {idx:02d}] refiner={tag} | suffix={suffix} | "
            f"training_method={tm} | refiner_input={ri} | update_rule={ur} | bay_loss={bl} | bay_router={br} | online_buffer_windows={ob}"
        )
    print(f"[Refined-CSV] Planned variant groups: {len(plan)}")

    return plan


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
    refiner_tag = _resolve_refiner_tag(refiner)
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
    refiner_tag = _resolve_refiner_tag(refiner)
    return build_split_summary_csv_paths(
        refiner_tag=refiner_tag,
        suffix=suffix,
        context_length=context_length,
        pred_len=pred_len,
        pred_len_avg=pred_len_avg,
    )


def _resolve_csv_path(args, dataset_name: str | None = None) -> Path:
    if args.csv_path:
        return Path(args.csv_path).expanduser().resolve()

    resolved_dataset = str(dataset_name or getattr(args, "dataset", "") or "").strip()
    if not resolved_dataset:
        raise ValueError("Either --csv_path or --dataset must be provided")

    cache_dir = Path(args.cache_dir).expanduser()
    csv_path = resolve_cached_csv_path(resolved_dataset, cache_dir=cache_dir)
    if csv_path.exists():
        return csv_path.resolve()

    if not args.auto_download:
        raise FileNotFoundError(
            f"CSV file not found: {csv_path}. Use --auto_download to fetch it into cache first."
        )

    return ensure_dataset_csv(
        resolved_dataset,
        cache_dir=cache_dir,
        force=bool(args.force_download),
        timeout=int(args.download_timeout),
    ).resolve()


def _configure_model_args(base_args, model_short_name: str):
    run_args = argparse.Namespace(**vars(base_args))
    run_args.model = normalize_model_name(model_short_name)
    run_args.tsfm_local_path = str(resolve_model_path(run_args.model, prefix=run_args.tsfm_model_prefix))
    return run_args


def _prepare_args_for_dataset(base_args, dataset_name: str):
    run_args = argparse.Namespace(**vars(base_args))
    run_args.dataset = _normalize_csv_dataset_name(dataset_name)
    run_args.csv_path = None

    csv_path = _resolve_csv_path(run_args, dataset_name=run_args.dataset)
    run_args.csv_path = str(csv_path)

    return run_args


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
    agg_metrics_flow: Dict | None,
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
        _write_one_row("refined", agg_metrics_flow)


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
        refiner_tag=_resolve_refiner_tag(refiner_value),
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
        agg_metrics_flow=rec.get("agg_metrics_flow"),
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
        flow_steps=int(rec.get("flow_steps", 0)),
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

    if str(rec.get("refiner_tag", "")).lower() in {"linear", "bay", "attn", "bay_attn"}:
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
            "bay_loss": rec.get("bay_loss", getattr(args, "bay_loss", None)),
            "bay_router": rec.get("bay_router", getattr(args, "bay_router", None)),
            "routing_temperature": rec.get("routing_temperature", getattr(args, "routing_temperature", None)),
            "ema_error_momentum": rec.get("ema_error_momentum", getattr(args, "ema_error_momentum", None)),
            "bay_huber_delta": float(getattr(args, "bay_huber_delta", 1.0)),
            "online_buffer_windows": rec.get("online_buffer_windows", getattr(args, "online_buffer_windows", None)),
            "force_gate_open": bool(getattr(args, "force_gate_open", False)),
            "channel_mix": bool(getattr(args, "channel_mix", True)),
            "bay_train_batch": int(getattr(args, "batch", 256)),
        }
        plot_path = save_refiner_loss_history_plot(
            logs_dir,
            dataset_name=dataset_result_name,
            model_name=model_short_name,
            refiner_name=f"{str(rec.get('refiner_tag', 'Linear'))}_{variant_suffix}_train",
            loss_history=train_loss_history,
            time_id=run_time_id,
        )
        val_plot_path = save_refiner_loss_history_plot(
            logs_dir,
            dataset_name=dataset_result_name,
            model_name=model_short_name,
            refiner_name=f"{str(rec.get('refiner_tag', 'Linear'))}_{variant_suffix}_val",
            loss_history=val_loss_history,
            time_id=run_time_id,
        )
        json_path = save_refiner_loss_history_json(
            logs_dir,
            dataset_name=dataset_result_name,
            model_name=model_short_name,
            refiner_name=f"{str(rec.get('refiner_tag', 'Linear'))}_{variant_suffix}",
            loss_history=train_loss_history,
            val_loss_history=val_loss_history,
            command_line=getattr(args, "_command_line", None),
            model_config=model_config,
            time_id=run_time_id,
        )
        print(f"[Refined-CSV] Refiner loss json saved: {json_path}")
        if plot_path is not None:
            print(f"[Refined-CSV] Refiner train loss plot saved: {plot_path}")
        if val_plot_path is not None:
            print(f"[Refined-CSV] Refiner val loss plot saved: {val_plot_path}")


def _persist_results_at_end(records: list[dict], args) -> None:
    if not records:
        print("[Refined-CSV] No result records to save.")
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
            print(f"[Refined-CSV] Summary MAE written: {out_paths['mae']}")
            print(f"[Refined-CSV] Summary MSE written: {out_paths['mse']}")

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
            print(f"[Refined-CSV] Pred-len average MAE summary written: {avg_out_paths['mae']}")
            print(f"[Refined-CSV] Pred-len average MSE summary written: {avg_out_paths['mse']}")


def _run_all_csv_matrix(args, device: torch.device, dataset_names: List[str], model_names: List[str], refiner_names: List[str]) -> None:
    records: list[dict] = []
    pred_len_values = _normalize_pred_len_values(getattr(args, "pred_len_values", [getattr(args, "pred_len", 96)]))
    speed_mode = bool(getattr(args, "speed", False))
    resume_enabled = bool(getattr(args, "resume_eval", False)) and (not speed_mode)
    speed_tables: dict[tuple[str, str, int], dict[str, dict]] = {}

    for refiner in refiner_names:
        variants = _expand_refiner_variants(refiner, args)
        for variant in variants:
            refiner_tag = str(variant["refiner_tag"])
            variant_suffix = str(variant["variant_suffix"])
            refiner_records_by_pred: dict[int, list[dict]] = {int(p): [] for p in pred_len_values}
            resume_completed_keys_by_pred: dict[int, set[tuple[str, str]]] = {int(p): set() for p in pred_len_values}
            resume_dataset_order_by_pred: dict[int, list[str]] = {int(p): list(dataset_names) for p in pred_len_values}
            resume_model_order_by_pred: dict[int, list[str]] = {
                int(p): [m for m in model_names if m in BACKEND_COMPATIBLE_MODELS]
                for p in pred_len_values
            }
            if (not resume_enabled) and (not speed_mode):
                for pred_len in pred_len_values:
                    summary_paths = _build_all_result_csv_paths(
                        refiner,
                        suffix=variant_suffix,
                        context_length=getattr(args, "context_length", None),
                        pred_len=int(pred_len),
                    )
                    for p in summary_paths.values():
                        if p.exists():
                            p.unlink()
                if len(pred_len_values) > 1:
                    avg_summary_paths = _build_all_result_csv_paths(
                        refiner,
                        suffix=variant_suffix,
                        context_length=getattr(args, "context_length", None),
                        pred_len_avg=True,
                    )
                    for p in avg_summary_paths.values():
                        if p.exists():
                            p.unlink()
            elif resume_enabled:
                for pred_len in pred_len_values:
                    summary_paths = _build_all_result_csv_paths(
                        refiner,
                        suffix=variant_suffix,
                        context_length=getattr(args, "context_length", None),
                        pred_len=int(pred_len),
                    )
                    existing_records, completed_keys, existing_ds_order, existing_model_order = _resume_load_existing_summary_records(
                        mae_csv_path=summary_paths["mae"],
                        mse_csv_path=summary_paths["mse"],
                        pred_len=int(pred_len),
                        refiner=refiner,
                        refiner_tag=refiner_tag,
                        variant_suffix=variant_suffix,
                        training_method=variant.get("training_method"),
                        refiner_input=variant.get("refiner_input"),
                        update_rule=variant.get("update_rule"),
                        online_buffer_windows=variant.get("online_buffer_windows"),
                    )
                    refiner_records_by_pred[int(pred_len)].extend(existing_records)
                    resume_completed_keys_by_pred[int(pred_len)] = set(completed_keys)
                    preferred_ds: list[str] = []
                    seen_ds: set[str] = set()
                    for ds in list(existing_ds_order) + list(dataset_names):
                        key = str(ds)
                        if key in seen_ds:
                            continue
                        seen_ds.add(key)
                        preferred_ds.append(key)
                    resume_dataset_order_by_pred[int(pred_len)] = preferred_ds

                    preferred_models: list[str] = []
                    seen_models: set[str] = set()
                    for m in list(existing_model_order) + [x for x in model_names if x in BACKEND_COMPATIBLE_MODELS]:
                        key = str(m)
                        if key in seen_models:
                            continue
                        seen_models.add(key)
                        preferred_models.append(key)
                    resume_model_order_by_pred[int(pred_len)] = preferred_models
                    records.extend(existing_records)

            model_order = [m for m in model_names if m in BACKEND_COMPATIBLE_MODELS]
            dataset_order = list(dataset_names)

            for model_short_name in model_names:
                if model_short_name not in BACKEND_COMPATIBLE_MODELS:
                    print(
                        f"[Refined-CSV][Skip] model={model_short_name} is registered but not yet supported by "
                        "the current Refined evaluator backend."
                    )
                    continue

                model_args = _configure_model_args(args, model_short_name)
                model_args.refiner = refiner
                model_args.refiner_variant_suffix = variant_suffix
                if variant.get("training_method") is not None:
                    model_args.training_method = variant.get("training_method")
                if variant.get("refiner_input") is not None:
                    model_args.refiner_input = variant.get("refiner_input")
                if variant.get("update_rule") is not None:
                    model_args.update_rule = variant.get("update_rule")
                if variant.get("online_buffer_windows") is not None:
                    model_args.online_buffer_windows = int(variant.get("online_buffer_windows"))
                if variant.get("bay_loss") is not None:
                    model_args.bay_loss = variant.get("bay_loss")
                if variant.get("bay_router") is not None:
                    model_args.bay_router = variant.get("bay_router")
                if variant.get("routing_temperature") is not None:
                    model_args.routing_temperature = float(variant.get("routing_temperature"))
                if variant.get("ema_error_momentum") is not None:
                    model_args.ema_error_momentum = float(variant.get("ema_error_momentum"))
                print(
                    f"\n[Refined-CSV][All] Running refiner={refiner_tag}({variant_suffix or 'default'}) model={model_short_name} "
                    f"| local_path={model_args.tsfm_local_path} | datasets={dataset_names}"
                )

                for dataset_name in dataset_names:
                    for pred_len in pred_len_values:
                        print(f"\n[Refined-CSV][All] Running dataset={dataset_name} | pred_len={pred_len} ...")
                        run_args = _prepare_args_for_dataset(model_args, dataset_name)
                        run_args.pred_len = int(pred_len)

                        if resume_enabled:
                            key = (str(dataset_name), str(model_short_name))
                            if key in resume_completed_keys_by_pred.get(int(pred_len), set()):
                                print(
                                    f"[Refined-CSV][Resume][Skip] refiner={refiner_tag}({variant_suffix or 'default'}) "
                                    f"model={model_short_name} dataset={dataset_name} pred_len={pred_len} has complete non-NaN summary entry.",
                                    flush=True,
                                )
                                del run_args
                                gc.collect()
                                if torch.cuda.is_available():
                                    torch.cuda.empty_cache()
                                continue

                        try:
                            eval_out = run_geoflow_csv_evaluation(args=run_args, device=device)
                            eval_out["dataset_label"] = dataset_name
                            eval_out["dataset_result_name"] = str(Path(str(run_args.csv_path)).stem)
                            eval_out["model_short_name"] = model_short_name
                            eval_out["refiner"] = refiner
                            eval_out["refiner_tag"] = refiner_tag
                            eval_out["variant_suffix"] = variant_suffix
                            eval_out["training_method"] = variant.get("training_method")
                            eval_out["refiner_input"] = variant.get("refiner_input")
                            eval_out["update_rule"] = variant.get("update_rule")
                            eval_out["online_buffer_windows"] = variant.get("online_buffer_windows")
                            eval_out["bay_loss"] = variant.get("bay_loss")
                            eval_out["bay_router"] = variant.get("bay_router")
                            eval_out["routing_temperature"] = variant.get("routing_temperature")
                            eval_out["ema_error_momentum"] = variant.get("ema_error_momentum")
                            eval_out["pred_len"] = int(pred_len)
                        except Exception as exc:
                            print(
                                f"[Refined-CSV][Skip][RunError] refiner={refiner_tag}({variant_suffix or 'default'}) "
                                f"model={model_short_name} dataset={dataset_name} pred_len={pred_len} failed: "
                                f"{type(exc).__name__}: {exc}",
                                flush=True,
                            )
                            print(traceback.format_exc(), flush=True)
                            eval_out = _build_failed_eval_record(
                                dataset_label=dataset_name,
                                model_short_name=model_short_name,
                                refiner=refiner,
                                refiner_tag=refiner_tag,
                                variant_suffix=variant_suffix,
                                training_method=variant.get("training_method"),
                                refiner_input=variant.get("refiner_input"),
                                update_rule=variant.get("update_rule"),
                                online_buffer_windows=variant.get("online_buffer_windows"),
                                bay_loss=variant.get("bay_loss"),
                                bay_router=variant.get("bay_router"),
                                routing_temperature=variant.get("routing_temperature"),
                                ema_error_momentum=variant.get("ema_error_momentum"),
                                pred_len=int(pred_len),
                            )

                        records.append(eval_out)
                        refiner_records_by_pred[int(pred_len)].append(eval_out)
                        if speed_mode:
                            speed_key = (str(dataset_name), str(model_short_name), int(pred_len))
                            speed_tables.setdefault(speed_key, {})[refiner_tag] = dict(eval_out.get("speed_stats", {}))
                            refiner_cols = [_resolve_refiner_tag(r) for r in refiner_names]
                            out_path = _write_speed_table(
                                dataset_name=str(dataset_name),
                                model_name=str(model_short_name),
                                pred_len=int(pred_len),
                                refiner_tags=refiner_cols,
                                stats_by_refiner=speed_tables[speed_key],
                            )
                            print(f"[Refined-CSV] Speed table written: {out_path}")
                        else:
                            _persist_single_record(eval_out, run_args)
                            out_paths = _refresh_refiner_summary_csv(
                                refiner_records=refiner_records_by_pred[int(pred_len)],
                                refiner_value=refiner,
                                dataset_order=resume_dataset_order_by_pred.get(int(pred_len), dataset_order),
                                model_order=resume_model_order_by_pred.get(int(pred_len), model_order),
                                context_length=getattr(args, "context_length", None),
                                pred_len=int(pred_len),
                            )
                            print(f"[Refined-CSV] Refiner MAE summary updated: {out_paths['mae']}")
                            print(f"[Refined-CSV] Refiner MSE summary updated: {out_paths['mse']}")

                            if len(pred_len_values) > 1:
                                merged_records: list[dict] = []
                                for rec_list in refiner_records_by_pred.values():
                                    merged_records.extend(rec_list)
                                avg_records = _build_pred_len_average_records(merged_records, pred_len_values)
                                avg_paths = _refresh_refiner_summary_csv(
                                    refiner_records=avg_records,
                                    refiner_value=refiner,
                                    dataset_order=resume_dataset_order_by_pred.get(int(pred_len), dataset_order),
                                    model_order=resume_model_order_by_pred.get(int(pred_len), model_order),
                                    context_length=getattr(args, "context_length", None),
                                    pred_len=None,
                                    pred_len_avg=True,
                                )
                                print(f"[Refined-CSV] Pred-len average MAE summary updated: {avg_paths['mae']}")
                                print(f"[Refined-CSV] Pred-len average MSE summary updated: {avg_paths['mse']}")

                        del run_args
                        del eval_out
                        gc.collect()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()

    if not speed_mode:
        _persist_results_at_end(records, args)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Refined refiner evaluation on local CSV time-series data",
        allow_abbrev=False,
    )

    parser.add_argument(
        "--dataset",
        nargs="+",
        default=["all"],
        help=f"One or more named CSV datasets (space-separated), or 'all'. Supported: {sorted(CSV_DATASET_SPECS.keys())}",
    )
    parser.add_argument("--csv_path", type=str, default=None, help="Direct path to CSV file. Overrides --dataset when provided")
    parser.add_argument(
        "--cache_dir",
        type=str,
        nargs="?",
        const=str(DEFAULT_CACHE_DIR),
        default=str(DEFAULT_CACHE_DIR),
        help="CSV dataset cache directory only (not used for model inference cache)",
    )
    parser.add_argument("--auto_download", action="store_true", help="Auto-download named dataset CSV into cache when missing")
    parser.add_argument("--force_download", action="store_true", help="Force re-download when using --auto_download")
    parser.add_argument("--download_timeout", type=int, default=120, help="HTTP timeout for CSV download")

    parser.add_argument("--target_column", type=str, default="all", help="Target column name or 'all' for multivariate forecasting")
    parser.add_argument("--pred_len", nargs="+", type=int, default=[96], help="One or more prediction lengths")
    parser.add_argument("--windows", type=int, default=None, help="Window count for instance generation")

    parser.add_argument("--device", default="cuda", help="cpu or cuda")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--download_online", action="store_true", help="Download TSFM model from Hugging Face when supported")
    parser.add_argument(
        "--model",
        nargs="+",
        default=["all"],
        help=f"One or more TSFM model names (space-separated), or 'all'. Supported: {TSFM_MODEL_ORDER}",
    )
    parser.add_argument("--tsfm_model_prefix", type=str, default=str(TSFM_MODEL_PATH_PREFIX), help="TSFM local model root directory prefix")
    parser.add_argument("--tsfm_local_path", default=None, help="Explicit TSFM local model path, only valid with a single model")
    parser.add_argument("--context_length", type=int, default=520, help="Base model context length")
    parser.add_argument(
        "--cache",
        "--cahce",
        dest="cache",
        action="store_true",
        help="Enable model inference cache reuse",
    )
    parser.add_argument(
        "--attn_maps",
        action="store_true",
        help="Enable attention map inspection mode (currently supports moirai-2 and chronos-2)",
    )

    parser.add_argument(
        "--training_method",
        nargs="+",
        default=["online"],
        help="Training method list for Linear/Attn/Bay/Bay_Attn. Supported: batch online",
    )
    parser.add_argument(
        "--refiner_input",
        nargs="+",
        default=["all"],
        help="Refiner input mode list for Linear/Attn/Bay/Bay_Attn. Supported: all xy x y e_past",
    )
    parser.add_argument(
        "--update_rule",
        nargs="+",
        default=["plain"],
        help="Update rule list. Supported: plain bayesian; Bay-only extensions: semi_prior prior",
    )
    parser.add_argument(
        "--online_buffer_windows",
        "--online_buffer_meta_windows",
        dest="online_buffer_windows",
        nargs="+",
        type=int,
        default=[3000],
        help=(
            "For online mode (Linear/Attn/Bay/Bay_Attn), number of stride=1 mini windows to buffer before each "
            "training trigger. Supports list input, e.g. --online_buffer_windows 512 3000 6000"
        ),
    )
    parser.add_argument(
        "--force_gate_open",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Refiner ablation (Linear/Attn/Bay/Bay_Attn): if enabled, force confidence gate c_t=1.0 and disable confidence routing effect.",
    )
    parser.add_argument(
        "--channel_mix",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Refiner ablation (Linear/Attn/Bay/Bay_Attn): enable/disable channel-mixing blocks; --no-channel_mix switches to CI-style channel-independent path.",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=256,
        help="Bay/Bay_Attn refiner training batch size.",
    )
    parser.add_argument(
        "--bay_loss",
        nargs="+",
        default=["mse"],
        help="Bay refiner observation loss variant list. Supported: mse mae huber",
    )
    parser.add_argument(
        "--bay_router",
        nargs="+",
        default=["boltzmann"],
        help="Bay refiner router variant list. Supported: boltzmann inema hard",
    )
    parser.add_argument(
        "--routing_temperature",
        nargs="+",
        type=float,
        default=[0.1],
        help="Bay refiner routing temperature list. Supports list input, e.g. --routing_temperature 0.1 0.2",
    )
    parser.add_argument(
        "--ema_error_momentum",
        nargs="+",
        type=float,
        default=[0.2],
        help="Bay refiner EMA error momentum list. Supports list input, e.g. --ema_error_momentum 0.2 0.3",
    )
    parser.add_argument(
        "--bay_huber_delta",
        type=float,
        default=1.0,
        help="Huber delta used when --bay_loss includes huber.",
    )
    parser.add_argument(
        "--baseline_router",
        action="store_true",
        help="Enable Boltzmann router for baseline refiners (AdaY/DSOF/TAFAS/SOLID/ELF).",
    )
    parser.add_argument(
        "--speed",
        action="store_true",
        help="Run speed evaluation only and emit a dedicated timing/FLOPS table.",
    )
    parser.add_argument(
        "--resume_eval",
        action="store_true",
        help="Resume interrupted evaluations by reusing existing non-NaN summary entries and rerunning only NaN/missing entries.",
    )
    parser.add_argument(
        "--refiner",
        nargs="+",
        default=["linear"],
        help="One or more refiners (space-separated), or 'all'.",
    )
    parser.add_argument(
        "--random_seed",
        type=int,
        default=None,
        help="Optional random seed used for reproducibility and appended to refiner suffix when provided.",
    )

    args = parser.parse_args()
    if getattr(args, "random_seed", None) is not None:
        seed_val = int(getattr(args, "random_seed"))
        random.seed(seed_val)
        np.random.seed(seed_val)
        torch.manual_seed(seed_val)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed_val)
    if bool(getattr(args, "attn_maps", False)) and bool(getattr(args, "cache", False)):
        raise ValueError("--attn_maps and --cache are mutually exclusive and cannot be used together")
    args._command_line = " ".join(shlex.quote(str(x)) for x in sys.argv)

    dataset_tokens = [str(x).strip() for x in (args.dataset or []) if str(x).strip()]
    if not dataset_tokens:
        dataset_tokens = ["all"]
    if any(str(x).lower() == "all" for x in dataset_tokens):
        selected_datasets = list(ALL_CSV_DATASETS)
    else:
        selected_datasets = [_normalize_csv_dataset_name(d) for d in dataset_tokens]

    model_tokens = [str(x).strip() for x in (args.model or []) if str(x).strip()]
    if not model_tokens:
        model_tokens = ["all"]
    if any(str(x).lower() == "all" for x in model_tokens):
        selected_models = list(TSFM_MODEL_ORDER)
    else:
        selected_models = [normalize_model_name(m) for m in model_tokens]

    refiner_tokens = [str(x).strip() for x in (args.refiner or []) if str(x).strip()]
    if not refiner_tokens:
        refiner_tokens = ["linear"]
    if any(str(x).lower() == "all" for x in refiner_tokens):
        selected_refiners = list(CANONICAL_REFINERS)
    else:
        selected_refiners = refiner_tokens
    invalid_refiners = [r for r in selected_refiners if r not in REFINER_CHOICES]
    if invalid_refiners:
        raise ValueError(f"Unsupported refiner names: {invalid_refiners}. Supported: {REFINER_CHOICES}")

    pred_len_values = _normalize_pred_len_values(getattr(args, "pred_len", [96]))
    args.pred_len_values = list(pred_len_values)
    args.pred_len = int(pred_len_values[0])
    print(f"[Refined-CSV] pred_len_values={pred_len_values}")

    _ = _preflight_validate_and_print_plan(args, selected_refiners=selected_refiners)

    if args.csv_path is None and any(str(x).lower() == "all" for x in dataset_tokens):
        args.auto_download = True

    if args.tsfm_local_path and len(selected_models) > 1:
        raise ValueError("--tsfm_local_path only supports a single model. Use --model with one value")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    if args.csv_path is not None:
        csv_path = Path(args.csv_path).expanduser().resolve()
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV file not found: {csv_path}")
        csv_stem = csv_path.stem
        speed_mode = bool(getattr(args, "speed", False))
        resume_enabled = bool(getattr(args, "resume_eval", False)) and (not speed_mode)
        speed_tables: dict[tuple[str, str, int], dict[str, dict]] = {}

        records: list[dict] = []
        for refiner in selected_refiners:
            variants = _expand_refiner_variants(refiner, args)
            for variant in variants:
                refiner_tag = str(variant["refiner_tag"])
                variant_suffix = str(variant["variant_suffix"])
                refiner_records_by_pred: dict[int, list[dict]] = {int(p): [] for p in pred_len_values}
                resume_completed_keys_by_pred: dict[int, set[tuple[str, str]]] = {int(p): set() for p in pred_len_values}
                resume_dataset_order_by_pred: dict[int, list[str]] = {int(p): [csv_stem] for p in pred_len_values}
                resume_model_order_by_pred: dict[int, list[str]] = {
                    int(p): [m for m in selected_models if m in BACKEND_COMPATIBLE_MODELS]
                    for p in pred_len_values
                }
                if (not resume_enabled) and (not speed_mode):
                    for pred_len in pred_len_values:
                        summary_paths = _build_all_result_csv_paths(
                            refiner,
                            suffix=variant_suffix,
                            context_length=getattr(args, "context_length", None),
                            pred_len=int(pred_len),
                        )
                        for p in summary_paths.values():
                            if p.exists():
                                p.unlink()
                    if len(pred_len_values) > 1:
                        avg_summary_paths = _build_all_result_csv_paths(
                            refiner,
                            suffix=variant_suffix,
                            context_length=getattr(args, "context_length", None),
                            pred_len_avg=True,
                        )
                        for p in avg_summary_paths.values():
                            if p.exists():
                                p.unlink()
                elif resume_enabled:
                    for pred_len in pred_len_values:
                        summary_paths = _build_all_result_csv_paths(
                            refiner,
                            suffix=variant_suffix,
                            context_length=getattr(args, "context_length", None),
                            pred_len=int(pred_len),
                        )
                        existing_records, completed_keys, existing_ds_order, existing_model_order = _resume_load_existing_summary_records(
                            mae_csv_path=summary_paths["mae"],
                            mse_csv_path=summary_paths["mse"],
                            pred_len=int(pred_len),
                            refiner=refiner,
                            refiner_tag=refiner_tag,
                            variant_suffix=variant_suffix,
                            training_method=variant.get("training_method"),
                            refiner_input=variant.get("refiner_input"),
                            update_rule=variant.get("update_rule"),
                            online_buffer_windows=variant.get("online_buffer_windows"),
                        )
                        refiner_records_by_pred[int(pred_len)].extend(existing_records)
                        resume_completed_keys_by_pred[int(pred_len)] = set(completed_keys)
                        preferred_ds: list[str] = []
                        seen_ds: set[str] = set()
                        for ds in list(existing_ds_order) + [csv_stem]:
                            key = str(ds)
                            if key in seen_ds:
                                continue
                            seen_ds.add(key)
                            preferred_ds.append(key)
                        resume_dataset_order_by_pred[int(pred_len)] = preferred_ds

                        preferred_models: list[str] = []
                        seen_models: set[str] = set()
                        for m in list(existing_model_order) + [x for x in selected_models if x in BACKEND_COMPATIBLE_MODELS]:
                            key = str(m)
                            if key in seen_models:
                                continue
                            seen_models.add(key)
                            preferred_models.append(key)
                        resume_model_order_by_pred[int(pred_len)] = preferred_models
                        records.extend(existing_records)

                model_order = [m for m in selected_models if m in BACKEND_COMPATIBLE_MODELS]
                dataset_order = [csv_stem]

                for model_short_name in selected_models:
                    if model_short_name not in BACKEND_COMPATIBLE_MODELS:
                        print(
                            f"[Refined-CSV][Skip] model={model_short_name} is registered but not yet supported by "
                            "the current Refined evaluator backend."
                        )
                        continue

                    run_args = _configure_model_args(args, model_short_name)
                    if args.tsfm_local_path:
                        run_args.tsfm_local_path = str(Path(args.tsfm_local_path).expanduser().resolve())
                    run_args.csv_path = str(csv_path)
                    run_args.dataset = None
                    run_args.refiner = refiner
                    run_args.refiner_variant_suffix = variant_suffix
                    if variant.get("training_method") is not None:
                        run_args.training_method = variant.get("training_method")
                    if variant.get("refiner_input") is not None:
                        run_args.refiner_input = variant.get("refiner_input")
                    if variant.get("update_rule") is not None:
                        run_args.update_rule = variant.get("update_rule")
                    if variant.get("online_buffer_windows") is not None:
                        run_args.online_buffer_windows = int(variant.get("online_buffer_windows"))
                    if variant.get("bay_loss") is not None:
                        run_args.bay_loss = variant.get("bay_loss")
                    if variant.get("bay_router") is not None:
                        run_args.bay_router = variant.get("bay_router")
                    if variant.get("routing_temperature") is not None:
                        run_args.routing_temperature = float(variant.get("routing_temperature"))
                    if variant.get("ema_error_momentum") is not None:
                        run_args.ema_error_momentum = float(variant.get("ema_error_momentum"))

                    for pred_len in pred_len_values:
                        if resume_enabled:
                            key = (str(Path(run_args.csv_path).stem), str(model_short_name))
                            if key in resume_completed_keys_by_pred.get(int(pred_len), set()):
                                print(
                                    f"[Refined-CSV][Resume][Skip] refiner={refiner_tag}({variant_suffix or 'default'}) "
                                    f"model={model_short_name} dataset={Path(run_args.csv_path).stem} pred_len={pred_len} has complete non-NaN summary entry.",
                                    flush=True,
                                )
                                continue
                        run_args.pred_len = int(pred_len)
                        print(
                            f"[Refined-CSV] Running refiner={refiner_tag}({variant_suffix or 'default'}) model={run_args.model} "
                            f"| local_path={run_args.tsfm_local_path} | dataset={Path(run_args.csv_path).stem} | pred_len={pred_len}"
                        )
                        try:
                            eval_out = run_geoflow_csv_evaluation(args=run_args, device=device)
                            eval_out["dataset_label"] = Path(run_args.csv_path).stem
                            eval_out["dataset_result_name"] = Path(run_args.csv_path).stem
                            eval_out["model_short_name"] = model_short_name
                            eval_out["refiner"] = refiner
                            eval_out["refiner_tag"] = refiner_tag
                            eval_out["variant_suffix"] = variant_suffix
                            eval_out["training_method"] = variant.get("training_method")
                            eval_out["refiner_input"] = variant.get("refiner_input")
                            eval_out["update_rule"] = variant.get("update_rule")
                            eval_out["online_buffer_windows"] = variant.get("online_buffer_windows")
                            eval_out["bay_loss"] = variant.get("bay_loss")
                            eval_out["bay_router"] = variant.get("bay_router")
                            eval_out["routing_temperature"] = variant.get("routing_temperature")
                            eval_out["ema_error_momentum"] = variant.get("ema_error_momentum")
                            eval_out["pred_len"] = int(pred_len)
                        except Exception as exc:
                            print(
                                f"[Refined-CSV][Skip][RunError] refiner={refiner_tag}({variant_suffix or 'default'}) "
                                f"model={model_short_name} dataset={Path(run_args.csv_path).stem} pred_len={pred_len} failed: "
                                f"{type(exc).__name__}: {exc}",
                                flush=True,
                            )
                            print(traceback.format_exc(), flush=True)
                            eval_out = _build_failed_eval_record(
                                dataset_label=Path(run_args.csv_path).stem,
                                model_short_name=model_short_name,
                                refiner=refiner,
                                refiner_tag=refiner_tag,
                                variant_suffix=variant_suffix,
                                training_method=variant.get("training_method"),
                                refiner_input=variant.get("refiner_input"),
                                update_rule=variant.get("update_rule"),
                                online_buffer_windows=variant.get("online_buffer_windows"),
                                bay_loss=variant.get("bay_loss"),
                                bay_router=variant.get("bay_router"),
                                routing_temperature=variant.get("routing_temperature"),
                                ema_error_momentum=variant.get("ema_error_momentum"),
                                pred_len=int(pred_len),
                            )

                        records.append(eval_out)
                        refiner_records_by_pred[int(pred_len)].append(eval_out)
                        if speed_mode:
                            speed_key = (str(Path(run_args.csv_path).stem), str(model_short_name), int(pred_len))
                            speed_tables.setdefault(speed_key, {})[refiner_tag] = dict(eval_out.get("speed_stats", {}))
                            refiner_cols = [_resolve_refiner_tag(r) for r in selected_refiners]
                            out_path = _write_speed_table(
                                dataset_name=str(Path(run_args.csv_path).stem),
                                model_name=str(model_short_name),
                                pred_len=int(pred_len),
                                refiner_tags=refiner_cols,
                                stats_by_refiner=speed_tables[speed_key],
                            )
                            print(f"[Refined-CSV] Speed table written: {out_path}")
                        else:
                            _persist_single_record(eval_out, run_args)
                            out_paths = _refresh_refiner_summary_csv(
                                refiner_records=refiner_records_by_pred[int(pred_len)],
                                refiner_value=refiner,
                                dataset_order=resume_dataset_order_by_pred.get(int(pred_len), dataset_order),
                                model_order=resume_model_order_by_pred.get(int(pred_len), model_order),
                                context_length=getattr(args, "context_length", None),
                                pred_len=int(pred_len),
                            )
                            print(f"[Refined-CSV] Refiner MAE summary updated: {out_paths['mae']}")
                            print(f"[Refined-CSV] Refiner MSE summary updated: {out_paths['mse']}")

                            if len(pred_len_values) > 1:
                                merged_records: list[dict] = []
                                for rec_list in refiner_records_by_pred.values():
                                    merged_records.extend(rec_list)
                                avg_records = _build_pred_len_average_records(merged_records, pred_len_values)
                                avg_paths = _refresh_refiner_summary_csv(
                                    refiner_records=avg_records,
                                    refiner_value=refiner,
                                    dataset_order=resume_dataset_order_by_pred.get(int(pred_len), dataset_order),
                                    model_order=resume_model_order_by_pred.get(int(pred_len), model_order),
                                    context_length=getattr(args, "context_length", None),
                                    pred_len=None,
                                    pred_len_avg=True,
                                )
                                print(f"[Refined-CSV] Pred-len average MAE summary updated: {avg_paths['mae']}")
                                print(f"[Refined-CSV] Pred-len average MSE summary updated: {avg_paths['mse']}")

                        del eval_out
                        gc.collect()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()

                    del run_args

        if not speed_mode:
            _persist_results_at_end(records, args)
        return

    _run_all_csv_matrix(
        args=args,
        device=device,
        dataset_names=selected_datasets,
        model_names=selected_models,
        refiner_names=selected_refiners,
    )


if __name__ == "__main__":
    main()
