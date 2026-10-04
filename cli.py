from __future__ import annotations
import argparse
import math
from itertools import product
from typing import List
from data.download_CSV import DEFAULT_CACHE_DIR, CSV_DATASET_SPECS
from model_registry import TSFM_MODEL_ORDER, TSFM_MODEL_PATH_PREFIX


REFINER_NAMES = {
    "orca_no_decay": "ORCA_NoDecay", "orca": "ORCA", "aday": "AdaY", "dsof": "DSOF",
    "tafas": "TAFAS", "solid": "SOLID", "elf": "ELF", "ridge": "Ridge", "ets": "ETS",
}
CANONICAL_REFINERS = list(REFINER_NAMES.values())
REFINER_CHOICES = CANONICAL_REFINERS + list(REFINER_NAMES)


def resolve_refiner_tag(refiner: str) -> str:
    key = str(refiner).strip().lower()
    if key not in REFINER_NAMES:
        raise ValueError(f"Unsupported refiner={refiner!r}. Supported: {CANONICAL_REFINERS}")
    return REFINER_NAMES[key]


ALL_CSV_DATASETS: List[str] = ["ETTh1", "ETTh2", "ETTm1", "ETTm2", "Exchange", "Weather", "Electricity", "Traffic"]


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
    if key not in {"plain", "bayesian"}:
        raise ValueError(
            f"Unsupported update_rule {value!r}. Supported: ['plain', 'bayesian']"
        )
    return key


def _normalize_router(value: str) -> str:
    key = str(value).strip().lower()
    if key == "ema":
        key = "inema"
    if key not in {"boltzmann", "inema", "hard"}:
        raise ValueError(
            f"Unsupported router {value!r}. Supported: ['boltzmann', 'inema', 'hard']"
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
    router: str | None = None,
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

    if refiner_tag in {"Ridge", "ETS"}:
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

    if refiner_tag in {"ORCA_NoDecay", "ORCA"}:
        parts: list[str] = []
        if str(training_method).strip().lower() == "batch":
            parts.append("batch")
        if refiner_input is not None:
            parts.append(str(refiner_input))
        if update_rule is not None:
            parts.append(str(update_rule))
        if online_buffer_windows is not None:
            parts.append(_online_buffer_tag(int(online_buffer_windows)))
        if refiner_tag in {"ORCA"} and args is not None:
            parts.append(f"batch{int(getattr(args, 'train_batch_size', 256))}")
        if refiner_tag == "ORCA":
            if routing_temperature is not None and not math.isclose(float(routing_temperature), 0.1, rel_tol=0.0, abs_tol=1e-12):
                parts.append(f"rt{_compact_float_tag(float(routing_temperature))}")
            if ema_error_momentum is not None and not math.isclose(float(ema_error_momentum), 0.2, rel_tol=0.0, abs_tol=1e-12):
                parts.append(f"ema{_compact_float_tag(float(ema_error_momentum))}")
        if refiner_tag == "ORCA" and router is not None:
            router_key = str(router).strip().lower()
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
    refiner_tag = resolve_refiner_tag(base_refiner)
    method = _normalize_training_method(args.training_method)
    if method == "batch" and refiner_tag not in {"AdaY", "DSOF", "TAFAS", "SOLID"}:
        raise ValueError(f"{refiner_tag} supports --training_method online only")
    inputs = args.refiner_input if refiner_tag in {"ORCA_NoDecay", "ORCA"} else [None]
    rules = args.update_rule if refiner_tag in {"ORCA_NoDecay", "ORCA"} else [None]
    out = []
    for refiner_input, update_rule in product(inputs, rules):
        uses_context = refiner_tag in {"ORCA_NoDecay", "ORCA"}
        variant = {
            "refiner": base_refiner,
            "refiner_tag": refiner_tag,
            "training_method": method,
            "refiner_input": refiner_input,
            "update_rule": update_rule,
            "online_buffer_windows": int(args.online_buffer_windows) if uses_context else None,
            "router": args.router if refiner_tag == "ORCA" else None,
            "routing_temperature": args.routing_temperature if refiner_tag == "ORCA" else None,
            "ema_error_momentum": args.ema_error_momentum if refiner_tag == "ORCA" else None,
        }
        variant["variant_suffix"] = _build_refiner_variant_suffix(
            refiner_tag, method, refiner_input, update_rule,
            online_buffer_windows=variant["online_buffer_windows"],
            router=variant["router"],
            routing_temperature=variant["routing_temperature"],
            ema_error_momentum=variant["ema_error_momentum"], args=args,
        )
        if not uses_context and method == "batch":
            variant["variant_suffix"] = "batch" + ("_" + variant["variant_suffix"] if variant["variant_suffix"] else "")
        out.append(variant)
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


def _preflight_validate_and_print_plan(args, *, selected_refiners: list[str]) -> list[dict]:
    args.training_method = _normalize_training_method(args.training_method)
    args.refiner_input = _unique_preserve_order([_normalize_refiner_input(x) for x in args.refiner_input])
    args.update_rule = _unique_preserve_order([_normalize_update_rule(x) for x in args.update_rule])
    args.router = _normalize_router(args.router)
    if args.online_buffer_windows <= 0 or args.train_batch_size <= 0 or args.batch_size <= 0:
        raise ValueError("Buffer capacity and batch sizes must be positive")
    plan = []
    for refiner in selected_refiners:
        plan.extend(_expand_refiner_variants(refiner, args))
    for idx, variant in enumerate(plan, start=1):
        print(f"[ORCA][Plan {idx:02d}] refiner={variant['refiner_tag']} | "
              f"training_method={variant['training_method']} | suffix={variant['variant_suffix'] or 'default'}")
    return plan


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run ORCA and baseline adapters on local CSV time-series data",
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

    parser.add_argument("--device", default="cuda", help="cpu, cuda, or a CUDA device such as cuda:1")
    parser.add_argument("--batch_size", type=int, default=1, help="Base-model inference batch size")
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
        dest="cache",
        action="store_true",
        help="Enable model inference cache reuse",
    )

    parser.add_argument(
        "--training_method",
        default="online",
        help="Training mode: online; batch is also supported by baseline adapters",
    )
    parser.add_argument(
        "--refiner_input",
        nargs="+",
        default=["all"],
        help="Refiner input mode list for ORCA/ORCA_NoDecay. Supported: all xy x y e_past",
    )
    parser.add_argument(
        "--update_rule",
        nargs="+",
        default=["plain"],
        help="Update rule list. Supported: plain bayesian",
    )
    parser.add_argument(
        "--online_buffer_windows",
        dest="online_buffer_windows",
        type=int,
        default=3000,
        help=(
            "For online mode (ORCA/ORCA_NoDecay), number of stride=1 mini windows to buffer before each "
            "training trigger"
        ),
    )
    parser.add_argument(
        "--force_gate_open",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Refiner ablation (ORCA/ORCA_NoDecay): if enabled, force confidence gate c_t=1.0 and disable confidence routing effect.",
    )
    parser.add_argument(
        "--channel_mix",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Refiner ablation (ORCA/ORCA_NoDecay): enable/disable channel-mixing blocks; --no-channel_mix switches to CI-style channel-independent path.",
    )
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=256,
        help="ORCA refiner training batch size.",
    )
    parser.add_argument(
        "--router",
        default="boltzmann",
        help="ORCA router. Supported: boltzmann inema hard",
    )
    parser.add_argument(
        "--routing_temperature",
        type=float,
        default=0.1,
        help="ORCA routing temperature",
    )
    parser.add_argument(
        "--ema_error_momentum",
        type=float,
        default=0.2,
        help="ORCA EMA error momentum",
    )
    parser.add_argument(
        "--baseline_router",
        action="store_true",
        help="Enable Boltzmann router for baseline refiners (AdaY/DSOF/TAFAS/SOLID/ELF).",
    )
    parser.add_argument(
        "--speed",
        action="store_true",
        help="Run speed evaluation only and emit a dedicated timing/FLOPs table.",
    )
    parser.add_argument(
        "--resume_eval",
        action="store_true",
        help="Resume interrupted evaluations by reusing existing non-NaN summary entries and rerunning only NaN/missing entries.",
    )
    parser.add_argument(
        "--refiner",
        nargs="+",
        default=["orca_no_decay"],
        help="One or more refiners (space-separated), or 'all'.",
    )
    parser.add_argument(
        "--random_seed",
        type=int,
        default=None,
        help="Optional random seed used for reproducibility and appended to refiner suffix when provided.",
    )

    return parser
