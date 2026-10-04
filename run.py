"""Run ORCA evaluations on chronological CSV streams."""

from __future__ import annotations

from cli import ALL_CSV_DATASETS, _expand_refiner_variants, _normalize_csv_dataset_name, _normalize_pred_len_values, _preflight_validate_and_print_plan, build_parser

if __name__ == "__main__":
    _cli_args = build_parser().parse_args()

from cli import CANONICAL_REFINERS, REFINER_CHOICES, resolve_refiner_tag
import argparse
import gc
import random
import shlex
import sys
import traceback
from pathlib import Path
from typing import List
import numpy as np
import torch
from data.download_CSV import ensure_dataset_csv, resolve_cached_csv_path
from eval.evaluator import run_csv_evaluation
from model_registry import BACKEND_COMPATIBLE_MODELS, TSFM_MODEL_ORDER, normalize_model_name, resolve_model_path
from reporting import _build_all_result_csv_paths, _build_failed_eval_record, _build_pred_len_average_records, _persist_results_at_end, _persist_single_record, _refresh_refiner_summary_csv, _resume_load_existing_summary_records, _write_speed_table


def _resolve_csv_path(args, dataset_name: str | None = None) -> Path:
    if args.csv_path:
        return Path(args.csv_path).expanduser().resolve()

    resolved_dataset = str(dataset_name or getattr(args, "dataset", "") or "").strip()
    if not resolved_dataset:
        raise ValueError("Either --csv_path or --dataset must be provided")

    cache_dir = Path(args.cache_dir).expanduser()
    csv_path = resolve_cached_csv_path(resolved_dataset, cache_dir=cache_dir)
    if csv_path.exists() and not (args.auto_download and args.force_download):
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
    run_args.tsfm_local_path = (
        str(Path(base_args.tsfm_local_path).expanduser().resolve())
        if base_args.tsfm_local_path else str(resolve_model_path(run_args.model, prefix=run_args.tsfm_model_prefix))
    )
    return run_args


def _prepare_args_for_dataset(base_args, dataset_name: str):
    run_args = argparse.Namespace(**vars(base_args))
    run_args.dataset = _normalize_csv_dataset_name(dataset_name)
    run_args.csv_path = None

    csv_path = _resolve_csv_path(run_args, dataset_name=run_args.dataset)
    run_args.csv_path = str(csv_path)

    return run_args


def _execute_configuration(run_args, device, *, dataset_label, model_short_name, refiner, variant):
    metadata = {"dataset_label": dataset_label,
                "dataset_result_name": Path(run_args.csv_path).stem,
                "model_short_name": model_short_name, "refiner": refiner,
                "pred_len": int(run_args.pred_len)}
    metadata.update({key: value for key, value in variant.items() if key != "refiner"})
    try:
        result = run_csv_evaluation(args=run_args, device=device)
    except Exception as exc:
        print(f"[ORCA][RunError] model={model_short_name} dataset={dataset_label} "
              f"pred_len={run_args.pred_len}: {type(exc).__name__}: {exc}", flush=True)
        traceback.print_exc()
        result = _build_failed_eval_record(
            dataset_label=dataset_label, model_short_name=model_short_name, refiner=refiner,
            refiner_tag=variant["refiner_tag"], variant_suffix=variant["variant_suffix"],
            training_method=variant.get("training_method"), refiner_input=variant.get("refiner_input"),
            update_rule=variant.get("update_rule"), online_buffer_windows=variant.get("online_buffer_windows"),
            router=variant.get("router"), routing_temperature=variant.get("routing_temperature"),
            ema_error_momentum=variant.get("ema_error_momentum"), pred_len=int(run_args.pred_len),
        )
        result["error"] = f"{type(exc).__name__}: {exc}"
    result.update(metadata)
    return result


def _raise_for_failed_runs(records):
    failed = [r for r in records if r.get("error")]
    if failed:
        print(f"[ORCA] {len(failed)} evaluation(s) failed:", file=sys.stderr)
        for record in failed:
            print(f"  {record['model_short_name']}/{record['dataset_label']}/"
                  f"H{record['pred_len']}/{record['refiner_tag']}: {record['error']}", file=sys.stderr)
        raise SystemExit(1)


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
                        f"[ORCA][Skip] model={model_short_name} is registered but not yet supported by "
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
                if variant.get("router") is not None:
                    model_args.router = variant.get("router")
                if variant.get("routing_temperature") is not None:
                    model_args.routing_temperature = float(variant.get("routing_temperature"))
                if variant.get("ema_error_momentum") is not None:
                    model_args.ema_error_momentum = float(variant.get("ema_error_momentum"))
                print(
                    f"\n[ORCA][All] Running refiner={refiner_tag}({variant_suffix or 'default'}) model={model_short_name} "
                    f"| local_path={model_args.tsfm_local_path} | datasets={dataset_names}"
                )

                for dataset_name in dataset_names:
                    for pred_len in pred_len_values:
                        print(f"\n[ORCA][All] Running dataset={dataset_name} | pred_len={pred_len} ...")
                        run_args = _prepare_args_for_dataset(model_args, dataset_name)
                        run_args.pred_len = int(pred_len)

                        if resume_enabled:
                            key = (str(dataset_name), str(model_short_name))
                            if key in resume_completed_keys_by_pred.get(int(pred_len), set()):
                                print(
                                    f"[ORCA][Resume][Skip] refiner={refiner_tag}({variant_suffix or 'default'}) "
                                    f"model={model_short_name} dataset={dataset_name} pred_len={pred_len} has complete non-NaN summary entry.",
                                    flush=True,
                                )
                                del run_args
                                gc.collect()
                                if torch.cuda.is_available():
                                    torch.cuda.empty_cache()
                                continue

                        eval_out = _execute_configuration(
                            run_args, device, dataset_label=dataset_name, model_short_name=model_short_name,
                            refiner=refiner, variant=variant,
                        )

                        records.append(eval_out)
                        refiner_records_by_pred[int(pred_len)].append(eval_out)
                        if speed_mode:
                            speed_key = (str(dataset_name), str(model_short_name), int(pred_len))
                            speed_tables.setdefault(speed_key, {})[refiner_tag] = dict(eval_out.get("speed_stats", {}))
                            refiner_cols = [resolve_refiner_tag(r) for r in refiner_names]
                            out_path = _write_speed_table(
                                dataset_name=str(dataset_name),
                                model_name=str(model_short_name),
                                pred_len=int(pred_len),
                                refiner_tags=refiner_cols,
                                stats_by_refiner=speed_tables[speed_key],
                            )
                            print(f"[ORCA] Speed table written: {out_path}")
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
                            print(f"[ORCA] Refiner MAE summary updated: {out_paths['mae']}")
                            print(f"[ORCA] Refiner MSE summary updated: {out_paths['mse']}")

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
                                print(f"[ORCA] Pred-len average MAE summary updated: {avg_paths['mae']}")
                                print(f"[ORCA] Pred-len average MSE summary updated: {avg_paths['mse']}")

                        del run_args
                        del eval_out
                        gc.collect()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()

    if not speed_mode:
        _persist_results_at_end(records, args)
    _raise_for_failed_runs(records)


def main(args=None) -> None:
    if args is None:
        args = build_parser().parse_args()
    if getattr(args, "random_seed", None) is not None:
        seed_val = int(getattr(args, "random_seed"))
        random.seed(seed_val)
        np.random.seed(seed_val)
        torch.manual_seed(seed_val)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed_val)
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
        refiner_tokens = ["orca_no_decay"]
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
    print(f"[ORCA] pred_len_values={pred_len_values}")

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
                            f"[ORCA][Skip] model={model_short_name} is registered but not yet supported by "
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
                    if variant.get("router") is not None:
                        run_args.router = variant.get("router")
                    if variant.get("routing_temperature") is not None:
                        run_args.routing_temperature = float(variant.get("routing_temperature"))
                    if variant.get("ema_error_momentum") is not None:
                        run_args.ema_error_momentum = float(variant.get("ema_error_momentum"))

                    for pred_len in pred_len_values:
                        if resume_enabled:
                            key = (str(Path(run_args.csv_path).stem), str(model_short_name))
                            if key in resume_completed_keys_by_pred.get(int(pred_len), set()):
                                print(
                                    f"[ORCA][Resume][Skip] refiner={refiner_tag}({variant_suffix or 'default'}) "
                                    f"model={model_short_name} dataset={Path(run_args.csv_path).stem} pred_len={pred_len} has complete non-NaN summary entry.",
                                    flush=True,
                                )
                                continue
                        run_args.pred_len = int(pred_len)
                        print(
                            f"[ORCA] Running refiner={refiner_tag}({variant_suffix or 'default'}) model={run_args.model} "
                            f"| local_path={run_args.tsfm_local_path} | dataset={Path(run_args.csv_path).stem} | pred_len={pred_len}"
                        )
                        eval_out = _execute_configuration(
                            run_args, device, dataset_label=Path(run_args.csv_path).stem, model_short_name=model_short_name,
                            refiner=refiner, variant=variant,
                        )

                        records.append(eval_out)
                        refiner_records_by_pred[int(pred_len)].append(eval_out)
                        if speed_mode:
                            speed_key = (str(Path(run_args.csv_path).stem), str(model_short_name), int(pred_len))
                            speed_tables.setdefault(speed_key, {})[refiner_tag] = dict(eval_out.get("speed_stats", {}))
                            refiner_cols = [resolve_refiner_tag(r) for r in selected_refiners]
                            out_path = _write_speed_table(
                                dataset_name=str(Path(run_args.csv_path).stem),
                                model_name=str(model_short_name),
                                pred_len=int(pred_len),
                                refiner_tags=refiner_cols,
                                stats_by_refiner=speed_tables[speed_key],
                            )
                            print(f"[ORCA] Speed table written: {out_path}")
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
                            print(f"[ORCA] Refiner MAE summary updated: {out_paths['mae']}")
                            print(f"[ORCA] Refiner MSE summary updated: {out_paths['mse']}")

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
                                print(f"[ORCA] Pred-len average MAE summary updated: {avg_paths['mae']}")
                                print(f"[ORCA] Pred-len average MSE summary updated: {avg_paths['mse']}")

                        del eval_out
                        gc.collect()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()

                    del run_args

        if not speed_mode:
            _persist_results_at_end(records, args)
        _raise_for_failed_runs(records)
        return

    _run_all_csv_matrix(
        args=args,
        device=device,
        dataset_names=selected_datasets,
        model_names=selected_models,
        refiner_names=selected_refiners,
    )


if __name__ == "__main__":
    main(_cli_args)
