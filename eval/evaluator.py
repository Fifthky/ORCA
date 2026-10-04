from __future__ import annotations
from cli import resolve_refiner_tag
import logging
import time
import warnings
import csv
from pathlib import Path
import numpy as np
import torch
from core.util.refiner_util import select_quantile_index
from data.csv_dataset import CsvSeriesDataset
from data.data_provider import compute_window_and_update_steps_for_test_data, filter_test_data_by_context_length, split_window_counts, slice_filtered_test_data
from eval.eval_util import build_progress_line, first_value, format_duration_dhms
from eval.online_training import OnlineRefinerPredictor
from eval.builders import _build_predictor, _build_refiner, _coerce_args_int, _resolve_online_buffer_windows, _resolve_refiner, _resolve_training_method
from eval.inference_cache import _CachedArrayPredictor, _build_infer_cache_path, _forecast_to_cache_record, _load_infer_cache, _resolve_infer_cache_enabled, _save_infer_cache
from eval.metrics import PRIMARY_METRIC_KEY_1, PRIMARY_METRIC_KEY_2, PRIMARY_METRIC_LABEL_1, PRIMARY_METRIC_LABEL_2, _align_gt_pred_windows, _audit_window_alignment, _build_gt_windows_from_labels, _compose_window_audit_rows, _compute_point_primary_metrics, _compute_train_channel_mean_abs_scale, _compute_train_global_mean_abs_scale, _entry_forecast_start, _forecast_samples_to_mean_window, _forecast_to_sample_array, _inject_scaled_primary_metrics, _inject_scaled_primary_metrics_channelwise, _project_core_metric_keys, _sanitize_forecast_for_eval, _to_static_mean_scale_tensor, _use_electricity_channelwise_norm


logging.getLogger("gluonts.model.forecast").setLevel(logging.ERROR)


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
        print(f"[ORCA] Dataset cache hit: {Path(key[0]).name}", flush=True)
        return cached

    ds = CsvSeriesDataset(
        csv_path=key[0],
        prediction_length=int(key[1]),
        target_column=key[2],
        windows=key[3],
    )
    _CSV_DATASET_CACHE[key] = ds
    print(f"[ORCA] Dataset cache miss -> loaded: {Path(key[0]).name}", flush=True)
    return ds


def _save_gate_confidence_csv(
    *,
    model_name: str,
    dataset_name: str,
    pred_len: int,
    time_index: int,
    gate_confidence: np.ndarray,
) -> Path:
    safe_model = str(model_name).replace("/", "_").replace(" ", "_")
    safe_dataset = str(dataset_name).replace("/", "_").replace(" ", "_")
    file_name = f"gate_confidence_orca_{safe_model}_{safe_dataset}_pred{int(pred_len)}.csv"
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


def run_csv_evaluation(args, device: torch.device) -> dict:
    ds = _get_or_create_csv_dataset(args)

    channel_names = list(getattr(ds, "selected_columns", list(ds.dataframe.columns)))
    if not channel_names:
        raise ValueError("No numeric channels found in CSV dataset")

    print(f"[ORCA] CSV={Path(args.csv_path).name} | pred_len={ds.prediction_length} | channels={len(channel_names)}")
    print(
        "[ORCA] Evaluation mode: multivariate windows (shared slicing), "
        "single-channel backends use sequential channel packing"
    )
    print(f"[ORCA] Batch config: batch_size={int(args.batch_size)}")
    dataset_name = Path(args.csv_path).stem
    large_data_channel = _is_large_data_channel(dataset_name=dataset_name, pred_len=int(ds.prediction_length))
    if large_data_channel:
        print(
            "[ORCA] Large-data channel ON: dataset in {Traffic, Electricity} and pred_len > 100.",
            flush=True,
        )


    stride = 1
    refined_windows_req = int(ds.windows) * int(ds.prediction_length)
    refined_update_test_data_raw, refined_windows = ds.build_test_data(distance=int(stride), windows=refined_windows_req)

    min_context_length = _coerce_args_int(getattr(args, "context_length", 0), 0)
    refined_update_test_data = filter_test_data_by_context_length(refined_update_test_data_raw, min_context_length)

    if len(refined_update_test_data.input) == 0:
        raise ValueError(
            f"No refined-update windows left after context-length filter: context_length={min_context_length}"
        )
    total_update_windows = int(len(refined_update_test_data.input))

    order_diag = _check_instance_order_monotonic(refined_update_test_data.input)
    if int(order_diag["comparable_pairs"]) > 0 and int(order_diag["non_increasing_pairs"]) > 0:
        raise RuntimeError(
            "Detected non-monotonic GluonTS instance order in refined update stream. "
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
    if training_method == "online" and refiner_key in {"orca_no_decay", "orca", "dsof", "tafas", "solid"}:
        # Online retrain trigger uses stride-1 mini-window buffer and is independent of eval split.
        refiner_train_window_count = min(max(1, int(online_buffer_windows)), int(total_update_windows))
        refiner_val_window_count = 1
    test_start = int(eval_train_window_count + eval_val_window_count)
    test_end = int(test_start + eval_test_window_count)

    # Single-stream execution for updates; metrics are computed on test split only.
    baseline_eval_data = slice_filtered_test_data(refined_update_test_data, start=test_start, end=test_end)
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
        refined_update_test_data,
        start=0,
        end=int(eval_train_window_count),
    )
    scale_entries = train_scale_partition.input if len(train_scale_partition.input) > 0 else refined_update_test_data.input
    use_channelwise_norm = _use_electricity_channelwise_norm(dataset_name)
    train_global_mean_scale = _compute_train_global_mean_abs_scale(scale_entries)
    train_channel_mean_scale = _compute_train_channel_mean_abs_scale(scale_entries) if use_channelwise_norm else None

    window_count_est, update_steps_est, stream_count = compute_window_and_update_steps_for_test_data(
        refined_update_test_data,
        pred_len=int(ds.prediction_length),
        stride=int(stride),
    )
    print(
        f"[ORCA][{Path(args.csv_path).stem}] stream_count={stream_count} | baseline_windows={ds.windows} | refined_windows={refined_windows} | "
        f"window_count_est={window_count_est} | refiner_update_steps_est={update_steps_est} | "
        f"context_filter={min_context_length} | baseline_eval_windows={len(baseline_eval_data.input)} | "
        f"refined_eval_windows={len(refined_update_test_data.input)} | target_dim={ds.target_dim}"
    )
    print(
        f"[ORCA] Window split (meta windows, for evaluation): total={total_baseline_windows} | "
        f"train={train_meta_window_count} | val={val_meta_window_count} | test={test_meta_window_count} | eval_test_partition={len(baseline_eval_data.input)}"
    )
    print(
        f"[ORCA] Window split (mini windows, for online updates): total={total_update_windows} | "
        f"train={eval_train_window_count} | val={eval_val_window_count} | test={eval_test_window_count}"
    )
    if training_method == "online" and refiner_key in {"orca_no_decay", "orca", "dsof", "tafas", "solid"}:
        print(
            f"[ORCA] online_buffer_windows={int(online_buffer_windows)} (unit=stride-1 mini windows)",
            flush=True,
        )
        print(
            f"[ORCA] Refiner retrain trigger windows: train_buffer={int(refiner_train_window_count)} | val_marker={int(refiner_val_window_count)}",
            flush=True,
        )
    print(
        "[ORCA] Single-stream mode active: refiner runs on full stride=1 stream; metrics/logs use test split only.",
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
            f"[ORCA] Refiner static scaler ({scale_mode} mean abs) ready: channels={int(static_mean_scale.shape[-1])} | preview=[{preview}]"
        )

    cache_requested = _resolve_infer_cache_enabled(args)
    refiner_for_eval = _resolve_refiner(args)
    cache_enabled = bool(cache_requested)
    cache_path = _build_infer_cache_path(args, Path(args.csv_path).stem) if cache_enabled else None
    cache_load_t0 = time.perf_counter()
    if cache_path is not None:
        print(f"[ORCA] Cache load stage start: path={cache_path}", flush=True)
    cache_payload = _load_infer_cache(cache_path) if cache_path is not None else None
    cache_load_dt = max(1e-9, float(time.perf_counter() - cache_load_t0))
    if cache_path is not None:
        print(
            f"[ORCA] Cache load stage done: path={cache_path} | elapsed={cache_load_dt:.2f}s",
            flush=True,
        )
    cached_dense_records = [] if cache_payload is None else list(cache_payload.get("dense_records", []))
    dense_needed = int(len(refined_update_test_data.input))
    dense_cache_hit = len(cached_dense_records) >= dense_needed
    use_dense_cache_pipeline = bool(cache_enabled)
    cache_build_oom_retry = bool(cache_enabled and (not dense_cache_hit))
    if bool(getattr(args, "speed", False)):
        cache_build_oom_retry = False
    if cache_enabled:
        if dense_cache_hit:
            print(f"[ORCA] Inference cache hit: {cache_path}")
        else:
            print(f"[ORCA] Inference cache miss: {cache_path}")

    base_predictor = _build_predictor(ds, args, device)
    base_model_single_infer_time = float("nan")
    if bool(getattr(args, "speed", False)):
        print("[ORCA] Speed mode: base-model 10-step re-inference benchmark start.", flush=True)
        base_model_single_infer_time = _measure_base_model_single_inference_seconds(
            base_predictor,
            refined_update_test_data.input,
            steps=10,
            batch_size=1,
        )
        if np.isfinite(base_model_single_infer_time):
            print(
                f"[ORCA] Speed mode: base-model avg single inference={base_model_single_infer_time * 1000.0:.4f} ms over 10 steps.",
                flush=True,
            )
        else:
            print("[ORCA] Speed mode: base-model 10-step benchmark unavailable (NaN).", flush=True)
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
                f"[ORCA] Building dense inference cache payload: total_dense_windows={dense_needed} (stride=1)",
                flush=True,
            )
            if large_data_channel:
                print(
                    "[ORCA] Large-data inference path active: keep high-throughput batch predict + progress/ETA logging.",
                    flush=True,
                )
            dense_batch_size = max(1, int(args.batch_size))
            dense_dt = 0.0
            while True:
                try:
                    dense_t0 = time.perf_counter()
                    for _ in dense_record_predictor.predict(refined_update_test_data.input, batch_size=dense_batch_size):
                        pass
                    dense_dt = max(1e-9, float(time.perf_counter() - dense_t0))
                    dense_records_full = list(getattr(dense_record_predictor, "raw_records", []))
                    break
                except Exception as exc:
                    if not (cache_build_oom_retry and _is_oom_exception(exc)):
                        raise
                    if dense_batch_size <= 1:
                        raise RuntimeError(
                            "[ORCA] Cache build OOM fallback exhausted at batch_size=1."
                        ) from exc
                    new_batch_size = max(1, int(dense_batch_size) // 4)
                    print(
                        f"[ORCA][InferBatch][cache-build] OOM fallback: batch_size {dense_batch_size} -> {new_batch_size}",
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
                f"[ORCA] Dense inference finished: collected={len(dense_records_full)} | elapsed={format_duration_dhms(dense_dt)} | throughput={len(dense_records_full)/dense_dt:.2f}it/s",
                flush=True,
            )
            if cache_enabled and cache_path is not None:
                if large_data_channel:
                    print(
                        "[ORCA] Large-data save path active: v3 memmap stream with large blocks.",
                        flush=True,
                    )
                _save_infer_cache(
                    cache_path,
                    dense_records=dense_records_full,
                    large_data_channel=bool(large_data_channel),
                )
                print(f"[ORCA] Inference cache saved: {cache_path}")
                print(
                    "[ORCA] Cache ready: continuing with refiner evaluation.",
                    flush=True,
                )

    if use_dense_cache_pipeline:
        baseline_eval_samples = list(dense_records_full[: int(len(refined_update_test_data.input))])
        if len(baseline_eval_samples) < int(len(refined_update_test_data.input)):
            raise ValueError(
                f"Dense inference outputs are insufficient for baseline evaluation: expected={len(refined_update_test_data.input)}, got={len(baseline_eval_samples)}"
            )
        baseline_predictor = _CachedArrayPredictor(baseline_eval_samples)
    else:
        baseline_predictor = base_predictor

    print(
        f"[ORCA] Baseline evaluate stage start: full_windows={len(refined_update_test_data.input)} | test_windows={len(baseline_eval_data.input)}",
        flush=True,
    )
    baseline_eval_t0 = time.perf_counter()
    baseline_recording_predictor = _RecordingPredictor(
        baseline_predictor,
        prediction_length=int(ds.prediction_length),
        target_dim=int(ds.target_dim),
    )
    for _ in baseline_recording_predictor.predict(refined_update_test_data.input, batch_size=args.batch_size):
        pass
    baseline_eval_dt = max(1e-9, float(time.perf_counter() - baseline_eval_t0))
    print(
        f"[ORCA] Baseline evaluate stage done: elapsed={baseline_eval_dt:.2f}s",
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


    refined_pred_windows_metric: list[np.ndarray] = []

    if isinstance(update_steps_est, int) and update_steps_est <= 0:
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
        agg_metrics_refined = dict(agg_metrics_base)
        final_window_count = int(window_count_est)
        final_update_steps = 0
        loss_history = []
        val_loss_history = []
        refined_pred_windows_metric = [w.copy() for w in baseline_pred_windows_metric]
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
            predict_batch_size=int(args.batch_size),
            static_mean_scale=static_mean_scale,
            buffered_update_records=(dense_records_full if use_dense_cache_pipeline else None),
            speed_mode=bool(getattr(args, "speed", False)),
        )
        predictor = _RecordingPredictor(
            predictor_raw,
            prediction_length=int(ds.prediction_length),
            target_dim=int(ds.target_dim),
        )
        for _ in predictor.predict(refined_update_test_data.input, batch_size=args.batch_size):
            pass

        if bool(getattr(args, "speed", False)):
            speed_stats = dict(getattr(predictor_raw, "get_speed_stats", lambda: {})())
            speed_stats["base_model_infer_time"] = float(base_model_single_infer_time)
            if refiner_key == "orca" and "refiner" in locals():
                last_gate = getattr(refiner, "last_gate_confidence", None)
                last_time = getattr(refiner, "last_gate_time_index", None)
                if last_gate is not None and last_time is not None:
                    gate_path = _save_gate_confidence_csv(
                        model_name=str(getattr(args, "model", "unknown")),
                        dataset_name=str(dataset_name),
                        pred_len=int(ds.prediction_length),
                        time_index=int(last_time),
                        gate_confidence=last_gate,
                    )
                    print(f"[ORCA] ORCA gate confidence saved: {gate_path}")

            refiner_tag = resolve_refiner_tag(refiner_key)
            print(
                f"[ORCA] Speed benchmark finished: infer_steps={int(getattr(predictor_raw, '_speed_infer_steps_recorded', 0))} | "
                f"train_calls={int(getattr(predictor_raw, 'refiner_update_calls', 0))}"
            )
            return {
                "dataset_name": dataset_name,
                "model_short_name": str(getattr(args, "model", "unknown")),
                "refiner_tag": refiner_tag,
                "pred_len": int(ds.prediction_length),
                "agg_metrics_base": {},
                "agg_metrics_refined": {},
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
                "speed_stats": speed_stats,
            }

        refined_recorded_windows_all = list(getattr(predictor, "recorded_pred_windows", []))
        refined_meta_all = list(getattr(predictor, "point_meta_windows", []))
        refined_recorded_windows = refined_recorded_windows_all[int(test_start):int(test_end)]
        refined_meta_windows = refined_meta_all[int(test_start):int(test_end)]
        _, refined_pred_windows_metric = _align_gt_pred_windows(
            baseline_gt_windows_metric,
            refined_recorded_windows,
        )
        _audit_window_alignment(
            baseline_gt_windows_metric,
            refined_pred_windows_metric,
            refined_meta_windows,
        )
        refined_window_audit_rows = _compose_window_audit_rows(
            baseline_gt_windows_metric,
            refined_pred_windows_metric,
            refined_meta_windows,
        )
        _ = baseline_window_audit_rows
        _ = refined_window_audit_rows
        refined_point_metrics = _compute_point_primary_metrics(
            baseline_gt_windows_metric,
            refined_pred_windows_metric,
        )

        if use_channelwise_norm:
            agg_metrics_base = _inject_scaled_primary_metrics_channelwise(
                base_point_metrics,
                gt_windows=baseline_gt_windows_metric,
                pred_windows=baseline_pred_windows_metric,
                channel_mean_abs_scale=train_channel_mean_scale,
            )
            agg_metrics_refined = _inject_scaled_primary_metrics_channelwise(
                refined_point_metrics,
                gt_windows=baseline_gt_windows_metric,
                pred_windows=refined_pred_windows_metric,
                channel_mean_abs_scale=train_channel_mean_scale,
            )
        else:
            agg_metrics_base = _inject_scaled_primary_metrics(
                base_point_metrics,
                mean_abs_scale=train_global_mean_scale,
            )
            agg_metrics_refined = _inject_scaled_primary_metrics(
                refined_point_metrics,
                mean_abs_scale=train_global_mean_scale,
            )
        final_window_count = int(predictor_raw.window_count)
        final_update_steps = int(predictor_raw.update_steps)
        loss_history = list(getattr(refiner, "loss_history", [])) if hasattr(refiner, "loss_history") else []
        val_loss_history = list(getattr(refiner, "val_loss_history", [])) if hasattr(refiner, "val_loss_history") else []
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
    agg_metrics_refined = dict(agg_metrics_refined or {})
    agg_metrics_base = _project_core_metric_keys(agg_metrics_base)
    agg_metrics_refined = _project_core_metric_keys(agg_metrics_refined)
    if agg_metrics_base is not None:
        print(
            f"[ORCA][Baseline] {PRIMARY_METRIC_LABEL_1}={first_value(agg_metrics_base.get(PRIMARY_METRIC_KEY_1)):.4f} | "
            f"{PRIMARY_METRIC_LABEL_2}={first_value(agg_metrics_base.get(PRIMARY_METRIC_KEY_2)):.4f}"
        )
    if agg_metrics_refined is not None:
        print(
            f"[ORCA][Refined] {PRIMARY_METRIC_LABEL_1}={first_value(agg_metrics_refined.get(PRIMARY_METRIC_KEY_1)):.4f} | "
            f"{PRIMARY_METRIC_LABEL_2}={first_value(agg_metrics_refined.get(PRIMARY_METRIC_KEY_2)):.4f}"
        )
    if 'refiner' in locals() and hasattr(refiner, "consume_unit_diagnostics_once"):
        try:
            diag_line = refiner.consume_unit_diagnostics_once()
            if diag_line:
                print(diag_line)
        except Exception:
            pass

    total_update_window_count = int(final_window_count)
    total_update_steps = int(final_update_steps)
    total_eval_window_count = int(len(baseline_eval_data.input))
    collected_losses: list = list(loss_history)
    collected_val_losses: list = list(val_loss_history)

    dataset_name = Path(args.csv_path).stem
    if agg_metrics_base:
        print(
            f"[ORCA][Baseline][All Channels Avg] {PRIMARY_METRIC_LABEL_1}={first_value(agg_metrics_base.get(PRIMARY_METRIC_KEY_1)):.4f} | "
            f"{PRIMARY_METRIC_LABEL_2}={first_value(agg_metrics_base.get(PRIMARY_METRIC_KEY_2)):.4f}"
        )
    if agg_metrics_refined:
        print(
            f"[ORCA][Refined][All Channels Avg] {PRIMARY_METRIC_LABEL_1}={first_value(agg_metrics_refined.get(PRIMARY_METRIC_KEY_1)):.4f} | "
            f"{PRIMARY_METRIC_LABEL_2}={first_value(agg_metrics_refined.get(PRIMARY_METRIC_KEY_2)):.4f}"
        )

    if refiner_key == "orca" and "refiner" in locals():
        last_gate = getattr(refiner, "last_gate_confidence", None)
        last_time = getattr(refiner, "last_gate_time_index", None)
        if last_gate is not None and last_time is not None:
            gate_path = _save_gate_confidence_csv(
                model_name=str(getattr(args, "model", "unknown")),
                dataset_name=str(dataset_name),
                pred_len=int(ds.prediction_length),
                time_index=int(last_time),
                gate_confidence=last_gate,
            )
            print(f"[ORCA] ORCA gate confidence saved: {gate_path}")

    refiner_tag = resolve_refiner_tag(_resolve_refiner(args))

    print(
        f"[ORCA] Final counts: eval_windows={total_eval_window_count} | "
        f"update_windows={total_update_window_count} | update_steps={total_update_steps}"
    )

    return {
        "dataset_name": dataset_name,
        "model_short_name": str(getattr(args, "model", "unknown")),
        "refiner_tag": refiner_tag,
        "pred_len": int(ds.prediction_length),
        "agg_metrics_base": agg_metrics_base,
        "agg_metrics_refined": agg_metrics_refined,
        "window_count": int(total_update_window_count),
        "update_steps": int(total_update_steps),
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
