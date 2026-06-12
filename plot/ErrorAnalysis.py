from __future__ import annotations

import argparse
import math
import os
import sys
import traceback
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Keep BLAS/OpenMP thread count bounded before importing numpy/scikit-learn.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans

from data.csv_dataset import CSV_DATASET_SPECS, CsvSeriesDataset
from data.download_CSV import DEFAULT_CACHE_DIR, resolve_cached_csv_path
from data.data_provider import filter_test_data_by_context_length
from core.util.refiner_util import select_quantile_index
from model_registry import (
    BACKEND_COMPATIBLE_MODELS,
    TSFM_MODEL_ORDER,
    normalize_model_name,
)


ALL_CSV_DATASETS = [
    "ETTh1",
    "ETTh2",
    "ETTm1",
    "ETTm2",
    "Exchange",
    "Weather",
    "Electricity",
    "Traffic",
]


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


def _safe_model_tag(model_name: str) -> str:
    return str(model_name).replace("-", "_").replace("/", "_")


def _build_infer_cache_path(model_name: str, dataset_name: str, context_length: int, pred_len: int) -> Path:
    safe_model = _safe_model_tag(model_name)
    safe_ds = str(dataset_name).replace("/", "_")
    cache_dir = PROJECT_ROOT / "data/model_infer_cache"
    version_suffix = "npyv3"
    return cache_dir / f"{safe_model}_{safe_ds}_ctx{int(context_length)}_pred{int(pred_len)}_s1_{version_suffix}"


def _infer_cache_component_paths(cache_path: Path) -> tuple[Path, Path, Path]:
    base = str(cache_path)
    payload_path = Path(base + ".payloads.npy")
    kinds_path = Path(base + ".kinds.npy")
    qkeys_path = Path(base + ".qkeys.npy")
    return payload_path, kinds_path, qkeys_path


def _np_load_with_mmap_fallback(path: Path):
    try:
        return np.load(path, allow_pickle=True, mmap_mode="r")
    except ValueError as exc:
        # Object-dtype .npy arrays cannot be memory-mapped.
        if "can't be memory-mapped" not in str(exc).lower():
            raise
        return np.load(path, allow_pickle=True)


def _load_existing_cache_dense_records(cache_path: Path) -> list[dict]:
    payload_path, kinds_path, qkeys_path = _infer_cache_component_paths(cache_path)
    if not (payload_path.exists() and kinds_path.exists() and qkeys_path.exists()):
        raise FileNotFoundError(f"Cache files missing: {cache_path}*")

    payloads = _np_load_with_mmap_fallback(payload_path)
    kinds = _np_load_with_mmap_fallback(kinds_path)
    qkeys = _np_load_with_mmap_fallback(qkeys_path)

    if not (len(payloads) == len(kinds) == len(qkeys)):
        raise RuntimeError(f"Cache files have inconsistent lengths: {cache_path}*")

    out: list[dict] = []
    for i in range(int(len(payloads))):
        q_raw = qkeys[i]
        out.append(
            {
                "kind": str(kinds[i]),
                "payload": payloads[i],
                "forecast_keys": list(q_raw) if q_raw is not None else None,
            }
        )
    return out


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

    if arr.ndim == 1:
        return np.asarray(arr.reshape(-1, 1), dtype=np.float32)
    if arr.ndim == 2:
        return np.asarray(arr.T, dtype=np.float32)
    return np.asarray(arr.reshape(arr.shape[0], -1).T, dtype=np.float32)


def _entry_to_channel_time(entry) -> np.ndarray:
    main = entry[0] if isinstance(entry, tuple) else entry
    if isinstance(main, dict) and "past_target" in main:
        arr = np.asarray(main["past_target"], dtype=np.float32)
    else:
        arr = np.asarray(main["target"], dtype=np.float32)

    if arr.ndim == 1:
        return np.asarray(arr.reshape(1, -1), dtype=np.float32)
    if arr.ndim == 2:
        return np.asarray(arr, dtype=np.float32)
    return np.asarray(arr.reshape(arr.shape[0], -1), dtype=np.float32)


def _fixed_context_tail(channel_time: np.ndarray, context_length: int) -> np.ndarray:
    x = np.asarray(channel_time, dtype=np.float32)
    if x.ndim != 2:
        x = np.asarray(x).reshape(1, -1).astype(np.float32)
    d, t = int(x.shape[0]), int(x.shape[1])
    c = int(context_length)
    if t >= c:
        return np.asarray(x[:, (t - c):t], dtype=np.float32)
    pad = np.zeros((d, c - t), dtype=np.float32)
    return np.concatenate([pad, x], axis=1)


def _sanitize_flat(arr: np.ndarray) -> np.ndarray:
    x = np.asarray(arr, dtype=np.float32).reshape(-1)
    if x.size == 0:
        return x
    finite = np.isfinite(x)
    if bool(np.all(finite)):
        return x
    if bool(np.any(finite)):
        fill = float(np.median(x[finite]))
    else:
        fill = 0.0
    return np.nan_to_num(x, nan=fill, posinf=fill, neginf=fill).astype(np.float32)


def _windows_mae(gt_windows: list[np.ndarray], pred_windows: list[np.ndarray]) -> np.ndarray:
    n = min(len(gt_windows), len(pred_windows))
    errors = np.zeros((n,), dtype=np.float32)
    for i in range(n):
        g = np.asarray(gt_windows[i], dtype=np.float32)
        p = np.asarray(pred_windows[i], dtype=np.float32)
        if g.ndim == 1:
            g = g.reshape(-1, 1)
        if p.ndim == 1:
            p = p.reshape(-1, 1)
        h = min(int(g.shape[0]), int(p.shape[0]))
        d = min(int(g.shape[1]), int(p.shape[1]))
        if h <= 0 or d <= 0:
            errors[i] = np.nan
            continue
        diff = np.abs(g[:h, :d] - p[:h, :d])
        errors[i] = float(np.mean(diff))

    finite = np.isfinite(errors)
    if not bool(np.all(finite)):
        if bool(np.any(finite)):
            fill = float(np.nanmedian(errors[finite]))
        else:
            fill = 0.0
        errors = np.nan_to_num(errors, nan=fill, posinf=fill, neginf=fill)
    return np.asarray(errors, dtype=np.float32)


def _cluster_labels(data: np.ndarray, n_clusters: int, tag: str) -> tuple[np.ndarray, int]:
    x = np.asarray(data, dtype=np.float32)
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    n = int(x.shape[0])
    if n <= 0:
        raise ValueError(f"No samples for clustering: {tag}")

    k = int(n_clusters)
    if k <= 0:
        raise ValueError(f"n_clusters must be positive, got {n_clusters}")
    if k > n:
        print(f"[ErrorAnalysis][Warn] {tag}: requested clusters={k}, samples={n}. Falling back to clusters={n}.")
        k = n

    model = KMeans(n_clusters=k, random_state=42, n_init="auto")
    labels = model.fit_predict(x)
    return np.asarray(labels, dtype=np.int32), k


def _resolve_dataset_csv_path(dataset_name: str) -> Path:
    cache_dir = Path(DEFAULT_CACHE_DIR).expanduser()
    if not cache_dir.is_absolute():
        cache_dir = PROJECT_ROOT / cache_dir
    csv_path = resolve_cached_csv_path(dataset_name, cache_dir=cache_dir)
    if not csv_path.exists():
        raise FileNotFoundError(
            "Dataset CSV cache not found. This script does not download automatically. "
            f"Please prepare cache first: {csv_path}"
        )
    return csv_path.resolve()


def _build_flow_update_stream(csv_path: Path, pred_len: int, context_length: int):
    ds = CsvSeriesDataset(
        csv_path=str(csv_path),
        prediction_length=int(pred_len),
        target_column="all",
        windows=None,
    )

    # Use non-overlapping windows by default (stride = prediction length)
    stride = int(pred_len)
    flow_windows_req = int(ds.windows) * int(ds.prediction_length)
    flow_update_test_data_raw, _ = ds.build_test_data(distance=int(stride), windows=flow_windows_req)
    flow_update_test_data = filter_test_data_by_context_length(flow_update_test_data_raw, int(context_length))

    if len(flow_update_test_data.input) <= 0:
        raise ValueError(
            "No windows left after context filter. "
            f"context_length={context_length}, csv={csv_path.name}"
        )

    return ds, flow_update_test_data


def _to_point_window_from_record_payload(arr: np.ndarray, *, forecast_keys: list[str] | None, kind: str) -> np.ndarray:
    x = np.asarray(arr, dtype=np.float32)

    # Already point forecast window.
    if x.ndim == 1:
        return x.reshape(-1, 1).astype(np.float32)

    if x.ndim == 2:
        q_idx = select_quantile_index(forecast_keys, int(x.shape[0]), target_quantile=0.5)
        if q_idx is not None:
            return x[int(q_idx)].reshape(-1, 1).astype(np.float32)
        if str(kind).lower() == "sample":
            return np.median(x, axis=0).reshape(-1, 1).astype(np.float32)
        # Fallback: if quantile keys are missing but this is quantile-like, use middle index.
        mid = int(x.shape[0] // 2)
        return x[mid].reshape(-1, 1).astype(np.float32)

    if x.ndim == 3:
        q_idx = select_quantile_index(forecast_keys, int(x.shape[0]), target_quantile=0.5)
        if q_idx is not None:
            point = np.asarray(x[int(q_idx)], dtype=np.float32)
        elif str(kind).lower() == "sample":
            point = np.asarray(np.median(x, axis=0), dtype=np.float32)
        else:
            mid = int(x.shape[0] // 2)
            point = np.asarray(x[mid], dtype=np.float32)
        if point.ndim == 1:
            return point.reshape(-1, 1).astype(np.float32)
        return point.astype(np.float32)

    # Generic fallback for unexpected ranks.
    point = np.asarray(np.median(x, axis=0), dtype=np.float32)
    if point.ndim == 1:
        return point.reshape(-1, 1).astype(np.float32)
    return point.astype(np.float32)


def _records_to_pred_windows(records: list[dict], prediction_length: int, target_dim: int) -> list[np.ndarray]:
    _ = prediction_length
    _ = target_dim
    out: list[np.ndarray] = []
    for rec in records:
        arr = np.asarray(rec.get("payload"), dtype=np.float32)
        keys_raw = rec.get("forecast_keys")
        forecast_keys = list(map(str, keys_raw)) if keys_raw is not None else None
        kind = str(rec.get("kind", "")).strip().lower()
        out.append(
            _to_point_window_from_record_payload(
                arr,
                forecast_keys=forecast_keys,
                kind=kind,
            )
        )
    return out


def _build_y_feature_vector(
    y_mode: str,
    x_ctx: np.ndarray,
    y_pred: np.ndarray,
    prev_error: float,
) -> np.ndarray:
    x_flat = _sanitize_flat(x_ctx)
    y_flat = _sanitize_flat(y_pred)
    e_flat = np.asarray([float(prev_error)], dtype=np.float32)

    if y_mode == "X":
        return x_flat
    if y_mode == "Y":
        return y_flat
    if y_mode == "XY":
        return np.concatenate([x_flat, y_flat], axis=0)
    if y_mode == "E":
        return e_flat
    if y_mode == "ALL":
        return np.concatenate([x_flat, y_flat, e_flat], axis=0)
    raise ValueError(f"Unsupported y_mode={y_mode!r}")


def _stack_feature_vectors(vectors: list[np.ndarray]) -> np.ndarray:
    if not vectors:
        raise ValueError("No feature vectors")
    length = int(vectors[0].shape[0])
    for i, v in enumerate(vectors):
        if int(v.shape[0]) != length:
            raise ValueError(
                f"Feature length mismatch at index {i}: {v.shape[0]} vs {length}. "
                "Please verify context_length/pred_len consistency."
            )
    return np.stack(vectors, axis=0).astype(np.float32)


def _plot_scatter_and_save(
    *,
    x_labels: np.ndarray,
    y_labels: np.ndarray,
    x_k: int,
    y_k: int,
    model_name: str,
    dataset_name: str,
    pred_len: int,
    context_length: int,
    y_mode: str,
    out_dir: Path,
) -> Path:
    n = int(min(len(x_labels), len(y_labels)))
    x = np.asarray(x_labels[:n], dtype=np.int32)
    y = np.asarray(y_labels[:n], dtype=np.int32)

    fig = plt.figure(figsize=(8, 7), dpi=140)
    ax = fig.add_subplot(111)

    # Build 2D histogram counts where x axis = error cluster, y axis = input cluster
    counts = np.zeros((int(y_k), int(x_k)), dtype=np.int32)
    for i in range(n):
        xi = int(x[i])
        yi = int(y[i])
        if 0 <= xi < int(x_k) and 0 <= yi < int(y_k):
            counts[yi, xi] += 1

    # Use a perceptually-meaningful colormap where low counts are light and high counts are darker.
    # Display as a grid of squares (no interpolation) with origin at lower-left.
    im = ax.imshow(
        counts,
        origin="lower",
        cmap="Blues",
        aspect="auto",
        interpolation="nearest",
        extent=(-0.5, float(x_k) - 0.5, -0.5, float(y_k) - 0.5),
    )
    ax.set_xlabel("Error Cluster (current window)")
    ax.set_ylabel(f"Input Cluster ({y_mode})")
    ax.set_title(
        f"Error vs Input Cluster\\nmodel={model_name} | dataset={dataset_name} | pred={pred_len} | ctx={context_length} | y={y_mode}"
    )
    ax.set_xticks(list(range(int(x_k))))
    ax.set_yticks(list(range(int(y_k))))
    # Draw thin grid lines between cells to emphasize tile boundaries but keep them subtle.
    ax.set_xticks([i - 0.5 for i in range(1, int(x_k))], minor=True)
    ax.set_yticks([i - 0.5 for i in range(1, int(y_k))], minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=0.8, alpha=0.6)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("Window Count (log-scaled)")

    out_dir.mkdir(parents=True, exist_ok=True)
    safe_model = _safe_model_tag(model_name)
    safe_ds = str(dataset_name).replace("/", "_")
    fname = (
        f"ErrorAnalysis_{safe_ds}_{safe_model}"
        f"_pred{int(pred_len)}_ctx{int(context_length)}"
        f"_xk{int(x_k)}_yk{int(y_k)}_y{str(y_mode)}.png"
    )
    out_path = out_dir / fname
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def _run_one(
    *,
    model_name: str,
    dataset_name: str,
    pred_len: int,
    context_length: int,
    x_classes: int,
    y_classes: int,
    y_mode: str,
) -> Path:
    csv_path = _resolve_dataset_csv_path(dataset_name)
    ds, flow_update_test_data = _build_flow_update_stream(csv_path, pred_len=pred_len, context_length=context_length)
    dense_needed = int(len(flow_update_test_data.input))

    cache_path = _build_infer_cache_path(
        model_name=model_name,
        dataset_name=dataset_name,
        context_length=int(context_length),
        pred_len=int(pred_len),
    )
    dense_records = _load_existing_cache_dense_records(cache_path)
    if len(dense_records) < dense_needed:
        raise RuntimeError(
            "Existing cache does not cover required windows. "
            f"cache_records={len(dense_records)}, required_windows={dense_needed}, cache={cache_path}"
        )

    used_records = list(dense_records[:dense_needed])
    pred_windows = _records_to_pred_windows(
        used_records,
        prediction_length=int(ds.prediction_length),
        target_dim=int(ds.target_dim),
    )

    gt_windows = [_extract_label_target(x) for x in flow_update_test_data.label]
    n = min(len(flow_update_test_data.input), len(pred_windows), len(gt_windows))
    if n <= 0:
        raise RuntimeError("No aligned windows for analysis")

    x_entries = list(flow_update_test_data.input[:n])
    pred_windows = list(pred_windows[:n])
    gt_windows = list(gt_windows[:n])

    errors = _windows_mae(gt_windows, pred_windows)
    prev_errors = np.zeros_like(errors)
    if int(errors.shape[0]) > 1:
        prev_errors[1:] = errors[:-1]

    y_vectors: list[np.ndarray] = []
    for i in range(n):
        x_raw = _entry_to_channel_time(x_entries[i])
        x_ctx = _fixed_context_tail(x_raw, int(context_length))
        y_pred = np.asarray(pred_windows[i], dtype=np.float32)
        y_vec = _build_y_feature_vector(
            y_mode=str(y_mode),
            x_ctx=x_ctx,
            y_pred=y_pred,
            prev_error=float(prev_errors[i]),
        )
        y_vectors.append(y_vec)

    y_matrix = _stack_feature_vectors(y_vectors)
    x_error_labels, x_k = _cluster_labels(errors.reshape(-1, 1), int(x_classes), tag="error")
    y_input_labels, y_k = _cluster_labels(y_matrix, int(y_classes), tag=f"input_{y_mode}")

    out_path = _plot_scatter_and_save(
        x_labels=x_error_labels,
        y_labels=y_input_labels,
        x_k=int(x_k),
        y_k=int(y_k),
        model_name=model_name,
        dataset_name=dataset_name,
        pred_len=int(pred_len),
        context_length=int(context_length),
        y_mode=str(y_mode),
        out_dir=PROJECT_ROOT / "results/plots/ErrorAnalysis",
    )
    return out_path


def _resolve_models(tokens: list[str]) -> list[str]:
    raw = [str(x).strip() for x in tokens if str(x).strip()]
    if not raw:
        raise ValueError("--model is required")
    if any(x.lower() == "all" for x in raw):
        return list(TSFM_MODEL_ORDER)
    return [normalize_model_name(x) for x in raw]


def _resolve_datasets(tokens: list[str]) -> list[str]:
    raw = [str(x).strip() for x in tokens if str(x).strip()]
    if not raw:
        raise ValueError("--dataset is required")
    if any(x.lower() == "all" for x in raw):
        return list(ALL_CSV_DATASETS)
    return [_normalize_csv_dataset_name(x) for x in raw]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Error analysis based on existing main-model inference cache (read-only).",
        allow_abbrev=False,
    )

    parser.add_argument(
        "--cache",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Must be enabled. This script only reads existing cache and will fail if disabled.",
    )
    parser.add_argument("--x_classes", type=int, default=10, help="Number of clusters for X-axis error classes")
    parser.add_argument("--y_classes", type=int, default=10, help="Number of clusters for Y-axis input classes")
    parser.add_argument("--model", nargs="+", required=True, help="Main model(s), supports space-separated list")
    parser.add_argument("--dataset", nargs="+", required=True, help="Dataset(s), supports space-separated list")
    parser.add_argument("--pred_len", type=int, default=96, help="Prediction length")
    parser.add_argument("--context_length", type=int, default=520, help="Input context length")
    parser.add_argument(
        "--y_axis",
        type=str,
        default="XY",
        choices=["X", "Y", "XY", "E", "ALL"],
        help="Y-axis category source",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if not bool(args.cache):
        raise RuntimeError(
            "ErrorAnalysis requires existing main-model cache. "
            "Please pass --cache and ensure cache files already exist."
        )

    if int(args.x_classes) <= 0 or int(args.y_classes) <= 0:
        raise ValueError("--x_classes and --y_classes must be positive integers")
    if int(args.pred_len) <= 0 or int(args.context_length) <= 0:
        raise ValueError("--pred_len and --context_length must be positive integers")

    selected_models = _resolve_models(list(args.model or []))
    selected_datasets = _resolve_datasets(list(args.dataset or []))

    print(f"[ErrorAnalysis] models={selected_models}")
    print(f"[ErrorAnalysis] datasets={selected_datasets}")
    print(
        f"[ErrorAnalysis] pred_len={int(args.pred_len)} | context_length={int(args.context_length)} | "
        f"x_classes={int(args.x_classes)} | y_classes={int(args.y_classes)} | y_axis={args.y_axis}"
    )

    for model_name in selected_models:
        if model_name not in BACKEND_COMPATIBLE_MODELS:
            print(
                f"[ErrorAnalysis][Skip] model={model_name} is not supported by current evaluation backend.",
                flush=True,
            )
            continue

        for dataset_name in selected_datasets:
            try:
                print(
                    f"[ErrorAnalysis] running model={model_name} dataset={dataset_name} pred_len={int(args.pred_len)}",
                    flush=True,
                )
                out_path = _run_one(
                    model_name=model_name,
                    dataset_name=dataset_name,
                    pred_len=int(args.pred_len),
                    context_length=int(args.context_length),
                    x_classes=int(args.x_classes),
                    y_classes=int(args.y_classes),
                    y_mode=str(args.y_axis),
                )
                print(f"[ErrorAnalysis] saved plot: {out_path}", flush=True)
            except Exception as exc:
                print(
                    f"[ErrorAnalysis][Failed] model={model_name} dataset={dataset_name} "
                    f"error={type(exc).__name__}: {exc}",
                    flush=True,
                )
                print(traceback.format_exc(), flush=True)


if __name__ == "__main__":
    main()
