from __future__ import annotations

import inspect
import re
import csv
import math
from datetime import datetime
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from gluonts.ev.metrics import MAE, MSE

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
    flow_steps: int,
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
        "flow_steps": int(flow_steps),
        "num_updates": num_updates,
        "loss_mean": loss_mean,
        "loss_last": loss_last,
        "loss_history": loss_history,
    }
    with open(log_path, "a") as f:
        f.write(json.dumps(log_obj, ensure_ascii=False) + "\n")


# Backward-compat alias for historical call sites.
append_meanflow_update_log = append_refiner_update_log


def _iter_module_candidates(obj):
    if obj is None:
        return
    seen_ids: set[int] = set()
    queue = [obj]
    while queue:
        cur = queue.pop(0)
        cur_id = id(cur)
        if cur_id in seen_ids:
            continue
        seen_ids.add(cur_id)
        yield cur

        for attr_name in ("model", "module", "pipeline", "inner_predictor", "net"):
            try:
                nxt = getattr(cur, attr_name, None)
            except Exception:
                nxt = None
            if nxt is not None:
                queue.append(nxt)


def _enable_attention_outputs(obj) -> None:
    for node in _iter_module_candidates(obj):
        cfg = getattr(node, "config", None)
        if cfg is None:
            continue
        for key, value in (("output_attentions", True), ("return_dict", True)):
            try:
                setattr(cfg, key, value)
            except Exception:
                pass


def _extract_tensor_list_from_output(output) -> list[np.ndarray]:
    out: list[np.ndarray] = []

    def _append_tensor(tensor_obj) -> None:
        if not isinstance(tensor_obj, torch.Tensor):
            return
        if int(tensor_obj.ndim) < 2:
            return
        arr = tensor_obj.detach().float().cpu().numpy()
        out.append(np.asarray(arr, dtype=np.float32))

    if isinstance(output, torch.Tensor):
        _append_tensor(output)
        return out

    if isinstance(output, (tuple, list)):
        for item in output:
            if isinstance(item, torch.Tensor):
                _append_tensor(item)
            elif isinstance(item, (tuple, list)):
                for sub_item in item:
                    _append_tensor(sub_item)
        return out

    if hasattr(output, "attentions"):
        attn = getattr(output, "attentions", None)
        if isinstance(attn, (tuple, list)):
            for item in attn:
                _append_tensor(item)
        elif isinstance(attn, torch.Tensor):
            _append_tensor(attn)

    return out


def _to_numpy_tensor(tensor_obj) -> np.ndarray | None:
    if not isinstance(tensor_obj, torch.Tensor):
        return None
    if int(tensor_obj.ndim) < 2:
        return None
    arr = tensor_obj.detach().float().cpu().numpy()
    return np.asarray(arr, dtype=np.float32)


def _maybe_build_attention_from_qk(q_arr: np.ndarray, k_arr: np.ndarray) -> np.ndarray | None:
    q = np.asarray(q_arr, dtype=np.float32)
    k = np.asarray(k_arr, dtype=np.float32)
    if q.ndim < 2 or k.ndim < 2:
        return None

    # Normalize to (B, T, C).
    if q.ndim == 2:
        q = q[None, ...]
    if k.ndim == 2:
        k = k[None, ...]
    if q.ndim > 3:
        q = q.reshape(q.shape[0], q.shape[1], -1)
    if k.ndim > 3:
        k = k.reshape(k.shape[0], k.shape[1], -1)

    b = min(int(q.shape[0]), int(k.shape[0]))
    t = min(int(q.shape[1]), int(k.shape[1]))
    c = min(int(q.shape[2]), int(k.shape[2]))
    if b <= 0 or t <= 0 or c <= 0:
        return None

    q = q[:b, :t, :c]
    k = k[:b, :t, :c]
    scale = max(1.0, float(np.sqrt(float(c))))
    logits = np.matmul(q, np.swapaxes(k, -1, -2)) / scale
    logits = logits - np.max(logits, axis=-1, keepdims=True)
    weights = np.exp(logits)
    denom = np.sum(weights, axis=-1, keepdims=True)
    denom = np.maximum(denom, 1e-6)
    attn = weights / denom
    return np.asarray(attn, dtype=np.float32)


def _collect_qk_weight_pairs_from_modules(obj) -> dict[int, dict[str, np.ndarray]]:
    layer_pairs: dict[int, dict[str, np.ndarray]] = {}
    direct_patterns = [
        re.compile(r"encoder\.layers\.(\d+)\.self_attn\.(q_proj|k_proj)\.weight$"),
        re.compile(r"model\.layers\.(\d+)\.self_attn\.(q_proj|k_proj)\.weight$"),
        re.compile(r"layers\.(\d+)\.self_attn\.(q_proj|k_proj)\.weight$"),
        re.compile(r"transformer\.h\.(\d+)\.attn\.(q_proj|k_proj)\.weight$"),
    ]
    fused_qkv_patterns = [
        re.compile(r"encoder\.layers\.(\d+)\.self_attn\.(?:qkv_proj|c_attn|in_proj)\.weight$"),
        re.compile(r"model\.layers\.(\d+)\.self_attn\.(?:qkv_proj|c_attn|in_proj)\.weight$"),
        re.compile(r"layers\.(\d+)\.self_attn\.(?:qkv_proj|c_attn|in_proj)\.weight$"),
        re.compile(r"transformer\.h\.(\d+)\.attn\.(?:qkv_proj|c_attn|in_proj)\.weight$"),
        re.compile(r".*\.in_proj_weight$"),
    ]
    for node in _iter_module_candidates(obj):
        if not hasattr(node, "state_dict"):
            continue
        try:
            sd = node.state_dict()
        except Exception:
            continue
        for key, value in sd.items():
            if not isinstance(value, torch.Tensor):
                continue
            arr = value.detach().float().cpu().numpy()
            if arr.ndim != 2:
                continue

            key_str = str(key)
            matched_direct = False
            for pat in direct_patterns:
                m = pat.search(key_str)
                if m is None:
                    continue
                idx = int(m.group(1))
                proj = str(m.group(2))
                layer_pairs.setdefault(idx, {})[proj] = np.asarray(arr, dtype=np.float32)
                matched_direct = True
                break
            if matched_direct:
                continue

            for pat in fused_qkv_patterns:
                m = pat.search(key_str)
                if m is None:
                    continue
                idx = int(m.group(1)) if m.groups() else 0
                if arr.shape[0] % 3 != 0:
                    continue
                chunk = int(arr.shape[0] // 3)
                q_w = np.asarray(arr[0:chunk, :], dtype=np.float32)
                k_w = np.asarray(arr[chunk : 2 * chunk, :], dtype=np.float32)
                layer_pairs.setdefault(idx, {})["q_proj"] = q_w
                layer_pairs.setdefault(idx, {})["k_proj"] = k_w
                break
    return layer_pairs


def _build_weight_proxy_attention_maps(obj, *, max_layers: int = 8) -> list[np.ndarray]:
    out: list[np.ndarray] = []
    layer_pairs = _collect_qk_weight_pairs_from_modules(obj)
    if not layer_pairs:
        return out

    for layer_idx in sorted(layer_pairs.keys()):
        pair = layer_pairs[layer_idx]
        if "q_proj" not in pair or "k_proj" not in pair:
            continue
        q_w = np.asarray(pair["q_proj"], dtype=np.float32)
        k_w = np.asarray(pair["k_proj"], dtype=np.float32)
        d = int(min(q_w.shape[1], k_w.shape[1]))
        if d <= 0:
            continue
        q_use = q_w[:, :d]
        k_use = k_w[:, :d]
        score = np.matmul(q_use, k_use.T) / max(1.0, float(np.sqrt(float(d))))
        score = score - np.max(score, axis=-1, keepdims=True)
        w = np.exp(score)
        w = w / np.maximum(np.sum(w, axis=-1, keepdims=True), 1e-6)
        # Keep a synthetic batch dimension for plotting compatibility.
        out.append(np.asarray(w[None, ...], dtype=np.float32))
        if len(out) >= int(max_layers):
            break

    return out


def _save_attention_map_images(output_dir: Path, maps: list[np.ndarray]) -> list[str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    saved_files: list[str] = []

    for idx, arr in enumerate(maps):
        arr_np = np.asarray(arr, dtype=np.float32)
        npy_path = output_dir / f"attn_map_{idx:03d}.npy"
        np.save(npy_path, arr_np)
        saved_files.append(str(npy_path))

        # Keep only one head/sample slice for visual inspection.
        if arr_np.ndim >= 4:
            view = arr_np[0, 0]
        elif arr_np.ndim == 3:
            view = arr_np[0]
        else:
            view = arr_np
        if view.ndim != 2:
            view = np.asarray(view).reshape(view.shape[0], -1)

        fig_path = output_dir / f"attn_map_{idx:03d}.png"
        plt.figure(figsize=(6, 5))
        plt.imshow(view, aspect="auto", origin="lower")
        plt.colorbar()
        plt.title(f"Attention Map {idx}")
        plt.xlabel("Key Index")
        plt.ylabel("Query Index")
        plt.tight_layout()
        plt.savefig(fig_path, dpi=180)
        plt.close()
        saved_files.append(str(fig_path))

    return saved_files


def _select_attention_view(arr: np.ndarray) -> np.ndarray:
    x = np.asarray(arr, dtype=np.float32)
    if x.ndim >= 4:
        view = x[0, 0]
    elif x.ndim == 3:
        view = x[0]
    else:
        view = x
    if view.ndim != 2:
        view = np.asarray(view, dtype=np.float32).reshape(view.shape[0], -1)
    return np.asarray(view, dtype=np.float32)


def _row_normalize_attention(view: np.ndarray) -> np.ndarray:
    x = np.asarray(view, dtype=np.float32)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    x = np.maximum(x, 0.0)
    row_sum = np.sum(x, axis=1, keepdims=True)
    invalid = row_sum <= 1e-12
    if np.any(invalid):
        fallback = np.ones_like(x, dtype=np.float32)
        fallback = fallback / np.maximum(np.sum(fallback, axis=1, keepdims=True), 1e-12)
        x = np.where(invalid, fallback, x)
        row_sum = np.sum(x, axis=1, keepdims=True)
    return x / np.maximum(row_sum, 1e-12)


def _attention_metrics_for_map(arr: np.ndarray) -> dict[str, float]:
    view = _select_attention_view(arr)
    p = _row_normalize_attention(view)
    cols = int(max(1, p.shape[1]))
    entropy = -np.sum(p * np.log(np.maximum(p, 1e-12)), axis=1)
    entropy_norm = entropy / np.log(float(cols)) if cols > 1 else entropy
    top1 = np.max(p, axis=1)
    return {
        "entropy": float(np.mean(entropy_norm)),
        "top1_mass": float(np.mean(top1)),
    }


def _write_attention_metric_outputs(output_dir: Path, maps: list[np.ndarray]) -> list[str]:
    saved: list[str] = []
    if not maps:
        return saved

    metric_rows: list[dict[str, float]] = []
    for idx, arr in enumerate(maps):
        m = _attention_metrics_for_map(arr)
        metric_rows.append(
            {
                "index": float(idx),
                "entropy": float(m["entropy"]),
                "top1_mass": float(m["top1_mass"]),
            }
        )

    csv_path = output_dir / "attention_metrics.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["index", "entropy", "top1_mass"])
        for row in metric_rows:
            writer.writerow([int(row["index"]), row["entropy"], row["top1_mass"]])
    saved.append(str(csv_path))

    entropy_vals = np.asarray([row["entropy"] for row in metric_rows], dtype=np.float32)
    top1_vals = np.asarray([row["top1_mass"] for row in metric_rows], dtype=np.float32)
    x_idx = np.arange(len(metric_rows), dtype=np.int32)

    seq_path = output_dir / "attention_metric_sequence.png"
    plt.figure(figsize=(10, 4))
    plt.plot(x_idx, entropy_vals, label="Entropy (normalized)", linewidth=1.2)
    plt.plot(x_idx, top1_vals, label="Top1 mass", linewidth=1.2)
    plt.xlabel("Map index")
    plt.ylabel("Metric value")
    plt.title("Attention Metrics Over Sequence")
    plt.legend()
    plt.tight_layout()
    plt.savefig(seq_path, dpi=180)
    plt.close()
    saved.append(str(seq_path))

    dist_path = output_dir / "attention_metric_distribution.png"
    plt.figure(figsize=(8, 4))
    bins = min(30, max(8, int(len(metric_rows) // 3)))
    ent_hist, ent_edges = np.histogram(entropy_vals, bins=bins, density=True)
    t1_hist, t1_edges = np.histogram(top1_vals, bins=bins, density=True)
    ent_centers = 0.5 * (ent_edges[:-1] + ent_edges[1:])
    t1_centers = 0.5 * (t1_edges[:-1] + t1_edges[1:])
    plt.plot(ent_centers, ent_hist, label="Entropy density", linewidth=1.4)
    plt.plot(t1_centers, t1_hist, label="Top1 mass density", linewidth=1.4)
    plt.xlabel("Metric value")
    plt.ylabel("Density")
    plt.title("Attention Metric Distribution")
    plt.legend()
    plt.tight_layout()
    plt.savefig(dist_path, dpi=180)
    plt.close()
    saved.append(str(dist_path))

    return saved


def inspect_attention_maps(
    *,
    model_name: str,
    predictor,
    eval_input,
    dataset_name: str,
    max_windows: int = 1,
) -> Dict:
    supported_models = {"moirai-2", "chronos-2", "sundial"}
    model_key = str(model_name).strip().lower()
    if model_key not in supported_models:
        raise ValueError(
            f"Attention map inspection currently supports only {sorted(supported_models)}; got {model_name!r}"
        )

    if eval_input is None or len(eval_input) == 0:
        raise ValueError("Cannot inspect attention maps: evaluation input is empty.")

    probe_count = max(1, int(max_windows))
    probe_entries = [eval_input[idx] for idx in range(min(probe_count, len(eval_input)))]

    _enable_attention_outputs(predictor)

    captured_maps: list[np.ndarray] = []
    captured_q: list[np.ndarray] = []
    captured_k: list[np.ndarray] = []
    hooks = []
    try:
        for candidate in _iter_module_candidates(predictor):
            if not hasattr(candidate, "named_modules"):
                continue
            for module_name, submodule in candidate.named_modules():
                if not hasattr(submodule, "register_forward_hook"):
                    continue
                name_lower = str(module_name).lower()
                cls_lower = str(type(submodule).__name__).lower()
                if "attn" not in name_lower and "attention" not in name_lower and "attn" not in cls_lower and "attention" not in cls_lower:
                    # Fallback probes for Moirai-style projection modules.
                    if ".q_proj" in name_lower or name_lower.endswith("q_proj"):
                        def _q_hook(_mod, _inp, output):
                            arr = _to_numpy_tensor(output)
                            if arr is not None:
                                captured_q.append(arr)
                        hooks.append(submodule.register_forward_hook(_q_hook))
                    if ".k_proj" in name_lower or name_lower.endswith("k_proj"):
                        def _k_hook(_mod, _inp, output):
                            arr = _to_numpy_tensor(output)
                            if arr is not None:
                                captured_k.append(arr)
                        hooks.append(submodule.register_forward_hook(_k_hook))
                    continue

                def _hook(_mod, _inp, output):
                    captured_maps.extend(_extract_tensor_list_from_output(output))

                hooks.append(submodule.register_forward_hook(_hook))

        predict_kwargs = {"batch_size": 1}
        predict_fn = getattr(predictor, "predict")
        try:
            sig = inspect.signature(predict_fn)
            if "batch_size" not in sig.parameters:
                predict_kwargs = {}
        except Exception:
            predict_kwargs = {"batch_size": 1}

        probe_iter = predict_fn(probe_entries, **predict_kwargs)
        for _ in probe_iter:
            break
    finally:
        for h in hooks:
            try:
                h.remove()
            except Exception:
                pass

    valid_maps: list[np.ndarray] = []
    for arr in captured_maps:
        x = np.asarray(arr, dtype=np.float32)
        if x.size == 0:
            continue
        if x.ndim < 2:
            continue
        if not np.isfinite(x).any():
            continue
        valid_maps.append(x)

    if not valid_maps and captured_q and captured_k:
        pair_count = min(len(captured_q), len(captured_k))
        for i in range(pair_count):
            approx_map = _maybe_build_attention_from_qk(captured_q[i], captured_k[i])
            if approx_map is None:
                continue
            if approx_map.size == 0 or not np.isfinite(approx_map).any():
                continue
            valid_maps.append(approx_map)

    if not valid_maps and model_key in {"moirai-2", "sundial"}:
        valid_maps.extend(_build_weight_proxy_attention_maps(predictor, max_layers=8))

    if not valid_maps:
        out_dir = Path("results/attn_maps") / f"{dataset_name}_{model_key.replace('-', '_')}"
        out_dir.mkdir(parents=True, exist_ok=True)
        return {
            "model_name": model_key,
            "dataset_name": str(dataset_name),
            "output_dir": str(out_dir),
            "num_maps": 0,
            "saved_files": [],
            "warning": "No usable attention tensors captured; check backend implementation details.",
            "q_proj_captured": int(len(captured_q)),
            "k_proj_captured": int(len(captured_k)),
            "weight_proxy_maps": 0,
        }

    out_dir = Path("results/attn_maps") / f"{dataset_name}_{model_key.replace('-', '_')}"
    saved_files = _save_attention_map_images(out_dir, valid_maps)
    saved_files.extend(_write_attention_metric_outputs(out_dir, valid_maps))
    return {
        "model_name": model_key,
        "dataset_name": str(dataset_name),
        "output_dir": str(out_dir),
        "num_maps": int(len(valid_maps)),
        "saved_files": saved_files,
        "q_proj_captured": int(len(captured_q)),
        "k_proj_captured": int(len(captured_k)),
        "weight_proxy_maps": int(len([m for m in valid_maps if np.asarray(m).ndim == 3 and np.asarray(m).shape[0] == 1])),
    }


def get_metrics() -> list:
    return [
        MSE(forecast_type="mean"),
        MAE(forecast_type="mean"),
    ]


def get_agg_metrics(result: tuple | pd.DataFrame | Dict | None) -> Dict | None:
    if result is None:
        return None
    if isinstance(result, tuple) and len(result) > 0 and isinstance(result[0], dict):
        return result[0]
    if isinstance(result, pd.DataFrame) and not result.empty:
        return result.iloc[0].to_dict()
    if isinstance(result, dict):
        return result
    return None


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
        "timesfm-2": "TimesFM-2",
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
            "timesfm-2",
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
            flow_v = _summary_to_float_nan(row_r[c] if c < len(row_r) else "nan")
            metrics_by_key[(ds, model_short)] = {
                "base": float(base_v),
                "flow": float(flow_v),
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
        flow_mae = float(mae_pair.get("flow", float("nan")))
        base_mse = float(mse_pair.get("base", float("nan")))
        flow_mse = float(mse_pair.get("flow", float("nan")))

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
            "agg_metrics_flow": {
                str(mae_metric_key): flow_mae,
                str(mse_metric_key): flow_mse,
            },
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
        existing_records.append(rec)

        if math.isfinite(flow_mae) and math.isfinite(flow_mse):
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
                flow_metric = first_value((rec.get("agg_metrics_flow") or {}).get(metric_key))
                change_metric = summary_pct_change(flow_metric, base_metric)

                row_vanilla.extend([
                    _summary_format_metric(base_metric),
                    _summary_format_change(change_metric),
                ])
                row_refined.extend([
                    _summary_format_metric(flow_metric),
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


def save_comparison_plots(
    *,
    dataset_name: str,
    baseline_gt_windows: list[np.ndarray],
    baseline_pred_windows: list[np.ndarray],
    flow_pred_windows: list[np.ndarray],
    refiner: str,
    pred_len: int,
    variant_suffix: str = "",
) -> tuple[Path, Path]:
    plot_dir = Path("results/details/plots")
    plot_dir.mkdir(parents=True, exist_ok=True)

    safe_name = dataset_name.replace("/", "_")
    refiner_tag = str(refiner)
    variant_tag = str(variant_suffix).strip()
    variant_tail = f"_{variant_tag}" if variant_tag else ""

    pred_tag = f"pred{int(pred_len)}"
    global_path = plot_dir / f"csv_global_{safe_name}_{refiner_tag}{variant_tail}_{pred_tag}.png"
    last_path = plot_dir / f"csv_last_window_{safe_name}_{refiner_tag}{variant_tail}_{pred_tag}.png"

    if not baseline_gt_windows or not baseline_pred_windows or not flow_pred_windows:
        return global_path, last_path

    # Plotting should reflect non-overlapping windows only; evaluator metrics can
    # still use dense stride=1 windows independently.
    step = max(1, int(pred_len))
    aligned_n = min(len(baseline_gt_windows), len(baseline_pred_windows), len(flow_pred_windows))
    select_idx = list(range(0, int(aligned_n), int(step)))
    if not select_idx:
        select_idx = [0]

    baseline_gt_plot = [baseline_gt_windows[i] for i in select_idx]
    baseline_pred_plot = [baseline_pred_windows[i] for i in select_idx]
    flow_pred_plot = [flow_pred_windows[i] for i in select_idx]

    global_gt = np.concatenate([w[:, -1] if w.ndim > 1 else w.ravel() for w in baseline_gt_plot])
    global_base = np.concatenate([w[:, -1] if w.ndim > 1 else w.ravel() for w in baseline_pred_plot])
    global_flow = np.concatenate([w[:, -1] if w.ndim > 1 else w.ravel() for w in flow_pred_plot])
    global_len = min(global_gt.shape[0], global_base.shape[0], global_flow.shape[0])

    plt.figure(figsize=(16, 5))
    plt.plot(global_gt[:global_len], label="GT", linewidth=1.2)
    plt.plot(global_base[:global_len], label="Baseline", linewidth=1.0)
    plt.plot(global_flow[:global_len], label="Refined", linewidth=1.0)
    plt.title(f"Global Test Comparison - {dataset_name}")
    plt.xlabel("Global step")
    plt.ylabel("Value")
    plt.legend()
    plt.tight_layout()
    plt.savefig(global_path, dpi=180)
    plt.close()

    last_gt = baseline_gt_plot[-1][:, -1] if baseline_gt_plot[-1].ndim > 1 else baseline_gt_plot[-1].ravel()
    last_base = baseline_pred_plot[-1][:, -1] if baseline_pred_plot[-1].ndim > 1 else baseline_pred_plot[-1].ravel()
    last_flow = flow_pred_plot[-1][:, -1] if flow_pred_plot[-1].ndim > 1 else flow_pred_plot[-1].ravel()
    last_len = min(last_gt.shape[0], last_base.shape[0], last_flow.shape[0])

    plt.figure(figsize=(12, 4))
    plt.plot(last_gt[:last_len], label="GT", linewidth=1.4)
    plt.plot(last_base[:last_len], label="Baseline", linewidth=1.1)
    plt.plot(last_flow[:last_len], label="Refined", linewidth=1.1)
    plt.title(f"Last Window Comparison - {dataset_name}")
    plt.xlabel("Horizon step")
    plt.ylabel("Value")
    plt.legend()
    plt.tight_layout()
    plt.savefig(last_path, dpi=180)
    plt.close()

    return global_path, last_path
