from __future__ import annotations
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
from gluonts.model.forecast import QuantileForecast, SampleForecast
from eval.eval_util import build_progress_line
from eval.metrics import _entry_forecast_start, _forecast_to_sample_array, _sanitize_forecast_array, _sanitize_quantile_payload


def _resolve_infer_cache_enabled(args) -> bool:
    return bool(getattr(args, "cache", False))


def _build_infer_cache_path(args, dataset_name: str) -> Path:
    model_raw = str(getattr(args, "model", "unknown"))
    cache_model_name = {"timesfm-2.5": "timesfm-2"}.get(model_raw, model_raw)
    safe_model = cache_model_name.replace("/", "_").replace("-", "_")
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
            print(f"[ORCA] Cache load: opening {cache_path.name}* ({cache_size_mb:.1f} MB)", flush=True)
        except Exception:
            print(f"[ORCA] Cache load: opening {cache_path.name}*", flush=True)

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
                f"[ORCA] Cache load: parallel decode start | records={record_count} | workers={workers}",
                flush=True,
            )
            with ThreadPoolExecutor(max_workers=workers) as executor:
                dense_records = list(executor.map(_decode_one, range(record_count)))
        else:
            dense_records = [_decode_one(i) for i in range(record_count)]

        print(f"[ORCA] Cache load: decoded dense records={len(dense_records)}", flush=True)
        return {
            "dense_records": dense_records,
        }
    except Exception:
        return None


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

    # Write regular-size caches as one contiguous array.
    if all_same_shape and first_shape is not None:
        est_bytes = int(record_count) * int(np.prod(np.asarray(first_shape, dtype=np.int64))) * 2
        if not bool(large_data_channel):
            print(
                f"[ORCA] Cache save(v3) mode=stack_save | records={record_count} | est={est_bytes/(1024.0**3):.2f} GiB",
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
                            prefix="[ORCA] Cache save(v3-stack)",
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
            f"[ORCA] Cache save(v3) mode=large_data_memmap_stream | records={record_count} | est={est_bytes/(1024.0**3):.2f} GiB",
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
                        prefix="[ORCA] Cache save(v3-memmap)",
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
        f"[ORCA] Cache save(v3-compat) fallback(object) start: records={record_count}",
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
                    prefix="[ORCA] Cache save(v3-compat)",
                    done=done,
                    total=record_count,
                    elapsed_seconds=elapsed,
                    unit="rec",
                ),
                flush=True,
            )
    payload_array = np.array(payloads, dtype=object)
    np.save(payload_path, payload_array, allow_pickle=True)


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
