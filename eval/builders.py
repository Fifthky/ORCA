from __future__ import annotations
from pathlib import Path
import numpy as np
import torch
from gluonts.model.forecast import Forecast, QuantileForecast, SampleForecast
from core.refiner_orca_no_decay import OnlineRefinerORCANoDecay
from core.refiner_orca import OnlineRefinerORCA
from core.refiner_AdaY import OnlineRefinerAdaY
from core.refiner_DSOF import OnlineRefinerDSOF
from core.refiner_TAFAS import OnlineRefinerTAFAS
from core.refiner_SOLID import OnlineRefinerSOLID
from core.refiner_ELF import OnlineRefinerELF
from core.refiner_ridge import OnlineRefinerRidge
from core.refiner_ets import OnlineRefinerETS
from data.csv_dataset import CsvSeriesDataset
from eval.model_backends import create_base_predictor, model_supports_multivariate, resolve_model_ref
from eval.metrics import _sanitize_forecast_array


def _coerce_args_int(val, default: int) -> int:
    """Convert an args attribute to int, tolerating list/tuple produced by argparse nargs."""
    if val is None:
        return default
    if isinstance(val, (list, tuple)):
        return int(val[0]) if val else default
    return int(val)


def _coerce_args_float(val, default: float) -> float:
    """Convert an args attribute to float, tolerating list/tuple produced by argparse nargs."""
    if val is None:
        return default
    if isinstance(val, (list, tuple)):
        return float(val[0]) if val else default
    return float(val)


def _resolve_refiner(args) -> str:
    return str(getattr(args, "refiner", "orca_no_decay")).lower()


def _resolve_training_method(args) -> str:
    raw = getattr(args, "training_method", "online")
    if isinstance(raw, (list, tuple)):
        value = str(raw[0]).strip().lower() if raw else "online"
    else:
        value = str(raw).strip().lower()

    # Be tolerant to accidentally stringified list values such as "['batch']".
    if value not in {"batch", "online"}:
        has_batch = "batch" in value
        has_online = "online" in value
        if has_batch and not has_online:
            value = "batch"
        elif has_online and not has_batch:
            value = "online"

    if value not in {"batch", "online"}:
        raise ValueError(f"Unsupported training_method={value!r}. Expected one of: batch, online")
    return value


def _resolve_refiner_input(args) -> str:
    raw = getattr(args, "refiner_input", "all")
    if isinstance(raw, (list, tuple)):
        value = str(raw[0]).strip().lower() if raw else "all"
    else:
        value = str(raw).strip().lower()

    # Be tolerant to accidentally stringified list values such as "['all']".
    if value not in {"all", "xy", "x", "y", "e_past", "epast"}:
        has_epast = "e_past" in value or "epast" in value
        if "all" in value:
            value = "all"
        elif "xy" in value:
            value = "xy"
        elif has_epast:
            value = "e_past"
        elif "x" in value and "y" not in value:
            value = "x"
        elif "y" in value and "x" not in value:
            value = "y"
    if value == "epast":
        value = "e_past"
    if value not in {"all", "xy", "x", "y", "e_past"}:
        raise ValueError(f"Unsupported refiner_input={value!r}. Expected one of: all, xy, x, y, e_past")
    return value


def _resolve_update_rule(args) -> str:
    raw = getattr(args, "update_rule", "plain")
    if isinstance(raw, (list, tuple)):
        value = str(raw[0]).strip().lower() if raw else "plain"
    else:
        value = str(raw).strip().lower()
    if value not in {"plain", "bayesian"}:
        raise ValueError(f"Unsupported update_rule={value!r}. Expected one of: plain, bayesian")
    return value


def _resolve_online_buffer_windows(args) -> int:
    raw = getattr(args, "online_buffer_windows", None)
    if isinstance(raw, (list, tuple)):
        raw = raw[0] if raw else None
    if raw is None:
        raw = 3000
    return max(1, int(raw))


def _resolve_force_gate_open(args) -> bool:
    return bool(getattr(args, "force_gate_open", False))


def _resolve_channel_mix(args) -> bool:
    return bool(getattr(args, "channel_mix", True))


def _resolve_routing_temperature(args) -> float:
    return _coerce_args_float(getattr(args, "routing_temperature", 0.1), 0.1)


def _resolve_ema_error_momentum(args) -> float:
    return _coerce_args_float(getattr(args, "ema_error_momentum", 0.2), 0.2)


def _resolve_router(args) -> str:
    raw = getattr(args, "router", "boltzmann")
    if isinstance(raw, (list, tuple)):
        value = str(raw[0]).strip().lower() if raw else "boltzmann"
    else:
        value = str(raw).strip().lower()
    if value == "ema":
        value = "inema"
    if value not in {"boltzmann", "inema", "hard"}:
        raise ValueError(
            f"Unsupported router={value!r}. Expected one of: boltzmann, inema, hard"
        )
    return value


def _resolve_aday_delta(args) -> float:
    """Paper-aligned safe default: smaller bounded step on ETT, standard elsewhere."""
    dataset_name = ""
    csv_path = getattr(args, "csv_path", None)
    if csv_path:
        try:
            dataset_name = Path(str(csv_path)).stem
        except Exception:
            dataset_name = ""
    if not dataset_name:
        dataset_name = str(getattr(args, "dataset", "") or "")

    key = str(dataset_name).strip().lower()
    if key.startswith("etth") or key.startswith("ettm"):
        return 0.01
    return 0.1


def _resolve_solid_period(args) -> int:
    dataset_name = ""
    csv_path = getattr(args, "csv_path", None)
    if csv_path:
        try:
            dataset_name = Path(str(csv_path)).stem
        except Exception:
            dataset_name = ""
    if not dataset_name:
        dataset_name = str(getattr(args, "dataset", "") or "")

    key = str(dataset_name).strip().lower()
    if key.startswith("etth") or key.startswith("wth"):
        return 24
    if key.startswith("ettm"):
        return 96
    if "electricity" in key:
        return 24
    if "traffic" in key:
        return 24
    if "illness" in key:
        return 52
    if "weather" in key:
        return 144
    if "exchange" in key:
        return 1
    return 24


class _SequentialChannelPackedPredictor:
    """Adapter: run single-channel backend per channel, then pack to multivariate forecast."""

    def __init__(self, base_predictor, target_dim: int) -> None:
        self.base_predictor = base_predictor
        self.prediction_length = int(getattr(base_predictor, "prediction_length", 1))
        self.context_length = getattr(base_predictor, "context_length", None)
        self.batch_size = int(getattr(base_predictor, "batch_size", 1))
        self.target_dim = int(max(1, target_dim))
        self._packed_batch_logged = False

    @staticmethod
    def _iter_entries(dataset) -> list[dict]:
        entries: list[dict] = []
        for entry in dataset:
            if isinstance(entry, tuple):
                entries.append(entry[0])
            else:
                entries.append(entry)
        return entries

    @staticmethod
    def _ensure_channel_time(arr: np.ndarray) -> np.ndarray:
        x = np.asarray(arr, dtype=np.float32)
        if x.ndim == 1:
            return x.reshape(1, -1)
        if x.ndim == 2:
            return x
        return x.reshape(x.shape[0], -1)

    @staticmethod
    def _forecast_start(entry: dict):
        start = entry.get("forecast_start", None)
        if start is not None:
            return start
        if "start" in entry and "target" in entry:
            target = np.asarray(entry["target"], dtype=np.float32)
            target_length = int(target.shape[0]) if target.ndim == 1 else int(target.shape[-1])
            return entry["start"] + target_length
        return entry.get("start", None)

    @staticmethod
    def _to_univariate_samples(arr: np.ndarray, *, expected_pred_len: int) -> np.ndarray:
        x = np.asarray(arr, dtype=np.float32)
        if x.ndim == 1:
            return x.reshape(1, -1)
        if x.ndim == 2:
            # Normalize ambiguous 2D samples to (S, L).
            # Some backends emit (L, S), especially when prediction length is known.
            l = int(expected_pred_len)
            if x.shape[1] == l and x.shape[0] != l:
                return x
            if x.shape[0] == l and x.shape[1] != l:
                return x.T
            return x
        if x.ndim == 3:
            l = int(expected_pred_len)
            # Accept common variants: (S, 1, L), (S, L, 1), (L, S, 1), (L, 1, S).
            if x.shape[1] == 1 and x.shape[2] == l:
                return x[:, 0, :]
            if x.shape[2] == 1 and x.shape[1] == l:
                return x[:, :, 0]
            if x.shape[0] == l and x.shape[2] == 1:
                return x[:, :, 0].T
            if x.shape[0] == l and x.shape[1] == 1:
                return x[:, 0, :].T

            if x.shape[-1] == l:
                return x.reshape(-1, l)
            if x.shape[0] == l:
                return x.reshape(l, -1).T
            return x.reshape(x.shape[0], -1)
        return x.reshape(1, -1)

    def _split_entry_per_channel(self, entry: dict) -> list[dict]:
        if "past_target" in entry:
            src = self._ensure_channel_time(entry["past_target"])
            field = "past_target"
        else:
            src = self._ensure_channel_time(entry["target"])
            field = "target"

        channel_entries: list[dict] = []
        for ch_idx in range(int(src.shape[0])):
            out = dict(entry)
            out[field] = np.asarray(src[ch_idx], dtype=np.float32)
            if "target" in out:
                tgt = self._ensure_channel_time(entry["target"])
                out["target"] = np.asarray(tgt[ch_idx], dtype=np.float32)
            out["item_id"] = f"{entry.get('item_id', 'item')}_ch{ch_idx}"
            channel_entries.append(out)
        return channel_entries

    def _pack_quantile_forecast(self, channel_forecasts: list[Forecast], entry: dict) -> QuantileForecast:
        raw_keys = list(map(str, getattr(channel_forecasts[0], "forecast_keys", [])))

        def _is_quantile_key(key: str) -> bool:
            k = str(key).strip().lower()
            if k.startswith("p") and len(k) > 1:
                try:
                    float(k[1:])
                    return True
                except Exception:
                    return False
            try:
                float(k)
                return True
            except Exception:
                return False

        q_keys = [k for k in raw_keys if _is_quantile_key(k)]
        if not q_keys:
            raise RuntimeError(
                f"No numeric quantile keys available for packing. raw_keys={raw_keys}"
            )

        channel_q_arrays: list[np.ndarray] = []
        pred_len: int | None = None
        for fcst in channel_forecasts:
            per_q: list[np.ndarray] = []
            for q in q_keys:
                q_arr = _sanitize_forecast_array(np.asarray(fcst.quantile(str(q)), dtype=np.float32))
                if q_arr.ndim == 2 and q_arr.shape[-1] == 1:
                    q_arr = q_arr[:, 0]
                if q_arr.ndim != 1:
                    q_arr = q_arr.reshape(-1)
                per_q.append(q_arr)
            q_stack = np.stack(per_q, axis=0)  # (Q, L)
            pred_len = q_stack.shape[1] if pred_len is None else min(pred_len, int(q_stack.shape[1]))
            channel_q_arrays.append(q_stack)

        if pred_len is None:
            pred_len = int(self.prediction_length)
        aligned = [arr[:, :pred_len] for arr in channel_q_arrays]
        packed = _sanitize_forecast_array(np.stack(aligned, axis=-1))  # (Q, L, D)
        return QuantileForecast(
            item_id=entry.get("item_id"),
            forecast_arrays=_sanitize_forecast_array(np.asarray(packed, dtype=np.float32)),
            start_date=self._forecast_start(entry),
            forecast_keys=q_keys,
        )

    def _pack_sample_forecast(self, channel_forecasts: list[Forecast], entry: dict) -> SampleForecast:
        channel_samples: list[np.ndarray] = []
        sample_count: int | None = None
        pred_len: int | None = None
        for fcst in channel_forecasts:
            arr = _sanitize_forecast_array(np.asarray(getattr(fcst, "samples"), dtype=np.float32))
            s = self._to_univariate_samples(arr, expected_pred_len=int(self.prediction_length))  # (S, L)
            sample_count = s.shape[0] if sample_count is None else min(sample_count, int(s.shape[0]))
            pred_len = s.shape[1] if pred_len is None else min(pred_len, int(s.shape[1]))
            channel_samples.append(s)

        if sample_count is None or pred_len is None:
            sample_count = 1
            pred_len = int(self.prediction_length)
        aligned = [arr[:sample_count, :pred_len] for arr in channel_samples]
        # Defensive fallback: if horizon got into axis-0 due backend shape oddities, transpose.
        if int(pred_len) != int(self.prediction_length) and int(sample_count) == int(self.prediction_length):
            aligned = [arr.T for arr in aligned]
            sample_count, pred_len = pred_len, sample_count
        # GluonTS multivariate SampleForecast expects (S, L, D).
        packed = _sanitize_forecast_array(np.stack(aligned, axis=-1))  # (S, L, D)
        return SampleForecast(
            samples=_sanitize_forecast_array(np.asarray(packed, dtype=np.float32)),
            start_date=self._forecast_start(entry),
            item_id=entry.get("item_id"),
        )

    def predict(self, dataset, **kwargs):
        entries = self._iter_entries(dataset)
        if not entries:
            return

        # Keep memory bounded but avoid one backend invocation per window.
        # Process windows in chunks and pack per-window forecasts from flattened channel outputs.
        request_batch = kwargs.get("batch_size", None)
        try:
            window_batch_size = max(1, int(request_batch)) if request_batch is not None else int(self.batch_size)
        except Exception:
            window_batch_size = int(self.batch_size)
        window_batch_size = max(1, int(window_batch_size))

        for start in range(0, len(entries), window_batch_size):
            window_chunk = entries[start : start + window_batch_size]
            flat_channel_entries: list[dict] = []
            channel_counts: list[int] = []
            for entry in window_chunk:
                ch_entries = self._split_entry_per_channel(entry)
                flat_channel_entries.extend(ch_entries)
                channel_counts.append(len(ch_entries))

            # Preserve window-level batching semantics for single-channel backends.
            # If a window has D channels, backend work scales with D flattened entries.
            backend_kwargs = dict(kwargs)
            channel_factor = max(channel_counts) if channel_counts else 1
            backend_kwargs["batch_size"] = max(1, int(window_batch_size) * int(channel_factor))
            if not bool(self._packed_batch_logged):
                print(
                    f"[ORCA][InferBatchPacked] window_batch={window_batch_size} | "
                    f"channel_factor={channel_factor} | backend_batch={int(backend_kwargs['batch_size'])}",
                    flush=True,
                )
                self._packed_batch_logged = True

            try:
                flat_fcsts = list(self.base_predictor.predict(flat_channel_entries, **backend_kwargs))
            except TypeError as exc:
                msg = str(exc)
                if "unexpected keyword argument" in msg and "batch_size" in msg:
                    kwargs_compat = dict(backend_kwargs)
                    kwargs_compat.pop("batch_size", None)
                    flat_fcsts = list(self.base_predictor.predict(flat_channel_entries, **kwargs_compat))
                else:
                    raise

            if len(flat_fcsts) != len(flat_channel_entries):
                raise RuntimeError(
                    f"Forecast count mismatch in sequential channel packing chunk: forecasts={len(flat_fcsts)} entries={len(flat_channel_entries)}"
                )

            offset = 0
            for entry, ch_count in zip(window_chunk, channel_counts):
                group_fcsts = flat_fcsts[offset : offset + ch_count]
                offset += ch_count

                has_samples = all(hasattr(fcst, "samples") and getattr(fcst, "samples", None) is not None for fcst in group_fcsts)
                has_quantile = all(getattr(fcst, "forecast_keys", None) for fcst in group_fcsts)

                if has_samples:
                    yield self._pack_sample_forecast(group_fcsts, entry)
                elif has_quantile:
                    yield self._pack_quantile_forecast(group_fcsts, entry)
                else:
                    raise RuntimeError(
                        f"Unsupported forecast objects in sequential channel packing: {type(group_fcsts[0]).__name__}"
                    )


def _build_predictor(ds: CsvSeriesDataset, args, device: torch.device):
    model_name = str(getattr(args, "model", "moirai-2"))
    model_ref = resolve_model_ref(
        model_name=model_name,
        tsfm_local_path=getattr(args, "tsfm_local_path", None),
        download_online=bool(getattr(args, "download_online", False)),
    )
    supports_mv = bool(model_supports_multivariate(model_name))
    backend_target_dim = int(ds.target_dim) if supports_mv else 1
    predictor = create_base_predictor(
        model_name=model_name,
        model_ref=model_ref,
        prediction_length=int(ds.prediction_length),
        context_length=int(args.context_length),
        target_dim=int(backend_target_dim),
        batch_size=int(args.batch_size),
        device=device,
        chronos_predict_batches_jointly=bool(getattr(args, "chronos_predict_batches_jointly", False)),
    )
    if int(ds.target_dim) > 1 and not supports_mv:
        return _SequentialChannelPackedPredictor(predictor, target_dim=int(ds.target_dim))
    return predictor


def _build_refiner(
    args,
    device: torch.device,
    *,
    target_dim: int,
    train_window_count: int,
    val_window_count: int,
):
    refiner = _resolve_refiner(args)
    if _resolve_training_method(args) == "batch" and refiner not in {"aday", "dsof", "tafas", "solid"}:
        raise ValueError(f"{refiner} requires training_method=online")
    if refiner == "orca_no_decay":
        return OnlineRefinerORCANoDecay(
            feature_dim=int(target_dim),
            device=device,
            collect_train_windows=int(train_window_count),
            collect_val_windows=int(val_window_count),
            refiner_input=_resolve_refiner_input(args),
            online_training=(_resolve_training_method(args) == "online"),
            update_rule=_resolve_update_rule(args),
            force_gate_open=_resolve_force_gate_open(args),
            channel_mix=_resolve_channel_mix(args),
        )
    if refiner == "orca":
        return OnlineRefinerORCA(
            feature_dim=int(target_dim),
            device=device,
            collect_train_windows=int(train_window_count),
            refiner_input=_resolve_refiner_input(args),
            update_rule=_resolve_update_rule(args),
            router=_resolve_router(args),
            routing_temperature=_resolve_routing_temperature(args),
            ema_error_momentum=_resolve_ema_error_momentum(args),
            force_gate_open=_resolve_force_gate_open(args),
            channel_mix=_resolve_channel_mix(args),
            train_batch_size=_coerce_args_int(getattr(args, "train_batch_size", 256), 256),
        )
    if refiner == "aday":
        # Keep AdaY close to paper/reference defaults while fitting this framework's online protocol.
        aday_lr = 1e-4
        aday_delta = _resolve_aday_delta(args)
        aday_grad_clip = 1e-2
        return OnlineRefinerAdaY(
            feature_dim=int(target_dim),
            lr=aday_lr,
            device=device,
            delta=aday_delta,
            grad_clip=aday_grad_clip,
            collect_train_windows=int(train_window_count),
            collect_val_windows=int(val_window_count),
            online_training=(_resolve_training_method(args) == "online"),
            baseline_router=bool(getattr(args, "baseline_router", False)),
            seq_len_hint=_coerce_args_int(getattr(args, "context_length", 0), 0),
            warmup_buffer_windows=_coerce_args_int(getattr(args, "online_buffer_windows", 3000), 3000),
            warmup_batch_size=_coerce_args_int(getattr(args, "train_batch_size", 256), 256),
            warmup_epochs=10,
            warmup_patience=3,
            warmup_val_ratio=0.1,
            online_lr=1e-6,
            online_grad_clip=1e-4,
        )
    if refiner == "dsof":
        return OnlineRefinerDSOF(
            feature_dim=int(target_dim),
            hidden_dim=128,
            num_blocks=3,
            lr=1e-3,
            device=device,
            collect_train_windows=int(train_window_count),
            collect_val_windows=int(val_window_count),
            online_training=(_resolve_training_method(args) == "online"),
            baseline_router=bool(getattr(args, "baseline_router", False)),
            seq_len_hint=_coerce_args_int(getattr(args, "context_length", 0), 0),
            replay_buffer_size=300,
            batch_replay_size=32,
            num_er_epochs=1,
            freq_er_update=1,
            td_enabled=True,
            td_k=1,
            discounted=0.9,
            warmup_buffer_windows=_coerce_args_int(getattr(args, "online_buffer_windows", 3000), 3000),
            warmup_batch_size=_coerce_args_int(getattr(args, "train_batch_size", 256), 256),
            warmup_epochs=10,
            warmup_patience=3,
            warmup_val_ratio=0.1,
            warmup_lr=1e-3,
            online_batch_lr=1e-4,
            online_td_lr=3e-4,
            grad_clip=1e-2,
        )
    if refiner == "tafas":
        return OnlineRefinerTAFAS(
            feature_dim=int(target_dim),
            device=device,
            collect_train_windows=int(train_window_count),
            collect_val_windows=int(val_window_count),
            online_training=(_resolve_training_method(args) == "online"),
            baseline_router=bool(getattr(args, "baseline_router", False)),
        )
    if refiner == "solid":
        solid_period = _resolve_solid_period(args)
        return OnlineRefinerSOLID(
            feature_dim=int(target_dim),
            device=device,
            collect_train_windows=int(train_window_count),
            collect_val_windows=int(val_window_count),
            online_training=(_resolve_training_method(args) == "online"),
            baseline_router=bool(getattr(args, "baseline_router", False)),
            period=int(solid_period),
        )
    if refiner == "elf":
        return OnlineRefinerELF(
            feature_dim=int(target_dim),
            stride=1,
            device=device,
            baseline_router=bool(getattr(args, "baseline_router", False)),
        )
    if refiner == "ridge":
        return OnlineRefinerRidge(
            feature_dim=int(target_dim),
            device=device,
            collect_train_windows=int(train_window_count),
        )
    if refiner == "ets":
        return OnlineRefinerETS(
            feature_dim=int(target_dim),
            device=device,
        )

    raise ValueError(f"Unsupported refiner={refiner!r}.")
