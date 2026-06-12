from __future__ import annotations

import contextlib
from dataclasses import dataclass
import io
import os
from pathlib import Path
from typing import Iterator

import numpy as np
import torch
from gluonts.itertools import batcher
from gluonts.model.forecast import QuantileForecast


HF_MODEL_IDS: dict[str, str] = {
    "chronos-2": "amazon/chronos-2",
    "moirai-2": "Salesforce/moirai-2.0-R-small",
    "tirex": "NX-AI/TiRex-1.1-gifteval",
    "timesfm-2": "google/timesfm-2.5-200m-pytorch",
    "sundial": "thuml/sundial-base-128m",
    "moirai-1-small": "Salesforce/moirai-1.0-R-small",
    "moirai-1-base": "Salesforce/moirai-1.1-R-base",
    "moirai-1-large": "Salesforce/moirai-1.1-R-large",
}

MULTIVARIATE_MODELS: set[str] = {
    "chronos-2",
    "moirai-1-small",
    "moirai-1-base",
    "moirai-1-large",
}


class BackendPredictorError(RuntimeError):
    pass


@contextlib.contextmanager
def _suppress_process_output(enabled: bool):
    if not enabled:
        yield
        return

    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    saved_stdout_fd = os.dup(1)
    saved_stderr_fd = os.dup(2)
    try:
        os.dup2(devnull_fd, 1)
        os.dup2(devnull_fd, 2)
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            yield
    finally:
        os.dup2(saved_stdout_fd, 1)
        os.dup2(saved_stderr_fd, 2)
        os.close(saved_stdout_fd)
        os.close(saved_stderr_fd)
        os.close(devnull_fd)


@dataclass
class _PredictorBase:
    prediction_length: int
    context_length: int | None
    batch_size: int
    _batch_usage_logged: bool = False

    def _iter_entries(self, dataset) -> list[dict]:
        entries: list[dict] = []
        for entry in dataset:
            if isinstance(entry, tuple):
                entries.append(entry[0])
            else:
                entries.append(entry)
        return entries

    def _effective_batch_size(self, kwargs: dict | None = None) -> int:
        if kwargs is None:
            kwargs = {}
        raw = kwargs.get("batch_size", None)
        if raw is None:
            return max(1, int(self.batch_size))
        try:
            return max(1, int(raw))
        except Exception:
            return max(1, int(self.batch_size))

    def _log_batch_usage_once(self, *, backend_name: str, requested: int, total_entries: int) -> None:
        if bool(getattr(self, "_batch_usage_logged", False)):
            return
        configured = max(1, int(self.batch_size))
        print(
            f"[Refined-CSV][InferBatch] backend={backend_name} | configured={configured} | "
            f"requested={int(requested)} | entries={int(total_entries)}",
            flush=True,
        )
        self._batch_usage_logged = True

    @staticmethod
    def _extract_target(entry: dict, *, context_length: int | None) -> np.ndarray:
        if "past_target" in entry:
            arr = np.asarray(entry["past_target"], dtype=np.float32)
        else:
            arr = np.asarray(entry["target"], dtype=np.float32)
        if context_length is None:
            return arr
        if arr.ndim == 1:
            return arr[-int(context_length):]
        return arr[..., -int(context_length):]

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


class Chronos2Predictor(_PredictorBase):
    def __init__(
        self,
        model_ref: str,
        prediction_length: int,
        context_length: int | None,
        batch_size: int,
        device: torch.device,
        quantile_levels: tuple[float, ...] = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9),
        predict_batches_jointly: bool = False,
    ) -> None:
        super().__init__(prediction_length=prediction_length, context_length=context_length, batch_size=batch_size)
        self.quantile_levels = quantile_levels
        self.predict_batches_jointly = bool(predict_batches_jointly)
        try:
            from chronos import BaseChronosPipeline
        except Exception as exc:
            raise BackendPredictorError(
                "chronos package is required for model 'chronos-2'. Install: pip install 'chronos-forecasting>=2.1'"
            ) from exc

        kwargs = {}
        if str(device).startswith("cuda"):
            kwargs["device_map"] = "cuda"
            kwargs["torch_dtype"] = torch.bfloat16
            kwargs["low_cpu_mem_usage"] = True

        # Prefer the explicit Chronos2Pipeline API from official docs when available.
        try:
            from chronos import Chronos2Pipeline

            self.pipeline = Chronos2Pipeline.from_pretrained(model_ref, **kwargs)
        except Exception:
            self.pipeline = BaseChronosPipeline.from_pretrained(model_ref, **kwargs)

    @staticmethod
    def _to_quantile_forecast_array(q_tensor: torch.Tensor) -> np.ndarray:
        q_np = q_tensor.detach().cpu().numpy()
        if q_np.ndim == 3:
            # Chronos2 output: (D, L, Q) -> GluonTS quantile payload: (Q, L, D)
            q_np = q_np.transpose(2, 1, 0)
            if q_np.shape[-1] == 1:
                q_np = q_np[:, :, 0]
            return np.asarray(q_np, dtype=np.float32)
        if q_np.ndim == 2:
            # Univariate fallback: (L, Q) -> (Q, L)
            return np.asarray(q_np.T, dtype=np.float32)
        return np.asarray(q_np, dtype=np.float32)

    def predict(self, dataset, **kwargs) -> Iterator[QuantileForecast]:
        entries = self._iter_entries(dataset)
        if not entries:
            return

        effective_batch_size = self._effective_batch_size(kwargs)
        model_batch_size = max(1, int(effective_batch_size))
        self._log_batch_usage_once(
            backend_name="chronos-2",
            requested=effective_batch_size,
            total_entries=len(entries),
        )

        for batch_entries in batcher(entries, batch_size=effective_batch_size):
            model_inputs = [
                {"target": self._extract_target(entry, context_length=self.context_length)}
                for entry in batch_entries
            ]

            cross_learning = bool(self.predict_batches_jointly)
            if cross_learning:
                item_ids = [str(entry.get("item_id", "<missing>")) for entry in batch_entries]
                # Avoid cross-series leakage when rolling windows of the same item are in one call.
                if len(set(item_ids)) < len(item_ids):
                    cross_learning = False

            try:
                quantiles, _ = self.pipeline.predict_quantiles(
                    inputs=model_inputs,
                    prediction_length=self.prediction_length,
                    batch_size=model_batch_size,
                    quantile_levels=list(self.quantile_levels),
                    cross_learning=cross_learning,
                )
            except TypeError:
                # Compatibility with older Chronos-2 API still using predict_batches_jointly.
                quantiles, _ = self.pipeline.predict_quantiles(
                    inputs=model_inputs,
                    prediction_length=self.prediction_length,
                    batch_size=model_batch_size,
                    quantile_levels=list(self.quantile_levels),
                    predict_batches_jointly=cross_learning,
                )

            for q_tensor, entry in zip(quantiles, batch_entries):
                q_arr = self._to_quantile_forecast_array(q_tensor)
                yield QuantileForecast(
                    item_id=entry.get("item_id"),
                    forecast_arrays=np.asarray(q_arr, dtype=np.float32),
                    start_date=self._forecast_start(entry),
                    forecast_keys=list(map(str, self.quantile_levels)),
                )


class Moirai1PredictorFactory:
    @staticmethod
    def create(
        model_ref: str,
        prediction_length: int,
        context_length: int,
        target_dim: int,
        batch_size: int,
        device: torch.device,
    ):
        try:
            from uni2ts.model.moirai import MoiraiForecast, MoiraiModule
        except Exception as exc:
            raise BackendPredictorError(
                "uni2ts Moirai backend is required. Install compatible uni2ts package."
            ) from exc

        module = MoiraiModule.from_pretrained(model_ref)
        model = MoiraiForecast(
            module=module,
            prediction_length=int(prediction_length),
            context_length=int(context_length),
            patch_size=32,
            num_samples=100,
            target_dim=int(target_dim),
            feat_dynamic_real_dim=0,
            past_feat_dynamic_real_dim=0,
        )
        return model.create_predictor(batch_size=int(batch_size), device=str(device))


class Moirai2PredictorFactory:
    @staticmethod
    def create(
        model_ref: str,
        prediction_length: int,
        context_length: int,
        target_dim: int,
        batch_size: int,
        device: torch.device,
    ):
        try:
            from uni2ts.model.moirai2 import Moirai2Forecast, Moirai2Module
        except Exception as exc:
            raise BackendPredictorError(
                "uni2ts Moirai2 backend is required. Install compatible uni2ts package with moirai2 support."
            ) from exc

        model = Moirai2Forecast(
            module=Moirai2Module.from_pretrained(model_ref),
            prediction_length=int(prediction_length),
            context_length=int(context_length),
            target_dim=int(target_dim),
            feat_dynamic_real_dim=0,
            past_feat_dynamic_real_dim=0,
        )
        return model.create_predictor(batch_size=int(batch_size), device=str(device))


class Moirai2QuantilePredictor(_PredictorBase):
    def __init__(
        self,
        model_ref: str,
        prediction_length: int,
        context_length: int,
        target_dim: int,
        batch_size: int,
        device: torch.device,
        quantile_levels: tuple[float, ...] = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9),
    ) -> None:
        super().__init__(prediction_length=prediction_length, context_length=context_length, batch_size=batch_size)
        self.quantile_levels = quantile_levels
        self.model_ref = str(model_ref)
        self.target_dim = int(target_dim)
        self.device = torch.device(device)
        try:
            from uni2ts.model.moirai2 import Moirai2Forecast, Moirai2Module
        except Exception as exc:
            raise BackendPredictorError(
                "uni2ts Moirai2 backend is required. Install compatible uni2ts package with moirai2 support."
            ) from exc

        self.model = Moirai2Forecast(
            module=Moirai2Module.from_pretrained(self.model_ref),
            prediction_length=int(self.prediction_length),
            context_length=int(context_length),
            target_dim=int(self.target_dim),
            feat_dynamic_real_dim=0,
            past_feat_dynamic_real_dim=0,
        ).to(self.device)

    def _normalize_input_for_moirai2(self, arr: np.ndarray) -> np.ndarray:
        x = np.asarray(arr, dtype=np.float32)
        if x.ndim == 1:
            return x
        if x.ndim != 2:
            return x.reshape(-1).astype(np.float32)
        if self.target_dim > 1:
            if x.shape[0] == self.target_dim and x.shape[1] != self.target_dim:
                return x.T.astype(np.float32)
            if x.shape[1] == self.target_dim:
                return x.astype(np.float32)
            return x.T.astype(np.float32) if x.shape[0] <= x.shape[1] else x.astype(np.float32)
        if x.shape[0] == 1:
            return x[0].astype(np.float32)
        if x.shape[1] == 1:
            return x[:, 0].astype(np.float32)
        return x.reshape(-1).astype(np.float32)

    def _predict_batch(self, batch_entries: list[dict]) -> np.ndarray:
        batch_inputs = [
            self._normalize_input_for_moirai2(self._extract_target(entry, context_length=self.context_length))
            for entry in batch_entries
        ]
        forecasts = self.model.predict(batch_inputs)
        if isinstance(forecasts, torch.Tensor):
            return forecasts.detach().cpu().numpy()
        return np.asarray(forecasts)

    def predict(self, dataset, **kwargs):
        entries = self._iter_entries(dataset)
        if not entries:
            return

        curr_batch_size = self._effective_batch_size(kwargs)
        self._log_batch_usage_once(
            backend_name="moirai-2",
            requested=curr_batch_size,
            total_entries=len(entries),
        )

        for batch_entries in batcher(entries, batch_size=curr_batch_size):
            forecasts_np = self._predict_batch(batch_entries)
            for item, entry in zip(forecasts_np, batch_entries):
                yield QuantileForecast(
                    item_id=entry.get("item_id"),
                    forecast_arrays=np.asarray(item, dtype=np.float32),
                    start_date=self._forecast_start(entry),
                    forecast_keys=list(map(str, self.quantile_levels)),
                )


class SundialPredictor(_PredictorBase):
    def __init__(
        self,
        model_ref: str,
        prediction_length: int,
        context_length: int | None,
        batch_size: int,
        device: torch.device,
        num_samples: int = 100,
        quantile_levels: tuple[float, ...] = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9),
    ) -> None:
        super().__init__(prediction_length=prediction_length, context_length=context_length, batch_size=batch_size)
        self.device = device
        self.num_samples = int(num_samples)
        q_levels = sorted({float(q) for q in quantile_levels})
        self.quantile_levels = tuple(q_levels)
        try:
            from transformers import AutoModelForCausalLM, __version__ as transformers_version
        except Exception as exc:
            raise BackendPredictorError(
                "transformers is required for model 'sundial'. Install: pip install transformers"
            ) from exc

        ver_parts = str(transformers_version).split(".")
        ver_mm = tuple(int(x) for x in ver_parts[:2] if x.isdigit())
        if len(ver_mm) < 2 or ver_mm != (4, 40):
            raise BackendPredictorError(
                "Sundial requires transformers==4.40.x for reliable inference. "
                f"Current version: {transformers_version}."
            )

        self.model = AutoModelForCausalLM.from_pretrained(model_ref, trust_remote_code=True).to(device)
        self.model.eval()

    def _to_1d_tensor(self, arr: np.ndarray) -> torch.Tensor:
        if arr.ndim == 1:
            out = arr
        else:
            out = arr[-1]
        if self.context_length is not None:
            out = out[-int(self.context_length):]
        return torch.as_tensor(self._sanitize_1d_series(np.asarray(out, dtype=np.float32)), dtype=torch.float32)

    @staticmethod
    def _sanitize_1d_series(arr: np.ndarray) -> np.ndarray:
        x = np.asarray(arr, dtype=np.float32).reshape(-1)
        if x.size == 0:
            return np.zeros((1,), dtype=np.float32)

        x = x.copy()
        x[~np.isfinite(x)] = np.nan
        finite_mask = np.isfinite(x)
        if not finite_mask.any():
            return np.zeros_like(x, dtype=np.float32)

        first_idx = int(np.argmax(finite_mask))
        first_val = float(x[first_idx])
        x[:first_idx] = first_val
        for i in range(first_idx + 1, x.shape[0]):
            if not np.isfinite(x[i]):
                x[i] = x[i - 1]
        return x.astype(np.float32)

    def _left_pad(self, tensors: list[torch.Tensor]) -> torch.Tensor:
        max_len = max(int(t.numel()) for t in tensors)
        padded = []
        for t in tensors:
            if t.numel() < max_len:
                pad = torch.full((max_len - int(t.numel()),), float("nan"), dtype=t.dtype)
                padded.append(torch.cat([pad, t], dim=-1))
            else:
                padded.append(t)
        return torch.stack(padded)

    def _samples_to_quantile_array(self, arr: np.ndarray) -> tuple[np.ndarray, list[float]]:
        x = np.asarray(arr, dtype=np.float32)

        def _grouped_rank_quantiles(sample_major: np.ndarray) -> tuple[np.ndarray, list[float]]:
            # sample_major shape: [S, ...]
            s = int(sample_major.shape[0])
            q_levels = list(self.quantile_levels)
            g = max(1, min(int(len(q_levels)), s))
            sorted_desc = np.sort(sample_major, axis=0)[::-1]
            groups = np.array_split(sorted_desc, g, axis=0)
            # High->low bucket means, then reverse to low->high quantile order.
            bucket_means = [np.mean(grp, axis=0, dtype=np.float32) for grp in groups]
            q_arr = np.stack(bucket_means[::-1], axis=0).astype(np.float32)
            # Keep user-facing quantile keys aligned with configured levels.
            if len(q_levels) >= g:
                return q_arr, q_levels[:g]
            return q_arr, q_levels

        if x.ndim == 1:
            x = x.reshape(1, -1)
        if x.ndim == 2:
            # Prefer sample-major layout [S, L]; transpose if shape looks like [L, S].
            pred_len = int(self.prediction_length)
            n_samples = int(self.num_samples)
            if x.shape[0] == pred_len and x.shape[1] != pred_len:
                x = x.T
            elif x.shape[1] == n_samples and x.shape[0] != n_samples:
                x = x.T
            quantiles, levels = _grouped_rank_quantiles(x)
            quantiles = np.maximum.accumulate(np.asarray(quantiles, dtype=np.float32), axis=0)
            return np.asarray(quantiles, dtype=np.float32), levels
        if x.ndim == 3:
            # Multivariate defensive path: try to move sample axis to dim-0.
            n_samples = int(self.num_samples)
            if x.shape[0] != n_samples and n_samples in x.shape:
                x = np.moveaxis(x, int(list(x.shape).index(n_samples)), 0)
            quantiles, levels = _grouped_rank_quantiles(x)
            quantiles = np.maximum.accumulate(np.asarray(quantiles, dtype=np.float32), axis=0)
            return np.asarray(quantiles, dtype=np.float32), levels
        raise BackendPredictorError(f"Unsupported Sundial sample shape: {tuple(x.shape)}")

    def predict(self, dataset, **kwargs) -> Iterator[QuantileForecast]:
        entries = self._iter_entries(dataset)
        batch_x_shape_default = int(self.context_length) if self.context_length is not None else 2880
        batch_x_shape = int(kwargs.get("batch_x_shape", batch_x_shape_default))
        effective_batch_size = self._effective_batch_size(kwargs)
        self._log_batch_usage_once(
            backend_name="sundial",
            requested=effective_batch_size,
            total_entries=len(entries),
        )
        for batch_entries in batcher(entries, batch_size=effective_batch_size):
            context_tensors = [
                self._to_1d_tensor(self._extract_target(entry, context_length=self.context_length))
                for entry in batch_entries
            ]
            batch_x = self._left_pad(context_tensors).to(self.device)
            if batch_x.shape[-1] > batch_x_shape:
                batch_x = batch_x[..., -batch_x_shape:]

            if torch.isnan(batch_x).any():
                from gluonts.transform import LastValueImputation

                x_np = batch_x.detach().cpu().numpy()
                imputed = []
                for i in range(x_np.shape[0]):
                    imputed.append(LastValueImputation()(x_np[i]))
                batch_x = torch.tensor(np.vstack(imputed), dtype=torch.float32, device=self.device)

            batch_x = torch.nan_to_num(batch_x, nan=0.0, posinf=1e6, neginf=-1e6)

            with torch.no_grad():
                if str(self.device).startswith("cuda"):
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        outputs = self.model.generate(
                            batch_x.float(),
                            max_new_tokens=int(self.prediction_length),
                            revin=True,
                            num_samples=int(self.num_samples),
                        )
                else:
                    outputs = self.model.generate(
                        batch_x.float(),
                        max_new_tokens=int(self.prediction_length),
                        revin=True,
                        num_samples=int(self.num_samples),
                    )

            out_np = outputs.detach().float().cpu().numpy()
            for out_item, entry in zip(out_np, batch_entries):
                arr = np.asarray(out_item, dtype=np.float32)
                if not np.isfinite(arr).all():
                    raise BackendPredictorError("Sundial produced non-finite forecast values.")
                q_arr, q_levels = self._samples_to_quantile_array(arr)
                yield QuantileForecast(
                    item_id=entry.get("item_id"),
                    forecast_arrays=q_arr,
                    start_date=self._forecast_start(entry),
                    forecast_keys=list(map(str, q_levels)),
                )


class TimesFM2Predictor(_PredictorBase):
    def __init__(
        self,
        model_ref: str,
        prediction_length: int,
        context_length: int | None,
        batch_size: int,
        device: torch.device,
    ) -> None:
        super().__init__(prediction_length=prediction_length, context_length=context_length, batch_size=batch_size)
        self.device = torch.device(device)
        try:
            from timesfm import configs
            from timesfm.timesfm_2p5 import timesfm_2p5_torch
        except Exception as exc:
            raise BackendPredictorError(
                "timesfm package is required for model 'timesfm-2'."
            ) from exc

        self._configs = configs
        self.model = timesfm_2p5_torch.TimesFM_2p5_200M_torch()
        self.quantiles = list(np.arange(1, 10) / 10.0)
        self._compiled_signature: tuple[int, int, int] | None = None
        self._load_checkpoint(model_ref)
        self._move_model_to_device()

    def _move_model_to_device(self) -> None:
        candidates = [
            getattr(self.model, "model", None),
            self.model,
        ]
        for module in candidates:
            if module is None or not hasattr(module, "to"):
                continue
            try:
                module.to(self.device)
            except Exception:
                continue

    def _try_load_safetensors_direct(self, weight_file: Path) -> bool:
        try:
            from safetensors.torch import load_file
        except Exception:
            return False

        try:
            state = load_file(str(weight_file))
        except Exception:
            return False

        # Editable-install variants may expose the torch module under different attributes.
        candidates = [
            getattr(self.model, "model", None),
            self.model,
        ]
        for module in candidates:
            if module is None or not hasattr(module, "load_state_dict"):
                continue
            try:
                module.load_state_dict(state, strict=False)
                return True
            except Exception:
                continue
        return False

    def _load_checkpoint(self, model_ref: str) -> None:
        errors: list[str] = []
        ref_path = Path(str(model_ref)).expanduser()
        is_local_path = ref_path.exists()

        if is_local_path:
            resolved = ref_path.resolve()
            candidate_paths: list[Path] = []

            if resolved.is_file():
                candidate_paths.append(resolved)
            elif resolved.is_dir():
                # TimesFM local installs typically use a concrete weights file path.
                preferred = resolved / "model.safetensors"
                if preferred.exists():
                    candidate_paths.append(preferred)

                for pat in ("*.safetensors", "**/*.safetensors"):
                    for p in sorted(resolved.glob(pat)):
                        if p not in candidate_paths:
                            candidate_paths.append(p)

                # Keep directory itself as the last local fallback for other builds.
                candidate_paths.append(resolved)

            for cand in candidate_paths:
                cand_str = str(cand)
                for call in (
                    lambda p=cand_str: self.model.load_checkpoint(path=p),
                    lambda p=cand_str: self.model.load_checkpoint(p),
                ):
                    try:
                        call()
                        return
                    except Exception as exc:
                        errors.append(f"local checkpoint load failed ({cand_str}): {type(exc).__name__}: {exc}")

                if cand.is_file() and cand.suffix == ".safetensors":
                    if self._try_load_safetensors_direct(cand):
                        return
                    errors.append(
                        f"local safetensors direct load failed ({cand_str})"
                    )

        for call in (
            lambda: self.model.load_checkpoint(repo_id=str(model_ref)),
            lambda: self.model.load_checkpoint(path=str(model_ref)),
            lambda: self.model.load_checkpoint(str(model_ref)),
            lambda: self.model.load_checkpoint(),
        ):
            try:
                call()
                return
            except Exception as exc:
                errors.append(f"checkpoint load attempt failed: {type(exc).__name__}: {exc}")

        raise BackendPredictorError(
            "Unable to load TimesFM checkpoint. "
            f"model_ref={model_ref}. Attempts: {' | '.join(errors)}"
        )

    def _to_1d(self, arr: np.ndarray) -> np.ndarray:
        if arr.ndim == 1:
            out = arr
        else:
            out = arr[-1]
        if self.context_length is not None:
            out = out[-int(self.context_length):]
        return np.asarray(out, dtype=np.float32)

    def predict(self, dataset, **kwargs) -> Iterator[QuantileForecast]:
        entries = self._iter_entries(dataset)
        effective_batch_size = self._effective_batch_size(kwargs)
        self._log_batch_usage_once(
            backend_name="timesfm-2",
            requested=effective_batch_size,
            total_entries=len(entries),
        )
        compile_batch_override = kwargs.get("timesfm_compile_batch_size", None)
        try:
            requested_compile_batch = (
                max(1, int(compile_batch_override))
                if compile_batch_override is not None
                else int(effective_batch_size)
            )
        except Exception:
            requested_compile_batch = int(effective_batch_size)

        compile_batch_cap = kwargs.get("timesfm_compile_batch_max", None)
        try:
            if compile_batch_cap is not None:
                requested_compile_batch = min(requested_compile_batch, max(1, int(compile_batch_cap)))
        except Exception:
            pass

        for batch_entries in batcher(entries, batch_size=effective_batch_size):
            self._move_model_to_device()
            context = [self._to_1d(self._extract_target(entry, context_length=self.context_length)) for entry in batch_entries]
            max_context = max(arr.shape[0] for arr in context)
            max_context = ((max_context + self.model.model.p - 1) // self.model.model.p) * self.model.model.p
            compiled_max_context = min(15360, int(max_context))
            compiled_batch_size = max(1, int(requested_compile_batch))

            compile_sig = (compiled_max_context, 1024, int(compiled_batch_size))
            if self._compiled_signature != compile_sig:
                self.model.compile(
                    forecast_config=self._configs.ForecastConfig(
                        max_context=compiled_max_context,
                        max_horizon=1024,
                        infer_is_positive=True,
                        use_continuous_quantile_head=True,
                        fix_quantile_crossing=True,
                        force_flip_invariance=True,
                        return_backcast=False,
                        normalize_inputs=True,
                        per_core_batch_size=int(compiled_batch_size),
                    )
                )
                self._compiled_signature = compile_sig

            preds_chunks: list[np.ndarray] = []
            for start in range(0, len(context), int(compiled_batch_size)):
                sub_context = context[start : start + int(compiled_batch_size)]
                _, sub_preds = self.model.forecast(horizon=int(self.prediction_length), inputs=sub_context)
                preds_chunks.append(np.asarray(sub_preds))
            full_preds = np.concatenate(preds_chunks, axis=0)

            full_preds = full_preds[:, 0 : int(self.prediction_length), 1:]
            quantile_batch = full_preds.transpose((0, 2, 1))
            for q_arr, entry in zip(quantile_batch, batch_entries):
                yield QuantileForecast(
                    item_id=entry.get("item_id"),
                    forecast_arrays=np.asarray(q_arr, dtype=np.float32),
                    forecast_keys=list(map(str, self.quantiles)),
                    start_date=self._forecast_start(entry),
                )


class TiRexPredictor(_PredictorBase):
    def __init__(
        self,
        model_ref: str,
        prediction_length: int,
        context_length: int | None,
        batch_size: int,
        device: torch.device,
    ) -> None:
        super().__init__(prediction_length=prediction_length, context_length=context_length, batch_size=batch_size)
        backend = "cuda" if str(device).startswith("cuda") else "cpu"
        self._tirex_resample_strategy = None
        self._tirex_suppress_stdout = True

        # Enforce offline behavior to avoid accidental network access in restricted environments.
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

        try:
            from tirex.base import PretrainedModel
        except Exception as exc:
            raise BackendPredictorError(
                "tirex-ts package is required for model 'tirex'. Install: pip install 'tirex-ts[all]==1.3.0'"
            ) from exc

        ref_path = Path(str(model_ref)).expanduser()
        local_ckpt: Path | None = None
        if ref_path.exists():
            resolved = ref_path.resolve()
            if resolved.is_file() and resolved.suffix == ".ckpt":
                local_ckpt = resolved
            elif resolved.is_dir():
                cand = resolved / "model.ckpt"
                if cand.exists() and cand.is_file():
                    local_ckpt = cand

        if local_ckpt is None:
            raise BackendPredictorError(
                "TiRex must be loaded from local checkpoint file in offline mode. "
                f"Expected a local path containing model.ckpt, got model_ref={model_ref}."
            )

        ckpt_str = str(local_ckpt)
        try:
            try:
                checkpoint = torch.load(ckpt_str, map_location=str(device), weights_only=True)
            except TypeError:
                checkpoint = torch.load(ckpt_str, map_location=str(device))

            if not isinstance(checkpoint, dict):
                raise BackendPredictorError(
                    f"TiRex checkpoint must be a dict, got {type(checkpoint).__name__}."
                )
            if "hyper_parameters" not in checkpoint or "state_dict" not in checkpoint:
                raise BackendPredictorError(
                    "TiRex checkpoint missing required keys: 'hyper_parameters' and/or 'state_dict'."
                )

            model_cls = PretrainedModel.REGISTRY.get("TiRex", None)
            if model_cls is None:
                # Fallback for potential casing/name variations across tirex versions.
                for name, cls in PretrainedModel.REGISTRY.items():
                    if str(name).lower() == "tirex":
                        model_cls = cls
                        break
            if model_cls is None:
                raise BackendPredictorError(
                    f"Unable to find TiRex class in PretrainedModel.REGISTRY: {list(PretrainedModel.REGISTRY.keys())}"
                )

            model = model_cls(backend=backend, **checkpoint["hyper_parameters"])
            if hasattr(model, "on_load_checkpoint"):
                model.on_load_checkpoint(checkpoint)
            model.load_state_dict(checkpoint["state_dict"])
            self.model = model.to(str(device))
        except Exception as exc:
            raise BackendPredictorError(
                "Unable to initialize TiRex from local checkpoint in strict offline mode using direct checkpoint bootstrap. "
                f"checkpoint={ckpt_str}. Error: {type(exc).__name__}: {exc}"
            ) from exc

    def _prepare_entry(self, entry: dict) -> dict:
        out = dict(entry)
        raw = self._extract_target(entry, context_length=self.context_length)
        arr = np.asarray(raw, dtype=np.float32)
        if arr.ndim > 1:
            arr = np.asarray(arr[-1], dtype=np.float32)
        arr = arr.reshape(-1)

        # Keep a strict, fixed input length per request: truncate or left-pad with NaN.
        if self.context_length is not None:
            ctx = max(1, int(self.context_length))
            if arr.shape[0] > ctx:
                arr = arr[-ctx:]
            elif arr.shape[0] < ctx:
                pad = np.full((ctx - arr.shape[0],), np.nan, dtype=np.float32)
                arr = np.concatenate([pad, arr], axis=0)

        out["target"] = arr
        out["past_target"] = arr
        return out

    def _forecast_gluon_quiet(self, batch_entries: list[dict], *, batch_size: int):
        call_kwargs = dict(
            prediction_length=int(self.prediction_length),
            output_type="gluonts",
            batch_size=max(1, int(batch_size)),
            resample_strategy=self._tirex_resample_strategy,
        )

        if not bool(self._tirex_suppress_stdout):
            return self.model.forecast_gluon(batch_entries, **call_kwargs)

        try:
            with _suppress_process_output(True):
                return self.model.forecast_gluon(batch_entries, **call_kwargs)
        except Exception:
            # Retry with visible output when an actual runtime error occurs.
            return self.model.forecast_gluon(batch_entries, **call_kwargs)

    def predict(self, dataset, **kwargs):
        entries = self._iter_entries(dataset)
        if not entries:
            return
        effective_batch_size = self._effective_batch_size(kwargs)
        self._log_batch_usage_once(
            backend_name="tirex",
            requested=effective_batch_size,
            total_entries=len(entries),
        )

        prepared_entries = [self._prepare_entry(entry) for entry in entries]
        for batch_entries in batcher(prepared_entries, batch_size=effective_batch_size):
            forecasts = self._forecast_gluon_quiet(
                batch_entries,
                batch_size=len(batch_entries),
            )
            for fcst in forecasts:
                yield fcst


def resolve_model_ref(model_name: str, tsfm_local_path: str | None, download_online: bool) -> str:
    if download_online:
        return HF_MODEL_IDS[model_name]

    if model_name == "tirex":
        if tsfm_local_path:
            return str(Path(tsfm_local_path).expanduser().resolve())
        raise BackendPredictorError(
            "TiRex offline mode requires --tsfm_local_path pointing to a local TiRex directory or model.ckpt file."
        )

    if tsfm_local_path:
        return str(Path(tsfm_local_path).expanduser().resolve())
    return HF_MODEL_IDS[model_name]


def model_supports_multivariate(model_name: str) -> bool:
    return model_name in MULTIVARIATE_MODELS


def create_base_predictor(
    *,
    model_name: str,
    model_ref: str,
    prediction_length: int,
    context_length: int,
    target_dim: int,
    batch_size: int,
    device: torch.device,
    chronos_predict_batches_jointly: bool = False,
):
    if model_name in {"moirai-1-small", "moirai-1-base", "moirai-1-large"}:
        return Moirai1PredictorFactory.create(
            model_ref=model_ref,
            prediction_length=prediction_length,
            context_length=context_length,
            target_dim=target_dim,
            batch_size=batch_size,
            device=device,
        )

    if model_name == "moirai-2":
        return Moirai2QuantilePredictor(
            model_ref=model_ref,
            prediction_length=prediction_length,
            context_length=context_length,
            target_dim=target_dim,
            batch_size=batch_size,
            device=device,
        )

    if model_name == "chronos-2":
        return Chronos2Predictor(
            model_ref=model_ref,
            prediction_length=prediction_length,
            context_length=context_length,
            batch_size=batch_size,
            device=device,
            predict_batches_jointly=bool(chronos_predict_batches_jointly),
        )

    if model_name == "timesfm-2":
        return TimesFM2Predictor(
            model_ref=model_ref,
            prediction_length=prediction_length,
            context_length=context_length,
            batch_size=batch_size,
            device=device,
        )

    if model_name == "sundial":
        return SundialPredictor(
            model_ref=model_ref,
            prediction_length=prediction_length,
            context_length=context_length,
            batch_size=batch_size,
            device=device,
        )

    if model_name == "tirex":
        return TiRexPredictor(
            model_ref=model_ref,
            prediction_length=prediction_length,
            context_length=context_length,
            batch_size=batch_size,
            device=device,
        )

    raise BackendPredictorError(f"Unsupported model backend: {model_name}")
