from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
from typing import List, Sequence

import numpy as np
import torch
import torch.nn.functional as F


def build_time_id() -> str:
	return datetime.now().strftime("%Y%m%d_%H%M%S_%f")


def sanitize_tensor(tensor: torch.Tensor) -> torch.Tensor:
	tensor = torch.where(torch.isnan(tensor), torch.zeros_like(tensor), tensor)
	tensor = torch.where(torch.isinf(tensor), torch.zeros_like(tensor), tensor)
	return tensor


def parse_quantile_key(key: str) -> float | None:
	k = str(key).strip().lower()
	if k.startswith("p") and len(k) > 1:
		k = k[1:]
	try:
		q = float(k)
	except Exception:
		return None
	if not np.isfinite(q):
		return None
	if q > 1.0 and q <= 100.0:
		q = q / 100.0
	if q < 0.0 or q > 1.0:
		return None
	return q


def select_quantile_index(forecast_keys: Sequence[str] | None, q_count: int, *, target_quantile: float = 0.5) -> int | None:
	if not forecast_keys:
		return None
	if int(q_count) <= 0:
		return None
	parsed: list[tuple[int, float]] = []
	for idx, key in enumerate(list(map(str, forecast_keys))):
		q = parse_quantile_key(key)
		if q is not None:
			parsed.append((idx, float(q)))
	if not parsed:
		return None
	best_idx, _ = min(parsed, key=lambda t: abs(float(t[1]) - float(target_quantile)))
	if 0 <= int(best_idx) < int(q_count):
		return int(best_idx)
	return None


def collapse_batch_median(tensor: torch.Tensor) -> torch.Tensor:
	if tensor.ndim == 3 and tensor.shape[0] > 1:
		return tensor.median(dim=0, keepdim=True).values
	return tensor


def align_sequence_length(tensor: torch.Tensor, target_len: int) -> torch.Tensor:
	current_len = int(tensor.shape[1])
	if current_len == int(target_len):
		return tensor
	if current_len > int(target_len):
		return tensor[:, -int(target_len):, :]
	pad_size = int(target_len) - current_len
	return F.pad(tensor, (0, 0, pad_size, 0))


def prepare_aligned_sequence_batch(
	seq_list: List[torch.Tensor],
	target_len: int,
) -> torch.Tensor:
	aligned: list[torch.Tensor] = []
	for x in seq_list:
		if x.ndim != 3:
			continue
		aligned.append(align_sequence_length(x, int(target_len)))
	if not aligned:
		raise RuntimeError("No valid input sequence found for aligned batch")
	return torch.cat(aligned, dim=0)


def extract_loss_history_values(loss_history: Sequence[object] | None) -> list[float]:
	flat: list[float] = []
	if not loss_history:
		return flat
	for entry in loss_history:
		if isinstance(entry, (list, tuple)):
			for v in entry:
				try:
					flat.append(float(v))
				except Exception:
					continue
			continue
		try:
			flat.append(float(entry))
		except Exception:
			continue
	return flat


def save_refiner_loss_history_json(
	log_dir: Path,
	*,
	dataset_name: str,
	model_name: str,
	refiner_name: str,
	loss_history: Sequence[object] | None,
	val_loss_history: Sequence[object] | None = None,
	command_line: str | None = None,
	model_config: dict | None = None,
	time_id: str | None = None,
) -> Path:
	run_time_id = str(time_id or build_time_id())
	timestamp = datetime.now().isoformat(timespec="seconds")
	train_flat = extract_loss_history_values(loss_history)
	val_flat = extract_loss_history_values(val_loss_history)

	safe_dataset = str(dataset_name).replace("/", "_")
	safe_model = str(model_name).replace("/", "_")
	safe_refiner = str(refiner_name).replace("/", "_")
	log_dir = Path(log_dir)
	log_dir.mkdir(parents=True, exist_ok=True)
	out_path = log_dir / f"{safe_dataset}_{safe_model}_{safe_refiner}_{run_time_id}_loss.json"

	payload = {
		"timestamp": timestamp,
		"time_id": run_time_id,
		"dataset_name": dataset_name,
		"model_name": model_name,
		"refiner_name": refiner_name,
		"command_line": command_line,
		"model_config": (model_config or {}),
		"loss_history": [float(v) for v in train_flat],
		"train_loss_history": [float(v) for v in train_flat],
		"val_loss_history": [float(v) for v in val_flat],
	}
	with open(out_path, "w", encoding="utf-8") as f:
		json.dump(payload, f, ensure_ascii=False, indent=2)

	return out_path
