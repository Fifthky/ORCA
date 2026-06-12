from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


@dataclass
class FilteredTestData:
    input: list
    label: list


def iter_test_inputs(test_data):
    if hasattr(test_data, "input"):
        return test_data.input
    return test_data


def entry_main(entry: dict | tuple) -> dict:
    if isinstance(entry, tuple):
        return entry[0]
    return entry


def entry_item_id(entry: dict | tuple) -> str:
    main = entry_main(entry)
    return str(main.get("item_id", "<missing>"))


def entry_channel_dim(entry: dict | tuple) -> int:
    main = entry_main(entry)
    if "past_target" in main:
        arr = np.asarray(main["past_target"], dtype=np.float32)
    else:
        arr = np.asarray(main["target"], dtype=np.float32)
    if arr.ndim <= 1:
        return 1
    return int(arr.shape[0])


def entry_series_length(entry: dict | tuple) -> int | None:
    main = entry_main(entry)
    if "past_target" in main:
        arr = np.asarray(main["past_target"], dtype=np.float32)
    elif "target" in main:
        arr = np.asarray(main["target"], dtype=np.float32)
    else:
        return None

    if arr.ndim == 1:
        return int(arr.shape[0])
    if arr.ndim >= 2:
        return int(arr.shape[1])
    return None


def is_contiguous_length_step(prev_len: int | None, curr_len: int | None, expected_step: int) -> bool:
    if prev_len is None or curr_len is None:
        return False
    if curr_len <= prev_len:
        return False
    return int(curr_len - prev_len) == int(expected_step)


def has_valid_prev_gt(entry: dict | tuple, pred_len: int) -> bool:
    main = entry_main(entry)
    if "past_target" in main:
        arr = np.asarray(main["past_target"], dtype=np.float32)
    else:
        arr = np.asarray(main["target"], dtype=np.float32)

    if "past_is_pad" in main:
        pad = np.asarray(main["past_is_pad"]).astype(bool)
        if pad.ndim == 1 and pad.shape[0] >= pred_len and bool(pad[-pred_len:].any()):
            return False
        if pad.ndim == 2 and pad.shape[-1] >= pred_len and bool(pad[..., -pred_len:].any()):
            return False

    if arr.ndim == 1:
        return int(arr.shape[0]) >= int(pred_len)
    if arr.ndim >= 2:
        return int(arr.shape[1]) >= int(pred_len)
    return False


def compute_window_and_update_steps_for_test_data(test_data, pred_len: int, stride: int) -> tuple[int, int, int]:
    window_count = 0
    flow_steps = 0
    has_prev_state = False
    prev_len: int | None = None

    for entry in iter_test_inputs(test_data):
        window_count += 1
        curr_len = entry_series_length(entry)

        if has_prev_state and has_valid_prev_gt(entry, int(pred_len)) and is_contiguous_length_step(prev_len, curr_len, int(stride)):
            flow_steps += 1

        has_prev_state = True
        prev_len = curr_len

    stream_count = 1 if window_count > 0 else 0
    return window_count, flow_steps, stream_count


def filter_test_data_by_context_length(test_data, min_context_length: int) -> FilteredTestData:
    min_ctx = int(max(0, min_context_length))
    if min_ctx <= 0:
        return FilteredTestData(input=list(test_data.input), label=list(test_data.label))

    filtered_input = []
    filtered_label = []
    for entry, label in zip(test_data.input, test_data.label):
        series_len = entry_series_length(entry)
        if series_len is not None and int(series_len) >= min_ctx:
            filtered_input.append(entry)
            filtered_label.append(label)

    return FilteredTestData(input=filtered_input, label=filtered_label)


def split_window_counts(
    total_windows: int,
    *,
    train_ratio: float = 0.7,
    val_ratio: float = 0.1,
) -> tuple[int, int, int]:
    total = int(max(0, total_windows))
    if total <= 0:
        return 0, 0, 0

    train_end = int(math.ceil(float(total) * float(train_ratio)))
    val_end = int(math.ceil(float(total) * float(train_ratio + val_ratio)))

    train_count = min(total, max(0, train_end))
    val_count = min(total - train_count, max(0, val_end - train_count))
    test_count = max(0, total - train_count - val_count)

    if test_count == 0 and total > 0:
        test_count = 1
        if val_count > 0:
            val_count -= 1
        elif train_count > 0:
            train_count -= 1

    return int(train_count), int(val_count), int(test_count)


def slice_filtered_test_data(test_data: FilteredTestData, start: int, end: int) -> FilteredTestData:
    n = min(len(test_data.input), len(test_data.label))
    s = max(0, min(int(start), n))
    e = max(s, min(int(end), n))
    return FilteredTestData(input=list(test_data.input[s:e]), label=list(test_data.label[s:e]))
