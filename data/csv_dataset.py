from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from gluonts.dataset.common import ListDataset
from gluonts.dataset.split import split as gluonts_split


CSV_DATASET_SPECS: dict[str, dict[str, Any]] = {
    "ETTh1": {"relative_path": "ETT-small/ETTh1.csv"},
    "ETTh2": {"relative_path": "ETT-small/ETTh2.csv"},
    "ETTm1": {"relative_path": "ETT-small/ETTm1.csv"},
    "ETTm2": {"relative_path": "ETT-small/ETTm2.csv"},
    "Exchange": {"relative_path": "exchange_rate/exchange_rate.csv"},
    "Weather": {"relative_path": "weather/weather.csv"},
    "Electricity": {"relative_path": "electricity/electricity.csv"},
    "Traffic": {"relative_path": "traffic/traffic.csv"},
}


class CsvSeriesDataset:
    def __init__(
        self,
        *,
        csv_path: str | Path,
        prediction_length: int,
        target_column: str | None = "all",
        windows: int | None = None,
    ) -> None:
        self.csv_path = Path(csv_path).expanduser().resolve()
        if not self.csv_path.is_file():
            raise FileNotFoundError(f"CSV file not found: {self.csv_path}")

        self.prediction_length = int(prediction_length)
        if self.prediction_length <= 0:
            raise ValueError(f"prediction_length must be >= 1, got {self.prediction_length}")

        self.target_column = None if target_column is None else str(target_column)

        raw_df = pd.read_csv(self.csv_path)
        self.freq = self._infer_frequency(raw_df, fallback="H")
        self.dataframe = self._normalize_dataframe(raw_df)
        self.numeric_columns = list(self.dataframe.columns)
        self.target_data, self.selected_columns = self._extract_target_data(self.dataframe, self.target_column)

        self.total_len = int(self.target_data.shape[-1])
        truncated = self.target_data
        target_payload = truncated
        self.gluonts_dataset = ListDataset(
            [
                {
                    "item_id": "stream_0",
                    "start": pd.Period("2000-01-01", freq=self.freq),
                    "target": target_payload,
                }
            ],
            freq=self.freq,
            one_dim_target=False,
        )

        default_windows = max(1, self.total_len // self.prediction_length)
        self.windows = int(windows) if windows is not None else int(default_windows)
        self.windows = max(1, self.windows)
        self.windows = min(self.windows, self.max_windows_for_distance(self.prediction_length))

        self.target_dim = int(truncated.shape[0])
        self._min_series_length = int(truncated.shape[-1])

    @staticmethod
    def _infer_frequency(df: pd.DataFrame, fallback: str = "H") -> str:
        if df.empty or df.shape[1] == 0:
            return str(fallback)

        ts_col = df.iloc[:, 0]
        parsed = pd.to_datetime(ts_col, errors="coerce")
        valid = parsed.dropna()
        if len(valid) >= 3:
            inferred = pd.infer_freq(valid)
            if inferred:
                return str(inferred)

        return str(fallback)

    @staticmethod
    def _normalize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            raise ValueError("CSV dataframe is empty")

        out_df = df.copy()
        if out_df.shape[1] >= 2 and not np.issubdtype(out_df.iloc[:, 0].dtype, np.number):
            out_df = out_df.iloc[:, 1:]

        numeric_df = out_df.select_dtypes(include=[np.number])
        if numeric_df.empty:
            raise ValueError("No numeric columns found in CSV after preprocessing")
        return numeric_df

    @staticmethod
    def _extract_target_data(df: pd.DataFrame, target_column: str | None) -> tuple[np.ndarray, list[str]]:
        if target_column is None or str(target_column).strip().lower() in {"all", "*"}:
            selected_columns = list(df.columns)
        elif target_column in df.columns:
            selected_columns = [str(target_column)]
        elif "OT" in df.columns:
            selected_columns = ["OT"]
        else:
            selected_columns = [str(df.columns[-1])]

        values = df[selected_columns].to_numpy(dtype=np.float32)
        if values.ndim != 2:
            values = values.reshape(values.shape[0], -1)
        # GluonTS multivariate convention: target shape (D, T)
        return values.T.astype(np.float32), selected_columns

    def max_windows_for_distance(self, distance: int) -> int:
        distance = int(distance)
        if distance <= 0:
            return 0
        if self.total_len < self.prediction_length:
            return 0
        return int((self.total_len - self.prediction_length) // distance + 1)

    def build_test_data(self, *, distance: int, windows: int | None = None):
        distance = int(distance)
        if distance <= 0:
            raise ValueError(f"distance must be >= 1, got {distance}")

        max_windows = self.max_windows_for_distance(distance)
        if max_windows <= 0:
            raise ValueError(
                f"No valid windows for distance={distance}. total_len={self.total_len}, pred_len={self.prediction_length}"
            )

        windows_req = int(windows) if windows is not None else int(self.windows)
        windows_use = max(1, min(windows_req, max_windows))

        _, test_template = gluonts_split(self.gluonts_dataset, offset=-self.total_len)
        return test_template.generate_instances(
            prediction_length=self.prediction_length,
            windows=windows_use,
            distance=distance,
        ), windows_use

    @property
    def test_data(self):
        test_data, _ = self.build_test_data(distance=self.prediction_length, windows=self.windows)
        return test_data

    @classmethod
    def from_named_dataset(
        cls,
        *,
        dataset_name: str,
        cache_dir: str | Path,
        prediction_length: int,
        target_column: str = "OT",
        windows: int | None = None,
    ) -> "CsvSeriesDataset":
        if dataset_name not in CSV_DATASET_SPECS:
            raise ValueError(
                f"Unsupported dataset_name={dataset_name!r}. Supported: {sorted(CSV_DATASET_SPECS.keys())}"
            )
        spec = CSV_DATASET_SPECS[dataset_name]
        csv_path = Path(cache_dir) / spec["relative_path"]

        return cls(
            csv_path=csv_path,
            prediction_length=prediction_length,
            target_column=target_column,
            windows=windows,
        )
