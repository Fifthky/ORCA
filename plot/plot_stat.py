from __future__ import annotations

import argparse
import csv
import math
import re
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATASETS = ["ETTh1", "ETTh2", "ETTm1", "ETTm2", "Exchange", "Weather"]
BASE_MODELS = ["Chronos-2", "Moirai-2", "TiRex", "TimesFM-2.5", "Sundial"]


def _normalize_name_for_match(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(text).strip().lower())


def _normalize_model_header(name: str) -> str:
    key = _normalize_name_for_match(name)
    mapping = {
        "chronos2": "Chronos-2",
        "moirai2": "Moirai-2",
        "tirex": "TiRex",
        "timesfm2": "TimesFM-2.5",
        "sundial": "Sundial",
    }
    if key not in mapping:
        raise ValueError(f"Unsupported model header: {name!r}")
    return mapping[key]


def _read_table(csv_path: Path) -> list[list[str]]:
    with open(csv_path, "r", newline="") as f:
        return list(csv.reader(f))


def _extract_model_columns(header: list[str]) -> list[tuple[str, int]]:
    # Return list of (model_name, change_col_index)
    model_cols: list[tuple[str, int]] = []
    # header in these CSVs interleaves value, percent so we scan and pick percent cols
    for idx in range(2, len(header)):
        name = str(header[idx]).strip()
        if not name or name.lower() == "datasets avg.":
            continue
        try:
            canonical = _normalize_model_header(name)
        except ValueError:
            continue
        # change percentage usually in the next column (idx+1)
        change_idx = idx + 1 if idx + 1 < len(header) else idx
        model_cols.append((canonical, change_idx))

    by_name = {name: change_idx for name, change_idx in model_cols}
    ordered = [(m, by_name[m]) for m in BASE_MODELS if m in by_name]
    if len(ordered) != len(BASE_MODELS):
        missing = [m for m in BASE_MODELS if m not in by_name]
        raise ValueError(f"Missing base model columns: {missing}")
    return ordered


def _extract_dataset_changes(rows: list[list[str]]) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    if not rows:
        return out
    model_cols = _extract_model_columns(rows[0])
    for row in rows[1:]:
        if len(row) < 2:
            continue
        dataset_name = str(row[0]).strip()
        row_type = str(row[1]).strip().lower()
        if not dataset_name or row_type != "vanilla":
            continue
        change_map: dict[str, float] = {}
        for model_name, change_idx in model_cols:
            cell = row[change_idx] if change_idx < len(row) else ""
            raw = str(cell).strip().replace("%", "")
            try:
                val = float(raw)
            except Exception:
                val = float("nan")
            change_map[model_name] = val
        out[dataset_name] = change_map
    return out


def _format_mean_std(mean: float, std: float) -> str:
    if not math.isfinite(mean):
        return "nan"
    if not math.isfinite(std):
        std = 0.0
    return f"{mean:.2f}%±{std:.2f}%"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate per-seed stat CSV for Bay seeds.")
    parser.add_argument("--input_dir", type=str, default="results/plot_abla")
    parser.add_argument("--output", type=str, default="results/plots/plot_abla/plot_stat.csv")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    input_dir = (PROJECT_ROOT / str(args.input_dir)).resolve()
    output_path = (PROJECT_ROOT / str(args.output)).resolve()

    seeds = [10, 42, 100, 200]
    seed_files: list[Path] = []
    for s in seeds:
        stem = f"results_csv_Bay_xy_bayesian_buf3000_batch256_seed{s}_pred96_all_csv_dataset_mse.csv"
        p = input_dir / stem
        if not p.is_file():
            raise FileNotFoundError(f"Expected exact seed CSV not found: {p}")
        seed_files.append(p)

    # values[dataset][model] = list of floats per seed
    values: dict[str, dict[str, list[float]]] = {ds: {m: [] for m in BASE_MODELS} for ds in DATASETS}

    for p in seed_files:
        rows = _read_table(p)
        dataset_changes = _extract_dataset_changes(rows)
        for ds in DATASETS:
            row_map = dataset_changes.get(ds, {})
            for m in BASE_MODELS:
                v = row_map.get(m, float("nan"))
                values[ds][m].append(v)

    # compute per-dataset per-model mean/std
    mean_std_cells: dict[str, dict[str, tuple[float, float]]] = {ds: {} for ds in DATASETS}
    for ds in DATASETS:
        for m in BASE_MODELS:
            arr = np.array(values[ds][m], dtype=float)
            mean = float(np.nanmean(arr))
            std = float(np.nanstd(arr, ddof=0))
            mean_std_cells[ds][m] = (mean, std)

    # dataset avg column: per seed average across models, then mean/std across seeds
    dataset_avg: dict[str, tuple[float, float]] = {}
    for ds in DATASETS:
        per_seed_avgs: list[float] = []
        for i in range(len(seeds)):
            vals_i = [values[ds][m][i] for m in BASE_MODELS]
            per_seed_avgs.append(float(np.nanmean(np.array(vals_i, dtype=float))))
        mean = float(np.nanmean(np.array(per_seed_avgs, dtype=float)))
        std = float(np.nanstd(np.array(per_seed_avgs, dtype=float), ddof=0))
        dataset_avg[ds] = (mean, std)

    # model avg row: per seed average across datasets, then mean/std across seeds
    model_avg: dict[str, tuple[float, float]] = {}
    for m in BASE_MODELS:
        per_seed_avgs: list[float] = []
        for i in range(len(seeds)):
            vals_i = [values[ds][m][i] for ds in DATASETS]
            per_seed_avgs.append(float(np.nanmean(np.array(vals_i, dtype=float))))
        mean = float(np.nanmean(np.array(per_seed_avgs, dtype=float)))
        std = float(np.nanstd(np.array(per_seed_avgs, dtype=float), ddof=0))
        model_avg[m] = (mean, std)

    # overall avg (bottom-right): per seed average across all dataset-model cells
    per_seed_overall: list[float] = []
    for i in range(len(seeds)):
        all_vals = []
        for ds in DATASETS:
            for m in BASE_MODELS:
                all_vals.append(values[ds][m][i])
        per_seed_overall.append(float(np.nanmean(np.array(all_vals, dtype=float))))
    overall_mean = float(np.nanmean(np.array(per_seed_overall, dtype=float)))
    overall_std = float(np.nanstd(np.array(per_seed_overall, dtype=float), ddof=0))

    # write CSV
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="") as f:
        w = csv.writer(f)
        header = ["Model"] + BASE_MODELS + ["Avg"]
        w.writerow(header)
        for ds in DATASETS:
            row = [ds]
            for m in BASE_MODELS:
                mean, std = mean_std_cells[ds][m]
                row.append(_format_mean_std(mean, std))
            mmean, mstd = dataset_avg[ds]
            row.append(_format_mean_std(mmean, mstd))
            w.writerow(row)

        # Models Avg. row
        row = ["Models Avg."]
        for m in BASE_MODELS:
            mean, std = model_avg[m]
            row.append(_format_mean_std(mean, std))
        row.append(_format_mean_std(overall_mean, overall_std))
        w.writerow(row)

    print(f"[plot_stat] Wrote summary CSV: {output_path}")


if __name__ == "__main__":
    main()
