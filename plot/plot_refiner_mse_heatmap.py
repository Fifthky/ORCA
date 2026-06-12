from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATASETS = ["ETTh1", "ETTh2", "ETTm1", "ETTm2", "Exchange", "Weather", "Electricity", "Traffic"]
# Subset used for boltzmann router comparison (six datasets)
SUB_DATASETS = ["ETTh1", "ETTh2", "ETTm1", "ETTm2", "Exchange", "Weather"]
BASE_MODELS = ["Chronos-2", "Moirai-2", "TiRex", "TimesFM-2.5", "Sundial"]

REFINER_FILES = [
    ("ORCA (ours)", "results_csv_Bay_xy_bayesian_buf3000_batch256_pred_avg_all_csv_dataset_mse.csv"),
    ("ELF", "results_csv_ELF_pred_avg_all_csv_dataset_mse.csv"),
    ("δ-Adapter", "results_csv_AdaY_pred_avg_all_csv_dataset_mse.csv"),
    ("DSOF", "results_csv_DSOF_pred_avg_all_csv_dataset_mse.csv"),
    ("TAFAS", "results_csv_TAFAS_pred_avg_all_csv_dataset_mse.csv"),
    ("SOLID", "results_csv_SOLID_pred_avg_all_csv_dataset_mse.csv"),
]


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


def _parse_percent(text: str) -> float:
    raw = str(text).strip().replace("%", "")
    if not raw:
        return float("nan")
    try:
        return float(raw)
    except Exception:
        return float("nan")


def _read_table(csv_path: Path) -> list[list[str]]:
    with open(csv_path, "r", newline="") as f:
        return list(csv.reader(f))


def _extract_model_columns(header: list[str]) -> list[tuple[str, int]]:
    model_cols: list[tuple[str, int]] = []
    max_col = max(0, len(header) - 1)
    for idx in range(2, max_col, 2):
        name = str(header[idx]).strip()
        if not name or name.lower() == "datasets avg.":
            continue
        canonical = _normalize_model_header(name)
        model_cols.append((canonical, idx + 1))

    by_name = {name: change_idx for name, change_idx in model_cols}
    ordered = [(m, by_name[m]) for m in BASE_MODELS if m in by_name]
    if len(ordered) != len(BASE_MODELS):
        missing = [m for m in BASE_MODELS if m not in by_name]
        raise ValueError(f"Missing base model columns: {missing}")
    return ordered


def _extract_dataset_changes(rows: list[list[str]]) -> dict[str, dict[str, float]]:
    if not rows:
        return {}
    model_cols = _extract_model_columns(rows[0])
    out: dict[str, dict[str, float]] = {}
    for row in rows[1:]:
        if len(row) < 2:
            continue
        dataset_name = str(row[0]).strip()
        row_type = str(row[1]).strip().lower()
        if dataset_name == "Models Avg." or row_type != "vanilla":
            continue
        change_map: dict[str, float] = {}
        for model_name, change_idx in model_cols:
            cell = row[change_idx] if change_idx < len(row) else ""
            change_map[model_name] = _parse_percent(cell)
        out[dataset_name] = change_map
    return out


def _resolve_refiner_files(input_dir: Path) -> list[tuple[str, Path]]:
    resolved: list[tuple[str, Path]] = []
    for label, name in REFINER_FILES:
        p = input_dir / name
        if p.is_file():
            resolved.append((label, p))
            continue
        if label == "DSOF":
            alt = input_dir / name.replace("DSOF", "OSDF")
            if alt.is_file():
                resolved.append(("OSDF", alt))
                continue
        raise FileNotFoundError(f"Missing input CSV file: {p}")
    # Enforce strict ORCA filename for the first entry
    if resolved:
        orca_expected = REFINER_FILES[0][1]
        orca_label = REFINER_FILES[0][0]
        # find resolved tuple for that label
        for lab, path in resolved:
            if lab == orca_label:
                if path.name != orca_expected:
                    raise FileNotFoundError(
                        f"ORCA CSV filename mismatch: expected exact filename {orca_expected}, found {path.name}"
                    )
                break
    return resolved


def _resolve_boltzmann_files(input_dir: Path) -> list[tuple[str, Path, Path]]:
    """
    Return list of tuples (label, router_path, baseline_path) for refiners excluding the first (ORCA/Bay).
    Filenames are matched exactly; router files are expected to be like
    results_csv_<REFINER>_router_pred_avg_all_csv_dataset_mse.csv and baseline is
    results_csv_<REFINER>_pred_avg_all_csv_dataset_mse.csv
    """
    out: list[tuple[str, Path, Path]] = []
    # skip the first refiner (ORCA / Bay)
    for label, base_name in REFINER_FILES[1:]:
        baseline = input_dir / base_name
        if not baseline.is_file():
            raise FileNotFoundError(f"Missing baseline CSV file for {label}: {baseline}")
        # construct router filename by inserting '_router' before '_pred_avg...'
        if base_name.endswith("_pred_avg_all_csv_dataset_mse.csv"):
            router_name = base_name.replace("_pred_avg_all_csv_dataset_mse.csv", "_router_pred_avg_all_csv_dataset_mse.csv")
        else:
            router_name = base_name.replace(".csv", "_router.csv")
        router = input_dir / router_name
        if not router.is_file():
            raise FileNotFoundError(f"Missing router CSV file for {label}: {router}")
        # ensure exact filenames differ
        if router.name == baseline.name:
            raise RuntimeError(f"Router and baseline filenames collide for {label}: {router.name}")
        out.append((label, router, baseline))
    return out


def _build_heatmap_matrix_for_datasets(dataset_changes: dict[str, dict[str, float]], datasets: list[str]) -> np.ndarray:
    out = np.full((len(BASE_MODELS), len(datasets)), np.nan, dtype=np.float32)
    for r_idx, model in enumerate(BASE_MODELS):
        for c_idx, ds in enumerate(datasets):
            val = float(dataset_changes.get(ds, {}).get(model, float("nan")))
            out[r_idx, c_idx] = -val if np.isfinite(val) else np.nan
    return out


def _plot_boltzmann_heatmaps(refiner_inputs: list[tuple[str, Path, Path]], output_path: Path) -> None:
    # Similar layout to _plot_heatmaps but for the 5 router refiners and SUB_DATASETS
    fig, axes = plt.subplots(1, len(refiner_inputs), figsize=(20, 3.5), dpi=600)
    fig.subplots_adjust(left=0.06, right=0.92, bottom=0.18, top=0.86, wspace=0.12)

    cmap = _build_colormap()
    norm = TwoSlopeNorm(vmin=-10.0, vcenter=0.0, vmax=10.0)

    TITLE_FS = 18
    XT_FS = 14
    YT_FS = 16
    TICK_FS = 14
    CB_LABEL_FS = 16

    last_im = None
    for idx, (title, router_path, baseline_path) in enumerate(refiner_inputs):
        rows_router = _read_table(router_path)
        rows_base = _read_table(baseline_path)
        dataset_changes_router = _extract_dataset_changes(rows_router)
        dataset_changes_base = _extract_dataset_changes(rows_base)

        data_router = _build_heatmap_matrix_for_datasets(dataset_changes_router, SUB_DATASETS)
        data_base = _build_heatmap_matrix_for_datasets(dataset_changes_base, SUB_DATASETS)

        # Strict NaN check
        if np.isnan(data_router).any():
            nan_coords = np.argwhere(np.isnan(data_router))
            nan_list = [tuple(map(int, x)) for x in nan_coords]
            raise ValueError(f"NaN detected in router data for {router_path.relative_to(PROJECT_ROOT)} at positions (model_idx, dataset_idx): {nan_list}.")

        clipped = np.clip(data_router, -10.0, 10.0)
        masked = np.ma.masked_invalid(clipped)

        # compute averages over the 5 models x 6 datasets
        try:
            base_avg = float(np.nanmean(data_base))
        except Exception:
            base_avg = float('nan')
        try:
            router_avg = float(np.nanmean(data_router))
        except Exception:
            router_avg = float('nan')

        display_title = f"{title}\n{base_avg:.2f}% -> {router_avg:.2f}%" if np.isfinite(base_avg) and np.isfinite(router_avg) else title

        ax = axes[idx]
        last_im = ax.imshow(masked, cmap=cmap, norm=norm, aspect="auto")
        ax.set_title(display_title, fontsize=TITLE_FS, pad=8, fontweight="bold")
        ax.set_xticks(range(len(SUB_DATASETS)))
        ax.set_xticklabels(SUB_DATASETS, fontsize=XT_FS, rotation=45, ha="right")
        if idx == 0:
            ax.set_yticks(range(len(BASE_MODELS)))
            ax.set_yticklabels(BASE_MODELS, fontsize=YT_FS, fontweight="bold")
        else:
            ax.set_yticks([])
        ax.tick_params(axis="x", length=0, labelsize=XT_FS)
        ax.tick_params(axis="y", length=0, labelsize=YT_FS)

    if last_im is not None:
        cbar = fig.colorbar(last_im, ax=axes, orientation="vertical", fraction=0.03, pad=0.02)
        cbar.set_ticks([-10, 0, 10])
        cbar.set_ticklabels(["<-10%", "0", ">10%"])
        cbar.ax.tick_params(labelsize=TICK_FS)
        cbar.ax.yaxis.set_label_position('left')
        cbar.ax.set_ylabel("MSE Drop (%)", fontsize=CB_LABEL_FS, rotation=90, va='center', labelpad=10)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=600, bbox_inches="tight")
    plt.close(fig)



def _build_heatmap_matrix(dataset_changes: dict[str, dict[str, float]]) -> np.ndarray:
    data = np.full((len(BASE_MODELS), len(DATASETS)), np.nan, dtype=np.float32)
    for r_idx, model in enumerate(BASE_MODELS):
        for c_idx, ds in enumerate(DATASETS):
            val = float(dataset_changes.get(ds, {}).get(model, float("nan")))
            data[r_idx, c_idx] = -val if np.isfinite(val) else np.nan
    return data


def _build_colormap() -> LinearSegmentedColormap:
    colors = [
        (0.0, "#E76F69"),
        (0.49, "#F9E5D6"),
        (0.51, "#E3F0D9"),
        (1.0, "#4D8A23"),
    ]
    cmap = LinearSegmentedColormap.from_list("mse_drop", colors)
    cmap.set_bad("#d0d0d0")
    return cmap


def _plot_heatmaps(refiner_inputs: list[tuple[str, Path]], output_path: Path) -> None:
    # Increased figure DPI and sizes for publication; fonts enlarged below.
    fig, axes = plt.subplots(1, len(refiner_inputs), figsize=(20, 3.5), dpi=600)
    fig.subplots_adjust(left=0.06, right=0.92, bottom=0.18, top=0.86, wspace=0.12)

    cmap = _build_colormap()
    norm = TwoSlopeNorm(vmin=-10.0, vcenter=0.0, vmax=10.0)

    # Font sizes (bumped)
    TITLE_FS = 18
    XT_FS = 14
    YT_FS = 16
    TICK_FS = 14
    CB_LABEL_FS = 16

    last_im = None
    for idx, (title, csv_path) in enumerate(refiner_inputs):
        rows = _read_table(csv_path)
        dataset_changes = _extract_dataset_changes(rows)
        data = _build_heatmap_matrix(dataset_changes)
        # Strict NaN check: require all 40 cells to be finite
        if np.isnan(data).any():
            nan_coords = np.argwhere(np.isnan(data))
            # convert to list of (model_idx, dataset_idx) for clarity
            nan_list = [tuple(map(int, x)) for x in nan_coords]
            raise ValueError(
                f"NaN detected in data for {csv_path.relative_to(PROJECT_ROOT)} at positions (model_idx, dataset_idx): {nan_list}."
                " Please check the CSV for missing values; averaging requires all 40 cells to be present."
            )
        # Compute average over the full 40 cells (models x datasets)
        try:
            avg_val = float(np.mean(data))
        except Exception:
            avg_val = float('nan')

        clipped = np.clip(data, -10.0, 10.0)
        masked = np.ma.masked_invalid(clipped)

        # display the average computed above (now guaranteed to be over all 40 cells)
        display_title = f"{title} ({avg_val:.2f}%)" if np.isfinite(avg_val) else f"{title}"

        ax = axes[idx]
        last_im = ax.imshow(masked, cmap=cmap, norm=norm, aspect="auto")
        ax.set_title(display_title, fontsize=TITLE_FS, pad=8, fontweight="bold")
        ax.set_xticks(range(len(DATASETS)))
        ax.set_xticklabels(DATASETS, fontsize=XT_FS, rotation=45, ha="right")
        if idx == 0:
            ax.set_yticks(range(len(BASE_MODELS)))
            ax.set_yticklabels(BASE_MODELS, fontsize=YT_FS, fontweight="bold")
        else:
            ax.set_yticks([])
        ax.tick_params(axis="x", length=0, labelsize=XT_FS)
        ax.tick_params(axis="y", length=0, labelsize=YT_FS)

    if last_im is not None:
        cbar = fig.colorbar(last_im, ax=axes, orientation="vertical", fraction=0.03, pad=0.02)
        # Keep colorbar position and ticks unchanged; place the label to the left of the colorbar.
        cbar.set_ticks([-10, 0, 10])
        cbar.set_ticklabels(["<-10%", "0", ">10%"])
        cbar.ax.tick_params(labelsize=TICK_FS)
        cbar.ax.yaxis.set_label_position('left')
        cbar.ax.set_ylabel("MSE Drop (%)", fontsize=CB_LABEL_FS, rotation=90, va='center', labelpad=10)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=600, bbox_inches="tight")
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plot refiner MSE drop heatmaps from summary CSVs.",
        allow_abbrev=False,
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default="results/plot_abla",
        help="Relative path from project root to the CSV directory",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/plots/plot_abla/plot_refiner_mse_heatmap.pdf",
        help="Relative output image path from project root",
    )
    parser.add_argument(
        "--boltzmann",
        action="store_true",
        help="When set, plot router variants for the five refiners (excluding Bay) using exact filenames",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    input_dir = (PROJECT_ROOT / str(args.input_dir)).resolve()
    output_path = (PROJECT_ROOT / str(args.output)).resolve()

    if args.boltzmann:
        refiner_inputs = _resolve_boltzmann_files(input_dir)
        # refiner_inputs is list of (label, router_path, baseline_path)
        _plot_boltzmann_heatmaps(refiner_inputs, output_path)
    else:
        refiner_inputs = _resolve_refiner_files(input_dir)
        _plot_heatmaps(refiner_inputs, output_path)
    print(f"[plot_refiner_mse_heatmap] Saved figure: {output_path}")


if __name__ == "__main__":
    main()
