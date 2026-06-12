from __future__ import annotations

import argparse
import csv
import math
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]

TARGET_DATASETS = ["ETTh1", "ETTh2", "ETTm1", "ETTm2", "Weather", "Exchange"]
BASE_MODELS = ["Chronos-2", "Moirai-2", "TiRex", "TimesFM-2.5", "Sundial"]
TARGET_INPUTS = ["xy", "all", "x", "y", "e_past"]

INPUT_LABELS = {
    "xy": "ORCA (proposed, Both Input & Pred.) ",
    "all": "All Three",
    "x": "Base Model Pred.",
    "y": "Base Model Input",
    "e_past": "Base Model Error",
}

INPUT_ORDER = {key: i for i, key in enumerate(TARGET_INPUTS)}


def _normalize_name_for_match(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(text).strip().lower())


def _parse_percent(text: str) -> float:
    raw = str(text).strip().replace("%", "")
    if not raw:
        return float("nan")
    return float(raw)


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


def _build_expected_filename(inp: str) -> str:
    """Build the exact expected CSV filename for a given refiner_input."""
    return f"results_csv_Bay_{inp}_bayesian_buf3000_batch256_pred_avg_all_csv_dataset_mse.csv"


def _resolve_input_files(input_dir: Path) -> dict[str, Path]:
    out: dict[str, Path] = {}
    for inp in TARGET_INPUTS:
        expected_name = _build_expected_filename(inp)
        p = input_dir / expected_name
        if not p.is_file():
            raise FileNotFoundError(
                f"Missing input CSV file: {expected_name}\n"
                f"Expected at: {p}\n"
                f"Available files: {[f.name for f in sorted(input_dir.glob('*.csv'))]}"
            )
        out[inp] = p
    return out


def _read_table(csv_path: Path) -> list[list[str]]:
    with open(csv_path, "r", newline="") as f:
        return list(csv.reader(f))


def _extract_model_columns(header: list[str]) -> list[tuple[str, int, int]]:
    model_cols: list[tuple[str, int, int]] = []
    for idx in range(2, len(header)):
        name = str(header[idx]).strip()
        if not name:
            continue
        if name.lower() == "datasets avg.":
            continue
        canonical = _normalize_model_header(name)
        model_cols.append((canonical, idx, idx + 1))

    by_name = {name: (name, v_idx, c_idx) for name, v_idx, c_idx in model_cols}
    ordered = [by_name[m] for m in BASE_MODELS if m in by_name]
    if len(ordered) != len(BASE_MODELS):
        missing = [m for m in BASE_MODELS if m not in by_name]
        raise ValueError(f"Missing base model columns: {missing}")
    return ordered


def _extract_dataset_changes(
    rows: list[list[str]], model_cols: list[tuple[str, int, int]]
) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for row in rows[1:]:
        if len(row) < 2:
            continue
        dataset_name = str(row[0]).strip()
        row_type = str(row[1]).strip().lower()
        if not dataset_name or row_type != "vanilla":
            continue

        change_map: dict[str, float] = {}
        for model_name, _, change_idx in model_cols:
            cell = row[change_idx] if change_idx < len(row) else ""
            try:
                change_map[model_name] = _parse_percent(cell)
            except Exception:
                change_map[model_name] = float("nan")
        out[dataset_name] = change_map
    return out


def _compute_target_six_changes(dataset_changes: dict[str, dict[str, float]]) -> dict[str, float]:
    available = [d for d in TARGET_DATASETS if d in dataset_changes]
    if not available:
        raise ValueError("No target datasets found in CSV for target-six average")

    out: dict[str, float] = {}
    for model in BASE_MODELS:
        vals: list[float] = []
        for ds in available:
            v = float(dataset_changes[ds].get(model, float("nan")))
            if math.isfinite(v):
                vals.append(v)
        out[model] = float(np.mean(vals)) if vals else float("nan")
    return out


def _extract_model_changes_for_plot(csv_path: Path) -> dict[str, float]:
    rows = _read_table(csv_path)
    if not rows:
        raise ValueError(f"CSV is empty: {csv_path}")
    model_cols = _extract_model_columns(rows[0])
    dataset_changes = _extract_dataset_changes(rows, model_cols)
    # Always compute from the target six datasets.
    return _compute_target_six_changes(dataset_changes)


def _extract_per_dataset_model_changes(csv_path: Path) -> dict[str, dict[str, float]]:
    """Return per-dataset change dict for a single CSV."""
    rows = _read_table(csv_path)
    if not rows:
        raise ValueError(f"CSV is empty: {csv_path}")
    model_cols = _extract_model_columns(rows[0])
    return _extract_dataset_changes(rows, model_cols)


def _build_ds_model_drop_data(
    input_to_ds_changes: dict[str, dict[str, dict[str, float]]]
) -> dict[str, dict[str, dict[str, float]]]:
    """Build ds_name -> model -> input -> drop_value structure.

    Returns dict[ds_name][model][input] = positive_drop_value.
    """
    out: dict[str, dict[str, dict[str, float]]] = {}
    for ds in TARGET_DATASETS:
        out[ds] = {}
        for model in BASE_MODELS:
            out[ds][model] = {}
            for inp in TARGET_INPUTS:
                ds_map = input_to_ds_changes[inp]
                if ds in ds_map and model in ds_map[ds]:
                    val = float(ds_map[ds][model])
                    out[ds][model][inp] = -val if math.isfinite(val) else 0.0
                else:
                    out[ds][model][inp] = 0.0
    return out


def _plot_input_ladder(
    ds_model_data: dict[str, dict[str, dict[str, float]]],
    output_path: Path,
) -> None:
    n_datasets = len(TARGET_DATASETS)
    n_inputs = len(TARGET_INPUTS)
    n_models = len(BASE_MODELS)

    fig_width_cm = 15
    fig_height_cm = 4.5
    fig, axes = plt.subplots(
        1,
        n_datasets,
        figsize=(fig_width_cm / 2.54, fig_height_cm / 2.54),
        dpi=600,
        gridspec_kw={"wspace": 0.3, "hspace": 0.25},
    )
    # Reserve bottom area for MSE label + legend, no right margin needed
    fig.subplots_adjust(left=0.2, right=0.95, top=0.75, bottom=0.18)
    if n_datasets == 1:
        axes = [axes]

    # Color scheme from plot_abla_structure.py (5 models).
    model_colors = ["#FFE8A7", "#FF9494", "#9CC2E0", "#A6D9A6", "#D1A6D6"]

    # Determine global x-axis limit considering both positive and negative drops.
    global_vmax_pos = 0.0
    global_vmax_neg = 0.0
    for ds in TARGET_DATASETS:
        for inp in TARGET_INPUTS:
            total = 0.0
            for model in BASE_MODELS:
                v = ds_model_data[ds][model][inp]
                if math.isfinite(v):
                    total += v
            avg = total / n_models
            global_vmax_pos = max(global_vmax_pos, avg)
            global_vmax_neg = min(global_vmax_neg, avg)
    x_margin = max(global_vmax_pos, abs(global_vmax_neg)) * 0.2
    xlim_left = global_vmax_neg - x_margin if global_vmax_neg < 0 else -x_margin
    xlim_right = global_vmax_pos + x_margin

    # Compute overall 30-value average (6 datasets x 5 models) per variant for y-axis labels.
    variant_overall_avg: dict[str, float] = {}
    for inp in TARGET_INPUTS:
        vals = []
        for ds in TARGET_DATASETS:
            for model in BASE_MODELS:
                v = ds_model_data[ds][model][inp]
                if math.isfinite(v):
                    vals.append(v)
        variant_overall_avg[inp] = float(np.mean(vals)) if vals else 0.0

    for ds_idx, (ax, ds_name) in enumerate(zip(axes, TARGET_DATASETS)):
        y_positions = np.arange(n_inputs)

        # Zero reference line (match x-axis style: black, same thickness).
        ax.axvline(x=0, color="#000000", linewidth=0.8, linestyle="-", zorder=1)

        for inp_idx, inp in enumerate(TARGET_INPUTS):
            # Compute per-model contributions (positive = improvement, negative = degradation).
            model_vals = []
            for model in BASE_MODELS:
                v = ds_model_data[ds_name][model][inp]
                model_vals.append(v if math.isfinite(v) else 0.0)

            # Stacked horizontal bar: positive to the right, negative to the left.
            pos_left = 0.0
            neg_left = 0.0
            for m_idx, val in enumerate(model_vals):
                scaled_val = val / n_models
                if scaled_val >= 0:
                    ax.barh(
                        inp_idx,
                        scaled_val,
                        left=pos_left,
                        height=0.7,
                        color=model_colors[m_idx],
                        edgecolor="none",
                        zorder=2,
                    )
                    pos_left += scaled_val
                else:
                    ax.barh(
                        inp_idx,
                        abs(scaled_val),
                        left=neg_left - abs(scaled_val),
                        height=0.7,
                        color=model_colors[m_idx],
                        edgecolor="none",
                        zorder=2,
                    )
                    neg_left -= abs(scaled_val)

            # Average of 5 models for this variant+dataset, always written on the right.
            avg_val = float(np.mean(model_vals))
            if avg_val >= 0:
                label_x = pos_left + max(global_vmax_pos, 0.5) * 0.02
            else:
                label_x = max(global_vmax_pos, 0.5) * 0.03
            ax.text(
                label_x,
                inp_idx,
                f"{avg_val:.1f}%",
                ha="left",
                va="center",
                fontsize=6,
                color="#333333",
            )

        # Y-axis labels only on the leftmost subplot.
        if ds_idx == 0:
            y_labels = []
            for inp in TARGET_INPUTS:
                avg = variant_overall_avg[inp]
                sign = "" if avg >= 0 else "-"
                display_val = abs(avg)
                raw_label = INPUT_LABELS[inp]
                display_label = raw_label if len(raw_label) <= 28 else raw_label[:20] + "\n" + raw_label[20:]
                y_labels.append(f"{display_label} ({sign}{display_val:.1f}%)")
            ax.set_yticks(range(n_inputs))
            ax.set_yticklabels(y_labels, fontsize=6.5, fontweight="bold")
        else:
            ax.set_yticks([])

        ax.set_title(ds_name, fontsize=6.5, pad=6)
        # Set x-axis limits: Exchange uses -5, others use -1
        ds_xlim_left = -5.0 if ds_name == "Exchange" else -1.0
        ax.set_xlim(ds_xlim_left, xlim_right)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.spines["bottom"].set_linewidth(0.8)
        ax.tick_params(axis="y", length=0)
        ax.tick_params(axis="x", labelsize=6)

    # Shared legend with gray rounded box background (horizontal, at bottom)
    legend_elements = [
        plt.Rectangle((0, 0), 1, 1, color=model_colors[i], edgecolor="none")
        for i in range(n_models)
    ]
    leg = fig.legend(
        legend_elements,
        BASE_MODELS,
        loc="upper center",
        bbox_to_anchor=(0.5, 1),
        bbox_transform=fig.transFigure,
        fontsize=6.5,
        frameon=True,
        fancybox=True,
        facecolor="#F3F3F3",
        edgecolor="black",
        ncol=5,
        handlelength=1.2,
        handleheight=0.6,
        columnspacing=1.5,
        borderpad=0.2,
        labelspacing=0.1,
        handletextpad=0.4,
    )
    leg.get_frame().set_linewidth(0.8)
    leg.get_frame().set_boxstyle("round,pad=0.3")
    leg.get_frame().set_alpha(0.5)
    # X-axis label centered at the bottom of the entire figure (above legend)
    fig.text(0.5, 0, "MSE Drop (%) (positive is better)", ha="center", va="bottom", fontsize=6.5, fontweight="bold")

    fig.suptitle(
        "Ablation on Adapter Input",
        fontsize=6.5,
        x=0.1,
        y=0.83,
        fontweight="bold",
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Save primary image (respect extension) and also export a 600 dpi PDF for publication.
    fig.savefig(output_path, bbox_inches="tight", dpi=600)
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Plot ablation on refiner input from summary CSVs. "
            "Horizontal ladder plot: 6 datasets x 5 input variants."
        ),
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
        default="results/plots/plot_abla/plot_abla_input_mse.pdf",
        help="Relative output image path from project root",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    input_dir = (PROJECT_ROOT / str(args.input_dir)).resolve()
    output_path = (PROJECT_ROOT / str(args.output)).resolve()

    input_files = _resolve_input_files(input_dir)
    print("[plot_abla_input] Matched files:")
    for inp in TARGET_INPUTS:
        print(f"  - {inp}: {input_files[inp].relative_to(PROJECT_ROOT)}")

    input_to_ds_changes: dict[str, dict[str, dict[str, float]]] = {}
    input_to_avg: dict[str, dict[str, float]] = {}
    print("[plot_abla_input] Per-dataset and six-dataset avg drops:")
    for inp in TARGET_INPUTS:
        csv_path = input_files[inp]
        ds_changes = _extract_per_dataset_model_changes(csv_path)
        input_to_ds_changes[inp] = ds_changes
        avg_changes = _extract_model_changes_for_plot(csv_path)
        input_to_avg[inp] = avg_changes
        desc = ", ".join(f"{m}={avg_changes.get(m, float('nan')):.2f}%" for m in BASE_MODELS)
        print(f"  - {inp}: {desc}")

    ds_model_data = _build_ds_model_drop_data(input_to_ds_changes)
    _plot_input_ladder(ds_model_data, output_path)
    print(f"[plot_abla_input] Saved figure: {output_path}")


if __name__ == "__main__":
    main()
