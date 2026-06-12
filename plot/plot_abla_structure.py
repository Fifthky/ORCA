from __future__ import annotations

import argparse
import csv
import math
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]

TARGET_DATASETS = ["ETTh1", "ETTh2", "ETTm1", "ETTm2", "Weather", "Exchange"]
BASE_MODELS = ["Chronos-2", "Moirai-2", "TiRex", "TimesFM-2.5", "Sundial"]


@dataclass(frozen=True)
class VariantSpec:
    key: str
    title: str
    color: str
    required_tokens: tuple[str, ...]
    forbidden_tokens: tuple[str, ...] = ()


VARIANTS: list[VariantSpec] = [
    VariantSpec(
        key="proposed",
        title="ORCA (proposed)",
        color="#FFE8A7",
        required_tokens=(
            "results_csv",
            "bay",
            "xy",
            "bayesian",
            "buf3000",
            "batch256",
            "pred_avg",
            "all_csv_dataset",
            "mse",
        ),
        forbidden_tokens=("gate_open", "ci", "plain", "linear"),
    ),
    VariantSpec(
        key="wo_router",
        title="w/o Boltzmann Router",
        color="#FF9494",
        required_tokens=(
            "results_csv",
            "bay",
            "xy",
            "bayesian",
            "buf3000",
            "batch256",
            "gate_open",
            "pred_avg",
            "all_csv_dataset",
            "mse",
        ),
    ),
    VariantSpec(
        key="wo_channel_mix",
        title="w/o Channel Mixing",
        color="#9CC2E0",
        required_tokens=(
            "results_csv",
            "bay",
            "xy",
            "bayesian",
            "buf3000",
            "batch256",
            "ci",
            "pred_avg",
            "all_csv_dataset",
            "mse",
        ),
    ),
    VariantSpec(
        key="wo_bayesian_prior",
        title="w/o bayesian prior loss",
        color="#A6D9A6",
        required_tokens=(
            "results_csv",
            "bay",
            "xy",
            "plain",
            "buf3000",
            "batch256",
            "pred_avg",
            "all_csv_dataset",
            "mse",
        ),
    ),
    VariantSpec(
        key="wo_decay_buffer",
        title="w/o decay buffer",
        color="#D1A6D6",
        required_tokens=(
            "results_csv",
            "linear",
            "xy",
            "pred_avg",
            "all_csv_dataset",
            "mse",
        ),
    ),
]


def _normalize_name_for_match(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(text).strip().lower())


def _parse_percent(text: str) -> float:
    raw = str(text).strip().replace("%", "")
    if not raw:
        return float("nan")
    return float(raw)


def _discover_csv_files(input_dir: Path) -> list[Path]:
    files = [p for p in sorted(input_dir.glob("*.csv")) if p.is_file()]
    return files


def _is_match(name_lower: str, spec: VariantSpec) -> bool:
    if any(tok not in name_lower for tok in spec.required_tokens):
        return False
    if any(tok in name_lower for tok in spec.forbidden_tokens):
        return False
    return True


def _resolve_variant_files(input_dir: Path) -> dict[str, Path]:
    files = _discover_csv_files(input_dir)
    if not files:
        raise FileNotFoundError(f"No CSV files found in: {input_dir}")

    matched: dict[str, Path] = {}
    for spec in VARIANTS:
        hit: Path | None = None
        for p in files:
            name_lower = p.name.lower()
            if _is_match(name_lower, spec):
                hit = p
                break
        if hit is None:
            raise FileNotFoundError(
                f"Cannot find file for variant={spec.key}. required={spec.required_tokens}, forbidden={spec.forbidden_tokens}"
            )
        matched[spec.key] = hit
    return matched


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

    # Preserve canonical base model ordering.
    by_name = {name: (name, value_idx, change_idx) for name, value_idx, change_idx in model_cols}
    ordered = [by_name[m] for m in BASE_MODELS if m in by_name]
    if len(ordered) != len(BASE_MODELS):
        missing = [m for m in BASE_MODELS if m not in by_name]
        raise ValueError(f"Missing base model columns: {missing}")
    return ordered


def _extract_dataset_changes(rows: list[list[str]], model_cols: list[tuple[str, int, int]]) -> dict[str, dict[str, float]]:
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


def _extract_models_avg_change_row(rows: list[list[str]], model_cols: list[tuple[str, int, int]]) -> dict[str, float]:
    for row in rows:
        if len(row) < 2:
            continue
        if str(row[0]).strip().lower() != "models avg.":
            continue
        if str(row[1]).strip().lower() != "change":
            continue

        out: dict[str, float] = {}
        for model_name, _, change_idx in model_cols:
            cell = row[change_idx] if change_idx < len(row) else ""
            try:
                out[model_name] = _parse_percent(cell)
            except Exception:
                out[model_name] = float("nan")
        return out
    raise ValueError("Cannot find final Models Avg./Change row")


def _compute_model_change_from_target_six(
    dataset_changes: dict[str, dict[str, float]],
) -> tuple[dict[str, float], bool]:
    available = [d for d in TARGET_DATASETS if d in dataset_changes]
    has_extra = len(set(dataset_changes.keys()) - set(TARGET_DATASETS)) > 0
    if not available:
        return {}, has_extra

    out: dict[str, float] = {}
    for model in BASE_MODELS:
        vals: list[float] = []
        for ds in available:
            v = float(dataset_changes[ds].get(model, float("nan")))
            if math.isfinite(v):
                vals.append(v)
        out[model] = float(np.mean(vals)) if vals else float("nan")
    return out, has_extra


def _extract_model_changes_for_plot(csv_path: Path) -> dict[str, float]:
    rows = _read_table(csv_path)
    if not rows:
        raise ValueError(f"CSV is empty: {csv_path}")

    model_cols = _extract_model_columns(rows[0])
    dataset_changes = _extract_dataset_changes(rows, model_cols)
    six_avg, has_extra = _compute_model_change_from_target_six(dataset_changes)

    # Requirement: if extra datasets are present, force six-dataset average.
    if has_extra and six_avg:
        return six_avg

    # Prefer six-dataset average whenever possible; otherwise fallback to final summary row.
    if six_avg:
        return six_avg
    return _extract_models_avg_change_row(rows, model_cols)


def _format_pct(v: float) -> str:
    if not math.isfinite(v):
        return "nan"
    return f"{v:.2f}%"


def _plot_ablation_bars(
    variant_to_changes: dict[str, dict[str, float]],
    output_path: Path,
    *,
    fig_title: str | None = None,
) -> None:
    n = len(VARIANTS)
    fig, axes = plt.subplots(1, n, figsize=(3.2 * n, 4), dpi=600, constrained_layout=True)
    if n == 1:
        axes = [axes]

    for ax, spec in zip(axes, VARIANTS):
        model_change = variant_to_changes[spec.key]
        # CSV stores improvement as negative delta for MSE. Convert to positive "drop ratio".
        drops = np.array([-float(model_change.get(m, float("nan"))) for m in BASE_MODELS], dtype=np.float64)
        drops = np.nan_to_num(drops, nan=0.0)

        x = np.arange(len(BASE_MODELS))
        bars = ax.bar(x, drops, color=spec.color, edgecolor="none", width=0.72)

        for b, v in zip(bars, drops):
            ax.text(
                b.get_x() + b.get_width() / 2.0,
                b.get_height() + max(0.02, float(np.max(drops) * 0.02 if np.max(drops) > 0 else 0.02)),
                f"{v:.2f}",
                ha="center",
                va="bottom",
                fontsize=15,
            )

        overall_drop = float(np.mean(drops)) if len(drops) > 0 else float("nan")
        ax.set_title(f"{spec.title}\nAvg {overall_drop:.2f}%", fontsize=16, pad=12, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(BASE_MODELS, fontsize=15, rotation=25, ha="right")
        ax.tick_params(axis="y", labelsize=15)

        if spec.key == VARIANTS[0].key:
            ax.set_ylabel("MSE Drop Ratio (%)", fontsize=16)

        # Match requested style: no grid, only left and bottom spines.
        ax.grid(False)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_linewidth(1.2)
        ax.spines["bottom"].set_linewidth(1.2)

    # Unify y-axis range across all subplots using the tallest subplot range.
    max_y = max(ax.get_ylim()[1] for ax in axes)
    for ax in axes:
        ax.set_ylim(0, max_y)

    if fig_title:
        fig.suptitle(fig_title, fontsize=18)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Plot ablation structure bars from summary CSVs under results/plot_abla. "
            "Each subplot is one ablation variant and bars are drop ratios over five base models."
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
        default="results/plots/plot_abla/plot_abla_structure_mse.pdf",
        help="Relative output image path from project root",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    input_dir = (PROJECT_ROOT / str(args.input_dir)).resolve()
    output_path = (PROJECT_ROOT / str(args.output)).resolve()

    variant_files = _resolve_variant_files(input_dir)
    print("[plot_abla_structure] Matched files:")
    for spec in VARIANTS:
        print(f"  - {spec.key}: {variant_files[spec.key].relative_to(PROJECT_ROOT)}")

    variant_to_changes: dict[str, dict[str, float]] = {}
    print("[plot_abla_structure] Model drops (negative change in CSV -> positive drop):")
    for spec in VARIANTS:
        csv_path = variant_files[spec.key]
        changes = _extract_model_changes_for_plot(csv_path)
        variant_to_changes[spec.key] = changes
        desc = ", ".join(f"{m}={_format_pct(changes.get(m, float('nan')))}" for m in BASE_MODELS)
        print(f"  - {spec.key}: {desc}")

    _plot_ablation_bars(
        variant_to_changes,
        output_path,
        fig_title="Structure Ablation (MSE Drop on ETTh1/ETTh2/ETTm1/ETTm2/Weather/Exchange)",
    )
    print(f"[plot_abla_structure] Saved figure: {output_path}")


if __name__ == "__main__":
    main()
