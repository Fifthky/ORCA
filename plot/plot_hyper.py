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
TARGET_BUFFERS = [2000, 3000, 4000]
TARGET_BATCHES = [128, 256, 512]


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


def _extract_buffer_batch_from_name(name: str) -> tuple[int | None, int | None]:
    lower = str(name).lower()
    buf_m = re.search(r"buf(\d+)", lower)
    bsz_m = re.search(r"batch(\d+)", lower)
    buf = int(buf_m.group(1)) if buf_m else None
    bsz = int(bsz_m.group(1)) if bsz_m else None
    return buf, bsz


def _is_standard_hyper_variant_file(path: Path) -> bool:
    name = path.name.lower()
    required = [
        "results_csv",
        "bay",
        "xy",
        "bayesian",
        "pred_avg",
        "all_csv_dataset",
        "mse",
    ]
    forbidden = ["gate_open", "ci", "plain", "linear"]

    if any(tok not in name for tok in required):
        return False
    if any(tok in name for tok in forbidden):
        return False
    if "ema" in name or "rt" in name:
        return False

    buf, bsz = _extract_buffer_batch_from_name(name)
    if buf not in set(TARGET_BUFFERS):
        return False
    if bsz not in set(TARGET_BATCHES):
        return False
    return True


def _resolve_hyper_files(input_dir: Path) -> dict[tuple[int, int], Path]:
    files = _discover_csv_files(input_dir)
    if not files:
        raise FileNotFoundError(f"No CSV files found in: {input_dir}")

    out: dict[tuple[int, int], Path] = {}
    for p in files:
        if not _is_standard_hyper_variant_file(p):
            continue
        buf, bsz = _extract_buffer_batch_from_name(p.name)
        if buf is None or bsz is None:
            continue
        key = (int(buf), int(bsz))
        if key in out:
            raise RuntimeError(f"Duplicate files for buf={buf}, batch={bsz}: {out[key].name} and {p.name}")
        out[key] = p

    missing = [(b, z) for b in TARGET_BUFFERS for z in TARGET_BATCHES if (b, z) not in out]
    if missing:
        raise FileNotFoundError(f"Missing hyper CSV files for combinations: {missing}")
    return out


def _float_to_tag(value: float) -> str:
    s = str(value)
    s = s.replace(".", "")
    return s


def _build_exact_filename_for_params(
    buf: int,
    bsz: int,
    *,
    ema_tag: str | None = None,
    rt_tag: str | None = None,
    ema_first: bool = True,
) -> str:
    parts = [f"results_csv_Bay_xy_bayesian_buf{buf}_batch{bsz}"]
    suffix_parts: list[str] = []
    if ema_first:
        if ema_tag:
            suffix_parts.append(f"ema{ema_tag}")
        if rt_tag:
            suffix_parts.append(f"rt{rt_tag}")
    else:
        if rt_tag:
            suffix_parts.append(f"rt{rt_tag}")
        if ema_tag:
            suffix_parts.append(f"ema{ema_tag}")
    parts.extend(suffix_parts)
    parts.append("pred_avg_all_csv_dataset_mse")
    return "_".join(parts)


def _resolve_exact_csv_file(input_dir: Path, stem_candidates: list[str]) -> Path:
    matches: list[Path] = []
    for stem in dict.fromkeys(stem_candidates):
        candidate = input_dir / f"{stem}.csv"
        if candidate.exists():
            matches.append(candidate)
    if not matches:
        raise FileNotFoundError(f"Expected exact CSV file not found. Tried: {stem_candidates}")
    if len(matches) > 1:
        names = ", ".join(p.name for p in matches)
        raise RuntimeError(f"Duplicate exact CSV files found: {names}")
    return matches[0]


def _load_changes_from_file(csv_path: Path) -> dict[str, float]:
    return _extract_model_changes_for_plot(csv_path)


def _resolve_left_panel_files(input_dir: Path) -> tuple[list[Path], list[float], list[float]]:
    ema_values = [0.05, 0.2, 0.5]
    rt_values = [0.05, 0.1, 0.5]
    left_buf = 3000
    left_bsz = 256

    files: list[Path] = []
    for ema in ema_values:
        ema_tag = None if abs(ema - 0.2) < 1e-12 else _float_to_tag(ema)
        for rt in rt_values:
            rt_tag = None if abs(rt - 0.1) < 1e-12 else _float_to_tag(rt)
            candidates = [
                _build_exact_filename_for_params(left_buf, left_bsz, ema_tag=ema_tag, rt_tag=rt_tag, ema_first=True),
                _build_exact_filename_for_params(left_buf, left_bsz, ema_tag=ema_tag, rt_tag=rt_tag, ema_first=False),
            ]
            files.append(_resolve_exact_csv_file(input_dir, candidates))

    return files, ema_values, rt_values


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
    # Always compute from the target six datasets to satisfy the requirement.
    return _compute_target_six_changes(dataset_changes)


def _build_model_line_matrix(hyper_to_changes: dict[tuple[int, int], dict[str, float]]) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for model in BASE_MODELS:
        mat = np.full((len(TARGET_BATCHES), len(TARGET_BUFFERS)), np.nan, dtype=np.float64)
        for r, batch in enumerate(TARGET_BATCHES):
            for c, buf in enumerate(TARGET_BUFFERS):
                change = float(hyper_to_changes[(buf, batch)].get(model, float("nan")))
                # CSV change is negative for improvement; convert to positive drop ratio.
                mat[r, c] = -change
        out[model] = mat
    return out


def _build_left_line_matrix(files: list[Path], ema_values: list[float], rt_values: list[float]) -> dict[str, np.ndarray]:
    left_matrix = {model: np.full((len(ema_values), len(rt_values)), np.nan, dtype=np.float64) for model in BASE_MODELS}
    index = 0
    for i in range(len(ema_values)):
        for j in range(len(rt_values)):
            changes = _load_changes_from_file(files[index])
            for model in BASE_MODELS:
                left_matrix[model][i, j] = -float(changes.get(model, float("nan")))
            index += 1
    return left_matrix


def _plot_combined_lines(input_dir: Path, output_path: Path) -> None:
    left_files, ema_values, rt_values = _resolve_left_panel_files(input_dir)

    model_colors = ["#FFE8A7", "#FF9494", "#9CC2E0", "#A6D9A6", "#D1A6D6"]
    model_markers = ["o", "s", "^", "D", "*"]

    left_matrix = _build_left_line_matrix(left_files, ema_values, rt_values)

    # Use two equal-width plot columns and place the legend in a separate axis
    # so the main plots can grow while the plot-to-legend gap stays configurable.
    fig = plt.figure(figsize=(11.2, 2), dpi=600, constrained_layout=False)
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1], wspace=0.03)
    ax_left = fig.add_subplot(gs[0, 0])
    ax_right = fig.add_subplot(gs[0, 1])
    ax_legend = fig.add_axes([0.92, 0.18, 0.085, 0.64])
    ax_legend.axis("off")

    left_points = len(ema_values) * len(rt_values)
    x_left = list(range(left_points))
    bottom_labels_left = [str(v) for v in rt_values] * len(ema_values)
    top_centers_left = [i * len(rt_values) + (len(rt_values) - 1) / 2 for i in range(len(ema_values))]

    left_all_y: list[float] = []
    for m_idx, model in enumerate(BASE_MODELS):
        y: list[float] = []
        mat = left_matrix[model]
        for i in range(len(ema_values)):
            for j in range(len(rt_values)):
                y.append(float(mat[i, j]))
        left_all_y.extend(y)
        ax_left.plot(
            x_left,
            y,
            label=model,
            color=model_colors[m_idx],
            marker=model_markers[m_idx],
            linestyle="-",
            linewidth=2,
            markersize=8,
        )

    ax_left.set_xticks(x_left)
    ax_left.set_xticklabels(bottom_labels_left, fontsize=10)
    ax_left_top = ax_left.twiny()
    ax_left_top.set_xlim(ax_left.get_xlim())
    ax_left_top.set_xticks(top_centers_left)
    ax_left_top.set_xticklabels([str(v) for v in ema_values], fontsize=10)
    ax_left_top.set_xlabel("EMA Momentum $\\alpha$", fontsize=12, labelpad=6)
    ax_left_top.xaxis.set_tick_params(pad=6)
    ax_left.set_xlim(-0.5, left_points - 0.5)
    ax_left.set_xlabel("Routing Temperature $\\tau$", fontsize=12)
    ax_left.set_ylabel("MSE Drop (%)", fontsize=12)
    ax_left.grid(True, linestyle="--", color="#BBBBBB", linewidth=0.8)

    hyper_files = _resolve_hyper_files(input_dir)
    hyper_to_changes: dict[tuple[int, int], dict[str, float]] = {}
    for key, p in hyper_files.items():
        hyper_to_changes[key] = _extract_model_changes_for_plot(p)

    model_to_mat = _build_model_line_matrix(hyper_to_changes)

    # Flatten and plot on ax_right similar to previous implementation
    n_points = len(TARGET_BUFFERS) * len(TARGET_BATCHES)
    x = list(range(n_points))
    right_all_y: list[float] = []
    for m_idx, model in enumerate(BASE_MODELS):
        mat = model_to_mat[model]
        y: list[float] = []
        for c in range(len(TARGET_BUFFERS)):
            for r in range(len(TARGET_BATCHES)):
                y.append(float(mat[r, c]))
        right_all_y.extend(y)
        ax_right.plot(x, y, label=model, color=model_colors[m_idx], marker=model_markers[m_idx], linewidth=2)

    bottom_labels: list[str] = []
    for _ in TARGET_BUFFERS:
        bottom_labels.extend([str(b) for b in TARGET_BATCHES])
    ax_right.set_xticks(x)
    ax_right.set_xticklabels(bottom_labels, fontsize=10)
    centers = [i * len(TARGET_BATCHES) + (len(TARGET_BATCHES) - 1) / 2 for i in range(len(TARGET_BUFFERS))]
    ax_top = ax_right.twiny()
    ax_top.set_xlim(ax_right.get_xlim())
    ax_top.set_xticks(centers)
    ax_top.set_xticklabels([str(b) for b in TARGET_BUFFERS], fontsize=10)
    ax_top.xaxis.set_tick_params(pad=6)
    ax_right.set_xlim(-0.5, n_points - 0.5)
    # Set separate bottom and top axis labels for right panel
    ax_right.set_xlabel("Batch Size", fontsize=12)
    ax_top.set_xlabel("Buffer Length", fontsize=12, labelpad=6)
    # Hide right panel Y labels but keep y-axis gridlines visible
    ax_right.tick_params(axis="y", which="both", left=False, labelleft=False)
    ax_right.xaxis.grid(True, linestyle="--", color="#BBBBBB", linewidth=0.8)
    ax_right.yaxis.grid(True, linestyle="--", color="#BBBBBB", linewidth=0.8)
    handles, labels = ax_right.get_legend_handles_labels()
    ax_legend.legend(handles, labels, loc="center", fontsize=12, frameon=True)

    # Compute shared Y limits from plotted data
    try:
        combined = np.array(left_all_y + right_all_y, dtype=float)
        ymin = float(np.nanmin(combined))
        ymax = float(np.nanmax(combined))
        if math.isfinite(ymin) and math.isfinite(ymax) and ymax > ymin:
            pad = max(1e-6, (ymax - ymin) * 0.05)
            ax_left.set_ylim(ymin - pad, ymax + pad)
            ax_right.set_ylim(ymin - pad, ymax + pad)
    except Exception:
        pass

    fig.subplots_adjust(left=0.075, right=0.885, top=0.92, bottom=0.22)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def _plot_buffer_batch_lines(model_to_mat: dict[str, np.ndarray], output_path: Path) -> None:
    # Convert matrix data to a single 9-point line plot.
    # X ordering: for each buffer in TARGET_BUFFERS, list TARGET_BATCHES (128,256,512).
    n_points = len(TARGET_BUFFERS) * len(TARGET_BATCHES)
    x = list(range(n_points))

    # Colors and markers borrowed from plot_abla_input.py
    model_colors = ["#FFE8A7", "#FF9494", "#9CC2E0", "#A6D9A6", "#D1A6D6"]
    model_markers = ["o", "s", "^", "D", "*"]

    fig, ax = plt.subplots(figsize=(12, 2), dpi=600, constrained_layout=True)
    ax.grid(True, linestyle="--", color="#BBBBBB", linewidth=1.0)

    for m_idx, model in enumerate(BASE_MODELS):
        mat = model_to_mat[model]
        # Flatten in order: for each buffer (col), for each batch (row)
        y: list[float] = []
        for c in range(len(TARGET_BUFFERS)):
            for r in range(len(TARGET_BATCHES)):
                y.append(float(mat[r, c]))

        ax.plot(
            x,
            y,
            label=model,
            color=model_colors[m_idx],
            marker=model_markers[m_idx],
            linestyle="-",
            linewidth=2,
            markersize=10,
        )

    # Bottom ticks: repeated batch sizes per buffer group (128,256,512, 128,256,512, ...)
    bottom_labels: list[str] = []
    for _ in TARGET_BUFFERS:
        bottom_labels.extend([str(b) for b in TARGET_BATCHES])
    ax.set_xticks(x)
    ax.set_xticklabels(bottom_labels, fontsize=12)

    # Top ticks: buffer labels centered above each group of 3 points
    centers = [i * len(TARGET_BATCHES) + (len(TARGET_BATCHES) - 1) / 2 for i in range(len(TARGET_BUFFERS))]
    ax_top = ax.twiny()
    ax_top.set_xlim(ax.get_xlim())
    ax_top.set_xticks(centers)
    ax_top.set_xticklabels([str(b) for b in TARGET_BUFFERS], fontsize=12)
    ax_top.xaxis.set_tick_params(pad=6)

    ax.set_xlim(-0.5, n_points - 0.5)
    ax.set_xlabel("Batch Size (bottom) — Buffer Length (top)", fontsize=12)
    ax.set_ylabel("MSE Drop (%)", fontsize=12)
    ax.grid(True, linestyle="--", color="#BBBBBB", linewidth=1.0)

    # Legend placed to the right outside the axes (keep style identical)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=12, frameon=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight", dpi=600)
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Plot hyper-parameter line charts for Bay-XY-bayesian variants. "
            "Each subplot is one 5-model line chart."
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
        default="results/plots/plot_abla/plot_hyper_mse.pdf",
        help="Relative output image path from project root",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    input_dir = (PROJECT_ROOT / str(args.input_dir)).resolve()
    output_path = (PROJECT_ROOT / str(args.output)).resolve()

    hyper_files = _resolve_hyper_files(input_dir)
    print("[plot_hyper] Matched files:")
    for batch in TARGET_BATCHES:
        for buf in TARGET_BUFFERS:
            p = hyper_files[(buf, batch)]
            print(f"  - batch={batch}, buf={buf}: {p.relative_to(PROJECT_ROOT)}")

    _plot_combined_lines(input_dir, output_path)
    print(f"[plot_hyper] Saved figure: {output_path}")


if __name__ == "__main__":
    main()
