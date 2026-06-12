from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _load_gate_csv(csv_path: Path) -> tuple[np.ndarray, np.ndarray]:
    rows = np.genfromtxt(str(csv_path), delimiter=",", skip_header=1)
    if rows.ndim == 1 and rows.size == 0:
        return np.array([]), np.array([])
    if rows.ndim == 1:
        rows = rows.reshape(1, -1)
    time_idx = rows[:, 0].astype(np.int64)
    conf = rows[:, 2].astype(np.float32)
    return time_idx, conf


def _plot_gate_confidence(csv_path: Path, out_dir: Path) -> Path:
    time_idx, conf = _load_gate_csv(csv_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{csv_path.stem}.pdf"

    plt.rcParams.update({
        "font.size": 14,
        "axes.labelsize": 16,
        "xtick.labelsize": 13,
        "ytick.labelsize": 13,
    })

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.set_facecolor("#f0f0f0")
    ax.grid(True, linestyle="--", linewidth=1.4, color="#808080")

    if time_idx.size > 0:
        x_vals = time_idx
        y_vals = conf
        ax.plot(x_vals, y_vals, marker="o", linestyle="", markersize=6)

    ax.set_xlabel("Time")
    ax.set_ylabel("Boltzmann Routing Weight")

    ax.annotate(
        "",
        xy=(0.0, 0.98),
        xytext=(0.0, 0.02),
        xycoords="axes fraction",
        arrowprops=dict(arrowstyle="->", linewidth=1.8, color="black"),
    )
    ax.text(
        0.02,
        0.98,
        "Foundation Model",
        transform=ax.transAxes,
        ha="left",
        va="top",
    )
    ax.text(
        0.02,
        0.02,
        "OrCA-Linear Adapter",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
    )

    fig.tight_layout()
    fig.savefig(out_path, dpi=600, format="pdf")
    plt.close(fig)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot Bay gate confidence CSVs.")
    parser.add_argument(
        "--input",
        type=str,
        default="results/details",
        help="CSV file or directory containing gate confidence CSVs.",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="results/details/plots",
        help="Output directory for PDF plots.",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    out_dir = Path(args.out_dir)

    if input_path.is_file():
        _plot_gate_confidence(input_path, out_dir)
        return

    csv_files = sorted(input_path.glob("gate_confidence_bay_*.csv"))
    for csv_path in csv_files:
        _plot_gate_confidence(csv_path, out_dir)


if __name__ == "__main__":
    main()
