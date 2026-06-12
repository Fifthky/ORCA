from __future__ import annotations

import argparse
import urllib.request
from pathlib import Path

from data.csv_dataset import CSV_DATASET_SPECS


DEFAULT_CACHE_DIR = Path("data/data_cache")


CSV_DOWNLOAD_URLS: dict[str, str] = {
    "ETTh1": "https://raw.githubusercontent.com/thuml/Autoformer/main/dataset/ETT-small/ETTh1.csv",
    "ETTh2": "https://raw.githubusercontent.com/thuml/Autoformer/main/dataset/ETT-small/ETTh2.csv",
    "ETTm1": "https://raw.githubusercontent.com/thuml/Autoformer/main/dataset/ETT-small/ETTm1.csv",
    "ETTm2": "https://raw.githubusercontent.com/thuml/Autoformer/main/dataset/ETT-small/ETTm2.csv",
    "Exchange": "https://raw.githubusercontent.com/thuml/Autoformer/main/dataset/exchange_rate/exchange_rate.csv",
    "Weather": "https://raw.githubusercontent.com/thuml/Autoformer/main/dataset/weather/weather.csv",
    "Electricity": "https://raw.githubusercontent.com/thuml/Autoformer/main/dataset/electricity/electricity.csv",
    "Traffic": "https://raw.githubusercontent.com/thuml/Autoformer/main/dataset/traffic/traffic.csv",
}


def resolve_cached_csv_path(dataset_name: str, cache_dir: str | Path = DEFAULT_CACHE_DIR) -> Path:
    if dataset_name not in CSV_DATASET_SPECS:
        raise ValueError(
            f"Unsupported dataset_name={dataset_name!r}. Supported: {sorted(CSV_DATASET_SPECS.keys())}"
        )
    rel_path = CSV_DATASET_SPECS[dataset_name]["relative_path"]
    return Path(cache_dir).expanduser() / rel_path


def ensure_dataset_csv(
    dataset_name: str,
    *,
    cache_dir: str | Path = DEFAULT_CACHE_DIR,
    force: bool = False,
    timeout: int = 120,
) -> Path:
    if dataset_name not in CSV_DOWNLOAD_URLS:
        raise ValueError(
            f"No download URL configured for dataset_name={dataset_name!r}. Supported: {sorted(CSV_DOWNLOAD_URLS.keys())}"
        )

    target_path = resolve_cached_csv_path(dataset_name, cache_dir=cache_dir)
    target_path.parent.mkdir(parents=True, exist_ok=True)

    if target_path.exists() and not force:
        return target_path

    url = CSV_DOWNLOAD_URLS[dataset_name]
    with urllib.request.urlopen(url, timeout=timeout) as response:
        data = response.read()
    target_path.write_bytes(data)
    return target_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Download benchmark CSV files into data/data_cache")
    parser.add_argument(
        "--dataset",
        type=str,
        default="all",
        help=f"Dataset name or 'all'. Supported: {', '.join(sorted(CSV_DOWNLOAD_URLS.keys()))}",
    )
    parser.add_argument("--cache_dir", type=str, default=str(DEFAULT_CACHE_DIR), help="CSV cache directory")
    parser.add_argument("--force", action="store_true", help="Force re-download even if file exists")
    parser.add_argument("--timeout", type=int, default=120, help="HTTP timeout in seconds")
    args = parser.parse_args()

    if args.dataset == "all":
        dataset_names = sorted(CSV_DOWNLOAD_URLS.keys())
    else:
        dataset_names = [args.dataset]

    for name in dataset_names:
        path = ensure_dataset_csv(
            name,
            cache_dir=args.cache_dir,
            force=bool(args.force),
            timeout=int(args.timeout),
        )
        print(f"[download_CSV] ready: {name} -> {path}")


if __name__ == "__main__":
    main()
