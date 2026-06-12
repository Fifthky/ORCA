from __future__ import annotations

from pathlib import Path


TSFM_MODEL_PATH_PREFIX = Path("path_to_your_model")

TSFM_MODEL_PATHS: dict[str, str] = {
    "chronos-2": "chronos-2",
    "moirai-2": "moirai-2.0-R-small",
    "tirex": "TiRex-1.1-gifteval",
    "timesfm-2": "timesfm-2.5-200m-pytorch",
    "sundial": "sundial-base-128m",
    "moirai-1-small": "moirai-1.0-R-small",
    "moirai-1-base": "moirai-1.1-R-base",
    "moirai-1-large": "moirai-1.1-R-large",
}

TSFM_MODEL_ORDER: list[str] = [
    "chronos-2",
    "moirai-2",
    "tirex",
    "timesfm-2",
    "sundial",
]

TSFM_MODEL_ALIASES: dict[str, str] = {
    "chronos2": "chronos-2",
    "chronos": "chronos-2",
    "moirai2": "moirai-2",
    "tirex": "tirex",
    "timesfm2": "timesfm-2",
    "timesfm-2": "timesfm-2",
    "moirai-small": "moirai-1-small",
    "moirai-base": "moirai-1-base",
    "moirai-large": "moirai-1-large",
    "moirai1-small": "moirai-1-small",
    "moirai1-base": "moirai-1-base",
    "moirai1-large": "moirai-1-large",
}

# Registered model names currently wired with a backend adapter.
BACKEND_COMPATIBLE_MODELS: set[str] = set(TSFM_MODEL_PATHS.keys())


def normalize_model_name(name: str | None) -> str | None:
    if name is None:
        return None
    key = str(name).strip().lower()
    if not key:
        return None
    key = TSFM_MODEL_ALIASES.get(key, key)
    if key not in TSFM_MODEL_PATHS:
        raise ValueError(
            f"Unsupported model short name {name!r}. Supported defaults: {TSFM_MODEL_ORDER}; all registered: {sorted(TSFM_MODEL_PATHS.keys())}"
        )
    return key


def resolve_model_path(model_short_name: str, prefix: str | Path | None = None) -> Path:
    key = normalize_model_name(model_short_name)
    if key is None:
        raise ValueError("model_short_name must not be empty")
    root = Path(prefix).expanduser() if prefix is not None else TSFM_MODEL_PATH_PREFIX
    return (root / TSFM_MODEL_PATHS[key]).resolve()


def parse_name_list(raw: str | None, *, default_values: list[str]) -> list[str]:
    if raw is None:
        return list(default_values)
    text = str(raw).strip()
    if not text:
        return list(default_values)
    if text.lower() == "all":
        return list(default_values)
    names = [x.strip() for x in text.split(",") if x.strip()]
    if not names:
        return list(default_values)
    return names
