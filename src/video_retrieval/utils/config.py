"""YAML configuration with .env and environment-variable overrides."""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Optional

import yaml

try:
    from dotenv import load_dotenv
except ImportError:  # python-dotenv is optional
    load_dotenv = None

ENV_PREFIX = "VR__"
# <root>/src/video_retrieval/utils/config.py -> parents[3] == <root>
PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _to_ns(value: Any) -> Any:
    if isinstance(value, dict):
        return SimpleNamespace(**{k: _to_ns(v) for k, v in value.items()})
    return value


def _apply_env_overrides(data: Dict[str, Any]) -> None:
    """VR__EMBEDDING__DEVICE=cuda  ->  data['embedding']['device'] = 'cuda'."""
    for key, raw in os.environ.items():
        if not key.startswith(ENV_PREFIX):
            continue
        parts = key[len(ENV_PREFIX):].lower().split("__")
        node = data
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = yaml.safe_load(raw)


def _abs(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else PROJECT_ROOT / p


def _resolve_sqlite(url: str) -> str:
    prefix = "sqlite:///"
    if url.startswith(prefix) and url != prefix + ":memory:":
        p = _abs(url[len(prefix):])
        p.parent.mkdir(parents=True, exist_ok=True)
        return prefix + p.as_posix()
    return url


def load_config(path: Optional[str] = None) -> SimpleNamespace:
    if load_dotenv:
        load_dotenv(PROJECT_ROOT / ".env")

    cfg_path = Path(path or os.getenv("VR_CONFIG") or "config/default.yaml")
    if not cfg_path.is_absolute() and not cfg_path.exists():
        cfg_path = PROJECT_ROOT / cfg_path
    with open(cfg_path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}

    _apply_env_overrides(data)
    if os.getenv("DATABASE_URL"):
        data["database"]["metadata_url"] = os.environ["DATABASE_URL"]

    data_dir = _abs(data["app"]["data_dir"])
    data["app"]["data_dir"] = str(data_dir)
    data["database"]["vector_dir"] = str(_abs(data["database"]["vector_dir"]))
    data["database"]["metadata_url"] = _resolve_sqlite(data["database"]["metadata_url"])

    data["paths"] = {
        "data": str(data_dir),
        "uploads": str(data_dir / "uploads"),
        "clips": str(data_dir / "clips"),
        "thumbnails": str(data_dir / "thumbnails"),
    }
    for p in data["paths"].values():
        Path(p).mkdir(parents=True, exist_ok=True)
    data["root"] = str(PROJECT_ROOT)
    return _to_ns(data)


@lru_cache(maxsize=1)
def get_config() -> SimpleNamespace:
    return load_config()
