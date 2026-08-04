"""Configuration and environment loading.

A deliberately small dotenv reader rather than a dependency: the file format we need is
`KEY=value` with optional quotes and `#` comments, and adding a package for that would be
the only reason it is in the dependency tree.
"""

from __future__ import annotations

import os
from pathlib import Path


def load_dotenv(path: Path | None = None, *, override: bool = False) -> dict[str, str]:
    """Load `.env` from `path`, or from the nearest ancestor of the CWD that has one.

    Real environment variables win by default, so `GROQ_API_KEY=... subtitler run` behaves
    the way anyone would expect.
    """
    env_path = path or _find_dotenv(Path.cwd())
    if env_path is None or not env_path.is_file():
        return {}

    loaded: dict[str, str] = {}
    for raw in env_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if not key:
            continue
        loaded[key] = value
        if override or key not in os.environ:
            os.environ[key] = value
    return loaded


def _find_dotenv(start: Path) -> Path | None:
    for directory in [start, *start.parents]:
        candidate = directory / ".env"
        if candidate.is_file():
            return candidate
    return None
