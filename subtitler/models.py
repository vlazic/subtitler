"""Model registry, download and cache management.

Two rules:

**Revisions are pinned to commit SHAs, not to `main`.** A floating tag means a benchmark
run from last month cannot be reproduced today, and the whole point of `bench` is to make
a decision that stays made.

**Downloading is an explicit command.** A first `subtitler run` on a fresh machine would
otherwise stall for ten minutes with no explanation while it pulls three gigabytes. `run`
errors with the exact command and the size unless `--auto-download` is passed.
"""

from __future__ import annotations

import os
import shutil
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

__all__ = [
    "MODELS",
    "ModelSpec",
    "cache_root",
    "download",
    "local_path",
    "remove",
    "resolve",
    "specs_for_backend",
]


@dataclass(frozen=True, slots=True)
class ModelSpec:
    name: str  # what the user types: "large-v3"
    backend: str  # "faster-whisper" | "mlx"
    repo_id: str
    revision: str  # a commit SHA, always
    approx_bytes: int
    license: str

    @property
    def key(self) -> str:
        return f"{self.backend}/{self.name}"

    @property
    def size_label(self) -> str:
        gb = self.approx_bytes / 1e9
        return f"{gb:.2f} GB" if gb >= 0.1 else f"{self.approx_bytes / 1e6:.0f} MB"


# Sizes and SHAs resolved from the Hugging Face API on 2026-08-04. To add or bump a model,
# re-resolve rather than guessing: `HfApi().model_info(repo, files_metadata=True)`.
MODELS: tuple[ModelSpec, ...] = (
    ModelSpec(
        name="large-v3",
        backend="faster-whisper",
        repo_id="Systran/faster-whisper-large-v3",
        revision="edaa852ec7e145841d8ffdb056a99866b5f0a478",
        approx_bytes=3_090_839_273,
        license="mit",
    ),
    ModelSpec(
        name="large-v3",
        backend="mlx",
        repo_id="mlx-community/whisper-large-v3-mlx",
        revision="49e6aa286ad60c14352c404340ded53710378a11",
        approx_bytes=3_083_522_487,
        license="mit",
    ),
    # Tiny exists for CI and for smoke-testing a fresh machine without a 3 GB download.
    # It is not good enough for Serbian and is never a default.
    ModelSpec(
        name="tiny",
        backend="faster-whisper",
        repo_id="Systran/faster-whisper-tiny",
        revision="d90ca5fe260221311c53c58e660288d3deb8d356",
        approx_bytes=78_207_087,
        license="mit",
    ),
    ModelSpec(
        name="tiny",
        backend="mlx",
        repo_id="mlx-community/whisper-tiny-mlx",
        revision="6caf9c55601caafbe6508a8b0d216bdf4783c4e8",
        approx_bytes=74_420_620,
        license="mit",
    ),
)

# What a user may type for each canonical name.
_ALIASES = {
    "large": "large-v3",
    "large-v3": "large-v3",
    "whisper-large-v3": "large-v3",
    "tiny": "tiny",
    "whisper-tiny": "tiny",
}


class ModelNotFound(LookupError):
    pass


def resolve(name: str, backend: str) -> ModelSpec:
    canonical = _ALIASES.get(name, name)
    for spec in MODELS:
        if spec.name == canonical and spec.backend == backend:
            return spec
    known = sorted({s.name for s in MODELS if s.backend == backend})
    raise ModelNotFound(f"no {backend} model named {name!r}; known: {', '.join(known)}")


def specs_for_backend(backend: str) -> tuple[ModelSpec, ...]:
    return tuple(s for s in MODELS if s.backend == backend)


def cache_root() -> Path:
    """Where Hugging Face puts things. Printed on every download so it is never a mystery."""
    for var in ("HF_HUB_CACHE", "HF_HOME"):
        value = os.environ.get(var)
        if value:
            return Path(value) / "hub" if var == "HF_HOME" else Path(value)
    return Path.home() / ".cache" / "huggingface" / "hub"


def _repo_dir(spec: ModelSpec) -> Path:
    return cache_root() / f"models--{spec.repo_id.replace('/', '--')}"


def local_path(spec: ModelSpec) -> Path | None:
    """The snapshot directory for the pinned revision, or None if it is not there."""
    snapshot = _repo_dir(spec) / "snapshots" / spec.revision
    if snapshot.is_dir() and any(snapshot.iterdir()):
        return snapshot
    return None


def is_cached(spec: ModelSpec) -> bool:
    return local_path(spec) is not None


def disk_usage(spec: ModelSpec) -> int:
    root = _repo_dir(spec)
    if not root.exists():
        return 0
    # follow_symlinks=False: the snapshot dir is symlinks into blobs/, and counting both
    # would double the number.
    return sum(f.stat().st_size for f in root.rglob("*") if f.is_file() and not f.is_symlink())


def download(spec: ModelSpec, *, progress: Callable[[str], None] | None = None) -> Path:
    """Fetch the pinned revision. Resumable: huggingface_hub keeps partial blobs."""
    from huggingface_hub import snapshot_download

    say = progress or (lambda _msg: None)
    say(f"downloading {spec.repo_id} @ {spec.revision[:12]} ({spec.size_label})")
    say(f"cache: {cache_root()}")

    path = snapshot_download(
        repo_id=spec.repo_id,
        revision=spec.revision,
        cache_dir=str(cache_root()),
    )
    say(f"ready: {path}")
    return Path(path)


def remove(spec: ModelSpec) -> bool:
    root = _repo_dir(spec)
    if not root.exists():
        return False
    shutil.rmtree(root)
    return True


def render_list(backend: str | None = None) -> str:
    """The `subtitler models list` table."""
    specs = [s for s in MODELS if backend is None or s.backend == backend]
    lines = [f"cache: {cache_root()}", ""]
    width = max(len(s.key) for s in specs)
    for spec in specs:
        if is_cached(spec):
            state = f"cached ({disk_usage(spec) / 1e9:.2f} GB on disk)"
        else:
            state = f"not downloaded ({spec.size_label})"
        lines.append(f"  {spec.key:<{width}}  {state}")
        lines.append(f"  {'':<{width}}  {spec.repo_id} @ {spec.revision[:12]} [{spec.license}]")
    return "\n".join(lines)
