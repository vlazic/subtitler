"""Engine registry and platform-aware default selection.

Two rules this module exists to hold:

  * `--engine auto` silently skips anything unavailable and reports what it settled on.
  * An **explicit** engine that cannot run is a hard error naming the fix, never a silent
    fallback. Falling back quietly is how you end up benchmarking the wrong backend.

Phase 1 registers the cloud engine only. The local engines (mlx on Apple Silicon,
faster-whisper elsewhere) join the registry in Phase 3 and take over the top of the
preference order at that point.
"""

from __future__ import annotations

import platform

from subtitler.engines.base import Availability, Engine, EngineUnavailable
from subtitler.engines.groq import GroqEngine

__all__ = ["EngineUnavailable", "available_engines", "is_apple_silicon", "resolve"]

# Preference order for `auto`, best first. Local engines are prepended in Phase 3 so the
# default path never uploads anything.
_AUTO_ORDER = ["groq", "groq-turbo"]


def is_apple_silicon() -> bool:
    return platform.system() == "Darwin" and platform.machine() in {"arm64", "aarch64"}


def _build(name: str, *, model: str, device: str) -> Engine:
    if name in {"groq", "groq-turbo"}:
        return GroqEngine("turbo" if name == "groq-turbo" else model)
    raise EngineUnavailable(
        name,
        "not implemented yet (local engines land in Phase 3)",
        "use --engine groq for now",
    )


def available_engines(*, model: str = "large-v3", device: str = "auto") -> dict[str, Availability]:
    """Probe every known engine. Used by `subtitler doctor` and by `auto` resolution."""
    result: dict[str, Availability] = {}
    for name in _AUTO_ORDER:
        try:
            result[name] = _build(name, model=model, device=device).availability()
        except EngineUnavailable as exc:
            result[name] = Availability(False, exc.reason, exc.fix)
    return result


def resolve(name: str = "auto", *, model: str = "large-v3", device: str = "auto") -> Engine:
    if name != "auto":
        engine = _build(name, model=model, device=device)
        avail = engine.availability()
        if not avail.ok:
            raise EngineUnavailable(name, avail.reason, avail.fix)
        return engine

    problems: list[str] = []
    for candidate in _AUTO_ORDER:
        try:
            engine = _build(candidate, model=model, device=device)
        except EngineUnavailable as exc:
            problems.append(f"  {candidate}: {exc.reason}")
            continue
        avail = engine.availability()
        if avail.ok:
            return engine
        problems.append(f"  {candidate}: {avail.reason}" + (f" ({avail.fix})" if avail.fix else ""))

    raise EngineUnavailable(
        "auto",
        "no engine is usable on this machine:\n" + "\n".join(problems),
        "run `subtitler doctor` for the full dependency report",
    )
