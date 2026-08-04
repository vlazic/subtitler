"""Engine registry and platform-aware default selection.

Two rules this module exists to hold:

  * `--engine auto` silently skips anything unavailable and reports what it settled on.
  * An **explicit** engine that cannot run is a hard error naming the fix, never a silent
    fallback. Falling back quietly is how you end up benchmarking the wrong backend, or
    uploading a private video to a cloud API because a local model failed to load.
"""

from __future__ import annotations

from collections.abc import Callable

from subtitler.doctor import detect_platform
from subtitler.engines.base import Availability, Engine, EngineUnavailable
from subtitler.engines.faster import FasterWhisperEngine
from subtitler.engines.groq import GroqEngine
from subtitler.engines.mlx import MlxWhisperEngine

__all__ = [
    "ALL_ENGINES",
    "EngineUnavailable",
    "available_engines",
    "default_order",
    "is_apple_silicon",
    "resolve",
]

ALL_ENGINES = ("mlx", "faster-whisper", "groq", "groq-turbo")

# `batch_size` is a faster-whisper-on-CUDA knob; every other backend takes it and drops it
# rather than the caller having to know which engine it applies to.
_BUILDERS: dict[str, Callable[[str, str, int], Engine]] = {
    "mlx": lambda model, device, _batch: MlxWhisperEngine(model, device=device),
    "faster-whisper": lambda model, device, batch: FasterWhisperEngine(
        model, device=device, batch_size=batch
    ),
    "groq": lambda model, _device, _batch: GroqEngine(model),
    "groq-turbo": lambda _model, _device, _batch: GroqEngine("turbo"),
}


def is_apple_silicon() -> bool:
    """Platform facts come from doctor.detect_platform, never from `platform` directly.

    One detector means the mac branch stays fakeable in tests, which is the only way the
    primary target gets exercised on a Linux dev machine.
    """
    return detect_platform().is_apple_silicon


def default_order(apple_silicon: bool | None = None) -> tuple[str, ...]:
    """Preference order for `auto`, best first.

    Local before cloud, always: the friend's video should not leave his laptop just
    because a download has not happened yet. Cloud is a fallback, not a shortcut.
    """
    mac = is_apple_silicon() if apple_silicon is None else apple_silicon
    local = ("mlx", "faster-whisper") if mac else ("faster-whisper", "mlx")
    return (*local, "groq", "groq-turbo")


def _build(name: str, *, model: str, device: str, batch_size: int = 0) -> Engine:
    builder = _BUILDERS.get(name)
    if builder is None:
        raise EngineUnavailable(name, "unknown engine", f"choose from: {', '.join(ALL_ENGINES)}")
    return builder(model, device, batch_size)


def available_engines(*, model: str = "large-v3", device: str = "auto") -> dict[str, Availability]:
    """Probe every known engine. Used by `auto` resolution and by reporting."""
    result: dict[str, Availability] = {}
    for name in ALL_ENGINES:
        try:
            result[name] = _build(name, model=model, device=device).availability()
        except EngineUnavailable as exc:
            result[name] = Availability(False, exc.reason, exc.fix)
        except LookupError as exc:  # the backend has no such model
            result[name] = Availability(False, str(exc), "")
    return result


def resolve(
    name: str = "auto",
    *,
    model: str = "large-v3",
    device: str = "auto",
    batch_size: int = 0,
) -> Engine:
    if name != "auto":
        engine = _build(name, model=model, device=device, batch_size=batch_size)
        avail = engine.availability()
        if not avail.ok:
            raise EngineUnavailable(name, avail.reason, avail.fix)
        return engine

    problems: list[str] = []
    for candidate in default_order():
        try:
            engine = _build(candidate, model=model, device=device, batch_size=batch_size)
            avail = engine.availability()
        except (EngineUnavailable, LookupError) as exc:
            problems.append(f"  {candidate}: {exc}")
            continue
        if avail.ok:
            return engine
        detail = f"  {candidate}: {avail.reason}"
        if avail.fix:
            detail += f"\n      fix: {avail.fix}"
        problems.append(detail)

    raise EngineUnavailable(
        "auto",
        "no engine is usable on this machine:\n" + "\n".join(problems),
        "run `subtitler doctor` for the full dependency report",
    )
