"""mlx-whisper adapter: the default on Apple Silicon, and the primary target.

Two things differ from faster-whisper and both are handled here rather than leaking into
the pipeline:

* **No VAD.** mlx-whisper has no voice-activity filter, so the shared silence gate in
  `base.drop_silent_segments` does that job instead. Without it Whisper's silence filler
  ("Hvala.", "Thank you.") lands in the subtitles.
* **A smaller kwarg surface.** Options are passed through a filter that keeps only what the
  installed version actually accepts, so a version bump degrades to fewer options rather
  than a TypeError mid-transcription.
"""

from __future__ import annotations

import inspect
import platform
import time
from pathlib import Path
from typing import Any

from subtitler import models
from subtitler.engines.base import (
    Availability,
    EngineUnavailable,
    ModelInfo,
    TranscribeOptions,
    collapse_repetition,
    drop_silent_segments,
)
from subtitler.model import Segment, Transcript, Word

BACKEND = "mlx"


class MlxWhisperEngine:
    name = BACKEND
    kind = "local"

    def __init__(self, model: str = "large-v3", *, device: str = "auto") -> None:
        self.spec = models.resolve(model, BACKEND)
        self.requested_device = device

    # ---------------------------------------------------------------- availability

    @staticmethod
    def platform_supported() -> bool:
        return platform.system() == "Darwin" and platform.machine() in {"arm64", "aarch64"}

    def availability(self) -> Availability:
        # Check the platform before the import: on Linux `uv sync --all-extras` cannot even
        # resolve mlx-whisper, so a bare ImportError would be a confusing way to say
        # "this engine is Apple Silicon only".
        if not self.platform_supported():
            return Availability(
                False,
                "mlx runs on Apple Silicon only",
                "use --engine faster-whisper on this machine",
            )
        try:
            import mlx_whisper  # noqa: F401
        except ImportError:
            return Availability(False, "mlx-whisper is not installed", "uv sync --extra mlx")
        if models.local_path(self.spec) is None:
            return Availability(
                False,
                f"the {self.spec.name} weights are not downloaded ({self.spec.size_label})",
                f"subtitler models download {self.spec.name}",
            )
        return Availability(True)

    def ensure_model(self, progress: Any = None) -> ModelInfo:
        path = models.local_path(self.spec) or models.download(self.spec, progress=progress)
        return ModelInfo(
            name=self.spec.name,
            revision=self.spec.revision,
            path=Path(path),
            size_bytes=self.spec.approx_bytes,
        )

    def describe(self) -> dict[str, Any]:
        return {
            "engine": self.name,
            "kind": self.kind,
            "model": self.spec.name,
            "repo": self.spec.repo_id,
            "revision": self.spec.revision,
            "device": "mps",
        }

    # ---------------------------------------------------------------- transcription

    def transcribe(self, audio: Path, opts: TranscribeOptions) -> Transcript:
        avail = self.availability()
        if not avail.ok:
            raise EngineUnavailable(self.name, avail.reason, avail.fix)

        import mlx_whisper

        path = models.local_path(self.spec)
        wanted: dict[str, Any] = {
            "path_or_hf_repo": str(path),
            "language": None if opts.language == "auto" else opts.language,
            "initial_prompt": opts.initial_prompt,
            "temperature": opts.temperature,
            "word_timestamps": opts.word_timestamps,
            "condition_on_previous_text": opts.condition_on_previous_text,
            "compression_ratio_threshold": opts.compression_ratio_threshold,
        }
        kwargs = _supported_kwargs(mlx_whisper.transcribe, wanted)

        started = time.monotonic()
        raw = mlx_whisper.transcribe(str(audio), **kwargs)
        runtime = time.monotonic() - started

        segments = []
        # See the same counters in `faster.py`: the benchmark needs to know how often the
        # decoder looped and how much silence filler the gate removed, and neither survives
        # into the transcript that is kept.
        collapsed = 0
        for seg in raw.get("segments") or []:
            raw_text = str(seg.get("text", "")).strip()
            text = collapse_repetition(raw_text)
            collapsed += text != raw_text
            if not text:
                continue
            segments.append(
                Segment(
                    start=float(seg["start"]),
                    end=float(seg["end"]),
                    text=text,
                    words=tuple(
                        Word(
                            start=float(w["start"]),
                            end=float(w["end"]),
                            text=str(w.get("word", "")).strip(),
                            prob=_opt_float(w.get("probability")),
                        )
                        for w in (seg.get("words") or [])
                        if "start" in w and "end" in w
                    ),
                    no_speech_prob=_opt_float(seg.get("no_speech_prob")),
                    avg_logprob=_opt_float(seg.get("avg_logprob")),
                    compression_ratio=_opt_float(seg.get("compression_ratio")),
                )
            )

        kept = drop_silent_segments(tuple(segments), audio)

        return Transcript(
            language=raw.get("language") or opts.language,
            duration=float(kept[-1].end if kept else 0.0),
            segments=kept,
            engine=self.name,
            model=self.spec.name,
            model_revision=self.spec.revision,
            runtime_s=runtime,
            params={
                "language": opts.language,
                "temperature": opts.temperature,
                "seed": opts.seed,
                "passed_kwargs": sorted(kwargs),
                "repetition_collapsed": collapsed,
                "silence_dropped": len(segments) - len(kept),
            },
        )


def _supported_kwargs(func: Any, wanted: dict[str, Any]) -> dict[str, Any]:
    """Keep only the kwargs this build of mlx-whisper accepts, dropping None values.

    mlx-whisper is younger than faster-whisper and its signature has moved. Filtering
    means a version bump loses an option rather than raising a TypeError partway through
    a transcription, and `params.passed_kwargs` records what actually got through so a
    benchmark result is never silently mis-attributed.
    """
    try:
        accepted = set(inspect.signature(func).parameters)
    except (TypeError, ValueError):
        return {k: v for k, v in wanted.items() if v is not None}

    # A **kwargs catch-all means everything is forwarded to the decoder.
    if any(
        p.kind is inspect.Parameter.VAR_KEYWORD for p in inspect.signature(func).parameters.values()
    ):
        return {k: v for k, v in wanted.items() if v is not None}

    return {k: v for k, v in wanted.items() if k in accepted and v is not None}


def _opt_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
