"""faster-whisper (CTranslate2) adapter.

The default local engine everywhere except Apple Silicon. Most of the non-obvious code
here is lifted from `record-audio/record_audio/transcribe.py`, where it was learned the
expensive way.
"""

from __future__ import annotations

import ctypes
import os
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

BACKEND = "faster-whisper"

# None = not attempted yet. Cache the RESULT, not merely the fact that it ran: caching a
# bare "already done" flag made the second call return True even when nothing loaded,
# which told the device probe CUDA was fine and produced a failure several seconds later.
_cuda_preloaded: bool | None = None


def preload_cuda_libraries() -> bool:
    """Load the venv's CUDA 12 libraries before CTranslate2 looks for them.

    CTranslate2 wheels link against CUDA 12 while a current driver ships 13, so the loader
    finds the system's 13 first and CTranslate2 fails with an unresolved symbol. Opening
    the pip-installed cu12 libraries with RTLD_GLOBAL first makes them win.

    Returns True if anything was loaded. Never raises: on a CPU-only machine there is
    simply nothing to preload.
    """
    global _cuda_preloaded
    if _cuda_preloaded is not None:
        return _cuda_preloaded

    loaded = False
    for package, libs in (
        ("nvidia.cublas.lib", ("libcublas.so.12", "libcublasLt.so.12")),
        ("nvidia.cudnn.lib", ("libcudnn.so.9", "libcudnn_ops.so.9", "libcudnn_cnn.so.9")),
    ):
        try:
            module = __import__(package, fromlist=["__file__"])
            lib_dir = Path(module.__file__ or "").parent
        except (ImportError, AttributeError):
            continue
        for name in libs:
            candidate = lib_dir / name
            if not candidate.exists():
                continue
            try:
                ctypes.CDLL(str(candidate), mode=ctypes.RTLD_GLOBAL)
                loaded = True
            except OSError:
                # A library that will not load is not fatal here: the device probe below
                # falls back to CPU, which is slow but correct.
                continue

    _cuda_preloaded = loaded
    return loaded


def _cuda_usable() -> bool:
    """A visible device is not enough: the CUDA 12 libraries have to actually load.

    `get_cuda_device_count()` reports the driver's devices, which on this dev box is 1
    even though the driver ships CUDA 13 and CTranslate2 wants 12. Trusting that count
    alone produces "Library libcublas.so.12 is not found or cannot be loaded" at model
    load time, several seconds later and with no hint about the cause. So: check that
    libcublas.so.12 is genuinely loadable before claiming CUDA.
    """
    if os.environ.get("SUBTITLER_FORCE_CPU"):
        return False
    try:
        import ctranslate2

        if ctranslate2.get_cuda_device_count() <= 0:
            return False
    except Exception:
        return False

    if preload_cuda_libraries():
        return True
    # Nothing was preloadable from the venv; fall back to whatever the system loader has.
    try:
        ctypes.CDLL("libcublas.so.12", mode=ctypes.RTLD_GLOBAL)
    except OSError:
        return False
    return True


class FasterWhisperEngine:
    name = BACKEND
    kind = "local"

    def __init__(self, model: str = "large-v3", *, device: str = "auto") -> None:
        self.spec = models.resolve(model, BACKEND)
        self.requested_device = device
        self._model: Any = None
        # Set once the model actually loads, which is the only reliable answer.
        self._resolved: tuple[str, str] | None = None
        self._fallback_reason: str | None = None

    # ---------------------------------------------------------------- availability

    def availability(self) -> Availability:
        try:
            import faster_whisper  # noqa: F401
        except ImportError:
            return Availability(False, "faster-whisper is not installed", "uv sync --extra local")
        if models.local_path(self.spec) is None:
            return Availability(
                False,
                f"the {self.spec.name} weights are not downloaded ({self.spec.size_label})",
                f"subtitler models download {self.spec.name}",
            )
        return Availability(True)

    def resolve_device(self) -> tuple[str, str]:
        """(device, compute_type). int8 on CPU is the difference between usable and not."""
        if self._resolved is not None:
            return self._resolved
        if self.requested_device == "cpu":
            return "cpu", "int8"
        if self.requested_device == "cuda":
            return "cuda", "float16"
        return ("cuda", "float16") if _cuda_usable() else ("cpu", "int8")

    def ensure_model(self, progress: Any = None) -> ModelInfo:
        path = models.local_path(self.spec) or models.download(self.spec, progress=progress)
        return ModelInfo(
            name=self.spec.name,
            revision=self.spec.revision,
            path=Path(path),
            size_bytes=self.spec.approx_bytes,
        )

    def describe(self) -> dict[str, Any]:
        device, compute = self.resolve_device()
        return {
            "engine": self.name,
            "kind": self.kind,
            "model": self.spec.name,
            "repo": self.spec.repo_id,
            "revision": self.spec.revision,
            "device": device,
            "compute_type": compute,
            "cuda_fallback": self._fallback_reason,
        }

    # ---------------------------------------------------------------- transcription

    def _load(self) -> Any:
        if self._model is not None:
            return self._model
        from faster_whisper import WhisperModel

        path = models.local_path(self.spec)
        if path is None:
            raise EngineUnavailable(
                self.name,
                "weights are not downloaded",
                f"subtitler models download {self.spec.name}",
            )

        device, compute_type = self.resolve_device()
        self._model = WhisperModel(str(path), device=device, compute_type=compute_type)
        self._resolved = (device, compute_type)
        return self._model

    def _force_cpu(self, reason: str) -> None:
        """Drop to CPU after a CUDA failure, recording why.

        Necessary because CTranslate2 initializes the device lazily: constructing
        WhisperModel with device="cuda" succeeds on a machine with no usable CUDA
        runtime, and the failure only appears once decoding starts. No probe can catch
        that ahead of time, so the retry has to wrap the actual work.
        """
        self._fallback_reason = reason.strip().splitlines()[0] if reason else "cuda unavailable"
        self._model = None
        self._resolved = ("cpu", "int8")

    def transcribe(self, audio: Path, opts: TranscribeOptions) -> Transcript:
        avail = self.availability()
        if not avail.ok:
            raise EngineUnavailable(self.name, avail.reason, avail.fix)

        try:
            return self._decode(audio, opts)
        except Exception as exc:
            # A CUDA runtime that is visible but unusable fails here, not at load time.
            # An explicit --device cuda should report that; auto should degrade to CPU,
            # which is slower but correct, and say so in the params.
            if self._resolved != ("cuda", "float16") or self.requested_device == "cuda":
                raise
            self._force_cpu(str(exc))
            return self._decode(audio, opts)

    def _decode(self, audio: Path, opts: TranscribeOptions) -> Transcript:
        import ctranslate2

        # A fixed seed does not make CTranslate2 bit-deterministic, but it removes one
        # source of run-to-run drift when comparing engines in the benchmark.
        ctranslate2.set_random_seed(opts.seed)

        model = self._load()
        started = time.monotonic()
        segments_iter, info = model.transcribe(
            str(audio),
            language=None if opts.language == "auto" else opts.language,
            initial_prompt=opts.initial_prompt,
            beam_size=opts.beam_size,
            temperature=opts.temperature,
            word_timestamps=opts.word_timestamps,
            # Off by default: this is the main driver of repetition loops, and it lets one
            # bad segment poison everything after it.
            condition_on_previous_text=opts.condition_on_previous_text,
            compression_ratio_threshold=opts.compression_ratio_threshold,
            vad_filter=opts.vad,
        )

        segments = []
        for seg in segments_iter:  # generator: work happens here
            text = collapse_repetition((seg.text or "").strip())
            if not text:
                continue
            segments.append(
                Segment(
                    start=float(seg.start),
                    end=float(seg.end),
                    text=text,
                    words=tuple(
                        Word(
                            start=float(w.start),
                            end=float(w.end),
                            text=str(w.word).strip(),
                            prob=float(w.probability) if w.probability is not None else None,
                        )
                        for w in (seg.words or [])
                    ),
                    no_speech_prob=_opt_float(getattr(seg, "no_speech_prob", None)),
                    avg_logprob=_opt_float(getattr(seg, "avg_logprob", None)),
                    compression_ratio=_opt_float(getattr(seg, "compression_ratio", None)),
                )
            )
        runtime = time.monotonic() - started

        kept = drop_silent_segments(tuple(segments), audio)
        device, compute_type = self.resolve_device()

        return Transcript(
            language=getattr(info, "language", None) or opts.language,
            duration=float(getattr(info, "duration", 0.0) or (kept[-1].end if kept else 0.0)),
            segments=kept,
            engine=self.name,
            model=self.spec.name,
            model_revision=self.spec.revision,
            runtime_s=runtime,
            params={
                "language": opts.language,
                "beam_size": opts.beam_size,
                "temperature": opts.temperature,
                "vad": opts.vad,
                "device": device,
                "compute_type": compute_type,
                "cuda_fallback": self._fallback_reason,
                "seed": opts.seed,
            },
        )


def _opt_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
