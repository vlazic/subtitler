"""Groq Whisper adapter.

Cloud, so it is a comparison baseline and a fallback rather than the default: the friend's
video should not have to leave his laptop. It is the first engine implemented because it
needs no 3 GB download, which makes the walking skeleton testable on day one.
"""

from __future__ import annotations

import json
import os
import random
import time
from pathlib import Path
from typing import Any

from subtitler.engines.base import (
    Availability,
    EngineUnavailable,
    ModelInfo,
    TranscribeOptions,
    collapse_repetition,
)
from subtitler.model import Segment, Transcript, Word

MODELS = {
    "large-v3": "whisper-large-v3",
    "whisper-large-v3": "whisper-large-v3",
    "turbo": "whisper-large-v3-turbo",
    "large-v3-turbo": "whisper-large-v3-turbo",
    "whisper-large-v3-turbo": "whisper-large-v3-turbo",
}

# Groq rejects oversized uploads (25 MB on the free tier, 100 MB on dev). A 16 kHz mono
# PCM WAV is ~1.9 MB per minute, so this is roughly 13 minutes. Splitting long inputs at
# silence boundaries lands in a later phase; until then the limit is reported honestly
# instead of surfacing as an opaque API error.
MAX_UPLOAD_BYTES = 24 * 1024 * 1024

_RETRY_STATUS = {408, 429, 500, 502, 503, 504}


class GroqEngine:
    kind = "cloud"

    def __init__(self, model: str = "large-v3", *, max_retries: int = 4) -> None:
        if model not in MODELS:
            raise EngineUnavailable(
                "groq", f"unknown model {model!r}", f"choose from {sorted(set(MODELS))}"
            )
        self.model = MODELS[model]
        self.name = "groq-turbo" if self.model.endswith("-turbo") else "groq"
        self.max_retries = max_retries

    # ---------------------------------------------------------------- availability

    @staticmethod
    def api_keys() -> list[str]:
        """Read the key pool.

        GROQ_API_KEYS is a comma-separated pool, carried over from the bash pipeline so a
        long batch does not exhaust one key's rate limit. Unlike that version, nothing is
        ever logged: it echoed the first 15 characters of the chosen key to stderr.
        """
        pool = os.environ.get("GROQ_API_KEYS", "")
        keys = [k.strip() for k in pool.split(",") if k.strip()]
        single = os.environ.get("GROQ_API_KEY", "").strip()
        if single and single not in keys:
            keys.append(single)
        return keys

    def availability(self) -> Availability:
        try:
            import groq  # noqa: F401
        except ImportError:
            return Availability(False, "the groq package is not installed", "uv sync --extra cloud")
        if not self.api_keys():
            return Availability(
                False,
                "no API key found",
                "set GROQ_API_KEY in your environment or in .env",
            )
        return Availability(True)

    def ensure_model(self, progress: Any = None) -> ModelInfo:
        """Nothing to download: the weights live on Groq's side."""
        return ModelInfo(name=self.model, revision="hosted")

    def describe(self) -> dict[str, Any]:
        return {"engine": self.name, "kind": self.kind, "model": self.model}

    # ---------------------------------------------------------------- transcription

    def transcribe(self, audio: Path, opts: TranscribeOptions) -> Transcript:
        avail = self.availability()
        if not avail.ok:
            raise EngineUnavailable(self.name, avail.reason, avail.fix)

        size = audio.stat().st_size
        if size > MAX_UPLOAD_BYTES:
            raise EngineUnavailable(
                self.name,
                f"{size / 1e6:.1f} MB exceeds the {MAX_UPLOAD_BYTES / 1e6:.0f} MB upload limit",
                "use a local engine (--engine faster-whisper or mlx) for long files",
            )

        payload = audio.read_bytes()
        started = time.monotonic()
        raw = self._request(audio.name, payload, opts)
        runtime = time.monotonic() - started

        return self._to_transcript(raw, runtime=runtime, opts=opts)

    def _request(self, filename: str, payload: bytes, opts: TranscribeOptions) -> dict[str, Any]:
        import groq

        keys = self.api_keys()
        last_error: Exception | None = None

        for attempt in range(self.max_retries):
            key = random.choice(keys)
            client = groq.Groq(api_key=key)
            kwargs: dict[str, Any] = {
                "file": (filename, payload),
                "model": self.model,
                "response_format": "verbose_json",
                "temperature": opts.temperature,
                "timestamp_granularities": ["segment", "word"],
            }
            # 'auto' means omit the field entirely and let the model decide. Every other
            # value is pinned, because Serbian gets detected as hr or bs on short clips.
            if opts.language and opts.language != "auto":
                kwargs["language"] = opts.language
            if opts.initial_prompt:
                kwargs["prompt"] = opts.initial_prompt

            try:
                response = client.audio.transcriptions.create(**kwargs)
            except Exception as exc:
                if self._retryable(exc) and attempt < self.max_retries - 1:
                    last_error = exc
                    time.sleep(min(2**attempt, 8))
                    continue
                # Surface API failures as an engine error with a fix, not as a traceback
                # out of the SDK. Account-level problems in particular ("organization
                # has been restricted", quota exhausted) are not the user's mistake and
                # should point at the local engine rather than at a stack trace.
                raise EngineUnavailable(
                    self.name,
                    _api_message(exc),
                    "use a local engine: --engine faster-whisper (or mlx on Apple Silicon)",
                ) from exc

            return json.loads(response.json()) if hasattr(response, "json") else dict(response)

        raise EngineUnavailable(self.name, f"request failed after retries: {last_error}", "")

    @staticmethod
    def _retryable(exc: Exception) -> bool:
        status = getattr(exc, "status_code", None) or getattr(
            getattr(exc, "response", None), "status_code", None
        )
        return status in _RETRY_STATUS

    def _to_transcript(
        self, raw: dict[str, Any], *, runtime: float, opts: TranscribeOptions
    ) -> Transcript:
        return parse_verbose_json(
            raw, opts=opts, engine=self.name, model=self.model, runtime=runtime
        )


def parse_verbose_json(
    raw: dict[str, Any],
    *,
    opts: TranscribeOptions,
    engine: str = "verbose_json",
    model: str = "unknown",
    runtime: float = 0.0,
) -> Transcript:
    """Parse a Whisper `verbose_json` response into a Transcript.

    Module-level and provider-neutral: OpenAI, Groq and local Whisper all emit this shape,
    and `subtitler convert` reads saved responses with it.
    """
    words_by_index = _group_words(raw.get("words") or [], raw.get("segments") or [])

    segments = []
    for i, seg in enumerate(raw.get("segments") or []):
        text = collapse_repetition((seg.get("text") or "").strip())
        if not text:
            continue
        segments.append(
            Segment(
                start=float(seg["start"]),
                end=float(seg["end"]),
                text=text,
                words=words_by_index.get(i, ()),
                no_speech_prob=_opt_float(seg.get("no_speech_prob")),
                avg_logprob=_opt_float(seg.get("avg_logprob")),
                compression_ratio=_opt_float(seg.get("compression_ratio")),
            )
        )

    return Transcript(
        language=raw.get("language") or opts.language,
        duration=float(raw.get("duration") or (segments[-1].end if segments else 0.0)),
        segments=tuple(segments),
        engine=engine,
        model=model,
        model_revision="hosted",
        runtime_s=runtime,
        params={
            "language": opts.language,
            "temperature": opts.temperature,
            "prompt": bool(opts.initial_prompt),
        },
    )


def _group_words(
    words: list[dict[str, Any]], segments: list[dict[str, Any]]
) -> dict[int, tuple[Word, ...]]:
    """Assign a flat word list to segments by timestamp.

    Groq returns words as one flat array rather than nested per segment, so they have to be
    bucketed. A word is assigned to the segment whose span contains its midpoint, which is
    stable even when the two disagree slightly at the boundaries.
    """
    if not words or not segments:
        return {}

    bounds = [(float(s["start"]), float(s["end"])) for s in segments]
    grouped: dict[int, list[Word]] = {}
    cursor = 0

    for w in words:
        try:
            start, end = float(w["start"]), float(w["end"])
        except (KeyError, TypeError, ValueError):
            continue
        mid = (start + end) / 2
        while cursor + 1 < len(bounds) and mid > bounds[cursor][1]:
            cursor += 1
        grouped.setdefault(cursor, []).append(
            Word(start=start, end=end, text=str(w.get("word", "")).strip())
        )

    return {i: tuple(ws) for i, ws in grouped.items() if ws}


def _api_message(exc: Exception) -> str:
    """A one-line reason from a Groq SDK error, without the stack."""
    body = getattr(exc, "body", None)
    if isinstance(body, dict):
        error = body.get("error")
        if isinstance(error, dict) and error.get("message"):
            return str(error["message"])
    status = getattr(exc, "status_code", None)
    return f"API error{f' {status}' if status else ''}: {exc}".strip()


def _opt_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
