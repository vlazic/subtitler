"""The intermediate data model every stage reads and writes.

Word-level timings are the contract. `cues.py` needs them to pick split points, and every
engine can supply them (faster-whisper and mlx via `word_timestamps=True`, Groq via
`timestamp_granularities`). When an engine does not, `synthesize_words` fills in.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class Word:
    start: float
    end: float
    text: str
    prob: float | None = None


@dataclass(frozen=True, slots=True)
class Segment:
    start: float
    end: float
    text: str
    words: tuple[Word, ...] = ()
    no_speech_prob: float | None = None
    avg_logprob: float | None = None
    compression_ratio: float | None = None


@dataclass(frozen=True, slots=True)
class Cue:
    """One displayed subtitle. `lines` is already wrapped: nothing downstream re-wraps."""

    index: int
    start: float
    end: float
    lines: tuple[str, ...]

    @property
    def text(self) -> str:
        return " ".join(self.lines)

    @property
    def duration(self) -> float:
        return self.end - self.start

    @property
    def cps(self) -> float:
        """Characters per second. The reading-speed metric that matters for burned-in text."""
        d = self.duration
        return len(self.text) / d if d > 0 else float("inf")


@dataclass(frozen=True, slots=True)
class Transcript:
    language: str
    duration: float
    segments: tuple[Segment, ...]
    engine: str
    model: str
    model_revision: str = ""
    runtime_s: float = 0.0
    params: dict[str, Any] = field(default_factory=dict)

    @property
    def text(self) -> str:
        return " ".join(s.text.strip() for s in self.segments if s.text.strip())

    @property
    def rtf(self) -> float:
        """Realtime factor. Below 1.0 means faster than the audio it processed."""
        return self.runtime_s / self.duration if self.duration > 0 else 0.0


def synthesize_words(segment: Segment) -> tuple[Word, ...]:
    """Distribute a segment's duration across its words by character length.

    A degraded fallback for engines that return no word timings, never a fatal error.
    Character length is a better proxy for spoken duration than word count is.
    """
    tokens = segment.text.split()
    if not tokens:
        return ()
    total = sum(len(t) for t in tokens)
    if total == 0:
        return ()
    span = max(segment.end - segment.start, 1e-6)
    words: list[Word] = []
    cursor = segment.start
    for token in tokens:
        share = span * (len(token) / total)
        words.append(Word(start=cursor, end=cursor + share, text=token))
        cursor += share
    # Absorb float drift into the last word so the segment boundary stays exact.
    last = words[-1]
    words[-1] = Word(start=last.start, end=segment.end, text=last.text, prob=last.prob)
    return tuple(words)


def ensure_words(segments: tuple[Segment, ...]) -> tuple[Segment, ...]:
    """Guarantee every segment carries word timings."""
    out = []
    for seg in segments:
        out.append(seg if seg.words else Segment(**{**asdict(seg), "words": synthesize_words(seg)}))
    return tuple(out)


# --------------------------------------------------------------------------------------
# Serialization. Every stage artifact is JSON so the cache is inspectable by hand.
# --------------------------------------------------------------------------------------


def transcript_to_dict(t: Transcript) -> dict[str, Any]:
    return {"schema_version": SCHEMA_VERSION, **asdict(t)}


def transcript_from_dict(data: dict[str, Any]) -> Transcript:
    version = data.get("schema_version", SCHEMA_VERSION)
    if version != SCHEMA_VERSION:
        raise ValueError(f"transcript schema v{version} is not v{SCHEMA_VERSION}; delete the cache")
    payload = {k: v for k, v in data.items() if k != "schema_version"}
    segments = tuple(
        Segment(
            **{
                **s,
                "words": tuple(Word(**w) for w in s.get("words", ())),
            }
        )
        for s in payload.pop("segments", [])
    )
    return Transcript(segments=segments, **payload)


def cues_to_dict(cues: tuple[Cue, ...]) -> dict[str, Any]:
    return {"schema_version": SCHEMA_VERSION, "cues": [asdict(c) for c in cues]}


def cues_from_dict(data: dict[str, Any]) -> tuple[Cue, ...]:
    return tuple(
        Cue(index=c["index"], start=c["start"], end=c["end"], lines=tuple(c["lines"]))
        for c in data["cues"]
    )


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))
