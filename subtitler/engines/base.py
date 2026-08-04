"""The one interface every transcription backend implements.

Also the home of the shared decode hygiene. The three constants below were learned the hard
way in `record-audio` and are absent from every earlier pipeline in this lineage; they are
applied by every adapter rather than reimplemented per engine.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from subtitler.model import Transcript

# Verbatim from gozba2/emisije/transcribe-audio.sh. Proven on this material; do not
# reword it without a benchmark run showing the change helps.
#
# Note the tension worth knowing about: "do not split sentences across lines" is right for
# a sidecar transcript and is exactly what cues.py then undoes for burn-in. The prompt
# steers recognition; cue layout is decided afterwards, from word timings.
SERBIAN_PROMPT = (
    "Nemoj deliti rečenice u više redova. "
    "Zadrži srpski jezik i latinično pismo. "
    "Koristi ispravna imena za ljude, knjige, filozofske škole itd."
)

# Whisper hallucinates filler ("Hvala.", "Thank you.", "Titlovi ...") over silence. Dropping
# silent spans outright beats filtering the hallucination afterwards, because by then it has
# already consumed a timestamp range and skewed the segment boundaries around it.
SILENT_PEAK_DBFS = -60.0

# Decoder repetition loops: the same short phrase emitted over and over.
MAX_REPEATS = 6
MAX_REPEAT_PHRASE_TOKENS = 8

RANDOM_SEED = 20260101


class EngineUnavailable(RuntimeError):
    """An engine was asked for but cannot run here. Always carries an actionable fix."""

    def __init__(self, engine: str, reason: str, fix: str = "") -> None:
        message = f"engine {engine!r} is unavailable: {reason}"
        if fix:
            message += f"\n  fix: {fix}"
        super().__init__(message)
        self.engine = engine
        self.reason = reason
        self.fix = fix


@dataclass(frozen=True, slots=True)
class Availability:
    ok: bool
    reason: str = ""
    fix: str = ""


@dataclass(frozen=True, slots=True)
class ModelInfo:
    name: str
    revision: str = ""
    path: Path | None = None
    size_bytes: int | None = None


@dataclass(frozen=True, slots=True)
class TranscribeOptions:
    language: str = "sr"
    initial_prompt: str | None = SERBIAN_PROMPT
    temperature: float = 0.0
    beam_size: int = 5
    word_timestamps: bool = True
    # Off by default: conditioning on previous text is the main driver of repetition loops
    # and of one bad segment poisoning everything after it.
    condition_on_previous_text: bool = False
    compression_ratio_threshold: float | None = 2.4
    vad: bool = True
    seed: int = RANDOM_SEED
    extra: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class Engine(Protocol):
    name: str
    kind: str  # "local" | "cloud"

    def availability(self) -> Availability: ...

    def ensure_model(self, progress: Any = None) -> ModelInfo: ...

    def transcribe(self, audio: Path, opts: TranscribeOptions) -> Transcript: ...

    def describe(self) -> dict[str, Any]: ...


# --------------------------------------------------------------------------------------
# Shared post-decode hygiene
# --------------------------------------------------------------------------------------


def peak_dbfs(wav_path: Path, start: float, end: float) -> float:
    """Peak level of one time span in a 16-bit PCM WAV, in dBFS.

    Deliberately stdlib only (`wave` + `array`): the silence gate has to work for every
    engine, including on a Mac where numpy is only present as an mlx transitive dependency.
    Reading just the frames for the span keeps this cheap even on a long file.
    """
    import wave
    from array import array

    try:
        with wave.open(str(wav_path), "rb") as wav:
            if wav.getsampwidth() != 2:
                return 0.0  # not 16-bit; do not guess, treat as loud and keep the segment
            rate = wav.getframerate()
            channels = wav.getnchannels()
            total = wav.getnframes()

            first = max(0, min(total, int(start * rate)))
            last = max(first, min(total, int(end * rate)))
            if last <= first:
                return -float("inf")

            wav.setpos(first)
            raw = wav.readframes(last - first)
    except (OSError, wave.Error):
        return 0.0

    samples = array("h")
    samples.frombytes(raw[: len(raw) - (len(raw) % (2 * channels))])
    if not samples:
        return -float("inf")

    peak = max(abs(s) for s in samples)
    if peak == 0:
        return -float("inf")
    import math

    return 20 * math.log10(peak / 32768.0)


def drop_silent_segments(
    segments: tuple[Any, ...],
    wav_path: Path,
    *,
    threshold_dbfs: float = SILENT_PEAK_DBFS,
) -> tuple[Any, ...]:
    """Remove segments whose audio is essentially silent.

    Whisper invents filler over silence ("Hvala.", "Thank you.", "Titlovi ..."). Dropping
    the span outright beats filtering the text afterwards: by then the hallucination has
    already claimed a timestamp range and skewed the segmentation around it.
    """
    if not wav_path.exists():
        return segments
    return tuple(s for s in segments if peak_dbfs(wav_path, s.start, s.end) > threshold_dbfs)


def prompt_echoed(text: str, prompt: str | None) -> tuple[int, str]:
    """How much of the steering prompt came back as transcript text, and which part.

    One detector for this failure, not a second one. `bench.metrics.prompt_echo` was
    written when `--denoise arnndn` made the **sequential** path open with the tail of
    `SERBIAN_PROMPT` instead of the first fifty words of a lecture, and it is exactly the
    check the decode path needs: a contiguous run of the prompt's own tokens in the
    prompt's own order, which no ordinary Serbian sentence produces by accident.

    Imported inside the function because `bench.metrics` imports `SERBIAN_PROMPT` from
    this module, and a module-level import here would close that circle.
    """
    if not prompt or not text:
        return 0, ""
    from subtitler.bench.metrics import prompt_echo

    return prompt_echo(text, prompt)


def collapse_repetition(
    text: str,
    *,
    max_repeats: int = MAX_REPEATS,
    max_phrase_tokens: int = MAX_REPEAT_PHRASE_TOKENS,
) -> str:
    """Collapse a phrase repeated more than `max_repeats` times in a row.

    Scans phrase lengths from longest to shortest so "ne znam ne znam ne znam" collapses as
    one two-word phrase rather than as two separate single-word runs.
    """
    tokens = text.split()
    if len(tokens) < max_repeats:
        return text

    for size in range(max_phrase_tokens, 0, -1):
        i = 0
        out: list[str] = []
        changed = False
        while i < len(tokens):
            phrase = tokens[i : i + size]
            if len(phrase) < size:
                out.extend(tokens[i:])
                break
            repeats = 1
            j = i + size
            while tokens[j : j + size] == phrase:
                repeats += 1
                j += size
            if repeats > max_repeats:
                out.extend(phrase)
                changed = True
                i = j
            else:
                out.extend(tokens[i : i + size])
                i += size
        if changed:
            tokens = out
    return " ".join(tokens)
