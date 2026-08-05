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

# The speech-free gate. Every threshold below was measured against this project's own two
# Serbian fixtures (25 segments of real speech) and against ffmpeg-generated music-only
# clips, rather than chosen from theory. The numbers are in `docs/STATUS.md`.
#
# Whisper's own `no_speech_threshold` default, and comfortably above the 0.436 that is the
# highest `no_speech_prob` any genuine speech segment in the fixtures produces.
NO_SPEECH_PROB_MAX = 0.60
# Mean word probability below which the decoder plainly did not recognise anything. The
# lowest a real segment reaches is 0.779, so this only ever fires on gibberish.
MIN_WORD_CONFIDENCE = 0.35
# A transcript this sparse is not a transcription. Both speech fixtures run at 1.40 and 1.70
# words per second of audio; the music-only clips produce 0.13 to 0.40.
SPEECHLESS_WORDS_PER_SECOND = 0.5
# Whisper's memorised non-speech boilerplate comes back *confident*, so confidence alone
# cannot be the test. What separates it is that genuine filler is maximally confident: the
# one real filler segment in the fixtures ("Hvala." in `gozba-sample.mp3`) scores 1.000,
# while every hallucinated one scores 0.797 to 0.856.
FILLER_CONFIDENCE_MAX = 0.95

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


def mean_word_confidence(segment: Any) -> float | None:
    """Mean probability over a segment's words, or None when the backend supplies none.

    None and 0.0 are different facts and are kept apart all the way to the gate: Groq
    returns word timings with no probabilities at all, and a missing confidence must never
    read as no confidence.
    """
    probs = [w.prob for w in getattr(segment, "words", ()) if w.prob is not None]
    return sum(probs) / len(probs) if probs else None


def is_speechless(segment: Any, *, words_per_second: float | None = None) -> str:
    """Why this segment holds no speech, or "" when it looks like speech.

    Three tests, in order of how directly the model incriminates itself. Each one exists
    because the one before it demonstrably misses the case that follows.

    1. **The no-speech head fired.** What `no_speech_prob` is for, and the cheapest true
       signal available. It catches the honest cases: a decoder that knows it is looking at
       music usually says so (0.80 to 0.94 on the music clips it fired for).

    2. **The words mean nothing.** A mean word probability under `MIN_WORD_CONFIDENCE` is a
       decoder emitting tokens it cannot justify, which is what a weak model does over
       noise ("..." at 0.004 to 0.219).

    3. **Memorised boilerplate in a transcript that carries no speech.** This is the case
       the first two miss, and it is the bug this gate was written for. Ten seconds of
       titles and music produced `Hvala što pratite kanal.` with `no_speech_prob` 0.14 and
       mean word confidence 0.80: the model is *confident*, because it is reciting YouTube
       outro text it learned rather than guessing at audio. Neither 1 nor 2 can be moved to
       catch it without destroying real speech, which reaches `no_speech_prob` 0.436 and
       mean confidence 0.779 in the very same fixtures. The three conjuncts are what make
       this safe, and each protects a specific real case:

       * the text is one of `bench.metrics.FILLER_PHRASES`, Whisper's known non-speech
         filler, which is already the project's list for exactly this and is used here to
         decide rather than merely to count;
       * the transcript around it carries almost no speech, which is what tells one
         hallucinated `Hvala.` apart from the genuine `Hvala.` in the middle of
         `gozba-sample.mp3`, a 109-second file running at 1.40 words per second;
       * and it is not maximally confident, which is what protects a legitimately short
         clip of someone actually saying thank you. Genuine filler scores 1.000.

    Returns a reason string rather than a bool so the pipeline can tell the user which test
    fired. A segment nothing incriminates is kept: an engine that supplies neither signal
    gets its text through untouched, because absence of evidence is not evidence, and a
    silent drop on a missing field is the very failure this gate exists to prevent.
    """
    no_speech = getattr(segment, "no_speech_prob", None)
    if no_speech is not None and no_speech >= NO_SPEECH_PROB_MAX:
        return f"the model reports no speech here (no_speech_prob {no_speech:.2f})"

    confidence = mean_word_confidence(segment)
    if confidence is not None and confidence < MIN_WORD_CONFIDENCE:
        return f"the words carry almost no confidence (mean {confidence:.2f})"

    if (
        words_per_second is not None
        and words_per_second < SPEECHLESS_WORDS_PER_SECOND
        # `is not None` and not merely "below the ceiling": an engine that reports no word
        # probabilities at all (Groq) abstains from this test rather than failing it, so
        # the only thing that can delete its text is its own no-speech head.
        and confidence is not None
        and confidence < FILLER_CONFIDENCE_MAX
    ):
        from subtitler.bench.metrics import filler_hits

        hits = filler_hits(segment.text)
        if hits:
            return (
                f"known non-speech filler ({', '.join(sorted(hits))!r}) in audio holding "
                f"{words_per_second:.2f} words per second"
            )

    return ""


def drop_speechless_segments(
    segments: tuple[Any, ...],
    *,
    duration: float = 0.0,
) -> tuple[tuple[Any, ...], list[str]]:
    """Remove segments that hold no speech, and say what was removed and why.

    The companion to `drop_silent_segments`, and applied by every engine for the same
    reason: the silence gate reads the waveform and so only ever catches *quiet* audio,
    while the failure that reaches a user is loud. Music, titles and applause all sail past
    a -60 dBFS peak check and still contain no speech for Whisper to transcribe, so it
    invents some.

    `duration` is the length of the audio, used for the word-rate term in `is_speechless`.
    Zero means "unknown", which disables that third test rather than guessing at it.

    Returns `(kept, reasons)`. The reasons are handed up rather than logged: an engine has
    no log channel, and the pipeline has to be able to tell the user that text was removed.
    """
    text_words = sum(len(s.text.split()) for s in segments)
    rate = (text_words / duration) if duration > 0 else None

    kept: list[Any] = []
    reasons: list[str] = []
    for seg in segments:
        reason = is_speechless(seg, words_per_second=rate)
        if reason:
            reasons.append(f"[{seg.start:.1f}s] {seg.text!r}: {reason}")
        else:
            kept.append(seg)
    return tuple(kept), reasons


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
