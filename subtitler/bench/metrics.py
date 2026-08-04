"""Every number the benchmark reports is computed here.

Non-negotiable 9: no leaderboard number ever comes from an LLM. Phase 8 may use an agent to
*adjudicate a reference transcript*, and that is the only place a model is allowed near this
data; the scoring itself is arithmetic over strings, and it lives in this file so that stays
easy to check.

Two families of metric, and the distinction matters for reading the report:

**Reference-based** (`score`) needs a ground-truth transcript, and answers "how much of this
is wrong". WER, CER, and the substitution/insertion/deletion split, plus WER_folded: the
same score with the Serbian diacritics folded away. The gap between WER and WER_folded is
the useful part. A model that hears the right word and writes `c` for `č` is one
find-and-replace from correct; a model that hears the wrong word is not.

**Reference-free** (`cue_stats`, `hallucination`, and the timings the runner collects) needs
nothing, and answers "is this output shaped like a usable subtitle track". Those numbers are
available for every cell in the matrix whether a reference exists or not, which is what
makes the harness useful before Phase 8 lands.

`jiwer` is imported inside the functions that need it. It is in the `bench` extra only, and
CI syncs without that extra: importing this module has to work anywhere, so that the
reference-free half stays testable on both runners.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

from subtitler.bench import normalize as norm
from subtitler.cues import CueConfig, display_len
from subtitler.model import Cue

__all__ = [
    "FILLER_PHRASES",
    "CueStats",
    "Hallucination",
    "ReferenceScore",
    "cue_stats",
    "filler_hits",
    "hallucination",
    "longest_repeated_ngram",
    "percentile",
    "score",
]

# Whisper's silence filler, in the two languages this material produces it in. These are
# counted, never removed: a count is evidence about a cell, while a removal would quietly
# improve the very output being measured. "Titlovi" and "Prevod" are the openings of the
# Serbian subtitling credits Whisper learned from its training data ("Titlovi: ...",
# "Prevod: ..."); "Hvala" and "Subscribe" are what it says over silence.
FILLER_PHRASES: tuple[str, ...] = (
    "hvala",
    "titlovi",
    "prevod",
    "subscribe",
    "thank you",
    "hvala na gledanju",
    "titlovi po sluhu",
)


@dataclass(frozen=True, slots=True)
class ReferenceScore:
    """WER and friends against a reference. Every field is a fraction, not a percentage."""

    wer: float
    wer_folded: float
    cer: float
    substitutions: int
    insertions: int
    deletions: int
    hits: int
    reference_words: int
    hypothesis_words: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class CueStats:
    """Shape of the produced subtitle track, measured against the layout limits."""

    count: int
    total_chars: int
    mean_cps: float
    p95_cps: float
    max_cps: float
    over_line_pct: float
    over_lines_pct: float
    over_dur_pct: float
    over_cps_pct: float
    under_min_dur_pct: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class Hallucination:
    """Heuristics, named as such. Each is evidence, none is a verdict."""

    longest_repeat_n: int
    longest_repeat_text: str
    repetition_collapsed: int | None
    silence_dropped: int | None
    filler_hits: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# --------------------------------------------------------------------------------------
# Reference-based
# --------------------------------------------------------------------------------------


def score(reference: str, hypothesis: str) -> ReferenceScore:
    """WER, WER_folded, CER and the error decomposition, over normalized text.

    Both sides go through the identical normalizer, which is the only way the number means
    anything: scoring `Ђорђе, 20 година.` against `đorđe 20 godina` unnormalized would
    report a total failure of a transcript that is word for word correct.

    An empty reference cannot produce a rate (the denominator is zero), so it raises rather
    than reporting a confident 0.0 or 1.0. The caller knows whether it has a reference; this
    function should not have to guess.
    """
    import jiwer

    ref = norm.normalize(reference)
    hyp = norm.normalize(hypothesis)
    if not ref:
        raise ValueError("cannot score against an empty reference")

    words = jiwer.process_words(ref, hyp)
    folded = jiwer.process_words(
        norm.normalize(reference, fold=True), norm.normalize(hypothesis, fold=True)
    )
    return ReferenceScore(
        wer=float(words.wer),
        wer_folded=float(folded.wer),
        # CER over the normalized strings, spaces included: a missing word boundary is a
        # real character error, and dropping spaces would hide it.
        cer=float(jiwer.cer(ref, hyp)),
        substitutions=int(words.substitutions),
        insertions=int(words.insertions),
        deletions=int(words.deletions),
        hits=int(words.hits),
        reference_words=len(ref.split()),
        hypothesis_words=len(hyp.split()),
    )


# --------------------------------------------------------------------------------------
# Reference-free
# --------------------------------------------------------------------------------------


def percentile(values: Sequence[float], fraction: float) -> float:
    """Nearest-rank percentile, stdlib only.

    numpy is in the `local` extra, not the core, and a benchmark metric must not be the
    reason a Mac needs it. Nearest-rank also has no interpolation to argue about: p95 of a
    20-element sample is the 19th value, which is a value that genuinely occurred.
    """
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = max(1, math.ceil(fraction * len(ordered)))
    return float(ordered[min(rank, len(ordered)) - 1])


def cue_stats(cues: Sequence[Cue], config: CueConfig | None = None) -> CueStats:
    """Reading speed and limit violations, as percentages of the cue count.

    Line width is measured with `cues.display_len`, the same function `lint` uses, so
    `<b>` and `<i>` count as markup rather than as width. A cue with a violation-free
    layout must never be reported as over the limit just because `--fix-markup html` was on.
    """
    cfg = config or CueConfig()
    if not cues:
        return CueStats(0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    cps = [c.cps for c in cues if math.isfinite(c.cps)]
    n = len(cues)
    over_line = sum(1 for c in cues if any(display_len(line) > cfg.max_line for line in c.lines))
    over_lines = sum(1 for c in cues if len(c.lines) > cfg.max_lines)
    over_dur = sum(1 for c in cues if c.duration > cfg.max_dur)
    over_cps = sum(1 for c in cues if c.cps > cfg.max_cps)
    under_dur = sum(1 for c in cues if c.duration < cfg.min_dur)

    return CueStats(
        count=n,
        total_chars=sum(display_len(c.text) for c in cues),
        mean_cps=(sum(cps) / len(cps)) if cps else 0.0,
        p95_cps=percentile(cps, 0.95),
        max_cps=max(cps) if cps else 0.0,
        over_line_pct=100.0 * over_line / n,
        over_lines_pct=100.0 * over_lines / n,
        over_dur_pct=100.0 * over_dur / n,
        over_cps_pct=100.0 * over_cps / n,
        under_min_dur_pct=100.0 * under_dur / n,
    )


def longest_repeated_ngram(text: str, *, min_n: int = 2) -> tuple[int, str]:
    """The longest word sequence that occurs more than once, as (length, text).

    The signature of a decoder repetition loop that `collapse_repetition` did not catch:
    that function only collapses a phrase repeated more than six times *consecutively*, so a
    model that emits the same nine words at 0:40 and again at 1:10 passes it untouched.

    Implemented over sorted suffixes: sort the token suffixes, and the longest repeat is the
    longest common prefix of some adjacent pair, which needs one linear scan instead of a
    dictionary per n. Overlaps count, deliberately: `ne znam ne znam ne znam` is a repeat
    whichever way it is cut.
    """
    words = norm.tokens(text)
    if len(words) < min_n * 2:
        return 0, ""

    suffixes = sorted(range(len(words)), key=lambda i: words[i:])
    best_len, best_at = 0, 0
    for a, b in itertools.pairwise(suffixes):
        shared = 0
        limit = len(words) - max(a, b)
        while shared < limit and words[a + shared] == words[b + shared]:
            shared += 1
        if shared > best_len:
            best_len, best_at = shared, a

    if best_len < min_n:
        return 0, ""
    return best_len, " ".join(words[best_at : best_at + best_len])


def filler_hits(text: str, phrases: Sequence[str] = FILLER_PHRASES) -> dict[str, int]:
    """How often each known filler phrase appears, over normalized whole tokens.

    Token-aligned rather than substring: `prevod` must not fire on `prevodilac`, and
    `hvala` must not fire inside a sentence that merely contains it as part of another
    word. A real "Hvala." cue is a whole token on its own.
    """
    words = norm.tokens(text)
    counts: dict[str, int] = {}
    for phrase in phrases:
        needle = norm.tokens(phrase)
        if not needle:
            continue
        found = sum(
            1 for i in range(len(words) - len(needle) + 1) if words[i : i + len(needle)] == needle
        )
        if found:
            counts[phrase] = found
    return counts


def hallucination(
    text: str,
    *,
    repetition_collapsed: int | None = None,
    silence_dropped: int | None = None,
) -> Hallucination:
    """Everything that suggests the model produced text the speaker did not say.

    The two counters are passed in rather than derived: they are recorded by the engine
    while it decodes (`Transcript.params`), because by the time the transcript exists the
    collapsed repetitions and the dropped silent segments are gone from it by definition.
    `None` means the engine did not report them, which is not the same as zero and is kept
    distinct all the way into the report.
    """
    length, phrase = longest_repeated_ngram(text)
    return Hallucination(
        longest_repeat_n=length,
        longest_repeat_text=phrase,
        repetition_collapsed=repetition_collapsed,
        silence_dropped=silence_dropped,
        filler_hits=filler_hits(text),
    )
