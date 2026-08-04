"""Segments in, display cues out. The file that separates a readable subtitle from a wall
of text.

Whisper emits one segment per utterance, which routinely runs 5 to 25 seconds unbroken.
Every earlier tool in this lineage wrote that straight out as a single cue, which is fine
for a scrolling transcript on a web page and unusable burned into a video.

The pipeline here, over word-level timings:

  1. split   at the most natural boundary, recursively, until every chunk fits
  2. wrap    into balanced lines that never break a phrase apart
  3. merge   adjacent scraps that are too small to read on their own
  4. time    enforce minimum duration and reading speed, using the gaps between cues

Reading speed cannot be fixed by splitting: characters per second is a ratio, so halving
the text and the time leaves it unchanged. It is fixed by extending a cue into the silence
that follows, and when there is no silence to borrow, `lint` reports it rather than the
code silently mangling the text.
"""

from __future__ import annotations

from dataclasses import dataclass

from subtitler.model import Cue, Segment, Word, ensure_words

# Serbian enclitics. These lean on the word before them and must never start a line:
# a break in front of "je" or "se" reads as a stutter.
CLITICS = frozenset(
    (
        "je",
        "su",
        "sam",
        "si",
        "smo",
        "ste",
        "bi",
        "bih",
        "bismo",
        "biste",
        "li",
        "ću",
        "ćeš",
        "će",
        "ćemo",
        "ćete",
        "me",
        "te",
        "ga",
        "ih",
        "im",
        "mu",
        "joj",
        "nas",
        "vas",
        "se",
        "ni",
        "mi",
        "ti",
    )
)

# Prepositions bind forward to their noun, so a break directly after one strands it.
PREPOSITIONS = frozenset(
    (
        "u",
        "na",
        "o",
        "po",
        "za",
        "sa",
        "s",
        "iz",
        "od",
        "do",
        "kod",
        "pred",
        "pod",
        "nad",
        "kroz",
        "uz",
        "niz",
        "pri",
        "prema",
        "bez",
        "oko",
        "protiv",
        "među",
        "radi",
        "zbog",
        "preko",
        "posle",
        "pre",
        "nakon",
        "tokom",
    )
)

# Conjunctions and relatives are good places to start a new cue.
CONJUNCTIONS = frozenset(
    (
        "i",
        "a",
        "ali",
        "ili",
        "jer",
        "pa",
        "te",
        "da",
        "koji",
        "koja",
        "koje",
        "kojeg",
        "kojoj",
        "kada",
        "kad",
        "ako",
        "dok",
        "zato",
        "mada",
        "iako",
        "nego",
        "već",
    )
)

_SENTENCE_END = (".", "!", "?", "…")
# \u2014 em dash, \u2013 en dash: written as escapes so the source stays unambiguous.
_CLAUSE_END = (",", ";", ":", "\u2014", "\u2013")

# A pause this long between words is a natural boundary even without punctuation.
_PAUSE_S = 0.25

# Worse than every real rank, so an allowed break always wins over a forbidden one.
_FORBIDDEN_RANK = 99


@dataclass(frozen=True, slots=True)
class CueConfig:
    """Defaults follow the usual broadcast conventions (BBC and Netflix style guides)."""

    max_line: int = 42
    max_lines: int = 2
    min_dur: float = 1.0
    max_dur: float = 7.0
    max_cps: float = 20.0
    min_gap: float = 0.08
    # Adjacent scraps shorter than this in both time and characters get merged.
    merge_under_chars: int = 20
    merge_gap: float = 0.3

    @property
    def max_chars(self) -> int:
        return self.max_line * self.max_lines


# --------------------------------------------------------------------------------------
# Word helpers
# --------------------------------------------------------------------------------------


def _text_of(words: tuple[Word, ...]) -> str:
    return " ".join(w.text for w in words if w.text)


def _chars(words: tuple[Word, ...]) -> int:
    return len(_text_of(words))


def _bare(token: str) -> str:
    return token.strip().strip("".join(_SENTENCE_END) + "".join(_CLAUSE_END) + "\"'()„“»«").lower()


def _break_allowed(words: tuple[Word, ...], index: int) -> bool:
    """May a break go before `words[index]`?"""
    if index <= 0 or index >= len(words):
        return False
    if _bare(words[index].text) in CLITICS:
        return False
    return _bare(words[index - 1].text) not in PREPOSITIONS


def _break_rank(words: tuple[Word, ...], index: int) -> int:
    """Lower is a better place to break. Ranks are tried in order, never mixed."""
    previous = words[index - 1].text.rstrip()
    following = _bare(words[index].text)

    if previous.endswith(_SENTENCE_END):
        return 0
    if previous.endswith(_CLAUSE_END):
        return 1
    if following in CONJUNCTIONS:
        return 2
    if words[index].start - words[index - 1].end >= _PAUSE_S:
        return 3
    return 4


def _best_break(words: tuple[Word, ...], cfg: CueConfig) -> int | None:
    """Pick the split point: best rank first, then closest to the middle by characters.

    Balancing by character count rather than by word count matters, because an unbalanced
    split produces a cue that is still too long and has to be split again anyway.
    """
    if len(words) < 2:
        return None

    total = _chars(words)
    best: tuple[int, float, int] | None = None  # (rank, imbalance, index)

    for index in range(1, len(words)):
        if not _break_allowed(words, index):
            continue
        left = _chars(words[:index])
        # Skip splits that cannot help: one side would still exceed a whole cue.
        imbalance = abs(left - (total - left)) / max(total, 1)
        candidate = (_break_rank(words, index), imbalance, index)
        if best is None or candidate < best:
            best = candidate

    return best[2] if best else None


# --------------------------------------------------------------------------------------
# 1. Splitting
# --------------------------------------------------------------------------------------


def _too_big(words: tuple[Word, ...], cfg: CueConfig) -> bool:
    """Does this run of words need splitting?

    Length always forces a split. Duration only does when there is enough text to make
    both halves readable on their own: a long timespan carrying few words means Whisper
    attributed silence to the phrase, and splitting "Misao Lokove filozofije" into a
    four-second "Misao Lokove" and a four-second "filozofije" is worse than leaving it
    whole and trimming how long it stays on screen (see `_timed`).
    """
    if not words:
        return False
    if _chars(words) > cfg.max_chars:
        return True
    duration = words[-1].end - words[0].start
    return duration > cfg.max_dur and _chars(words) > cfg.max_line


def split_words(words: tuple[Word, ...], cfg: CueConfig) -> list[tuple[Word, ...]]:
    """Recursively split a run of words until every piece fits a cue."""
    if not words:
        return []
    if not _too_big(words, cfg):
        return [words]

    index = _best_break(words, cfg)
    if index is None:
        # Nothing may be broken (a single very long word, or every boundary forbidden).
        # Returning it whole is honest: lint will flag it.
        return [words]

    return split_words(words[:index], cfg) + split_words(words[index:], cfg)


# --------------------------------------------------------------------------------------
# 2. Wrapping
# --------------------------------------------------------------------------------------


def wrap_words(words: tuple[Word, ...], cfg: CueConfig) -> tuple[str, ...]:
    """Lay a cue's words out over at most `max_lines` balanced lines."""
    text = _text_of(words)
    if len(text) <= cfg.max_line or len(words) < 2:
        return (text,) if text else ()

    if cfg.max_lines == 2:
        index = _best_wrap_point(words, cfg)
        if index is not None:
            return (_text_of(words[:index]), _text_of(words[index:]))

    return wrap_text(text, max_line=cfg.max_line, max_lines=cfg.max_lines)


def _best_wrap_point(words: tuple[Word, ...], cfg: CueConfig) -> int | None:
    """The break that balances two lines best while respecting the phrase rules."""
    total = _chars(words)
    best: tuple[int, float, int] | None = None

    for index in range(1, len(words)):
        left = _chars(words[:index])
        right = total - left - 1  # the space that becomes the newline
        if left > cfg.max_line or right > cfg.max_line:
            continue
        # A forbidden break is still better than an overflowing line, so it is ranked
        # last rather than excluded outright. It must rank strictly worse than an
        # ordinary allowed break, though: sharing rank 4 with them let a better-balanced
        # forbidden break win, which is how "s / jedne strane" stranded a preposition.
        rank = _break_rank(words, index) if _break_allowed(words, index) else _FORBIDDEN_RANK
        candidate = (rank, abs(left - right) / max(total, 1), index)
        if best is None or candidate < best:
            best = candidate

    return best[2] if best else None


def wrap_text(text: str, *, max_line: int, max_lines: int) -> tuple[str, ...]:
    """Greedy word wrap with a balancing pass. Used where word timings are unavailable."""
    words = text.split()
    if not words:
        return ()

    lines: list[str] = []
    current = ""
    for word in words:
        candidate = f"{current} {word}".strip()
        if current and len(candidate) > max_line:
            lines.append(current)
            current = word
        else:
            current = candidate
    if current:
        lines.append(current)

    if len(lines) == 2:
        lines = _balance(lines[0], lines[1], max_line)
    return tuple(lines[:max_lines]) if len(lines) > max_lines else tuple(lines)


def _balance(first: str, second: str, max_line: int) -> list[str]:
    """A 40/4 split satisfies the character limit and still reads badly."""
    best = [first, second]
    best_delta = abs(len(first) - len(second))
    words = first.split()
    for cut in range(len(words) - 1, 0, -1):
        left = " ".join(words[:cut])
        right = " ".join(words[cut:] + second.split())
        if len(right) > max_line:
            break
        delta = abs(len(left) - len(right))
        if delta < best_delta:
            best, best_delta = [left, right], delta
    return best


# --------------------------------------------------------------------------------------
# 3 and 4. Merging and timing
# --------------------------------------------------------------------------------------


def _merge_scraps(groups: list[tuple[Word, ...]], cfg: CueConfig) -> list[tuple[Word, ...]]:
    """Fold a too-short fragment into its neighbour when they belong together.

    A 0.4 second cue reading "Da." flashes past unread. Merging it with the next cue is
    better than leaving it, as long as the result still fits.
    """
    out: list[tuple[Word, ...]] = []
    for group in groups:
        if not out:
            out.append(group)
            continue

        previous = out[-1]
        prev_dur = previous[-1].end - previous[0].start
        gap = group[0].start - previous[-1].end
        combined = previous + group

        too_short = prev_dur < cfg.min_dur and _chars(previous) < cfg.merge_under_chars
        if too_short and gap < cfg.merge_gap and not _too_big(combined, cfg):
            out[-1] = combined
        else:
            out.append(group)
    return out


def _timed(groups: list[tuple[Word, ...]], cfg: CueConfig) -> list[Cue]:
    """Give every cue a readable duration, borrowing from the silence that follows.

    Never overlaps the next cue, never shrinks a cue, and never stretches past max_dur.
    A cue that still cannot meet the reading-speed limit is left alone for lint to
    report: the alternative is deleting words the speaker actually said.
    """
    cues: list[Cue] = []
    for i, words in enumerate(groups):
        lines = wrap_words(words, cfg)
        if not lines:
            continue

        start = words[0].start
        text_len = len(" ".join(lines))

        # Trim a span that outlives its text. A subtitle does not need to sit on screen
        # through trailing silence, and the alternative (splitting) produces one-word cues.
        end = min(words[-1].end, start + cfg.max_dur)

        wanted = max(end, start + cfg.min_dur, start + text_len / cfg.max_cps)
        wanted = min(wanted, start + cfg.max_dur)

        if i + 1 < len(groups):
            ceiling = groups[i + 1][0].start - cfg.min_gap
            wanted = min(wanted, max(ceiling, end))

        cues.append(Cue(index=len(cues) + 1, start=start, end=max(end, wanted), lines=lines))
    return cues


# --------------------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------------------


def segments_to_cues(
    segments: tuple[Segment, ...], config: CueConfig | None = None
) -> tuple[Cue, ...]:
    cfg = config or CueConfig()
    segments = ensure_words(segments)

    groups: list[tuple[Word, ...]] = []
    for segment in segments:
        words = tuple(w for w in segment.words if w.text.strip())
        if not words:
            continue
        groups.extend(split_words(words, cfg))

    groups = _merge_scraps(groups, cfg)
    return tuple(_timed(groups, cfg))


def lint_cues(cues: tuple[Cue, ...], config: CueConfig | None = None) -> list[str]:
    """Report every cue-quality violation. An empty list means the file meets the bar."""
    cfg = config or CueConfig()
    problems: list[str] = []

    for i, cue in enumerate(cues):
        label = f"cue {cue.index}"
        if len(cue.lines) > cfg.max_lines:
            problems.append(f"{label}: {len(cue.lines)} lines (max {cfg.max_lines})")
        for n, line in enumerate(cue.lines, start=1):
            if len(line) > cfg.max_line:
                problems.append(f"{label} line {n}: {len(line)} chars (max {cfg.max_line})")
        if cue.duration < cfg.min_dur - 1e-6:
            problems.append(f"{label}: {cue.duration:.2f}s is under the {cfg.min_dur}s minimum")
        if cue.duration > cfg.max_dur + 1e-6:
            problems.append(f"{label}: {cue.duration:.2f}s exceeds the {cfg.max_dur}s maximum")
        if cue.cps > cfg.max_cps + 1e-6:
            problems.append(f"{label}: {cue.cps:.1f} chars/sec exceeds {cfg.max_cps}")
        if i + 1 < len(cues) and cues[i + 1].start < cue.end - 1e-6:
            problems.append(f"{label}: overlaps cue {cues[i + 1].index}")
    return problems
