"""Segments in, display cues out.

This is the file that separates a readable subtitle from a wall of text. Whisper emits one
segment per utterance, which can run 5 to 25 seconds unbroken; every earlier tool in this
lineage wrote that straight out as a single cue.

Phase 1 implements the interface and a greedy wrap. Phase 4 replaces `segments_to_cues`
with the real splitter (sentence and clause boundaries, balanced lines, Serbian clitic
rules, duration and reading-speed enforcement). Nothing outside this module needs to change
when that lands.
"""

from __future__ import annotations

from dataclasses import dataclass

from subtitler.model import Cue, Segment, ensure_words


@dataclass(frozen=True, slots=True)
class CueConfig:
    """Defaults follow the usual broadcast conventions (BBC/Netflix style guides)."""

    max_line: int = 42
    max_lines: int = 2
    min_dur: float = 1.0
    max_dur: float = 7.0
    max_cps: float = 20.0
    min_gap: float = 0.08


def wrap_text(text: str, *, max_line: int, max_lines: int) -> tuple[str, ...]:
    """Greedy word wrap, then balance the last two lines.

    Balancing matters: a 40/4 split reads far worse than 22/22 even though both satisfy the
    character limit.
    """
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
    """Move words back from the first line while both lines still fit and the split evens out."""
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


def segments_to_cues(
    segments: tuple[Segment, ...], config: CueConfig | None = None
) -> tuple[Cue, ...]:
    """Phase 1: one cue per segment, greedily wrapped.

    Known to violate max_dur and max_cps on long segments. `lint_cues` reports those
    violations honestly rather than the pipeline pretending they are not there.
    """
    cfg = config or CueConfig()
    segments = ensure_words(segments)

    cues: list[Cue] = []
    for seg in segments:
        text = seg.text.strip()
        if not text:
            continue
        lines = wrap_text(text, max_line=cfg.max_line, max_lines=cfg.max_lines)
        if not lines:
            continue
        cues.append(Cue(index=len(cues) + 1, start=seg.start, end=seg.end, lines=lines))
    return tuple(cues)


def lint_cues(cues: tuple[Cue, ...], config: CueConfig | None = None) -> list[str]:
    """Report every cue-quality violation. Empty list means the file meets the bar."""
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
