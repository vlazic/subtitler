"""SRT and VTT writing, parsing, and validation.

Timestamps go through integer milliseconds before formatting. The code this replaced used
`f"{seconds:06.3f}"`, which renders 59.9996 as "60.000" and emits the invalid timestamp
00:00:60.000 that strict parsers reject.
"""

from __future__ import annotations

import re
from pathlib import Path

from subtitler.model import Cue

_VTT_HEADER = "WEBVTT"

# Accepts both HH:MM:SS,mmm (SRT) and HH:MM:SS.mmm (VTT), and the MM:SS.mmm short form
# that some tools emit for clips under an hour.
_TIME_RE = re.compile(r"(?:(\d+):)?(\d{1,2}):(\d{2})[.,](\d{1,3})")
_ARROW_RE = re.compile(r"\s*-->\s*")


def _clock(seconds: float, sep: str) -> str:
    ms_total = round(max(seconds, 0.0) * 1000)
    hours, ms_total = divmod(ms_total, 3_600_000)
    minutes, ms_total = divmod(ms_total, 60_000)
    secs, ms = divmod(ms_total, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}{sep}{ms:03d}"


def srt_clock(seconds: float) -> str:
    return _clock(seconds, ",")


def vtt_clock(seconds: float) -> str:
    return _clock(seconds, ".")


def parse_clock(text: str) -> float:
    match = _TIME_RE.fullmatch(text.strip())
    if not match:
        raise ValueError(f"unparseable timestamp: {text!r}")
    hours, minutes, secs, frac = match.groups()
    ms = int(frac.ljust(3, "0"))
    return int(hours or 0) * 3600 + int(minutes) * 60 + int(secs) + ms / 1000


# --------------------------------------------------------------------------------------
# Writing
# --------------------------------------------------------------------------------------


def render_srt(cues: tuple[Cue, ...]) -> str:
    blocks = [
        f"{i}\n{srt_clock(c.start)} --> {srt_clock(c.end)}\n" + "\n".join(c.lines)
        for i, c in enumerate(cues, start=1)
    ]
    return "\n\n".join(blocks) + "\n" if blocks else ""


def render_vtt(cues: tuple[Cue, ...]) -> str:
    # Build blocks and join once. Seeding the list with "WEBVTT\n" and joining with "\n"
    # while each cue also starts with "\n" is what produced the stray double blank line in
    # every VTT the old converter emitted.
    blocks = [f"{vtt_clock(c.start)} --> {vtt_clock(c.end)}\n" + "\n".join(c.lines) for c in cues]
    return "\n\n".join([_VTT_HEADER, *blocks]) + "\n"


def write_srt(path: Path, cues: tuple[Cue, ...]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_srt(cues), encoding="utf-8")
    return path


def write_vtt(path: Path, cues: tuple[Cue, ...]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_vtt(cues), encoding="utf-8")
    return path


# --------------------------------------------------------------------------------------
# Parsing. Needed by `subtitler burn` and `subtitler lint`, which take a subtitle file
# rather than a pipeline run.
# --------------------------------------------------------------------------------------


def parse_subtitles(text: str) -> tuple[Cue, ...]:
    """Parse SRT or VTT. The two differ only in the header and the decimal separator."""
    body = text.lstrip("﻿")
    if body.lstrip().startswith(_VTT_HEADER):
        body = body.split("\n", 1)[1] if "\n" in body else ""

    cues: list[Cue] = []
    for raw_block in re.split(r"\n\s*\n", body.strip()):
        block = raw_block.strip()
        if not block:
            continue
        lines = block.split("\n")
        # An SRT block opens with a bare cue number; VTT may carry an optional cue id.
        if "-->" not in lines[0]:
            lines = lines[1:]
        if not lines or "-->" not in lines[0]:
            continue
        start_text, end_text = _ARROW_RE.split(lines[0], maxsplit=1)
        # VTT allows cue settings after the end time, e.g. "align:start position:10%".
        end_text = end_text.split()[0] if end_text.split() else end_text
        payload = tuple(line for line in lines[1:] if line.strip())
        if not payload:
            continue
        cues.append(
            Cue(
                index=len(cues) + 1,
                start=parse_clock(start_text),
                end=parse_clock(end_text),
                lines=payload,
            )
        )
    return tuple(cues)


def read_subtitles(path: Path) -> tuple[Cue, ...]:
    return parse_subtitles(path.read_text(encoding="utf-8"))


# --------------------------------------------------------------------------------------
# Validation. The Deno tool this replaced ran a real validate step; keeping one means a
# malformed file is caught here rather than by a player.
# --------------------------------------------------------------------------------------


def validate_vtt(text: str) -> list[str]:
    """Structural problems only. Cue-quality rules live in `lint`."""
    problems: list[str] = []
    if not text.lstrip("﻿").startswith(_VTT_HEADER):
        problems.append("missing WEBVTT header")

    cues = parse_subtitles(text)
    if not cues:
        problems.append("no cues found")

    previous_end = -1.0
    for cue in cues:
        label = f"cue {cue.index} at {vtt_clock(cue.start)}"
        if cue.end <= cue.start:
            problems.append(f"{label}: end is not after start")
        if cue.start < previous_end - 1e-6:
            problems.append(f"{label}: overlaps the previous cue")
        if not any(line.strip() for line in cue.lines):
            problems.append(f"{label}: empty text")
        previous_end = max(previous_end, cue.end)
    return problems
