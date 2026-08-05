"""The human pass over an adjudicated reference: hear the span, settle the span.

`agents.py` produces a *consensus pseudo-reference* and is honest about what that is worth.
Every WER derived from it carries `*`, `human_verified` stays `false`, and the spans the
adjudicator was unsure about plus the ones its critic disputed are listed in
`benchmarks/references/review-queue.md`. That queue is the only thing standing between this
project's quality numbers and being real, and nothing in it can be delegated to a model: the
whole point is that somebody hears the audio.

What *can* be removed is the friction around the listening. Done by hand the job is reading
a 44-row markdown table, scrubbing an audio file to each timestamp, and hand-editing a
reference `.txt` for every correction. This module turns it into one span per screen with
the audio already playing.

**It reads the structured outputs, not the markdown.** `agents/outputs/adjudicate-<clip>.json`
carries every span with its candidates, its chosen reading, its reason and its confidence;
`critique-references.json` carries the critic's findings. The queue table is those two lists
concatenated, and roughly a quarter of its rows are a critic finding sitting on top of the
adjudicator span it disputes, so merging them turns 44 rows into about 35 stops. Re-parsing
the table this module could regenerate would be the wrong direction entirely.

**Anchoring is a substring match, deliberately.** A span's `chosen` text is a quotation from
one reference line, so a correction is `line.replace(chosen, corrected, 1)` and nothing else
in the file can move. The span's window fixes which line, which matters for the one reading
(`Dakle,`) that occurs three times. Two spans are not quotations at all: one elides its
middle with `...` and one records an *omission* rather than a reading. They anchor to
nothing, and are offered as a whole-line edit instead of being quietly dropped.

**Verification is per clip, and it is not a claim of perfection.** `human_verified` lives in
each clip's own `meta.json` and `report._verified` keys off the clip, so a clean clip becomes
real without waiting for a poor one. What the flip means is that every *flagged* span was
checked; an error every engine made identically was never flagged and was never put in front
of the reviewer. `VERIFIED_NOTE` says exactly that, in the artifact, because the artifact is
what someone finds later.

Nothing here computes or reports a score (non-negotiable 9), and the one ffmpeg invocation
goes through `media.play_span_cmd` (non-negotiable 1).
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from subtitler import media
from subtitler.bench.agents import REVIEW_QUEUE, WINDOW_S, review_table, timestamp

__all__ = [
    "CONVENTIONS",
    "CRITIQUE_OUTPUT",
    "DECISIONS",
    "LEAD_IN_S",
    "TAIL_S",
    "VERIFIED_NOTE",
    "Anchor",
    "Candidate",
    "Convention",
    "Critic",
    "Decision",
    "Span",
    "Summary",
    "anchor",
    "apply_convention",
    "apply_text",
    "clip_paths",
    "load_decisions",
    "load_queue",
    "play_cmd",
    "resolved_keys",
    "review",
    "verify_clip",
    "write_decisions",
]

# Alongside the queue it resolves, because it is the same artifact at a different stage and
# somebody reading one will want the other.
DECISIONS = "review-decisions.json"

# The critic's response, named by `agents.plan` as `agents/outputs/<task_id>.json`.
CRITIQUE_OUTPUT = "critique-references.json"

# A span that starts on the disputed word is hard to judge: the ear needs the run-up to it,
# and the tail confirms the word did not continue. Chosen to be short enough that 35 spans
# stay a twenty-minute job.
LEAD_IN_S = 1.5
TAIL_S = 0.5


@dataclass(frozen=True, slots=True)
class Candidate:
    """One engine's reading of a span, named as the reviewer will see it."""

    source: str
    text: str


@dataclass(frozen=True, slots=True)
class Critic:
    """The adversarial pass's objection to a span, when it raised one."""

    issue: str
    severity: str
    recommendation: str


@dataclass(frozen=True, slots=True)
class Span:
    """One stop: what the engines said, what the adjudicator chose, and where it landed.

    `line` is the reference line the chosen reading was written into, derived from the
    window index rather than searched for, because it is what disambiguates a reading that
    occurs in more than one window.
    """

    clip: str
    start: float
    end: float
    chosen: str
    candidates: tuple[Candidate, ...] = ()
    reason: str = ""
    confidence: str = ""
    critic: Critic | None = None
    line: int = -1

    @property
    def key(self) -> tuple[str, str, str]:
        """Stable across regenerations of the queue, and readable in the decisions file.

        Not the timestamp alone: two spans in `gozba-sample` share the window 60.0-75.0, and
        the pair is only told apart by what was chosen.
        """
        return (self.clip, f"{self.start:.1f}", self.chosen)

    @property
    def flagged_by(self) -> str:
        if self.critic is None:
            return f"adjudicator ({self.confidence})"
        return f"adjudicator ({self.confidence}) + critic: {self.critic.issue} ({self.critic.severity})"


@dataclass(frozen=True, slots=True)
class Anchor:
    """Where a span's chosen reading sits: one line, one substring of it."""

    line: int
    text: str


@dataclass(frozen=True, slots=True)
class Decision:
    """What the reviewer said about one span. `skip` leaves it unresolved on purpose."""

    clip: str
    start: float
    chosen: str
    verdict: str
    text: str = ""

    @property
    def key(self) -> tuple[str, str, str]:
        return (self.clip, f"{self.start:.1f}", self.chosen)

    def to_dict(self) -> dict[str, Any]:
        return {
            "clip": self.clip,
            "start": self.start,
            "chosen": self.chosen,
            "verdict": self.verdict,
            "text": self.text,
        }


@dataclass(frozen=True, slots=True)
class Convention:
    """A decision that applies to a whole clip rather than to one span.

    A spelling convention cannot be settled span by span: taking it one occurrence at a time
    invites answering it differently in two places, and each answer moves WER on every other
    occurrence. So it is asked once, before the spans, as a set of concrete edits.
    """

    clip: str
    label: str
    why: str
    edits: tuple[tuple[str, str], ...]


# The only convention this project has needed so far, and the critic named it the single
# highest-leverage decision in the whole reference: the choice moves WER on every occurrence
# of the name and no adjudication can settle a spelling. `gozba-sample.txt` is currently
# inconsistent with itself, Serbianized in the adjectives (`Lokove`, `Lokovoj`) and English
# in the noun. Written out as three case-inflected edits rather than one find-and-replace,
# because `Locke` -> `Lok` would produce `Johna Lok` and a reviewer confirming a table can
# see that; a regex cannot.
CONVENTIONS: tuple[Convention, ...] = (
    Convention(
        clip="gozba-sample",
        label="Serbianize the name `Locke`",
        why=(
            "The reference already writes the adjectives Serbianized (`Lokove filozofije`, "
            "`Lokovoj filozofiji`) while the engines wrote the noun in English. One "
            "convention, applied to every occurrence."
        ),
        edits=(
            ("o filozofiji Johna Locke", "o filozofiji Džona Loka"),
            ("zašto je uopšte John Locke značajan", "zašto je uopšte Džon Lok značajan"),
            ("da je Locke začetnik", "da je Lok začetnik"),
        ),
    ),
)

# Replaces `agents.CAVEAT` in a verified clip's meta.json. The second sentence is the point:
# flipping the flag does not make the reference correct, it makes the *flagged* spans
# checked, and the class of error the adjudication is blind to was never shown to anybody.
VERIFIED_NOTE = (
    "Human-verified against the audio: every span the adjudicator flagged and every finding "
    "its critic raised was heard and settled by a speaker of the language. What that does "
    "not cover is the text nobody flagged: an error every engine made the same way is "
    "invisible to a consensus adjudication and was never put in front of the reviewer. WER "
    "against this reference is a real number and is not a guarantee of a perfect transcript."
)


# --------------------------------------------------------------------------- loading


def _window_lines(windows: Sequence[Mapping[str, Any]]) -> dict[int, int]:
    """Window index to reference line index.

    Not the identity: `agents.reference_text` drops windows that came back empty, so a clip
    where the engines heard nothing for fifteen seconds has fewer lines than windows and
    every window after the gap shifts up.
    """
    mapping: dict[int, int] = {}
    line = 0
    for window in sorted(windows, key=lambda w: int(w.get("index", 0))):
        if str(window.get("text", "")).strip():
            mapping[int(window["index"])] = line
            line += 1
    return mapping


def _line_for(start: float, lines: Mapping[int, int]) -> int:
    """The reference line a timestamp falls on, or -1 when its window was empty."""
    return lines.get(int(start // WINDOW_S), -1)


def load_queue(run_dir: Path, *, clip: str | None = None) -> list[Span]:
    """Every span needing a human, merged, in the order they are heard.

    A critic finding on the same clip and window as an adjudicator span is attached to that
    span rather than listed after it: they are one stop, and the critic's objection is the
    most useful thing to read while deciding. A finding with no such span stands alone.
    """
    outputs = run_dir / "agents" / "outputs"
    spans: list[Span] = []
    lines_by_clip: dict[str, dict[int, int]] = {}

    for path in sorted(outputs.glob("adjudicate-*.json")):
        data = _read_json(path)
        clip_id = str(data.get("clip") or path.stem.removeprefix("adjudicate-"))
        lines_by_clip[clip_id] = _window_lines(data.get("windows") or [])
        for raw in data.get("spans") or []:
            start = float(raw.get("start", 0.0))
            spans.append(
                Span(
                    clip=clip_id,
                    start=start,
                    end=float(raw.get("end", start)),
                    chosen=str(raw.get("chosen", "")),
                    candidates=tuple(
                        Candidate(str(c.get("source", "")), str(c.get("text", "")))
                        for c in raw.get("candidates") or []
                    ),
                    reason=str(raw.get("reason", "")),
                    confidence=str(raw.get("confidence", "")),
                    line=_line_for(start, lines_by_clip[clip_id]),
                )
            )

    critique = _read_json(outputs / CRITIQUE_OUTPUT)
    for finding in critique.get("findings") or []:
        clip_id = str(finding.get("clip", ""))
        start = float(finding.get("start", 0.0))
        critic = Critic(
            issue=str(finding.get("issue", "other")),
            severity=str(finding.get("severity", "")),
            recommendation=str(finding.get("recommendation", "")),
        )
        match = next(
            (
                i
                for i, span in enumerate(spans)
                if span.clip == clip_id
                and span.critic is None
                and int(span.start // WINDOW_S) == int(start // WINDOW_S)
                and _overlaps(span, start, float(finding.get("end", start)))
            ),
            None,
        )
        if match is not None:
            spans[match] = replace(spans[match], critic=critic)
            continue
        spans.append(
            Span(
                clip=clip_id,
                start=start,
                end=float(finding.get("end", start)),
                chosen=str(finding.get("reference_text", "")),
                candidates=tuple(
                    Candidate(*_split_candidate(str(c))) for c in finding.get("candidates") or []
                ),
                reason=str(finding.get("recommendation", "")),
                critic=critic,
                line=_line_for(start, lines_by_clip.get(clip_id, {})),
            )
        )

    spans.sort(key=lambda s: (s.clip, s.start, s.chosen))
    return [s for s in spans if clip is None or s.clip == clip]


def _overlaps(span: Span, start: float, end: float) -> bool:
    """Two timestamps describing the same stretch of audio, allowing for rounding."""
    return span.start < end + 0.5 and start < span.end + 0.5


def _split_candidate(text: str) -> tuple[str, str]:
    """`"groq: some reading"` to `("groq", "some reading")`.

    The critic's candidates are prose, not the adjudicator's structured pairs. Splitting on
    the first colon recovers the engine name; anything else is shown whole under no name
    rather than mangled.
    """
    source, sep, reading = text.partition(":")
    if sep and " " not in source.strip():
        return source.strip(), reading.strip()
    return "", text.strip()


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def clip_paths(run_dir: Path, *, root: Path | None = None) -> dict[str, Path]:
    """Clip id to the media file, read from the run's own record of what it transcribed.

    The path in `results.json` is relative to the repository, which is what makes a
    committed run reusable in someone else's checkout.
    """
    payload = _read_json(run_dir / "results.json")
    base = root or Path.cwd()
    found: dict[str, Path] = {}
    for record in payload.get("results") or []:
        clip_id = str(record.get("clip_id", ""))
        clip = str(record.get("clip", ""))
        if clip_id and clip and clip_id not in found:
            found[clip_id] = base / clip
    return found


# --------------------------------------------------------------------------- anchoring


def anchor(span: Span, lines: Sequence[str], *, text: str | None = None) -> Anchor | None:
    """The one line a span's reading sits on, or `None` if it is not a quotation.

    Tries the span's own line first, which is what tells the three occurrences of `Dakle,`
    apart. Falling back to a whole-file search covers a span whose reading was quoted across
    a window boundary, which really happens: `daš ga jednom državnom službenom licu` is
    timestamped in window 7 and written on line 8. `None` is a real answer, not a failure:
    one span in `gozba-sample` elides its middle with `...` and one in `uvod-u-pravo` records
    an omission rather than a reading, and neither can be replaced by substring.

    `text` overrides what to look for, because a clip-wide convention may have rewritten the
    line since the adjudicator quoted it. See `current_text`.
    """
    needle = (text if text is not None else span.chosen).strip()
    if not needle:
        return None
    if 0 <= span.line < len(lines) and needle in lines[span.line]:
        return Anchor(line=span.line, text=needle)
    hits = [i for i, line in enumerate(lines) if needle in line]
    return Anchor(line=hits[0], text=needle) if len(hits) == 1 else None


def apply_text(lines: Sequence[str], at: Anchor, new_text: str) -> list[str]:
    """Replace the anchored reading, on its line only, once."""
    out = list(lines)
    out[at.line] = out[at.line].replace(at.text, new_text, 1)
    return out


def apply_convention(lines: Sequence[str], convention: Convention) -> tuple[list[str], list[str]]:
    """`(lines, applied)`. An edit whose target is absent is skipped, not an error.

    Absent usually means already applied, which is what a second run of an interrupted
    review looks like.
    """
    out = list(lines)
    applied: list[str] = []
    for current, replacement in convention.edits:
        for i, line in enumerate(out):
            if current in line:
                out[i] = line.replace(current, replacement)
                applied.append(f"{current} -> {replacement}")
                break
    return out, applied


def current_text(span: Span, conventions: Sequence[Convention]) -> str:
    """What the reference says for this span *now*, after the conventions already applied.

    A convention and a span can quote the same words: `da je Locke začetnik` is both an edit
    in the `Locke` convention and the head of a flagged span. Applying the convention first
    would leave that span unable to find itself in the file, so the adjudicator's `chosen`
    stays the span's identity and this is what gets matched and displayed.
    """
    text = span.chosen
    for convention in conventions:
        if convention.clip != span.clip:
            continue
        for old, new in convention.edits:
            text = text.replace(old, new)
    return text


# --------------------------------------------------------------------------- decisions


def load_decisions(references: Path) -> list[Decision]:
    data = _read_json(references / DECISIONS)
    out = []
    for raw in data.get("decisions") or []:
        out.append(
            Decision(
                clip=str(raw.get("clip", "")),
                start=float(raw.get("start", 0.0)),
                chosen=str(raw.get("chosen", "")),
                verdict=str(raw.get("verdict", "")),
                text=str(raw.get("text", "")),
            )
        )
    return out


def write_decisions(
    references: Path,
    decisions: Sequence[Decision],
    *,
    conventions: Sequence[str] = (),
    run: str = "",
    now: str | None = None,
) -> Path:
    """The record of what was settled. Written by `review`'s `save`, after the text it describes."""
    path = references / DECISIONS
    payload = {
        "run": run,
        "updated_utc": now or datetime.now(UTC).isoformat(),
        "conventions": list(conventions),
        "decisions": [d.to_dict() for d in decisions],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def resolved_keys(decisions: Sequence[Decision]) -> set[tuple[str, str, str]]:
    """Spans that are settled. A skip is recorded and does not count as settled."""
    return {d.key for d in decisions if d.verdict in ("accept", "replace")}


# --------------------------------------------------------------------------- playback


def play_cmd(
    clip: Path, span: Span, *, lead_in: float = LEAD_IN_S, tail: float = TAIL_S
) -> list[str]:
    """The span with its run-up, built by `media` because ffplay is ffmpeg."""
    return media.play_span_cmd(
        clip, start=max(0.0, span.start - lead_in), end=max(span.end + tail, span.start + tail)
    )


# --------------------------------------------------------------------------- the session


@dataclass(frozen=True, slots=True)
class Summary:
    """What one sitting achieved, and what is left."""

    answered: int
    skipped: int
    remaining: int
    verified: tuple[str, ...]
    conventions: tuple[str, ...]
    written: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "answered": self.answered,
            "skipped": self.skipped,
            "remaining": self.remaining,
            "verified": list(self.verified),
            "conventions": list(self.conventions),
            "written": list(self.written),
        }


PROMPT = "[enter] accept  [1-9] pick  [e] edit  [r] replay  [s] skip  [q] save+quit\n> "


def _render(span: Span, position: int, total: int, at: Anchor | None, text: str) -> list[str]:
    lines = [
        "",
        "─" * 78,
        f"{span.clip}  [{position}/{total}]  "
        f"{timestamp(span.start)}-{timestamp(span.end)}   {span.flagged_by}",
        "",
    ]
    for i, candidate in enumerate(span.candidates, start=1):
        label = candidate.source or "candidate"
        lines.append(f"  {i}  {label:<16}{candidate.text}")
    if not span.candidates:
        lines.append("  (no candidate readings recorded)")
    lines += ["", f"  reference -> {text}"]
    if text != span.chosen:
        lines.append(f"  (a convention changed this from: {span.chosen})")
    if span.reason:
        lines.append(f"  why: {span.reason}")
    if span.critic is not None and span.critic.recommendation != span.reason:
        lines.append(f"  critic: {span.critic.recommendation}")
    if at is None:
        lines.append(
            "  NOTE: this reading is not one quotable stretch of a reference line, so "
            "accepting it changes nothing. [e] retypes the whole line instead."
        )
    lines.append("")
    return lines


def _declined(label: str) -> str:
    """Recorded so a convention that was turned down is not offered again every session."""
    return f"declined: {label}"


def review(
    run_dir: Path,
    *,
    references: Path,
    clip: str | None = None,
    root: Path | None = None,
    dry_run: bool = False,
    verified_by: str = "",
    ask: Callable[[str], str] = input,
    play: Callable[[Sequence[str]], None] | None = None,
    log: Callable[[str], None] = print,
    now: str | None = None,
) -> Summary:
    """Walk the queue with the audio, and write what the reviewer decides.

    Injected `ask` and `play` for the same reason `gui/session.py` exists: the loop's
    decisions are the part worth testing and neither a sound card nor a terminal should be
    required to test them.
    """
    spans = load_queue(run_dir, clip=clip)
    if not spans:
        log(f"nothing flagged in {run_dir}")
        return Summary(0, 0, 0, (), (), ())

    decisions = load_decisions(references)
    done = resolved_keys(decisions)
    media_by_clip = clip_paths(run_dir, root=root)
    player = play if play is not None else _play
    # Carried across the whole session and written on every save. Reloading it from disk per
    # span, or omitting it from the per-answer write, would drop the record of a convention
    # the first time a span was answered after it.
    convention_log = list(_prior_conventions(references))
    started_with = list(convention_log)
    applied_conventions = [c for c in CONVENTIONS if c.label in convention_log]
    written: set[str] = set()
    answered = skipped = 0
    todo = [s for s in spans if s.key not in done]

    log(
        f"{len(spans)} span(s) flagged in {run_dir.name}; {len(spans) - len(todo)} already "
        f"settled, {len(todo)} to go."
    )

    lines_by_clip = {c: _read_lines(references / f"{c}.txt") for c in {s.clip for s in spans}}
    quit_early = False

    def save() -> None:
        """Everything one answer changed, flushed together: the text first, then the record.

        A deliberate `[q]` is not the only way a session ends. A killed process, a closed
        laptop and a Ctrl-C all stop the loop wherever it stands, so anything held in memory
        until the end is not "resumable", it is lost. Writing the decisions per answer and the
        reference text only at the end was exactly that trap: the next session reads the
        answer, skips the span, never re-offers the convention, and the correction the
        reviewer actually made is gone with nothing to say it ever existed.

        Order matters between the two writes. Text first means a process killed between them
        leaves a correction on disk that no decision claims, which the next session re-offers
        (its `chosen` no longer matches the line, so it arrives as a whole-line edit and says
        so). Decisions first would leave a claim with no correction, which is the silent
        loss. Rewriting every clip this session touched, rather than only the last one, is
        idempotent and costs a few kilobytes, and it repairs a flush that was cut short.
        """
        if dry_run:
            return
        for clip_id in sorted(written):
            _write_lines(references / f"{clip_id}.txt", lines_by_clip[clip_id])
        write_decisions(
            references, decisions, conventions=convention_log, run=run_dir.name, now=now
        )

    for clip_id in sorted({s.clip for s in todo}):
        if quit_early:
            break
        for convention in CONVENTIONS:
            if convention.clip != clip_id:
                continue
            if convention.label in convention_log or _declined(convention.label) in convention_log:
                continue
            # Asked only when it would actually change this clip's reference. A convention is
            # declared here as a constant, against one particular committed reference, so any
            # other run of the same clip id -- a fixture, a re-transcription, a reference
            # somebody already fixed by hand -- may not contain a word of it. Offering it
            # anyway would be a question with no consequence whose only effect is to teach the
            # reviewer that the questions can be answered without reading them. The preview is
            # the real edit, so what is shown and what is applied cannot disagree.
            rewritten, pending = apply_convention(lines_by_clip[clip_id], convention)
            if not pending:
                continue
            log("")
            log(f"{clip_id}: {convention.label}")
            log(f"  {convention.why}")
            for edit in pending:
                log(f"    {edit}")
            answer = ask("apply this convention to the whole clip? [Y/n] ").strip().lower()
            if answer not in ("", "y", "yes"):
                convention_log.append(_declined(convention.label))
                log("  declined; it will not be offered again")
                save()
                continue
            lines_by_clip[clip_id] = rewritten
            convention_log.append(convention.label)
            applied_conventions.append(convention)
            written.add(clip_id)
            log(f"  applied {len(pending)} edit(s)")
            save()

        clip_todo = [s for s in todo if s.clip == clip_id]
        media_path = media_by_clip.get(clip_id)
        if media_path is None or not media_path.exists():
            log(f"{clip_id}: no media file found, so nothing will play. Reading only.")
            media_path = None

        for i, span in enumerate(clip_todo, start=1):
            reading = current_text(span, applied_conventions)
            at = anchor(span, lines_by_clip[clip_id], text=reading)
            for line in _render(span, i, len(clip_todo), at, reading):
                log(line)
            if media_path is not None:
                player(play_cmd(media_path, span))

            while True:
                answer = ask(PROMPT).strip()
                lowered = answer.lower()

                if lowered == "r":
                    if media_path is not None:
                        player(play_cmd(media_path, span))
                    continue
                if lowered == "q":
                    quit_early = True
                    break
                if lowered == "s":
                    decisions = _record(
                        decisions, Decision(span.clip, span.start, span.chosen, "skip")
                    )
                    skipped += 1
                    break

                text: str
                if lowered == "":
                    text = reading
                elif lowered == "e":
                    typed = ask("corrected text> ").strip()
                    if not typed:
                        log("  nothing typed; asking again")
                        continue
                    text = typed
                elif answer.isdigit() and 1 <= int(answer) <= len(span.candidates):
                    text = span.candidates[int(answer) - 1].text
                else:
                    log("  not one of the choices")
                    continue

                verdict = "accept" if text == reading else "replace"
                if verdict == "replace":
                    if at is None:
                        lines_by_clip[clip_id] = _replace_line(
                            lines_by_clip[clip_id], span.line, text
                        )
                    else:
                        lines_by_clip[clip_id] = apply_text(lines_by_clip[clip_id], at, text)
                    written.add(clip_id)
                    log(f"  -> {text}")
                decisions = _record(
                    decisions, Decision(span.clip, span.start, span.chosen, verdict, text)
                )
                answered += 1
                break

            save()
            if quit_early:
                break

    done = resolved_keys(decisions)
    remaining = [s for s in spans if s.key not in done]
    # A clip is verifiable when nothing it had flagged is still open. Per clip rather than
    # per run because that is where the flag lives and where the report reads it: the clean
    # clip should not wait on the poor one.
    verified = [
        c for c in sorted({s.clip for s in spans}) if not any(s.clip == c for s in remaining)
    ]
    fresh = tuple(c for c in convention_log if c not in started_with)

    if dry_run:
        log("")
        log("--dry-run: nothing written")
        return Summary(answered, skipped, len(remaining), tuple(verified), fresh, ())

    # The text and the decisions are already on disk from the last answer; this is the same
    # write again, for the session that answered nothing and for the one that quit. What is
    # left below is derived state -- the flag and the queue -- which a later run recomputes
    # from the decisions, and which therefore must not be reached by an abandoned session.
    save()
    for clip_id in verified:
        if not (references / f"{clip_id}.meta.json").exists():
            log(f"{clip_id}: no meta.json to mark verified; skipping the flag")
            continue
        verify_clip(
            references,
            clip_id,
            spans=len([s for s in spans if s.clip == clip_id]),
            by=verified_by,
            now=now,
        )
    _rewrite_queue(run_dir, references, remaining, verified, run=run_dir.name)

    log("")
    log(
        f"answered {answered}, skipped {skipped}, {len(remaining)} still open."
        + (f" human_verified: {', '.join(verified)}." if verified else "")
    )
    if verified:
        log(f"now: subtitler bench report {run_dir}")

    return Summary(
        answered, skipped, len(remaining), tuple(verified), fresh, tuple(sorted(written))
    )


def _record(decisions: Sequence[Decision], decision: Decision) -> list[Decision]:
    """Last answer for a span wins, so re-reviewing a span replaces rather than appends."""
    return [d for d in decisions if d.key != decision.key] + [decision]


def _prior_conventions(references: Path) -> list[str]:
    return [str(c) for c in _read_json(references / DECISIONS).get("conventions") or []]


def _replace_line(lines: Sequence[str], index: int, text: str) -> list[str]:
    out = list(lines)
    if 0 <= index < len(out):
        out[index] = text
    return out


def _read_lines(path: Path) -> list[str]:
    if not path.exists():
        return []
    return path.read_text(encoding="utf-8").rstrip("\n").split("\n")


def _write_lines(path: Path, lines: Sequence[str]) -> None:
    path.write_text("\n".join(lines).rstrip("\n") + "\n", encoding="utf-8")


def _play(cmd: Sequence[str]) -> None:
    """Best effort. A reviewer with no ffplay still gets the text and the timestamps."""
    import subprocess

    try:
        subprocess.run(list(cmd), check=False, capture_output=True)
    except (OSError, ValueError):
        return


# --------------------------------------------------------------------------- finishing


def verify_clip(
    references: Path,
    clip_id: str,
    *,
    spans: int,
    by: str = "",
    now: str | None = None,
) -> Path:
    """Flip `human_verified` for one clip, and say what the flip does and does not mean."""
    path = references / f"{clip_id}.meta.json"
    meta = _read_json(path)
    if not meta:
        raise ValueError(f"{path} does not exist; there is no reference to verify")
    meta["human_verified"] = True
    meta["verified_utc"] = now or datetime.now(UTC).isoformat()
    meta["verified_spans"] = spans
    if by:
        meta["verified_by"] = by
    meta["note"] = VERIFIED_NOTE
    path.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def _rewrite_queue(
    run_dir: Path,
    references: Path,
    remaining: Sequence[Span],
    verified: Sequence[str],
    *,
    run: str = "",
) -> Path:
    """The queue, minus what has been settled, rendered by the same code that wrote it.

    Rebuilding the adjudicator's shape rather than editing the markdown keeps one renderer:
    the next `bench agents --merge` and this module cannot drift apart in how a row reads.
    """
    adjudications: dict[str, dict[str, Any]] = {}
    findings: list[dict[str, Any]] = []
    for span in remaining:
        if span.critic is not None:
            findings.append(
                {
                    "clip": span.clip,
                    "start": span.start,
                    "end": span.end,
                    "issue": span.critic.issue,
                    "severity": span.critic.severity,
                    "reference_text": span.chosen,
                    "recommendation": span.critic.recommendation,
                    "candidates": [f"{c.source}: {c.text}" for c in span.candidates],
                }
            )
        adjudications.setdefault(span.clip, {"spans": []})["spans"].append(
            {
                "start": span.start,
                "end": span.end,
                "chosen": span.chosen,
                "reason": span.reason,
                "confidence": span.confidence,
                "candidates": [{"source": c.source, "text": c.text} for c in span.candidates],
            }
        )

    body = review_table(adjudications, {"findings": findings}, run=run)
    if verified:
        note = (
            "\n> **Verified against the audio: "
            + ", ".join(f"`{c}`" for c in sorted(verified))
            + ".** Those clips carry `human_verified: true` and their WER is no longer marked "
            "provisional. What that covers is every span listed here; see each "
            "`meta.json` for what it does not.\n"
        )
        body = body.replace("\n| clip | time |", note + "\n| clip | time |", 1)
    path = references / REVIEW_QUEUE
    path.write_text(body, encoding="utf-8")
    return path
