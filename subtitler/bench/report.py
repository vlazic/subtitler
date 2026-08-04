"""results.json in, report.md out. Pure: a dict goes in, a string comes out.

No I/O, no jiwer, no clock, no model. That is what makes the report testable on a runner
that has never seen the `bench` extra, and it is what guarantees the document can never
disagree with the JSON it was rendered from.

The report's first job is not the leaderboard, it is the caveats above it. A benchmark whose
reference is a machine transcript measures agreement between models rather than correctness,
and a table of four-decimal numbers is very good at hiding that. So the "What this run
cannot answer" section is rendered first, from facts in the payload, and every provisional
number carries a marker in its own cell rather than in a footnote nobody reads.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

__all__ = ["leaderboard_rows", "render"]

# Attached to any WER derived from a reference nobody has verified.
PROVISIONAL = "*"


def _pct(value: float | None, digits: int = 1) -> str:
    return "n/a" if value is None else f"{100 * value:.{digits}f}"


def _num(value: float | None, digits: int = 2) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def _table(header: Sequence[str], rows: Sequence[Sequence[str]]) -> list[str]:
    if not rows:
        return ["_(nothing to show)_"]
    return [
        "| " + " | ".join(header) + " |",
        "|" + "|".join("---" for _ in header) + "|",
        *("| " + " | ".join(row) + " |" for row in rows),
    ]


def _verified(payload: dict[str, Any], clip_id: str) -> bool:
    meta = payload.get("references", {}).get(clip_id) or {}
    return bool(meta.get("human_verified"))


def leaderboard_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Successful cells, best first.

    Ranked by WER when a reference exists, and by RTF when none does. The fallback is
    stated in the report rather than silently substituted: speed is not quality, and a table
    sorted by speed under a heading that says "leaderboard" would imply it is.
    """
    rows = [r for r in payload.get("results", []) if r.get("ok")]
    scored = [r for r in rows if (r.get("reference_score") or {}).get("wer") is not None]
    if scored:
        return sorted(rows, key=lambda r: (r.get("reference_score") or {}).get("wer", float("inf")))
    return sorted(rows, key=lambda r: r.get("rtf") or float("inf"))


def render(payload: dict[str, Any]) -> str:
    results = payload.get("results", [])
    ok = [r for r in results if r.get("ok")]
    failed = [r for r in results if not r.get("ok")]
    any_reference = any((r.get("reference_score") or {}).get("wer") is not None for r in ok)

    lines: list[str] = ["# Benchmark run", ""]
    lines += _provenance(payload)
    lines += _caveats(payload, ok, failed, any_reference)
    lines += _leaderboard(payload, any_reference)
    lines += _quality(payload, ok)
    lines += _fix_axis(ok)
    lines += _failures(failed)
    lines += _normalization_note()
    lines += _environment(payload)
    return "\n".join(lines).rstrip() + "\n"


def _provenance(payload: dict[str, Any]) -> list[str]:
    git = payload.get("git", {}) or {}
    config = payload.get("config", {}) or {}
    dirty = git.get("dirty")
    sha = git.get("sha") or "unknown"
    state = "clean" if dirty is False else ("DIRTY, not reproducible" if dirty else "unknown")
    lines = [
        f"- created: `{payload.get('created_utc', '')}`",
        f"- commit: `{sha}` on `{git.get('branch') or '?'}` ({state})",
        f"- clips: {', '.join(config.get('clips', [])) or 'none'}",
        f"- denoisers: {', '.join(config.get('denoisers', [])) or 'none'}",
        f"- engines: {', '.join(config.get('engines', [])) or 'none'}"
        f" ({config.get('model', '?')}, device {config.get('device', '?')}"
        + (f", batch {config['batch_size']}" if config.get("batch_size") else "")
        + ")",
    ]
    if payload.get("rescored_utc"):
        lines.append(f"- metrics recomputed from the kept transcripts: `{payload['rescored_utc']}`")
    return [*lines, ""]


def _caveats(
    payload: dict[str, Any],
    ok: Sequence[dict[str, Any]],
    failed: Sequence[dict[str, Any]],
    any_reference: bool,
) -> list[str]:
    lines = ["## What this run cannot answer", ""]

    references = payload.get("references", {}) or {}
    absent = sorted(cid for cid, meta in references.items() if meta.get("status") != "present")
    unverified = sorted(
        cid
        for cid, meta in references.items()
        if meta.get("status") == "present" and not meta.get("human_verified")
    )

    if absent:
        lines += [
            f"- **No reference transcript for {', '.join(absent)}.** WER, CER and the error "
            "decomposition are therefore not reported for those clips: this run measures "
            "shape, speed and hallucination signals only. Phase 8 (LLM adjudication of "
            "reference transcripts) is what fills that gap; nothing here invents one.",
        ]
    if unverified:
        lines += [
            f"- **The reference for {', '.join(unverified)} is not human-verified.** Every WER "
            f"derived from it is marked `{PROVISIONAL}` and is provisional: an unverified "
            "reference measures agreement between models, not correctness.",
        ]
    if not any_reference:
        lines += [
            "- **The leaderboard below is ordered by realtime factor, not by quality.** "
            "Speed is not accuracy. Nothing in this run ranks transcription quality.",
        ]

    cloud = [r for r in failed if "groq" in str(r.get("engine_requested", ""))]
    if cloud:
        reasons = sorted({str(r.get("error", "")).strip() for r in cloud})
        lines += [
            "- **The cloud baseline did not run.** "
            + "; ".join(reasons)
            + ". PRD acceptance criterion 4 (does local `large-v3` beat "
            "`groq/whisper-large-v3-turbo` on Serbian) is therefore **unanswered** by this "
            "run, and would still be unanswered with a working key while no reference exists.",
        ]
    elif not any(str(r.get("engine_requested", "")).startswith("groq") for r in [*ok, *failed]):
        lines += [
            "- **No cloud engine was in the matrix**, so this run says nothing about "
            "PRD acceptance criterion 4 (local versus `groq/whisper-large-v3-turbo`).",
        ]

    if not any(r.get("fix") for r in ok):
        lines += [
            "- **The `--fix` axis was not run**, so PRD open question 4 (does the correction "
            "pass improve WER or hurt it) is untouched here.",
        ]
    return [*lines, ""]


def _leaderboard(payload: dict[str, Any], any_reference: bool) -> list[str]:
    rows = leaderboard_rows(payload)
    heading = "## Leaderboard" + ("" if any_reference else " (by speed: no reference exists)")
    header = [
        "#",
        "clip",
        "denoise",
        "engine",
        "fix",
        "WER %",
        "WER folded %",
        "CER %",
        "sub/ins/del",
        "RTF",
        "wall s",
        "peak MB",
    ]
    table = []
    for i, r in enumerate(rows, start=1):
        score = r.get("reference_score") or {}
        mark = "" if _verified(payload, r.get("clip_id", "")) else PROVISIONAL
        wer = _pct(score.get("wer")) + (mark if score.get("wer") is not None else "")
        folded = _pct(score.get("wer_folded")) + (
            mark if score.get("wer_folded") is not None else ""
        )
        table.append(
            [
                str(i),
                r.get("clip_id", ""),
                r.get("denoise", ""),
                r.get("engine", r.get("engine_requested", "")),
                "yes" if r.get("fix") else "no",
                wer,
                folded,
                _pct(score.get("cer")),
                (
                    f"{score.get('substitutions')}/{score.get('insertions')}/"
                    f"{score.get('deletions')}"
                    if score
                    else "n/a"
                ),
                _num(r.get("rtf"), 3),
                _num(r.get("wall_s"), 1),
                _num(r.get("peak_rss_mb"), 0),
            ]
        )
    return [heading, "", *_table(header, table), ""]


def _quality(payload: dict[str, Any], ok: Sequence[dict[str, Any]]) -> list[str]:
    """Cue shape and the hallucination heuristics. Available with or without a reference."""
    header = [
        "cell",
        "cues",
        "mean CPS",
        "p95 CPS",
        "max CPS",
        "over line %",
        "over dur %",
        "over CPS %",
        "under min dur %",
        "longest repeat",
        "prompt echo",
        "collapses",
        "silence dropped",
        "filler",
    ]
    rows = []
    for r in sorted(ok, key=lambda r: r.get("cell_id", "")):
        cue = r.get("cue_stats", {}) or {}
        hal = r.get("hallucination", {}) or {}
        filler = hal.get("filler_hits", {}) or {}
        repeat = hal.get("longest_repeat_n", 0)
        repeat_text = hal.get("longest_repeat_text", "")
        rows.append(
            [
                r.get("cell_id", ""),
                str(cue.get("count", 0)),
                _num(cue.get("mean_cps"), 1),
                _num(cue.get("p95_cps"), 1),
                _num(cue.get("max_cps"), 1),
                _num(cue.get("over_line_pct"), 1),
                _num(cue.get("over_dur_pct"), 1),
                _num(cue.get("over_cps_pct"), 1),
                _num(cue.get("under_min_dur_pct"), 1),
                f"{repeat} (`{repeat_text}`)" if repeat else "0",
                (
                    f"**{hal.get('prompt_echo_n')}** (`{hal.get('prompt_echo_text')}`)"
                    if hal.get("prompt_echo_n")
                    else "0"
                ),
                "n/a"
                if hal.get("repetition_collapsed") is None
                else str(hal["repetition_collapsed"]),
                "n/a" if hal.get("silence_dropped") is None else str(hal["silence_dropped"]),
                ", ".join(f"{k}x{v}" for k, v in sorted(filler.items())) or "none",
            ]
        )
    echoed = [
        r.get("cell_id", "") for r in ok if (r.get("hallucination", {}) or {}).get("prompt_echo_n")
    ]
    warning = (
        [
            f"**{len(echoed)} cell(s) echoed the Serbian steering prompt back as transcript "
            f"text**: {', '.join(sorted(echoed))}. That is decoder output standing where "
            "speech should be, so the affected transcript is missing whatever was said "
            "there. Worth reading before trusting any other number in that row.",
            "",
        ]
        if echoed
        else []
    )
    return ["## Cue shape and hallucination signals", "", *warning, *_table(header, rows), ""]


def _fix_axis(ok: Sequence[dict[str, Any]]) -> list[str]:
    fixed = [r for r in ok if r.get("fix")]
    if not fixed:
        return []
    rows = [
        [
            r.get("cell_id", ""),
            _pct(r.get("fix_change_rate")),
            str((r.get("fix_report") or {}).get("changed", "n/a")),
            _num(r.get("wall_s"), 1),
        ]
        for r in sorted(fixed, key=lambda r: r.get("cell_id", ""))
    ]
    return [
        "## The `--fix` axis",
        "",
        "`change %` is the word-level distance between the corrected cell and the identical "
        "uncorrected one. It measures **how much the model rewrote, not whether the rewrite "
        "was right**. Whether `--fix` improves WER or hurts it (PRD open question 4) needs a "
        "reference transcript to answer, and is answered in the leaderboard above only when "
        "one exists.",
        "",
        *_table(["cell", "change %", "cues changed", "wall s"], rows),
        "",
    ]


def _failures(failed: Sequence[dict[str, Any]]) -> list[str]:
    if not failed:
        return []
    rows = [
        [r.get("cell_id", ""), r.get("engine_requested", ""), f"`{r.get('error', '')}`"]
        for r in failed
    ]
    return [
        "## Cells that did not run",
        "",
        "Kept rather than dropped: a cell that could not run is a result about this machine "
        "or this account, and a matrix that silently omitted it would read as if it had "
        "never been asked for.",
        "",
        *_table(["cell", "engine", "error"], rows),
        "",
    ]


def _normalization_note() -> list[str]:
    return [
        "## How the text was normalized",
        "",
        "Applied identically to hypothesis and reference, in this order: NFC, Serbian "
        "Cyrillic to Latin via a hand-written table, lowercase, punctuation to spaces "
        "(including the Serbian quotes), whitespace collapsed. `WER folded` repeats the "
        "score with `č ć` folded to `c`, `đ` to `dj`, `š` to `s` and `ž` to `z`; the gap "
        "between the two columns separates hearing the wrong word from writing `c` for `č`.",
        "",
        "**Digits and abbreviations are deliberately not normalized in v1.** `20` scores as "
        "a substitution against `dvadeset`, and `npr.` against `na primer`. Both inflate "
        "every WER here, and both inflate it equally for every engine in the matrix, so the "
        "ranking survives while the absolute numbers are pessimistic.",
        "",
    ]


def _environment(payload: dict[str, Any]) -> list[str]:
    config = payload.get("config", {}) or {}
    cue = config.get("cues", {}) or {}
    return [
        "## Environment",
        "",
        "Full detail in `env.json` next to this file: `doctor --json`, the OS, CPU, RAM and "
        "GPU, and the version of every library that can change a transcript.",
        "",
        f"Cue limits in force: max {cue.get('max_line', '?')} chars per line, "
        f"{cue.get('max_lines', '?')} lines, {cue.get('min_dur', '?')}-{cue.get('max_dur', '?')}s, "
        f"{cue.get('max_cps', '?')} CPS.",
        "",
    ]
