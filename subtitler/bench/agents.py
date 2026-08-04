"""Adjudicated reference transcripts, and the honest name for what they are.

`benchmarks/references/<clip>.txt` is what turns the matrix from a shape-and-speed report
into a leaderboard. There is no human transcript of either fixture, so this module builds one
the only way the material allows: it takes the transcripts several engines already produced
for the same clip, aligns them by timestamp into a side-by-side view, and hands that to an
LLM to adjudicate into one best-guess verbatim Serbian text.

**What that reference is, exactly.** A *consensus pseudo-reference*. The adjudicator cannot
hear the audio. It works at the text level with Serbian language knowledge, which catches a
large class of real errors (`povenuli smo` is not a Serbian word and `pomenuli smo` is;
`državne gane` is not a phrase and `državne organe` is) and is blind to every error where all
engines agreed on the same wrong word. So a WER computed against it **ranks the engines
against each other; it does not establish truth**. That sentence is not confined to a commit
message: `meta.json` carries it, `report.py` prints it above the leaderboard, and every
number derived from it is marked provisional until a human flips `human_verified`.

**Why a second, adversarial agent.** The failure that would quietly invalidate the whole
comparison is not a wrong word, it is a *smoothed* one. An adjudicator that writes the
grammatically correct sentence instead of the one that was spoken produces a reference that
systematically penalises every raw engine and rewards the `--fix` pass, which is precisely
the axis PRD open question 4 asks about. `ref-critic` reads the reference against the same
aligned view looking for exactly that, plus spans where a real disagreement was papered over
with a confident single reading.

**Non-negotiable 9 is intact.** Agents produce *text*. Every number still comes out of
`metrics.py`, from that text, by arithmetic. Nothing here computes or reports a score.

**The mechanism is a manifest, not an API call.** This module has no model client in it. It
emits `agent-tasks.json` (role, id, prompt file, input file, output path) and later merges
the JSON that came back. That keeps the LLM outside the package: the agent definitions live
in `.claude/agents/`, so the same manifest can be driven by Claude Code, by a human with a
chat window, or by a script, and the merge is a pure function of files on disk either way.
Schemas are strict and validated on the way in. A malformed response is retried once and then
recorded `status: "failed"`, which leaves that clip without a reference: a run that reports no
WER is a worse result than a run that reports one, and both are better than a run that
reports one built out of a broken response.

The two stages are idempotent, so `--merge` can be run as often as it is useful:

1. `subtitler bench agents <run>` writes the manifest, the aligned inputs and one prompt per
   task. The critic is `blocked` until there is something to criticise.
2. `subtitler bench agents <run> --merge` validates whatever outputs exist, writes the
   references it can, generates the critic's input from them, and folds the critic's findings
   into the review queue when they arrive.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

__all__ = [
    "ADJUDICATOR",
    "CRITIC",
    "MAX_ATTEMPTS",
    "REVIEW_QUEUE",
    "WINDOW_S",
    "Source",
    "Window",
    "align",
    "merge",
    "plan",
    "reference_text",
    "render_alignment",
    "review_table",
    "select_sources",
    "validate_adjudication",
    "validate_critique",
    "write_plan",
]

ADJUDICATOR = "ref-adjudicator"
CRITIC = "ref-critic"

# One row of the side-by-side view. Long enough that a sentence is rarely cut in half and a
# disagreement has its context in the same row, short enough that the row stays readable and
# that a span's timestamp is precise enough to find in an audio player by hand.
WINDOW_S = 15.0

# Try once, retry once, then it is a failure and the clip goes without a reference.
MAX_ATTEMPTS = 2

REVIEW_QUEUE = "review-queue.md"

CONFIDENCES = ("low", "medium")
ISSUES = ("smoothed", "grammar_corrected", "other")
SEVERITIES = ("high", "medium", "low")

# Repeated verbatim into every meta.json this module writes. The caveat travels with the
# artifact, because the artifact is what someone will find in six months.
CAVEAT = (
    "Consensus pseudo-reference: adjudicated from engine transcripts by an LLM that cannot "
    "hear the audio. It catches errors that are not Serbian and is blind to any error every "
    "engine made the same way. WER against it ranks engines against each other and does not "
    "establish truth. Not human-verified: resolve the flagged spans in "
    f"benchmarks/references/{REVIEW_QUEUE} against the audio before trusting it."
)


@dataclass(frozen=True, slots=True)
class Source:
    """One engine transcript feeding an adjudication, named as the agent will see it."""

    cell_id: str
    label: str
    engine: str
    denoise: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class Window:
    """One timestamp-aligned row: what each source said between `start` and `end`."""

    index: int
    start: float
    end: float
    texts: dict[str, str] = field(default_factory=dict)


# --------------------------------------------------------------------------------------
# Choosing what feeds the reference
# --------------------------------------------------------------------------------------


def select_sources(payload: Mapping[str, Any], clip_id: str) -> list[Source]:
    """The cells whose transcripts are allowed to vote, one per engine.

    Three exclusions, each of which would otherwise corrupt the reference rather than enrich
    it:

    **Corrected cells never vote.** A `--fix` cell is already an LLM rewrite. Feeding it into
    the reference would make the reference agree with the correction pass by construction,
    and open question 4 asks whether that pass helps or hurts. Only `nofix` cells vote.

    **A cell that echoed the steering prompt never votes.** `metrics.prompt_echo` exists
    because `--denoise arnndn` put the tail of `SERBIAN_PROMPT` where the first fifty words of
    a lecture should be. That text is decoder output, not a reading of the audio.

    **One cell per engine.** Five denoiser variants of the same engine are five nearly
    identical opinions, and stacking them would let one engine outvote the others on a word
    none of them heard differently anyway. `none` is preferred when it ran, because it is the
    default path and the least processed audio.
    """
    candidates = [
        r
        for r in payload.get("results", [])
        if r.get("ok")
        and r.get("clip_id") == clip_id
        and not r.get("fix")
        and not (r.get("hallucination") or {}).get("prompt_echo_n")
        and r.get("cell_id")
    ]
    best: dict[str, dict[str, Any]] = {}
    for record in sorted(
        candidates,
        key=lambda r: (
            str(r.get("engine_requested", "")),
            0 if r.get("denoise") == "none" else 1,
            str(r.get("denoise", "")),
        ),
    ):
        engine = str(record.get("engine_requested", ""))
        best.setdefault(engine, record)
    return [
        Source(
            cell_id=str(r["cell_id"]),
            label=engine,
            engine=engine,
            denoise=str(r.get("denoise", "")),
        )
        for engine, r in sorted(best.items())
    ]


# --------------------------------------------------------------------------------------
# The side-by-side view
# --------------------------------------------------------------------------------------


def align(
    cues_by_label: Mapping[str, Sequence[Any]],
    *,
    window_s: float = WINDOW_S,
) -> list[Window]:
    """Bucket every source's cues into fixed windows, keyed on the cue's start time.

    Fixed windows rather than a pairwise alignment on purpose. Engines disagree about cue
    boundaries constantly (one emits `Misao lokove filozofije` and `ukratko izraženo` as two
    cues where another emits them as one), so any alignment that tries to match cue to cue
    spends its effort on a difference that does not exist in the audio. A window has none of
    that: whatever was said between 0:15 and 0:30 lands in the same row for every engine, and
    the adjudicator reads a row at a time.

    Empty windows are dropped. A gap in the middle of a clip is silence every engine agreed
    about, and a row of three blanks is noise in a document a model has to read carefully.
    """
    if window_s <= 0:
        raise ValueError("window_s must be positive")

    buckets: dict[int, dict[str, list[str]]] = {}
    for label, cues in cues_by_label.items():
        for cue in cues:
            index = int(max(0.0, float(cue.start)) // window_s)
            buckets.setdefault(index, {}).setdefault(label, []).append(cue.text.strip())

    windows = []
    for index in sorted(buckets):
        texts = {
            label: " ".join(part for part in parts if part)
            for label, parts in sorted(buckets[index].items())
        }
        texts = {label: text for label, text in texts.items() if text}
        if texts:
            windows.append(
                Window(index=index, start=index * window_s, end=(index + 1) * window_s, texts=texts)
            )
    return windows


def timestamp(seconds: float) -> str:
    """`mm:ss`, which is what an audio player's scrubber shows and a reviewer types."""
    total = round(seconds)
    return f"{total // 60:02d}:{total % 60:02d}"


def render_alignment(clip_id: str, sources: Sequence[Source], windows: Sequence[Window]) -> str:
    """The adjudicator's input document: one block per window, one line per engine."""
    lines = [
        f"# Aligned engine transcripts: {clip_id}",
        "",
        "Each block is one time window. Each line inside it is what one engine transcribed in "
        "that window. The engines are independent readings of the same audio; where they "
        "differ, at most one of them can be right, and sometimes none is.",
        "",
        "Sources:",
        "",
    ]
    lines += [f"- `{s.label}`: cell `{s.cell_id}` (denoise `{s.denoise}`)" for s in sources]
    lines += ["", f"{len(windows)} windows.", ""]
    for window in windows:
        lines.append(
            f"## window {window.index} [{timestamp(window.start)}-{timestamp(window.end)}]"
        )
        lines.append("")
        for source in sources:
            text = window.texts.get(source.label, "")
            lines.append(f"- `{source.label}`: {text or '(nothing)'}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def render_critic_input(
    adjudications: Mapping[str, Mapping[str, Any]],
    alignments: Mapping[str, str],
) -> str:
    """The critic's input: every adjudicated reference next to the view it was built from."""
    lines = [
        "# Adjudicated references, for adversarial review",
        "",
        "For each clip: the reference the adjudicator produced, the spans it already admitted "
        "were uncertain, and the aligned engine transcripts it worked from. Your job is the "
        "spans it did **not** admit.",
        "",
    ]
    for clip_id in sorted(adjudications):
        data = adjudications[clip_id]
        lines += [f"## {clip_id}", "", "### Adjudicated reference, window by window", ""]
        for window in data.get("windows", []):
            lines.append(f"- window {window['index']}: {window['text']}")
        spans = data.get("spans", [])
        lines += ["", f"### Spans the adjudicator already flagged ({len(spans)})", ""]
        for span in spans:
            readings = "; ".join(
                f"`{c['source']}`: {c['text']}" for c in span.get("candidates", [])
            )
            lines.append(
                f"- [{timestamp(span['start'])}-{timestamp(span['end'])}] chose "
                f"**{span['chosen']}** from {readings} ({span.get('reason', '')})"
            )
        lines += ["", "### The aligned engine transcripts", "", alignments.get(clip_id, ""), ""]
    return "\n".join(lines).rstrip() + "\n"


# --------------------------------------------------------------------------------------
# Prompts
# --------------------------------------------------------------------------------------

_ADJUDICATOR_SCHEMA = """{
  "role": "ref-adjudicator",
  "clip": "<clip id>",
  "windows": [{"index": <int>, "text": "<verbatim Serbian for that window>"}],
  "spans": [
    {
      "start": <seconds, float>,
      "end": <seconds, float>,
      "chosen": "<the reading you put in the reference>",
      "candidates": [{"source": "<engine label>", "text": "<that engine's reading>"}],
      "reason": "<why, in one sentence>",
      "confidence": "low" | "medium"
    }
  ],
  "notes": "<anything a human reviewer should know, or an empty string>"
}"""

_CRITIC_SCHEMA = """{
  "role": "ref-critic",
  "findings": [
    {
      "clip": "<clip id>",
      "start": <seconds, float>,
      "end": <seconds, float>,
      "issue": "smoothed" | "grammar_corrected" | "other",
      "reference_text": "<what the reference says there>",
      "candidates": ["<engine reading>", "..."],
      "recommendation": "<what a human should check, in one sentence>",
      "severity": "high" | "medium" | "low"
    }
  ],
  "verdict": "<one paragraph: is this reference fit to score against, and where is it weakest>"
}"""


def _adjudicator_prompt(clip_id: str, task: Mapping[str, Any], windows: Sequence[Window]) -> str:
    return f"""# Task: adjudicate a reference transcript for `{clip_id}`

Your role definition is `.claude/agents/{ADJUDICATOR}.md`. Read it first; it is the part of
this task that does not change per clip.

- Input: `{task["input"]}`
- Output: write JSON, and nothing else, to `{task["output"]}`
- Windows to cover: {len(windows)}, indices {windows[0].index} to {windows[-1].index}

Produce exactly this shape:

```json
{_ADJUDICATOR_SCHEMA}
```

Hard requirements, each of which makes the output invalid if broken:

1. `windows` must contain **every** index in the input, exactly once, in ascending order.
   A window whose engines all said nothing gets an empty string, not a missing entry.
2. Transcribe, do not edit. The reference must be what the speaker **said**, including a
   false start, a repeated word or an ungrammatical sentence. If the engines agree on
   something clumsy, the clumsy version is the reference. Writing the corrected sentence
   would make every raw engine look wrong and the `--fix` pass look right, which would
   invalidate the benchmark this reference exists to serve.
3. Every place where the engines disagree in a way you could not resolve with certainty from
   the language alone goes in `spans`, with all the candidate readings. Under-flagging is the
   expensive mistake here: a flagged span costs a human thirty seconds with the audio, an
   unflagged wrong one silently moves every WER in the table.
4. Serbian Latin script, the speaker's own dialect and word order, ordinary punctuation.
5. No commentary outside the JSON.
"""


def _critic_prompt(task: Mapping[str, Any], clips: Sequence[str]) -> str:
    return f"""# Task: find what the adjudicator got wrong

Your role definition is `.claude/agents/{CRITIC}.md`. Read it first.

- Input: `{task["input"]}`
- Output: write JSON, and nothing else, to `{task["output"]}`
- Clips under review: {", ".join(clips)}

Produce exactly this shape:

```json
{_CRITIC_SCHEMA}
```

You are adversarial. The two failures worth your attention:

1. **Silent grammar correction.** The reference reads as better Serbian than any engine
   produced. That is not adjudication, it is editing, and a reference edited that way
   penalises every raw engine and rewards the correction pass under test.
2. **A smoothed-over disagreement.** The engines genuinely differed, the adjudicator picked
   one confidently and did not flag it.

`clip` must be one of the clips listed above and `findings` may be empty if there is honestly
nothing to report. No commentary outside the JSON.
"""


# --------------------------------------------------------------------------------------
# The manifest
# --------------------------------------------------------------------------------------


def _agents_dir(run_dir: Path) -> Path:
    return run_dir / "agents"


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _load_cues(path: Path) -> tuple[Any, ...]:
    from subtitler.render import read_subtitles

    return read_subtitles(path) if path.exists() else ()


def plan(run_dir: Path, *, window_s: float = WINDOW_S) -> dict[str, Any]:
    """Build the manifest and every input document, from a finished run directory.

    Pure with respect to the references: this reads transcripts and writes nothing outside
    `<run>/agents/`. Nothing lands in `benchmarks/references/` until an adjudication has come
    back and validated.
    """
    results_path = run_dir / "results.json"
    if not results_path.exists():
        raise ValueError(f"{results_path} does not exist; is that a benchmark run directory?")
    payload = json.loads(results_path.read_text(encoding="utf-8"))

    clip_ids = sorted(
        {str(r.get("clip_id", "")) for r in payload.get("results", []) if r.get("ok")}
    )
    tasks: list[dict[str, Any]] = []
    inputs: dict[str, str] = {}

    for clip_id in clip_ids:
        if not clip_id:
            continue
        sources = select_sources(payload, clip_id)
        if len(sources) < 2:
            # One opinion is not a consensus. Say so and skip rather than adjudicate a single
            # engine's transcript into a "reference" that would score it against itself.
            continue
        cues = {s.label: _load_cues(run_dir / "transcripts" / f"{s.cell_id}.srt") for s in sources}
        windows = align(cues, window_s=window_s)
        if not windows:
            continue
        task_id = f"adjudicate-{clip_id}"
        tasks.append(
            {
                "role": ADJUDICATOR,
                "task_id": task_id,
                "clip": clip_id,
                "agent": f".claude/agents/{ADJUDICATOR}.md",
                "prompt": f"agents/prompts/{task_id}.md",
                "input": f"agents/inputs/{task_id}.md",
                "output": f"agents/outputs/{task_id}.json",
                "sources": [s.to_dict() for s in sources],
                "windows": [w.index for w in windows],
                "status": "pending",
                "attempts": 0,
                "errors": [],
                "output_sha": "",
            }
        )
        inputs[task_id] = render_alignment(clip_id, sources, windows)

    if tasks:
        critic_id = "critique-references"
        tasks.append(
            {
                "role": CRITIC,
                "task_id": critic_id,
                "clip": "",
                "agent": f".claude/agents/{CRITIC}.md",
                "prompt": f"agents/prompts/{critic_id}.md",
                "input": f"agents/inputs/{critic_id}.md",
                "output": f"agents/outputs/{critic_id}.json",
                "sources": [],
                "windows": [],
                # There is nothing to criticise until the adjudications come back, so the
                # critic's input does not exist yet and `--merge` is what writes it.
                "status": "blocked",
                "attempts": 0,
                "errors": [],
                "output_sha": "",
            }
        )

    return {
        "schema_version": 1,
        "run": run_dir.name,
        "window_s": window_s,
        "clips": clip_ids,
        "tasks": tasks,
        "_inputs": inputs,
    }


def write_plan(run_dir: Path, manifest: dict[str, Any]) -> Path:
    """Write `agent-tasks.json`, the inputs and the prompts. Returns the manifest path."""
    agents = _agents_dir(run_dir)
    (agents / "inputs").mkdir(parents=True, exist_ok=True)
    (agents / "prompts").mkdir(parents=True, exist_ok=True)
    (agents / "outputs").mkdir(parents=True, exist_ok=True)

    inputs = manifest.pop("_inputs", {})
    windows_by_clip: dict[str, list[Window]] = {}
    for task in manifest["tasks"]:
        if task["role"] != ADJUDICATOR:
            continue
        text = inputs.get(task["task_id"], "")
        (run_dir / task["input"]).write_text(text, encoding="utf-8")
        windows_by_clip[task["clip"]] = [
            Window(index=i, start=i * manifest["window_s"], end=(i + 1) * manifest["window_s"])
            for i in task["windows"]
        ]
        (run_dir / task["prompt"]).write_text(
            _adjudicator_prompt(task["clip"], task, windows_by_clip[task["clip"]]),
            encoding="utf-8",
        )

    clips = [t["clip"] for t in manifest["tasks"] if t["role"] == ADJUDICATOR]
    for task in manifest["tasks"]:
        if task["role"] == CRITIC:
            (run_dir / task["prompt"]).write_text(_critic_prompt(task, clips), encoding="utf-8")

    path = agents / "agent-tasks.json"
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


# --------------------------------------------------------------------------------------
# Validation. Strict on purpose: a field this reads wrong is a reference nobody can trust.
# --------------------------------------------------------------------------------------


def _is_number(value: Any) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool)


def _nonempty_str(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def validate_adjudication(data: Any, *, clip_id: str, window_indices: Sequence[int]) -> list[str]:
    """Every reason this response cannot become a reference, or an empty list."""
    errors: list[str] = []
    if not isinstance(data, dict):
        return ["the response is not a JSON object"]
    if data.get("role") != ADJUDICATOR:
        errors.append(f"role must be {ADJUDICATOR!r}, got {data.get('role')!r}")
    if data.get("clip") != clip_id:
        errors.append(f"clip must be {clip_id!r}, got {data.get('clip')!r}")

    windows = data.get("windows")
    if not isinstance(windows, list) or not windows:
        errors.append("windows must be a non-empty list")
    else:
        seen: list[int] = []
        for i, window in enumerate(windows):
            if not isinstance(window, dict):
                errors.append(f"windows[{i}] is not an object")
                continue
            if not isinstance(window.get("index"), int) or isinstance(window.get("index"), bool):
                errors.append(f"windows[{i}].index is not an integer")
                continue
            if not isinstance(window.get("text"), str):
                errors.append(f"windows[{i}].text is not a string")
            seen.append(int(window["index"]))
        expected = list(window_indices)
        if seen and sorted(seen) != sorted(expected):
            missing = sorted(set(expected) - set(seen))
            extra = sorted(set(seen) - set(expected))
            errors.append(
                f"windows must cover every input window exactly once: "
                f"missing {missing}, unexpected {extra}"
            )
        elif seen != sorted(seen):
            errors.append("windows must be in ascending index order")

    spans = data.get("spans")
    if not isinstance(spans, list):
        errors.append("spans must be a list (empty is allowed)")
    else:
        errors += _validate_spans(spans)

    if not isinstance(data.get("notes", ""), str):
        errors.append("notes must be a string")
    return errors


def _validate_spans(spans: Sequence[Any]) -> list[str]:
    errors: list[str] = []
    for i, span in enumerate(spans):
        where = f"spans[{i}]"
        if not isinstance(span, dict):
            errors.append(f"{where} is not an object")
            continue
        if not _is_number(span.get("start")) or not _is_number(span.get("end")):
            errors.append(f"{where}.start and .end must be numbers")
        elif float(span["end"]) < float(span["start"]):
            errors.append(f"{where} ends before it starts")
        if not _nonempty_str(span.get("chosen")):
            errors.append(f"{where}.chosen must be a non-empty string")
        if not _nonempty_str(span.get("reason")):
            errors.append(f"{where}.reason must be a non-empty string")
        if span.get("confidence") not in CONFIDENCES:
            errors.append(f"{where}.confidence must be one of {', '.join(CONFIDENCES)}")
        candidates = span.get("candidates")
        if not isinstance(candidates, list) or not candidates:
            errors.append(f"{where}.candidates must be a non-empty list")
            continue
        for j, candidate in enumerate(candidates):
            if (
                not isinstance(candidate, dict)
                or not _nonempty_str(candidate.get("source"))
                or not isinstance(candidate.get("text"), str)
            ):
                errors.append(f"{where}.candidates[{j}] needs a source and a text")
    return errors


def validate_critique(data: Any, *, clips: Sequence[str]) -> list[str]:
    """Every reason this critique cannot be recorded, or an empty list."""
    errors: list[str] = []
    if not isinstance(data, dict):
        return ["the response is not a JSON object"]
    if data.get("role") != CRITIC:
        errors.append(f"role must be {CRITIC!r}, got {data.get('role')!r}")
    if not _nonempty_str(data.get("verdict")):
        errors.append("verdict must be a non-empty string")

    findings = data.get("findings")
    if not isinstance(findings, list):
        errors.append("findings must be a list (empty is allowed)")
        return errors

    for i, finding in enumerate(findings):
        where = f"findings[{i}]"
        if not isinstance(finding, dict):
            errors.append(f"{where} is not an object")
            continue
        if finding.get("clip") not in clips:
            errors.append(f"{where}.clip must be one of {', '.join(clips)}")
        if not _is_number(finding.get("start")) or not _is_number(finding.get("end")):
            errors.append(f"{where}.start and .end must be numbers")
        if finding.get("issue") not in ISSUES:
            errors.append(f"{where}.issue must be one of {', '.join(ISSUES)}")
        if finding.get("severity") not in SEVERITIES:
            errors.append(f"{where}.severity must be one of {', '.join(SEVERITIES)}")
        if not _nonempty_str(finding.get("recommendation")):
            errors.append(f"{where}.recommendation must be a non-empty string")
        if not isinstance(finding.get("reference_text", ""), str):
            errors.append(f"{where}.reference_text must be a string")
        candidates = finding.get("candidates", [])
        if not isinstance(candidates, list) or any(not isinstance(c, str) for c in candidates):
            errors.append(f"{where}.candidates must be a list of strings")
    return errors


# --------------------------------------------------------------------------------------
# Merging what came back
# --------------------------------------------------------------------------------------


def reference_text(windows: Sequence[Mapping[str, Any]]) -> str:
    """The reference file's contents: one window per line, in index order.

    Line breaks rather than one paragraph because a human has to read this against an audio
    player, and a window per line is a coarse index into the clip. The normalizer collapses
    whitespace before anything is scored, so the layout cannot change a number.
    """
    lines = [str(w.get("text", "")).strip() for w in sorted(windows, key=lambda w: w["index"])]
    return "\n".join(line for line in lines if line).strip() + "\n"


def _read_output(run_dir: Path, task: Mapping[str, Any]) -> tuple[Any, str, str]:
    """`(parsed, raw, error)`. A missing file is not an error, it is work not done yet."""
    path = run_dir / task["output"]
    if not path.exists():
        return None, "", "missing"
    raw = path.read_text(encoding="utf-8")
    try:
        return json.loads(raw), raw, ""
    except json.JSONDecodeError as exc:
        return None, raw, f"the response is not valid JSON: {exc}"


def _record_failure(task: dict[str, Any], errors: Sequence[str], sha: str) -> None:
    """Count one attempt against a *new* response, and fail on the second.

    The sha is what makes that honest: merging twice over the same unchanged bad file must
    not burn both attempts, or the operator loses the retry to their own `--merge`.
    """
    if task.get("output_sha") != sha:
        task["attempts"] = int(task.get("attempts", 0)) + 1
        task["output_sha"] = sha
    task["errors"] = list(errors)
    task["status"] = "failed" if int(task["attempts"]) >= MAX_ATTEMPTS else "retry"


def merge(
    run_dir: Path,
    *,
    references: Path,
    log: Callable[[str], None] = print,
) -> dict[str, Any]:
    """Fold every returned response into references, meta and the review queue.

    Deterministic, and idempotent: the same files on disk always produce the same references,
    in the same order, whatever order the responses arrived in. The critic never edits the
    reference text. It can only add a span to the human review queue and a sentence to the
    meta, because an adversarial agent silently rewriting the artifact it is auditing is the
    one thing worse than the smoothing it is looking for.
    """
    manifest_path = _agents_dir(run_dir) / "agent-tasks.json"
    if not manifest_path.exists():
        raise ValueError(f"{manifest_path} does not exist; run `bench agents <run>` first")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    references.mkdir(parents=True, exist_ok=True)

    tasks = manifest.get("tasks", [])
    adjudications: dict[str, dict[str, Any]] = {}
    clips = [t["clip"] for t in tasks if t["role"] == ADJUDICATOR]

    for task in tasks:
        if task["role"] != ADJUDICATOR:
            continue
        data, raw, error = _read_output(run_dir, task)
        if error == "missing":
            task["status"] = "pending"
            log(f"{task['task_id']}: no response yet at {task['output']}")
            continue
        errors = (
            [error]
            if error
            else validate_adjudication(
                data, clip_id=task["clip"], window_indices=task.get("windows", [])
            )
        )
        if errors:
            _record_failure(task, errors, _digest(raw))
            log(
                f"{task['task_id']}: {task['status']} after {task['attempts']} attempt(s): {errors[0]}"
            )
            continue
        task["status"] = "ok"
        task["errors"] = []
        task["output_sha"] = _digest(raw)
        task["attempts"] = max(1, int(task.get("attempts", 0)))
        adjudications[task["clip"]] = data

    critique, critic_task = _merge_critique(run_dir, tasks, clips, adjudications, log)

    written = []
    for clip_id in sorted(adjudications):
        written.append(
            _write_reference(
                references,
                run_dir,
                clip_id,
                adjudications[clip_id],
                next(t for t in tasks if t["role"] == ADJUDICATOR and t["clip"] == clip_id),
                critique,
            )
        )

    if adjudications:
        (references / REVIEW_QUEUE).write_text(
            review_table(adjudications, critique, run=manifest.get("run", "")), encoding="utf-8"
        )
        log(f"wrote {references / REVIEW_QUEUE}")

    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return {
        "run": manifest.get("run", ""),
        "references": written,
        "spans": sum(len(a.get("spans", [])) for a in adjudications.values()),
        "critic_findings": len(critique.get("findings", [])) if critique else 0,
        "critic_status": critic_task.get("status", "") if critic_task else "",
        "failed": sorted(t["task_id"] for t in tasks if t["status"] == "failed"),
        "retry": sorted(t["task_id"] for t in tasks if t["status"] == "retry"),
        "pending": sorted(t["task_id"] for t in tasks if t["status"] in ("pending", "blocked")),
    }


def _merge_critique(
    run_dir: Path,
    tasks: Sequence[dict[str, Any]],
    clips: Sequence[str],
    adjudications: Mapping[str, Mapping[str, Any]],
    log: Callable[[str], None],
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Read the critic's response, and write its input the moment one can be written."""
    critic_task = next((t for t in tasks if t["role"] == CRITIC), None)
    if critic_task is None:
        return {}, None

    data, raw, error = _read_output(run_dir, critic_task)
    if error == "missing":
        if adjudications and len(adjudications) == len(clips):
            alignments = {
                clip: (run_dir / f"agents/inputs/adjudicate-{clip}.md").read_text(encoding="utf-8")
                for clip in adjudications
                if (run_dir / f"agents/inputs/adjudicate-{clip}.md").exists()
            }
            (run_dir / critic_task["input"]).write_text(
                render_critic_input(adjudications, alignments), encoding="utf-8"
            )
            critic_task["status"] = "pending"
            log(f"{critic_task['task_id']}: input ready at {critic_task['input']}")
        else:
            critic_task["status"] = "blocked"
        return {}, critic_task

    errors = [error] if error else validate_critique(data, clips=clips)
    if errors:
        _record_failure(critic_task, errors, _digest(raw))
        log(
            f"{critic_task['task_id']}: {critic_task['status']} after "
            f"{critic_task['attempts']} attempt(s): {errors[0]}"
        )
        return {}, critic_task

    critic_task["status"] = "ok"
    critic_task["errors"] = []
    critic_task["output_sha"] = _digest(raw)
    critic_task["attempts"] = max(1, int(critic_task.get("attempts", 0)))
    return data, critic_task


def _write_reference(
    references: Path,
    run_dir: Path,
    clip_id: str,
    adjudication: Mapping[str, Any],
    task: Mapping[str, Any],
    critique: Mapping[str, Any],
) -> str:
    """The reference itself, and the provenance that says what it is worth."""
    text = reference_text(adjudication.get("windows", []))
    (references / f"{clip_id}.txt").write_text(text, encoding="utf-8")

    findings = [f for f in critique.get("findings", []) if f.get("clip") == clip_id]
    meta = {
        "clip": clip_id,
        "reference": f"{clip_id}.txt",
        "status": "present",
        "human_verified": False,
        "adjudicated": True,
        "method": "llm-adjudicated consensus of engine transcripts, aligned by timestamp",
        "run": run_dir.name,
        "engine_cells": [s["cell_id"] for s in task.get("sources", [])],
        "engines": [s["label"] for s in task.get("sources", [])],
        "windows": len(adjudication.get("windows", [])),
        "spans_flagged": len(adjudication.get("spans", [])),
        "critic_findings": len(findings),
        "critic_verdict": str(critique.get("verdict", "")),
        "review_queue": f"benchmarks/references/{REVIEW_QUEUE}",
        "agent": {
            "role": task.get("role", ADJUDICATOR),
            "task_id": task.get("task_id", ""),
            "attempts": int(task.get("attempts", 0)),
        },
        "source": (
            f"adjudicated from {len(task.get('sources', []))} engine transcripts in "
            f"benchmarks/results/{run_dir.name}"
        ),
        "words": len(text.split()),
        "characters": len(text),
        "note": CAVEAT,
    }
    (references / f"{clip_id}.meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return f"{clip_id}.txt"


def review_table(
    adjudications: Mapping[str, Mapping[str, Any]],
    critique: Mapping[str, Any],
    *,
    run: str = "",
) -> str:
    """The human review list: every uncertain span, with the audio timestamp to check it at.

    This is the document that turns `human_verified: false` into `true`. It is deliberately
    short and sorted by clip then timestamp, so a reviewer can open the audio once, scrub to
    each row in order, and answer a yes/no question at each one.
    """
    total = sum(len(a.get("spans", [])) for a in adjudications.values()) + len(
        critique.get("findings", [])
    )
    lines = [
        "# Reference review queue",
        "",
        f"{total} span(s) need one human with the audio. "
        + (f"From run `{run}`. " if run else "")
        + "Each row is a place where the engines disagreed and an LLM that cannot hear the "
        "audio had to choose, or where the critic thinks that choice was wrong.",
        "",
        "**Until every row is resolved, `human_verified` stays `false` in every "
        "`*.meta.json`, and every WER in the benchmark report is marked provisional.** "
        "To resolve a row: open the clip at the timestamp, listen, and correct the line in "
        "`benchmarks/references/<clip>.txt` if the chosen reading is wrong. When the list is "
        "clear, set `human_verified: true` and re-run `subtitler bench report <run>`.",
        "",
        "| clip | time | candidate readings | chosen | why | flagged by |",
        "|---|---|---|---|---|---|",
    ]
    rows: list[tuple[str, float, str]] = []
    for clip_id in sorted(adjudications):
        for span in adjudications[clip_id].get("spans", []):
            candidates = " / ".join(
                f"`{c['source']}`: {c['text']}" for c in span.get("candidates", [])
            )
            rows.append(
                (
                    clip_id,
                    float(span["start"]),
                    f"| {clip_id} | {timestamp(span['start'])} | {candidates} | "
                    f"**{span['chosen']}** | {span.get('reason', '')} | "
                    f"adjudicator ({span.get('confidence', '')}) |",
                )
            )
    for finding in critique.get("findings", []):
        candidates = " / ".join(str(c) for c in finding.get("candidates", []))
        rows.append(
            (
                str(finding.get("clip", "")),
                float(finding.get("start", 0.0)),
                f"| {finding.get('clip', '')} | {timestamp(float(finding.get('start', 0.0)))} | "
                f"{candidates} | {finding.get('reference_text', '')} | "
                f"{finding.get('recommendation', '')} | "
                f"critic: {finding.get('issue', '')} ({finding.get('severity', '')}) |",
            )
        )
    lines += [row for _, _, row in sorted(rows, key=lambda r: (r[0], r[1], r[2]))]
    if not rows:
        lines.append("| _(none flagged)_ | | | | | |")

    verdict = str(critique.get("verdict", "")).strip()
    if verdict:
        lines += ["", "## The critic's verdict", "", verdict]
    lines += [
        "",
        "## What this reference is",
        "",
        CAVEAT,
        "",
    ]
    return "\n".join(lines).rstrip() + "\n"
