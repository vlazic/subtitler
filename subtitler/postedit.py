"""The optional LLM correction pass.

One job: correct the *text* of already-timed cues. Timestamps are never sent to the model,
so they cannot come back wrong. The cue list that goes in and the cue list that comes out
have the same length, the same indices, and byte-identical `start` and `end` values, unless
`--drop-intro-phrases` explicitly removes whole cues.

That is a direct response to how the bash pipeline this replaces did it. It chunked the VTT
with `awk` by counting blank lines, which can cut a cue in half; it asked the model to
"keep the timestamps, just correct the text", which puts the clock inside the thing being
rewritten; and it then needed a separate validate step plus two repair scripts that could
only report "broken", never *which* chunk. Here the unit of work is a parsed `Cue`, the
model only ever sees a numbered list of strings, and a batch whose reply does not match the
request exactly is discarded on the spot with a warning naming the batch and its cue range.

Three rules this module exists to enforce:

1. **No sampling parameters unless asked for.** Current Claude models (Opus 5, Sonnet 5,
   Opus 4.7/4.8) reject `temperature`, `top_p` and `top_k` with a 400, and LiteLLM forwards
   whatever it is handed. So the kwargs dict is built empty and `temperature` is added only
   when the user passed `--fix-temperature`. There is no default to "helpfully" send.
2. **Validate before accepting.** Same count, same index set, every text a string. Anything
   else and the batch is rejected and the original cues are kept.
3. **The prompt is a file, not a string literal.** `prompts/postedit.md` is generic;
   `prompts/gozba.md` is the domain variant. Only the editorial policy comes from the file:
   the JSON contract is appended here, because a prompt file that could break the contract
   would silently reject every batch.

LiteLLM is imported lazily inside `_litellm_complete`. It costs about 1.7 seconds to
import, `--fix` is off by default, and it lives behind the `fix` extra, so a normal run
must not pay for it or require it.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable, Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from subtitler.cues import CueConfig, wrap_edited, wrap_text
from subtitler.model import Cue

DEFAULT_MODEL = "anthropic/claude-sonnet-5"
DEFAULT_PROMPT = "postedit"
DEFAULT_BATCH_SIZE = 40
DEFAULT_WORKERS = 4
DEFAULT_TIMEOUT_S = 120.0

# Retries for transport-level failures (a 429, a dropped connection), handled by LiteLLM.
# Not retries for a reply that failed validation: those are counted separately below.
NETWORK_RETRIES = 2

# A batch whose reply does not validate is asked once more before it is discarded. One
# retry, not a loop: a model that cannot follow the contract twice will not follow it on
# the fifth attempt, and the caller is paying per attempt.
VALIDATION_ATTEMPTS = 2

# Where prompt files are looked up, in order. The first is the packaged copy (the wheel
# force-includes `prompts/` as `subtitler/prompts/`), the second is the repo checkout.
PROMPT_DIRS: tuple[Path, ...] = (
    Path(__file__).resolve().parent / "prompts",
    Path(__file__).resolve().parent.parent / "prompts",
)

# Appended to every prompt file. The model never sees a timestamp.
CONTRACT = """
## Input and output format

The user message is a JSON array of objects, each `{"i": <integer>, "text": "<subtitle>"}`.

Reply with a JSON array of exactly the same length, containing exactly the same `i` values,
in the same order, each `{"i": <integer>, "text": "<corrected subtitle>"}`.

Reply with the JSON array and nothing else: no prose before it, no explanation after it, no
code fence. Never add, remove, reorder or renumber items. `i` is an opaque identifier: copy
it through unchanged.
""".strip()


class FixError(RuntimeError):
    """The correction pass could not run at all. Always carries an actionable fix."""


class BatchRejected(ValueError):
    """A reply did not match its request. The batch is discarded, the originals kept."""


@dataclass(frozen=True, slots=True)
class FixConfig:
    model: str = DEFAULT_MODEL
    prompt: str = DEFAULT_PROMPT
    batch_size: int = DEFAULT_BATCH_SIZE
    workers: int = DEFAULT_WORKERS
    # None means "send no sampling parameters at all", which is the only safe default.
    temperature: float | None = None
    markup: str = "strip"  # strip | html
    drop_intro_phrases: Path | None = None
    timeout_s: float = DEFAULT_TIMEOUT_S


@dataclass(slots=True)
class FixReport:
    model: str = ""
    batches: int = 0
    rejected: list[str] = field(default_factory=list)
    changed: int = 0
    dropped: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "batches": self.batches,
            "rejected_batches": len(self.rejected),
            "changed_cues": self.changed,
            "dropped_cues": self.dropped,
        }


# --------------------------------------------------------------------------------------
# Markdown. Ported from gozba2/emisije/replace_markdown_formatting.py.
# --------------------------------------------------------------------------------------

# Non-greedy, and anchored so `**` never matches as two `*`. Applied to one cue's text at
# a time, so a run that spans a line break is not a case that can arise.
_BOLD_RE = re.compile(r"\*\*(?!\s)(.+?)(?<!\s)\*\*")
_ITALIC_RE = re.compile(r"(?<!\*)\*(?!\s|\*)(.+?)(?<!\s|\*)\*(?!\*)")
_CODE_RE = re.compile(r"`([^`]+)`")
# A leading `#` or `-`/`*` bullet: a model that decides a cue is a heading or a list item.
_LEAD_MARKER_RE = re.compile(r"^\s*(?:#{1,6}\s+|[-*+]\s+(?=\S))")


def strip_markdown(text: str, *, html: bool = False) -> str:
    """Remove markdown emphasis a model emitted despite being told not to.

    The original script only ever converted `**x**` to `<b>x</b>` and `*x*` to `<i>x</i>`,
    because its output was a `.vtt` for a web player. That is now the `html=True` branch
    and it is not the default: the default output is burned into video through libass,
    which renders `<b>` as the four literal characters `<b>`, and an `.srt` that reads
    `<b>Aristotel</b>` is worse than one that reads `Aristotel`.

    Bold is handled before italic in both branches, so `**x**` is never seen as `*` + `*x*`.
    """
    text = _LEAD_MARKER_RE.sub("", text)
    if html:
        text = _BOLD_RE.sub(r"<b>\1</b>", text)
        text = _ITALIC_RE.sub(r"<i>\1</i>", text)
    else:
        text = _BOLD_RE.sub(r"\1", text)
        text = _ITALIC_RE.sub(r"\1", text)
    text = _CODE_RE.sub(r"\1", text)
    return text.strip()


# --------------------------------------------------------------------------------------
# Intro phrases. Ported from gozba2/emisije/remove_intro.py.
# --------------------------------------------------------------------------------------

_MARKUP_RE = re.compile(r"</?[a-zA-Z][^>]*>")
# Straight and typographic quotes, plus the emphasis markers, written as escapes the way
# `cues.py` writes its dashes: a curly quote and a straight one are indistinguishable in a
# character class at a glance, and this one decides whether a cue is deleted.
_NOISE_RE = re.compile("[\"'*_\u201e\u201c\u201d\u00bb\u00ab\u2018\u2019]")


def _normalize_phrase(text: str) -> str:
    """Fold away everything that made the original need five spellings of one sentence.

    `remove_intro.py` listed the same station ident five times: bare, lowercased, in
    straight quotes, in `<i>` tags, and in asterisks. Normalising markup, quotes and case
    away on both sides collapses those to one line in the phrase file.
    """
    text = _MARKUP_RE.sub(" ", text)
    text = _NOISE_RE.sub(" ", text)
    return " ".join(text.split()).casefold()


def load_intro_phrases(path: Path) -> tuple[str, ...]:
    """One phrase per line. Blank lines and `#` comments are ignored."""
    lines = path.read_text(encoding="utf-8").splitlines()
    return tuple(
        line.strip() for line in lines if line.strip() and not line.lstrip().startswith("#")
    )


def drop_intro_cues(cues: Sequence[Cue], phrases: Iterable[str]) -> tuple[Cue, ...]:
    """Drop the cues that carry a boilerplate phrase, and only those cues.

    The original deleted *everything before* the matched line as well, so one ident line
    appearing 40 seconds in silently threw away the first 40 seconds of the show. Here a
    cue is dropped when it matches and kept when it does not; the survivors keep their
    timestamps exactly and are renumbered so the `.srt` stays sequential.
    """
    needles = tuple(n for n in (_normalize_phrase(p) for p in phrases) if n)
    if not needles:
        return tuple(cues)

    kept: list[Cue] = []
    for cue in cues:
        haystack = _normalize_phrase(cue.text)
        if any(needle in haystack for needle in needles):
            continue
        kept.append(Cue(index=len(kept) + 1, start=cue.start, end=cue.end, lines=cue.lines))
    return tuple(kept)


# --------------------------------------------------------------------------------------
# Prompts
# --------------------------------------------------------------------------------------


def load_prompt(name_or_path: str) -> str:
    """Resolve a prompt by path, or by bare name from the prompt directories.

    Everything above the first `---` rule is documentation for whoever edits the file, not
    instruction for the model, so it is dropped. The contract is appended afterwards, where
    no edit to the file can remove it.
    """
    candidate = Path(name_or_path).expanduser()
    if candidate.suffix and candidate.is_file():
        text = candidate.read_text(encoding="utf-8")
    else:
        stem = candidate.name.removesuffix(".md")
        for directory in PROMPT_DIRS:
            path = directory / f"{stem}.md"
            if path.is_file():
                text = path.read_text(encoding="utf-8")
                break
        else:
            known = sorted({p.stem for d in PROMPT_DIRS if d.is_dir() for p in d.glob("*.md")})
            raise FixError(
                f"no prompt {name_or_path!r}; built in: {', '.join(known) or 'none found'}, "
                "or pass a path to a .md file"
            )

    _, sep, body = text.partition("\n---\n")
    policy = (body if sep else text).strip()
    if not policy:
        raise FixError(f"prompt {name_or_path!r} is empty")
    return f"{policy}\n\n{CONTRACT}"


# --------------------------------------------------------------------------------------
# Batching and validation. The part the bash pipeline got wrong.
# --------------------------------------------------------------------------------------


def batch_cues(cues: Sequence[Cue], size: int) -> list[tuple[Cue, ...]]:
    """Fixed-size batches of whole cues. A cue is never split across a request."""
    if size < 1:
        raise FixError(f"--fix-batch must be at least 1, got {size}")
    return [tuple(cues[i : i + size]) for i in range(0, len(cues), size)]


def render_request(batch: Sequence[Cue]) -> str:
    """The user message: indices and text, no timestamps."""
    return json.dumps(
        [{"i": cue.index, "text": cue.text} for cue in batch],
        ensure_ascii=False,
        indent=None,
    )


_FENCE_RE = re.compile(r"^\s*```(?:json)?\s*(.*?)\s*```\s*$", re.DOTALL)


def parse_reply(reply: str, batch: Sequence[Cue]) -> dict[int, str]:
    """Validate a reply against the batch that produced it, or raise `BatchRejected`.

    Count and index set must match exactly. Nothing partial is accepted: taking the items
    that happen to line up and leaving the rest would produce a file whose cues came from
    two different passes, and no message anywhere saying so.
    """
    text = (reply or "").strip()
    if not text:
        raise BatchRejected("empty reply")

    fenced = _FENCE_RE.match(text)
    if fenced:
        # Told not to fence it, some models fence it anyway. Cheap to accept.
        text = fenced.group(1).strip()

    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise BatchRejected(f"reply is not JSON ({exc.msg})") from exc

    if not isinstance(data, list):
        raise BatchRejected(f"reply is a {type(data).__name__}, expected a JSON array")
    if len(data) != len(batch):
        raise BatchRejected(f"reply has {len(data)} items, expected {len(batch)}")

    out: dict[int, str] = {}
    for item in data:
        if not isinstance(item, dict) or "i" not in item or "text" not in item:
            raise BatchRejected("an item is not an object with 'i' and 'text'")
        index, value = item["i"], item["text"]
        if not isinstance(index, int) or isinstance(index, bool):
            raise BatchRejected(f"index {index!r} is not an integer")
        if not isinstance(value, str):
            raise BatchRejected(f"text for index {index} is a {type(value).__name__}")
        if not value.strip():
            raise BatchRejected(f"text for index {index} is empty")
        out[index] = value

    expected = {cue.index for cue in batch}
    if set(out) != expected:
        missing = sorted(expected - set(out))
        extra = sorted(set(out) - expected)
        detail = []
        if missing:
            detail.append(f"missing {missing}")
        if extra:
            detail.append(f"unexpected {extra}")
        raise BatchRejected("index set does not match: " + ", ".join(detail))
    return out


def _reflow(lines: tuple[str, ...], tokens: list[str]) -> tuple[str, ...] | None:
    """Rebuild `lines` from `tokens`, keeping the same number of words per line.

    `strip_markdown` only ever removes a marker or swaps it for a tag glued to the word it
    marks, so the marked-up text and the plain text always have the same token count. That
    is what lets the break be chosen on one and applied to the other. If they ever diverge,
    return None rather than guess, and the caller falls back.
    """
    counts = [len(line.split()) for line in lines]
    if sum(counts) != len(tokens):
        return None
    out: list[str] = []
    cursor = 0
    for count in counts:
        out.append(" ".join(tokens[cursor : cursor + count]))
        cursor += count
    return tuple(out)


def _relayout(text: str, marked: str, cue: Cue, cfg: CueConfig) -> tuple[str, ...]:
    """Re-wrap a corrected cue's text, through the same wrapper the splitter uses.

    The wrapping itself is `cues.wrap_edited`, which is shared with the GUI's hand editor
    and carries the explanation of why the greedy `wrap_text` is the wrong one. What is
    specific to this module is the markup layer below.

    `text` is the markup-free version and `marked` is what actually gets written. The break
    is chosen on `text` and applied to `marked`, because `<b>` and `</b>` are seven
    characters of nothing on a web player: measuring them against `max_line` pushed a
    two-line cue in the gozba fixture onto three lines for emphasis nobody can see.
    """
    lines = wrap_edited(text, start=cue.start, end=cue.end, config=cfg)
    if marked == text:
        return lines
    return _reflow(lines, marked.split()) or wrap_text(
        marked, max_line=cfg.max_line, max_lines=len(marked.split()) or 1
    )


# --------------------------------------------------------------------------------------
# The model call
# --------------------------------------------------------------------------------------

Completer = Callable[[str, str], str]


def _import_litellm() -> Any:
    try:
        import litellm
    except ImportError as exc:
        raise FixError("--fix needs LiteLLM.\n  fix: uv sync --extra fix") from exc
    litellm.suppress_debug_info = True
    return litellm


def litellm_completer(cfg: FixConfig) -> Completer:
    """Bind a `FixConfig` to a callable that takes (system, user) and returns the reply.

    The import happens here rather than inside the callable so a machine without the `fix`
    extra is told so once, before any batching, instead of having every batch fail with an
    ImportError and the run then report "every batch failed" for what is a one-line install.
    """
    _import_litellm()

    def complete(system: str, user: str) -> str:
        return _litellm_complete(system, user, cfg)

    return complete


def _litellm_complete(system: str, user: str, cfg: FixConfig) -> str:
    litellm = _import_litellm()

    # Built empty on purpose. Anthropic's current models return a 400 for `temperature`,
    # `top_p` or `top_k`, and LiteLLM forwards every sampling parameter it is given, so the
    # only way to be correct across providers is to send none unless the user asked.
    kwargs: dict[str, Any] = {
        "model": cfg.model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "timeout": cfg.timeout_s,
        "num_retries": NETWORK_RETRIES,
    }
    if cfg.temperature is not None:
        kwargs["temperature"] = cfg.temperature

    response = litellm.completion(**kwargs)
    try:
        return response.choices[0].message.content or ""
    except (AttributeError, IndexError, KeyError) as exc:
        raise BatchRejected(f"unreadable response from {cfg.model}: {exc}") from exc


# --------------------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------------------


def fix_cues(
    cues: Sequence[Cue],
    cfg: FixConfig,
    *,
    cue_config: CueConfig | None = None,
    complete: Completer | None = None,
    log: Callable[[str], None] = print,
) -> tuple[tuple[Cue, ...], FixReport]:
    """Correct cue text. Returns the new cues and a report of what happened.

    Every returned cue keeps the `start` and `end` it came in with. The only structural
    change this function can make is removing whole cues, and only when
    `drop_intro_phrases` is set.
    """
    layout = cue_config or CueConfig()
    report = FixReport(model=cfg.model)
    if not cues:
        return tuple(cues), report

    call = complete or litellm_completer(cfg)
    system = load_prompt(cfg.prompt)
    batches = batch_cues(cues, cfg.batch_size)
    report.batches = len(batches)

    def run(job: tuple[int, tuple[Cue, ...]]) -> tuple[int, dict[int, str] | str]:
        number, batch = job
        span = f"batch {number}/{len(batches)} (cues {batch[0].index}-{batch[-1].index})"
        last = ""
        for attempt in range(1, VALIDATION_ATTEMPTS + 1):
            try:
                return number, parse_reply(call(system, render_request(batch)), batch)
            except BatchRejected as exc:
                last = str(exc)
                if attempt < VALIDATION_ATTEMPTS:
                    log(f"fix: {span} rejected ({last}); retrying once")
            except Exception as exc:
                # Auth, quota, connection: not worth a second attempt at the same wall.
                return number, f"{type(exc).__name__}: {exc}"
        return number, last

    workers = max(1, min(cfg.workers, len(batches)))
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="fix") as pool:
        results = dict(pool.map(run, enumerate(batches, start=1)))

    corrected: dict[int, str] = {}
    for number, batch in enumerate(batches, start=1):
        outcome = results[number]
        if isinstance(outcome, dict):
            corrected.update(outcome)
            continue
        span = f"batch {number}/{len(batches)} (cues {batch[0].index}-{batch[-1].index})"
        report.rejected.append(f"{span}: {outcome}")
        log(f"fix: {span} discarded, keeping the original text ({outcome})")

    if report.rejected and len(report.rejected) == len(batches):
        # Nothing came back usable. Writing the untouched file and reporting success is
        # exactly the failure mode this project refuses: a run that says it corrected the
        # text and did not.
        raise FixError(f"every batch failed. First: {report.rejected[0]}")

    out: list[Cue] = []
    for cue in cues:
        replacement = corrected.get(cue.index)
        if replacement is None:
            out.append(cue)
            continue
        plain = strip_markdown(replacement)
        marked = strip_markdown(replacement, html=True) if cfg.markup == "html" else plain
        if not plain or marked == cue.text:
            # Most cues come back untouched, and the splitter's original break used real
            # word timings that cannot be recovered here. Re-wrapping an unchanged string
            # would trade a better line break for an identical one, and would also count
            # the cue as "changed" when nothing about it changed.
            out.append(cue)
            continue
        lines = _relayout(plain, marked, cue, layout)
        if lines != cue.lines:
            report.changed += 1
        out.append(Cue(index=cue.index, start=cue.start, end=cue.end, lines=lines))

    result = tuple(out)
    if cfg.drop_intro_phrases is not None:
        before = len(result)
        result = drop_intro_cues(result, load_intro_phrases(cfg.drop_intro_phrases))
        report.dropped = before - len(result)
        if report.dropped:
            log(f"fix: dropped {report.dropped} cue(s) matching {cfg.drop_intro_phrases.name}")

    return result, report


def cache_params(cfg: FixConfig, layout: CueConfig) -> dict[str, Any]:
    """What the `fix` stage's cache key must cover.

    The prompt's *content* is in the key, not its name: editing `prompts/postedit.md` must
    invalidate the corrected cues, and it would not if only "postedit" were recorded. The
    line layout is in it because the corrected text is re-wrapped here. `workers` is
    deliberately absent: thread count cannot change the result, and putting it in the key
    would mean `--fix-workers 8` re-ran the whole pass for nothing.
    """
    payload: dict[str, Any] = {
        "model": cfg.model,
        "prompt": cfg.prompt,
        "prompt_sha": hashlib.sha256(load_prompt(cfg.prompt).encode("utf-8")).hexdigest()[:16],
        "batch_size": cfg.batch_size,
        "temperature": cfg.temperature,
        "markup": cfg.markup,
        "max_line": layout.max_line,
        "max_lines": layout.max_lines,
    }
    if cfg.drop_intro_phrases is not None:
        phrases = load_intro_phrases(cfg.drop_intro_phrases)
        payload["intro_phrases"] = hashlib.sha256("\n".join(phrases).encode("utf-8")).hexdigest()[
            :16
        ]
    return payload
