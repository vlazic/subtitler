"""Hand corrections to the cue text, and the one place they are allowed to live.

The GUI lets a human read the cues before the burn and fix what the recognizer heard
wrong. Where those corrections are stored is the whole design problem, and there are two
obvious answers that are both wrong:

* **Write them into `cues.json`.** That artifact belongs to the `cues` stage, whose key is
  derived from the transcript and `CueConfig`. Neither changed when the human typed, so the
  next run recomputes the stage, the key still matches, and `segments_to_cues` overwrites
  every correction without a word.
* **Put them in the `cues` stage's key.** Then the corrections survive, and every one of
  them invalidates the transcript-derived artifact that has nothing to do with them.

So they live in `edits.json`, which is *nobody's* artifact: no stage writes it, so no stage
can clobber it. It is read as an input by a stage of its own, `edit`, that sits between
`cues`/`fix` and `burn`:

    cues -> [fix] -> edit -> burn
                      ^
                  edits.json (written by the editor, read by the pipeline)

`edits.json` records the key of the cues it was written against. That is what makes a
changed transcript safe: a new model, a different denoiser or `--force transcribe` changes
the `cues` key, the recorded `base_key` no longer matches, and the corrections are *not*
applied to text they were never about. They are reported, not deleted, because cue 41 of
the old transcript is not cue 41 of the new one and quietly re-pointing them would be the
worst possible outcome. Nothing is thrown away either: switch back to the model they were
made under and they line up again and apply.

The `edit` stage's own key is the upstream key plus a digest of the corrections, so the
burn re-runs exactly when the text it would render has changed, and not when anything else
has.

**Text only.** An edit may change what a cue says and nothing else. The timings came from
real word timings and are what `lint` measures reading speed against; letting the editor
move them would put the one number a human cannot judge by eye under the mouse. Cues are
never added or removed either, which is what keeps a correction addressable by its index
across re-runs.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from subtitler.cache import text_id
from subtitler.cues import CueConfig, display_len, lint_cues, wrap_edited
from subtitler.model import Cue

SCHEMA_VERSION = 1
EDITS_NAME = "edits.json"
ARTIFACT_NAME = "edited.json"


class EditError(ValueError):
    """A correction that cannot be stored, named so a form can point at the cue."""


@dataclass(frozen=True, slots=True)
class EditSet:
    """The corrections on disk: which cues, what they should say, and against what."""

    base_key: str
    texts: dict[int, str]

    def __bool__(self) -> bool:
        return bool(self.texts)

    def digest(self) -> str:
        """A stable id for this exact set of corrections, for the stage key.

        The corrections themselves would work as cache params, but a meta file carrying a
        paragraph of Serbian per corrected cue stops being readable by hand, which is the
        property `cache.py` picks one slot per stage to protect.
        """
        payload = json.dumps(
            sorted((index, text) for index, text in self.texts.items()),
            ensure_ascii=False,
            separators=(",", ":"),
        )
        return text_id(payload)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "base_key": self.base_key,
            "edits": [{"index": i, "text": self.texts[i]} for i in sorted(self.texts)],
        }


def path_for(work: Path) -> Path:
    return work / EDITS_NAME


def normalize(text: str) -> str:
    """One space between words, none at the ends. Layout is decided by `wrap_edited`."""
    return " ".join(str(text).split())


def build(base_key: str, raw: Any) -> EditSet:
    """Validate whatever the editor posted into an `EditSet`.

    Accepts a mapping of cue index to text, which is what JSON gives back for a dict
    whose keys are numbers. Blank text is refused rather than treated as "delete this
    cue": deleting one would renumber the rest, and every correction here is addressed by
    the index it was made against.
    """
    if not str(base_key).strip():
        raise EditError("the corrections do not say which cues they were made against")
    if not isinstance(raw, dict):
        raise EditError("expected a mapping of cue number to corrected text")

    texts: dict[int, str] = {}
    for key, value in raw.items():
        try:
            index = int(key)
        except (TypeError, ValueError) as exc:
            raise EditError(f"{key!r} is not a cue number") from exc
        text = normalize(value if isinstance(value, str) else "")
        if not text:
            raise EditError(f"cue {index} would be left empty; a cue must say something")
        texts[index] = text
    return EditSet(base_key=str(base_key).strip(), texts=texts)


def save(work: Path, edit_set: EditSet) -> Path:
    path = path_for(work)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(edit_set.to_dict(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return path


def clear(work: Path) -> None:
    path_for(work).unlink(missing_ok=True)


def load(work: Path) -> EditSet | None:
    """Read the corrections, or None when there are none to read.

    An unreadable or wrong-schema file is treated as absent rather than fatal. It is the
    one file in the work directory a human is invited to open, and a typo in it must not
    make the pipeline refuse to run at all.
    """
    path = path_for(work)
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(data, dict) or data.get("schema_version") != SCHEMA_VERSION:
        return None
    texts: dict[int, str] = {}
    for item in data.get("edits", []):
        try:
            index = int(item["index"])
        except (KeyError, TypeError, ValueError):
            continue
        text = normalize(item.get("text", ""))
        if text:
            texts[index] = text
    base_key = str(data.get("base_key") or "")
    return EditSet(base_key=base_key, texts=texts) if base_key else None


# --------------------------------------------------------------------------------------
# Applying them
# --------------------------------------------------------------------------------------


def relayout(cue: Cue, text: str, config: CueConfig | None = None) -> Cue:
    """The corrected cue: same index, same clock, re-broken lines.

    The break goes through `cues.wrap_edited`, never the greedy `wrap_text`, for the reason
    written out there. A cue whose text is unchanged is returned untouched, break and all:
    the splitter chose that break from real word timings and nothing here can do better.
    """
    wanted = normalize(text)
    if wanted == normalize(cue.text):
        return cue
    return Cue(
        index=cue.index,
        start=cue.start,
        end=cue.end,
        lines=wrap_edited(wanted, start=cue.start, end=cue.end, config=config),
    )


def apply_edits(
    cues: tuple[Cue, ...], edit_set: EditSet, config: CueConfig | None = None
) -> tuple[tuple[Cue, ...], list[int]]:
    """Returns the corrected cues and the indices that actually changed.

    An edit naming a cue that is not there is ignored rather than fatal, and an edit whose
    text matches what the cue already says counts as no change, so re-approving without
    typing anything leaves the burn cached.
    """
    changed: list[int] = []
    out: list[Cue] = []
    for cue in cues:
        wanted = edit_set.texts.get(cue.index)
        if wanted is None:
            out.append(cue)
            continue
        edited = relayout(cue, wanted, config)
        if edited is not cue:
            changed.append(cue.index)
        out.append(edited)
    return tuple(out), changed


# --------------------------------------------------------------------------------------
# What the editor shows
# --------------------------------------------------------------------------------------


def cue_report(cue: Cue, config: CueConfig | None = None) -> dict[str, Any]:
    """One cue as the editor needs it: the clock, the layout, and why it is flagged.

    `lint_cues` on a single cue is deliberate. It is the same function the CLI's `lint`
    command runs and the same one the run summary reports, so a cue marked in the window
    is a cue the file will be reported for, with the identical wording. Running it over a
    one-cue tuple skips only the overlap check, which the editor cannot cause: it never
    moves a timestamp.
    """
    cfg = config or CueConfig()
    return {
        "index": cue.index,
        "start": round(cue.start, 3),
        "end": round(cue.end, 3),
        "duration": round(cue.duration, 3),
        # Over the visible characters, for the same reason `lint` measures those: markup
        # is weight, not width.
        "cps": round(display_len(cue.text) / cue.duration, 1) if cue.duration > 0 else None,
        "chars": display_len(cue.text),
        "text": cue.text,
        "lines": list(cue.lines),
        # Per line, so the editor can put the number that decides the verdict next to the
        # line it is about. `display_len` and not `len`, because markup is weight, not width.
        "line_widths": [display_len(line) for line in cue.lines],
        "problems": lint_cues((cue,), cfg),
    }


def cue_reports(cues: tuple[Cue, ...], config: CueConfig | None = None) -> list[dict[str, Any]]:
    return [cue_report(cue, config) for cue in cues]
