"""The page's controls to a `RunConfig`, and back to the command line that matches it.

The GUI adds no capability: it builds the same `RunConfig` that `cli._cmd_run` builds, so
this module is where every `subtitler run` flag has to be expressible or it is not in the
GUI at all. Keeping it pure (no HTTP, no threads, nothing but `exists()` on the paths the
user picked) is what lets `tests/test_gui.py` cover the whole option surface without a
browser.

`command_line()` exists for a second reason beyond being testable: the GUI prints the
equivalent command for every run, so a user who starts in the window can move to the
terminal, and so a bug report from the friend arrives as a command rather than a
description of which boxes were ticked.
"""

from __future__ import annotations

import shlex
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from subtitler import postedit
from subtitler.cues import CueConfig
from subtitler.gui.jobs import STAGES
from subtitler.pipeline import RunConfig

# The choice sets are duplicated nowhere: `/api/options` serves these to the page, and the
# validator below rejects anything outside them. A control the page renders is therefore a
# control this module accepts, by construction.
ENGINES: tuple[str, ...] = ("auto", "mlx", "faster-whisper", "groq", "groq-turbo")
DEVICES: tuple[str, ...] = ("auto", "cpu", "cuda")
DENOISERS: tuple[str, ...] = ("none", "afftdn", "arnndn", "anlmdn", "speech")
STYLE_PRESETS: tuple[str, ...] = ("outline", "box", "minimal")
MARKUP: tuple[str, ...] = ("strip", "html")
FORCE_STAGES: tuple[str, ...] = (
    "",
    "all",
    "extract",
    "denoise",
    "transcribe",
    "cues",
    "fix",
    "burn",
)

# Whisper supports far more, and `lang` is a free-text field in the page for that reason.
# These are the ones worth a click, with Serbian first because that is what this is tuned
# for. `auto` is offered last and labelled, since non-negotiable 2 says it is a trap.
LANGUAGES: tuple[tuple[str, str], ...] = (
    ("sr", "Serbian"),
    ("hr", "Croatian"),
    ("bs", "Bosnian"),
    ("sl", "Slovenian"),
    ("mk", "Macedonian"),
    ("en", "English"),
    ("de", "German"),
    ("fr", "French"),
    ("it", "Italian"),
    ("es", "Spanish"),
    ("ru", "Russian"),
    ("auto", "detect (not recommended)"),
)


class FormError(ValueError):
    """A rejected control, named, so the page can point at it instead of at the form."""

    def __init__(self, field: str, message: str) -> None:
        super().__init__(message)
        self.field = field
        self.message = message

    def to_dict(self) -> dict[str, str]:
        return {"field": self.field, "error": self.message}


# --------------------------------------------------------------------------------------
# Coercion. Everything arrives from JSON, so everything may be the wrong type or blank.
# --------------------------------------------------------------------------------------


def _text(payload: Mapping[str, Any], key: str, default: str = "") -> str:
    value = payload.get(key, default)
    if value is None:
        return ""
    return str(value).strip()


def _flag(payload: Mapping[str, Any], key: str, default: bool = False) -> bool:
    value = payload.get(key, default)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _choice(payload: Mapping[str, Any], key: str, default: str, choices: Sequence[str]) -> str:
    value = _text(payload, key) or default
    if value not in choices:
        raise FormError(
            key, f"{value!r} is not one of: {', '.join(c or '(none)' for c in choices)}"
        )
    return value


def _whole(payload: Mapping[str, Any], key: str, default: int, *, low: int, high: int) -> int:
    raw = _text(payload, key)
    if raw == "":
        return default
    try:
        value = int(float(raw))
    except ValueError as exc:
        raise FormError(key, f"{raw!r} is not a number") from exc
    if not low <= value <= high:
        raise FormError(key, f"must be between {low} and {high}")
    return value


def _number(
    payload: Mapping[str, Any], key: str, default: float, *, low: float, high: float
) -> float:
    raw = _text(payload, key)
    if raw == "":
        return default
    try:
        value = float(raw)
    except ValueError as exc:
        raise FormError(key, f"{raw!r} is not a number") from exc
    if not low <= value <= high:
        raise FormError(key, f"must be between {low} and {high}")
    return value


def _existing_file(payload: Mapping[str, Any], key: str, *, required: bool) -> Path | None:
    raw = _text(payload, key)
    if not raw:
        if required:
            raise FormError(key, "choose a file first")
        return None
    path = Path(raw).expanduser()
    if not path.exists():
        raise FormError(key, f"no such file: {path}")
    if path.is_dir():
        raise FormError(key, f"{path} is a folder, not a file")
    return path


# --------------------------------------------------------------------------------------
# The form itself
# --------------------------------------------------------------------------------------


def build_config(payload: Mapping[str, Any]) -> RunConfig:
    """Validate a page payload and return the `RunConfig` the pipeline will be handed.

    Raises `FormError` naming the offending control. Nothing is started, no directory is
    created and no file is written: a rejected form must cost the user nothing.
    """
    source = _existing_file(payload, "input", required=True)
    assert source is not None

    out_raw = _text(payload, "out_dir")
    out_dir: Path | None = None
    if out_raw:
        out_dir = Path(out_raw).expanduser()
        if out_dir.exists() and not out_dir.is_dir():
            raise FormError("out_dir", f"{out_dir} is a file, not a folder")

    prompt: str | None = _text(payload, "prompt") or None
    prompt_file = _existing_file(payload, "prompt_file", required=False)
    if prompt is None and prompt_file is not None:
        prompt = prompt_file.read_text(encoding="utf-8").strip()

    canvas = _text(payload, "canvas") or "1280x720"
    _check_canvas(canvas)

    min_dur = _number(payload, "min_dur", 1.0, low=0.1, high=60.0)
    max_dur = _number(payload, "max_dur", 7.0, low=0.2, high=120.0)
    if min_dur >= max_dur:
        raise FormError("min_dur", "the shortest cue must be shorter than the longest")

    fix = _fix_config(payload) if _flag(payload, "fix") else None

    return RunConfig(
        input=source,
        out_dir=out_dir,
        engine=_choice(payload, "engine", "auto", ENGINES),
        model=_text(payload, "model") or "large-v3",
        device=_choice(payload, "device", "auto", DEVICES),
        batch_size=_whole(payload, "batch_size", 0, low=0, high=64),
        language=_text(payload, "lang") or "sr",
        prompt=prompt,
        denoise=_choice(payload, "denoise", "none", DENOISERS),
        burn=_flag(payload, "burn", True),
        soft_mux=_flag(payload, "soft_mux"),
        srt_only=_flag(payload, "srt_only"),
        canvas=canvas,
        canvas_color=_text(payload, "canvas_color") or "0x101010",
        style_preset=_choice(payload, "style_preset", "outline", STYLE_PRESETS),
        font=_text(payload, "font") or None,
        font_size=_whole(payload, "font_size", 0, low=0, high=400) or None,
        cues=CueConfig(
            max_line=_whole(payload, "max_line", 42, low=10, high=120),
            max_lines=_whole(payload, "max_lines", 2, low=1, high=4),
            min_dur=min_dur,
            max_dur=max_dur,
            max_cps=_number(payload, "max_cps", 20.0, low=1.0, high=99.0),
        ),
        fix=fix,
        force=_choice(payload, "force", "", FORCE_STAGES) or None,
        dry_run=_flag(payload, "dry_run"),
        verbose=_whole(payload, "verbose", 0, low=0, high=3),
    )


def _check_canvas(spec: str) -> None:
    """Reject a bad canvas here rather than three ffmpeg calls into the run.

    `pipeline.parse_canvas` raises `MediaError` for the same input, but it does so after
    `probe` and `extract` have already run, which in the GUI reads as "it worked for a
    while and then broke".
    """
    width, _, height = spec.lower().partition("x")
    if not (width.isdigit() and height.isdigit()) or int(width) <= 0 or int(height) <= 0:
        raise FormError("canvas", f"{spec!r} is not a size like 1280x720")


def _fix_config(payload: Mapping[str, Any]) -> postedit.FixConfig:
    phrases = _existing_file(payload, "drop_intro_phrases", required=False)
    temperature_raw = _text(payload, "fix_temperature")
    return postedit.FixConfig(
        model=_text(payload, "fix_model") or postedit.DEFAULT_MODEL,
        prompt=_text(payload, "fix_prompt") or postedit.DEFAULT_PROMPT,
        batch_size=_whole(payload, "fix_batch", postedit.DEFAULT_BATCH_SIZE, low=1, high=500),
        workers=_whole(payload, "fix_workers", postedit.DEFAULT_WORKERS, low=1, high=32),
        # Left as None unless the user typed one: current Claude models reject
        # temperature/top_p/top_k with a 400, so "unset" is not the same as "0".
        temperature=(
            _number(payload, "fix_temperature", 0.0, low=0.0, high=2.0) if temperature_raw else None
        ),
        markup=_choice(payload, "fix_markup", "strip", MARKUP),
        drop_intro_phrases=phrases,
    )


# --------------------------------------------------------------------------------------
# The equivalent command line
# --------------------------------------------------------------------------------------

_DEFAULTS = RunConfig(input=Path("."))
_FIX_DEFAULTS = postedit.FixConfig()


def to_argv(cfg: RunConfig) -> list[str]:
    """`subtitler run ...` for this config, carrying only what differs from the defaults.

    A command that repeats every default is unreadable and teaches the reader nothing about
    which knob mattered.
    """
    argv = ["subtitler", "run", str(cfg.input)]

    def opt(flag: str, value: Any, default: Any) -> None:
        if value != default and value is not None:
            argv.extend([flag, str(value)])

    if cfg.out_dir is not None:
        argv.extend(["-o", str(cfg.out_dir)])
    opt("--engine", cfg.engine, _DEFAULTS.engine)
    opt("--model", cfg.model, _DEFAULTS.model)
    opt("--device", cfg.device, _DEFAULTS.device)
    opt("--batch-size", cfg.batch_size, _DEFAULTS.batch_size)
    opt("--lang", cfg.language, _DEFAULTS.language)
    opt("--prompt", cfg.prompt, _DEFAULTS.prompt)
    opt("--denoise", cfg.denoise, _DEFAULTS.denoise)
    if not cfg.burn:
        argv.append("--no-burn")
    if cfg.soft_mux:
        argv.append("--soft-mux")
    if cfg.srt_only:
        argv.append("--srt-only")
    opt("--canvas", cfg.canvas, _DEFAULTS.canvas)
    opt("--canvas-color", cfg.canvas_color, _DEFAULTS.canvas_color)
    opt("--style-preset", cfg.style_preset, _DEFAULTS.style_preset)
    opt("--font", cfg.font, _DEFAULTS.font)
    opt("--font-size", cfg.font_size, _DEFAULTS.font_size)
    opt("--max-line", cfg.cues.max_line, _DEFAULTS.cues.max_line)
    opt("--max-lines", cfg.cues.max_lines, _DEFAULTS.cues.max_lines)
    opt("--min-dur", cfg.cues.min_dur, _DEFAULTS.cues.min_dur)
    opt("--max-dur", cfg.cues.max_dur, _DEFAULTS.cues.max_dur)
    opt("--max-cps", cfg.cues.max_cps, _DEFAULTS.cues.max_cps)

    if cfg.fix is not None:
        argv.append("--fix")
        opt("--fix-model", cfg.fix.model, _FIX_DEFAULTS.model)
        opt("--fix-prompt", cfg.fix.prompt, _FIX_DEFAULTS.prompt)
        opt("--fix-batch", cfg.fix.batch_size, _FIX_DEFAULTS.batch_size)
        opt("--fix-workers", cfg.fix.workers, _FIX_DEFAULTS.workers)
        opt("--fix-temperature", cfg.fix.temperature, _FIX_DEFAULTS.temperature)
        opt("--fix-markup", cfg.fix.markup, _FIX_DEFAULTS.markup)
        opt("--drop-intro-phrases", cfg.fix.drop_intro_phrases, _FIX_DEFAULTS.drop_intro_phrases)

    opt("--force", cfg.force, _DEFAULTS.force)
    if cfg.dry_run:
        argv.append("--dry-run")
    if cfg.verbose:
        argv.append("-" + "v" * cfg.verbose)
    return argv


def command_line(cfg: RunConfig) -> str:
    return " ".join(shlex.quote(part) for part in to_argv(cfg))


# The three optional stages, and the flag that decides each. Kept next to the config that
# sets them, because a progress bar promising a stage the run will skip is a progress bar
# that always ends looking unfinished.
def stages_for(cfg: RunConfig) -> tuple[str, ...]:
    skip = set()
    if cfg.denoise == "none":
        skip.add("denoise")
    if cfg.fix is None:
        skip.add("fix")
    if cfg.srt_only or not cfg.burn:
        skip.add("burn")
    return tuple(stage for stage in STAGES if stage not in skip)


def options() -> dict[str, Any]:
    """Everything the page needs to render its controls, so the two cannot drift."""
    return {
        "engines": list(ENGINES),
        "devices": list(DEVICES),
        "denoisers": list(DENOISERS),
        "style_presets": list(STYLE_PRESETS),
        "markup": list(MARKUP),
        "force_stages": list(FORCE_STAGES),
        "languages": [{"code": code, "label": label} for code, label in LANGUAGES],
        "defaults": {
            "engine": _DEFAULTS.engine,
            "model": _DEFAULTS.model,
            "device": _DEFAULTS.device,
            "batch_size": _DEFAULTS.batch_size,
            "lang": _DEFAULTS.language,
            "denoise": _DEFAULTS.denoise,
            "burn": _DEFAULTS.burn,
            "soft_mux": _DEFAULTS.soft_mux,
            "srt_only": _DEFAULTS.srt_only,
            "canvas": _DEFAULTS.canvas,
            "canvas_color": _DEFAULTS.canvas_color,
            "style_preset": _DEFAULTS.style_preset,
            "max_line": _DEFAULTS.cues.max_line,
            "max_lines": _DEFAULTS.cues.max_lines,
            "min_dur": _DEFAULTS.cues.min_dur,
            "max_dur": _DEFAULTS.cues.max_dur,
            "max_cps": _DEFAULTS.cues.max_cps,
            "fix_model": _FIX_DEFAULTS.model,
            "fix_prompt": _FIX_DEFAULTS.prompt,
            "fix_batch": _FIX_DEFAULTS.batch_size,
            "fix_workers": _FIX_DEFAULTS.workers,
            "fix_markup": _FIX_DEFAULTS.markup,
        },
    }
