"""ASS subtitle generation and the ffmpeg burn-in commands.

Two decisions drive this file.

**A real .ass file, not `force_style` on an SRT.** A style block can express outline,
shadow, box, margins and resolution-relative sizing; a `force_style` string is a fragile
one-liner that has to be re-escaped every time it changes.

**Filter paths are never escaped, they are avoided.** `-vf "ass=/Users/x/My Videos/a:b's.ass"`
passes through shell, filtergraph and filter-option parsing, each with its own rules. So
the subtitle file is written into a temp directory under the fixed ASCII name `subs.ass`,
the bundled fonts are copied in beside it, and ffmpeg runs with `cwd` set there against the
literal filter `ass=subs.ass:fontsdir=fonts`. Input and output paths stay as argv entries,
where only the shell would matter, and there is no shell.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

from subtitler.media import MediaError, run
from subtitler.model import Cue

ASSETS = Path(__file__).parent / "assets"
FONT_DIR = ASSETS / "fonts"
BUNDLED_FONT = "Noto Sans"

# ASS alignment is numpad-style: 2 is bottom-center, 8 is top-center.
_ALIGN_BOTTOM = 2
_ALIGN_TOP = 8


@dataclass(frozen=True, slots=True)
class Style:
    """Resolution-independent style. Sizes are fractions of the video height."""

    name: str
    font_frac: float
    outline_frac: float
    shadow_frac: float
    border_style: int  # 1 = outline + shadow, 3 = opaque box
    margin_v_frac: float
    margin_h_frac: float
    primary: str = "FFFFFFFF"  # RRGGBBAA, alpha 0xFF = fully opaque
    outline_colour: str = "000000FF"
    back_colour: str = "00000080"
    bold: bool = False


PRESETS: dict[str, Style] = {
    # Readable over arbitrary footage without covering it. The default.
    "outline": Style(
        name="outline",
        font_frac=0.052,
        outline_frac=0.0035,
        shadow_frac=0.0020,
        border_style=1,
        margin_v_frac=0.06,
        margin_h_frac=0.06,
        bold=True,
    ),
    # Best over busy or bright video, where an outline alone stops being enough.
    "box": Style(
        name="box",
        font_frac=0.046,
        outline_frac=0.0025,
        shadow_frac=0.0,
        border_style=3,
        margin_v_frac=0.06,
        margin_h_frac=0.08,
        back_colour="000000B4",
    ),
    "minimal": Style(
        name="minimal",
        font_frac=0.040,
        outline_frac=0.0018,
        shadow_frac=0.0,
        border_style=1,
        margin_v_frac=0.05,
        margin_h_frac=0.08,
    ),
}


def rgba_to_ass(rgba: str) -> str:
    """RRGGBBAA to ASS &HAABBGGRR.

    Two traps in one conversion: ASS orders the colour bytes backwards (BGR), and its alpha
    channel is inverted (00 is opaque, FF is transparent). Hand-written `force_style`
    strings get one or both wrong, which is why this is a tested function.
    """
    value = rgba.strip().lstrip("#")
    if len(value) == 6:
        value += "FF"
    if len(value) != 8:
        raise ValueError(f"expected RRGGBB or RRGGBBAA, got {rgba!r}")
    r, g, b, a = (value[i : i + 2] for i in (0, 2, 4, 6))
    inverted_alpha = f"{255 - int(a, 16):02X}"
    return f"&H{inverted_alpha}{b}{g}{r}".upper()


def ass_clock(seconds: float) -> str:
    """ASS uses H:MM:SS.cc with centisecond precision and a single-digit hour."""
    cs_total = round(max(seconds, 0.0) * 100)
    hours, cs_total = divmod(cs_total, 360_000)
    minutes, cs_total = divmod(cs_total, 6_000)
    secs, cs = divmod(cs_total, 100)
    return f"{hours:d}:{minutes:02d}:{secs:02d}.{cs:02d}"


def _escape_text(line: str) -> str:
    """ASS treats braces as override blocks and backslashes as escapes."""
    return line.replace("\\", "\\\\").replace("{", "\\{").replace("}", "\\}")


def build_ass(
    cues: tuple[Cue, ...],
    *,
    width: int,
    height: int,
    style: Style,
    font_name: str = BUNDLED_FONT,
    font_size: int | None = None,
    position: str = "bottom",
) -> str:
    font = font_size or max(12, round(height * style.font_frac))
    outline = max(1, round(height * style.outline_frac))
    shadow = max(0, round(height * style.shadow_frac))
    margin_v = max(10, round(height * style.margin_v_frac))
    margin_h = max(10, round(width * style.margin_h_frac))
    align = _ALIGN_TOP if position == "top" else _ALIGN_BOTTOM

    header = "\n".join(
        [
            "[Script Info]",
            "ScriptType: v4.00+",
            f"PlayResX: {width}",
            f"PlayResY: {height}",
            # WrapStyle 2 disables libass line wrapping entirely. cues.py already decided
            # every break and emits it as \N; letting libass re-wrap would silently undo
            # the reading-speed and line-balance work.
            "WrapStyle: 2",
            "ScaledBorderAndShadow: yes",
            "YCbCr Matrix: TV.709",
            "",
        ]
    )

    style_format = (
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, "
        "BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, "
        "BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding"
    )
    style_line = ",".join(
        str(v)
        for v in [
            "Style: Default",
            font_name,
            font,
            rgba_to_ass(style.primary),
            rgba_to_ass("0000FFFF"),
            rgba_to_ass(style.outline_colour),
            rgba_to_ass(style.back_colour),
            -1 if style.bold else 0,
            0,
            0,
            0,
            100,
            100,
            0,
            0,
            style.border_style,
            outline,
            shadow,
            align,
            margin_h,
            margin_h,
            margin_v,
            1,
        ]
    )

    events_format = (
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text"
    )
    dialogues = [
        "Dialogue: 0,{start},{end},Default,,0,0,0,,{text}".format(
            start=ass_clock(c.start),
            end=ass_clock(c.end),
            text="\\N".join(_escape_text(line) for line in c.lines),
        )
        for c in cues
    ]

    return "\n".join(
        [
            header,
            "[V4+ Styles]",
            style_format,
            style_line,
            "",
            "[Events]",
            events_format,
            *dialogues,
            "",
        ]
    )


# --------------------------------------------------------------------------------------
# Command builders. Pure functions, asserted on directly in tests.
# --------------------------------------------------------------------------------------

# Every option is named, including the filename (`f` is a documented alias for `filename`
# on 4.4 and 8.x alike).
#
# This is not cosmetic. When a filter is MISSING, ffmpeg given a positional argument
# reports "No option name near 'subs.ass:fontsdir=fonts'" rather than "No such filter:
# 'ass'", because it fails while parsing options before it ever resolves the name. That
# error reads like a syntax incompatibility and sends you off fixing the wrong thing. With
# every option named, a missing filter says so plainly.
_FILTER = "ass=f=subs.ass:fontsdir=fonts"


def burn_video_cmd(src: Path, dst: Path, *, crf: int = 20, preset: str = "medium") -> list[str]:
    """Burn onto real video. Video must be re-encoded; `-c:v copy` cannot rasterize text."""
    return [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-v",
        "error",
        "-i",
        str(src),
        "-vf",
        _FILTER,
        "-c:v",
        "libx264",
        "-preset",
        preset,
        "-crf",
        str(crf),
        # Without yuv420p the output may not play in QuickTime or Safari, and the target
        # user is on a Mac.
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        "-map",
        "0:v:0",
        "-map",
        "0:a:0",
        "-movflags",
        "+faststart",
        str(dst),
    ]


def burn_canvas_cmd(
    audio: Path,
    dst: Path,
    *,
    duration: float,
    width: int,
    height: int,
    color: str = "0x101010",
    fps: int = 25,
    crf: int = 23,
) -> list[str]:
    """Burn onto a generated canvas for audio-only input.

    Differences from the recipe this replaces: an explicit `d=` on the lavfi source and a
    matching `-t`, because `-shortest` against an infinite source is unreliable on ffmpeg
    4.x; `-tune stillimage`, which shrinks a static canvas by roughly an order of
    magnitude; `-pix_fmt yuv420p`; `+faststart`; and near-black rather than pure black,
    which avoids banding around the text outline.
    """
    return [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-v",
        "error",
        "-f",
        "lavfi",
        "-i",
        f"color=c={color}:s={width}x{height}:r={fps}:d={duration:.3f}",
        "-i",
        str(audio),
        "-vf",
        _FILTER,
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        str(crf),
        "-tune",
        "stillimage",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        "-t",
        f"{duration:.3f}",
        "-movflags",
        "+faststart",
        str(dst),
    ]


def soft_mux_cmd(src: Path, subs: Path, dst: Path, *, language: str = "srp") -> list[str]:
    """Mux a subtitle track without re-encoding. mov_text drops all styling."""
    codec = "ass" if dst.suffix.lower() == ".mkv" else "mov_text"
    return [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-v",
        "error",
        "-i",
        str(src),
        "-i",
        str(subs),
        "-c",
        "copy",
        "-c:s",
        codec,
        "-metadata:s:s:0",
        f"language={language}",
        str(dst),
    ]


def preview_cmd(
    src: Path,
    dst: Path,
    *,
    at: float,
    lavfi: str | None = None,
) -> list[str]:
    """One still frame with the subtitles rendered, for comparing styles by eye."""
    cmd = ["ffmpeg", "-y", "-hide_banner", "-v", "error"]
    if lavfi:
        # Audio-only source: there is no picture to seek into, so synthesize one.
        cmd += ["-f", "lavfi", "-i", lavfi]
    else:
        cmd += ["-ss", f"{at:.3f}", "-i", str(src)]
    cmd += ["-vf", _FILTER, "-frames:v", "1", str(dst)]
    return cmd


def even(value: int) -> int:
    """yuv420p requires even dimensions; an odd canvas size fails the encode outright."""
    return value - (value % 2)


# --------------------------------------------------------------------------------------
# Execution
# --------------------------------------------------------------------------------------


def preview(
    cues: tuple[Cue, ...],
    out_dir: Path,
    *,
    video: Path | None = None,
    audio: Path | None = None,
    width: int,
    height: int,
    canvas_color: str = "0x101010",
    presets: tuple[str, ...] = ("outline", "box", "minimal"),
    dry_run: bool = False,
) -> list[Path]:
    """Render one still per style preset, at a moment where a cue is actually on screen.

    A table of font sizes and outline widths tells you nothing about whether text is
    readable over your own footage. Three images do.
    """
    if not cues:
        raise MediaError("no cues to preview")

    width, height = even(width), even(height)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Pick the longest cue: the most text on screen is the hardest case for legibility,
    # and the widest margin for a seek landing slightly off.
    target = max(cues, key=lambda c: len(c.text))
    at = (target.start + target.end) / 2

    written: list[Path] = []
    for name in presets:
        if name not in PRESETS:
            raise MediaError(f"unknown style preset {name!r}; choose from {sorted(PRESETS)}")
        dst = out_dir / f"preview-{name}.png"
        # `-ss` before `-i` rebases timestamps to zero, so the cue is written starting at
        # zero and the seek lands on it regardless of where it sits in the timeline.
        shifted = Cue(index=1, start=0.0, end=target.duration, lines=target.lines)
        ass_text = build_ass(
            (shifted,), width=width, height=height, style=PRESETS[name], font_name=BUNDLED_FONT
        )

        with TemporaryDirectory(prefix="subtitler-preview-") as tmp:
            work = Path(tmp)
            (work / "subs.ass").write_text(ass_text, encoding="utf-8")
            shutil.copytree(FONT_DIR, work / "fonts", ignore=shutil.ignore_patterns("*.md"))
            if video is not None:
                cmd = preview_cmd(video.resolve(), dst.resolve(), at=at)
            else:
                cmd = preview_cmd(
                    Path("."),
                    dst.resolve(),
                    at=0.0,
                    lavfi=f"color=c={canvas_color}:s={width}x{height}:d=1",
                )
            run(cmd, cwd=work, dry_run=dry_run)
        written.append(dst)

    return written


def burn(
    cues: tuple[Cue, ...],
    dst: Path,
    *,
    video: Path | None = None,
    audio: Path | None = None,
    width: int,
    height: int,
    duration: float,
    style_preset: str = "outline",
    font_name: str | None = None,
    font_size: int | None = None,
    position: str = "bottom",
    canvas_color: str = "0x101010",
    fps: int = 25,
    dry_run: bool = False,
) -> Path:
    """Render `cues` onto `video`, or onto a generated canvas carrying `audio`."""
    if (video is None) == (audio is None):
        raise MediaError("burn() takes exactly one of video= or audio=")
    if style_preset not in PRESETS:
        raise MediaError(f"unknown style preset {style_preset!r}; choose from {sorted(PRESETS)}")

    width, height = even(width), even(height)
    ass_text = build_ass(
        cues,
        width=width,
        height=height,
        style=PRESETS[style_preset],
        font_name=font_name or BUNDLED_FONT,
        font_size=font_size,
        position=position,
    )

    dst.parent.mkdir(parents=True, exist_ok=True)

    with TemporaryDirectory(prefix="subtitler-burn-") as tmp:
        work = Path(tmp)
        (work / "subs.ass").write_text(ass_text, encoding="utf-8")
        # A local fonts/ dir keeps the filter argument a fixed ASCII literal and makes the
        # bundled font available without touching the system font cache.
        shutil.copytree(FONT_DIR, work / "fonts", ignore=shutil.ignore_patterns("*.md"))

        if video is not None:
            cmd = burn_video_cmd(video.resolve(), dst.resolve())
        else:
            cmd = burn_canvas_cmd(
                audio.resolve(),  # type: ignore[union-attr]
                dst.resolve(),
                duration=duration,
                width=width,
                height=height,
                color=canvas_color,
                fps=fps,
            )
        run(cmd, cwd=work, dry_run=dry_run)

    return dst
