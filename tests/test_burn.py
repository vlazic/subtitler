"""Command construction and ASS generation.

These assert on argv lists rather than running ffmpeg, so they catch the flags that are
easy to lose in a refactor and that only fail on someone else's machine: yuv420p, the
explicit duration, the fixed filter literal.
"""

from __future__ import annotations

import pytest

from subtitler.burn import (
    PRESETS,
    ass_clock,
    build_ass,
    burn_canvas_cmd,
    burn_video_cmd,
    even,
    rgba_to_ass,
    soft_mux_cmd,
)
from subtitler.model import Cue


def cue(index: int, start: float, end: float, *lines: str) -> Cue:
    return Cue(index=index, start=start, end=end, lines=lines)


class TestColour:
    @pytest.mark.parametrize(
        ("rgba", "expected"),
        [
            ("FFFFFFFF", "&H00FFFFFF"),  # opaque white
            ("000000FF", "&H00000000"),  # opaque black
            ("FF0000FF", "&H000000FF"),  # opaque red: bytes reverse to BGR
            ("00000080", "&H7F000000"),  # half-transparent black
        ],
    )
    def test_rgba_to_ass(self, rgba: str, expected: str) -> None:
        """ASS is &HAABBGGRR: colour bytes reversed and alpha inverted. Both are easy to
        get wrong by hand, which is the whole reason this is a function."""
        assert rgba_to_ass(rgba) == expected

    def test_six_digit_input_is_treated_as_opaque(self) -> None:
        assert rgba_to_ass("FFFFFF") == "&H00FFFFFF"

    def test_bad_input_raises(self) -> None:
        with pytest.raises(ValueError, match="RRGGBB"):
            rgba_to_ass("xyz")


class TestAssClock:
    @pytest.mark.parametrize(
        ("seconds", "expected"),
        [(0.0, "0:00:00.00"), (1.5, "0:00:01.50"), (3661.004, "1:01:01.00")],
    )
    def test_format(self, seconds: float, expected: str) -> None:
        assert ass_clock(seconds) == expected


class TestAss:
    def _ass(self, **kwargs) -> str:
        return build_ass(
            (cue(1, 0.0, 2.0, "prvi red", "drugi red"),),
            width=1920,
            height=1080,
            style=PRESETS["outline"],
            **kwargs,
        )

    def test_wrapstyle_disables_libass_wrapping(self) -> None:
        """cues.py owns line layout. If libass re-wraps, the reading-speed and
        line-balance work is silently undone at render time."""
        assert "WrapStyle: 2" in self._ass()

    def test_playres_matches_the_video(self) -> None:
        text = self._ass()
        assert "PlayResX: 1920" in text
        assert "PlayResY: 1080" in text

    def test_line_breaks_become_backslash_n(self) -> None:
        assert "prvi red\\Ndrugi red" in self._ass()

    def test_braces_are_escaped(self) -> None:
        text = build_ass(
            (cue(1, 0.0, 1.0, "an {override} attempt"),),
            width=640,
            height=360,
            style=PRESETS["minimal"],
        )
        assert "\\{override\\}" in text

    def test_font_size_scales_with_height(self) -> None:
        small = build_ass((cue(1, 0, 1, "x"),), width=640, height=360, style=PRESETS["outline"])
        large = build_ass((cue(1, 0, 1, "x"),), width=3840, height=2160, style=PRESETS["outline"])
        assert _font_size(small) < _font_size(large)

    def test_explicit_font_size_wins(self) -> None:
        assert _font_size(self._ass(font_size=99)) == 99

    def test_top_position_uses_alignment_eight(self) -> None:
        assert _style_field(self._ass(position="top"), "Alignment") == "8"

    def test_bottom_position_is_the_default(self) -> None:
        assert _style_field(self._ass(), "Alignment") == "2"

    def test_box_preset_uses_border_style_three(self) -> None:
        text = build_ass((cue(1, 0, 1, "x"),), width=1280, height=720, style=PRESETS["box"])
        assert _style_field(text, "BorderStyle") == "3"

    def test_all_presets_generate(self) -> None:
        for name, style in PRESETS.items():
            text = build_ass((cue(1, 0, 1, "x"),), width=1280, height=720, style=style)
            assert "[Events]" in text, name


def _style_field(ass_text: str, field: str) -> str:
    """Read a named field off the Default style, indexing by the Format line.

    Positional indexing into the style line is exactly the kind of thing that silently
    reads the wrong column after someone adds a field.
    """
    fmt_line = next(line for line in ass_text.splitlines() if line.startswith("Format: Name,"))
    names = [n.strip() for n in fmt_line.removeprefix("Format:").split(",")]
    style_line = next(line for line in ass_text.splitlines() if line.startswith("Style: "))
    values = [v.strip() for v in style_line.removeprefix("Style:").split(",")]
    return values[names.index(field)]


def _font_size(ass_text: str) -> int:
    return int(_style_field(ass_text, "Fontsize"))


class TestCommands:
    def test_video_burn_reencodes_with_yuv420p(self, tmp_path) -> None:
        """Without yuv420p, QuickTime and Safari may refuse the file, and the target
        user is on a Mac."""
        cmd = burn_video_cmd(tmp_path / "in.mp4", tmp_path / "out.mp4")
        assert "-pix_fmt" in cmd and cmd[cmd.index("-pix_fmt") + 1] == "yuv420p"
        assert "libx264" in cmd
        assert "+faststart" in cmd

    def test_filter_is_a_fixed_ascii_literal(self, tmp_path) -> None:
        """Paths are never interpolated into the filtergraph: the subtitle file is written
        into a temp dir under a fixed name and ffmpeg runs with cwd set there."""
        for cmd in (
            burn_video_cmd(tmp_path / "in.mp4", tmp_path / "out.mp4"),
            burn_canvas_cmd(
                tmp_path / "a.wav", tmp_path / "o.mp4", duration=1.0, width=640, height=360
            ),
        ):
            assert cmd[cmd.index("-vf") + 1] == "ass=subs.ass:fontsdir=fonts"

    def test_canvas_passes_duration_twice_and_never_uses_shortest(self, tmp_path) -> None:
        """-shortest against an infinite lavfi source is unreliable on ffmpeg 4.x, which
        is what Ubuntu 22.04 ships."""
        cmd = burn_canvas_cmd(
            tmp_path / "a.wav", tmp_path / "o.mp4", duration=109.061, width=1280, height=720
        )
        assert "-shortest" not in cmd
        assert "d=109.061" in cmd[cmd.index("-i") + 1]
        assert cmd[cmd.index("-t") + 1] == "109.061"
        assert "stillimage" in cmd

    def test_soft_mux_picks_the_codec_from_the_container(self, tmp_path) -> None:
        mp4 = soft_mux_cmd(tmp_path / "i.mp4", tmp_path / "s.srt", tmp_path / "o.mp4")
        mkv = soft_mux_cmd(tmp_path / "i.mkv", tmp_path / "s.ass", tmp_path / "o.mkv")
        assert mp4[mp4.index("-c:s") + 1] == "mov_text"
        assert mkv[mkv.index("-c:s") + 1] == "ass"


class TestEven:
    @pytest.mark.parametrize(("value", "expected"), [(1080, 1080), (1081, 1080), (721, 720)])
    def test_rounds_down(self, value: int, expected: int) -> None:
        """yuv420p requires even dimensions; an odd canvas fails the encode outright."""
        assert even(value) == expected
