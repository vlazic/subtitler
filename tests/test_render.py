from __future__ import annotations

import pytest

from subtitler.model import Cue
from subtitler.render import (
    parse_clock,
    parse_subtitles,
    render_srt,
    render_vtt,
    srt_clock,
    validate_vtt,
    vtt_clock,
)


def cue(index: int, start: float, end: float, *lines: str) -> Cue:
    return Cue(index=index, start=start, end=end, lines=lines)


class TestClocks:
    @pytest.mark.parametrize(
        ("seconds", "expected"),
        [
            (0.0, "00:00:00,000"),
            (1.5, "00:00:01,500"),
            (61.25, "00:01:01,250"),
            (3661.001, "01:01:01,001"),
        ],
    )
    def test_srt_clock(self, seconds: float, expected: str) -> None:
        assert srt_clock(seconds) == expected

    def test_rounding_never_emits_sixty_seconds(self) -> None:
        """The bug this project inherited: f"{59.9996:06.3f}" renders "60.000".

        That produces the invalid timestamp 00:00:60.000, which strict parsers reject.
        Going through integer milliseconds first carries into the minute correctly.
        """
        assert srt_clock(59.9996) == "00:01:00,000"
        assert vtt_clock(59.9996) == "00:01:00.000"
        assert srt_clock(3599.9999) == "01:00:00,000"

    def test_negative_clamps_to_zero(self) -> None:
        assert srt_clock(-1.0) == "00:00:00,000"

    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("00:00:01,500", 1.5),
            ("00:00:01.500", 1.5),
            ("01:01:01,001", 3661.001),
            ("01:01.500", 61.5),
        ],
    )
    def test_parse_clock(self, text: str, expected: float) -> None:
        assert parse_clock(text) == pytest.approx(expected)

    def test_clock_roundtrip(self) -> None:
        for value in (0.0, 0.001, 12.345, 59.999, 3661.5):
            assert parse_clock(srt_clock(value)) == pytest.approx(value, abs=0.001)


class TestVtt:
    def test_header_has_exactly_one_blank_line_after_it(self) -> None:
        """The old converter emitted two, by seeding the list with "WEBVTT\\n"
        and then joining with "\\n" while each cue also began with "\\n"."""
        text = render_vtt((cue(1, 0.0, 1.0, "zdravo"),))
        assert text.startswith("WEBVTT\n\n00:00:00.000 --> 00:00:01.000\nzdravo\n")
        assert "\n\n\n" not in text

    def test_validate_accepts_generated_output(self) -> None:
        cues = (cue(1, 0.0, 2.0, "prvi"), cue(2, 2.5, 4.0, "drugi"))
        assert validate_vtt(render_vtt(cues)) == []

    def test_validate_reports_structural_problems(self) -> None:
        problems = validate_vtt("00:00:01.000 --> 00:00:00.500\nnazad\n")
        assert any("WEBVTT" in p for p in problems)
        assert any("end is not after start" in p for p in problems)

    def test_validate_reports_overlap(self) -> None:
        text = render_vtt((cue(1, 0.0, 3.0, "prvi"), cue(2, 2.0, 4.0, "drugi")))
        assert any("overlaps" in p for p in validate_vtt(text))


class TestSrt:
    def test_cues_are_numbered_from_one(self) -> None:
        text = render_srt((cue(7, 0.0, 1.0, "a"), cue(9, 1.0, 2.0, "b")))
        assert text.splitlines()[0] == "1"
        assert "2\n00:00:01,000 --> 00:00:02,000\nb" in text

    def test_multiline_cue_is_preserved(self) -> None:
        text = render_srt((cue(1, 0.0, 2.0, "prvi red", "drugi red"),))
        assert "prvi red\ndrugi red" in text

    def test_empty_input_renders_empty(self) -> None:
        assert render_srt(()) == ""


class TestParsing:
    def test_roundtrip_srt(self) -> None:
        original = (cue(1, 0.0, 2.0, "prvi"), cue(2, 2.5, 4.25, "drugi", "red"))
        parsed = parse_subtitles(render_srt(original))
        assert [c.lines for c in parsed] == [c.lines for c in original]
        assert [c.start for c in parsed] == [c.start for c in original]

    def test_roundtrip_vtt(self) -> None:
        original = (cue(1, 0.0, 2.0, "prvi"), cue(2, 2.5, 4.25, "drugi"))
        parsed = parse_subtitles(render_vtt(original))
        assert [c.text for c in parsed] == ["prvi", "drugi"]

    def test_vtt_cue_settings_after_end_time_are_ignored(self) -> None:
        text = "WEBVTT\n\n00:00:00.000 --> 00:00:02.000 align:start position:10%\nzdravo\n"
        (parsed,) = parse_subtitles(text)
        assert parsed.end == pytest.approx(2.0)
        assert parsed.text == "zdravo"

    def test_serbian_diacritics_survive(self) -> None:
        original = (cue(1, 0.0, 2.0, "čćđšž ČĆĐŠŽ"),)
        assert parse_subtitles(render_srt(original))[0].text == "čćđšž ČĆĐŠŽ"
