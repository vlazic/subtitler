from __future__ import annotations

from itertools import pairwise

import pytest

from subtitler.cues import CueConfig, lint_cues, segments_to_cues, wrap_text
from subtitler.model import Cue, Segment, Word, synthesize_words


def seg(start: float, end: float, text: str) -> Segment:
    return Segment(start=start, end=end, text=text)


class TestWrap:
    def test_short_text_stays_on_one_line(self) -> None:
        assert wrap_text("zdravo svima", max_line=42, max_lines=2) == ("zdravo svima",)

    def test_respects_the_character_limit(self) -> None:
        text = "Misao Lokove filozofije, ukratko izraženo, sastoji se u ovome."
        lines = wrap_text(text, max_line=42, max_lines=2)
        assert all(len(line) <= 42 for line in lines)
        assert " ".join(lines) == text

    def test_balances_two_lines(self) -> None:
        """A 40/4 split satisfies the character limit and still reads badly."""
        text = "a" * 38 + " bb cc"
        first, second = wrap_text(text, max_line=42, max_lines=2)
        assert abs(len(first) - len(second)) < 38

    def test_never_loses_words_when_it_fits(self) -> None:
        text = "jedan dva tri četiri pet šest sedam osam"
        assert " ".join(wrap_text(text, max_line=20, max_lines=3)) == text

    def test_empty_input(self) -> None:
        assert wrap_text("   ", max_line=42, max_lines=2) == ()


class TestSynthesizeWords:
    def test_covers_the_whole_segment(self) -> None:
        words = synthesize_words(seg(10.0, 14.0, "jedan dva tri"))
        assert words[0].start == pytest.approx(10.0)
        assert words[-1].end == pytest.approx(14.0)

    def test_is_monotonic(self) -> None:
        words = synthesize_words(seg(0.0, 5.0, "a bb ccc dddd"))
        assert len(words) == 4
        for a, b in pairwise(words):
            assert a.end <= b.start + 1e-9

    def test_longer_words_get_more_time(self) -> None:
        short, long = synthesize_words(seg(0.0, 10.0, "a dddddddddd"))
        assert (long.end - long.start) > (short.end - short.start)

    def test_empty_text(self) -> None:
        assert synthesize_words(seg(0.0, 1.0, "   ")) == ()


class TestSegmentsToCues:
    def test_one_cue_per_segment(self) -> None:
        cues = segments_to_cues((seg(0, 2, "prvi"), seg(2, 4, "drugi")))
        assert [c.text for c in cues] == ["prvi", "drugi"]

    def test_indices_are_sequential(self) -> None:
        cues = segments_to_cues((seg(0, 2, "a"), seg(2, 4, "b"), seg(4, 6, "c")))
        assert [c.index for c in cues] == [1, 2, 3]

    def test_blank_segments_are_dropped(self) -> None:
        assert segments_to_cues((seg(0, 2, "   "),)) == ()

    def test_timings_are_preserved(self) -> None:
        (cue,) = segments_to_cues((seg(1.25, 3.75, "zdravo"),))
        assert (cue.start, cue.end) == (1.25, 3.75)

    def test_existing_word_timings_are_kept(self) -> None:
        words = (Word(0.0, 0.5, "prvi"), Word(0.5, 1.0, "drugi"))
        segment = Segment(start=0.0, end=1.0, text="prvi drugi", words=words)
        assert segments_to_cues((segment,))[0].text == "prvi drugi"


class TestLint:
    cfg = CueConfig()

    def _cue(self, start: float, end: float, *lines: str) -> Cue:
        return Cue(index=1, start=start, end=end, lines=lines)

    def test_clean_cue_has_no_violations(self) -> None:
        assert lint_cues((self._cue(0.0, 3.0, "kratak red"),), self.cfg) == []

    def test_flags_a_long_line(self) -> None:
        problems = lint_cues((self._cue(0.0, 6.0, "x" * 60),), self.cfg)
        assert any("chars (max 42)" in p for p in problems)

    def test_flags_too_many_lines(self) -> None:
        problems = lint_cues((self._cue(0.0, 6.0, "a", "b", "c"),), self.cfg)
        assert any("3 lines" in p for p in problems)

    def test_flags_a_short_cue(self) -> None:
        problems = lint_cues((self._cue(0.0, 0.4, "brzo"),), self.cfg)
        assert any("under the 1.0s minimum" in p for p in problems)

    def test_flags_a_long_cue(self) -> None:
        problems = lint_cues((self._cue(0.0, 25.0, "dugo"),), self.cfg)
        assert any("exceeds the 7.0s maximum" in p for p in problems)

    def test_flags_reading_speed(self) -> None:
        problems = lint_cues((self._cue(0.0, 1.0, "x" * 40),), self.cfg)
        assert any("chars/sec" in p for p in problems)

    def test_flags_overlap(self) -> None:
        cues = (
            Cue(index=1, start=0.0, end=3.0, lines=("prvi",)),
            Cue(index=2, start=2.0, end=5.0, lines=("drugi",)),
        )
        assert any("overlaps" in p for p in lint_cues(cues, self.cfg))
