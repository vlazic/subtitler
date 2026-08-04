from __future__ import annotations

from itertools import pairwise

import pytest

from subtitler.cues import (
    CLITICS,
    PREPOSITIONS,
    CueConfig,
    lint_cues,
    segments_to_cues,
    wrap_text,
)
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


# ======================================================================================
# The real splitter (Phase 4). Everything above tests the primitives; this tests the
# behaviour that makes a subtitle readable.
# ======================================================================================


def words_from(text: str, *, start: float = 0.0, per_word: float = 0.4, gap_at: dict | None = None):
    """Build word timings from a sentence, optionally inserting a pause before a word."""
    gap_at = gap_at or {}
    out = []
    cursor = start
    for i, token in enumerate(text.split()):
        cursor += gap_at.get(i, 0.0)
        out.append(Word(start=cursor, end=cursor + per_word, text=token))
        cursor += per_word
    return tuple(out)


def seg_with_words(text: str, **kw) -> Segment:
    words = words_from(text, **kw)
    return Segment(start=words[0].start, end=words[-1].end, text=text, words=words)


class TestSplitting:
    cfg = CueConfig()

    def test_a_long_segment_becomes_several_cues(self) -> None:
        """The whole point: one Whisper segment is not one readable cue."""
        text = (
            "Misao Lokove filozofije, ukratko izraženo, sastoji se u ovome. "
            "Da se opšta predstava, da se ono što je istinito, to jest, "
            "da se saznanje zasniva na iskustvu."
        )
        cues = segments_to_cues((seg_with_words(text),), self.cfg)
        assert len(cues) > 1
        assert lint_cues(cues, self.cfg) == []

    def test_prefers_a_sentence_boundary(self) -> None:
        text = "Prva rečenica je gotova. Druga rečenica počinje ovde i traje malo duže nego prva."
        cues = segments_to_cues((seg_with_words(text),), self.cfg)
        assert cues[0].text.endswith(".")

    def test_never_starts_a_cue_with_a_clitic(self) -> None:
        """A break in front of "je" or "se" reads as a stutter."""
        text = (
            "Ovo je jedna veoma duga rečenica koja se mora prelomiti negde "
            "jer je predugačka da stane u jedan jedini titl."
        )
        cues = segments_to_cues((seg_with_words(text),), self.cfg)
        for cue in cues:
            for line in cue.lines:
                assert line.split()[0].lower() not in CLITICS, line

    def test_never_strands_a_preposition_at_a_line_end(self) -> None:
        """Regression: "s / jedne strane" split the preposition from its noun, because a
        forbidden break shared rank 4 with ordinary breaks and won on balance."""
        text = (
            "Propisuje se kao put saznanja s jedne strane iskustvo i posmatranje, "
            "a s druge strane analiziranje i isticanje opštih odredaba."
        )
        cues = segments_to_cues((seg_with_words(text),), self.cfg)
        for cue in cues:
            for line in cue.lines[:-1]:
                assert line.split()[-1].lower() not in PREPOSITIONS, line

    def test_a_pause_decides_where_to_split_when_punctuation_is_absent(self) -> None:
        """A pause is a tiebreaker for WHERE to split, not a trigger to split.

        A cue that fits every rule is left alone even if it spans a silence; splitting it
        would create two cues where one was already readable.
        """
        first = "prva grupa reci ovde bez ikakve interpunkcije nikakve"
        second = "druga grupa reci dolazi tek posle jasne duge pauze ovde"
        text = f"{first} {second}"
        pause_index = len(first.split())

        cues = segments_to_cues((seg_with_words(text, gap_at={pause_index: 1.5}),), self.cfg)
        assert len(cues) > 1, "this text is long enough to require a split"
        # The boundary landed on the pause rather than mid-phrase.
        assert cues[1].text.startswith("druga grupa")

    def test_a_cue_that_already_fits_is_never_split(self) -> None:
        text = "kratka recenica koja lepo staje u jedan titl"
        assert len(segments_to_cues((seg_with_words(text),), self.cfg)) == 1

    def test_output_always_satisfies_the_length_rules(self) -> None:
        text = " ".join(f"rec{i}" for i in range(80))
        cues = segments_to_cues((seg_with_words(text),), self.cfg)
        for cue in cues:
            assert len(cue.lines) <= self.cfg.max_lines
            assert all(len(line) <= self.cfg.max_line for line in cue.lines)


class TestTiming:
    cfg = CueConfig()

    def test_a_short_cue_is_extended_into_the_following_silence(self) -> None:
        segments = (
            seg_with_words("Da.", start=0.0, per_word=0.3),
            seg_with_words("Sledeca recenica dolazi mnogo kasnije.", start=10.0),
        )
        cues = segments_to_cues(segments, self.cfg)
        assert cues[0].duration >= self.cfg.min_dur

    def test_extension_never_overlaps_the_next_cue(self) -> None:
        segments = (
            seg_with_words("Da.", start=0.0, per_word=0.3),
            seg_with_words("Odmah zatim.", start=0.5),
        )
        cues = segments_to_cues(segments, self.cfg)
        for a, b in pairwise(cues):
            assert a.end <= b.start + 1e-9

    def test_a_cue_is_never_shortened(self) -> None:
        text = "Ova recenica traje tacno onoliko koliko traje."
        (cue,) = segments_to_cues((seg_with_words(text, per_word=0.2),), self.cfg)
        assert cue.end >= 0.2 * len(text.split())

    def test_no_cue_exceeds_the_maximum_duration(self) -> None:
        text = " ".join(f"rec{i}" for i in range(40))
        cues = segments_to_cues((seg_with_words(text, per_word=1.0),), self.cfg)
        assert all(c.duration <= self.cfg.max_dur + 1e-6 for c in cues)


class TestMerging:
    def test_adjacent_scraps_are_merged(self) -> None:
        """A 0.3 second cue reading "Da." flashes past unread."""
        cfg = CueConfig()
        segments = (
            seg_with_words("Da.", start=0.0, per_word=0.3),
            seg_with_words("Naravno.", start=0.4, per_word=0.3),
        )
        cues = segments_to_cues(segments, cfg)
        assert len(cues) == 1
        assert "Da." in cues[0].text and "Naravno." in cues[0].text

    def test_a_distant_scrap_is_left_alone(self) -> None:
        cfg = CueConfig()
        segments = (
            seg_with_words("Da.", start=0.0, per_word=0.3),
            seg_with_words("Naravno.", start=30.0, per_word=0.3),
        )
        assert len(segments_to_cues(segments, cfg)) == 2


class TestAcceptanceCriterion:
    def test_real_transcript_produces_zero_violations(self) -> None:
        """The PRD's bar: every cue at most 2 lines of 42 chars, 1.0 to 7.0 seconds, at
        most 20 chars per second. Run against a real Groq response, not a synthetic one."""
        import json
        from pathlib import Path

        from subtitler.engines.base import TranscribeOptions
        from subtitler.engines.groq import parse_verbose_json

        fixture = Path(__file__).resolve().parents[1] / "fixtures" / "groq_verbose.json"
        transcript = parse_verbose_json(
            json.loads(fixture.read_text(encoding="utf-8")), opts=TranscribeOptions()
        )
        cues = segments_to_cues(transcript.segments, CueConfig())
        assert cues
        assert lint_cues(cues, CueConfig()) == []


class TestGolden:
    """The committed expected output.

    Any change to a splitting rule shows up here as a reviewable diff rather than as a
    silent shift in what the tool produces. Regenerate deliberately, never reflexively:
      uv run python -c "..." > tests/golden/groq_verbose.srt
    """

    def test_matches_the_committed_output(self) -> None:
        import json
        from pathlib import Path

        from subtitler.engines.base import TranscribeOptions
        from subtitler.engines.groq import parse_verbose_json
        from subtitler.render import render_srt

        root = Path(__file__).resolve().parents[1]
        raw = json.loads((root / "fixtures" / "groq_verbose.json").read_text(encoding="utf-8"))
        transcript = parse_verbose_json(raw, opts=TranscribeOptions())
        produced = render_srt(segments_to_cues(transcript.segments, CueConfig()))
        expected = (root / "tests" / "golden" / "groq_verbose.srt").read_text(encoding="utf-8")
        assert produced == expected
