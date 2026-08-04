"""Benchmark metrics.

Split deliberately: the reference-free half is stdlib and runs everywhere, and the WER half
needs `jiwer` from the `bench` extra, which CI does not sync. Marking the whole file as
requiring jiwer would silently stop testing the half that guards the numbers most people
read, since a run with no reference reports nothing else.
"""

from __future__ import annotations

import pytest

from subtitler.bench import metrics
from subtitler.cues import CueConfig
from subtitler.model import Cue


def cue(index: int, start: float, end: float, *lines: str) -> Cue:
    return Cue(index=index, start=start, end=end, lines=tuple(lines))


class TestPercentile:
    def test_nearest_rank_picks_a_real_value(self):
        values = list(range(1, 21))  # 1..20
        assert metrics.percentile(values, 0.95) == 19

    def test_median(self):
        assert metrics.percentile([1, 2, 3], 0.5) == 2

    def test_empty(self):
        assert metrics.percentile([], 0.95) == 0.0

    def test_single_value(self):
        assert metrics.percentile([7.5], 0.95) == 7.5


class TestCueStats:
    def test_empty_cues(self):
        stats = metrics.cue_stats(())
        assert stats.count == 0
        assert stats.mean_cps == 0.0

    def test_counts_and_reading_speed(self):
        cues = (cue(1, 0.0, 2.0, "dvadeset slova ovde!"), cue(2, 2.0, 4.0, "deset"))
        stats = metrics.cue_stats(cues)
        assert stats.count == 2
        assert stats.mean_cps == pytest.approx((20 / 2 + 5 / 2) / 2)

    def test_violations_are_percentages_of_the_cue_count(self):
        cfg = CueConfig(max_line=10, max_lines=1, min_dur=1.0, max_dur=3.0, max_cps=5.0)
        cues = (
            cue(1, 0.0, 3.5, "a" * 20),  # over line, over duration, 5.7 cps
            cue(2, 3.5, 4.0, "ok"),  # under minimum duration
        )
        stats = metrics.cue_stats(cues, cfg)
        assert stats.over_line_pct == 50.0
        assert stats.over_dur_pct == 50.0
        assert stats.over_cps_pct == 50.0
        assert stats.under_min_dur_pct == 50.0

    def test_markup_is_not_width(self):
        """`display_len` is what lint measures, so `<b>` must not count as three columns."""
        cfg = CueConfig(max_line=10)
        stats = metrics.cue_stats((cue(1, 0.0, 2.0, "<b>kratko</b>"),), cfg)
        assert stats.over_line_pct == 0.0

    def test_a_zero_duration_cue_does_not_poison_the_mean(self):
        """`Cue.cps` is infinite at zero duration; an infinite mean would report nothing."""
        cues = (cue(1, 0.0, 0.0, "instant"), cue(2, 1.0, 2.0, "normalno"))
        stats = metrics.cue_stats(cues)
        assert stats.mean_cps == pytest.approx(8.0)


class TestLongestRepeatedNgram:
    def test_no_repeat(self):
        assert metrics.longest_repeated_ngram("svaka reč je drugačija ovde") == (0, "")

    def test_finds_the_repeated_phrase(self):
        length, text = metrics.longest_repeated_ngram("ne znam ne znam ne znam kraj recenice")
        assert length >= 4
        assert text.startswith("ne znam ne znam")

    def test_ignores_a_single_repeated_word(self):
        """One word repeated is Serbian, not a decoder loop. `min_n` is 2 for that reason."""
        assert metrics.longest_repeated_ngram("i tako i ovako") == (0, "")

    def test_is_normalized_before_matching(self):
        length, text = metrics.longest_repeated_ngram("Добар дан. Dobar dan.")
        assert (length, text) == (2, "dobar dan")

    def test_short_input(self):
        assert metrics.longest_repeated_ngram("dve reci") == (0, "")


class TestFillerHits:
    def test_counts_whole_tokens(self):
        assert metrics.filler_hits("Hvala. Hvala.") == {"hvala": 2}

    def test_does_not_fire_inside_a_longer_word(self):
        """`prevod` must not match `prevodilac`, or every translation lecture is a
        hallucination."""
        assert metrics.filler_hits("prevodilac je govorio") == {}

    def test_multiword_phrase(self):
        hits = metrics.filler_hits("hvala na gledanju")
        assert hits["hvala na gledanju"] == 1
        assert hits["hvala"] == 1

    def test_nothing_found(self):
        assert metrics.filler_hits("obican tekst bez ijedne poznate fraze") == {}


class TestHallucination:
    def test_carries_the_engine_counters_through(self):
        result = metrics.hallucination("tekst", repetition_collapsed=3, silence_dropped=2)
        assert result.repetition_collapsed == 3
        assert result.silence_dropped == 2

    def test_none_is_not_zero(self):
        """A cloud engine cannot run the silence gate. "Not measured" must stay distinct."""
        result = metrics.hallucination("tekst")
        assert result.repetition_collapsed is None
        assert result.silence_dropped is None

    def test_to_dict_is_json_ready(self):
        payload = metrics.hallucination("Hvala. Hvala.").to_dict()
        assert payload["filler_hits"] == {"hvala": 2}


class TestScore:
    """The WER half. Needs jiwer, which is in the `bench` extra only."""

    @pytest.fixture(autouse=True)
    def _needs_jiwer(self):
        pytest.importorskip("jiwer")

    def test_identical_text_scores_zero(self):
        result = metrics.score("dobar dan svete", "dobar dan svete")
        assert result.wer == 0.0
        assert result.cer == 0.0

    def test_scripts_and_punctuation_do_not_count_as_errors(self):
        """The whole point of normalizing both sides before scoring."""
        result = metrics.score("Ђорђе је дошао.", "Đorđe je došao!")
        assert result.wer == 0.0

    def test_one_wrong_word_in_four(self):
        result = metrics.score("jedan dva tri cetiri", "jedan dva pet cetiri")
        assert result.wer == pytest.approx(0.25)
        assert (result.substitutions, result.insertions, result.deletions) == (1, 0, 0)

    def test_deletion_and_insertion_are_reported_apart(self):
        result = metrics.score("jedan dva tri", "jedan dva tri cetiri")
        assert result.insertions == 1
        assert result.deletions == 0

    def test_folding_separates_a_missing_diacritic_from_a_wrong_word(self):
        """The single most useful number for Serbian.

        `caša` for `čaša` is a keyboard problem; `kaša` for `čaša` is a recognition
        problem. Unfolded they are the same substitution, folded only one survives.
        """
        diacritic_only = metrics.score("čaša je čista", "casa je cista")
        assert diacritic_only.wer == pytest.approx(2 / 3)
        assert diacritic_only.wer_folded == 0.0

        real_error = metrics.score("čaša je čista", "kaša je čista")
        assert real_error.wer == pytest.approx(1 / 3)
        assert real_error.wer_folded == pytest.approx(1 / 3)

    def test_word_counts_are_of_the_normalized_text(self):
        result = metrics.score("Добар дан!", "dobar dan")
        assert (result.reference_words, result.hypothesis_words) == (2, 2)

    def test_an_empty_reference_raises_instead_of_reporting_zero(self):
        with pytest.raises(ValueError, match="empty reference"):
            metrics.score("   ", "nešto")

    def test_an_empty_hypothesis_is_a_total_loss_not_a_crash(self):
        result = metrics.score("jedan dva tri", "")
        assert result.wer == 1.0
        assert result.deletions == 3
