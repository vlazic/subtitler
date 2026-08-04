"""Hand corrections: where they live, what re-wraps them, and when they go stale.

The two properties this file exists to nail down are the ones that are invisible when they
break. A correction that gets silently overwritten on the next run looks like the editor
never saved. A correction that gets applied to a *different* transcript looks like the
editor scrambled the subtitles. Both are cache-shaped bugs, so both are tested against the
real `StageCache` rather than against a mock of one.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from subtitler import cache as cache_mod
from subtitler import edits as edits_mod
from subtitler.cues import CueConfig, wrap_edited, wrap_text
from subtitler.model import Cue


def cue(index: int, start: float, end: float, *lines: str) -> Cue:
    return Cue(index=index, start=start, end=end, lines=lines)


class TestRewrapping:
    def test_a_corrected_cue_is_re_broken_by_wrap_words_not_the_greedy_wrapper(self) -> None:
        """The bug `CLAUDE.md` already records, now reachable from the editor too.

        Wrapping corrected text with the greedy `wrap_text` put the break after "da",
        leaving the clitic "se" to open line two. `cues.CLITICS` exists precisely to forbid
        that break, and only `wrap_words` (which `wrap_edited` goes through) consults it.
        Both wrappers satisfy the 42-character limit here, so a length assertion would pass
        on the wrong one: the break position is the thing under test.
        """
        text = "Da se opšta predstava, da se ono što je istinito,"

        greedy = wrap_text(text, max_line=42, max_lines=2)
        assert greedy[1].startswith("se ")  # the bug, still there in the greedy wrapper

        proper = wrap_edited(text, start=0.0, end=3.0)
        assert proper == ("Da se opšta predstava,", "da se ono što je istinito,")
        assert not proper[1].startswith("se ")

    def test_a_cue_nobody_touched_keeps_the_break_the_splitter_chose(self) -> None:
        """The splitter picked that break from real word timings, including the pauses
        between them. Nothing downstream has those, so re-wrapping an unchanged cue can
        only ever lose information."""
        original = cue(3, 1.0, 4.0, "Da se opšta predstava, da", "se ono što je istinito,")
        same_text = " ".join(original.lines)
        assert edits_mod.relayout(original, same_text) is original

    def test_whitespace_alone_is_not_a_correction(self) -> None:
        """A textarea hands back whatever the user's cursor did. Two spaces between words
        must not count as an edit and re-break a line that was fine."""
        original = cue(1, 0.0, 2.0, "Prvi red", "drugi red")
        assert edits_mod.relayout(original, "  Prvi red   drugi   red \n") is original

    def test_a_real_correction_comes_back_with_the_same_clock_and_index(self) -> None:
        """The editor may change what a cue says and nothing else. Reading speed is
        measured against the duration, so a mouse that could drag a timestamp would put the
        one number a human cannot judge by eye under the mouse."""
        original = cue(7, 12.5, 15.25, "Aristotel kaze")
        edited = edits_mod.relayout(original, "Aristotel kaže")
        assert (edited.index, edited.start, edited.end) == (7, 12.5, 15.25)
        assert edited.text == "Aristotel kaže"


class TestTheFileOnDisk:
    def test_a_blank_correction_is_refused_rather_than_deleting_the_cue(self) -> None:
        """Deleting one would renumber the rest, and every correction here is addressed by
        the index it was made against."""
        with pytest.raises(edits_mod.EditError, match="cue 4"):
            edits_mod.build("k1", {"4": "   "})

    def test_a_correction_with_no_base_key_is_refused(self) -> None:
        with pytest.raises(edits_mod.EditError):
            edits_mod.build("", {"1": "tekst"})

    def test_a_round_trip_through_the_file_preserves_the_corrections(self, tmp_path: Path) -> None:
        built = edits_mod.build("abc123", {"2": "drugi", "9": "deveti"})
        edits_mod.save(tmp_path, built)
        assert edits_mod.load(tmp_path) == built

    def test_no_file_at_all_is_the_only_thing_that_reads_as_no_corrections(
        self, tmp_path: Path
    ) -> None:
        """Deleting the file is how a user says "run without them", and the only way."""
        assert edits_mod.load(tmp_path) is None

    def test_an_unreadable_file_says_what_is_wrong_and_where(self, tmp_path: Path) -> None:
        """Regression: this used to read as "no corrections", so a trailing comma in the one
        file a human is invited to open produced a run that reported success and burned the
        uncorrected words in with nothing said. Hand-typed text is not reconstructible, so
        losing it silently is worse than stopping."""
        edits_mod.path_for(tmp_path).write_text(
            '{"schema_version": 1, "base_key": "k", "edits": [],}', encoding="utf-8"
        )
        with pytest.raises(edits_mod.EditFileError) as caught:
            edits_mod.load(tmp_path)
        message = str(caught.value)
        assert "not valid JSON" in message
        assert "line 1, column 52" in message  # where, not merely that
        assert edits_mod.EDITS_NAME in message  # and the way back

    def test_a_file_from_another_schema_names_both_versions(self, tmp_path: Path) -> None:
        edits_mod.path_for(tmp_path).write_text(
            json.dumps({"schema_version": 99, "base_key": "k", "edits": []}), encoding="utf-8"
        )
        with pytest.raises(edits_mod.EditFileError, match="schema_version 99"):
            edits_mod.load(tmp_path)

    def test_the_key_being_mistyped_is_not_read_as_an_empty_correction_set(
        self, tmp_path: Path
    ) -> None:
        """`"edit"` for `"edits"` is the typo this file invites, and the one that used to be
        indistinguishable from having made no corrections."""
        edits_mod.path_for(tmp_path).write_text(
            json.dumps({"schema_version": 1, "base_key": "k", "edit": [{"index": 1, "text": "a"}]}),
            encoding="utf-8",
        )
        with pytest.raises(edits_mod.EditFileError, match='no "edits" key'):
            edits_mod.load(tmp_path)

    def test_an_entry_that_names_no_cue_is_refused_rather_than_dropped(
        self, tmp_path: Path
    ) -> None:
        """A correction silently skipped is a correction lost, and the file still says it
        is there."""
        edits_mod.path_for(tmp_path).write_text(
            json.dumps({"schema_version": 1, "base_key": "k", "edits": [{"text": "a"}]}),
            encoding="utf-8",
        )
        with pytest.raises(edits_mod.EditFileError, match="which cue"):
            edits_mod.load(tmp_path)

    def test_corrections_with_no_base_key_are_refused(self, tmp_path: Path) -> None:
        edits_mod.path_for(tmp_path).write_text(
            json.dumps({"schema_version": 1, "edits": [{"index": 1, "text": "a"}]}),
            encoding="utf-8",
        )
        with pytest.raises(edits_mod.EditFileError, match="base_key"):
            edits_mod.load(tmp_path)

    def test_the_digest_changes_with_the_text_and_not_with_the_order(self) -> None:
        """It is the `edit` stage's only cache parameter, so it has to move when the words
        move and stay put when nothing but the dict insertion order did."""
        one = edits_mod.build("k", {"1": "prvi", "2": "drugi"})
        same = edits_mod.build("k", {"2": "drugi", "1": "prvi"})
        other = edits_mod.build("k", {"1": "prvi", "2": "treći"})
        assert one.digest() == same.digest()
        assert one.digest() != other.digest()


class TestApplying:
    CUES = (
        cue(1, 0.0, 2.0, "Prva replika"),
        cue(2, 2.2, 5.0, "Da se opšta predstava, da", "se ono što je istinito,"),
        cue(3, 5.2, 7.0, "Treća replika"),
    )

    def test_only_the_named_cue_changes(self) -> None:
        edited, changed = edits_mod.apply_edits(
            self.CUES, edits_mod.build("k", {"2": "Da se opšta predstava, da se ono što je tačno,"})
        )
        assert changed == [2]
        assert edited[0] is self.CUES[0]
        assert edited[2] is self.CUES[2]
        assert edited[1].lines[1].startswith("da ")

    def test_re_approving_without_typing_anything_counts_as_no_change(self) -> None:
        """Which is what leaves the burn cached: the `edit` stage's key must not move
        because the editor was opened and closed again."""
        _edited, changed = edits_mod.apply_edits(
            self.CUES, edits_mod.build("k", {"1": "Prva replika"})
        )
        assert changed == []

    def test_a_correction_naming_a_cue_that_is_gone_is_ignored_not_fatal(self) -> None:
        edited, changed = edits_mod.apply_edits(self.CUES, edits_mod.build("k", {"99": "nema"}))
        assert changed == []
        assert edited == self.CUES


class TestTheQualityReport:
    def test_a_cue_that_meets_the_bar_carries_no_notes(self) -> None:
        report = edits_mod.cue_report(cue(1, 0.0, 3.0, "Kratka i citljiva replika."))
        assert report["problems"] == []
        assert report["duration"] == 3.0

    def test_the_same_wording_lint_would_use(self) -> None:
        """The marks in the window and the violations in the run summary have to be the
        same sentences, or the user is told two different things about one file."""
        long_line = "a" * 60
        report = edits_mod.cue_report(cue(4, 0.0, 1.0, long_line))
        assert any("60 chars (max 42)" in p for p in report["problems"])
        assert any("chars/sec exceeds 20" in p for p in report["problems"])

    def test_markup_is_weight_not_width(self) -> None:
        """`<b>` is seven characters of nothing on a player, and `cues.display_len` is what
        `lint` measures, so the editor must not report a violation on a line that reads
        perfectly."""
        marked = "<b>" + "a" * 40 + "</b>"
        report = edits_mod.cue_report(cue(1, 0.0, 5.0, marked), CueConfig(max_cps=99.0))
        assert report["line_widths"] == [40]
        assert report["problems"] == []


def test_the_edit_stage_sits_between_the_text_and_the_burn() -> None:
    """`--force cues` has to take the corrections and the burn with it; `--force edit` has
    to leave the transcript alone."""
    order = cache_mod.STAGE_ORDER
    assert order.index("cues") < order.index("edit") < order.index("burn")
    assert cache_mod.invalidated_from("edit") == frozenset({"edit", "burn", "mux"})
    assert "transcribe" not in cache_mod.invalidated_from("edit")
