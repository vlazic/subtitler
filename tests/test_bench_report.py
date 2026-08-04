"""The generated report.

`report.render` is a pure function of the payload, so these are string assertions on a
hand-built dict. That is the point of keeping it pure: the document that a human reads and
draws a conclusion from is testable without a GPU, a model or a reference transcript.

Most of what is asserted here is the caveats. A benchmark report's failure mode is not a
wrong number, it is a right number read as if it meant something it does not.
"""

from __future__ import annotations

from subtitler.bench import report


def cell(**overrides):
    base = {
        "cell_id": "clip__none__faster-whisper__large-v3__nofix",
        "clip_id": "clip",
        "denoise": "none",
        "engine": "faster-whisper",
        "engine_requested": "faster-whisper",
        "fix": False,
        "ok": True,
        "rtf": 0.05,
        "wall_s": 12.3,
        "peak_rss_mb": 2048.0,
        "reference_score": None,
        "cue_stats": {"count": 40, "mean_cps": 14.2, "p95_cps": 19.9},
        "hallucination": {
            "longest_repeat_n": 0,
            "longest_repeat_text": "",
            "repetition_collapsed": 0,
            "silence_dropped": 1,
            "filler_hits": {},
        },
    }
    return {**base, **overrides}


def payload(**overrides):
    base = {
        "created_utc": "2026-08-04T10:00:00+00:00",
        "git": {"sha": "abc1234", "branch": "main", "dirty": False},
        "config": {
            "clips": ["fixtures/clip.mp3"],
            "denoisers": ["none"],
            "engines": ["faster-whisper"],
            "model": "large-v3",
            "device": "cuda",
            "batch_size": 0,
            "cues": {
                "max_line": 42,
                "max_lines": 2,
                "min_dur": 1.0,
                "max_dur": 7.0,
                "max_cps": 20.0,
            },
        },
        "references": {"clip": {"status": "absent", "human_verified": False}},
        "results": [cell()],
    }
    return {**base, **overrides}


class TestCaveats:
    def test_a_missing_reference_is_stated_before_any_number(self):
        text = report.render(payload())
        assert "No reference transcript for clip" in text
        assert text.index("cannot answer") < text.index("Leaderboard")

    def test_without_a_reference_the_leaderboard_says_it_is_sorted_by_speed(self):
        text = report.render(payload())
        assert "Leaderboard (by speed: no reference exists)" in text

    def test_an_unverified_reference_is_called_provisional(self):
        text = report.render(
            payload(
                references={"clip": {"status": "present", "human_verified": False}},
                results=[cell(reference_score={"wer": 0.12, "wer_folded": 0.09, "cer": 0.05})],
            )
        )
        assert "not human-verified" in text
        assert "agreement between models, not correctness" in text
        assert f"12.0{report.PROVISIONAL}" in text

    def test_a_verified_reference_carries_no_marker(self):
        text = report.render(
            payload(
                references={"clip": {"status": "present", "human_verified": True}},
                results=[cell(reference_score={"wer": 0.12, "wer_folded": 0.09, "cer": 0.05})],
            )
        )
        assert f"12.0{report.PROVISIONAL}" not in text
        assert "| 12.0 |" in text

    def test_a_dirty_tree_is_flagged_in_the_provenance(self):
        text = report.render(payload(git={"sha": "abc", "branch": "main", "dirty": True}))
        assert "DIRTY, not reproducible" in text

    def test_a_restricted_cloud_engine_is_named_as_the_reason_criterion_4_is_unanswered(self):
        text = report.render(
            payload(
                results=[
                    cell(),
                    cell(
                        cell_id="clip__none__groq-turbo__large-v3__nofix",
                        ok=False,
                        engine_requested="groq-turbo",
                        error="EngineUnavailable: organization_restricted",
                    ),
                ]
            )
        )
        assert "organization_restricted" in text
        assert "criterion 4" in text
        assert "**unanswered**" in text

    def test_no_cloud_cell_at_all_still_says_criterion_4_is_untouched(self):
        assert "acceptance criterion 4" in report.render(payload())


class TestLeaderboard:
    def test_sorted_by_wer_when_a_reference_exists(self):
        rows = report.leaderboard_rows(
            payload(
                results=[
                    cell(cell_id="worse", reference_score={"wer": 0.4}),
                    cell(cell_id="better", reference_score={"wer": 0.2}),
                ]
            )
        )
        assert [r["cell_id"] for r in rows] == ["better", "worse"]

    def test_sorted_by_rtf_when_no_reference_exists(self):
        rows = report.leaderboard_rows(
            payload(results=[cell(cell_id="slow", rtf=0.9), cell(cell_id="fast", rtf=0.1)])
        )
        assert [r["cell_id"] for r in rows] == ["fast", "slow"]

    def test_failed_cells_are_not_ranked(self):
        rows = report.leaderboard_rows(payload(results=[cell(), cell(cell_id="dead", ok=False)]))
        assert [r["cell_id"] for r in rows] == ["clip__none__faster-whisper__large-v3__nofix"]

    def test_failed_cells_are_still_reported(self):
        text = report.render(
            payload(results=[cell(ok=False, error="EngineUnavailable: no weights")])
        )
        assert "Cells that did not run" in text
        assert "no weights" in text


class TestFixAxis:
    def test_the_section_is_absent_when_the_axis_was_not_run(self):
        text = report.render(payload())
        assert "## The `--fix` axis" not in text
        assert "The `--fix` axis was not run" in text

    def test_change_rate_is_labelled_as_change_not_accuracy(self):
        text = report.render(
            payload(
                results=[
                    cell(),
                    cell(
                        cell_id="clip__none__faster-whisper__large-v3__fix",
                        fix=True,
                        fix_change_rate=0.07,
                        fix_report={"changed_cues": 12, "model": "openai/gpt-4o"},
                    ),
                ]
            )
        )
        assert "how much the model rewrote, not whether the rewrite" in text
        assert "7.0" in text
        assert "| 12 |" in text

    def test_a_cell_that_reused_a_cached_transcript_is_marked_as_such(self):
        """Otherwise its wall clock reads as three seconds of transcription."""
        text = report.render(
            payload(results=[cell(cached_stages=["extract", "transcribe", "cues"])])
        )
        assert "extract,transcribe,cues" in text
        assert "is what it did **not** have to do" in text


class TestAlwaysPresent:
    def test_the_normalization_caveats_are_in_every_report(self):
        text = report.render(payload())
        assert "Digits and abbreviations are deliberately not normalized" in text

    def test_a_prompt_echo_is_called_out_above_the_table(self):
        text = report.render(
            payload(
                results=[
                    cell(
                        hallucination={
                            "longest_repeat_n": 0,
                            "longest_repeat_text": "",
                            "repetition_collapsed": 0,
                            "silence_dropped": 0,
                            "prompt_echo_n": 7,
                            "prompt_echo_text": "koristi ispravna imena za ljude knjige",
                            "filler_hits": {},
                        }
                    )
                ]
            )
        )
        assert "echoed the Serbian steering prompt" in text
        assert "koristi ispravna imena" in text

    def test_no_warning_when_nothing_echoed(self):
        assert "echoed the Serbian steering prompt" not in report.render(payload())

    def test_an_empty_run_still_renders(self):
        text = report.render(payload(results=[]))
        assert "# Benchmark run" in text
        assert "_(nothing to show)_" in text
