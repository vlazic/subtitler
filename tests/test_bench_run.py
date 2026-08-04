"""The matrix runner.

Every test here injects a fake cell runner. Running the real one means loading a 3 GB model
per cell, which is what the benchmark is for and what a test suite must never do. What is
under test is the parts that decide *what* runs, in *what order*, and what is written down
afterwards, all of which are the things that quietly invalidate a result.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from subtitler.bench import run as bench
from subtitler.cues import CueConfig

FIXTURE = Path("fixtures/gozba-sample.mp3")


def fake_cell(spec: bench.CellSpec) -> dict:
    """A believable payload without a model behind it."""
    text = "dobar dan svete" + (" ispravljeno" if spec.fix else "")
    return {
        "ok": True,
        "error": "",
        "wall_s": 1.0,
        "peak_rss_mb": 100.0,
        "engine": spec.engine,
        "cached_stages": [],
        "audio_s": 10.0,
        "decode_s": 0.5,
        "rtf": 0.05,
        "language": "sr",
        "model": spec.model,
        "model_revision": "abc123",
        "engine_params": {"repetition_collapsed": 0, "silence_dropped": 1},
        "segments": 1,
        "fix_report": {"changed": 1} if spec.fix else None,
        "text": text,
        "cues": [{"index": 1, "start": 0.0, "end": 2.0, "lines": [text]}],
        "lint_violations": 0,
    }


def config(tmp_path: Path, **overrides) -> bench.BenchConfig:
    base = {
        "clips": (FIXTURE,),
        "denoisers": ("none",),
        "engines": ("faster-whisper",),
        "out_root": tmp_path / "results",
        "references": tmp_path / "references",
        "work": tmp_path / "work",
        "allow_dirty": True,
        "cues": CueConfig(),
    }
    return bench.BenchConfig(**{**base, **overrides})


class TestCellSpec:
    def test_cell_id_is_readable_and_filesystem_safe(self):
        spec = bench.CellSpec(clip=Path("a/gozba-sample.mp3"), denoise="afftdn")
        assert spec.cell_id == "gozba-sample__afftdn__faster-whisper__large-v3__nofix"
        assert "/" not in spec.cell_id

    def test_the_fix_axis_is_in_the_id(self):
        spec = bench.CellSpec(clip=Path("x.mp3"), fix=True)
        assert spec.cell_id.endswith("__fix")

    def test_batch_size_is_in_the_id_because_it_changes_the_transcript(self):
        """Batched decoding drops the steering prompt, so it is a different transcript."""
        spec = bench.CellSpec(clip=Path("x.mp3"), batch_size=16)
        assert "__b16__" in spec.cell_id


class TestExpand:
    def test_clip_outermost_then_denoiser_then_engine(self):
        """The ordering the shared stage cache depends on.

        One slot per stage means `denoise.wav` is overwritten whenever the preset changes,
        so any nesting that alternates denoisers re-runs the denoiser for every engine.
        """
        cfg = bench.BenchConfig(
            clips=(Path("a.mp3"), Path("b.mp3")),
            denoisers=("none", "afftdn"),
            engines=("faster-whisper", "mlx"),
        )
        order = [(c.clip.name, c.denoise, c.engine) for c in bench.expand(cfg)]
        assert order[:4] == [
            ("a.mp3", "none", "faster-whisper"),
            ("a.mp3", "none", "mlx"),
            ("a.mp3", "afftdn", "faster-whisper"),
            ("a.mp3", "afftdn", "mlx"),
        ]
        assert order[4][0] == "b.mp3"

    def test_the_fix_axis_doubles_every_cell(self):
        cfg = bench.BenchConfig(clips=(Path("a.mp3"),), denoisers=("none",), fix_axis=True)
        cells = bench.expand(cfg)
        assert [c.fix for c in cells] == [False, True]

    def test_the_corrected_cell_follows_its_own_baseline(self):
        """So it hits the transcript the uncorrected cell just cached."""
        cfg = bench.BenchConfig(clips=(Path("a.mp3"),), denoisers=("none", "afftdn"), fix_axis=True)
        cells = bench.expand(cfg)
        assert [(c.denoise, c.fix) for c in cells] == [
            ("none", False),
            ("none", True),
            ("afftdn", False),
            ("afftdn", True),
        ]


class TestResolveClips:
    def test_an_explicit_file(self, tmp_path):
        clip = tmp_path / "one.mp3"
        clip.write_bytes(b"x")
        assert bench.resolve_clips(str(clip), root=tmp_path) == (clip,)

    def test_a_comma_separated_list(self, tmp_path):
        a, b = tmp_path / "a.mp3", tmp_path / "b.m4a"
        a.write_bytes(b"x")
        b.write_bytes(b"x")
        assert bench.resolve_clips(f"{a},{b}", root=tmp_path) == (a, b)

    def test_a_directory_takes_the_media_in_it(self, tmp_path):
        (tmp_path / "a.mp3").write_bytes(b"x")
        (tmp_path / "notes.txt").write_text("not media")
        assert bench.resolve_clips(str(tmp_path), root=tmp_path) == (tmp_path / "a.mp3",)

    def test_an_empty_clips_directory_falls_back_to_the_fixtures(self, tmp_path):
        """A fresh checkout has no clips: the interesting ones are too big to commit."""
        (tmp_path / "benchmarks" / "clips").mkdir(parents=True)
        clips = bench.resolve_clips(None, root=tmp_path)
        assert [c.name for c in clips] == ["gozba-sample.mp3", "uvod-u-pravo.m4a"]

    def test_a_missing_file_is_an_error_not_an_empty_matrix(self, tmp_path):
        with pytest.raises(ValueError, match="clip not found"):
            bench.resolve_clips(str(tmp_path / "nope.mp3"), root=tmp_path)

    def test_a_directory_with_no_media_is_an_error(self, tmp_path):
        (tmp_path / "notes.txt").write_text("x")
        with pytest.raises(ValueError, match="no media files"):
            bench.resolve_clips(str(tmp_path), root=tmp_path)


class TestReferences:
    def test_absent_reference_reads_as_none(self, tmp_path):
        assert bench.load_reference("clip", tmp_path) is None

    def test_an_empty_reference_file_is_not_a_reference(self, tmp_path):
        (tmp_path / "clip.txt").write_text("   \n")
        assert bench.load_reference("clip", tmp_path) is None

    def test_meta_for_an_absent_reference_says_so(self, tmp_path):
        meta = bench.reference_meta("clip", tmp_path)
        assert meta["status"] == "absent"
        assert meta["human_verified"] is False
        assert "Phase 8" in meta["note"]

    def test_meta_never_claims_a_reference_was_verified(self, tmp_path):
        """The harness cannot know whether a human read it, so it only carries it forward."""
        (tmp_path / "clip.txt").write_text("dobar dan")
        assert bench.reference_meta("clip", tmp_path)["human_verified"] is False

    def test_meta_carries_forward_what_the_file_already_claimed(self, tmp_path):
        (tmp_path / "clip.txt").write_text("dobar dan")
        (tmp_path / "clip.meta.json").write_text(
            json.dumps({"human_verified": True, "source": "adjudicated by hand"})
        )
        meta = bench.reference_meta("clip", tmp_path)
        assert meta["human_verified"] is True
        assert meta["source"] == "adjudicated by hand"

    def test_unreadable_meta_does_not_take_the_run_down(self, tmp_path):
        (tmp_path / "clip.txt").write_text("dobar dan")
        (tmp_path / "clip.meta.json").write_text("{not json")
        assert bench.reference_meta("clip", tmp_path)["human_verified"] is False


class TestGitState:
    def test_a_non_repository_reports_no_sha_rather_than_raising(self, tmp_path):
        state = bench.git_state(tmp_path)
        assert state["sha"] is None
        assert state["dirty"] is None


class TestRunMatrix:
    def test_writes_the_whole_run_directory(self, tmp_path):
        run_dir = bench.run_matrix(
            config(tmp_path),
            repo=tmp_path,
            log=lambda _m: None,
            runner=fake_cell,
            env_collector=lambda: {"host": "fake"},
        )
        assert (run_dir / "results.json").exists()
        assert (run_dir / "env.json").exists()
        assert (run_dir / "report.md").exists()
        assert (run_dir / "transcripts").is_dir()

    def test_keeps_every_hypothesis_so_a_metric_can_be_recomputed(self, tmp_path):
        run_dir = bench.run_matrix(
            config(tmp_path),
            repo=tmp_path,
            log=lambda _m: None,
            runner=fake_cell,
            env_collector=dict,
        )
        kept = sorted(p.name for p in (run_dir / "transcripts").iterdir())
        assert kept == [
            "gozba-sample__none__faster-whisper__large-v3__nofix.srt",
            "gozba-sample__none__faster-whisper__large-v3__nofix.txt",
        ]

    def test_records_the_git_sha(self, tmp_path):
        run_dir = bench.run_matrix(
            config(tmp_path),
            repo=Path.cwd(),
            log=lambda _m: None,
            runner=fake_cell,
            env_collector=dict,
        )
        payload = json.loads((run_dir / "results.json").read_text())
        assert payload["git"]["sha"]

    def test_refuses_a_dirty_tree_without_allow_dirty(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            bench, "git_state", lambda _repo: {"sha": "abc", "dirty": True, "dirty_paths": ["x.py"]}
        )
        with pytest.raises(ValueError, match="working tree is dirty"):
            bench.run_matrix(
                config(tmp_path, allow_dirty=False),
                repo=tmp_path,
                log=lambda _m: None,
                runner=fake_cell,
            )

    def test_writes_a_reference_meta_for_every_clip(self, tmp_path):
        bench.run_matrix(
            config(tmp_path),
            repo=tmp_path,
            log=lambda _m: None,
            runner=fake_cell,
            env_collector=dict,
        )
        meta = json.loads((tmp_path / "references" / "gozba-sample.meta.json").read_text())
        assert meta["status"] == "absent"

    def test_a_failed_cell_is_recorded_rather_than_aborting_the_matrix(self, tmp_path):
        def half_dead(spec):
            if spec.denoise == "afftdn":
                return {"ok": False, "error": "EngineUnavailable: organization_restricted"}
            return fake_cell(spec)

        run_dir = bench.run_matrix(
            config(tmp_path, denoisers=("none", "afftdn")),
            repo=tmp_path,
            log=lambda _m: None,
            runner=half_dead,
            env_collector=dict,
        )
        payload = json.loads((run_dir / "results.json").read_text())
        assert [r["ok"] for r in payload["results"]] == [True, False]
        assert "organization_restricted" in (run_dir / "report.md").read_text()

    def test_reference_free_metrics_are_present_without_a_reference(self, tmp_path):
        run_dir = bench.run_matrix(
            config(tmp_path),
            repo=tmp_path,
            log=lambda _m: None,
            runner=fake_cell,
            env_collector=dict,
        )
        record = json.loads((run_dir / "results.json").read_text())["results"][0]
        assert record["reference_score"] is None
        assert record["cue_stats"]["count"] == 1
        assert record["hallucination"]["silence_dropped"] == 1

    def test_a_reference_produces_a_wer(self, tmp_path):
        pytest.importorskip("jiwer")
        (tmp_path / "references").mkdir(parents=True)
        (tmp_path / "references" / "gozba-sample.txt").write_text("Dobar dan, svete!")
        run_dir = bench.run_matrix(
            config(tmp_path),
            repo=tmp_path,
            log=lambda _m: None,
            runner=fake_cell,
            env_collector=dict,
        )
        record = json.loads((run_dir / "results.json").read_text())["results"][0]
        assert record["reference_score"]["wer"] == 0.0


class TestRescore:
    def test_a_reference_added_later_scores_a_run_that_already_happened(self, tmp_path):
        """The Phase 8 seam, and the reason every hypothesis is kept on disk."""
        pytest.importorskip("jiwer")
        run_dir = bench.run_matrix(
            config(tmp_path),
            repo=tmp_path,
            log=lambda _m: None,
            runner=fake_cell,
            env_collector=dict,
        )
        assert (
            json.loads((run_dir / "results.json").read_text())["results"][0]["reference_score"]
            is None
        )

        (tmp_path / "references" / "gozba-sample.txt").write_text("dobar dan svetu")
        bench.rescore(run_dir, references=tmp_path / "references", log=lambda _m: None)

        record = json.loads((run_dir / "results.json").read_text())["results"][0]
        assert record["reference_score"]["wer"] == pytest.approx(1 / 3)

    def test_a_hand_corrected_transcript_on_disk_wins_over_the_json_copy(self, tmp_path):
        pytest.importorskip("jiwer")
        run_dir = bench.run_matrix(
            config(tmp_path),
            repo=tmp_path,
            log=lambda _m: None,
            runner=fake_cell,
            env_collector=dict,
        )
        transcript = (
            run_dir / "transcripts" / "gozba-sample__none__faster-whisper__large-v3__nofix.txt"
        )
        transcript.write_text("potpuno drugacije reci ovde\n")
        (tmp_path / "references" / "gozba-sample.txt").write_text("potpuno drugacije reci ovde")
        bench.rescore(run_dir, references=tmp_path / "references", log=lambda _m: None)
        record = json.loads((run_dir / "results.json").read_text())["results"][0]
        assert record["reference_score"]["wer"] == 0.0

    def test_a_directory_without_results_is_a_clear_error(self, tmp_path):
        with pytest.raises(ValueError, match="benchmark run directory"):
            bench.rescore(tmp_path, references=tmp_path, log=lambda _m: None)


class TestFixDelta:
    def test_change_rate_is_measured_against_the_matching_uncorrected_cell(self, tmp_path):
        pytest.importorskip("jiwer")
        run_dir = bench.run_matrix(
            config(tmp_path, fix_axis=True),
            repo=tmp_path,
            log=lambda _m: None,
            runner=fake_cell,
            env_collector=dict,
        )
        records = json.loads((run_dir / "results.json").read_text())["results"]
        corrected = next(r for r in records if r["fix"])
        # The fake corrector appends one word to three, which is one insertion in three.
        assert corrected["fix_change_rate"] == pytest.approx(1 / 3)

    def test_an_uncorrected_cell_has_no_change_rate(self, tmp_path):
        run_dir = bench.run_matrix(
            config(tmp_path),
            repo=tmp_path,
            log=lambda _m: None,
            runner=fake_cell,
            env_collector=dict,
        )
        record = json.loads((run_dir / "results.json").read_text())["results"][0]
        assert "fix_change_rate" not in record


class TestParseAxis:
    def test_none_means_no_restriction(self):
        assert bench.parse_axis(None, valid=("a", "b"), label="thing") is None

    def test_a_comma_separated_subset(self):
        assert bench.parse_axis("a, b", valid=("a", "b", "c"), label="thing") == ("a", "b")

    def test_an_unknown_value_names_the_valid_ones(self):
        with pytest.raises(ValueError, match="unknown thing: z"):
            bench.parse_axis("z", valid=("a", "b"), label="thing")


class TestLatestRun:
    def test_none_when_there_are_no_runs(self, tmp_path):
        assert bench.latest_run(tmp_path) is None

    def test_the_newest_run_wins(self, tmp_path):
        for stamp in ("2026-08-04T09-00-00Z", "2026-08-04T16-00-00Z"):
            (tmp_path / stamp).mkdir()
            (tmp_path / stamp / "results.json").write_text("{}")
        assert bench.latest_run(tmp_path).name == "2026-08-04T16-00-00Z"

    def test_a_half_written_run_directory_is_skipped(self, tmp_path):
        (tmp_path / "2026-08-04T09-00-00Z").mkdir()
        (tmp_path / "2026-08-04T09-00-00Z" / "results.json").write_text("{}")
        (tmp_path / "2026-08-04T16-00-00Z").mkdir()  # crashed before writing results
        assert bench.latest_run(tmp_path).name == "2026-08-04T09-00-00Z"


class TestWiring:
    def test_the_bench_subcommand_is_handled_and_not_pending(self):
        from subtitler.cli import _HANDLERS, _PENDING, build_parser

        args = build_parser().parse_args(["bench", "run"])
        assert args.command == "bench"
        assert args.action == "run"
        assert "bench" in _HANDLERS
        assert "bench" not in _PENDING

    def test_the_clips_axis_defaults_to_resolution_at_runtime(self):
        """`--clips` has no path default: the fallback lives in `resolve_clips`."""
        from subtitler.cli import build_parser

        assert build_parser().parse_args(["bench", "run"]).clips is None

    def test_agents_is_still_phase_8(self, capsys):
        """The seam this phase deliberately stops at."""
        from subtitler.cli import main

        assert main(["bench", "agents"]) == 2
        assert "Phase 8" in capsys.readouterr().err


class TestRunDirName:
    def test_utc_iso_without_the_characters_a_filesystem_dislikes(self):
        from datetime import UTC, datetime

        name = bench.run_dir_name(datetime(2026, 8, 4, 16, 30, 12, tzinfo=UTC))
        assert name == "2026-08-04T16-30-12Z"
        assert ":" not in name

    def test_the_format_sorts_chronologically(self):
        from datetime import UTC, datetime

        early = bench.run_dir_name(datetime(2026, 8, 4, 9, 0, 0, tzinfo=UTC))
        late = bench.run_dir_name(datetime(2026, 8, 4, 16, 0, 0, tzinfo=UTC))
        assert sorted([late, early]) == [early, late]
