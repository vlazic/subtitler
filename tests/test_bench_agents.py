"""Reference adjudication: what feeds it, what it accepts, and what it writes.

No model runs here, and none has to: the module deliberately has no client in it. What is
under test is the part that decides which transcripts get a vote, the schema that a response
has to satisfy before it can become a reference, and the merge, which must be deterministic
because a benchmark's ground truth cannot depend on the order two files landed in.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from subtitler.bench import agents
from subtitler.model import Cue
from subtitler.render import write_srt

CLIP = "gozba-sample"


def cell(cell_id: str, *, engine: str, denoise: str = "none", **overrides) -> dict:
    record = {
        "cell_id": cell_id,
        "clip_id": CLIP,
        "denoise": denoise,
        "engine_requested": engine,
        "ok": True,
        "fix": False,
        "hallucination": {"prompt_echo_n": 0},
    }
    return {**record, **overrides}


def cues(*texts: str, step: float = 5.0) -> tuple[Cue, ...]:
    return tuple(
        Cue(index=i + 1, start=i * step, end=(i + 1) * step, lines=(text,))
        for i, text in enumerate(texts)
    )


def run_dir_with(tmp_path: Path, records: list[dict], transcripts: dict[str, tuple[Cue, ...]]):
    run_dir = tmp_path / "2026-08-04T00-00-00Z"
    (run_dir / "transcripts").mkdir(parents=True)
    (run_dir / "results.json").write_text(
        json.dumps({"results": records}, ensure_ascii=False), encoding="utf-8"
    )
    for cell_id, track in transcripts.items():
        write_srt(run_dir / "transcripts" / f"{cell_id}.srt", track)
    return run_dir


def planned(tmp_path: Path) -> Path:
    """A run with two engines that disagree in the second window, planned and written."""
    records = [
        cell("a", engine="faster-whisper"),
        cell("b", engine="groq"),
    ]
    run_dir = run_dir_with(
        tmp_path,
        records,
        {
            "a": cues("Misao lokove filozofije", "pomenuli smo državne organe", step=20.0),
            "b": cues("Misao Lokove filozofije", "povenuli smo državne gane", step=20.0),
        },
    )
    agents.write_plan(run_dir, agents.plan(run_dir))
    return run_dir


def adjudication(clip: str = CLIP, **overrides) -> dict:
    payload = {
        "role": agents.ADJUDICATOR,
        "clip": clip,
        "windows": [
            {"index": 0, "text": "Misao Lokove filozofije"},
            {"index": 1, "text": "pomenuli smo državne organe"},
        ],
        "spans": [
            {
                "start": 10.0,
                "end": 20.0,
                "chosen": "pomenuli smo",
                "candidates": [
                    {"source": "faster-whisper", "text": "pomenuli smo"},
                    {"source": "groq", "text": "povenuli smo"},
                ],
                "reason": "povenuli is not a Serbian word",
                "confidence": "medium",
            }
        ],
        "notes": "",
    }
    return {**payload, **overrides}


def critique(**overrides) -> dict:
    payload = {
        "role": agents.CRITIC,
        "findings": [
            {
                "clip": CLIP,
                "start": 0.0,
                "end": 10.0,
                "issue": "grammar_corrected",
                "reference_text": "Misao Lokove filozofije",
                "candidates": ["Misao lokove filozofije"],
                "recommendation": "check whether the speaker capitalised anything at all",
                "severity": "low",
            }
        ],
        "verdict": "Usable, weakest on proper nouns.",
    }
    return {**payload, **overrides}


def write_output(run_dir: Path, task_id: str, data) -> None:
    path = run_dir / "agents" / "outputs" / f"{task_id}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        data if isinstance(data, str) else json.dumps(data, ensure_ascii=False), encoding="utf-8"
    )


class TestSelectSources:
    def test_one_cell_per_engine_preferring_no_denoiser(self):
        payload = {
            "results": [
                cell("fw-arnndn", engine="faster-whisper", denoise="arnndn"),
                cell("fw-none", engine="faster-whisper", denoise="none"),
                cell("groq-afftdn", engine="groq", denoise="afftdn"),
            ]
        }
        chosen = agents.select_sources(payload, CLIP)
        assert [s.cell_id for s in chosen] == ["fw-none", "groq-afftdn"]

    def test_a_corrected_cell_never_votes(self):
        """A --fix cell is already an LLM rewrite; the reference must not agree with it."""
        payload = {
            "results": [
                cell("fw-fix", engine="faster-whisper", fix=True),
                cell("groq", engine="groq"),
            ]
        }
        assert [s.cell_id for s in agents.select_sources(payload, CLIP)] == ["groq"]

    def test_a_prompt_echoing_cell_never_votes(self):
        payload = {
            "results": [
                cell("fw", engine="faster-whisper", hallucination={"prompt_echo_n": 7}),
                cell("groq", engine="groq"),
            ]
        }
        assert [s.cell_id for s in agents.select_sources(payload, CLIP)] == ["groq"]

    def test_a_failed_cell_and_another_clip_are_both_ignored(self):
        payload = {
            "results": [
                cell("dead", engine="groq-turbo", ok=False),
                cell("elsewhere", engine="groq", clip_id="other"),
                cell("fw", engine="faster-whisper"),
            ]
        }
        assert [s.cell_id for s in agents.select_sources(payload, CLIP)] == ["fw"]


class TestAlign:
    def test_cues_land_in_the_window_their_start_falls_in(self):
        windows = agents.align(
            {"a": cues("prvo", "drugo", step=20.0), "b": cues("prvo b", "drugo b", step=20.0)},
            window_s=15.0,
        )
        assert [w.index for w in windows] == [0, 1]
        assert windows[0].texts == {"a": "prvo", "b": "prvo b"}
        assert windows[1].texts == {"a": "drugo", "b": "drugo b"}

    def test_two_cues_in_one_window_are_joined_in_time_order(self):
        windows = agents.align({"a": cues("prvo", "drugo", step=2.0)}, window_s=15.0)
        assert windows[0].texts == {"a": "prvo drugo"}

    def test_a_window_nobody_spoke_in_is_dropped(self):
        track = (
            Cue(index=1, start=0.0, end=1.0, lines=("prvo",)),
            Cue(index=2, start=40.0, end=41.0, lines=("kasnije",)),
        )
        windows = agents.align({"a": track}, window_s=15.0)
        assert [w.index for w in windows] == [0, 2]

    def test_a_source_that_said_nothing_in_a_window_is_absent_from_it(self):
        windows = agents.align(
            {"a": cues("prvo"), "b": (Cue(index=1, start=20.0, end=21.0, lines=("posle",)),)},
            window_s=15.0,
        )
        assert windows[0].texts == {"a": "prvo"}

    def test_a_zero_window_is_refused_rather_than_dividing_by_it(self):
        with pytest.raises(ValueError):
            agents.align({"a": cues("prvo")}, window_s=0.0)


class TestPlan:
    def test_the_manifest_names_a_role_an_input_and_an_output_per_task(self, tmp_path):
        run_dir = planned(tmp_path)
        manifest = json.loads((run_dir / "agents/agent-tasks.json").read_text())
        roles = [t["role"] for t in manifest["tasks"]]
        assert roles == [agents.ADJUDICATOR, agents.CRITIC]
        for task in manifest["tasks"]:
            assert task["task_id"] and task["prompt"] and task["output"]
            assert (run_dir / task["prompt"]).exists()

    def test_the_aligned_input_shows_every_engine_side_by_side(self, tmp_path):
        run_dir = planned(tmp_path)
        text = (run_dir / f"agents/inputs/adjudicate-{CLIP}.md").read_text()
        assert "povenuli smo državne gane" in text
        assert "pomenuli smo državne organe" in text
        assert "## window 0" in text

    def test_the_critic_starts_blocked_because_there_is_nothing_to_criticise(self, tmp_path):
        run_dir = planned(tmp_path)
        manifest = json.loads((run_dir / "agents/agent-tasks.json").read_text())
        critic = next(t for t in manifest["tasks"] if t["role"] == agents.CRITIC)
        assert critic["status"] == "blocked"
        assert not (run_dir / critic["input"]).exists()

    def test_a_clip_with_one_engine_is_not_adjudicated(self, tmp_path):
        """One opinion is not a consensus, and scoring an engine against itself is a zero."""
        run_dir = run_dir_with(tmp_path, [cell("only", engine="groq")], {"only": cues("sam")})
        manifest = agents.plan(run_dir)
        assert manifest["tasks"] == []

    def test_a_directory_that_is_not_a_run_says_so(self, tmp_path):
        with pytest.raises(ValueError, match=r"results\.json"):
            agents.plan(tmp_path)


class TestValidateAdjudication:
    def test_a_well_formed_response_has_no_errors(self):
        assert (
            agents.validate_adjudication(adjudication(), clip_id=CLIP, window_indices=[0, 1]) == []
        )

    def test_a_missing_window_is_an_error(self):
        data = adjudication(windows=[{"index": 0, "text": "prvo"}])
        errors = agents.validate_adjudication(data, clip_id=CLIP, window_indices=[0, 1])
        assert any("missing [1]" in e for e in errors)

    def test_the_wrong_clip_is_an_error(self):
        errors = agents.validate_adjudication(
            adjudication(clip="other"), clip_id=CLIP, window_indices=[0, 1]
        )
        assert any("clip must be" in e for e in errors)

    def test_a_span_without_candidates_is_an_error(self):
        data = adjudication(
            spans=[
                {
                    "start": 1.0,
                    "end": 2.0,
                    "chosen": "x",
                    "candidates": [],
                    "reason": "y",
                    "confidence": "low",
                }
            ]
        )
        errors = agents.validate_adjudication(data, clip_id=CLIP, window_indices=[0, 1])
        assert any("candidates" in e for e in errors)

    def test_an_invented_confidence_level_is_an_error(self):
        data = adjudication()
        data["spans"][0]["confidence"] = "certain"
        errors = agents.validate_adjudication(data, clip_id=CLIP, window_indices=[0, 1])
        assert any("confidence" in e for e in errors)

    def test_no_spans_at_all_is_allowed(self):
        data = adjudication(spans=[])
        assert agents.validate_adjudication(data, clip_id=CLIP, window_indices=[0, 1]) == []

    def test_a_json_array_is_not_an_adjudication(self):
        assert agents.validate_adjudication([], clip_id=CLIP, window_indices=[0])


class TestValidateCritique:
    def test_a_well_formed_critique_has_no_errors(self):
        assert agents.validate_critique(critique(), clips=[CLIP]) == []

    def test_an_empty_findings_list_is_a_real_answer(self):
        assert agents.validate_critique(critique(findings=[]), clips=[CLIP]) == []

    def test_a_finding_about_a_clip_that_is_not_in_the_run_is_an_error(self):
        data = critique()
        data["findings"][0]["clip"] = "nowhere"
        assert any("clip must be" in e for e in agents.validate_critique(data, clips=[CLIP]))

    def test_an_invented_issue_type_is_an_error(self):
        data = critique()
        data["findings"][0]["issue"] = "vibes"
        assert any("issue" in e for e in agents.validate_critique(data, clips=[CLIP]))

    def test_a_critique_without_a_verdict_is_an_error(self):
        assert any(
            "verdict" in e for e in agents.validate_critique(critique(verdict=" "), clips=[CLIP])
        )


class TestMerge:
    def test_a_valid_adjudication_becomes_a_reference_with_its_provenance(self, tmp_path):
        run_dir = planned(tmp_path)
        write_output(run_dir, f"adjudicate-{CLIP}", adjudication())
        references = tmp_path / "references"

        summary = agents.merge(run_dir, references=references, log=lambda _m: None)

        assert summary["references"] == [f"{CLIP}.txt"]
        text = (references / f"{CLIP}.txt").read_text()
        assert text == "Misao Lokove filozofije\npomenuli smo državne organe\n"
        meta = json.loads((references / f"{CLIP}.meta.json").read_text())
        assert meta["adjudicated"] is True
        assert meta["human_verified"] is False
        assert meta["engines"] == ["faster-whisper", "groq"]
        assert meta["engine_cells"] == ["a", "b"]
        assert meta["spans_flagged"] == 1
        assert "cannot hear the audio" in meta["note"]

    def test_no_response_yet_is_not_a_failure(self, tmp_path):
        run_dir = planned(tmp_path)
        summary = agents.merge(run_dir, references=tmp_path / "refs", log=lambda _m: None)
        assert summary["failed"] == []
        assert f"adjudicate-{CLIP}" in summary["pending"]
        assert not (tmp_path / "refs" / f"{CLIP}.txt").exists()

    def test_a_malformed_response_is_retried_once_and_then_fails(self, tmp_path):
        run_dir = planned(tmp_path)
        references = tmp_path / "references"
        write_output(run_dir, f"adjudicate-{CLIP}", "{not json at all")

        first = agents.merge(run_dir, references=references, log=lambda _m: None)
        assert first["failed"] == []
        task = json.loads((run_dir / "agents/agent-tasks.json").read_text())["tasks"][0]
        assert (task["status"], task["attempts"]) == ("retry", 1)

        write_output(run_dir, f"adjudicate-{CLIP}", adjudication(windows=[]))
        second = agents.merge(run_dir, references=references, log=lambda _m: None)
        assert second["failed"] == [f"adjudicate-{CLIP}"]
        assert not (references / f"{CLIP}.txt").exists()

    def test_merging_twice_over_one_bad_response_does_not_burn_the_retry(self, tmp_path):
        """Otherwise an operator loses the retry to their own second `--merge`."""
        run_dir = planned(tmp_path)
        write_output(run_dir, f"adjudicate-{CLIP}", "{not json")
        for _ in range(3):
            summary = agents.merge(run_dir, references=tmp_path / "refs", log=lambda _m: None)
        assert summary["failed"] == []
        assert summary["retry"] == [f"adjudicate-{CLIP}"]

    def test_a_retry_that_validates_recovers_the_task(self, tmp_path):
        run_dir = planned(tmp_path)
        references = tmp_path / "references"
        write_output(run_dir, f"adjudicate-{CLIP}", "{not json")
        agents.merge(run_dir, references=references, log=lambda _m: None)
        write_output(run_dir, f"adjudicate-{CLIP}", adjudication())
        summary = agents.merge(run_dir, references=references, log=lambda _m: None)
        assert summary["failed"] == [] and summary["retry"] == []
        assert (references / f"{CLIP}.txt").exists()

    def test_the_critics_input_is_written_once_every_clip_is_adjudicated(self, tmp_path):
        run_dir = planned(tmp_path)
        write_output(run_dir, f"adjudicate-{CLIP}", adjudication())
        agents.merge(run_dir, references=tmp_path / "refs", log=lambda _m: None)
        text = (run_dir / "agents/inputs/critique-references.md").read_text()
        assert "pomenuli smo državne organe" in text
        assert "povenuli smo državne gane" in text

    def test_the_critic_adds_to_the_queue_and_never_to_the_reference(self, tmp_path):
        run_dir = planned(tmp_path)
        references = tmp_path / "references"
        write_output(run_dir, f"adjudicate-{CLIP}", adjudication())
        agents.merge(run_dir, references=references, log=lambda _m: None)
        before = (references / f"{CLIP}.txt").read_text()

        write_output(run_dir, "critique-references", critique())
        summary = agents.merge(run_dir, references=references, log=lambda _m: None)

        assert summary["critic_findings"] == 1
        assert (references / f"{CLIP}.txt").read_text() == before
        queue = (references / agents.REVIEW_QUEUE).read_text()
        assert "critic: grammar_corrected" in queue
        assert "Usable, weakest on proper nouns." in queue
        meta = json.loads((references / f"{CLIP}.meta.json").read_text())
        assert meta["critic_findings"] == 1

    def test_the_review_queue_lists_every_span_with_a_timestamp(self, tmp_path):
        run_dir = planned(tmp_path)
        references = tmp_path / "references"
        write_output(run_dir, f"adjudicate-{CLIP}", adjudication())
        agents.merge(run_dir, references=references, log=lambda _m: None)
        queue = (references / agents.REVIEW_QUEUE).read_text()
        assert "| 00:10 |" in queue
        assert "povenuli smo" in queue and "**pomenuli smo**" in queue
        assert "human_verified" in queue

    def test_the_merge_is_byte_identical_when_run_again(self, tmp_path):
        run_dir = planned(tmp_path)
        references = tmp_path / "references"
        write_output(run_dir, f"adjudicate-{CLIP}", adjudication())
        write_output(run_dir, "critique-references", critique())
        agents.merge(run_dir, references=references, log=lambda _m: None)
        first = [p.read_bytes() for p in sorted(references.iterdir())]
        agents.merge(run_dir, references=references, log=lambda _m: None)
        assert [p.read_bytes() for p in sorted(references.iterdir())] == first

    def test_merging_without_a_manifest_says_what_to_run_first(self, tmp_path):
        with pytest.raises(ValueError, match="bench agents"):
            agents.merge(tmp_path, references=tmp_path / "refs", log=lambda _m: None)


class TestReferenceText:
    def test_windows_are_joined_in_index_order_whatever_order_they_arrived_in(self):
        text = agents.reference_text([{"index": 1, "text": "drugo"}, {"index": 0, "text": "prvo"}])
        assert text == "prvo\ndrugo\n"

    def test_a_silent_window_contributes_no_line(self):
        text = agents.reference_text([{"index": 0, "text": "prvo"}, {"index": 1, "text": "  "}])
        assert text == "prvo\n"


class TestWiring:
    def test_agents_without_a_manifest_exits_nonzero_rather_than_tracebacking(self, tmp_path):
        from subtitler.cli import main

        assert main(["bench", "agents", str(tmp_path), "--out", str(tmp_path)]) == 1

    def test_the_plan_and_merge_round_trip_through_the_cli(self, tmp_path, capsys):
        from subtitler.cli import main

        run_dir = run_dir_with(
            tmp_path,
            [cell("a", engine="faster-whisper"), cell("b", engine="groq")],
            {"a": cues("prvo"), "b": cues("prvo b")},
        )
        references = tmp_path / "references"
        assert main(["bench", "agents", str(run_dir), "--references", str(references)]) == 0
        write_output(
            run_dir,
            f"adjudicate-{CLIP}",
            adjudication(windows=[{"index": 0, "text": "prvo"}], spans=[]),
        )
        assert (
            main(["bench", "agents", str(run_dir), "--references", str(references), "--merge"]) == 0
        )
        assert (references / f"{CLIP}.txt").read_text() == "prvo\n"
