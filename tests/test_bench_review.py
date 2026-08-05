"""The human pass over a reference, driven with no audio and no terminal.

The listening cannot be tested and is not what is under test. What is: that the queue a
reviewer walks is the queue the adjudicator and its critic actually produced, that a
correction lands on exactly the words it was aimed at, that quitting halfway loses nothing,
and above all that `human_verified` cannot be raised while anything is still open. That flag
is the difference between every WER in this project being provisional and being real, so the
test that it does not flip early matters more than the ones that check it flips at all.

"Halfway" is not only `[q]`. `TestInterrupted` ends the session the way it really ends -- by
killing it between two questions -- and then runs a second one over the same directory,
because the only definition of "resumable" that means anything is that the state left behind
is a state the next session can read.

The real fixtures are used where the shape of the real data is the point: the two spans that
anchor to nothing and the one reading that occurs three times are properties of this
project's own reference, not of a contrived one.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from subtitler.bench import review

CLIP = "gozba-sample"
RUN = "2026-08-04T15-21-56Z"
REAL_RUN = Path("benchmarks/results") / RUN
REAL_REFS = Path("benchmarks/references")

real_data = pytest.mark.skipif(
    not (REAL_RUN / "agents" / "outputs" / f"adjudicate-{CLIP}.json").exists(),
    reason="the committed adjudication is not in this checkout",
)


# --------------------------------------------------------------------------- fixtures


def adjudication(clip: str = CLIP, *, windows: list[dict], spans: list[dict]) -> dict:
    return {"role": "ref-adjudicator", "clip": clip, "windows": windows, "spans": spans}


def span_json(start: float, chosen: str, **overrides) -> dict:
    record = {
        "start": start,
        "end": start + 15.0,
        "chosen": chosen,
        "candidates": [
            {"source": "faster-whisper", "text": chosen},
            {"source": "groq", "text": chosen.replace("a", "o")},
        ],
        "reason": "two engines against one",
        "confidence": "low",
    }
    return {**record, **overrides}


def build_run(tmp_path: Path, *, adjudications: list[dict], findings: list[dict] | None = None):
    """A run directory and a references directory holding what the adjudicator wrote."""
    run_dir = tmp_path / RUN
    outputs = run_dir / "agents" / "outputs"
    outputs.mkdir(parents=True)
    refs = tmp_path / "references"
    refs.mkdir()

    results = []
    for data in adjudications:
        clip = data["clip"]
        (outputs / f"adjudicate-{clip}.json").write_text(
            json.dumps(data, ensure_ascii=False), encoding="utf-8"
        )
        text = "\n".join(w["text"] for w in data["windows"] if w["text"].strip())
        (refs / f"{clip}.txt").write_text(text + "\n", encoding="utf-8")
        (refs / f"{clip}.meta.json").write_text(
            json.dumps(
                {"clip": clip, "status": "present", "human_verified": False, "note": "caveat"},
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        results.append({"clip_id": clip, "clip": f"fixtures/{clip}.mp3", "ok": True})

    (outputs / review.CRITIQUE_OUTPUT).write_text(
        json.dumps({"role": "ref-critic", "findings": findings or [], "verdict": "v"}),
        encoding="utf-8",
    )
    (run_dir / "results.json").write_text(
        json.dumps({"results": results}, ensure_ascii=False), encoding="utf-8"
    )
    return run_dir, refs


def simple(tmp_path: Path, **kwargs):
    """One clip, two windows, two flagged spans, one per window."""
    return build_run(
        tmp_path,
        adjudications=[
            adjudication(
                windows=[
                    {"index": 0, "text": "prvo je ovako i tako dalje"},
                    {"index": 1, "text": "drugo je onako i tako dalje"},
                ],
                spans=[span_json(0.0, "prvo je ovako"), span_json(15.0, "drugo je onako")],
            )
        ],
        **kwargs,
    )


class Answers:
    """A scripted reviewer. Runs out of answers loudly rather than hanging."""

    def __init__(self, *answers: str) -> None:
        self.pending = list(answers)
        self.asked: list[str] = []

    def __call__(self, prompt: str) -> str:
        self.asked.append(prompt)
        if not self.pending:
            raise AssertionError(f"the session asked more than it was told: {prompt!r}")
        return self.pending.pop(0)


class Killed(RuntimeError):
    """The process going away between two questions, raised from where the reviewer sits."""


class Crashes(Answers):
    """A scripted reviewer whose machine dies when the script runs out, rather than a failure."""

    def __call__(self, prompt: str) -> str:
        if not self.pending:
            self.asked.append(prompt)
            raise Killed(prompt)
        return super().__call__(prompt)


def silent(_cmd) -> None:
    """A player that plays nothing, which is what a test runner has."""


def run_session(run_dir, refs, *answers, **kwargs):
    ask = Answers(*answers)
    summary = review.review(
        run_dir,
        references=refs,
        ask=ask,
        play=silent,
        log=lambda _m: None,
        now="2026-08-05T00:00:00+00:00",
        **kwargs,
    )
    return summary, ask


def crash_session(run_dir, refs, *answers, **kwargs):
    """Answer what is scripted, then die at the next prompt. Returns the log lines."""
    logged: list[str] = []
    ask = Crashes(*answers)
    with pytest.raises(Killed):
        review.review(
            run_dir,
            references=refs,
            ask=ask,
            play=silent,
            log=logged.append,
            now="2026-08-05T00:00:00+00:00",
            **kwargs,
        )
    return logged


# --------------------------------------------------------------------------- loading


class TestLoadQueue:
    def test_a_critic_finding_on_the_same_window_becomes_one_stop_not_two(
        self, tmp_path: Path
    ) -> None:
        """The queue table lists them separately; a reviewer should hear the span once."""
        run_dir, _ = simple(
            tmp_path,
            findings=[
                {
                    "clip": CLIP,
                    "start": 0.0,
                    "end": 15.0,
                    "issue": "smoothed",
                    "severity": "high",
                    "reference_text": "prvo je ovako",
                    "recommendation": "listen again",
                    "candidates": ["groq: prvo je onako"],
                }
            ],
        )
        spans = review.load_queue(run_dir)

        assert len(spans) == 2
        merged = next(s for s in spans if s.start == 0.0)
        assert merged.critic is not None
        assert merged.critic.severity == "high"
        assert merged.reason == "two engines against one", "the adjudicator's reason survives"
        assert "critic: smoothed (high)" in merged.flagged_by

    def test_a_finding_with_no_adjudicator_span_stands_on_its_own(self, tmp_path: Path) -> None:
        run_dir, _ = simple(
            tmp_path,
            findings=[
                {
                    "clip": CLIP,
                    "start": 40.0,
                    "end": 44.0,
                    "issue": "grammar_corrected",
                    "severity": "medium",
                    "reference_text": "nesto drugo",
                    "recommendation": "check it",
                    "candidates": ["groq: nesto trece"],
                }
            ],
        )
        spans = review.load_queue(run_dir)

        assert len(spans) == 3
        alone = next(s for s in spans if s.start == 40.0)
        assert alone.critic is not None
        assert alone.candidates == (review.Candidate("groq", "nesto trece"),)

    def test_two_critic_findings_do_not_both_land_on_one_span(self, tmp_path: Path) -> None:
        """A span already carrying an objection is not a home for the next one."""
        finding = {
            "clip": CLIP,
            "start": 0.0,
            "end": 15.0,
            "issue": "smoothed",
            "severity": "low",
            "reference_text": "prvo je ovako",
            "recommendation": "one",
            "candidates": [],
        }
        run_dir, _ = simple(tmp_path, findings=[finding, {**finding, "recommendation": "two"}])
        spans = review.load_queue(run_dir)

        assert len(spans) == 3
        assert sum(1 for s in spans if s.critic is not None) == 2

    def test_a_clip_filter_leaves_the_other_clip_alone(self, tmp_path: Path) -> None:
        run_dir, _ = build_run(
            tmp_path,
            adjudications=[
                adjudication(
                    windows=[{"index": 0, "text": "prvo"}], spans=[span_json(0.0, "prvo")]
                ),
                adjudication(
                    "other",
                    windows=[{"index": 0, "text": "drugo"}],
                    spans=[span_json(0.0, "drugo")],
                ),
            ],
        )
        assert {s.clip for s in review.load_queue(run_dir)} == {CLIP, "other"}
        assert {s.clip for s in review.load_queue(run_dir, clip=CLIP)} == {CLIP}

    def test_an_empty_window_shifts_every_line_after_it(self, tmp_path: Path) -> None:
        """`reference_text` drops empty windows, so window index is not line index."""
        run_dir, _ = build_run(
            tmp_path,
            adjudications=[
                adjudication(
                    windows=[
                        {"index": 0, "text": "prvo"},
                        {"index": 1, "text": "   "},
                        {"index": 2, "text": "trece"},
                    ],
                    spans=[span_json(30.0, "trece")],
                )
            ],
        )
        assert review.load_queue(run_dir)[0].line == 1

    def test_a_run_with_no_adjudication_is_empty_rather_than_an_error(self, tmp_path: Path) -> None:
        (tmp_path / RUN / "agents" / "outputs").mkdir(parents=True)
        assert review.load_queue(tmp_path / RUN) == []


class TestClipPaths:
    def test_the_media_comes_from_the_run_s_own_record(self, tmp_path: Path) -> None:
        run_dir, _ = simple(tmp_path)
        found = review.clip_paths(run_dir, root=Path("/repo"))
        assert found == {CLIP: Path("/repo/fixtures/gozba-sample.mp3")}


# --------------------------------------------------------------------------- anchoring


class TestAnchor:
    def test_the_span_s_own_line_wins_over_an_identical_reading_elsewhere(self) -> None:
        """`Dakle,` occurs three times in the real reference and means three things."""
        lines = ["Dakle, prvo.", "Dakle, drugo.", "Dakle, trece."]
        span = review.Span(clip=CLIP, start=30.0, end=32.0, chosen="Dakle,", line=2)
        at = review.anchor(span, lines)
        assert at is not None
        assert at.line == 2

    def test_a_reading_written_on_the_next_line_is_still_found(self) -> None:
        """A span timestamped in one window can be written into the next one's line."""
        lines = ["prvo je ovako", "daš ga jednom državnom službenom licu"]
        span = review.Span(
            clip=CLIP, start=0.0, end=3.0, chosen="daš ga jednom državnom službenom licu", line=0
        )
        at = review.anchor(span, lines)
        assert at is not None
        assert at.line == 1

    def test_a_reading_that_is_not_a_quotation_anchors_to_nothing(self) -> None:
        lines = ["prvo je ovako i tako dalje"]
        for chosen in ('(omitted: "bukvalno tako")', "prvo ... dalje"):
            span = review.Span(clip=CLIP, start=0.0, end=3.0, chosen=chosen, line=0)
            assert review.anchor(span, lines) is None, chosen

    def test_an_ambiguous_reading_with_no_usable_line_is_refused(self) -> None:
        """Two possible homes and no window to choose between them: better to ask."""
        lines = ["Dakle, prvo.", "Dakle, drugo."]
        span = review.Span(clip=CLIP, start=0.0, end=3.0, chosen="Dakle,", line=-1)
        assert review.anchor(span, lines) is None

    def test_an_override_is_what_gets_looked_for(self) -> None:
        lines = ["da je Lok začetnik modernog empirizma"]
        span = review.Span(clip=CLIP, start=0.0, end=3.0, chosen="da je Locke začetnik", line=0)
        assert review.anchor(span, lines) is None
        assert review.anchor(span, lines, text="da je Lok začetnik") is not None


class TestApply:
    def test_a_correction_touches_its_line_and_no_other(self) -> None:
        lines = ["prvo je ovako", "prvo je ovako"]
        at = review.Anchor(line=1, text="prvo je ovako")
        assert review.apply_text(lines, at, "prvo je onako") == ["prvo je ovako", "prvo je onako"]

    def test_only_the_first_occurrence_on_that_line_moves(self) -> None:
        lines = ["Dakle, i opet Dakle, kraj"]
        at = review.Anchor(line=0, text="Dakle,")
        assert review.apply_text(lines, at, "Znači,") == ["Znači, i opet Dakle, kraj"]


class TestConventions:
    def test_every_edit_is_applied_once_and_reported(self) -> None:
        convention = review.Convention(
            clip=CLIP, label="l", why="w", edits=(("Johna Locke", "Džona Loka"),)
        )
        lines, applied = review.apply_convention(["o filozofiji Johna Locke, dalje"], convention)
        assert lines == ["o filozofiji Džona Loka, dalje"]
        assert applied == ["Johna Locke -> Džona Loka"]

    def test_an_absent_target_is_skipped_rather_than_an_error(self) -> None:
        """Absent normally means already applied, which is what a resumed session sees."""
        convention = review.Convention(
            clip=CLIP, label="l", why="w", edits=(("Johna Locke", "Džona Loka"),)
        )
        lines, applied = review.apply_convention(["nothing to see"], convention)
        assert lines == ["nothing to see"]
        assert applied == []

    def test_the_reading_a_convention_rewrote_is_what_gets_matched(self) -> None:
        """A convention and a span can quote the same words. The span must still be found."""
        span = review.Span(clip=CLIP, start=0.0, end=3.0, chosen="da je Locke začetnik", line=0)
        assert review.current_text(span, review.CONVENTIONS) == "da je Lok začetnik"

    def test_a_convention_for_another_clip_does_not_touch_this_one(self) -> None:
        span = review.Span(clip="other", start=0.0, end=3.0, chosen="da je Locke začetnik", line=0)
        assert review.current_text(span, review.CONVENTIONS) == "da je Locke začetnik"

    def test_the_locke_convention_inflects_rather_than_replacing_a_word(self) -> None:
        """`Locke` -> `Lok` alone would produce `Johna Lok`, which is not Serbian."""
        locke = next(c for c in review.CONVENTIONS if c.clip == CLIP)
        assert ("o filozofiji Johna Locke", "o filozofiji Džona Loka") in locke.edits
        assert all("Locke" in old for old, _ in locke.edits)


# --------------------------------------------------------------------------- the session


class TestSession:
    def test_accepting_everything_changes_no_text_and_verifies_the_clip(
        self, tmp_path: Path
    ) -> None:
        run_dir, refs = simple(tmp_path)
        before = (refs / f"{CLIP}.txt").read_text(encoding="utf-8")

        summary, _ = run_session(run_dir, refs, "", "")

        assert summary.answered == 2
        assert summary.remaining == 0
        assert summary.verified == (CLIP,)
        assert (refs / f"{CLIP}.txt").read_text(encoding="utf-8") == before
        assert json.loads((refs / f"{CLIP}.meta.json").read_text())["human_verified"] is True

    def test_a_correction_is_written_into_the_reference(self, tmp_path: Path) -> None:
        run_dir, refs = simple(tmp_path)

        summary, _ = run_session(run_dir, refs, "e", "prvo je onako", "")

        assert summary.answered == 2
        lines = (refs / f"{CLIP}.txt").read_text(encoding="utf-8").splitlines()
        assert lines[0] == "prvo je onako i tako dalje"
        assert lines[1] == "drugo je onako i tako dalje", "the untouched line is untouched"

    def test_picking_an_engine_reading_takes_that_engine_s_words(self, tmp_path: Path) -> None:
        run_dir, refs = simple(tmp_path)

        run_session(run_dir, refs, "2", "")

        assert (refs / f"{CLIP}.txt").read_text(encoding="utf-8").startswith("prvo je ovoko")

    def test_an_unparseable_answer_asks_again_instead_of_guessing(self, tmp_path: Path) -> None:
        run_dir, refs = simple(tmp_path)

        summary, ask = run_session(run_dir, refs, "9", "zzz", "", "")

        assert summary.answered == 2
        assert ask.pending == []

    def test_replay_does_not_count_as_an_answer(self, tmp_path: Path) -> None:
        run_dir, refs = simple(tmp_path)
        played: list[list[str]] = []

        ask = Answers("r", "", "")
        review.review(
            run_dir,
            references=refs,
            ask=ask,
            play=played.append,
            log=lambda _m: None,
            now="2026-08-05T00:00:00+00:00",
        )

        assert len(played) == 3, "once on arrival at each span, once more for the replay"

    def test_the_audio_is_played_with_a_run_up_to_the_disputed_words(self, tmp_path: Path) -> None:
        span = review.Span(clip=CLIP, start=45.0, end=60.0, chosen="x")
        cmd = review.play_cmd(Path("/clips/a.mp3"), span)

        assert cmd[0] == "ffplay"
        assert "-ss" in cmd and cmd[cmd.index("-ss") + 1] == "43.500"
        assert cmd[cmd.index("-t") + 1] == "17.000", (
            "the 15s span plus both the lead-in and the tail"
        )

    def test_a_span_at_the_very_start_does_not_seek_before_zero(self) -> None:
        span = review.Span(clip=CLIP, start=0.0, end=3.0, chosen="x")
        cmd = review.play_cmd(Path("/clips/a.mp3"), span)
        assert cmd[cmd.index("-ss") + 1] == "0.000"


class TestVerification:
    def test_a_skipped_span_keeps_the_clip_unverified(self, tmp_path: Path) -> None:
        """The whole point of the flag. A skip is recorded and settles nothing."""
        run_dir, refs = simple(tmp_path)

        summary, _ = run_session(run_dir, refs, "s", "")

        assert summary.skipped == 1
        assert summary.remaining == 1
        assert summary.verified == ()
        assert json.loads((refs / f"{CLIP}.meta.json").read_text())["human_verified"] is False

    def test_quitting_early_keeps_the_clip_unverified(self, tmp_path: Path) -> None:
        run_dir, refs = simple(tmp_path)

        summary, _ = run_session(run_dir, refs, "", "q")

        assert summary.answered == 1
        assert summary.remaining == 1
        assert summary.verified == ()

    def test_one_clip_can_be_verified_while_another_is_not(self, tmp_path: Path) -> None:
        run_dir, refs = build_run(
            tmp_path,
            adjudications=[
                adjudication(
                    windows=[{"index": 0, "text": "prvo"}], spans=[span_json(0.0, "prvo")]
                ),
                adjudication(
                    "other",
                    windows=[{"index": 0, "text": "drugo"}],
                    spans=[span_json(0.0, "drugo")],
                ),
            ],
        )

        summary, _ = run_session(run_dir, refs, "", clip=CLIP)

        assert summary.verified == (CLIP,)
        assert json.loads((refs / f"{CLIP}.meta.json").read_text())["human_verified"] is True
        assert json.loads((refs / "other.meta.json").read_text())["human_verified"] is False

    def test_verifying_replaces_the_caveat_with_what_the_flag_actually_means(
        self, tmp_path: Path
    ) -> None:
        run_dir, refs = simple(tmp_path)

        run_session(run_dir, refs, "", "", verified_by="a reviewer")

        meta = json.loads((refs / f"{CLIP}.meta.json").read_text())
        assert meta["verified_by"] == "a reviewer"
        assert meta["verified_spans"] == 2
        assert meta["note"] == review.VERIFIED_NOTE
        assert "invisible to a consensus adjudication" in meta["note"]

    def test_verify_clip_refuses_a_reference_that_is_not_there(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="no reference"):
            review.verify_clip(tmp_path, "missing", spans=1)


class TestResume:
    def test_a_settled_span_is_not_asked_about_again(self, tmp_path: Path) -> None:
        run_dir, refs = simple(tmp_path)
        run_session(run_dir, refs, "", "q")

        summary, ask = run_session(run_dir, refs, "")

        assert summary.answered == 1
        assert ask.pending == [], "exactly one span was left to ask about"
        assert summary.verified == (CLIP,)

    def test_a_skipped_span_comes_back(self, tmp_path: Path) -> None:
        run_dir, refs = simple(tmp_path)
        run_session(run_dir, refs, "s", "s")

        summary, _ = run_session(run_dir, refs, "", "")

        assert summary.answered == 2
        assert summary.verified == (CLIP,)

    def test_every_answer_is_on_disk_before_the_next_question(self, tmp_path: Path) -> None:
        """Quitting is one way a session ends; a closed laptop is another."""
        run_dir, refs = simple(tmp_path)
        seen: list[int] = []

        def ask(_prompt: str) -> str:
            seen.append(len(review.load_decisions(refs)))
            return ""

        review.review(run_dir, references=refs, ask=ask, play=silent, log=lambda _m: None, now="t")
        assert seen == [0, 1], "the second question is asked with the first answer already saved"

    def test_a_declined_convention_is_not_offered_twice(self, tmp_path: Path) -> None:
        run_dir, refs = build_run(
            tmp_path,
            adjudications=[
                adjudication(
                    windows=[{"index": 0, "text": "o filozofiji Johna Locke, dalje"}],
                    spans=[span_json(0.0, "o filozofiji Johna Locke")],
                )
            ],
        )
        _, first = run_session(run_dir, refs, "n", "")
        assert any("convention" in p for p in first.asked)

        _, second = run_session(run_dir, refs)
        assert not any("convention" in p for p in second.asked)

    def test_an_applied_convention_survives_the_next_answer_s_save(self, tmp_path: Path) -> None:
        """The per-answer write must carry the conventions, or the first answer erases them."""
        run_dir, refs = build_run(
            tmp_path,
            adjudications=[
                adjudication(
                    windows=[{"index": 0, "text": "o filozofiji Johna Locke, dalje"}],
                    spans=[span_json(0.0, "o filozofiji Johna Locke")],
                )
            ],
        )
        run_session(run_dir, refs, "y", "")

        saved = json.loads((refs / review.DECISIONS).read_text(encoding="utf-8"))
        assert saved["conventions"] == ["Serbianize the name `Locke`"]

    def test_a_convention_rewrites_the_span_it_quotes_rather_than_orphaning_it(
        self, tmp_path: Path
    ) -> None:
        run_dir, refs = build_run(
            tmp_path,
            adjudications=[
                adjudication(
                    windows=[{"index": 0, "text": "o filozofiji Johna Locke, dalje"}],
                    spans=[span_json(0.0, "o filozofiji Johna Locke")],
                )
            ],
        )
        summary, _ = run_session(run_dir, refs, "y", "")

        text = (refs / f"{CLIP}.txt").read_text(encoding="utf-8")
        assert text.strip() == "o filozofiji Džona Loka, dalje"
        assert summary.answered == 1, "accepting after a convention is still an accept"
        assert summary.verified == (CLIP,)


class TestInterrupted:
    """The session that was never quit, only stopped. `[q]` is the polite case, not the real one.

    Each of these ends the first session inside the loop and then starts a second one over the
    same directory. The assertion that matters is never "the file was written" on its own: it
    is that the second session and the files agree, because the decisions file is what tells
    it which spans to skip, and a span it skips is a span nobody will ever look at again.
    """

    def test_a_correction_outlives_the_process_that_recorded_it(self, tmp_path: Path) -> None:
        """Recording the answer and not the text is worse than recording neither."""
        run_dir, refs = simple(tmp_path)

        crash_session(run_dir, refs, "e", "prvo je onako")

        assert len(review.load_decisions(refs)) == 1, "the answer was recorded"
        assert (refs / f"{CLIP}.txt").read_text(encoding="utf-8").splitlines()[0] == (
            "prvo je onako i tako dalje"
        ), "and so was the correction it describes"

        summary, ask = run_session(run_dir, refs, "")

        assert summary.answered == 1
        assert ask.pending == [], "the settled span is not offered again"
        assert summary.verified == (CLIP,)
        lines = (refs / f"{CLIP}.txt").read_text(encoding="utf-8").splitlines()
        assert lines[0] == "prvo je onako i tako dalje", (
            "the span the resumed session skipped is still corrected in the reference"
        )
        assert lines[1] == "drugo je onako i tako dalje"

    def test_a_convention_survives_the_crash_that_logged_it(self, tmp_path: Path) -> None:
        """A convention is the worst thing to lose: it is logged, so it is never re-offered.

        Logging it while leaving the text alone puts the file and the log permanently out of
        step, and `current_text` then hunts for words the file does not contain, so the span
        it rewrote stops anchoring and quietly becomes unfixable.
        """
        run_dir, refs = build_run(
            tmp_path,
            adjudications=[
                adjudication(
                    windows=[{"index": 0, "text": "o filozofiji Johna Locke, dalje"}],
                    spans=[span_json(0.0, "o filozofiji Johna Locke")],
                )
            ],
        )

        crash_session(run_dir, refs, "y")

        saved = json.loads((refs / review.DECISIONS).read_text(encoding="utf-8"))
        assert saved["conventions"] == ["Serbianize the name `Locke`"]
        assert (refs / f"{CLIP}.txt").read_text(encoding="utf-8").strip() == (
            "o filozofiji Džona Loka, dalje"
        ), "the log says it was applied, so the reference has to say so too"

        logged: list[str] = []
        ask = Answers("")
        summary = review.review(
            run_dir, references=refs, ask=ask, play=silent, log=logged.append, now="t"
        )

        assert not any("convention" in p for p in ask.asked), "already logged, so not re-offered"
        assert not any("not one quotable stretch" in line for line in logged), (
            "the span still anchors: the file and the convention log agree about the name"
        )
        assert summary.answered == 1
        assert (refs / f"{CLIP}.txt").read_text(encoding="utf-8").strip() == (
            "o filozofiji Džona Loka, dalje"
        )

    def test_a_crash_in_a_dry_run_still_writes_nothing(self, tmp_path: Path) -> None:
        """Flushing per answer must not become a way for `--dry-run` to touch the reference."""
        run_dir, refs = simple(tmp_path)
        before = (refs / f"{CLIP}.txt").read_text(encoding="utf-8")

        crash_session(run_dir, refs, "e", "prvo je onako", dry_run=True)

        assert (refs / f"{CLIP}.txt").read_text(encoding="utf-8") == before
        assert not (refs / review.DECISIONS).exists()

    def test_ctrl_c_is_not_caught_and_cannot_raise_the_flag(self, tmp_path: Path) -> None:
        """The interrupt propagates. Catching it to run the finisher would verify an
        abandoned session, which is the one thing this module must never do."""
        run_dir, refs = simple(tmp_path)

        def ask(_prompt: str) -> str:
            if review.load_decisions(refs):
                raise KeyboardInterrupt
            return ""

        with pytest.raises(KeyboardInterrupt):
            review.review(
                run_dir, references=refs, ask=ask, play=silent, log=lambda _m: None, now="t"
            )

        assert len(review.load_decisions(refs)) == 1, "the answer before the interrupt is safe"
        assert json.loads((refs / f"{CLIP}.meta.json").read_text())["human_verified"] is False

        summary, _ = run_session(run_dir, refs, "")

        assert summary.answered == 1
        assert summary.verified == (CLIP,)
        assert json.loads((refs / f"{CLIP}.meta.json").read_text())["human_verified"] is True

    def test_a_session_with_nothing_left_finishes_what_the_interrupt_skipped(
        self, tmp_path: Path
    ) -> None:
        """Why Ctrl-C needs no handler: the flag and the queue are derived, and a re-run
        derives them again from the decisions, asking the reviewer nothing."""
        run_dir, refs = simple(tmp_path)
        spans = review.load_queue(run_dir)
        review.write_decisions(
            refs,
            [review.Decision(s.clip, s.start, s.chosen, "accept", s.chosen) for s in spans],
            run=RUN,
        )

        summary, ask = run_session(run_dir, refs)

        assert ask.asked == [], "every span was already settled"
        assert summary.answered == 0
        assert summary.verified == (CLIP,)
        assert json.loads((refs / f"{CLIP}.meta.json").read_text())["human_verified"] is True
        assert "Verified against the audio" in (refs / "review-queue.md").read_text(
            encoding="utf-8"
        )


class TestDryRun:
    def test_nothing_reaches_disk(self, tmp_path: Path) -> None:
        run_dir, refs = simple(tmp_path)
        before = (refs / f"{CLIP}.txt").read_text(encoding="utf-8")

        summary, _ = run_session(run_dir, refs, "e", "prvo je onako", "", dry_run=True)

        assert summary.answered == 2
        assert (refs / f"{CLIP}.txt").read_text(encoding="utf-8") == before
        assert not (refs / review.DECISIONS).exists()
        assert json.loads((refs / f"{CLIP}.meta.json").read_text())["human_verified"] is False


class TestQueueRewrite:
    def test_a_settled_span_leaves_the_queue_and_an_open_one_stays(self, tmp_path: Path) -> None:
        run_dir, refs = simple(tmp_path)

        run_session(run_dir, refs, "", "s")

        queue = (refs / "review-queue.md").read_text(encoding="utf-8")
        assert "drugo je onako" in queue
        assert "prvo je ovako" not in queue

    def test_a_verified_clip_is_named_above_the_table(self, tmp_path: Path) -> None:
        run_dir, refs = simple(tmp_path)

        run_session(run_dir, refs, "", "")

        queue = (refs / "review-queue.md").read_text(encoding="utf-8")
        assert "Verified against the audio" in queue
        assert f"`{CLIP}`" in queue


# --------------------------------------------------------------------------- real data


@real_data
class TestAgainstTheCommittedReference:
    """The shapes that matter here are properties of this project's own reference."""

    def test_the_forty_four_queue_rows_are_thirty_five_stops(self) -> None:
        spans = review.load_queue(REAL_RUN)
        assert len(spans) == 35
        assert sum(1 for s in spans if s.critic is not None) == 9, "9 rows were duplicates"

    def test_every_span_anchors_except_the_two_that_are_not_quotations(self) -> None:
        spans = review.load_queue(REAL_RUN)
        unanchored = []
        for clip in sorted({s.clip for s in spans}):
            lines = (REAL_REFS / f"{clip}.txt").read_text(encoding="utf-8").strip().split("\n")
            unanchored += [
                s.chosen for s in spans if s.clip == clip and review.anchor(s, lines) is None
            ]

        assert sorted(unanchored) == [
            '(omitted: "bukvalno tako")',
            "o filozofiji Johna Locke ... zašto je uopšte John Locke značajan",
        ]

    def test_the_thrice_written_dakle_resolves_to_its_own_window(self) -> None:
        spans = review.load_queue(REAL_RUN, clip="uvod-u-pravo")
        lines = (REAL_REFS / "uvod-u-pravo.txt").read_text(encoding="utf-8").strip().split("\n")
        dakle = next(s for s in spans if s.chosen == "Dakle,")

        assert sum(1 for line in lines if "Dakle," in line) == 3, "three homes to choose from"
        at = review.anchor(dakle, lines)
        assert at is not None
        assert at.line == 6

    def test_the_locke_convention_still_matches_the_committed_reference(self) -> None:
        """A convention whose targets have drifted would silently do nothing."""
        text = (REAL_REFS / f"{CLIP}.txt").read_text(encoding="utf-8")
        locke = next(c for c in review.CONVENTIONS if c.clip == CLIP)
        for old, _ in locke.edits:
            assert old in text, old
