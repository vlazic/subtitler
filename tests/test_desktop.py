"""The native window, without a window.

**Nothing here calls `Tk()`.** Both CI runners are headless and macOS has no Xvfb, so a
test that opened a window would be a test that only ever runs on the maintainer's desk.
What is covered instead is everything `window.py` delegates: the phase machine, the cue
editor's model, where corrections are staged, and the one string conversion the window
itself owns. `window.py` is imported (which touches no display) so that a syntax error or
a renamed attribute is caught rather than discovered by double-clicking an icon.

The mac branches are covered by faking `Platform`, exactly as `tests/test_doctor.py` does,
because the maintainer develops on Linux and there is no other way to reach them.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from subtitler import edits as edits_mod
from subtitler.cues import CueConfig
from subtitler.doctor import Platform
from subtitler.gui import session as session_mod
from subtitler.gui.session import EditorModel, Player, Refusal, Session
from subtitler.model import Cue
from subtitler.pipeline import RunConfig, RunResult

REPO = Path(__file__).resolve().parent.parent
FIXTURE = REPO / "fixtures" / "tiny-10s.wav"

MAC = Platform(system="Darwin", machine="arm64", brew_prefix=Path("/opt/homebrew"))
LINUX = Platform(system="Linux", machine="x86_64", distro_id="pop", distro_like="ubuntu debian")


def inline(work) -> None:
    """A `spawn` that runs the job on the caller's stack.

    Every job in this file therefore finishes before `start()` returns: no sleeps, no
    polling loops, and no thread left running into the next test.
    """
    work()


def cue(index: int, start: float, end: float, *lines: str) -> Cue:
    return Cue(index=index, start=start, end=end, lines=lines)


CUES = (
    cue(1, 0.0, 3.0, "Misao Lokove filozofije,", "ukratko izrazeno."),
    cue(2, 3.0, 7.0, "Da se opsta predstava,", "da se ono sto je istinito,"),
)


def fake_result(cfg: RunConfig, out: Path, cues: tuple[Cue, ...] = CUES) -> RunResult:
    return RunResult(
        input=FIXTURE,
        source=str(FIXTURE),
        srt=out / "tiny-10s.srt",
        cues=cues,
        cues_key="basekey000000001",
        engine="fake",
    )


def make_session(tmp_path: Path, **kwargs) -> Session:
    kwargs.setdefault("plat", LINUX)
    kwargs.setdefault("spawn", inline)
    kwargs.setdefault("runner", lambda cfg, log=print: fake_result(cfg, tmp_path))
    return Session(**kwargs)


def form(tmp_path: Path, **extra) -> dict:
    payload = {"input": str(FIXTURE), "out_dir": str(tmp_path), "review": True}
    payload.update(extra)
    return payload


# --------------------------------------------------------------------------------------
# Whether a window is possible at all
# --------------------------------------------------------------------------------------


class TestTkAvailability:
    def test_it_reports_what_importing_tkinter_actually_does(self) -> None:
        """The whole native front end rests on this one answer, so it is asserted against
        the real import rather than trusted.

        `importlib.util.find_spec("tkinter")` would say yes on a build with no `_tkinter`
        at all, which is precisely the build the question is about, so the check has to be
        an import. Importing it opens no display.
        """
        expected = True
        try:
            import tkinter  # noqa: F401
        except Exception:
            expected = False
        assert session_mod.tk_available() is expected

    def test_the_fallback_sentence_names_the_fix_and_the_alternative(self) -> None:
        """A user who reads this has no window; the message has to be self-contained.

        Both platforms are named because the sentence is printed by the machine that
        cannot show a window, and the reader is by definition not going to look one up.
        It must also not imply Tk is normally absent: the uv-managed Python `make setup`
        installs bundles it, on macOS included, so this path means the interpreter came
        from somewhere else.
        """
        assert "tkinter" in session_mod.TK_MISSING
        assert "browser" in session_mod.TK_MISSING
        assert "brew install python-tk@3.12" in session_mod.TK_MISSING
        assert "python3-tk" in session_mod.TK_MISSING
        assert "uv" in session_mod.TK_MISSING

    def test_the_window_and_doctor_answer_the_question_with_one_function(self) -> None:
        """Otherwise the panel could report a native window on a machine that just fell
        back to the browser, which is the most confusing thing a diagnostic can do."""
        from subtitler import doctor

        assert session_mod.tk_available is doctor.tk_importable

    def test_a_machine_with_no_display_is_told_which_interface_opened(self) -> None:
        assert "browser" in session_mod.NO_DISPLAY


class TestWindowImports:
    def test_the_widget_module_imports_without_opening_anything(self) -> None:
        """Catches a typo in the one file no other test can execute.

        Importing `tkinter` and `subtitler.gui.window` connects to no display; only `Tk()`
        does, and there is none at module level.
        """
        pytest.importorskip("tkinter")
        from subtitler.gui import window

        assert window.POLL_MS > 0
        assert window.MIN_SIZE < window.START_SIZE

    def test_the_language_box_label_never_reaches_the_engine(self) -> None:
        """Regression guard: the combobox shows `sr (Serbian)` because a friend does not
        know ISO codes, and `RunConfig.language` is passed to Whisper verbatim and is not
        validated against a list. Sending the label would ask for a language called
        "sr (Serbian)"."""
        pytest.importorskip("tkinter")
        from subtitler.gui.window import language_code

        assert language_code("sr (Serbian)") == "sr"
        assert language_code("auto (detect (not recommended))") == "auto"
        assert language_code("  en  ") == "en"
        assert language_code("") == ""

    def test_the_window_offers_a_control_for_every_option_the_form_takes(self) -> None:
        """The GUI adds no capability and must lose none either.

        `forms` is where a `subtitler run` flag becomes expressible in a front end, so any
        flag it grew and the window did not is a flag the friend cannot reach. No display
        is needed for this: `window.CONTROLS` is a declaration, and `Window.__init__`
        asserts the widgets it actually builds match it.
        """
        pytest.importorskip("tkinter")
        from subtitler.gui import forms
        from subtitler.gui.window import CONTROLS

        missing = set(forms.options()["defaults"]) - set(CONTROLS)
        assert missing == set()

    def test_every_control_is_a_name_the_validator_reads(self) -> None:
        """The other direction: a box the window draws that `build_config` ignores is a
        setting the user changes and the run does not honour.

        Proven by changing each control away from its default one at a time and requiring
        the resulting `RunConfig` to differ, which no amount of ignoring can fake.
        """
        pytest.importorskip("tkinter")
        from subtitler.gui import forms
        from subtitler.gui.window import CONTROLS

        base = {"input": str(FIXTURE)}
        # A value that is valid for the control and is not what it already holds.
        probes = {
            "input": str(REPO / "fixtures" / "gozba-sample.mp3"),
            "start": "5",
            "end": "9",
            "out_dir": str(REPO),
            "engine": "faster-whisper",
            "model": "tiny",
            "device": "cpu",
            "lang": "en",
            "denoise": "afftdn",
            "prompt": "Neki tekst.",
            "max_line": "38",
            "max_lines": "3",
            "min_dur": "1.5",
            "max_dur": "8",
            "max_cps": "17",
            "style_preset": "box",
            "canvas": "1920x1080",
            "canvas_color": "0x202020",
            "font_size": "40",
            "font": "Some Font",
            "review": True,
            "burn": False,
            "srt_only": True,
            "soft_mux": True,
            "fix": True,
            "force": "cues",
            "batch_size": "8",
            "verbose": "2",
            "dry_run": True,
        }
        for key in CONTROLS:
            if key not in probes:
                continue  # the --fix sub-options; covered below, they need `fix` on too
            assert forms.build_config({**base, key: probes[key]}) != forms.build_config(base), key

    def test_every_fix_control_reaches_the_correction_pass(self) -> None:
        pytest.importorskip("tkinter")
        from subtitler.gui import forms

        base = {"input": str(FIXTURE), "fix": True}
        for key, value in (
            ("fix_model", "anthropic/claude-x"),
            ("fix_prompt", "some-other-prompt"),
            ("fix_batch", "13"),
            ("fix_workers", "3"),
            ("fix_temperature", "0.4"),
            ("fix_markup", "html"),
        ):
            assert forms.build_config({**base, key: value}).fix != forms.build_config(base).fix, key


# --------------------------------------------------------------------------------------
# The editor's model
# --------------------------------------------------------------------------------------


class TestEditorModel:
    def make(self) -> EditorModel:
        return EditorModel.from_cues(CUES, CueConfig(), "basekey000000001")

    def test_it_opens_with_the_cues_as_the_run_produced_them(self) -> None:
        model = self.make()
        assert model.order == (1, 2)
        assert model.texts[1] == "Misao Lokove filozofije, ukratko izrazeno."
        assert model.edits() == {}
        assert model.flagged() == []

    def test_an_unchanged_cue_keeps_the_break_the_splitter_chose(self) -> None:
        """Selecting a cue and clicking away must not re-wrap it.

        The splitter picked that break from real word timings; `wrap_edited` has to
        synthesize timings and can only do worse. `edits.relayout` returns the cue itself
        when the text matches, and this is the test that the model routes through it
        rather than re-wrapping a string it happens to be holding.
        """
        model = self.make()
        model.set_text(1, "Misao Lokove filozofije, ukratko izrazeno.")
        assert model.rows[1]["lines"] == ["Misao Lokove filozofije,", "ukratko izrazeno."]
        assert model.is_edited(1) is False

    def test_changed_text_is_re_wrapped_and_re_linted(self) -> None:
        model = self.make()
        row = model.set_text(1, "Misao Lokove filosofije, ukratko receno, sastoji se u ovome.")
        assert model.is_edited(1) is True
        assert len(row["lines"]) == 2
        assert row["chars"] == len("Misao Lokove filosofije, ukratko receno, sastoji se u ovome.")
        assert row["problems"] == []

    def test_a_cue_made_too_long_is_flagged_in_lints_own_words(self) -> None:
        """The marks in the window and the lines `subtitler lint` prints come from one
        function, so a cue flagged here is a cue the file is reported for."""
        model = self.make()
        row = model.set_text(1, "rec " * 40)
        assert any("chars/sec exceeds" in problem for problem in row["problems"])
        assert model.flagged() == [1]
        assert "1 flagged" in model.summary()

    def test_only_the_cues_that_changed_are_saved(self) -> None:
        """Sending all of them would rekey the `edit` stage whenever anything upstream
        re-wrapped, and would put the whole transcript in a file meant to be read by hand."""
        model = self.make()
        model.set_text(1, "Nesto sasvim drugo.")
        model.set_text(2, model.texts[2])
        assert model.edits() == {1: "Nesto sasvim drugo."}

    def test_reverting_puts_the_original_break_back(self) -> None:
        model = self.make()
        model.set_text(1, "Nesto sasvim drugo.")
        model.revert(1)
        assert model.is_edited(1) is False
        assert model.rows[1]["lines"] == ["Misao Lokove filozofije,", "ukratko izrazeno."]

    def test_an_emptied_cue_is_named_before_the_burn_rather_than_after(self) -> None:
        """`edits.build` refuses these too, but only once the corrections are collected and
        the second run is starting. Naming the cue lets the window select it."""
        model = self.make()
        model.set_text(2, "   ")
        assert model.blank() == [2]

    def test_the_summary_counts_what_the_user_asks_about(self) -> None:
        model = self.make()
        assert model.summary() == "2 cues, nothing flagged"
        model.set_text(1, "Nesto sasvim drugo.")
        assert model.summary() == "2 cues, nothing flagged, 1 corrected"


# --------------------------------------------------------------------------------------
# The session
# --------------------------------------------------------------------------------------


class TestSessionStart:
    def test_a_bad_form_names_the_control_and_starts_nothing(self, tmp_path: Path) -> None:
        session = make_session(tmp_path)
        refusal = session.start({"input": ""})
        assert isinstance(refusal, Refusal)
        assert refusal.field == "input"
        assert session.phase == session_mod.IDLE
        assert session.jobs.current is None

    def test_a_review_run_lands_in_the_editor(self, tmp_path: Path) -> None:
        session = make_session(tmp_path)
        assert session.start(form(tmp_path)) is None
        session.poll()
        assert session.phase == session_mod.REVIEW
        assert session.editor is not None
        assert session.editor.base_key == "basekey000000001"
        assert len(session.editor) == 2

    def test_a_run_without_review_finishes_with_no_editor(self, tmp_path: Path) -> None:
        session = make_session(tmp_path)
        session.start(form(tmp_path, review=False, srt_only=True))
        session.poll()
        assert session.phase == session_mod.DONE
        assert session.editor is None

    def test_the_stage_list_matches_what_this_config_will_actually_do(self, tmp_path: Path) -> None:
        """A progress bar promising a burn a `--srt-only` run will skip always ends looking
        unfinished."""
        session = make_session(tmp_path)
        session.start(form(tmp_path, review=False, srt_only=True))
        assert "burn" not in session.stages
        assert "transcribe" in session.stages

    def test_a_failure_reaches_the_user_as_a_message_not_a_traceback(self, tmp_path: Path) -> None:
        """Whoever double-clicked an icon has no terminal to read a traceback in."""

        def explode(cfg, log=print):
            raise RuntimeError("ffmpeg went home")

        session = make_session(tmp_path, runner=explode)
        session.start(form(tmp_path))
        session.poll()
        assert session.phase == session_mod.FAILED
        assert "ffmpeg went home" in session.error
        assert "Traceback" in session.detail

    def test_the_log_arrives_once_and_in_order(self, tmp_path: Path) -> None:
        """`poll` is incremental: the window appends what it is handed, so a line served
        twice would appear twice."""
        lines: list[str] = []

        def chatty(cfg, log=print):
            log("input: tiny-10s.wav")
            log("transcribed: 2 segments")
            return fake_result(cfg, tmp_path)

        session = make_session(tmp_path, runner=chatty)
        session.start(form(tmp_path))
        first = session.poll()
        lines.extend(first["lines"])
        second = session.poll()
        lines.extend(second["lines"])
        assert lines == [
            f"$ {session.command}",
            "input: tiny-10s.wav",
            "transcribed: 2 segments",
        ]
        assert second["lines"] == []

    def test_each_ending_is_announced_exactly_once(self, tmp_path: Path) -> None:
        """The handshake the window redraws on, and why it is not "has the phase changed".

        Two runs in a row both end in `DONE`, so a window comparing phases would draw the
        first result and never the second; a timer running four times a second against the
        same comparison would draw the first one forever.
        """
        session = make_session(tmp_path)
        session.start(form(tmp_path, review=False))
        session.poll()
        assert session.take_finished() == "run"
        assert session.take_finished() is None
        session.poll()
        assert session.take_finished() is None

        session.start(form(tmp_path, review=False))
        session.poll()
        assert session.take_finished() == "run"

    def test_the_furthest_stage_reached_drives_the_progress_bar(self, tmp_path: Path) -> None:
        def chatty(cfg, log=print):
            log("input: tiny-10s.wav")
            log("transcribed: 2 segments")
            return fake_result(cfg, tmp_path)

        session = make_session(tmp_path, runner=chatty)
        session.start(form(tmp_path))
        assert session.poll()["stage"] == "transcribe"


class TestSessionApprove:
    def approved(self, tmp_path: Path, correction: str | None = "Ispravljeno."):
        session = make_session(tmp_path)
        session.start(form(tmp_path))
        session.poll()
        if correction is not None:
            session.editor.set_text(1, correction)
        refusal = session.approve()
        session.poll()
        return session, refusal

    def test_approving_writes_the_corrections_where_the_pipeline_reads_them(
        self, tmp_path: Path
    ) -> None:
        _session, refusal = self.approved(tmp_path)
        assert refusal is None
        saved = edits_mod.load(tmp_path / ".subtitler" / FIXTURE.stem)
        assert saved is not None
        assert saved.texts == {1: "Ispravljeno."}
        assert saved.base_key == "basekey000000001"

    def test_the_second_run_no_longer_stops_for_review(self, tmp_path: Path) -> None:
        """Otherwise approving would land back in the editor and never burn."""
        seen: list[RunConfig] = []
        session = make_session(
            tmp_path, runner=lambda cfg, log=print: seen.append(cfg) or fake_result(cfg, tmp_path)
        )
        session.start(form(tmp_path))
        session.poll()
        session.approve()
        assert [cfg.review for cfg in seen] == [True, False]

    def test_approving_with_nothing_typed_clears_an_earlier_set(self, tmp_path: Path) -> None:
        """A correction the user undid must not come back from the file the previous
        approval left behind."""
        self.approved(tmp_path, "Prva verzija.")
        session = make_session(tmp_path)
        session.start(form(tmp_path))
        session.poll()
        session.approve()
        assert edits_mod.load(tmp_path / ".subtitler" / FIXTURE.stem) is None

    def test_an_emptied_cue_is_refused_and_nothing_is_started(self, tmp_path: Path) -> None:
        session = make_session(tmp_path)
        session.start(form(tmp_path))
        session.poll()
        session.editor.set_text(2, "  ")
        refusal = session.approve()
        assert refusal is not None
        assert "cue 2" in refusal.message
        assert session.phase == session_mod.REVIEW

    def test_approving_before_a_run_says_so_rather_than_raising(self, tmp_path: Path) -> None:
        assert make_session(tmp_path).approve() == Refusal("", "there is nothing to approve yet")


class TestStageEdits:
    def test_it_writes_and_clears_the_same_file_the_edit_stage_reads(self, tmp_path: Path) -> None:
        cfg = RunConfig(input=FIXTURE, out_dir=tmp_path)
        session_mod.stage_edits(cfg, "key1", {3: "Tekst."})
        work = tmp_path / ".subtitler" / FIXTURE.stem
        assert edits_mod.load(work).texts == {3: "Tekst."}
        session_mod.stage_edits(cfg, "key1", {})
        assert edits_mod.load(work) is None

    def test_corrections_with_no_transcript_behind_them_are_refused(self, tmp_path: Path) -> None:
        """Without a base key nothing can tell whether they are still about this text."""
        cfg = RunConfig(input=FIXTURE, out_dir=tmp_path)
        with pytest.raises(edits_mod.EditError):
            session_mod.stage_edits(cfg, "", {1: "Tekst."})


class TestResults:
    def test_the_finished_files_come_back_labelled_and_only_if_they_exist(
        self, tmp_path: Path
    ) -> None:
        session = make_session(tmp_path)
        session.start(form(tmp_path, review=False))
        session.poll()
        # `fake_result` names an srt that was never written, so nothing is offered.
        assert session.outputs() == []
        (tmp_path / "tiny-10s.srt").write_text("1\n", encoding="utf-8")
        assert session.outputs() == [("subtitles", tmp_path / "tiny-10s.srt")]

    def test_a_run_that_warns_hands_the_warning_to_the_window(self, tmp_path: Path) -> None:
        """The warning that exists says the transcript may be the steering prompt rather
        than the speech. Leaving it in the log would put it above the fold of a scrolling
        panel and below the notice of the person it is for."""

        def warned(cfg, log=print):
            result = fake_result(cfg, tmp_path)
            result.warnings.append("the transcript may be the prompt, not the speech")
            return result

        session = make_session(tmp_path, runner=warned)
        session.start(form(tmp_path, review=False))
        session.poll()
        assert session.warnings == ["the transcript may be the prompt, not the speech"]

    def test_a_run_with_nothing_to_say_says_nothing(self, tmp_path: Path) -> None:
        session = make_session(tmp_path)
        assert session.warnings == []
        session.start(form(tmp_path, review=False))
        session.poll()
        assert session.warnings == []

    def test_the_reveal_button_says_finder_on_a_mac_and_folder_everywhere_else(self) -> None:
        assert Session(plat=MAC, spawn=inline).reveal_label() == "Show in Finder"
        assert Session(plat=LINUX, spawn=inline).reveal_label() == "Open folder"

    def test_revealing_a_file_uses_the_platforms_own_command(self, tmp_path: Path) -> None:
        seen: list[list[str]] = []
        target = tmp_path / "out.mp4"
        target.write_bytes(b"")
        Session(plat=MAC, spawn=inline, opener=lambda argv: seen.append(list(argv))).reveal(target)
        assert seen == [["open", "-R", str(target)]]

    def test_a_desktop_with_no_xdg_open_reports_the_folder_instead_of_failing(
        self, tmp_path: Path
    ) -> None:
        """xdg-open is not installed everywhere, and failing to open a folder must not read
        like a failure of the run that filled it."""

        def broken(argv):
            raise OSError("no xdg-open")

        message = Session(plat=LINUX, spawn=inline, opener=broken).reveal(tmp_path)
        assert str(tmp_path) in message
        assert "could not open" in message


# --------------------------------------------------------------------------------------
# The dependency panel
# --------------------------------------------------------------------------------------


class TestDoctorPanel:
    """`doctor` inside the window, for the user who has no terminal to run it in.

    The single most common way this project fails on the primary target is an ffmpeg built
    without libass, and the only other sign of it is a burn that dies at the very end of a
    long run.
    """

    def statuses(self, plat: Platform):
        from subtitler.doctor import DEPS, Probe, diagnose

        return diagnose(plat, Probe(which=lambda _n: None, tk_importable=lambda: True), DEPS)

    def test_the_report_arrives_with_the_rendered_text_and_the_blockers(self) -> None:
        session = Session(plat=LINUX, spawn=inline, diagnose=self.statuses)
        session.start_doctor()
        assert session.doctor_report is not None
        assert "ffmpeg" in session.doctor_report["text"]
        assert "ffmpeg" in session.doctor_report["blocking"]
        assert session.doctor_running is False

    def test_it_is_computed_once_and_then_reused(self, tmp_path: Path) -> None:
        """It shells out a dozen times; a timer running four times a second must not."""
        calls: list[int] = []
        session = make_session(
            tmp_path, diagnose=lambda plat: calls.append(1) or self.statuses(plat)
        )
        session.start_doctor()
        session.start_doctor()
        assert len(calls) == 1
        session.start_doctor(refresh=True)
        assert len(calls) == 2

    def test_it_can_be_read_while_a_run_is_going(self, tmp_path: Path) -> None:
        """Deliberately off the job manager. A panel that refused to open because the
        machine was busy would be shut at exactly the moment somebody went looking for it.
        """
        session = make_session(tmp_path, runner=lambda cfg, log=print: fake_result(cfg, tmp_path))
        session.start(form(tmp_path))
        assert session.start_doctor() is False  # inline: it has already finished
        assert session.doctor_report is not None

    def test_a_broken_check_does_not_wedge_the_panel_shut(self, tmp_path: Path) -> None:
        def broken(_plat):
            raise OSError("nvidia-smi went away")

        session = make_session(tmp_path, diagnose=broken)
        session.start_doctor()
        assert "nvidia-smi went away" in session.doctor_report["text"]
        assert session.doctor_running is False


# --------------------------------------------------------------------------------------
# The weights
# --------------------------------------------------------------------------------------


class TestModels:
    def test_the_backend_offered_is_the_one_this_machine_would_use(self) -> None:
        assert Session(plat=MAC, spawn=inline).backend == "mlx"
        assert Session(plat=LINUX, spawn=inline).backend == "faster-whisper"

    def test_each_model_says_whether_it_is_already_on_disk(self, tmp_path: Path) -> None:
        rows = make_session(tmp_path).model_rows()
        assert rows
        assert {row["backend"] for row in rows} == {"faster-whisper"}
        for row in rows:
            assert isinstance(row["cached"], bool)
            assert row["size"]

    def test_a_download_reports_its_progress_into_the_same_log(self, tmp_path: Path) -> None:
        """`EngineUnavailable` names a terminal command, which is a dead end for somebody
        who was handed an icon."""
        session = make_session(
            tmp_path,
            downloader=lambda spec, progress=None: (
                progress and progress("fetching model.bin"),
                tmp_path / "snapshot",
            )[1],
        )
        assert session.download_model("tiny") is None
        snapshot = session.poll()
        assert snapshot["kind"] == "download"
        assert "fetching model.bin" in snapshot["lines"]
        assert session.take_finished() == "download"
        assert session.error == ""

    def test_an_unknown_model_names_the_control_rather_than_raising(self, tmp_path: Path) -> None:
        refusal = make_session(tmp_path).download_model("enormous-v9")
        assert refusal is not None
        assert refusal.field == "model"

    def test_a_download_that_fails_leaves_the_corrections_alone(self, tmp_path: Path) -> None:
        """A set of cues waiting for approval must survive somebody clicking Download on
        the wrong thing: the phase belongs to the run, and only the run may change it."""

        def explode(spec, progress=None):
            raise OSError("the network went away")

        session = make_session(tmp_path, downloader=explode)
        session.start(form(tmp_path))
        session.poll()
        assert session.phase == session_mod.REVIEW

        session.download_model("tiny")
        session.poll()
        assert session.take_finished() == "download"
        assert "the network went away" in session.error
        assert session.phase == session_mod.REVIEW
        assert session.editor is not None

    def test_nothing_else_may_start_while_the_weights_are_arriving(self, tmp_path: Path) -> None:
        """Three gigabytes of download underneath a transcription helps nobody."""
        held: list[object] = []
        session = make_session(
            tmp_path, spawn=held.append, downloader=lambda spec, progress=None: 0
        )
        assert session.download_model("tiny") is None
        assert session.busy is True
        refusal = session.start(form(tmp_path))
        assert refusal is not None and "still running" in refusal.message

    def test_the_bar_measures_the_bytes_that_landed(self, tmp_path: Path) -> None:
        """There is nothing to report: `snapshot_download` draws its bars on a terminal
        nobody launched this from, and its callback emits three lines and no numbers."""
        session = make_session(tmp_path, spawn=lambda work: None)
        assert session.download_fraction() is None
        session.download_model("tiny")
        fraction = session.download_fraction()
        assert fraction is not None
        assert 0.0 <= fraction <= 1.0


# --------------------------------------------------------------------------------------
# Hearing a cue
# --------------------------------------------------------------------------------------


class TestPlayer:
    def test_it_plays_exactly_the_span_of_the_cue(self) -> None:
        seen: list[list[str]] = []
        player = Player(spawn=lambda argv: seen.append(list(argv)), which=lambda name: "/bin/x")
        assert player.play(FIXTURE, 4.0, 6.5) == ""
        assert seen[0][-1] == str(FIXTURE)
        assert "-ss" in seen[0] and "4.000" in seen[0]
        assert "2.500" in seen[0]

    def test_a_machine_without_ffplay_gets_a_sentence_rather_than_an_exception(self) -> None:
        player = Player(spawn=lambda argv: None, which=lambda name: None)
        assert "not installed" in player.play(FIXTURE, 0.0, 1.0)

    def test_starting_a_second_cue_stops_the_first(self) -> None:
        """Otherwise clicking down a list plays every cue at once."""

        class FakeProc:
            def __init__(self) -> None:
                self.killed = False

            def poll(self):
                return None

            def terminate(self):
                self.killed = True

        procs: list[FakeProc] = []

        def spawn(argv):
            procs.append(FakeProc())
            return procs[-1]

        player = Player(spawn=spawn, which=lambda name: "/bin/x")
        player.play(FIXTURE, 0.0, 1.0)
        player.play(FIXTURE, 1.0, 2.0)
        assert procs[0].killed is True
        assert procs[1].killed is False
