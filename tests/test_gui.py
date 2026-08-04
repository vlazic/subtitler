"""The GUI, without a window.

Both CI runners are headless and macOS has no Xvfb, so a GUI that could only be tested by
clicking would be a GUI that is never tested on the primary target. Everything here is
therefore reachable without a display: the form is a pure function, the file picker takes a
faked `Platform` exactly as `doctor.py` does, the worker thread is behind an injectable
`spawn`, and the last class drives the real HTTP server over a real socket on an ephemeral
port, which is the same code path a browser uses.

The mac branches (`~/Movies`, `open -R`) are covered by faking `Platform`, because the
maintainer develops on Linux and there is no other way to exercise them.
"""

from __future__ import annotations

import json
import shutil
import threading
import urllib.error
import urllib.request
from pathlib import Path

import pytest

from subtitler import postedit
from subtitler.cues import CueConfig
from subtitler.doctor import Platform
from subtitler.gui import files, forms, jobs
from subtitler.gui.app import GuiApp
from subtitler.gui.server import build_server, is_local_host_header
from subtitler.model import Segment, Transcript, Word
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


# --------------------------------------------------------------------------------------
# The form
# --------------------------------------------------------------------------------------


class TestBuildConfig:
    def test_an_empty_form_still_produces_the_cli_defaults(self) -> None:
        """The window and the command line must start from the same place.

        A GUI whose blank form differs from `subtitler run FILE` would quietly produce
        different subtitles for the same file depending on which one the user reached for.
        """
        cfg = forms.build_config({"input": str(FIXTURE)})
        reference = RunConfig(input=FIXTURE)
        assert (cfg.engine, cfg.model, cfg.device, cfg.language, cfg.denoise) == (
            reference.engine,
            reference.model,
            reference.device,
            reference.language,
            reference.denoise,
        )
        assert cfg.cues == CueConfig()
        assert cfg.burn is True
        assert cfg.fix is None

    def test_every_control_the_page_renders_reaches_the_run_config(self) -> None:
        cfg = forms.build_config(
            {
                "input": str(FIXTURE),
                "out_dir": "/tmp",
                "engine": "faster-whisper",
                "model": "tiny",
                "device": "cuda",
                "batch_size": "16",
                "lang": "hr",
                "prompt": "  a steering prompt  ",
                "denoise": "speech",
                "srt_only": True,
                "soft_mux": True,
                "canvas": "1920x1080",
                "canvas_color": "0x000000",
                "style_preset": "box",
                "font": "Noto Sans",
                "font_size": "36",
                "max_line": "38",
                "max_lines": "3",
                "min_dur": "0.8",
                "max_dur": "6.0",
                "max_cps": "17",
                "force": "transcribe",
                "dry_run": True,
                "verbose": "2",
            }
        )
        assert cfg.out_dir == Path("/tmp")
        assert (cfg.engine, cfg.model, cfg.device, cfg.batch_size) == (
            "faster-whisper",
            "tiny",
            "cuda",
            16,
        )
        assert (cfg.language, cfg.prompt, cfg.denoise) == ("hr", "a steering prompt", "speech")
        assert (cfg.srt_only, cfg.soft_mux, cfg.style_preset) == (True, True, "box")
        assert (cfg.canvas, cfg.canvas_color) == ("1920x1080", "0x000000")
        assert (cfg.font, cfg.font_size) == ("Noto Sans", 36)
        assert cfg.cues == CueConfig(
            max_line=38, max_lines=3, min_dur=0.8, max_dur=6.0, max_cps=17.0
        )
        assert (cfg.force, cfg.dry_run, cfg.verbose) == ("transcribe", True, 2)

    def test_a_missing_input_is_a_named_field_not_a_stack_trace(self) -> None:
        with pytest.raises(forms.FormError) as exc:
            forms.build_config({})
        assert exc.value.field == "input"

    def test_a_path_that_does_not_exist_is_rejected_before_anything_starts(self) -> None:
        """The audience types nothing, but they can rename a file between picking and
        starting. Failing here costs them a click; failing inside the pipeline costs them
        a half-written work directory and a traceback."""
        with pytest.raises(forms.FormError) as exc:
            forms.build_config({"input": "/definitely/not/here.mp4"})
        assert exc.value.field == "input"
        assert "no such file" in exc.value.message

    def test_a_folder_is_not_an_input_file(self, tmp_path: Path) -> None:
        with pytest.raises(forms.FormError, match="folder"):
            forms.build_config({"input": str(tmp_path)})

    def test_an_output_path_that_is_a_file_is_rejected(self, tmp_path: Path) -> None:
        clash = tmp_path / "already.txt"
        clash.write_text("x", encoding="utf-8")
        with pytest.raises(forms.FormError) as exc:
            forms.build_config({"input": str(FIXTURE), "out_dir": str(clash)})
        assert exc.value.field == "out_dir"

    def test_a_bad_canvas_is_caught_here_rather_than_three_ffmpeg_calls_later(self) -> None:
        """`pipeline.parse_canvas` raises for the same input, but only after probe and
        extract have run, which in a window reads as "it worked and then broke"."""
        with pytest.raises(forms.FormError) as exc:
            forms.build_config({"input": str(FIXTURE), "canvas": "1920 by 1080"})
        assert exc.value.field == "canvas"

    def test_cue_durations_must_be_the_right_way_round(self) -> None:
        with pytest.raises(forms.FormError) as exc:
            forms.build_config({"input": str(FIXTURE), "min_dur": "9", "max_dur": "4"})
        assert exc.value.field == "min_dur"

    def test_numbers_outside_their_range_name_the_range(self) -> None:
        with pytest.raises(forms.FormError, match="between 10 and 120"):
            forms.build_config({"input": str(FIXTURE), "max_line": "500"})

    def test_an_unknown_choice_is_refused_rather_than_passed_through(self) -> None:
        """The page builds its dropdowns from `forms.options()`, so this can only happen to
        a hand-made request. It must still not reach `resolve()` as an engine name."""
        with pytest.raises(forms.FormError) as exc:
            forms.build_config({"input": str(FIXTURE), "engine": "whisper.cpp"})
        assert exc.value.field == "engine"

    def test_blank_boxes_fall_back_to_the_default_instead_of_zero(self) -> None:
        """An emptied number field posts "", and `int("")` is not the user asking for 0."""
        cfg = forms.build_config(
            {"input": str(FIXTURE), "max_line": "", "font_size": "", "batch_size": ""}
        )
        assert cfg.cues.max_line == 42
        assert cfg.font_size is None
        assert cfg.batch_size == 0

    def test_the_prompt_can_come_from_a_file(self, tmp_path: Path) -> None:
        prompt = tmp_path / "p.txt"
        prompt.write_text("verbatim steering\n", encoding="utf-8")
        cfg = forms.build_config({"input": str(FIXTURE), "prompt_file": str(prompt)})
        assert cfg.prompt == "verbatim steering"


class TestFixForm:
    def test_the_correction_pass_is_off_unless_the_box_is_ticked(self) -> None:
        """`--fix` is the only stage that costs money. Every fix field can be filled in and
        it still must not run without the checkbox."""
        cfg = forms.build_config(
            {"input": str(FIXTURE), "fix_model": "openai/gpt-4o", "fix_batch": "10"}
        )
        assert cfg.fix is None

    def test_ticking_it_carries_every_fix_control(self, tmp_path: Path) -> None:
        phrases = tmp_path / "intro.txt"
        phrases.write_text("dobrodosli\n", encoding="utf-8")
        cfg = forms.build_config(
            {
                "input": str(FIXTURE),
                "fix": True,
                "fix_model": "openai/gpt-4o",
                "fix_prompt": "gozba",
                "fix_batch": "12",
                "fix_workers": "2",
                "fix_markup": "html",
                "drop_intro_phrases": str(phrases),
            }
        )
        assert cfg.fix == postedit.FixConfig(
            model="openai/gpt-4o",
            prompt="gozba",
            batch_size=12,
            workers=2,
            temperature=None,
            markup="html",
            drop_intro_phrases=phrases,
        )

    def test_an_untouched_temperature_box_stays_unset_rather_than_becoming_zero(self) -> None:
        """Current Claude models reject temperature/top_p/top_k with a 400, so `postedit`
        sends nothing unless asked. A blank text field must not turn into 0.0."""
        cfg = forms.build_config({"input": str(FIXTURE), "fix": True, "fix_temperature": ""})
        assert cfg.fix is not None
        assert cfg.fix.temperature is None

        typed = forms.build_config({"input": str(FIXTURE), "fix": True, "fix_temperature": "0.3"})
        assert typed.fix is not None
        assert typed.fix.temperature == 0.3


class TestCommandLine:
    def test_defaults_produce_a_command_with_nothing_but_the_file(self) -> None:
        """A command that repeats every default teaches the reader nothing about which
        knob mattered, and is the one thing the friend will paste into a bug report."""
        cfg = forms.build_config({"input": str(FIXTURE)})
        assert forms.to_argv(cfg) == ["subtitler", "run", str(FIXTURE)]

    def test_each_changed_control_appears_once(self) -> None:
        cfg = forms.build_config(
            {
                "input": str(FIXTURE),
                "out_dir": "/tmp/out",
                "engine": "faster-whisper",
                "model": "tiny",
                "denoise": "speech",
                "srt_only": True,
                "style_preset": "box",
                "max_line": "30",
                "batch_size": "16",
                "lang": "en",
                "force": "transcribe",
            }
        )
        argv = forms.to_argv(cfg)
        assert argv[:3] == ["subtitler", "run", str(FIXTURE)]
        for flag, value in (
            ("-o", "/tmp/out"),
            ("--engine", "faster-whisper"),
            ("--model", "tiny"),
            ("--denoise", "speech"),
            ("--style-preset", "box"),
            ("--max-line", "30"),
            ("--batch-size", "16"),
            ("--lang", "en"),
            ("--force", "transcribe"),
        ):
            assert argv[argv.index(flag) + 1] == value
        assert "--srt-only" in argv
        assert argv.count("--engine") == 1

    def test_turning_the_burn_off_uses_the_negated_flag(self) -> None:
        """`--burn` is a BooleanOptionalAction defaulting to True, so "off" is `--no-burn`
        and there is no `--burn false` to emit."""
        cfg = forms.build_config({"input": str(FIXTURE), "burn": False})
        assert "--no-burn" in forms.to_argv(cfg)

    def test_the_fix_flags_only_appear_with_the_fix_flag(self) -> None:
        plain = forms.to_argv(forms.build_config({"input": str(FIXTURE)}))
        assert not [a for a in plain if a.startswith("--fix")]

        fixed = forms.to_argv(
            forms.build_config({"input": str(FIXTURE), "fix": True, "fix_model": "openai/gpt-4o"})
        )
        assert "--fix" in fixed
        assert fixed[fixed.index("--fix-model") + 1] == "openai/gpt-4o"

    def test_a_path_with_a_space_survives_the_quoting(self, tmp_path: Path) -> None:
        spaced = tmp_path / "a talk.wav"
        shutil.copy(FIXTURE, spaced)
        line = forms.command_line(forms.build_config({"input": str(spaced)}))
        assert "'" in line or '"' in line
        assert str(spaced) in line.replace("'", "")

    def test_verbosity_is_a_repeated_letter_not_a_number(self) -> None:
        cfg = forms.build_config({"input": str(FIXTURE), "verbose": "2"})
        assert "-vv" in forms.to_argv(cfg)


class TestStageList:
    """Which chips the progress bar shows, which is a function of the config alone."""

    def test_the_optional_stages_are_left_out_when_they_will_not_run(self) -> None:
        """A `burn` chip that never lights up on a `--srt-only` run reads as a step that
        failed, and the run looks unfinished for as long as the page is open."""
        cfg = forms.build_config({"input": str(FIXTURE), "srt_only": True})
        assert forms.stages_for(cfg) == ("probe", "extract", "transcribe", "cues", "render")

    def test_asking_for_them_puts_them_back(self) -> None:
        cfg = forms.build_config({"input": str(FIXTURE), "denoise": "speech", "fix": True})
        assert forms.stages_for(cfg) == jobs.STAGES

    def test_turning_the_burn_off_drops_the_burn_chip_too(self) -> None:
        cfg = forms.build_config({"input": str(FIXTURE), "burn": False})
        assert "burn" not in forms.stages_for(cfg)

    def test_the_job_carries_the_narrowed_list_to_the_page(self) -> None:
        app = make_app()
        started = body(
            app.handle(
                "POST",
                "/api/run",
                body=json.dumps({"input": str(FIXTURE), "srt_only": True}).encode(),
                token="tok",
            )
        )
        assert "burn" not in started["stages"]
        assert "denoise" not in started["stages"]


def test_the_page_and_the_validator_offer_the_same_choices() -> None:
    """`/api/options` is what the dropdowns are built from, so a value the page can show
    and the validator would reject is not reachable through the UI at all."""
    opts = forms.options()
    for key, expected in (
        ("engines", forms.ENGINES),
        ("devices", forms.DEVICES),
        ("denoisers", forms.DENOISERS),
        ("style_presets", forms.STYLE_PRESETS),
    ):
        assert opts[key] == list(expected)
    for engine in opts["engines"]:
        forms.build_config({"input": str(FIXTURE), "engine": engine})


# --------------------------------------------------------------------------------------
# The file picker, and the one macOS branch
# --------------------------------------------------------------------------------------


class TestPlaces:
    def test_macos_offers_movies_and_linux_offers_videos(self, tmp_path: Path) -> None:
        """The same folder has two names on the two platforms, and a shortcut pointing at
        a folder the user does not have is worse than no shortcut."""
        for name in ("Desktop", "Movies", "Videos"):
            (tmp_path / name).mkdir()
        mac = {p.name for p in files.places(MAC, tmp_path)}
        linux = {p.name for p in files.places(LINUX, tmp_path)}
        assert "Movies" in mac and "Videos" not in mac
        assert "Videos" in linux and "Movies" not in linux

    def test_home_is_always_first_even_on_a_bare_account(self, tmp_path: Path) -> None:
        found = files.places(MAC, tmp_path)
        assert found[0].name == "Home"
        assert found[0].path == tmp_path
        assert len(found) == 1


class TestReveal:
    def test_macos_selects_the_file_and_linux_opens_its_folder(self) -> None:
        """`open -R` is the only one of the two that can put the cursor on the file.
        Nothing on freedesktop reveals a file, so Linux settles for the folder; passing
        `-R` to `xdg-open` would simply fail."""
        target = Path("/home/x/clip.subbed.mp4")
        assert files.reveal_command(MAC, target, is_dir=False) == ["open", "-R", str(target)]
        assert files.reveal_command(LINUX, target, is_dir=False) == ["xdg-open", "/home/x"]

    def test_a_folder_is_opened_rather_than_revealed_on_both(self) -> None:
        folder = Path("/home/x/out")
        assert files.reveal_command(MAC, folder, is_dir=True) == ["open", str(folder)]
        assert files.reveal_command(LINUX, folder, is_dir=True) == ["xdg-open", str(folder)]

    def test_opening_a_file_with_its_default_app(self) -> None:
        assert files.open_command(MAC, Path("/a/b.mp4")) == ["open", "/a/b.mp4"]
        assert files.open_command(LINUX, Path("/a/b.mp4")) == ["xdg-open", "/a/b.mp4"]


class TestListing:
    @pytest.fixture
    def tree(self, tmp_path: Path) -> Path:
        (tmp_path / "Zed").mkdir()
        (tmp_path / "alpha").mkdir()
        (tmp_path / ".hidden").mkdir()
        for name in ("talk.mp4", "song.MP3", "notes.txt", ".secret.wav"):
            (tmp_path / name).write_bytes(b"x" * 10)
        return tmp_path

    def test_folders_come_first_and_only_media_is_offered(self, tree: Path) -> None:
        listing = files.list_dir(tree)
        names = [e["name"] for e in listing["entries"]]
        assert names == ["alpha", "Zed", "song.MP3", "talk.mp4"]

    def test_the_media_filter_is_case_insensitive(self, tree: Path) -> None:
        """A file off a camera or a Windows share is as likely to be .MP3 as .mp3, and a
        picker that hides it looks broken rather than strict."""
        kinds = {e["name"]: e["kind"] for e in files.list_dir(tree)["entries"]}
        assert kinds["song.MP3"] == "audio"
        assert kinds["talk.mp4"] == "video"

    def test_show_everything_includes_the_non_media_files(self, tree: Path) -> None:
        names = [e["name"] for e in files.list_dir(tree, media_only=False)["entries"]]
        assert "notes.txt" in names

    def test_hidden_entries_stay_hidden_unless_asked_for(self, tree: Path) -> None:
        assert ".hidden" not in [e["name"] for e in files.list_dir(tree)["entries"]]
        shown = files.list_dir(tree, media_only=False, show_hidden=True)["entries"]
        assert ".hidden" in [e["name"] for e in shown]

    def test_a_broken_child_is_skipped_rather_than_failing_the_whole_listing(
        self, tree: Path
    ) -> None:
        """A dangling symlink in Downloads must not make Downloads unbrowsable."""
        (tree / "dangling.mp4").symlink_to(tree / "gone.mp4")
        names = [e["name"] for e in files.list_dir(tree)["entries"]]
        assert "dangling.mp4" not in names
        assert "talk.mp4" in names

    def test_the_crumbs_walk_back_to_the_root(self, tree: Path) -> None:
        listing = files.list_dir(tree)
        assert listing["crumbs"][0]["path"] == "/"
        assert listing["crumbs"][-1]["path"] == str(tree)

    def test_a_file_is_not_a_directory(self, tree: Path) -> None:
        with pytest.raises(NotADirectoryError):
            files.list_dir(tree / "talk.mp4")


# --------------------------------------------------------------------------------------
# Progress
# --------------------------------------------------------------------------------------


class TestStages:
    # Copied from the `log(...)` calls in `pipeline.py`. If a message there is reworded,
    # this list stops matching and the progress display stops moving, which is exactly the
    # kind of silent regression a window makes invisible.
    PIPELINE_LINES = (
        ("input: talk.mp4 (109.0s, audio only)", "probe"),
        ("extract: cached", "extract"),
        ("denoise: speech", "denoise"),
        ("denoise: cached (speech)", "denoise"),
        ("engine: faster-whisper (large-v3, cuda)", "transcribe"),
        ("batched decoding: 16 chunks at a time. The steering prompt is NOT sent", "transcribe"),
        ("transcribe: cached (42 segments)", "transcribe"),
        ("transcribed: 42 segments in 8.1s (rtf 0.07)", "transcribe"),
        ("cues: cached (61 cues)", "cues"),
        ("fix: 61 cues through anthropic/claude-sonnet-5", "fix"),
        ("wrote: talk.srt, talk.vtt (61 cues)", "render"),
        ("lint: 3 cue-quality violations (see `subtitler lint talk.srt`)", "render"),
        ("burn: cached (talk.subbed.mp4)", "burn"),
        ("burned: talk.subbed.mp4 (1280x720)", "burn"),
    )

    @pytest.mark.parametrize(("line", "stage"), PIPELINE_LINES)
    def test_every_line_the_pipeline_logs_maps_to_a_stage(self, line: str, stage: str) -> None:
        assert jobs.stage_for_line(line) == stage

    def test_an_unrecognised_line_is_not_forced_into_a_stage(self) -> None:
        assert jobs.stage_for_line("--batch-size 16 ignored: batching only helps on CUDA") is None
        assert jobs.stage_for_line("") is None

    def test_the_display_never_moves_backwards(self) -> None:
        """`lint:` maps to render and is emitted after `wrote:`, but a burn follows both,
        and a late warning from an earlier stage must not rewind the tick marks."""
        log = [
            "input: a.wav (10.0s, audio only)",
            "engine: fake (fake-1, cpu)",
            "wrote: a.srt, a.vtt (3 cues)",
            "burned: a.subbed.mp4 (1280x720)",
            "lint: 1 cue-quality violations",
        ]
        assert jobs.stage_of(log) == "burn"

    def test_no_recognised_line_yet_means_no_stage_rather_than_the_first_one(self) -> None:
        assert jobs.stage_of(["$ subtitler run a.wav"]) is None


class TestJobs:
    def test_a_job_collects_the_log_callback_line_by_line(self) -> None:
        manager = jobs.JobManager(spawn=inline)

        def work(log):
            log("input: a.wav (10.0s, audio only)")
            log("wrote: a.srt, a.vtt (3 cues)")
            return {"cue_count": 3}

        job = manager.start("run", "a.wav", work)
        snap = job.snapshot(now=1.0)
        assert snap["status"] == "done"
        assert snap["lines"] == [
            "input: a.wav (10.0s, audio only)",
            "wrote: a.srt, a.vtt (3 cues)",
        ]
        assert snap["stage"] == "render"
        assert snap["result"] == {"cue_count": 3}

    def test_since_returns_only_what_the_page_has_not_seen(self) -> None:
        """The page polls once a second for minutes; re-sending the whole log each time
        would make a long run's traffic quadratic in its length."""
        manager = jobs.JobManager(spawn=inline)
        job = manager.start("run", "a", lambda log: [log(f"line {i}") for i in range(5)] and {})
        assert job.snapshot(since=3, now=1.0)["lines"] == ["line 3", "line 4"]
        assert job.snapshot(since=0, now=1.0)["next"] == 5

    def test_a_failing_job_records_the_error_instead_of_killing_the_worker(self) -> None:
        """A crash in a thread that nobody joins is a page that spins forever. The job has
        to end in a state the browser can render."""
        manager = jobs.JobManager(spawn=inline)

        def boom(_log):
            raise RuntimeError("ffmpeg is not installed")

        job = manager.start("run", "a.wav", boom)
        snap = job.snapshot(now=1.0)
        assert snap["status"] == "error"
        assert "ffmpeg is not installed" in snap["error"]
        assert "Traceback" in snap["detail"]

    def test_a_second_job_is_refused_while_one_is_running(self) -> None:
        manager = jobs.JobManager(spawn=lambda work: None)  # never runs, so it stays running
        manager.start("run", "first", lambda log: {})
        with pytest.raises(jobs.Busy):
            manager.start("run", "second", lambda log: {})

    def test_a_finished_job_is_still_readable(self) -> None:
        """The result panel is drawn from the last poll after the job ended, so the
        manager must not discard it the moment the worker returns."""
        manager = jobs.JobManager(spawn=inline)
        job = manager.start("run", "a", lambda log: {"srt": "/tmp/a.srt"})
        assert manager.snapshot(job.id)["result"] == {"srt": "/tmp/a.srt"}


# --------------------------------------------------------------------------------------
# The API
# --------------------------------------------------------------------------------------


def fake_result(cfg: RunConfig, out: Path) -> RunResult:
    word = Word(start=0.0, end=1.0, text="proba")
    transcript = Transcript(
        language="sr",
        duration=10.0,
        segments=(Segment(start=0.0, end=1.0, text="proba", words=(word,)),),
        engine="fake",
        model="fake-1",
        runtime_s=0.5,
    )
    return RunResult(input=cfg.input, srt=out / "a.srt", vtt=out / "a.vtt", transcript=transcript)


def make_app(
    *, plat: Platform = LINUX, runner=None, opener=None, home: Path | None = None
) -> GuiApp:
    return GuiApp(
        token="tok",
        plat=plat,
        home=home or Path.home(),
        spawn=inline,
        runner=runner or (lambda cfg, log=print: fake_result(cfg, Path("/tmp"))),
        opener=opener or (lambda argv: None),
        diagnose=lambda _plat: [],
    )


def body(response) -> dict:
    return json.loads(response.body)


class TestRoutes:
    def test_the_api_is_closed_without_the_token(self) -> None:
        """The server binds to loopback, but every page the user visits can also reach
        loopback. Without the token, any site they open could list their home directory."""
        app = make_app()
        assert app.handle("GET", "/api/files", {"path": "/"}, token=None).status == 403
        assert app.handle("GET", "/api/files", {"path": "/"}, token="wrong").status == 403
        assert app.handle("GET", "/api/options", token="tok").status == 200

    def test_the_page_itself_needs_no_token_because_it_carries_none_yet(self) -> None:
        response = make_app().handle("GET", "/", token=None)
        assert response.status == 200
        assert response.content_type.startswith("text/html")
        assert b"subtitler" in response.body

    def test_the_static_route_cannot_walk_out_of_its_directory(self) -> None:
        assert make_app().handle("GET", "/../../pyproject.toml", token=None).status == 404

    def test_an_unknown_route_is_a_404_and_not_a_500(self) -> None:
        assert make_app().handle("GET", "/api/nope", token="tok").status == 404

    def test_options_carries_the_platform_and_its_shortcuts(self) -> None:
        payload = body(make_app(plat=MAC).handle("GET", "/api/options", token="tok"))
        assert payload["platform"]["is_macos"] is True
        assert payload["reveal_label"] == "Show in Finder"
        assert payload["defaults"]["lang"] == "sr"

    def test_the_reveal_label_is_not_finder_on_linux(self) -> None:
        payload = body(make_app(plat=LINUX).handle("GET", "/api/options", token="tok"))
        assert payload["reveal_label"] == "Open folder"

    def test_preview_validates_without_starting_anything(self) -> None:
        app = make_app()
        ok = body(
            app.handle(
                "POST",
                "/api/preview",
                body=json.dumps({"input": str(FIXTURE), "model": "tiny"}).encode(),
                token="tok",
            )
        )
        assert ok["ok"] is True
        assert "--model tiny" in ok["command"]
        assert app.jobs.current is None

        bad = body(app.handle("POST", "/api/preview", body=b"{}", token="tok"))
        assert bad == {"ok": False, "field": "input", "error": "choose a file first"}

    def test_a_run_starts_a_job_and_reports_where_the_files_went(self) -> None:
        app = make_app()
        started = body(
            app.handle(
                "POST", "/api/run", body=json.dumps({"input": str(FIXTURE)}).encode(), token="tok"
            )
        )
        assert started["ok"] is True
        snap = body(app.handle("GET", "/api/job", {"id": started["id"]}, token="tok"))
        assert snap["status"] == "done"
        assert snap["result"]["srt"] == "/tmp/a.srt"
        assert snap["result"]["out_dir"] == "/tmp"

    def test_the_command_line_is_the_first_thing_in_the_log(self) -> None:
        """So a run that goes wrong can be reproduced in a terminal by copying one line."""
        app = make_app()
        started = body(
            app.handle(
                "POST",
                "/api/run",
                body=json.dumps({"input": str(FIXTURE), "srt_only": True}).encode(),
                token="tok",
            )
        )
        snap = body(app.handle("GET", "/api/job", {"id": started["id"]}, token="tok"))
        assert snap["lines"][0].startswith("$ subtitler run ")
        assert "--srt-only" in snap["lines"][0]

    def test_a_rejected_form_never_reaches_the_pipeline(self) -> None:
        def explode(cfg, log=print):  # pragma: no cover - must not be called
            raise AssertionError("the runner ran on an invalid form")

        app = make_app(runner=explode)
        response = app.handle("POST", "/api/run", body=b'{"input": "/nope.mp4"}', token="tok")
        assert response.status == 400
        assert body(response)["field"] == "input"

    def test_a_pipeline_failure_becomes_an_error_the_page_can_show(self) -> None:
        """`EngineUnavailable` is the common one: no model downloaded yet. The window has
        to say so rather than spin."""

        def unavailable(cfg, log=print):
            log("input: a.wav (10.0s, audio only)")
            raise RuntimeError("the large-v3 weights are not downloaded")

        app = make_app(runner=unavailable)
        started = body(
            app.handle(
                "POST", "/api/run", body=json.dumps({"input": str(FIXTURE)}).encode(), token="tok"
            )
        )
        snap = body(app.handle("GET", "/api/job", {"id": started["id"]}, token="tok"))
        assert snap["status"] == "error"
        assert "not downloaded" in snap["error"]

    def test_two_runs_at_once_are_refused_with_a_409(self) -> None:
        app = GuiApp(token="tok", plat=LINUX, spawn=lambda work: None, diagnose=lambda _p: [])
        first = app.handle(
            "POST", "/api/run", body=json.dumps({"input": str(FIXTURE)}).encode(), token="tok"
        )
        assert first.status == 200
        second = app.handle(
            "POST", "/api/run", body=json.dumps({"input": str(FIXTURE)}).encode(), token="tok"
        )
        assert second.status == 409

    def test_reveal_runs_the_platform_command_for_the_faked_platform(self) -> None:
        """The only way the mac branch of the results panel is ever exercised, since CI's
        macOS runner has no Finder to talk to either."""
        calls: list[list[str]] = []
        app = make_app(plat=MAC, opener=lambda argv: calls.append(list(argv)))
        response = app.handle(
            "POST", "/api/reveal", body=json.dumps({"path": str(FIXTURE)}).encode(), token="tok"
        )
        assert response.status == 200
        assert calls == [["open", "-R", str(FIXTURE)]]

    def test_reveal_refuses_a_path_that_is_not_there(self) -> None:
        response = make_app().handle("POST", "/api/reveal", body=b'{"path": "/nope"}', token="tok")
        assert response.status == 400

    def test_a_file_manager_that_will_not_start_is_a_message_not_a_crash(self) -> None:
        """xdg-open is not installed on every Linux box, and failing to open a folder must
        not read like a failure of the run that produced the folder."""

        def broken(_argv):
            raise OSError("No such file or directory: 'xdg-open'")

        response = make_app(opener=broken).handle(
            "POST", "/api/reveal", body=json.dumps({"path": str(FIXTURE)}).encode(), token="tok"
        )
        assert response.status == 500
        assert str(FIXTURE.parent) in body(response)["error"]

    def test_the_files_route_reports_a_bad_path_as_a_400(self) -> None:
        response = make_app().handle("GET", "/api/files", {"path": "/no/such/dir"}, token="tok")
        assert response.status == 400

    def test_the_model_list_names_the_backend_this_machine_would_use(self) -> None:
        """Apple Silicon downloads mlx weights and everything else downloads
        faster-whisper ones. Offering the wrong ones is a 3 GB mistake."""
        assert (
            body(make_app(plat=MAC).handle("GET", "/api/models", token="tok"))["backend"] == "mlx"
        )
        linux = body(make_app(plat=LINUX).handle("GET", "/api/models", token="tok"))
        assert linux["backend"] == "faster-whisper"
        assert {m["name"] for m in linux["models"]} == {"large-v3", "tiny"}

    def test_an_unknown_model_download_is_refused_before_a_job_starts(self) -> None:
        app = make_app()
        response = app.handle(
            "POST", "/api/models/download", body=b'{"name": "gigantic"}', token="tok"
        )
        assert response.status == 400
        assert app.jobs.current is None


class TestDoctorRoute:
    def test_the_first_call_answers_immediately_and_checks_in_the_background(self) -> None:
        """`diagnose` shells out a dozen times and can take seconds. Blocking the first
        paint to tell the user their ffmpeg lacks libass trades one annoyance for another.
        """
        calls: list[Platform] = []

        def slow(plat: Platform):
            calls.append(plat)
            return []

        app = GuiApp(token="tok", plat=LINUX, spawn=inline, diagnose=slow)
        first = body(app.handle("GET", "/api/doctor", token="tok"))
        assert first["ready"] is True  # inline spawn: the work already happened
        assert calls == [LINUX]

    def test_the_report_is_cached_until_the_user_asks_again(self) -> None:
        calls: list[Platform] = []
        app = GuiApp(
            token="tok", plat=LINUX, spawn=inline, diagnose=lambda p: calls.append(p) or []
        )
        app.handle("GET", "/api/doctor", token="tok")
        app.handle("GET", "/api/doctor", token="tok")
        assert len(calls) == 1
        app.handle("GET", "/api/doctor", {"refresh": "1"}, token="tok")
        assert len(calls) == 2

    def test_a_check_that_raises_leaves_a_readable_panel_rather_than_a_spinner(self) -> None:
        def broken(_plat):
            raise OSError("nvidia-smi went away")

        app = GuiApp(token="tok", plat=LINUX, spawn=inline, diagnose=broken)
        payload = body(app.handle("GET", "/api/doctor", token="tok"))
        assert payload["ready"] is True
        assert "nvidia-smi went away" in payload["text"]


# --------------------------------------------------------------------------------------
# Over a real socket
# --------------------------------------------------------------------------------------


class TestHostHeader:
    def test_only_a_loopback_host_is_answered(self) -> None:
        """A page on the internet cannot read a cross-origin response, but it can point its
        own hostname at 127.0.0.1 and become same-origin. The token stops that; refusing an
        unexpected Host is the second lock and costs a string comparison."""
        assert is_local_host_header("127.0.0.1:8000")
        assert is_local_host_header("localhost:53211")
        assert is_local_host_header("[::1]:9")
        assert not is_local_host_header("evil.example.com:8000")
        assert not is_local_host_header(None)


class TestOverHttp:
    """The same code path a browser uses, on an ephemeral loopback port.

    Every route above is tested by calling `handle` directly; this class exists to prove
    the wiring in between is real, because that is the part no unit test can reach and no
    headless runner can click.
    """

    @pytest.fixture
    def server(self):
        app = make_app()
        srv, _ = build_server(host="127.0.0.1", port=0, app=app)
        thread = threading.Thread(target=srv.serve_forever, daemon=True)
        thread.start()
        yield f"http://127.0.0.1:{srv.server_address[1]}", app
        srv.shutdown()
        srv.server_close()

    @staticmethod
    def _get(url: str, token: str | None = "tok"):
        request = urllib.request.Request(url)
        if token:
            request.add_header("X-Subtitler-Token", token)
        return urllib.request.urlopen(request, timeout=10)

    def test_the_page_is_served(self, server) -> None:
        base, _ = server
        with self._get(f"{base}/", token=None) as response:
            assert response.status == 200
            assert b"<title>subtitler</title>" in response.read()

    def test_the_api_answers_json_with_the_token_header(self, server) -> None:
        base, _ = server
        with self._get(f"{base}/api/options") as response:
            payload = json.loads(response.read())
        assert payload["engines"][0] == "auto"

    def test_the_token_can_also_ride_in_the_query_string(self, server) -> None:
        """Which is how the page gets it in the first place: the URL subtitler opens."""
        base, _ = server
        with self._get(f"{base}/api/options?t=tok", token=None) as response:
            assert response.status == 200

    def test_a_request_without_the_token_is_refused_over_the_wire(self, server) -> None:
        base, _ = server
        with pytest.raises(urllib.error.HTTPError) as exc:
            self._get(f"{base}/api/options", token=None)
        assert exc.value.code == 403

    def test_a_run_can_be_driven_end_to_end_over_http(self, server) -> None:
        base, _app = server
        request = urllib.request.Request(
            f"{base}/api/run",
            data=json.dumps({"input": str(FIXTURE), "srt_only": True}).encode(),
            headers={"Content-Type": "application/json", "X-Subtitler-Token": "tok"},
        )
        with urllib.request.urlopen(request, timeout=10) as response:
            started = json.loads(response.read())
        assert started["ok"] is True

        with self._get(f"{base}/api/job?id={started['id']}&since=0") as response:
            snapshot = json.loads(response.read())
        assert snapshot["status"] == "done"
        assert snapshot["result"]["srt"] == "/tmp/a.srt"


# --------------------------------------------------------------------------------------
# Wiring
# --------------------------------------------------------------------------------------


def test_the_gui_subcommand_exists_and_defaults_to_loopback() -> None:
    from subtitler.cli import _HANDLERS, build_parser

    args = build_parser().parse_args(["gui"])
    assert args.command == "gui"
    assert args.host == "127.0.0.1"
    assert args.port == 0
    assert args.no_browser is False
    assert "gui" in _HANDLERS


def test_the_page_ships_with_the_package() -> None:
    """The HTML is package data, not a source file that only exists in a checkout. If the
    wheel stops carrying it, `subtitler gui` serves a 404 to the browser it just opened."""
    from subtitler.gui.app import STATIC_DIR

    assert (STATIC_DIR / "index.html").is_file()


def test_importing_the_gui_needs_no_display_and_starts_no_server() -> None:
    """Both CI runners are headless and every test in this file imports these modules.
    Nothing here may bind a socket, open a browser, or need DISPLAY at import time."""
    import subtitler.gui.app
    import subtitler.gui.server

    assert "tkinter" not in dir(subtitler.gui.app)
    assert "tkinter" not in dir(subtitler.gui.server)
