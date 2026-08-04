"""The stage cache as the pipeline actually uses it.

The engine and the burn are faked, because what is under test is which stages run, not
whether Whisper or libass work; those have their own tests and their own CI steps. ffmpeg
does run, for extraction and probing, so the work-directory wiring is real.

The criterion these defend is PRD acceptance 6: running the same command twice hits the
cache, and the second run produces identical output.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import pytest

from subtitler import burn as burn_mod
from subtitler import edits as edits_mod
from subtitler import media, pipeline, postedit
from subtitler.cues import CueConfig
from subtitler.model import Segment, Transcript, Word
from subtitler.pipeline import RunConfig, run_pipeline

FIXTURE = Path(__file__).resolve().parent.parent / "fixtures" / "tiny-10s.wav"
needs_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")

pytestmark = needs_ffmpeg


class FakeEngine:
    """Counts calls, so a test can assert a stage did not run rather than that it was fast."""

    name = "fake"
    kind = "local"

    def __init__(self, text: str = "Ovo je proba srpskog teksta za titlove"):
        self.calls = 0
        self.text = text

    def describe(self) -> dict:
        return {"engine": self.name, "model": "fake-1", "device": "cpu"}

    def availability(self):  # pragma: no cover - never consulted, resolve() is patched
        raise NotImplementedError

    def ensure_model(self, progress=None):  # pragma: no cover
        raise NotImplementedError

    def transcribe(self, audio: Path, opts) -> Transcript:
        self.calls += 1
        tokens = self.text.split()
        step = 8.0 / len(tokens)
        words = tuple(
            Word(start=1.0 + i * step, end=1.0 + (i + 1) * step, text=t)
            for i, t in enumerate(tokens)
        )
        segment = Segment(start=1.0, end=9.0, text=self.text, words=words)
        return Transcript(
            language="sr",
            duration=10.0,
            segments=(segment,),
            engine=self.name,
            model="fake-1",
            runtime_s=0.5,
        )


class FakeBurner:
    """Stands in for the ffmpeg encode, which is five seconds of the two-second budget."""

    def __init__(self):
        self.calls = 0
        self.last: dict = {}

    def __call__(self, cues, dst: Path, **kwargs) -> Path:
        self.calls += 1
        self.last = kwargs
        self.cues = cues
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(b"fake mp4 " + str(len(cues)).encode())
        return dst


@pytest.fixture
def fakes(monkeypatch):
    engine = FakeEngine()
    burner = FakeBurner()
    monkeypatch.setattr(pipeline, "resolve", lambda *a, **k: engine)
    monkeypatch.setattr(burn_mod, "burn", burner)
    return engine, burner


def cfg(tmp_path: Path, **overrides) -> RunConfig:
    return RunConfig(input=FIXTURE, out_dir=tmp_path, engine="fake", **overrides)


class TestReRunIsFree:
    def test_second_run_skips_every_expensive_stage(self, tmp_path, fakes):
        engine, burner = fakes
        run_pipeline(cfg(tmp_path), log=lambda _m: None)
        assert (engine.calls, burner.calls) == (1, 1)

        second = run_pipeline(cfg(tmp_path), log=lambda _m: None)
        assert (engine.calls, burner.calls) == (1, 1)
        assert set(second.cached) == {"extract", "transcribe", "cues", "burn"}

    def test_second_run_output_is_byte_identical(self, tmp_path, fakes):
        """Byte-identical, not merely equivalent.

        The SRT and VTT are deliberately re-rendered on every run rather than cached, so
        this re-derives them from cues.json and compares. A cache that returned stale text,
        or a renderer that was not deterministic, fails here.
        """
        run_pipeline(cfg(tmp_path), log=lambda _m: None)
        before = {p.name: p.read_bytes() for p in sorted(tmp_path.glob("tiny-10s.*"))}
        assert before

        run_pipeline(cfg(tmp_path), log=lambda _m: None)
        after = {p.name: p.read_bytes() for p in sorted(tmp_path.glob("tiny-10s.*"))}
        assert after == before

    def test_the_work_directory_holds_one_meta_per_stage(self, tmp_path, fakes):
        run_pipeline(cfg(tmp_path), log=lambda _m: None)
        work = tmp_path / ".subtitler" / "tiny-10s"
        names = {p.name for p in work.iterdir()}
        assert names == {
            "extract.wav",
            "extract.meta.json",
            "transcribe.json",
            "transcribe.meta.json",
            "cues.json",
            "cues.meta.json",
            "burn.meta.json",
        }

    def test_nothing_is_written_outside_the_output_directory(self, tmp_path, fakes):
        """Non-negotiable 4: the code this replaced dropped temp_audio.wav into the CWD."""
        run_pipeline(cfg(tmp_path), log=lambda _m: None)
        assert (tmp_path / ".subtitler" / "tiny-10s" / "extract.wav").exists()
        assert not (Path.cwd() / "extract.wav").exists()


class TestInvalidation:
    def test_force_transcribe_reruns_transcribe_and_burn(self, tmp_path, fakes):
        engine, burner = fakes
        run_pipeline(cfg(tmp_path), log=lambda _m: None)
        run_pipeline(cfg(tmp_path, force="transcribe"), log=lambda _m: None)
        assert (engine.calls, burner.calls) == (2, 2)

    def test_force_burn_leaves_the_transcript_alone(self, tmp_path, fakes):
        engine, burner = fakes
        run_pipeline(cfg(tmp_path), log=lambda _m: None)
        run_pipeline(cfg(tmp_path, force="burn"), log=lambda _m: None)
        assert (engine.calls, burner.calls) == (1, 2)

    def test_bare_force_reruns_everything(self, tmp_path, fakes):
        engine, burner = fakes
        run_pipeline(cfg(tmp_path), log=lambda _m: None)
        result = run_pipeline(cfg(tmp_path, force="all"), log=lambda _m: None)
        assert (engine.calls, burner.calls) == (2, 2)
        assert result.cached == ()

    def test_a_cue_setting_change_does_not_retranscribe(self, tmp_path, fakes):
        """The point of chaining keys instead of hashing the whole command line.

        `--max-line 30` changes the cues and therefore the burn; it cannot change the
        words, and re-running a 75-second transcription for it would be the whole reason
        the cache exists, undone.
        """
        engine, burner = fakes
        run_pipeline(cfg(tmp_path), log=lambda _m: None)
        run_pipeline(cfg(tmp_path, cues=CueConfig(max_line=30)), log=lambda _m: None)
        assert (engine.calls, burner.calls) == (1, 2)

    def test_a_style_change_only_reburns(self, tmp_path, fakes):
        engine, burner = fakes
        run_pipeline(cfg(tmp_path), log=lambda _m: None)
        second = run_pipeline(cfg(tmp_path, style_preset="box"), log=lambda _m: None)
        assert (engine.calls, burner.calls) == (1, 2)
        assert "cues" in second.cached

    def test_a_new_engine_retranscribes(self, tmp_path, fakes, monkeypatch):
        """`describe()` is in the key, so a different model or device is a different run.

        Keying on `--model` alone would serve a large-v3 int8 CPU transcript for a run that
        asked for float16 on CUDA, which is a different transcript.
        """
        engine, _ = fakes
        run_pipeline(cfg(tmp_path), log=lambda _m: None)

        other = FakeEngine()
        other.describe = lambda: {"engine": "fake", "model": "fake-2", "device": "cpu"}
        monkeypatch.setattr(pipeline, "resolve", lambda *a, **k: other)
        run_pipeline(cfg(tmp_path), log=lambda _m: None)
        assert (engine.calls, other.calls) == (1, 1)

    def test_a_changed_source_file_invalidates_everything(self, tmp_path, fakes):
        """Same path, same name, different bytes. This is what content addressing buys."""
        engine, _ = fakes
        local = tmp_path / "clip.wav"
        local.write_bytes(FIXTURE.read_bytes())
        run_pipeline(RunConfig(input=local, out_dir=tmp_path, engine="fake"), log=lambda _m: None)

        # Half the fixture: different content, different length, same path.
        raw = FIXTURE.read_bytes()
        local.write_bytes(raw[: 44 + (len(raw) - 44) // 2])
        run_pipeline(RunConfig(input=local, out_dir=tmp_path, engine="fake"), log=lambda _m: None)
        assert engine.calls == 2

    def test_unknown_force_stage_is_rejected_before_any_work(self, tmp_path, fakes):
        engine, _ = fakes
        with pytest.raises(ValueError, match="unknown stage"):
            run_pipeline(cfg(tmp_path, force="transcript"), log=lambda _m: None)
        assert engine.calls == 0
        assert not (tmp_path / ".subtitler").exists()


class TestDenoiseStage:
    def test_denoise_is_its_own_cached_stage(self, tmp_path, fakes):
        result = run_pipeline(cfg(tmp_path, denoise="afftdn"), log=lambda _m: None)
        work = tmp_path / ".subtitler" / "tiny-10s"
        assert (work / "denoise.wav").exists()
        assert (work / "denoise.meta.json").exists()
        assert result.cached == ()

        again = run_pipeline(cfg(tmp_path, denoise="afftdn"), log=lambda _m: None)
        assert "denoise" in again.cached

    def test_switching_the_preset_reuses_the_extraction(self, tmp_path, fakes):
        """The reason denoise was split out of extract.

        On a 3 GB video the extraction is the expensive half. Changing the denoiser must
        not demux the source again, and the Phase 7 engine x denoiser matrix depends on it.
        """
        run_pipeline(cfg(tmp_path, denoise="afftdn"), log=lambda _m: None)
        second = run_pipeline(cfg(tmp_path, denoise="anlmdn"), log=lambda _m: None)
        assert second.cached == ("extract",)

    def test_no_denoise_leaves_no_denoise_artifact(self, tmp_path, fakes):
        run_pipeline(cfg(tmp_path), log=lambda _m: None)
        assert not (tmp_path / ".subtitler" / "tiny-10s" / "denoise.wav").exists()


class TestDryRun:
    def test_a_dry_run_writes_no_cache(self, tmp_path, fakes):
        """A dry run prints commands instead of running them.

        Committing a meta for a stage that never produced its artifact would make the next
        real run skip work it has not done.
        """
        run_pipeline(cfg(tmp_path, dry_run=True), log=lambda _m: None)
        work = tmp_path / ".subtitler" / "tiny-10s"
        assert not work.exists() or not any(work.glob("*.meta.json"))

    def test_a_dry_run_does_not_consume_the_cache(self, tmp_path, fakes):
        engine, _ = fakes
        run_pipeline(cfg(tmp_path), log=lambda _m: None)
        result = run_pipeline(cfg(tmp_path, dry_run=True), log=lambda _m: None)
        assert result.cached == ()
        assert engine.calls == 1


class TestFixStage:
    """`--fix` as a pipeline stage: cached, chained, and unable to move a timestamp.

    The model is a stub. What is under test is the wiring: that the corrected cues reach
    the renderer and the burn, that a second run does not pay for the same tokens twice,
    and that the clock survives the round trip through the cache.
    """

    @staticmethod
    def _fix_cfg(**overrides) -> postedit.FixConfig:
        return postedit.FixConfig(workers=1, **overrides)

    @pytest.fixture
    def model(self, monkeypatch):
        """A stub model that upper-cases every cue and counts the calls it was billed.

        Patched at `litellm_completer`, not at `_litellm_complete`. The former is what
        binds the config to a callable, and it imports LiteLLM eagerly so a machine without
        the `fix` extra is told once instead of once per batch. Patching below that point
        made the whole class fail on CI, which syncs without the extra on purpose.
        """
        calls = []

        def complete(_system: str, user: str) -> str:
            calls.append(user)
            items = json.loads(user)
            return json.dumps(
                [{"i": it["i"], "text": it["text"].upper()} for it in items], ensure_ascii=False
            )

        monkeypatch.setattr(postedit, "litellm_completer", lambda _cfg: complete)
        return calls

    def test_fix_changes_the_text_and_moves_no_timestamp(self, tmp_path, fakes, model):
        """The Phase 6 acceptance criterion, end to end through the real pipeline.

        Same run twice, once without `--fix` and once with: the SRT text differs, and every
        `-->` line is byte-identical between the two files.
        """
        plain = run_pipeline(cfg(tmp_path / "plain"), log=lambda _m: None)
        fixed = run_pipeline(cfg(tmp_path / "fixed", fix=self._fix_cfg()), log=lambda _m: None)

        plain_srt = plain.srt.read_text(encoding="utf-8")
        fixed_srt = fixed.srt.read_text(encoding="utf-8")
        assert plain_srt != fixed_srt

        clocks = lambda text: [ln for ln in text.splitlines() if "-->" in ln]  # noqa: E731
        assert clocks(plain_srt) == clocks(fixed_srt)
        assert [c.text for c in fixed.cues] == [c.text.upper() for c in plain.cues]

    def test_the_vtt_gets_the_same_treatment_as_the_srt(self, tmp_path, fakes, model):
        plain = run_pipeline(cfg(tmp_path / "plain"), log=lambda _m: None)
        fixed = run_pipeline(cfg(tmp_path / "fixed", fix=self._fix_cfg()), log=lambda _m: None)
        plain_vtt = plain.vtt.read_text(encoding="utf-8")
        fixed_vtt = fixed.vtt.read_text(encoding="utf-8")
        assert [ln for ln in plain_vtt.splitlines() if "-->" in ln] == [
            ln for ln in fixed_vtt.splitlines() if "-->" in ln
        ]
        assert plain_vtt != fixed_vtt

    def test_a_second_run_does_not_pay_for_the_same_tokens_twice(self, tmp_path, fakes, model):
        """`fix` is the only stage that costs money, so a cache miss here is the expensive one."""
        run_pipeline(cfg(tmp_path, fix=self._fix_cfg()), log=lambda _m: None)
        billed = len(model)
        assert billed > 0

        second = run_pipeline(cfg(tmp_path, fix=self._fix_cfg()), log=lambda _m: None)
        assert len(model) == billed
        assert "fix" in second.cached

    def test_the_fix_stage_writes_one_artifact_and_one_meta(self, tmp_path, fakes, model):
        run_pipeline(cfg(tmp_path, fix=self._fix_cfg()), log=lambda _m: None)
        work = tmp_path / ".subtitler" / "tiny-10s"
        assert (work / "fix.json").exists()
        assert (work / "fix.meta.json").exists()

    def test_no_fix_flag_means_no_fix_artifact_and_no_model_call(self, tmp_path, fakes, model):
        run_pipeline(cfg(tmp_path), log=lambda _m: None)
        assert not (tmp_path / ".subtitler" / "tiny-10s" / "fix.json").exists()
        assert model == []

    def test_the_burn_sees_the_corrected_cues(self, tmp_path, fakes, model):
        """`fix` chains between `cues` and `burn`, so turning it on must re-burn.

        A cached burn built from the uncorrected cues would put the wrong text on screen
        while every sidecar file showed the right one.
        """
        _, burner = fakes
        run_pipeline(cfg(tmp_path), log=lambda _m: None)
        assert burner.calls == 1
        run_pipeline(cfg(tmp_path, fix=self._fix_cfg()), log=lambda _m: None)
        assert burner.calls == 2

    def test_switching_the_fix_model_reruns_only_the_fix_and_the_burn(self, tmp_path, fakes, model):
        engine, burner = fakes
        run_pipeline(cfg(tmp_path, fix=self._fix_cfg()), log=lambda _m: None)
        result = run_pipeline(
            cfg(tmp_path, fix=self._fix_cfg(model="openai/gpt-4o")), log=lambda _m: None
        )
        assert engine.calls == 1
        assert burner.calls == 2
        assert set(result.cached) == {"extract", "transcribe", "cues"}

    def test_force_cues_takes_the_fix_with_it(self, tmp_path, fakes, model):
        run_pipeline(cfg(tmp_path, fix=self._fix_cfg()), log=lambda _m: None)
        billed = len(model)
        run_pipeline(cfg(tmp_path, fix=self._fix_cfg(), force="cues"), log=lambda _m: None)
        assert len(model) == 2 * billed

    def test_editing_the_prompt_file_invalidates_the_correction(self, tmp_path, fakes, model):
        """The prompt's content is in the key, not just its name."""
        prompt = tmp_path / "p.md"
        prompt.write_text("Correct the text.", encoding="utf-8")
        run_pipeline(cfg(tmp_path, fix=self._fix_cfg(prompt=str(prompt))), log=lambda _m: None)
        billed = len(model)

        prompt.write_text("Correct the text, and be strict.", encoding="utf-8")
        run_pipeline(cfg(tmp_path, fix=self._fix_cfg(prompt=str(prompt))), log=lambda _m: None)
        assert len(model) > billed

    def test_the_json_summary_reports_what_the_pass_did(self, tmp_path, fakes, model):
        result = run_pipeline(cfg(tmp_path, fix=self._fix_cfg()), log=lambda _m: None)
        assert result.to_dict()["fix"]["changed_cues"] == len(result.cues)
        assert result.to_dict()["fix"]["rejected_batches"] == 0

    def test_a_warm_run_still_reports_what_the_cold_run_did(self, tmp_path, fakes, model):
        """The report is stored with the cues, so `--json` does not go blank on a re-run."""
        cold = run_pipeline(cfg(tmp_path, fix=self._fix_cfg()), log=lambda _m: None)
        warm = run_pipeline(cfg(tmp_path, fix=self._fix_cfg()), log=lambda _m: None)
        assert warm.to_dict()["fix"] == cold.to_dict()["fix"]

    def test_a_run_without_fix_never_imports_litellm(self, tmp_path, fakes, monkeypatch):
        """LiteLLM costs about 1.7s to import and lives behind an extra CI does not sync."""
        monkeypatch.delitem(sys.modules, "litellm", raising=False)
        run_pipeline(cfg(tmp_path), log=lambda _m: None)
        assert "litellm" not in sys.modules


class TestTrimStage:
    """`--start`/`--end`, and the two things about them that are easy to get wrong."""

    def test_the_recognizer_is_given_the_fragment_not_the_whole_file(self, tmp_path, fakes):
        """Trimming before extraction is what makes cue timestamps relative for free.

        The extracted WAV is three seconds long, so every timestamp the engine produces is
        measured from the start of the fragment and nothing downstream has to subtract
        anything. Extracting first and cutting later would leave every cue offset by the
        start time, and nothing in the output would say so.
        """
        run_pipeline(cfg(tmp_path, start="2", end="5"), log=lambda _m: None)
        work = tmp_path / ".subtitler" / "tiny-10s"
        assert _duration(work / "trim.wav") == pytest.approx(3.0, abs=0.3)
        assert _duration(work / "extract.wav") == pytest.approx(3.0, abs=0.3)

    def test_the_burn_gets_the_trimmed_media(self, tmp_path, fakes):
        """Otherwise the exported video is the full-length source carrying subtitles that
        match three seconds of it."""
        _, burner = fakes
        run_pipeline(cfg(tmp_path, start="2", end="5"), log=lambda _m: None)
        used = burner.last.get("video") or burner.last.get("audio")
        assert used == tmp_path / ".subtitler" / "tiny-10s" / "trim.wav"
        assert burner.last["duration"] == pytest.approx(3.0, abs=0.3)

    def test_trim_is_its_own_cached_stage(self, tmp_path, fakes):
        result = run_pipeline(cfg(tmp_path, start="2", end="5"), log=lambda _m: None)
        work = tmp_path / ".subtitler" / "tiny-10s"
        assert (work / "trim.meta.json").exists()
        assert "trim" not in result.cached

        again = run_pipeline(cfg(tmp_path, start="2", end="5"), log=lambda _m: None)
        assert "trim" in again.cached

    def test_no_window_means_no_trim_artifact(self, tmp_path, fakes):
        """The untrimmed path must be exactly what it was before this stage existed, keys
        included, or every cached run in the world misses once for nothing."""
        run_pipeline(cfg(tmp_path), log=lambda _m: None)
        work = tmp_path / ".subtitler" / "tiny-10s"
        assert not (work / "trim.wav").exists()
        assert not (work / "trim.meta.json").exists()

    def test_moving_the_window_retranscribes(self, tmp_path, fakes):
        engine, _ = fakes
        run_pipeline(cfg(tmp_path, start="2", end="5"), log=lambda _m: None)
        run_pipeline(cfg(tmp_path, start="3", end="6"), log=lambda _m: None)
        assert engine.calls == 2

    def test_an_end_before_the_start_is_rejected_before_any_work(self, tmp_path, fakes):
        engine, _ = fakes
        with pytest.raises(pipeline.media.MediaError, match="must be after"):
            run_pipeline(cfg(tmp_path, start="5", end="2"), log=lambda _m: None)
        assert engine.calls == 0
        assert not (tmp_path / ".subtitler").exists()

    def test_a_nonsense_timecode_is_rejected_before_any_work(self, tmp_path, fakes):
        with pytest.raises(pipeline.media.MediaError, match="HH:MM:SS"):
            run_pipeline(cfg(tmp_path, start="ten past"), log=lambda _m: None)
        assert not (tmp_path / ".subtitler").exists()

    def test_a_start_past_the_end_says_how_long_the_source_is(self, tmp_path, fakes):
        """Regression: `--start` past the end of the source made ffmpeg write a few hundred
        bytes of container header and exit 0, and the failure surfaced as a raw ffprobe dump
        from the next stage. The duration is the one fact that makes the timecode obviously
        wrong, so it is in the message."""
        engine, _ = fakes
        with pytest.raises(pipeline.media.MediaError) as caught:
            run_pipeline(cfg(tmp_path, start="30"), log=lambda _m: None)
        message = str(caught.value)
        assert "at or past the end" in message
        assert media.format_timecode(media.probe(FIXTURE).duration) in message
        assert engine.calls == 0

    def test_a_fragment_that_does_not_probe_is_never_committed(self, tmp_path, fakes, monkeypatch):
        """Regression: the `trim` stage committed on ffmpeg's exit code alone, so one bad
        window turned a work directory into one that printed `trim: cached` and died in
        ffprobe on every run afterwards, forever.

        The husk is written here the way ffmpeg wrote it: file present, exit code 0.
        """

        def husk(src, dst, *, start=0.0, end=None, dry_run=False):
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_bytes(b"\xff\xfb" + b"\x00" * 347)
            return dst

        monkeypatch.setattr(pipeline.media, "trim", husk)
        with pytest.raises(pipeline.media.MediaError, match="ffprobe cannot"):
            run_pipeline(cfg(tmp_path, start="2", end="5"), log=lambda _m: None)
        work = tmp_path / ".subtitler" / "tiny-10s"
        assert (work / "trim.wav").exists()
        assert not (work / "trim.meta.json").exists()

    def test_a_cache_poisoned_by_the_old_behaviour_recovers(self, tmp_path, fakes):
        """The other half: a work directory that already holds a committed husk, written by
        a version that trusted the exit code. It must be re-cut, not served."""
        run_pipeline(cfg(tmp_path, start="2", end="5"), log=lambda _m: None)
        work = tmp_path / ".subtitler" / "tiny-10s"
        assert (work / "trim.meta.json").exists()
        (work / "trim.wav").write_bytes(b"\xff\xfb" + b"\x00" * 347)  # poison, meta intact

        lines: list[str] = []
        run_pipeline(cfg(tmp_path, start="2", end="5"), log=lines.append)
        assert any("unreadable" in line for line in lines)
        assert _duration(work / "trim.wav") == pytest.approx(3.0, abs=0.3)


def _duration(path: Path) -> float:
    return pipeline.media.probe(path).duration


class FakeFetcher:
    """Stands in for yt-dlp. Copies the fixture in rather than downloading one.

    CI never touches YouTube: a test that downloads is flaky, rate-limited and slow. What
    is verified here is the wiring around the download (the key, the shape asked for, the
    span asked for, where it lands), and the real thing is verified by hand and recorded in
    the README.

    A windowed call cuts the fixture, because that is what the real one now does: yt-dlp is
    handed a `Section` and its ffmpeg downloader seeks the remote file, so what lands on
    disk is already the fragment and the `trim` stage has nothing to do.
    """

    def __init__(self):
        self.calls: list[str] = []
        self.windows: list[tuple[float, float | None]] = []

    def __call__(self, url, dst_dir, *, kind="video", progress=None, start=0.0, end=None):
        from subtitler.fetch import Fetched

        self.calls.append(kind)
        self.windows.append((start, end))
        dst_dir = Path(dst_dir)
        dst_dir.mkdir(parents=True, exist_ok=True)
        path = dst_dir / "fetch.wav"
        windowed = bool(start) or end is not None
        if windowed:
            media.trim(FIXTURE, path, start=start, end=end)
        else:
            path.write_bytes(FIXTURE.read_bytes())
        if progress:
            progress("fetching: 100%")
        return Fetched(
            path=path,
            url=url,
            id="vid123",
            title="Neki Naslov",
            kind=kind,
            section_start=start if windowed else None,
            section_end=end,
        )


@pytest.fixture
def fetcher(monkeypatch):
    fake = FakeFetcher()
    monkeypatch.setattr(pipeline.fetch_mod, "fetch", fake)
    return fake


URL = "https://www.youtube.com/watch?v=vid123"


class TestFetchStage:
    def test_a_url_run_names_its_outputs_from_the_title(self, tmp_path, fakes, fetcher):
        result = run_pipeline(
            RunConfig(url=URL, out_dir=tmp_path, engine="fake"), log=lambda _m: None
        )
        assert result.srt == tmp_path / "Neki-Naslov.srt"
        assert result.srt.exists()
        assert result.to_dict()["source"] == URL

    def test_the_download_lands_in_the_work_directory_not_the_cwd(self, tmp_path, fakes, fetcher):
        """Non-negotiable 4. A URL has no directory of its own to write beside, which is
        exactly the case where writing into the CWD is tempting."""
        run_pipeline(RunConfig(url=URL, out_dir=tmp_path, engine="fake"), log=lambda _m: None)
        work = tmp_path / ".subtitler" / pipeline.fetch_mod.work_stem(URL)
        assert (work / "fetch.wav").exists()
        assert (work / "fetch.json").exists()
        assert not (Path.cwd() / "fetch.wav").exists()

    def test_a_url_run_without_an_output_directory_says_so(self, tmp_path, fakes, fetcher):
        with pytest.raises(pipeline.media.MediaError, match="-o DIR"):
            run_pipeline(RunConfig(url=URL, engine="fake"), log=lambda _m: None)
        assert fetcher.calls == []

    def test_a_second_run_does_not_download_again(self, tmp_path, fakes, fetcher):
        for _ in range(2):
            run_pipeline(RunConfig(url=URL, out_dir=tmp_path, engine="fake"), log=lambda _m: None)
        assert fetcher.calls == ["video"]

    def test_the_window_is_asked_of_the_site_and_never_cut_twice(self, tmp_path, fakes, fetcher):
        """Regression: `--start`/`--end` were not passed to yt-dlp, so `run URL --start
        1:00:00 --end 1:01:00` downloaded a whole four-hour source to keep sixty seconds.

        The span goes to the download now, which also means the `trim` stage must not run:
        what landed already starts at `--start`, and cutting it again at an absolute
        1:00:00 would ask for an hour into a one-minute file.
        """
        run_pipeline(
            RunConfig(url=URL, out_dir=tmp_path, engine="fake", start="2", end="5"),
            log=lambda _m: None,
        )
        assert fetcher.windows == [(2.0, 5.0)]

        work = tmp_path / ".subtitler" / pipeline.fetch_mod.work_stem(URL)
        assert not (work / "trim.meta.json").exists()
        assert _duration(work / "fetch.wav") == pytest.approx(3.0, abs=0.3)
        assert _duration(work / "extract.wav") == pytest.approx(3.0, abs=0.3)

    def test_moving_the_window_fetches_the_new_window(self, tmp_path, fakes, fetcher):
        """The deliberate reversal that came with passing the range through.

        `fetch` used to leave the window out of its key so that moving `--start` re-cut a
        cached download. But the download was the entire source, which is the cost this
        stage exists to avoid; now it is the window, so a different window is a different
        file and has to be fetched. What is re-fetched is three seconds, not four hours.
        """
        run_pipeline(
            RunConfig(url=URL, out_dir=tmp_path, engine="fake", start="2", end="5"),
            log=lambda _m: None,
        )
        result = run_pipeline(
            RunConfig(url=URL, out_dir=tmp_path, engine="fake", start="3", end="6"),
            log=lambda _m: None,
        )
        assert fetcher.windows == [(2.0, 5.0), (3.0, 6.0)]
        assert "fetch" not in result.cached

    def test_a_second_run_of_the_same_window_does_not_download_again(
        self, tmp_path, fakes, fetcher
    ):
        for _ in range(2):
            run_pipeline(
                RunConfig(url=URL, out_dir=tmp_path, engine="fake", start="2", end="5"),
                log=lambda _m: None,
            )
        assert fetcher.calls == ["video"]

    def test_srt_only_asks_for_audio(self, tmp_path, fakes, fetcher):
        """Downloading 1080p to produce a text file spends the user's bandwidth on
        nothing, and only this run knows that no pixel will ever be looked at."""
        run_pipeline(
            RunConfig(url=URL, out_dir=tmp_path, engine="fake", srt_only=True),
            log=lambda _m: None,
        )
        assert fetcher.calls == ["audio"]

    def test_switching_to_a_burn_fetches_the_video(self, tmp_path, fakes, fetcher):
        """The audio already on disk has no pixels in it, so this miss is correct."""
        run_pipeline(
            RunConfig(url=URL, out_dir=tmp_path, engine="fake", srt_only=True),
            log=lambda _m: None,
        )
        run_pipeline(RunConfig(url=URL, out_dir=tmp_path, engine="fake"), log=lambda _m: None)
        assert fetcher.calls == ["audio", "video"]

    def test_force_fetch_downloads_again(self, tmp_path, fakes, fetcher):
        """The escape hatch for the one thing the key cannot see: an upload that changed
        behind a URL that did not."""
        run_pipeline(RunConfig(url=URL, out_dir=tmp_path, engine="fake"), log=lambda _m: None)
        run_pipeline(
            RunConfig(url=URL, out_dir=tmp_path, engine="fake", force="fetch"),
            log=lambda _m: None,
        )
        assert fetcher.calls == ["video", "video"]

    def test_a_corrupt_record_downloads_again_instead_of_crashing(self, tmp_path, fakes, fetcher):
        run_pipeline(RunConfig(url=URL, out_dir=tmp_path, engine="fake"), log=lambda _m: None)
        work = tmp_path / ".subtitler" / pipeline.fetch_mod.work_stem(URL)
        (work / "fetch.json").write_text("{truncated", encoding="utf-8")
        run_pipeline(RunConfig(url=URL, out_dir=tmp_path, engine="fake"), log=lambda _m: None)
        assert fetcher.calls == ["video", "video"]

    def test_a_deleted_download_is_fetched_again(self, tmp_path, fakes, fetcher):
        """The meta may be valid while the file it describes is gone."""
        run_pipeline(RunConfig(url=URL, out_dir=tmp_path, engine="fake"), log=lambda _m: None)
        work = tmp_path / ".subtitler" / pipeline.fetch_mod.work_stem(URL)
        (work / "fetch.wav").unlink()
        run_pipeline(RunConfig(url=URL, out_dir=tmp_path, engine="fake"), log=lambda _m: None)
        assert fetcher.calls == ["video", "video"]

    def test_progress_reaches_the_run_log(self, tmp_path, fakes, fetcher):
        """The GUI streams this log line by line, so the download has to appear in it."""
        lines: list[str] = []
        run_pipeline(RunConfig(url=URL, out_dir=tmp_path, engine="fake"), log=lines.append)
        assert any("fetching: 100%" in line for line in lines)


class TestFromSource:
    """One place decides whether the user typed a path or a URL."""

    def test_a_url_becomes_a_url(self):
        assert RunConfig.from_source(URL).url == URL
        assert RunConfig.from_source(URL).input is None

    def test_a_path_becomes_a_path(self):
        assert RunConfig.from_source("clip.mp4").input == Path("clip.mp4")
        assert RunConfig.from_source("clip.mp4").url is None

    def test_the_other_options_pass_through(self):
        assert RunConfig.from_source(URL, start="1:00", srt_only=True).start == "1:00"

    def test_a_file_run_never_imports_yt_dlp(self, tmp_path, fakes, monkeypatch):
        """yt-dlp lives behind an extra CI does not sync, so a file run must not touch it.

        Same rule as LiteLLM: an optional dependency that a common path imports is not
        optional, it is a dependency with a broken install line.
        """
        monkeypatch.delitem(sys.modules, "yt_dlp", raising=False)
        run_pipeline(cfg(tmp_path, start="2", end="5"), log=lambda _m: None)
        assert "yt_dlp" not in sys.modules


class TestReviewStop:
    """`--review` stops once the subtitle files exist, before anything is encoded."""

    def test_the_subtitles_are_written_and_no_video_is(self, tmp_path, fakes):
        _engine, burner = fakes
        result = run_pipeline(cfg(tmp_path, review=True), log=lambda _m: None)
        assert result.srt is not None and result.srt.exists()
        assert result.video is None
        assert burner.calls == 0

    def test_the_cues_and_the_key_they_were_made_against_come_back(self, tmp_path, fakes):
        """The editor records its corrections against this key, so a run that did not hand
        one out could not have its corrections checked for staleness later."""
        result = run_pipeline(cfg(tmp_path, review=True), log=lambda _m: None)
        assert result.cues
        assert len(result.cues_key) == 16

    def test_approving_afterwards_pays_only_for_the_burn(self, tmp_path, fakes):
        """The whole point of stopping rather than running a second, separate pipeline: a
        45-minute transcription must not happen twice because the user read the text."""
        engine, burner = fakes
        run_pipeline(cfg(tmp_path, review=True), log=lambda _m: None)
        assert (engine.calls, burner.calls) == (1, 0)

        second = run_pipeline(cfg(tmp_path), log=lambda _m: None)
        assert (engine.calls, burner.calls) == (1, 1)
        assert set(second.cached) >= {"extract", "transcribe", "cues"}

    def test_a_url_review_still_fetches_the_video(self, tmp_path, fakes, fetcher):
        """`wants_video` must not be answered by "is this run going to burn right now".
        A review run that downloaded audio only would make the approval re-download."""
        run_pipeline(
            RunConfig(url=URL, out_dir=tmp_path, engine="fake", review=True),
            log=lambda _m: None,
        )
        assert fetcher.calls == ["video"]


class TestEditStage:
    """Hand corrections: applied, survived, and correctly refused when they go stale."""

    @staticmethod
    def _review(tmp_path, **overrides):
        return run_pipeline(cfg(tmp_path, review=True, **overrides), log=lambda _m: None)

    def test_a_correction_reaches_the_srt_and_the_burn(self, tmp_path, fakes):
        _engine, burner = fakes
        first = self._review(tmp_path)
        work = pipeline.work_dir(cfg(tmp_path))
        edits_mod.save(work, edits_mod.build(first.cues_key, {"1": "Sasvim drugačiji tekst"}))

        result = run_pipeline(cfg(tmp_path), log=lambda _m: None)
        assert result.cues[0].text == "Sasvim drugačiji tekst"
        assert "Sasvim drugačiji tekst" in result.srt.read_text(encoding="utf-8")
        assert burner.calls == 1
        # And the pixels, not merely the text file beside them.
        assert burner.cues[0].text == "Sasvim drugačiji tekst"

    def test_the_corrections_survive_a_re_run_of_the_cues_stage(self, tmp_path, fakes):
        """The trap this design exists for. `cues.json` is the `cues` stage's artifact, so
        anything written into it is recomputed away on the next run without a word. The
        corrections live in a file no stage writes."""
        first = self._review(tmp_path)
        work = pipeline.work_dir(cfg(tmp_path))
        edits_mod.save(work, edits_mod.build(first.cues_key, {"1": "Ostaje posle ponovnog rada"}))

        run_pipeline(cfg(tmp_path, force="cues"), log=lambda _m: None)
        again = run_pipeline(cfg(tmp_path), log=lambda _m: None)
        assert again.cues[0].text == "Ostaje posle ponovnog rada"

    def test_a_second_run_with_the_same_corrections_does_not_re_burn(self, tmp_path, fakes):
        """The other half of the trap: putting the corrections in the `cues` key would
        re-run the burn every time the editor was opened."""
        _engine, burner = fakes
        first = self._review(tmp_path)
        work = pipeline.work_dir(cfg(tmp_path))
        edits_mod.save(work, edits_mod.build(first.cues_key, {"1": "Jedna izmena"}))

        run_pipeline(cfg(tmp_path), log=lambda _m: None)
        assert burner.calls == 1
        second = run_pipeline(cfg(tmp_path), log=lambda _m: None)
        assert burner.calls == 1
        assert "burn" in second.cached and "edit" in second.cached

    def test_changing_the_text_re_burns_and_changing_nothing_else_does(self, tmp_path, fakes):
        _engine, burner = fakes
        first = self._review(tmp_path)
        work = pipeline.work_dir(cfg(tmp_path))
        edits_mod.save(work, edits_mod.build(first.cues_key, {"1": "Prva verzija"}))
        run_pipeline(cfg(tmp_path), log=lambda _m: None)
        assert burner.calls == 1

        edits_mod.save(work, edits_mod.build(first.cues_key, {"1": "Druga verzija"}))
        run_pipeline(cfg(tmp_path), log=lambda _m: None)
        assert burner.calls == 2

    def test_a_correction_made_against_another_transcript_is_reported_not_applied(
        self, tmp_path, fakes
    ):
        """Cue 1 of the old transcript is not cue 1 of the new one. Re-pointing the
        corrections at whatever now holds that index is the worst available outcome, so
        they are named in the log and skipped."""
        first = self._review(tmp_path)
        work = pipeline.work_dir(cfg(tmp_path))
        edits_mod.save(work, edits_mod.build(first.cues_key, {"1": "Za stari prepis"}))

        lines: list[str] = []
        # A different cue layout is a different `cues` key, exactly as a new model is.
        stale = run_pipeline(cfg(tmp_path, cues=CueConfig(max_line=20)), log=lines.append)
        assert stale.cues[0].text != "Za stari prepis"
        assert stale.edits == {"applied": [], "stale": 1, "base_key": first.cues_key}
        assert any("different transcript" in line for line in lines)

    def test_nothing_is_deleted_so_going_back_makes_them_apply_again(self, tmp_path, fakes):
        first = self._review(tmp_path)
        work = pipeline.work_dir(cfg(tmp_path))
        edits_mod.save(work, edits_mod.build(first.cues_key, {"1": "Vraca se"}))

        run_pipeline(cfg(tmp_path, cues=CueConfig(max_line=20)), log=lambda _m: None)
        back = run_pipeline(cfg(tmp_path), log=lambda _m: None)
        assert back.cues[0].text == "Vraca se"

    def test_a_run_with_no_corrections_writes_no_edit_stage_at_all(self, tmp_path, fakes):
        """So a user who never opens the editor gets byte-identical behaviour and the same
        cache keys as before this existed."""
        run_pipeline(cfg(tmp_path), log=lambda _m: None)
        work = pipeline.work_dir(cfg(tmp_path))
        assert not (work / "edit.meta.json").exists()
        assert not (work / "edited.json").exists()

    @pytest.mark.parametrize(
        ("fault", "text", "expected"),
        [
            ("invalid JSON", "{ not json", "not valid JSON"),
            (
                "a schema from another version",
                '{"schema_version": 99, "base_key": "k", "edits": []}',
                "schema_version",
            ),
            (
                '"edit" typed for "edits"',
                '{"schema_version": 1, "base_key": "k", "edit": [{"index": 1, "text": "a"}]}',
                'no "edits" key',
            ),
            (
                "no base_key",
                '{"schema_version": 1, "edits": [{"index": 1, "text": "a"}]}',
                "base_key",
            ),
        ],
    )
    def test_a_malformed_edits_file_stops_the_run_loudly(
        self, tmp_path, fakes, fault, text, expected
    ):
        """Regression: a malformed `edits.json` was read as "no corrections", so a typo in
        the one file a human is invited to open produced a run that reported success, burned
        the uncorrected words in, and said nothing at all.

        Loud on the same channel as the stale-key report, because a GUI run shows the user
        that log and not the traceback.
        """
        self._review(tmp_path)
        work = pipeline.work_dir(cfg(tmp_path))
        edits_mod.path_for(work).write_text(text, encoding="utf-8")

        lines: list[str] = []
        with pytest.raises(edits_mod.EditFileError, match=expected):
            run_pipeline(cfg(tmp_path), log=lines.append)
        assert any(expected in line for line in lines), lines
        assert any(edits_mod.EDITS_NAME in line for line in lines), f"{fault}: {lines}"

    def test_deleting_the_file_is_the_documented_way_back(self, tmp_path, fakes):
        """The escape hatch the message names has to be real: the run must not stay broken
        once the corrections are gone."""
        self._review(tmp_path)
        work = pipeline.work_dir(cfg(tmp_path))
        edits_mod.path_for(work).write_text("{ not json", encoding="utf-8")
        with pytest.raises(edits_mod.EditFileError):
            run_pipeline(cfg(tmp_path), log=lambda _m: None)

        edits_mod.clear(work)
        assert run_pipeline(cfg(tmp_path), log=lambda _m: None).srt.exists()


class TestSoftMux:
    """`--soft-mux` used to be accepted by both interfaces and read by neither."""

    @pytest.fixture
    def muxer(self, monkeypatch):
        calls: list[dict] = []

        def fake(src, subs, dst, *, language="srp", dry_run=False):
            calls.append({"src": src, "subs": subs, "dst": dst, "language": language})
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_bytes(b"muxed")
            return dst

        monkeypatch.setattr(burn_mod, "soft_mux", fake)
        return calls

    def test_the_flag_actually_produces_a_file(self, tmp_path, fakes, muxer):
        result = run_pipeline(cfg(tmp_path, soft_mux=True), log=lambda _m: None)
        assert result.muxed is not None and result.muxed.exists()
        assert result.to_dict()["muxed"] == str(result.muxed)
        assert len(muxer) == 1

    def test_without_the_flag_nothing_is_muxed(self, tmp_path, fakes, muxer):
        assert run_pipeline(cfg(tmp_path), log=lambda _m: None).muxed is None
        assert muxer == []

    def test_the_track_carries_the_rendered_srt(self, tmp_path, fakes, muxer):
        result = run_pipeline(cfg(tmp_path, soft_mux=True), log=lambda _m: None)
        assert muxer[0]["subs"] == result.srt
        assert muxer[0]["language"] == "srp"

    def test_an_audio_only_input_gets_the_track_on_the_burned_canvas(self, tmp_path, fakes, muxer):
        """There is no other picture to attach it to, and refusing would make the flag do
        nothing on exactly the inputs this tool is used for most."""
        result = run_pipeline(cfg(tmp_path, soft_mux=True), log=lambda _m: None)
        assert muxer[0]["src"] == result.video

    def test_srt_only_says_it_is_skipping_rather_than_failing(self, tmp_path, fakes, muxer):
        lines: list[str] = []
        result = run_pipeline(cfg(tmp_path, soft_mux=True, srt_only=True), log=lines.append)
        assert result.muxed is None
        assert muxer == []
        assert any("soft-mux: skipped" in line for line in lines)

    def test_no_video_anywhere_is_a_sentence_not_a_traceback(self, tmp_path, fakes, muxer):
        lines: list[str] = []
        result = run_pipeline(cfg(tmp_path, soft_mux=True, burn=False), log=lines.append)
        assert result.muxed is None
        assert any("no video" in line for line in lines)

    def test_a_second_run_does_not_re_mux(self, tmp_path, fakes, muxer):
        run_pipeline(cfg(tmp_path, soft_mux=True), log=lambda _m: None)
        second = run_pipeline(cfg(tmp_path, soft_mux=True), log=lambda _m: None)
        assert len(muxer) == 1
        assert "mux" in second.cached

    def test_a_hand_correction_re_muxes(self, tmp_path, fakes, muxer):
        """The track is the text, so the chain has to reach it. Keying the mux on the burn
        alone would ship a switchable track that still said the uncorrected thing."""
        first = run_pipeline(cfg(tmp_path, soft_mux=True, review=True), log=lambda _m: None)
        run_pipeline(cfg(tmp_path, soft_mux=True), log=lambda _m: None)
        assert len(muxer) == 1

        work = pipeline.work_dir(cfg(tmp_path))
        edits_mod.save(work, edits_mod.build(first.cues_key, {"1": "Ispravljeno"}))
        run_pipeline(cfg(tmp_path, soft_mux=True), log=lambda _m: None)
        assert len(muxer) == 2


class TestWorkDirIsPredictable:
    """The editor has to find the same directory the pipeline will, without running it."""

    def test_a_file_run_uses_the_stem_beside_the_output(self, tmp_path):
        assert pipeline.work_dir(cfg(tmp_path)) == tmp_path / ".subtitler" / FIXTURE.stem
        assert pipeline.output_dir(cfg(tmp_path)) == tmp_path

    def test_a_url_run_is_named_from_the_url_not_from_a_title(self, tmp_path):
        """The title costs a network round trip, and a warm run must not pay for one."""
        config = RunConfig(url=URL, out_dir=tmp_path)
        assert pipeline.work_dir(config).parent == tmp_path / ".subtitler"
        assert pipeline.work_dir(config).name.startswith("url-")

    def test_a_url_with_nowhere_to_write_is_refused_here_too(self):
        with pytest.raises(media.MediaError, match="output directory"):
            pipeline.output_dir(RunConfig(url=URL))
