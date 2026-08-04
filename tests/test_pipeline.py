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
from subtitler import pipeline, postedit
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

    def __call__(self, cues, dst: Path, **kwargs) -> Path:
        self.calls += 1
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
        """A stub LiteLLM that upper-cases every cue and counts the calls it was billed."""
        calls = []

        def complete(_system: str, user: str, _cfg) -> str:
            calls.append(user)
            items = json.loads(user)
            return json.dumps(
                [{"i": it["i"], "text": it["text"].upper()} for it in items], ensure_ascii=False
            )

        monkeypatch.setattr(postedit, "_litellm_complete", complete)
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
