"""Engine selection, the model registry, and the shared decode hygiene.

None of this needs a model or a GPU: selection, availability messaging, the silence gate
and every CUDA decision are all exercisable on any machine, which is the point. The GPU
paths below are driven through fakes for the same reason `test_doctor.py` fakes `Platform`:
CI has no NVIDIA card and the maintainer's friend has a Mac.
"""

from __future__ import annotations

import importlib
import struct
import sys
import types
import wave
from pathlib import Path
from typing import ClassVar

import pytest

from subtitler import models
from subtitler.engines import ALL_ENGINES, EngineUnavailable, _build, default_order, resolve
from subtitler.engines.base import (
    SILENT_PEAK_DBFS,
    TranscribeOptions,
    drop_silent_segments,
    drop_speechless_segments,
    mean_word_confidence,
    peak_dbfs,
)
from subtitler.engines.faster import (
    DEFAULT_BATCH_SIZE,
    FasterWhisperEngine,
    _nvidia_lib_dirs,
    preload_cuda_libraries,
)
from subtitler.engines.mlx import MlxWhisperEngine, _supported_kwargs
from subtitler.model import Segment, Word


class TestDefaultOrder:
    def test_apple_silicon_prefers_mlx(self) -> None:
        assert default_order(apple_silicon=True)[0] == "mlx"

    def test_elsewhere_prefers_faster_whisper(self) -> None:
        assert default_order(apple_silicon=False)[0] == "faster-whisper"

    def test_local_always_beats_cloud(self) -> None:
        """The friend's video should not leave his laptop just because a download has
        not happened yet. Cloud is a fallback, never a shortcut."""
        for mac in (True, False):
            order = default_order(apple_silicon=mac)
            assert order.index("faster-whisper") < order.index("groq")
            assert order.index("mlx") < order.index("groq")

    def test_order_covers_every_known_engine(self) -> None:
        assert set(default_order(apple_silicon=True)) == set(ALL_ENGINES)


class TestExplicitResolution:
    def test_unknown_engine_names_the_alternatives(self) -> None:
        with pytest.raises(EngineUnavailable) as exc:
            resolve("whisper.cpp")
        assert "faster-whisper" in str(exc.value)

    def test_mlx_on_linux_is_a_hard_error_not_a_fallback(self) -> None:
        """Silently falling back would mean benchmarking a different backend than the one
        that was asked for, or uploading a private file to a cloud API."""
        engine = MlxWhisperEngine("large-v3")
        if engine.platform_supported():
            pytest.skip("this machine is Apple Silicon")
        avail = engine.availability()
        assert not avail.ok
        assert "Apple Silicon" in avail.reason
        assert "faster-whisper" in avail.fix

    def test_unavailable_engine_carries_an_actionable_fix(self) -> None:
        engine = FasterWhisperEngine("large-v3")
        avail = engine.availability()
        if avail.ok:
            pytest.skip("faster-whisper is installed with weights present")
        assert avail.fix, "an unavailable engine must always say how to fix it"


class TestModelRegistry:
    def test_aliases_resolve(self) -> None:
        assert models.resolve("large", "faster-whisper").name == "large-v3"
        assert models.resolve("whisper-large-v3", "mlx").name == "large-v3"

    def test_unknown_model_lists_what_exists(self) -> None:
        with pytest.raises(models.ModelNotFound) as exc:
            models.resolve("medium", "mlx")
        assert "large-v3" in str(exc.value)

    def test_revisions_are_pinned_to_commit_shas(self) -> None:
        """A floating tag means a benchmark from last month cannot be reproduced today."""
        for spec in models.MODELS:
            assert len(spec.revision) == 40, f"{spec.key} is not pinned to a commit SHA"
            assert spec.revision != "main"
            int(spec.revision, 16)  # raises if it is not hex

    def test_every_backend_has_the_default_model(self) -> None:
        for backend in ("faster-whisper", "mlx"):
            assert models.resolve("large-v3", backend)

    def test_size_labels_are_human_readable(self) -> None:
        assert models.resolve("large-v3", "mlx").size_label.endswith("GB")
        assert models.resolve("tiny", "mlx").size_label.endswith("MB")

    def test_cache_root_honours_hf_hub_cache(self, monkeypatch, tmp_path) -> None:
        monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path))
        assert models.cache_root() == tmp_path

    def test_cache_root_honours_hf_home(self, monkeypatch, tmp_path) -> None:
        monkeypatch.delenv("HF_HUB_CACHE", raising=False)
        monkeypatch.setenv("HF_HOME", str(tmp_path))
        assert models.cache_root() == tmp_path / "hub"

    def test_uncached_model_has_no_local_path(self, monkeypatch, tmp_path) -> None:
        monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path))
        assert models.local_path(models.resolve("large-v3", "mlx")) is None


def write_wav(path: Path, spans: list[tuple[float, int]], rate: int = 16000) -> Path:
    """Build a 16-bit mono WAV from (duration_seconds, amplitude) spans."""
    frames = bytearray()
    for duration, amplitude in spans:
        for i in range(int(duration * rate)):
            value = amplitude if (i // 8) % 2 == 0 else -amplitude
            frames += struct.pack("<h", value)
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(rate)
        wav.writeframes(bytes(frames))
    return path


class TestSilenceGate:
    def test_loud_span_is_well_above_the_threshold(self, tmp_path) -> None:
        path = write_wav(tmp_path / "a.wav", [(1.0, 20000)])
        assert peak_dbfs(path, 0.0, 1.0) > SILENT_PEAK_DBFS

    def test_digital_silence_is_negative_infinity(self, tmp_path) -> None:
        path = write_wav(tmp_path / "a.wav", [(1.0, 0)])
        assert peak_dbfs(path, 0.0, 1.0) == -float("inf")

    def test_very_quiet_span_falls_below_the_threshold(self, tmp_path) -> None:
        # -60 dBFS is roughly amplitude 32 out of 32768.
        path = write_wav(tmp_path / "a.wav", [(1.0, 5)])
        assert peak_dbfs(path, 0.0, 1.0) < SILENT_PEAK_DBFS

    def test_gate_drops_the_silent_segment_only(self, tmp_path) -> None:
        """Whisper invents filler over silence. Dropping the span beats filtering the
        text afterwards, because by then it has already claimed a timestamp range."""
        path = write_wav(tmp_path / "a.wav", [(1.0, 20000), (1.0, 0), (1.0, 20000)])
        segments = (
            Segment(start=0.0, end=1.0, text="stvarno"),
            Segment(start=1.0, end=2.0, text="Hvala."),  # the classic hallucination
            Segment(start=2.0, end=3.0, text="opet stvarno"),
        )
        kept = drop_silent_segments(segments, path)
        assert [s.text for s in kept] == ["stvarno", "opet stvarno"]

    def test_missing_file_keeps_everything(self, tmp_path) -> None:
        """Never silently delete transcript content because a probe could not run."""
        segments = (Segment(start=0.0, end=1.0, text="zdravo"),)
        assert drop_silent_segments(segments, tmp_path / "nope.wav") == segments

    def test_zero_length_span(self, tmp_path) -> None:
        path = write_wav(tmp_path / "a.wav", [(1.0, 20000)])
        assert peak_dbfs(path, 0.5, 0.5) == -float("inf")


def spoken(start: float, end: float, text: str, *, prob: float, no_speech: float = 0.05) -> Segment:
    """A segment carrying a flat per-word confidence, which is what the gate reads."""
    tokens = text.split()
    step = (end - start) / max(len(tokens), 1)
    return Segment(
        start=start,
        end=end,
        text=text,
        words=tuple(
            Word(start=start + i * step, end=start + (i + 1) * step, text=t, prob=prob)
            for i, t in enumerate(tokens)
        ),
        no_speech_prob=no_speech,
    )


class TestSpeechFreeGate:
    """Regression: speech-free audio produced confident hallucination.

    The first 10 seconds of a YouTube clip, titles and music and no speech at all, came back
    as the single cue `Hvala što pratite kanal.` with nothing said about it. The -60 dBFS
    silence gate cannot catch it because music is loud, and neither can `no_speech_prob` or
    word confidence on their own: measured on this material the hallucination scores
    `no_speech_prob` 0.14 and mean word confidence 0.80, while genuine speech in the
    project's own fixtures reaches 0.436 and drops to 0.779. The numbers below are those
    measurements, not invented ones.
    """

    def test_the_youtube_music_hallucination_is_dropped(self) -> None:
        """The named bug, with its measured values: confident, low no-speech, still wrong."""
        segments = (spoken(0.0, 0.46, "Hvala što pratite kanal.", prob=0.80, no_speech=0.14),)
        kept, reasons = drop_speechless_segments(segments, duration=10.0)
        assert kept == ()
        assert len(reasons) == 1
        assert "hvala" in reasons[0]

    def test_the_genuine_hvala_in_gozba_sample_survives(self) -> None:
        """`fixtures/gozba-sample.mp3` really does contain someone saying "Hvala.".

        It scores 1.000, the highest confidence in the whole corpus, and it sits in a
        109-second transcript running at 1.40 words per second. Both facts are what keep it,
        and this test is what stops the gate from costing the fixture one of its 20 cues.
        """
        segments = (
            spoken(6.5, 20.2, "Misao lokove filozofije, ukratko izraženo", prob=0.96),
            spoken(72.0, 73.0, "Dobrodošli na gozbu.", prob=0.935, no_speech=0.067),
            spoken(73.3, 74.0, "Hvala.", prob=1.0, no_speech=0.067),
            spoken(75.9, 83.5, "Razgovaramo o filozofiji Johna Locke", prob=0.984),
        )
        kept, reasons = drop_speechless_segments(segments, duration=109.0)
        assert kept == segments
        assert reasons == []

    def test_the_worst_real_speech_in_the_fixtures_survives(self) -> None:
        """`uvod-u-pravo.m4a` reaches no_speech_prob 0.436 and mean confidence 0.779.

        Every threshold in the gate has to clear these, or the fix costs more than the bug.
        """
        segments = (
            spoken(
                0.6, 19.6, "Imamo pravna pravila koja su ona pravila", prob=0.779, no_speech=0.012
            ),
            spoken(
                95.6,
                107.5,
                "Dakle, imate potređenu državnu službenu lice",
                prob=0.865,
                no_speech=0.436,
            ),
        )
        kept, _ = drop_speechless_segments(segments, duration=164.5)
        assert kept == segments

    def test_the_no_speech_head_is_believed_when_it_fires(self) -> None:
        segments = (spoken(0.0, 5.0, "Svičnava.", prob=0.96, no_speech=0.94),)
        kept, reasons = drop_speechless_segments(segments, duration=10.0)
        assert kept == ()
        assert "no_speech_prob" in reasons[0]

    def test_words_carrying_no_confidence_are_dropped(self) -> None:
        """A weak model over noise emits tokens it cannot justify (measured: 0.004 to 0.219)."""
        segments = (spoken(1.0, 2.0, "...", prob=0.02, no_speech=0.1),)
        kept, reasons = drop_speechless_segments(segments, duration=10.0)
        assert kept == ()
        assert "confidence" in reasons[0]

    def test_filler_inside_a_real_transcript_is_kept(self) -> None:
        """The word-rate term is what tells one filler cue apart from a filler transcript."""
        segments = (
            spoken(0.0, 10.0, "Danas govorimo o filozofiji i o mnogim drugim stvarima", prob=0.9),
            spoken(10.0, 11.0, "Hvala.", prob=0.88),
        )
        kept, reasons = drop_speechless_segments(segments, duration=11.0)
        assert kept == segments
        assert reasons == []

    def test_an_engine_supplying_neither_signal_keeps_its_text(self) -> None:
        """Absence of evidence is not evidence. A silent drop on a missing field is exactly
        the failure this gate exists to prevent, so nothing is removed without a reason."""
        segments = (Segment(start=0.0, end=1.0, text="Hvala što pratite kanal."),)
        kept, reasons = drop_speechless_segments(segments, duration=10.0)
        assert kept == segments
        assert reasons == []

    def test_unknown_duration_disables_the_word_rate_test(self) -> None:
        """Zero means "not measured", and a rate is not guessed from a missing denominator."""
        segments = (spoken(0.0, 0.46, "Hvala što pratite kanal.", prob=0.80, no_speech=0.14),)
        kept, _ = drop_speechless_segments(segments, duration=0.0)
        assert kept == segments

    def test_mean_word_confidence_ignores_words_that_carry_none(self) -> None:
        seg = Segment(
            start=0.0,
            end=1.0,
            text="a b",
            words=(Word(0.0, 0.5, "a", 0.8), Word(0.5, 1.0, "b", None)),
        )
        assert mean_word_confidence(seg) == pytest.approx(0.8)
        assert mean_word_confidence(Segment(start=0.0, end=1.0, text="a")) is None


class TestPromptEchoRetry:
    """Regression: a clip shorter than 30 seconds echoed the steering prompt as transcript.

    The first 10 seconds of a YouTube episode (titles and music, no speech) came back as
    the single cue "Zadrži srpski jezik i latinično pismo." with word confidences of 0.02
    to 0.11, and passing a different `--prompt` produced that prompt instead. `_prompt_for`
    only dropped the prompt for batched decoding, on the claim that the sequential path is
    safe because it resets after the first 30-second window: a clip shorter than one window
    never reaches that reset, and on a longer file window one is exposed just the same,
    which is how `--denoise arnndn` lost the opening of a 164-second lecture.
    """

    ECHO = "Zadrži srpski jezik i latinično pismo."
    SPEECH = "Ovo je ono što je govornik zaista rekao u snimku."

    def fake_faster_whisper(self, monkeypatch, engine, texts):
        """Give `_decode_with` a decoder that returns `texts[n]` on its n-th call."""
        seen: list[str | None] = []

        def transcribe(path, **kwargs):
            seen.append(kwargs["initial_prompt"])
            segment = types.SimpleNamespace(
                start=0.0, end=5.0, text=texts[len(seen) - 1], words=None
            )
            return iter([segment]), types.SimpleNamespace(language="sr", duration=10.0)

        monkeypatch.setitem(
            sys.modules, "ctranslate2", types.SimpleNamespace(set_random_seed=lambda _seed: None)
        )
        monkeypatch.setattr(
            engine, "_load", lambda: types.SimpleNamespace(transcribe=transcribe), raising=False
        )
        monkeypatch.setattr(engine, "resolve_device", lambda: ("cpu", "int8"))
        return seen

    def test_an_echoing_decode_is_redone_without_the_prompt(self, monkeypatch) -> None:
        engine = FasterWhisperEngine("large-v3", device="cpu")
        seen = self.fake_faster_whisper(monkeypatch, engine, [self.ECHO, self.SPEECH])

        # `nope.wav` does not exist, so the silence gate keeps every segment: what is under
        # test is the retry, not the gate.
        transcript = engine._decode(Path("nope.wav"), TranscribeOptions())

        assert transcript.text == self.SPEECH, "the prompt-free decode is the one kept"
        assert seen[0] and seen[1] is None, "the retry must not carry the prompt"
        assert transcript.params["initial_prompt"] is False
        assert "latinično pismo" in transcript.params["prompt_echo_retry"]

    def test_a_clean_decode_keeps_the_prompt_and_decodes_once(self, monkeypatch) -> None:
        """The cost of the fix on every run that never echoes has to be exactly zero."""
        engine = FasterWhisperEngine("large-v3", device="cpu")
        seen = self.fake_faster_whisper(monkeypatch, engine, [self.SPEECH])

        transcript = engine._decode(Path("nope.wav"), TranscribeOptions())

        assert transcript.text == self.SPEECH
        assert seen == [TranscribeOptions().initial_prompt], "one decode, prompt attached"
        assert transcript.params["initial_prompt"] is True
        assert "prompt_echo_retry" not in transcript.params

    def test_a_custom_prompt_is_the_one_checked_for(self, monkeypatch) -> None:
        """The echo follows whatever `--prompt` put in, which is what proved the mechanism."""
        engine = FasterWhisperEngine("large-v3", device="cpu")
        custom = "Ovo je moj sopstveni pomoćni tekst"
        seen = self.fake_faster_whisper(monkeypatch, engine, [custom, self.SPEECH])

        transcript = engine._decode(Path("nope.wav"), TranscribeOptions(initial_prompt=custom))

        assert transcript.text == self.SPEECH
        assert seen == [custom, None]

    def test_mlx_has_the_same_retry(self, monkeypatch, tmp_path) -> None:
        """mlx-whisper carries `initial_prompt` into the first window too, so the hole is
        identical there and it is the primary target's default engine. Driven on Linux
        through a fake module, per non-negotiable 5."""
        seen: list[str | None] = []
        texts = [self.ECHO, self.SPEECH]

        def transcribe(path, **kwargs):
            seen.append(kwargs.get("initial_prompt"))
            return {
                "language": "sr",
                "segments": [{"start": 0.0, "end": 5.0, "text": texts[len(seen) - 1]}],
            }

        monkeypatch.setitem(
            sys.modules, "mlx_whisper", types.SimpleNamespace(transcribe=transcribe)
        )
        monkeypatch.setattr("subtitler.engines.mlx.models.local_path", lambda _spec: tmp_path)
        monkeypatch.setattr(MlxWhisperEngine, "platform_supported", staticmethod(lambda: True))

        transcript = MlxWhisperEngine("large-v3").transcribe(Path("nope.wav"), TranscribeOptions())

        assert transcript.text == self.SPEECH
        assert seen[0] and seen[1] is None
        assert "latinično pismo" in transcript.params["prompt_echo_retry"]


class TestMlxKwargFiltering:
    def test_keeps_only_accepted_names(self) -> None:
        def fake(audio, *, language=None, word_timestamps=False): ...

        out = _supported_kwargs(fake, {"language": "sr", "word_timestamps": True, "vad": True})
        assert out == {"language": "sr", "word_timestamps": True}

    def test_drops_none_values(self) -> None:
        def fake(audio, *, language=None, initial_prompt=None): ...

        assert _supported_kwargs(fake, {"language": "sr", "initial_prompt": None}) == {
            "language": "sr"
        }

    def test_a_kwargs_catch_all_forwards_everything(self) -> None:
        def fake(audio, **kwargs): ...

        assert _supported_kwargs(fake, {"anything": 1, "nothing": None}) == {"anything": 1}


class TestCudaLibraryDiscovery:
    def test_namespace_packages_have_no_dunder_file(self, tmp_path, monkeypatch) -> None:
        """Regression: the preload looked for the CUDA libraries in the wrong directory.

        `nvidia`, `nvidia.cublas` and `nvidia.cublas.lib` ship no `__init__.py`, so they
        are namespace packages and `__file__` is None. The original code did
        `Path(module.__file__ or "").parent`, which is `Path(".")`, so it looked for
        libcublas.so.12 in the current working directory, never found it, and reported
        "nothing to preload" on a machine with every library installed. It had never
        loaded a single one.
        """
        lib = tmp_path / "fakenvidia" / "cublas" / "lib"
        lib.mkdir(parents=True)
        (lib / "libcublas.so.12").write_bytes(b"not really a library")
        monkeypatch.syspath_prepend(str(tmp_path))
        importlib.invalidate_caches()
        module = importlib.import_module("fakenvidia")
        try:
            assert module.__file__ is None, "the trap only exists for namespace packages"
            assert Path(module.__file__ or "").parent == Path("."), "the bug, reproduced"
            assert _nvidia_lib_dirs(list(module.__path__)) == [lib]
        finally:
            sys.modules.pop("fakenvidia", None)

    def test_missing_directories_are_not_an_error(self, tmp_path) -> None:
        assert _nvidia_lib_dirs([str(tmp_path)]) == []

    def test_preload_survives_libraries_that_will_not_load(self, tmp_path, monkeypatch) -> None:
        """A CPU-only machine, and a broken install, must both degrade to False.

        The engine's answer to "is CUDA usable" is allowed to be no; it is never allowed
        to be an exception raised while merely asking.
        """
        lib = tmp_path / "cublas" / "lib"
        lib.mkdir(parents=True)
        (lib / "libcublas.so.12").write_bytes(b"\x00 not an ELF file")
        monkeypatch.setattr("subtitler.engines.faster._cuda_preloaded", None)
        monkeypatch.setattr("subtitler.engines.faster._nvidia_lib_dirs", lambda roots=None: [lib])
        assert preload_cuda_libraries() is False

    def test_an_explicit_device_cuda_still_preloads(self, tmp_path, monkeypatch) -> None:
        """Regression: `--device cuda` skipped the preload entirely and died mid-decode.

        `resolve_device()` reaches `preload_cuda_libraries()` only through `_cuda_usable()`,
        and it only calls that on `--device auto`. Asking for cuda by name therefore handed
        CTranslate2 an unprepared loader, which searched a system toolkit that ships CUDA
        11.5 and raised `Library libcublas.so.12 is not found or cannot be loaded` at the
        first decoded window, on a box where `subtitler doctor` had just reported the CUDA
        runtime as usable. The GUI is what surfaced it: its Processor dropdown makes "cuda"
        one click away rather than something only a benchmark run ever typed.
        """
        calls: list[int] = []
        monkeypatch.setattr(
            "subtitler.engines.faster.preload_cuda_libraries", lambda: calls.append(1) or True
        )
        monkeypatch.setattr("subtitler.engines.faster.models.local_path", lambda _spec: tmp_path)

        loaded: list[dict] = []

        class FakeWhisperModel:
            def __init__(self, path, **kwargs):
                loaded.append(kwargs)

        monkeypatch.setitem(
            sys.modules, "faster_whisper", types.SimpleNamespace(WhisperModel=FakeWhisperModel)
        )

        engine = FasterWhisperEngine("tiny", device="cuda")
        engine._load()
        assert calls == [1], "the preload must run before CTranslate2 looks for libcublas"
        assert loaded == [{"device": "cuda", "compute_type": "float16"}]

    def test_a_cpu_run_does_not_touch_the_cuda_libraries(self, tmp_path, monkeypatch) -> None:
        """dlopening several hundred megabytes of CUDA on a CPU-only run is pure latency."""
        calls: list[int] = []
        monkeypatch.setattr(
            "subtitler.engines.faster.preload_cuda_libraries", lambda: calls.append(1) or True
        )
        monkeypatch.setattr("subtitler.engines.faster.models.local_path", lambda _spec: tmp_path)
        monkeypatch.setitem(
            sys.modules,
            "faster_whisper",
            types.SimpleNamespace(WhisperModel=lambda path, **kwargs: object()),
        )
        FasterWhisperEngine("tiny", device="cpu")._load()
        assert calls == []


class TestBatching:
    """`--batch-size` is a CUDA-only knob and must stay invisible everywhere else."""

    def test_off_by_default(self) -> None:
        assert FasterWhisperEngine("large-v3").effective_batch_size() == 0
        assert FasterWhisperEngine("large-v3").describe()["batch_size"] == 0

    def test_ignored_on_cpu(self) -> None:
        """BatchedInferencePipeline runs on CPU and is slower there than the sequential
        path, so honouring the flag after a CUDA fallback makes a bad run worse."""
        engine = FasterWhisperEngine("large-v3", device="cpu", batch_size=DEFAULT_BATCH_SIZE)
        assert engine.resolve_device() == ("cpu", "int8")
        assert engine.effective_batch_size() == 0

    def test_honoured_on_cuda(self) -> None:
        engine = FasterWhisperEngine("large-v3", device="cuda", batch_size=8)
        assert engine.effective_batch_size() == 8
        assert engine.describe()["batch_size"] == 8

    def test_negative_batch_size_means_off(self) -> None:
        assert (
            FasterWhisperEngine("large-v3", device="cuda", batch_size=-4).effective_batch_size()
            == 0
        )

    def test_batching_drops_the_steering_prompt(self) -> None:
        """Regression, measured on a 54-minute Serbian episode: batched decoding echoed
        the steering prompt back as transcript text over and over and lost 15% of the
        speech. `generate_segment_batched` passes `initial_prompt` as `previous_tokens`
        for every window in the file, and unlike the sequential path there is no
        `prompt_reset_since` that moves past it after the first one."""
        opts = TranscribeOptions()
        assert opts.initial_prompt  # the Serbian prompt is on by default
        assert FasterWhisperEngine._prompt_for(opts, 0) == opts.initial_prompt
        assert FasterWhisperEngine._prompt_for(opts, 16) is None

    def test_the_registry_passes_it_only_to_faster_whisper(self) -> None:
        """Every builder accepts it so callers need not know which engine uses it.

        `_build` rather than `resolve`, because `resolve` also checks availability and the
        weights are not downloaded on a CI runner."""
        engine = _build("faster-whisper", model="large-v3", device="cuda", batch_size=8)
        assert isinstance(engine, FasterWhisperEngine)
        assert engine.requested_batch_size == 8
        # mlx and groq take the argument and ignore it rather than raising.
        for name in ("mlx", "groq", "groq-turbo"):
            assert _build(name, model="large-v3", device="auto", batch_size=8)

    def test_batching_without_vad_is_rejected_with_the_fix(self) -> None:
        """faster-whisper raises "No clip timestamps found" from deep inside the batched
        generator: with the VAD off it has nothing to chunk on."""
        engine = FasterWhisperEngine("large-v3", device="cuda", batch_size=8)
        with pytest.raises(ValueError) as exc:
            engine._decode(Path("nope.wav"), TranscribeOptions(vad=False))
        assert "--batch-size 0" in str(exc.value)


class TestOptions:
    def test_serbian_prompt_is_the_default(self) -> None:
        opts = TranscribeOptions()
        assert opts.language == "sr"
        assert opts.initial_prompt and "latinično pismo" in opts.initial_prompt

    def test_conditioning_on_previous_text_is_off(self) -> None:
        """It is the main driver of repetition loops, and lets one bad segment poison
        everything after it."""
        assert TranscribeOptions().condition_on_previous_text is False


class TestGroqKeyPool:
    """A key pool whose dead key can end the run is not a pool.

    Found by the Phase 7 benchmark matrix, not by a unit test: one of the two keys in the
    maintainer's pool answers "Organization has been restricted", and `random.choice` drew
    it for about half the cloud cells. A 400 is not retryable, so an identical `bench run`
    failed a different random half of its cloud cells on every attempt.
    """

    @staticmethod
    def _fake_groq(dead: set[str], seen: list[str]) -> types.ModuleType:
        class Restricted(Exception):
            status_code = 400
            body: ClassVar[dict] = {"error": {"message": "Organization has been restricted."}}

        class Response:
            @staticmethod
            def json() -> str:
                return '{"language": "sr", "duration": 1.0, "segments": []}'

        class Client:
            def __init__(self, api_key: str) -> None:
                self.api_key = api_key
                self.audio = self

            @property
            def transcriptions(self):  # the SDK's client.audio.transcriptions.create
                return self

            def create(self, **_kwargs):
                seen.append(self.api_key)
                if self.api_key in dead:
                    raise Restricted()
                return Response()

        module = types.ModuleType("groq")
        module.Groq = Client
        return module

    def _engine(self, monkeypatch, keys: str):
        from subtitler.engines import groq as groq_engine

        monkeypatch.setenv("GROQ_API_KEYS", keys)
        monkeypatch.delenv("GROQ_API_KEY", raising=False)
        # The pool is shuffled to spread load, which would make "the dead key was tried
        # first" a coin flip. Pinning the order is what makes the fallback observable.
        monkeypatch.setattr(groq_engine.random, "sample", lambda seq, k: list(seq)[:k])
        return groq_engine.GroqEngine("turbo")

    def test_a_restricted_key_falls_through_to_a_live_one(self, monkeypatch) -> None:
        seen: list[str] = []
        monkeypatch.setitem(sys.modules, "groq", self._fake_groq({"dead"}, seen))
        engine = self._engine(monkeypatch, "dead,live")
        result = engine._request("a.wav", b"x", TranscribeOptions())
        assert result["language"] == "sr"
        assert seen == ["dead", "live"]

    def test_every_key_gets_a_turn_before_the_run_gives_up(self, monkeypatch) -> None:
        seen: list[str] = []
        monkeypatch.setitem(sys.modules, "groq", self._fake_groq({"a", "b"}, seen))
        engine = self._engine(monkeypatch, "a,b")
        with pytest.raises(EngineUnavailable) as exc:
            engine._request("a.wav", b"x", TranscribeOptions())
        assert "restricted" in str(exc.value)
        assert set(seen) == {"a", "b"}

    def test_a_pool_larger_than_the_retry_budget_is_still_exhausted(self, monkeypatch) -> None:
        """`max_retries` counts attempts at one key; it must not cap how many keys are tried."""
        seen: list[str] = []
        monkeypatch.setitem(sys.modules, "groq", self._fake_groq({"a", "b", "c", "d"}, seen))
        engine = self._engine(monkeypatch, "a,b,c,d,e")
        engine.max_retries = 2
        assert engine._request("a.wav", b"x", TranscribeOptions())["language"] == "sr"
        assert "e" in seen

    def test_one_key_still_fails_once_and_says_why(self, monkeypatch) -> None:
        seen: list[str] = []
        monkeypatch.setitem(sys.modules, "groq", self._fake_groq({"only"}, seen))
        engine = self._engine(monkeypatch, "only")
        with pytest.raises(EngineUnavailable) as exc:
            engine._request("a.wav", b"x", TranscribeOptions())
        assert seen == ["only"]
        assert "local engine" in str(exc.value)
