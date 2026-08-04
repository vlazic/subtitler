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
import wave
from pathlib import Path

import pytest

from subtitler import models
from subtitler.engines import ALL_ENGINES, EngineUnavailable, _build, default_order, resolve
from subtitler.engines.base import (
    SILENT_PEAK_DBFS,
    TranscribeOptions,
    drop_silent_segments,
    peak_dbfs,
)
from subtitler.engines.faster import (
    DEFAULT_BATCH_SIZE,
    FasterWhisperEngine,
    _nvidia_lib_dirs,
    preload_cuda_libraries,
)
from subtitler.engines.mlx import MlxWhisperEngine, _supported_kwargs
from subtitler.model import Segment


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
