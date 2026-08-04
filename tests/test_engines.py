"""Engine selection, the model registry, and the shared decode hygiene.

None of this needs a model or a GPU: selection, availability messaging and the silence
gate are all exercisable on any machine, which is the point.
"""

from __future__ import annotations

import struct
import wave
from pathlib import Path

import pytest

from subtitler import models
from subtitler.engines import ALL_ENGINES, EngineUnavailable, default_order, resolve
from subtitler.engines.base import (
    SILENT_PEAK_DBFS,
    TranscribeOptions,
    drop_silent_segments,
    peak_dbfs,
)
from subtitler.engines.faster import FasterWhisperEngine
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


class TestOptions:
    def test_serbian_prompt_is_the_default(self) -> None:
        opts = TranscribeOptions()
        assert opts.language == "sr"
        assert opts.initial_prompt and "latinično pismo" in opts.initial_prompt

    def test_conditioning_on_previous_text_is_off(self) -> None:
        """It is the main driver of repetition loops, and lets one bad segment poison
        everything after it."""
        assert TranscribeOptions().condition_on_previous_text is False
