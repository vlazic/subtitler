"""ffmpeg command construction for extraction and the denoise presets.

Mostly argv assertions: these catch the flags that are easy to lose in a refactor and that
only fail on someone else's machine. The two that actually run ffmpeg are marked and skip
when it is absent.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from subtitler import media
from subtitler.media import (
    DENOISE_FILTERS,
    RNNOISE_LOCAL_NAME,
    RNNOISE_MODEL,
    TARGET_CHANNELS,
    TARGET_SR,
    MediaError,
    denoise_cmd,
    denoise_filter,
    extract_audio_cmd,
)

# The 10-second fixture rather than the 109-second one: `anlmdn` and `speech` are not
# cheap, and eight filter runs over the long clip put ten seconds into every `make test`.
FIXTURE = Path(__file__).resolve().parent.parent / "fixtures" / "tiny-10s.wav"
needs_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")


@pytest.fixture(scope="module")
def extracted(tmp_path_factory):
    """One canonical 16 kHz mono WAV, shared by every execution test in this module."""
    dst = tmp_path_factory.mktemp("extract") / "extract.wav"
    media.extract_audio(FIXTURE, dst)
    return dst


class TestExtractCommand:
    def test_canonical_shape(self):
        cmd = extract_audio_cmd(Path("in.mp4"), Path("out.wav"))
        assert cmd[:1] == ["ffmpeg"]
        assert cmd[-1] == "out.wav"
        for flag, value in (("-ac", str(TARGET_CHANNELS)), ("-ar", str(TARGET_SR))):
            assert cmd[cmd.index(flag) + 1] == value
        assert "-vn" in cmd
        assert cmd[cmd.index("-c:a") + 1] == "pcm_s16le"

    def test_extraction_never_denoises(self):
        """Denoise is its own stage now, and this is the assertion that keeps it that way.

        Folding the filter back into the extract command would put the denoiser choice in
        the extraction's cache key, so `--denoise afftdn` on a 3 GB video would demux the
        whole source again rather than filtering the WAV that is already on disk.
        """
        assert "-af" not in extract_audio_cmd(Path("in.mp4"), Path("out.wav"))


class TestDenoiseFilter:
    @pytest.mark.parametrize("name", ["afftdn", "anlmdn", "speech"])
    def test_presets_without_a_model_resolve(self, name):
        assert denoise_filter(name) == DENOISE_FILTERS[name]

    def test_none_is_empty(self):
        assert denoise_filter("none") == ""

    def test_unknown_preset_names_the_choices(self):
        with pytest.raises(MediaError, match="afftdn"):
            denoise_filter("rnnoise")

    def test_arnndn_without_a_model_is_an_error_not_a_broken_filtergraph(self):
        with pytest.raises(MediaError, match="rnnoise model"):
            denoise_filter("arnndn")

    def test_arnndn_option_is_named(self):
        """`m=` rather than a positional argument.

        When a filter is missing, ffmpeg given a positional argument reports "No option
        name near ..." instead of "No such filter", which sends you off fixing the wrong
        thing. Same rule as `ass=f=subs.ass` in burn.py.
        """
        assert (
            denoise_filter("arnndn", Path(RNNOISE_LOCAL_NAME)) == f"arnndn=m={RNNOISE_LOCAL_NAME}"
        )


class TestDenoiseCommand:
    def test_output_shape_is_restated(self):
        """`speech` ends in loudnorm, which resamples to 192 kHz internally.

        Without `-ar` on the output, that rate reaches the file and the recognizer gets
        audio in a shape the extraction stage promised it would never see.
        """
        cmd = denoise_cmd(Path("in.wav"), Path("out.wav"), preset="speech")
        assert cmd[cmd.index("-ar") + 1] == str(TARGET_SR)
        assert cmd[cmd.index("-ac") + 1] == str(TARGET_CHANNELS)
        assert cmd[cmd.index("-af") + 1] == DENOISE_FILTERS["speech"]

    def test_none_is_refused(self):
        """A no-op pass would rewrite the WAV for nothing. The caller must skip the stage."""
        with pytest.raises(MediaError, match="no-op"):
            denoise_cmd(Path("in.wav"), Path("out.wav"), preset="none")


class TestBundledRnnoiseModel:
    def test_the_model_ships(self):
        """`arnndn` cannot run without it, so a preset offered by `--denoise` and missing
        its model file is a preset that does not exist."""
        assert RNNOISE_MODEL.exists()
        assert RNNOISE_MODEL.stat().st_size > 1000

    def test_the_local_name_has_no_colon(self):
        """The whole colon-avoidance strategy rests on this name being boring."""
        assert ":" not in RNNOISE_LOCAL_NAME
        assert RNNOISE_LOCAL_NAME.isascii()


@needs_ffmpeg
class TestDenoiseExecution:
    """These run ffmpeg. They are the only proof that the presets are real."""

    @pytest.mark.parametrize("preset", ["afftdn", "arnndn", "anlmdn", "speech"])
    def test_preset_runs_and_changes_the_audio(self, preset, extracted, tmp_path):
        """Every preset must produce audio, and audio that differs from no denoising.

        A filter that silently passes through (a missing model, an option ffmpeg accepts
        and ignores) would leave the benchmark comparing a denoiser against itself.
        """
        dst = tmp_path / f"{preset}.wav"
        media.denoise_audio(extracted, dst, preset=preset)
        assert dst.stat().st_size > 0
        assert dst.read_bytes() != extracted.read_bytes()

    @pytest.mark.parametrize("preset", ["afftdn", "arnndn", "anlmdn", "speech"])
    def test_output_stays_16k_mono_pcm(self, preset, extracted, tmp_path):
        dst = tmp_path / f"{preset}.wav"
        media.denoise_audio(extracted, dst, preset=preset)
        proc = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "stream=codec_name,sample_rate,channels",
                "-of",
                "default=nw=1:nk=1",
                str(dst),
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        assert proc.stdout.split() == ["pcm_s16le", str(TARGET_SR), str(TARGET_CHANNELS)]

    def test_arnndn_survives_a_colon_in_every_path(self, extracted, tmp_path):
        """The acceptance test for the model-path strategy.

        ffmpeg filter options are colon separated, so `arnndn=m=/Users/x/a:b.rnnn` splits
        into a nonsense option. The fix is not escaping, which no single form survives
        shell, filtergraph and filter-option parsing: the model is copied into a temp dir
        as a fixed ASCII name and ffmpeg runs with cwd set there. Input and output stay as
        argv entries, where only a shell would matter, and there is no shell.
        """
        hostile = tmp_path / "a dir: with 'quotes'"
        hostile.mkdir()
        model = hostile / "model: with 'quotes'.rnnn"
        model.write_bytes(RNNOISE_MODEL.read_bytes())
        src = hostile / "in: file.wav"
        src.write_bytes(extracted.read_bytes())
        dst = hostile / "out: file.wav"

        media.denoise_audio(src, dst, preset="arnndn", rnnoise_model=model)
        assert dst.stat().st_size > 0

    def test_a_missing_model_names_the_fix(self, extracted, tmp_path):
        with pytest.raises(MediaError, match="--denoise"):
            media.denoise_audio(
                extracted, tmp_path / "o.wav", preset="arnndn", rnnoise_model=tmp_path / "gone.rnnn"
            )
