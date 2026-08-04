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
    format_timecode,
    parse_timecode,
    trim_cmd,
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


class TestParseTimecode:
    @pytest.mark.parametrize(
        ("text", "seconds"),
        [
            ("0", 0.0),
            ("42", 42.0),
            ("90", 90.0),  # a bare number is seconds and may exceed 59
            ("1:30", 90.0),
            ("01:30", 90.0),
            ("10:00", 600.0),
            ("1:00:00", 3600.0),
            ("01:02:03", 3723.0),
            ("2.5", 2.5),
            ("0:02.250", 2.25),
            (" 1:30 ", 90.0),
        ],
    )
    def test_the_three_forms(self, text, seconds):
        assert parse_timecode(text) == pytest.approx(seconds)

    @pytest.mark.parametrize("text", ["", "abc", "1:2:3:4", "-5", "1:-2", "1:75", "1:2:99"])
    def test_nonsense_is_refused_with_the_accepted_forms(self, text):
        with pytest.raises(MediaError):
            parse_timecode(text)

    def test_a_clock_field_over_59_is_an_error_not_a_silent_reinterpretation(self):
        """`--start 1:75` is a typo for `1:15` or for `75`, and there is no way to tell.

        Quietly reading it as 135 seconds would cut the wrong fragment and say nothing.
        """
        with pytest.raises(MediaError, match="under 60"):
            parse_timecode("1:75")

    def test_round_trips_through_format(self):
        assert parse_timecode(format_timecode(3723.5)) == pytest.approx(3723.5)


class TestTrimCommand:
    def test_it_is_a_stream_copy_not_a_re_encode(self):
        """The regression this exists for: cutting three minutes out of an hour has to
        take a second, not five minutes, and must not re-compress the source.
        """
        cmd = trim_cmd(Path("in.mp4"), Path("out.mp4"), start=600.0, end=780.0)
        assert cmd[cmd.index("-c") + 1] == "copy"
        assert "libx264" not in cmd
        assert "-crf" not in cmd

    def test_the_seek_comes_before_the_input(self):
        """After `-i` ffmpeg decodes and throws away everything up to the start, which on
        an hour-long source is the entire cost this is avoiding."""
        cmd = trim_cmd(Path("in.mp4"), Path("out.mp4"), start=600.0, end=780.0)
        assert cmd.index("-ss") < cmd.index("-i")

    def test_the_end_becomes_a_duration(self):
        """`-t`, not `-to`: where `-to` is measured after an input seek is not the same on
        ffmpeg 4.4 and 8.x, and this project supports both."""
        cmd = trim_cmd(Path("in.mp4"), Path("out.mp4"), start=600.0, end=780.0)
        assert "-to" not in cmd
        assert cmd[cmd.index("-t") + 1] == "180.000"

    def test_timestamps_are_rebased_to_zero(self):
        """The whole reason the fragment's first cue reads 00:00:00 rather than 00:10:00.

        Without this the copied packets keep the source's timestamps, the extracted WAV
        starts at 600s, and every cue derived from it is ten minutes out.
        """
        cmd = trim_cmd(Path("in.mp4"), Path("out.mp4"), start=600.0)
        assert cmd[cmd.index("-avoid_negative_ts") + 1] == "make_zero"

    def test_an_open_ended_cut_has_no_duration(self):
        cmd = trim_cmd(Path("in.mp4"), Path("out.mp4"), start=600.0)
        assert "-t" not in cmd
        assert cmd[cmd.index("-ss") + 1] == "600.000"

    def test_a_cut_from_the_beginning_has_no_seek(self):
        cmd = trim_cmd(Path("in.mp4"), Path("out.mp4"), end=30.0)
        assert "-ss" not in cmd
        assert cmd[cmd.index("-t") + 1] == "30.000"

    def test_paths_are_argv_entries_not_a_shell_line(self):
        cmd = trim_cmd(Path("a b: c.mp4"), Path("out d.mp4"), end=1.0)
        assert cmd[cmd.index("-i") + 1] == "a b: c.mp4"
        assert cmd[-1] == "out d.mp4"

    def test_an_end_before_the_start_is_refused(self):
        with pytest.raises(MediaError, match="must be after"):
            trim_cmd(Path("in.mp4"), Path("out.mp4"), start=100.0, end=50.0)

    def test_a_no_op_window_is_refused(self):
        """Copying the whole file to a second name would waste the disk and the time, and
        the caller has to skip the stage instead. Same rule as `--denoise none`."""
        with pytest.raises(MediaError, match="nothing to trim"):
            trim_cmd(Path("in.mp4"), Path("out.mp4"))


@pytest.fixture(scope="module")
def container(tmp_path_factory):
    """A real mp4, because a raw WAV carries no timestamps at all.

    The property under test is that the fragment's clock is rebased to zero, and a format
    with no clock cannot show it either way. One keyframe per second so a stream copy can
    cut where it is asked to.
    """
    dst = tmp_path_factory.mktemp("container") / "clip.mp4"
    media.run(
        [
            "ffmpeg", "-y", "-hide_banner", "-v", "error",
            "-f", "lavfi", "-i", "color=c=black:s=160x120:r=10:d=10",
            "-i", str(FIXTURE),
            "-c:v", "libx264", "-preset", "ultrafast", "-g", "10", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-t", "10", str(dst),
        ]
    )  # fmt: skip
    return dst


@needs_ffmpeg
class TestTrimExecution:
    """These run ffmpeg. They are the only proof the fragment really starts at zero."""

    def test_the_fragment_has_the_requested_duration(self, container, tmp_path):
        dst = tmp_path / "cut.mp4"
        media.trim(container, dst, start=2.0, end=5.0)
        assert _probe(dst, "format=duration") == pytest.approx(3.0, abs=0.3)

    def test_the_fragment_starts_at_zero_not_at_the_cut_point(self, container, tmp_path):
        """The acceptance criterion for relative timestamps.

        Everything downstream reads this file and nothing knows about the window, so if
        its clock still said 2.0 here, every cue in the SRT would be two seconds late.
        """
        dst = tmp_path / "cut.mp4"
        media.trim(container, dst, start=2.0, end=5.0)
        assert _probe(dst, "format=start_time") == pytest.approx(0.0, abs=0.05)

    def test_the_audio_the_recognizer_gets_is_the_fragment(self, container, tmp_path):
        """The chain the pipeline actually runs: trim, then extract.

        Extracting from the fragment is what makes the timestamps relative for free. If
        this ever became "extract, then trim", the WAV would be the full ten seconds.
        """
        cut = tmp_path / "cut.mp4"
        media.trim(container, cut, start=2.0, end=5.0)
        wav = media.extract_audio(cut, tmp_path / "extract.wav")
        assert _probe(wav, "format=duration") == pytest.approx(3.0, abs=0.3)

    def test_an_open_ended_cut_runs_to_the_end(self, container, tmp_path):
        whole = _probe(container, "format=duration")
        dst = tmp_path / "tail.mp4"
        media.trim(container, dst, start=4.0)
        assert _probe(dst, "format=duration") == pytest.approx(whole - 4.0, abs=0.3)

    def test_the_video_stream_survives_the_copy(self, container, tmp_path):
        """The burn happens after the trim, so the fragment has to still have pixels."""
        dst = tmp_path / "cut.mp4"
        media.trim(container, dst, start=2.0, end=5.0)
        assert media.probe(dst).has_video


def _probe(path: Path, entry: str) -> float:
    proc = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", entry, "-of", "default=nw=1:nk=1", str(path)],
        capture_output=True,
        text=True,
        check=True,
    )
    return float(proc.stdout.strip().splitlines()[0])


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
