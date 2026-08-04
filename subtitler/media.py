"""Every ffmpeg and ffprobe invocation in the project lives here or in burn.py.

Rules this file exists to enforce:
  * commands are built as lists and run with shell=False, never as f-string shell lines
  * every builder is a pure function so it can be asserted on in a test without running
  * nothing is written into the caller's working directory
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

# ffmpeg 4.4 is what Ubuntu 22.04 ships; Homebrew ships 7.x/8.x. Every flag used anywhere
# in this project must work on both, so the floor is 4.4 and the code branches where the
# two genuinely differ (see `supports_fps_mode`).
MIN_FFMPEG = (4, 4)

# The sample rate and layout every Whisper implementation wants. Extracting straight to
# this shape means no engine has to resample and the stage cache holds one canonical file.
TARGET_SR = 16000
TARGET_CHANNELS = 1

# ffmpeg denoise presets. All are built-in filters, so there is no compiled dependency and
# macOS and Linux behave identically. `arnndn` is the same algorithm as the vendored
# rnnoise C build this project replaced.
DENOISE_FILTERS: dict[str, str] = {
    "none": "",
    "afftdn": "afftdn=nf=-25",
    "arnndn": "arnndn=m={rnnoise_model}",
    "anlmdn": "anlmdn",
    "speech": "highpass=f=80,afftdn=nf=-25,loudnorm",
}

# The bundled RNNoise weights. `arnndn` cannot run without a model file, and requiring the
# user to find one would make the preset a lie. See assets/rnnoise/README.md for provenance.
RNNOISE_MODEL = Path(__file__).parent / "assets" / "rnnoise" / "sh.rnnn"

# The name the model is copied to inside the temp working directory. Fixed, ASCII, no
# colon: see `denoise_audio`.
RNNOISE_LOCAL_NAME = "rnnoise.rnnn"

# ffmpeg's own player, which every ffmpeg build this project accepts ships alongside it.
# Only the cue editor uses it, and only to hear one cue at a time.
PLAYER = "ffplay"

# The shortest span worth asking a player for. A cue can be a fraction of a second, and
# ffplay rounds a duration to about 10 ms, so zero would play silence.
MIN_PLAY_SPAN = 0.05


class MediaError(RuntimeError):
    """ffmpeg or ffprobe failed, or the input is not usable."""


@dataclass(frozen=True, slots=True)
class MediaInfo:
    path: Path
    duration: float
    has_video: bool
    has_audio: bool
    width: int | None
    height: int | None
    fps: float | None
    sample_rate: int | None
    channels: int | None

    @property
    def is_audio_only(self) -> bool:
        return self.has_audio and not self.has_video


# --------------------------------------------------------------------------------------
# Command builders. Pure: they return argv lists and touch nothing.
# --------------------------------------------------------------------------------------


def probe_cmd(src: Path) -> list[str]:
    return [
        "ffprobe",
        "-v",
        "error",
        "-print_format",
        "json",
        "-show_format",
        "-show_streams",
        str(src),
    ]


def extract_audio_cmd(src: Path, dst: Path) -> list[str]:
    """Extract a canonical 16 kHz mono PCM WAV.

    Always mono: the earlier code in gozba2 asked ffmpeg for stereo and then let librosa
    silently downmix, which wasted the work and made the two halves of the pipeline
    disagree about the channel layout.

    Denoising deliberately does *not* happen here. It used to, as one extra `-af` on this
    same command, but that made the denoiser choice part of the extraction's cache key, so
    changing `--denoise` demuxed the source video all over again. As a separate pass over a
    16 kHz mono WAV it costs about a second, and every denoiser gets to reuse one extraction.
    """
    return [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-v",
        "error",
        "-i",
        str(src),
        "-vn",
        "-ac",
        str(TARGET_CHANNELS),
        "-ar",
        str(TARGET_SR),
        "-c:a",
        "pcm_s16le",
        str(dst),
    ]


def denoise_cmd(
    src: Path, dst: Path, *, preset: str, rnnoise_model: Path | None = None
) -> list[str]:
    """Filter an extracted WAV through a denoise preset, staying in the canonical shape.

    `-ac`/`-ar` are re-stated rather than assumed: the `speech` preset ends in `loudnorm`,
    which resamples to 192 kHz internally, and without them that rate would reach the file.
    """
    filt = denoise_filter(preset, rnnoise_model)
    if not filt:
        raise MediaError(f"denoise preset {preset!r} is a no-op; do not run a pass for it")
    return [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-v",
        "error",
        "-i",
        str(src),
        "-vn",
        "-af",
        filt,
        "-ac",
        str(TARGET_CHANNELS),
        "-ar",
        str(TARGET_SR),
        "-c:a",
        "pcm_s16le",
        str(dst),
    ]


def trim_cmd(src: Path, dst: Path, *, start: float = 0.0, end: float | None = None) -> list[str]:
    """Cut a fragment out of a source without re-encoding it.

    `-c copy` is the whole point. Re-encoding an hour to keep three minutes costs minutes
    of CPU for a result that is bit-for-bit worse than the source; a stream copy is one
    demux pass and finishes in about a second.

    Three deliberate choices:

    * **`-ss` goes before `-i`.** After `-i` it decodes and discards everything up to the
      start, which on an hour-long source is the whole cost we are avoiding. Before `-i`
      it is a demuxer seek.
    * **`-t` (a duration), not `-to`.** Where `-to` is measured when the input was seeked
      is not the same on ffmpeg 4.4 and 8.x, and this project supports both. A duration
      means the same thing everywhere, so the end is converted to one here.
    * **`-avoid_negative_ts make_zero`.** Without it the fragment keeps the source's
      timestamps, so a cut at 10:00 produces a file whose first packet claims to be at
      600s, and every cue derived from it would be offset by ten minutes. This is what
      makes "the first cue starts near 00:00:00" true rather than hoped for.

    The cost of a stream copy, stated plainly: video can only be cut at a keyframe, so the
    fragment may begin up to one keyframe interval (a few seconds on a typical YouTube
    mp4) before the requested start. Nothing downstream is misaligned by that, because the
    transcript is made from this file and not from the original; only the boundary is
    approximate. The pipeline re-probes the result rather than trusting `end - start`.
    """
    if start < 0:
        raise MediaError(f"start must not be negative, got {start}")
    if end is not None and end <= start:
        raise MediaError(
            f"end ({format_timecode(end)}) must be after start ({format_timecode(start)})"
        )
    if not start and end is None:
        raise MediaError("neither a start nor an end was given; there is nothing to trim")

    cmd = ["ffmpeg", "-y", "-hide_banner", "-v", "error"]
    if start:
        cmd += ["-ss", f"{start:.3f}"]
    cmd += ["-i", str(src)]
    if end is not None:
        cmd += ["-t", f"{end - start:.3f}"]
    # -sn/-dn: a copied subtitle or data stream is the one thing that can make an
    # otherwise fine stream copy fail on a container change, and neither is transcribed.
    cmd += ["-c", "copy", "-avoid_negative_ts", "make_zero", "-sn", "-dn", str(dst)]
    return cmd


def play_span_cmd(src: Path, *, start: float, end: float) -> list[str]:
    """Play the span one cue covers, and nothing else.

    Here rather than in the GUI because ffplay is ffmpeg: non-negotiable 1 says every
    invocation is built by a function with a command-construction test, and "it only
    plays audio" is not an exemption. The editor uses it because reading speed is the one
    number in the quality report a human cannot judge by eye.

    * **`-ss` before `-i`**, for the same reason `trim_cmd` does it: a demuxer seek rather
      than decoding an hour to reach minute 58.
    * **`-nodisp`.** The input is a WAV, but ffplay opens an SDL window for the waveform
      even so, and a second window appearing over the editor is not what "listen" means.
    * **`-autoexit`.** Without it ffplay sits there after the span has finished and the
      next click starts a second one on top of it.
    * A floor under the duration, because a cue can be shorter than the 10 ms ffplay
      rounds to and a request for zero seconds plays nothing at all.
    """
    if start < 0:
        raise MediaError(f"start must not be negative, got {start}")
    return [
        PLAYER,
        "-hide_banner",
        "-loglevel",
        "error",
        "-nodisp",
        "-autoexit",
        "-ss",
        f"{start:.3f}",
        "-t",
        f"{max(end - start, MIN_PLAY_SPAN):.3f}",
        str(src),
    ]


def parse_timecode(value: str) -> float:
    """`SS`, `MM:SS` or `HH:MM:SS`, with optional fractional seconds, to seconds.

    A bare number is seconds and may exceed 59, because `--start 90` is a thing people
    type. Once a colon appears the fields below the first are clock fields and are bounded,
    so `1:75` is rejected rather than quietly read as 135 seconds.
    """
    text = (value or "").strip()
    if not text:
        raise MediaError("empty timecode; expected SS, MM:SS or HH:MM:SS")
    parts = text.split(":")
    if len(parts) > 3:
        raise MediaError(f"invalid timecode {value!r}; expected SS, MM:SS or HH:MM:SS")
    try:
        numbers = [float(part) for part in parts]
    except ValueError as exc:
        raise MediaError(f"invalid timecode {value!r}; expected SS, MM:SS or HH:MM:SS") from exc
    if any(number < 0 for number in numbers):
        raise MediaError(f"invalid timecode {value!r}; time does not run backwards")
    if len(parts) > 1 and any(number >= 60 for number in numbers[1:]):
        raise MediaError(f"invalid timecode {value!r}; minutes and seconds must be under 60")
    seconds = 0.0
    for number in numbers:
        seconds = seconds * 60 + number
    return seconds


def format_timecode(seconds: float) -> str:
    """The inverse, for log lines and error messages. Always HH:MM:SS.mmm."""
    whole = int(seconds)
    return f"{whole // 3600:02d}:{whole % 3600 // 60:02d}:{whole % 60:02d}.{round((seconds - whole) * 1000):03d}"


def denoise_filter(name: str, rnnoise_model: Path | None = None) -> str:
    """Resolve a denoise preset name to an ffmpeg filter string."""
    if name not in DENOISE_FILTERS:
        raise MediaError(f"unknown denoise preset {name!r}; choose from {sorted(DENOISE_FILTERS)}")
    template = DENOISE_FILTERS[name]
    if "{rnnoise_model}" in template:
        if rnnoise_model is None:
            raise MediaError("the arnndn preset needs an rnnoise model file")
        # ffmpeg filter options are colon separated, so a real path would split the option
        # the moment it contained a colon, and there is no escaping that survives shell,
        # filtergraph and filter-option parsing intact. Callers pass the fixed relative
        # name the model was copied to inside a temp cwd. See `denoise_audio`.
        return template.format(rnnoise_model=str(rnnoise_model))
    return template


# --------------------------------------------------------------------------------------
# Execution
# --------------------------------------------------------------------------------------


def run(
    cmd: list[str], *, cwd: Path | None = None, dry_run: bool = False
) -> subprocess.CompletedProcess[str]:
    """Run a command with shell=False. `dry_run` prints and returns without executing."""
    if dry_run:
        print("+ " + " ".join(cmd))
        return subprocess.CompletedProcess(cmd, 0, "", "")
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(cwd) if cwd else None,
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError as exc:
        raise MediaError(
            f"{cmd[0]} is not installed or not on PATH. Run: subtitler doctor --install"
        ) from exc
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout or "").strip().splitlines()[-12:]
        raise MediaError(f"{cmd[0]} failed ({proc.returncode}):\n" + "\n".join(tail))
    return proc


def probe(src: Path, *, dry_run: bool = False) -> MediaInfo:
    if not dry_run and not src.exists():
        raise MediaError(f"input not found: {src}")
    proc = run(probe_cmd(src), dry_run=dry_run)
    if dry_run:
        return MediaInfo(src, 0.0, False, True, None, None, None, TARGET_SR, TARGET_CHANNELS)

    data = json.loads(proc.stdout or "{}")
    streams = data.get("streams", [])
    video = next((s for s in streams if s.get("codec_type") == "video"), None)
    audio = next((s for s in streams if s.get("codec_type") == "audio"), None)

    if audio is None:
        raise MediaError(f"{src} has no audio stream; there is nothing to transcribe")

    duration = _duration_of(data, audio)

    # A cover-art JPEG inside an mp3 shows up as a video stream. Treating that as video
    # would send an audio file down the real-video burn path and produce a 1-frame clip.
    is_real_video = bool(video) and video.get("disposition", {}).get("attached_pic", 0) != 1

    return MediaInfo(
        path=src,
        duration=duration,
        has_video=is_real_video,
        has_audio=True,
        width=int(video["width"]) if is_real_video and video.get("width") else None,
        height=int(video["height"]) if is_real_video and video.get("height") else None,
        fps=_parse_fps(video.get("r_frame_rate")) if is_real_video else None,
        sample_rate=int(audio["sample_rate"]) if audio.get("sample_rate") else None,
        channels=int(audio["channels"]) if audio.get("channels") else None,
    )


def _duration_of(data: dict, audio_stream: dict) -> float:
    for candidate in (data.get("format", {}).get("duration"), audio_stream.get("duration")):
        try:
            value = float(candidate)
        except (TypeError, ValueError):
            continue
        if value > 0:
            return value
    raise MediaError("could not determine duration from ffprobe output")


def _parse_fps(rate: str | None) -> float | None:
    if not rate:
        return None
    try:
        num, _, den = rate.partition("/")
        return float(num) / float(den) if den and float(den) else float(num)
    except (ValueError, ZeroDivisionError):
        return None


def extract_audio(src: Path, dst: Path, *, dry_run: bool = False) -> Path:
    dst.parent.mkdir(parents=True, exist_ok=True)
    run(extract_audio_cmd(src, dst), dry_run=dry_run)
    return dst


def trim(
    src: Path,
    dst: Path,
    *,
    start: float = 0.0,
    end: float | None = None,
    dry_run: bool = False,
) -> Path:
    """Stream-copy the fragment between `start` and `end` into `dst`. See `trim_cmd`."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    run(trim_cmd(src, dst, start=start, end=end), dry_run=dry_run)
    return dst


def denoise_audio(
    src: Path,
    dst: Path,
    *,
    preset: str,
    rnnoise_model: Path | None = None,
    dry_run: bool = False,
) -> Path:
    """Run one denoise preset over an extracted WAV.

    `arnndn` takes a model *path* as a filter option, and filter options are colon
    separated, so the moment the path contains a colon the filtergraph splits in the wrong
    place. This is the same problem `burn.py` has with `ass=f=...`, and it gets the same
    answer: copy the model into a temp directory under a fixed ASCII name, run ffmpeg with
    `cwd` set there, and reference the relative name. Input and output stay as argv entries
    where no parsing beyond the shell applies, and there is no shell. No escaping is
    attempted, because no escaping is correct for all three parsers at once.
    """
    if preset not in DENOISE_FILTERS:
        raise MediaError(
            f"unknown denoise preset {preset!r}; choose from {sorted(DENOISE_FILTERS)}"
        )
    dst.parent.mkdir(parents=True, exist_ok=True)

    if "{rnnoise_model}" not in DENOISE_FILTERS[preset]:
        run(denoise_cmd(src, dst, preset=preset), dry_run=dry_run)
        return dst

    model = rnnoise_model or RNNOISE_MODEL
    if not model.exists():
        raise MediaError(
            f"the arnndn preset needs an rnnoise model and {model} is missing; "
            "reinstall subtitler, or pick another --denoise preset"
        )
    with TemporaryDirectory(prefix="subtitler-denoise-") as tmp:
        work = Path(tmp)
        shutil.copyfile(model, work / RNNOISE_LOCAL_NAME)
        cmd = denoise_cmd(
            src.resolve(), dst.resolve(), preset=preset, rnnoise_model=Path(RNNOISE_LOCAL_NAME)
        )
        run(cmd, cwd=work, dry_run=dry_run)
    return dst


# --------------------------------------------------------------------------------------
# Capability probing. Used by doctor.py and by the burn path's version branches.
# --------------------------------------------------------------------------------------


def ffmpeg_version() -> tuple[int, int] | None:
    """Return (major, minor), or None if ffmpeg is missing or unparseable."""
    if shutil.which("ffmpeg") is None:
        return None
    try:
        proc = subprocess.run(
            ["ffmpeg", "-version"], capture_output=True, text=True, check=False, timeout=15
        )
    except (OSError, subprocess.SubprocessError):
        return None
    match = re.search(r"ffmpeg version n?(\d+)\.(\d+)", proc.stdout or "")
    if not match:
        return None
    return int(match.group(1)), int(match.group(2))


def supports_fps_mode(version: tuple[int, int] | None) -> bool:
    """`-fps_mode` replaced `-vsync` in ffmpeg 5.0. Both exist in 5.x; only one in 4.x."""
    return bool(version and version[0] >= 5)
