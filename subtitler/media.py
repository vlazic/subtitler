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


def extract_audio_cmd(
    src: Path, dst: Path, *, denoise: str = "none", rnnoise_model: Path | None = None
) -> list[str]:
    """Extract a canonical 16 kHz mono PCM WAV, optionally denoising on the way.

    Always mono: the earlier code in gozba2 asked ffmpeg for stereo and then let librosa
    silently downmix, which wasted the work and made the two halves of the pipeline
    disagree about the channel layout.
    """
    cmd = ["ffmpeg", "-y", "-hide_banner", "-v", "error", "-i", str(src), "-vn"]
    filt = denoise_filter(denoise, rnnoise_model)
    if filt:
        cmd += ["-af", filt]
    cmd += [
        "-ac",
        str(TARGET_CHANNELS),
        "-ar",
        str(TARGET_SR),
        "-c:a",
        "pcm_s16le",
        str(dst),
    ]
    return cmd


def denoise_filter(name: str, rnnoise_model: Path | None = None) -> str:
    """Resolve a denoise preset name to an ffmpeg filter string."""
    if name not in DENOISE_FILTERS:
        raise MediaError(f"unknown denoise preset {name!r}; choose from {sorted(DENOISE_FILTERS)}")
    template = DENOISE_FILTERS[name]
    if "{rnnoise_model}" in template:
        if rnnoise_model is None:
            raise MediaError("the arnndn preset needs an rnnoise model file")
        # ffmpeg filter options are colon separated, so a path with a colon would split the
        # option. Callers pass a temp-dir-relative path for exactly this reason.
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


def extract_audio(
    src: Path,
    dst: Path,
    *,
    denoise: str = "none",
    rnnoise_model: Path | None = None,
    dry_run: bool = False,
) -> Path:
    dst.parent.mkdir(parents=True, exist_ok=True)
    run(extract_audio_cmd(src, dst, denoise=denoise, rnnoise_model=rnnoise_model), dry_run=dry_run)
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
