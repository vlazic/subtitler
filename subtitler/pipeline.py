"""Stage orchestration.

    probe -> extract -> [denoise] -> transcribe -> cues -> [fix] -> render -> [burn]

Every stage is a pure-ish function elsewhere; this module only sequences them and owns the
work directory. The content-addressed stage cache lands in Phase 5, which is why the work
directory already carries the artifacts as JSON rather than keeping them in memory.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from subtitler import burn as burn_mod
from subtitler import media, render
from subtitler.cues import CueConfig, lint_cues, segments_to_cues
from subtitler.engines import resolve
from subtitler.engines.base import TranscribeOptions
from subtitler.model import Cue, Transcript, cues_to_dict, transcript_to_dict, write_json

DEFAULT_CANVAS = (1280, 720)


@dataclass(frozen=True, slots=True)
class RunConfig:
    input: Path
    out_dir: Path | None = None
    engine: str = "auto"
    model: str = "large-v3"
    device: str = "auto"
    language: str = "sr"
    prompt: str | None = None
    denoise: str = "none"
    burn: bool = True
    soft_mux: bool = False
    srt_only: bool = False
    canvas: str = "1280x720"
    canvas_color: str = "0x101010"
    style_preset: str = "outline"
    font: str | None = None
    font_size: int | None = None
    cues: CueConfig = field(default_factory=CueConfig)
    dry_run: bool = False
    verbose: int = 0


@dataclass(slots=True)
class RunResult:
    input: Path
    srt: Path | None = None
    vtt: Path | None = None
    video: Path | None = None
    transcript: Transcript | None = None
    cues: tuple[Cue, ...] = ()
    lint: list[str] = field(default_factory=list)
    engine: str = ""
    elapsed_s: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "input": str(self.input),
            "srt": str(self.srt) if self.srt else None,
            "vtt": str(self.vtt) if self.vtt else None,
            "video": str(self.video) if self.video else None,
            "engine": self.engine,
            "cue_count": len(self.cues),
            "lint_violations": self.lint,
            "elapsed_s": round(self.elapsed_s, 2),
            "rtf": round(self.transcript.rtf, 3) if self.transcript else None,
        }


def parse_canvas(spec: str) -> tuple[int, int]:
    try:
        width, height = (int(part) for part in spec.lower().split("x", 1))
    except ValueError as exc:
        raise media.MediaError(f"invalid canvas {spec!r}; expected WIDTHxHEIGHT") from exc
    if width <= 0 or height <= 0:
        raise media.MediaError(f"invalid canvas {spec!r}")
    return burn_mod.even(width), burn_mod.even(height)


def run_pipeline(cfg: RunConfig, *, log: Any = print) -> RunResult:
    started = time.monotonic()
    src = cfg.input.expanduser().resolve()
    out_dir = (cfg.out_dir or src.parent).expanduser().resolve()
    work = out_dir / ".subtitler" / src.stem
    stem = src.stem

    info = media.probe(src, dry_run=cfg.dry_run)
    log(f"input: {src.name} ({info.duration:.1f}s, {'video' if info.has_video else 'audio only'})")

    audio_wav = work / "audio16k.wav"
    media.extract_audio(
        src,
        audio_wav,
        denoise=cfg.denoise,
        rnnoise_model=None,
        dry_run=cfg.dry_run,
    )

    engine = resolve(cfg.engine, model=cfg.model, device=cfg.device)
    log(f"engine: {engine.name} ({engine.describe().get('model')})")

    opts = TranscribeOptions(language=cfg.language)
    if cfg.prompt is not None:
        opts = TranscribeOptions(language=cfg.language, initial_prompt=cfg.prompt or None)

    if cfg.dry_run:
        log("dry run: stopping before transcription")
        return RunResult(input=src, engine=engine.name, elapsed_s=time.monotonic() - started)

    transcript = engine.transcribe(audio_wav, opts)
    write_json(work / "transcript.json", transcript_to_dict(transcript))
    log(
        f"transcribed: {len(transcript.segments)} segments in "
        f"{transcript.runtime_s:.1f}s (rtf {transcript.rtf:.2f})"
    )

    cues = segments_to_cues(transcript.segments, cfg.cues)
    write_json(work / "cues.json", cues_to_dict(cues))

    srt_path = render.write_srt(out_dir / f"{stem}.srt", cues)
    vtt_path = render.write_vtt(out_dir / f"{stem}.vtt", cues)
    log(f"wrote: {srt_path.name}, {vtt_path.name} ({len(cues)} cues)")

    problems = lint_cues(cues, cfg.cues)
    if problems:
        # Phase 1 uses a naive one-cue-per-segment split, so violations here are expected
        # and reported rather than hidden. The Phase 4 splitter is what makes them go away.
        log(f"lint: {len(problems)} cue-quality violations (see `subtitler lint {srt_path.name}`)")

    result = RunResult(
        input=src,
        srt=srt_path,
        vtt=vtt_path,
        transcript=transcript,
        cues=cues,
        lint=problems,
        engine=engine.name,
    )

    if cfg.burn and not cfg.srt_only:
        result.video = _burn(cfg, src, info, audio_wav, cues, out_dir / f"{stem}.subbed.mp4", log)

    result.elapsed_s = time.monotonic() - started
    return result


def _burn(
    cfg: RunConfig,
    src: Path,
    info: media.MediaInfo,
    audio_wav: Path,
    cues: tuple[Cue, ...],
    dst: Path,
    log: Any,
) -> Path:
    if info.has_video and info.width and info.height:
        width, height = info.width, info.height
        video, audio = src, None
    else:
        width, height = parse_canvas(cfg.canvas)
        # Burn the original audio rather than the 16 kHz mono extraction: the extraction is
        # shaped for the recognizer, not for listening.
        video, audio = None, src
    burn_mod.burn(
        cues,
        dst,
        video=video,
        audio=audio,
        width=width,
        height=height,
        duration=info.duration,
        style_preset=cfg.style_preset,
        font_name=cfg.font,
        font_size=cfg.font_size,
        canvas_color=cfg.canvas_color,
        dry_run=cfg.dry_run,
    )
    log(f"burned: {dst.name} ({width}x{height})")
    return dst
