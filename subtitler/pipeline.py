"""Stage orchestration.

    probe -> extract -> [denoise] -> transcribe -> cues -> [fix] -> render -> [burn]

Every stage is a pure-ish function elsewhere; this module only sequences them and owns the
work directory.

Each expensive stage goes through `cache.StageCache`: consult, and on a miss run it and
commit. The keys chain, so a change invalidates exactly what is downstream of it. `cache.py`
explains the chain; the shape to keep here is that a stage's `params` must contain
everything that can change its output and nothing that cannot, or the cache is either stale
or useless.

Render is not cached. Writing an SRT from cues.json takes a millisecond, and re-deriving it
on every run is what makes "the second run produces byte-identical output" a fact that gets
re-established rather than an artifact nobody touched.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from subtitler import burn as burn_mod
from subtitler import cache as cache_mod
from subtitler import media, postedit, render
from subtitler.cues import CueConfig, lint_cues, segments_to_cues
from subtitler.engines import resolve
from subtitler.engines.base import TranscribeOptions
from subtitler.model import (
    Cue,
    Transcript,
    cues_from_dict,
    cues_to_dict,
    read_json,
    transcript_from_dict,
    transcript_to_dict,
    write_json,
)

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
    # None means no correction pass. `--fix` is the only thing that sets it, so a run
    # without the flag never imports LiteLLM and never needs an API key.
    fix: postedit.FixConfig | None = None
    force: str | None = None
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
    cached: tuple[str, ...] = ()
    fix: dict[str, Any] | None = None
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
            "cached_stages": list(self.cached),
            "fix": self.fix,
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

    # Before any work: a typo in --force must be rejected on the spot, not after ffprobe
    # has already run and printed a line that implies the run started successfully.
    forced = cache_mod.invalidated_from(cfg.force)

    info = media.probe(src, dry_run=cfg.dry_run)
    log(f"input: {src.name} ({info.duration:.1f}s, {'video' if info.has_video else 'audio only'})")

    cache = cache_mod.StageCache(
        work,
        forced=forced,
        # A dry run prints commands instead of running them, so it must neither read the
        # cache (it would report stages as done that it never did) nor write to it.
        enabled=not cfg.dry_run,
    )
    source_id = cache_mod.content_id(src) if cache.enabled else "dry-run"

    audio_wav, audio_key = _audio(cfg, cache, src, source_id, work, log)

    engine = resolve(cfg.engine, model=cfg.model, device=cfg.device)
    log(f"engine: {engine.name} ({engine.describe().get('model')})")

    opts = TranscribeOptions(language=cfg.language)
    if cfg.prompt is not None:
        opts = TranscribeOptions(language=cfg.language, initial_prompt=cfg.prompt or None)

    if cfg.dry_run:
        log("dry run: stopping before transcription")
        return RunResult(input=src, engine=engine.name, elapsed_s=time.monotonic() - started)

    transcript, transcribe_key = _transcribe(cfg, cache, engine, opts, audio_wav, audio_key, log)
    cues, cues_key = _cues(cfg, cache, transcript, transcribe_key, log)

    fix_report: dict[str, Any] | None = None
    if cfg.fix is not None:
        cues, cues_key, fix_report = _fix(cfg, cache, cues, cues_key, log)

    srt_path = render.write_srt(out_dir / f"{stem}.srt", cues)
    vtt_path = render.write_vtt(out_dir / f"{stem}.vtt", cues)
    log(f"wrote: {srt_path.name}, {vtt_path.name} ({len(cues)} cues)")

    problems = lint_cues(cues, cfg.cues)
    if problems:
        log(f"lint: {len(problems)} cue-quality violations (see `subtitler lint {srt_path.name}`)")

    result = RunResult(
        input=src,
        srt=srt_path,
        vtt=vtt_path,
        transcript=transcript,
        cues=cues,
        lint=problems,
        engine=engine.name,
        fix=fix_report,
    )

    if cfg.burn and not cfg.srt_only:
        result.video = _burn(
            cfg, cache, src, source_id, info, cues, cues_key, out_dir / f"{stem}.subbed.mp4", log
        )

    result.cached = tuple(cache.hits)
    result.elapsed_s = time.monotonic() - started
    return result


# --------------------------------------------------------------------------------------
# Stages. Each returns its artifact plus its cache key, which is the next stage's input.
# --------------------------------------------------------------------------------------


def _audio(
    cfg: RunConfig,
    cache: cache_mod.StageCache,
    src: Path,
    source_id: str,
    work: Path,
    log: Any,
) -> tuple[Path, str]:
    """extract, then optionally denoise. Returns the WAV the recognizer should read."""
    extract_wav = work / "extract.wav"
    entry = cache.begin(
        "extract",
        input_hash=source_id,
        params={"sample_rate": media.TARGET_SR, "channels": media.TARGET_CHANNELS},
        artifacts=(extract_wav,),
    )
    if entry.hit:
        log("extract: cached")
    else:
        media.extract_audio(src, extract_wav, dry_run=cfg.dry_run)
        cache.commit(entry)

    if cfg.denoise == "none":
        return extract_wav, entry.key

    denoise_wav = work / "denoise.wav"
    params: dict[str, Any] = {"preset": cfg.denoise, "filter": media.DENOISE_FILTERS[cfg.denoise]}
    if "{rnnoise_model}" in media.DENOISE_FILTERS[cfg.denoise] and media.RNNOISE_MODEL.exists():
        # Key on the weights, not on their filename: swapping in a different model must
        # invalidate the denoised audio and everything derived from it.
        params["rnnoise_model"] = cache_mod.content_id(media.RNNOISE_MODEL)

    d_entry = cache.begin("denoise", input_hash=entry.key, params=params, artifacts=(denoise_wav,))
    if d_entry.hit:
        log(f"denoise: cached ({cfg.denoise})")
    else:
        media.denoise_audio(extract_wav, denoise_wav, preset=cfg.denoise, dry_run=cfg.dry_run)
        cache.commit(d_entry)
        log(f"denoise: {cfg.denoise}")
    return denoise_wav, d_entry.key


def _transcribe(
    cfg: RunConfig,
    cache: cache_mod.StageCache,
    engine: Any,
    opts: TranscribeOptions,
    audio_wav: Path,
    audio_key: str,
    log: Any,
) -> tuple[Transcript, str]:
    artifact = cache.work / "transcribe.json"
    entry = cache.begin(
        "transcribe",
        input_hash=audio_key,
        # `describe()` carries the resolved device and compute type, not just the model
        # name: a large-v3 decoded int8 on CPU is not the same transcript as float16 on
        # CUDA, and keying on `--model` alone would silently serve one for the other.
        params={"engine": engine.describe(), "options": asdict(opts)},
        artifacts=(artifact,),
    )
    if entry.hit:
        transcript = transcript_from_dict(read_json(artifact))
        log(f"transcribe: cached ({len(transcript.segments)} segments)")
        return transcript, entry.key

    transcript = engine.transcribe(audio_wav, opts)
    write_json(artifact, transcript_to_dict(transcript))
    cache.commit(entry)
    log(
        f"transcribed: {len(transcript.segments)} segments in "
        f"{transcript.runtime_s:.1f}s (rtf {transcript.rtf:.2f})"
    )
    return transcript, entry.key


def _cues(
    cfg: RunConfig,
    cache: cache_mod.StageCache,
    transcript: Transcript,
    transcribe_key: str,
    log: Any,
) -> tuple[tuple[Cue, ...], str]:
    artifact = cache.work / "cues.json"
    entry = cache.begin(
        "cues",
        input_hash=transcribe_key,
        params=asdict(cfg.cues),
        artifacts=(artifact,),
    )
    if entry.hit:
        cues = cues_from_dict(read_json(artifact))
        log(f"cues: cached ({len(cues)} cues)")
        return cues, entry.key

    cues = segments_to_cues(transcript.segments, cfg.cues)
    write_json(artifact, cues_to_dict(cues))
    cache.commit(entry)
    return cues, entry.key


def _fix(
    cfg: RunConfig,
    cache: cache_mod.StageCache,
    cues: tuple[Cue, ...],
    cues_key: str,
    log: Any,
) -> tuple[tuple[Cue, ...], str, dict[str, Any]]:
    """The optional LLM correction pass, as its own cached stage.

    Cached like every other stage, and the reason matters more here than elsewhere: this
    is the only stage that costs money. A re-run that re-billed the same 40 batches to
    produce the same file would be the cache's most expensive miss.

    Its key chains from `cues`, so `burn` sees the corrected cues and re-burns exactly when
    the correction changed. The report is stored alongside the cues so a warm run can still
    say what the cold run did.
    """
    assert cfg.fix is not None
    artifact = cache.work / "fix.json"
    entry = cache.begin(
        "fix",
        input_hash=cues_key,
        params=postedit.cache_params(cfg.fix, cfg.cues),
        artifacts=(artifact,),
    )
    if entry.hit:
        payload = read_json(artifact)
        fixed = cues_from_dict(payload)
        log(f"fix: cached ({len(fixed)} cues)")
        return fixed, entry.key, payload.get("report", {})

    log(f"fix: {len(cues)} cues through {cfg.fix.model}")
    fixed, report = postedit.fix_cues(cues, cfg.fix, cue_config=cfg.cues, log=log)
    write_json(artifact, {**cues_to_dict(fixed), "report": report.to_dict()})
    cache.commit(entry)
    log(
        f"fix: {report.changed}/{len(cues)} cues changed"
        + (f", {len(report.rejected)} batch(es) discarded" if report.rejected else "")
    )
    return fixed, entry.key, report.to_dict()


def _burn(
    cfg: RunConfig,
    cache: cache_mod.StageCache,
    src: Path,
    source_id: str,
    info: media.MediaInfo,
    cues: tuple[Cue, ...],
    cues_key: str,
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

    # Burn is not in the list of stages the phase brief names, and it is cached anyway,
    # because it is the only remaining expensive step: re-encoding a 109-second canvas
    # costs five seconds, which alone breaks the "a re-run finishes in under 2 seconds"
    # acceptance criterion. The source content id is in the key as well as the cues key,
    # since the pixels and the audio come from the source, not from the cues.
    entry = cache.begin(
        "burn",
        input_hash=cues_key,
        params={
            "source": source_id,
            "style_preset": cfg.style_preset,
            "font": cfg.font,
            "font_size": cfg.font_size,
            "canvas_color": cfg.canvas_color,
            "width": width,
            "height": height,
            "duration": round(info.duration, 3),
        },
        artifacts=(dst,),
    )
    if entry.hit:
        log(f"burn: cached ({dst.name})")
        return dst

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
    cache.commit(entry)
    log(f"burned: {dst.name} ({width}x{height})")
    return dst
