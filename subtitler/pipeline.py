"""Stage orchestration.

    [fetch] -> [trim] -> probe -> extract -> [denoise] -> transcribe -> cues -> [fix]
            -> render -> [burn]

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
from subtitler import fetch as fetch_mod
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
    # Exactly one of these two says where the media comes from. `from_source` picks.
    input: Path | None = None
    url: str | None = None
    # The fragment to keep, as the user typed it: `SS`, `MM:SS` or `HH:MM:SS`. Kept as text
    # rather than seconds for the same reason `canvas` is, so that one parser with one error
    # message serves the CLI, the GUI and anything else that builds a config.
    start: str | None = None
    end: str | None = None
    out_dir: Path | None = None
    engine: str = "auto"
    model: str = "large-v3"
    device: str = "auto"
    # 0 means decode sequentially. Only faster-whisper on CUDA honours it; see
    # `FasterWhisperEngine._prompt_for` for what it costs.
    batch_size: int = 0
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

    @classmethod
    def from_source(cls, source: str | Path, **kwargs: Any) -> RunConfig:
        """Build a config from whatever the user typed.

        Anything starting with http:// or https:// is a URL and goes to `url`; everything
        else is a path. This is the one place that decision is made, so the CLI, the GUI and
        any script agree on it rather than each writing their own sniffing.
        """
        if fetch_mod.is_url(source):
            return cls(url=str(source).strip(), **kwargs)
        return cls(input=Path(source), **kwargs)

    @property
    def source(self) -> str:
        return self.url or str(self.input or "")


@dataclass(slots=True)
class RunResult:
    input: Path
    source: str = ""
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
            # What the user asked for, which is not the same file for a URL or a trim run.
            "source": self.source or str(self.input),
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


def trim_window(cfg: RunConfig) -> tuple[float, float | None]:
    """`--start`/`--end` as seconds. Public so a form layer can validate before running."""
    start = media.parse_timecode(cfg.start) if cfg.start else 0.0
    end = media.parse_timecode(cfg.end) if cfg.end else None
    if end is not None and end <= start:
        raise media.MediaError(
            f"--end ({media.format_timecode(end)}) must be after "
            f"--start ({media.format_timecode(start)})"
        )
    return start, end


def wants_video(cfg: RunConfig) -> bool:
    """Whether this run will produce a picture, and so whether it needs one.

    The only consumer of a video stream is the burn. A `--srt-only` or `--no-burn` run that
    downloaded 1080p would spend the user's bandwidth to produce a text file.
    """
    return cfg.burn and not cfg.srt_only


def run_pipeline(cfg: RunConfig, *, log: Any = print) -> RunResult:
    started = time.monotonic()

    # Before any work: a typo in --force or in --start must be rejected on the spot, not
    # after ffprobe has printed a line implying the run started, and certainly not after a
    # 200 MB download.
    forced = cache_mod.invalidated_from(cfg.force)
    start_s, end_s = trim_window(cfg)

    if cfg.url:
        if cfg.out_dir is None:
            # Non-negotiable 4. A file input has a directory of its own to write beside;
            # a URL has none, and defaulting to the CWD is exactly what that rule forbids.
            raise media.MediaError(
                "a URL run needs an output directory: pass -o DIR. "
                "Nothing is ever written into the current directory."
            )
        out_dir = cfg.out_dir.expanduser().resolve()
        # Named from the URL, not from the video's title: the directory has to exist before
        # anything is downloaded, and learning the title costs a network round trip that a
        # warm run must not pay.
        work = out_dir / ".subtitler" / fetch_mod.work_stem(cfg.url)
    elif cfg.input is not None:
        src_in = cfg.input.expanduser().resolve()
        out_dir = (cfg.out_dir or src_in.parent).expanduser().resolve()
        work = out_dir / ".subtitler" / src_in.stem
    else:
        raise media.MediaError("nothing to transcribe: pass a file or a URL")

    cache = cache_mod.StageCache(
        work,
        forced=forced,
        # A dry run prints commands instead of running them, so it must neither read the
        # cache (it would report stages as done that it never did) nor write to it.
        enabled=not cfg.dry_run,
    )

    if cfg.url:
        src, media_id, stem, label = _fetch(cfg, cache, work, log)
    else:
        src = cfg.input.expanduser().resolve()  # type: ignore[union-attr]
        media_id = cache_mod.content_id(src) if cache.enabled else "dry-run"
        stem, label = src.stem, src.name

    if start_s or end_s is not None:
        src, media_id = _trim(cfg, cache, src, media_id, work, start_s, end_s, log)
        label = f"{label} [{media.format_timecode(start_s)} to " + (
            f"{media.format_timecode(end_s)}]" if end_s is not None else "the end]"
        )

    # Probed after the trim, never before: the burn's `-t` and the canvas length have to be
    # the fragment's real duration, and a stream copy lands on a keyframe, so `end - start`
    # is arithmetic rather than fact.
    info = media.probe(src, dry_run=cfg.dry_run)
    log(f"input: {label} ({info.duration:.1f}s, {'video' if info.has_video else 'audio only'})")

    audio_wav, audio_key = _audio(cfg, cache, src, media_id, work, log)

    engine = resolve(cfg.engine, model=cfg.model, device=cfg.device, batch_size=cfg.batch_size)
    described = engine.describe()
    log(f"engine: {engine.name} ({described.get('model')}, {described.get('device', 'n/a')})")
    if described.get("batch_size"):
        # Loud, because it changes the transcript and not just its speed.
        log(
            f"batched decoding: {described['batch_size']} chunks at a time. "
            "The steering prompt is NOT sent in this mode; batched decoding echoes it "
            "back as transcript text. Use --batch-size 0 to keep it."
        )
    elif cfg.batch_size and engine.name == "faster-whisper":
        log(f"--batch-size {cfg.batch_size} ignored: batching only helps on CUDA")

    opts = TranscribeOptions(language=cfg.language)
    if cfg.prompt is not None:
        opts = TranscribeOptions(language=cfg.language, initial_prompt=cfg.prompt or None)

    if cfg.dry_run:
        log("dry run: stopping before transcription")
        return RunResult(
            input=src,
            source=cfg.source,
            engine=engine.name,
            elapsed_s=time.monotonic() - started,
        )

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
        source=cfg.source,
        srt=srt_path,
        vtt=vtt_path,
        transcript=transcript,
        cues=cues,
        lint=problems,
        engine=engine.name,
        fix=fix_report,
    )

    if wants_video(cfg):
        # `src` here is the trimmed fragment when one was asked for, which is the point:
        # burning onto the original would export a full-length video carrying subtitles
        # that match three minutes of it.
        result.video = _burn(
            cfg, cache, src, media_id, info, cues, cues_key, out_dir / f"{stem}.subbed.mp4", log
        )

    result.cached = tuple(cache.hits)
    result.elapsed_s = time.monotonic() - started
    return result


# --------------------------------------------------------------------------------------
# Stages. Each returns its artifact plus its cache key, which is the next stage's input.
# --------------------------------------------------------------------------------------


def _fetch(
    cfg: RunConfig,
    cache: cache_mod.StageCache,
    work: Path,
    log: Any,
) -> tuple[Path, str, str, str]:
    """Download the URL into the work directory. Returns (media, key, stem, label).

    The stage's key is the URL plus which shape was asked for, and nothing else: there is
    nothing to content-address before the download has happened, and asking the site
    whether the upload changed would put a network round trip on every warm run. So
    changing `--start`, the engine or the style re-uses the download, and `--force fetch`
    is how you say "the upstream video changed".

    The downloaded file's extension is not fixed in advance (it is whatever the site
    serves), so the cached record `fetch.json` is consulted first and names the artifact
    the cache then checks for. An unreadable record is removed rather than trusted, which
    turns it into a plain "missing artifact" miss.
    """
    assert cfg.url is not None
    kind = "video" if wants_video(cfg) else "audio"
    info_path = work / fetch_mod.INFO_NAME

    if cfg.dry_run:
        # Nothing is downloaded, and nothing may be read from the cache, so the rest of the
        # dry run works from the name the download would have had.
        log(f"dry run: would fetch {cfg.url} ({kind})")
        nominal = work / f"{fetch_mod.DOWNLOAD_STEM}.mp4"
        return nominal, "dry-run", fetch_mod.work_stem(cfg.url), cfg.url

    known = fetch_mod.read_info(info_path)
    if known is None and info_path.exists():
        info_path.unlink()

    entry = cache.begin(
        "fetch",
        input_hash=fetch_mod.url_id(cfg.url),
        params=fetch_mod.cache_params(kind),
        artifacts=(info_path, known.path) if known else (info_path,),
    )
    if entry.hit and known is not None:
        log(f"fetch: cached ({known.path.name}, {known.title or known.id})")
        return known.path, entry.key, known.stem, known.title or known.path.name

    log(f"fetching {cfg.url} ({kind})")
    fetched = fetch_mod.fetch(cfg.url, work, kind=kind, progress=log)
    fetch_mod.write_info(info_path, fetched)
    cache.commit(entry)
    log(f"fetched: {fetched.title or fetched.id} -> {fetched.path.name}")
    return fetched.path, entry.key, fetched.stem, fetched.title or fetched.path.name


def _trim(
    cfg: RunConfig,
    cache: cache_mod.StageCache,
    src: Path,
    media_id: str,
    work: Path,
    start: float,
    end: float | None,
    log: Any,
) -> tuple[Path, str]:
    """Cut the fragment out before anything else looks at the media.

    Position is the design. Trimming here means the extraction, the transcript, the cues
    and the burn all see a file that *begins* at the fragment, so cue timestamps come out
    relative to it with no offset arithmetic anywhere, and the burn re-encodes three
    minutes rather than an hour. Doing it at the end would need every one of those to know
    about a window, and the first thing to get it wrong would be silent.
    """
    dst = work / f"trim{src.suffix or '.mp4'}"
    window = f"{media.format_timecode(start)} to " + (
        media.format_timecode(end) if end is not None else "the end"
    )
    entry = cache.begin(
        "trim",
        input_hash=media_id,
        # The two timecodes and nothing else: the cut is a stream copy, so no codec,
        # quality or size setting can change what comes out.
        params={"start": round(start, 3), "end": round(end, 3) if end is not None else None},
        artifacts=(dst,),
    )
    if entry.hit:
        log(f"trim: cached ({window})")
        return dst, entry.key

    media.trim(src, dst, start=start, end=end, dry_run=cfg.dry_run)
    cache.commit(entry)
    log(f"trimmed: {window} (stream copy)")
    return dst, entry.key


def _audio(
    cfg: RunConfig,
    cache: cache_mod.StageCache,
    src: Path,
    media_id: str,
    work: Path,
    log: Any,
) -> tuple[Path, str]:
    """extract, then optionally denoise. Returns the WAV the recognizer should read.

    `media_id` is the id of the media actually being transcribed: the source file's content
    id normally, and the `trim` stage's key when a fragment was cut, so that changing the
    window re-extracts from the new fragment rather than serving the old one's audio.
    """
    extract_wav = work / "extract.wav"
    entry = cache.begin(
        "extract",
        input_hash=media_id,
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
    media_id: str,
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
            # The media that supplies the pixels and the audio: the trimmed fragment when
            # there is one, so a re-cut re-burns.
            "source": media_id,
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
