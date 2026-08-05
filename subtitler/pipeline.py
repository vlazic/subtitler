"""Stage orchestration.

    [fetch] -> [trim] -> probe -> extract -> [denoise] -> transcribe -> cues -> [fix]
            -> [edit] -> render -> [burn] -> [mux]

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
from subtitler import edits as edits_mod
from subtitler import fetch as fetch_mod
from subtitler import media, postedit, render
from subtitler.cues import CueConfig, lint_cues, segments_to_cues
from subtitler.engines import resolve
from subtitler.engines.base import TranscribeOptions, prompt_echoed
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
    # Stop once the subtitle files exist, before any video is encoded. What the GUI's
    # editor runs first, so a human can read the cues and correct them while the burn
    # (the only expensive step left) has not happened yet.
    review: bool = False
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
    muxed: Path | None = None
    transcript: Transcript | None = None
    cues: tuple[Cue, ...] = ()
    lint: list[str] = field(default_factory=list)
    # Things that went wrong without failing the run, in the order they were noticed. A
    # `lint` entry says a cue is hard to read; a warning here says the text may not be what
    # anybody said, which is not a cue-quality note and must not be reported as one.
    warnings: list[str] = field(default_factory=list)
    engine: str = ""
    cached: tuple[str, ...] = ()
    fix: dict[str, Any] | None = None
    # The key of the stage that produced `cues`. What hand corrections are recorded
    # against, so a later run can tell whether they are still about this text.
    cues_key: str = ""
    edits: dict[str, Any] | None = None
    elapsed_s: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "input": str(self.input),
            # What the user asked for, which is not the same file for a URL or a trim run.
            "source": self.source or str(self.input),
            "srt": str(self.srt) if self.srt else None,
            "vtt": str(self.vtt) if self.vtt else None,
            "video": str(self.video) if self.video else None,
            "muxed": str(self.muxed) if self.muxed else None,
            "engine": self.engine,
            "cue_count": len(self.cues),
            "lint_violations": self.lint,
            "warnings": self.warnings,
            "cached_stages": list(self.cached),
            "fix": self.fix,
            "edits": self.edits,
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


def output_dir(cfg: RunConfig) -> Path:
    """Where this run's files land. Public so a form can show it before anything starts.

    A URL has no directory of its own to write beside, and defaulting to the current one is
    exactly what non-negotiable 4 forbids, so `-o` is required for it. Raising here rather
    than inside the run means the GUI can say so while the button is still unpressed.
    """
    if cfg.url:
        if cfg.out_dir is None:
            raise media.MediaError(
                "a URL run needs an output directory: pass -o DIR. "
                "Nothing is ever written into the current directory."
            )
        return cfg.out_dir.expanduser().resolve()
    if cfg.input is None:
        raise media.MediaError("nothing to transcribe: pass a file or a URL")
    src = cfg.input.expanduser().resolve()
    return (cfg.out_dir or src.parent).expanduser().resolve()


def work_dir(cfg: RunConfig) -> Path:
    """The stage cache for this run, and where `edits.json` is read from and written to.

    Derived from the config alone and never from anything the run learned, so the editor
    can find the same directory the pipeline will, without having run anything first. For a
    URL that means naming it from the URL rather than from the video's title: the title
    costs a network round trip a warm run must not pay.
    """
    out = output_dir(cfg)
    stem = fetch_mod.work_stem(cfg.url) if cfg.url else cfg.input.expanduser().resolve().stem  # type: ignore[union-attr]
    return out / ".subtitler" / stem


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

    # Non-negotiable 4 lives in `output_dir`: a URL run without `-o` is refused there.
    out_dir = output_dir(cfg)
    work = work_dir(cfg)

    cache = cache_mod.StageCache(
        work,
        forced=forced,
        # A dry run prints commands instead of running them, so it must neither read the
        # cache (it would report stages as done that it never did) nor write to it.
        enabled=not cfg.dry_run,
    )

    if cfg.url:
        # A URL run asks the site for the window, so what lands is already the fragment and
        # the `trim` stage has nothing left to cut. See `_fetch`.
        src, media_id, stem, label = _fetch(cfg, cache, work, log, start=start_s, end=end_s)
        already_cut = True
    else:
        src = cfg.input.expanduser().resolve()  # type: ignore[union-attr]
        media_id = cache_mod.content_id(src) if cache.enabled else "dry-run"
        stem, label = src.stem, src.name
        already_cut = False

    if start_s or end_s is not None:
        if not already_cut:
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
    warnings = _prompt_echo_warnings(transcript, opts, log)
    warnings += _speechless_warnings(transcript, log)
    cues, cues_key = _cues(cfg, cache, transcript, transcribe_key, log)

    fix_report: dict[str, Any] | None = None
    if cfg.fix is not None:
        cues, cues_key, fix_report = _fix(cfg, cache, cues, cues_key, log)

    # The key the editor's corrections are recorded against is the one for the cues as they
    # are *presented*, which is after `fix` when it ran.
    review_key = cues_key
    cues, cues_key, edit_report = _edit(cfg, cache, cues, cues_key, log)

    srt_path = render.write_srt(out_dir / f"{stem}.srt", cues)
    vtt_path = render.write_vtt(out_dir / f"{stem}.vtt", cues)
    log(f"wrote: {srt_path.name}, {vtt_path.name} ({len(cues)} cues)")
    warnings += _no_cues_warning(cues, transcript, log)

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
        warnings=warnings,
        engine=engine.name,
        fix=fix_report,
        cues_key=review_key,
        edits=edit_report,
    )

    if cfg.review:
        # Everything cheap is done and nothing has been encoded. The cues are on disk as
        # .srt and .vtt either way, so stopping here costs a caller that ignores `review`
        # nothing but the video it did not ask for yet.
        log(f"review: {len(cues)} cues ready to check")
        result.cached = tuple(cache.hits)
        result.elapsed_s = time.monotonic() - started
        return result

    if wants_video(cfg):
        # `src` here is the trimmed fragment when one was asked for, which is the point:
        # burning onto the original would export a full-length video carrying subtitles
        # that match three minutes of it.
        result.video, burn_key = _burn(
            cfg, cache, src, media_id, info, cues, cues_key, out_dir / f"{stem}.subbed.mp4", log
        )
    else:
        burn_key = ""

    if cfg.soft_mux:
        result.muxed = _mux(
            cfg,
            cache,
            src=src,
            media_id=media_id,
            info=info,
            burned=result.video,
            burn_key=burn_key,
            subs=srt_path,
            cues_key=cues_key,
            out_dir=out_dir,
            stem=stem,
            log=log,
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
    *,
    start: float = 0.0,
    end: float | None = None,
) -> tuple[Path, str, str, str]:
    """Download the URL into the work directory. Returns (media, key, stem, label).

    The stage's key is the URL, which shape was asked for, and **the window**: there is
    nothing to content-address before the download has happened, and asking the site
    whether the upload changed would put a network round trip on every warm run, so
    `--force fetch` is how you say "the upstream video changed".

    The window is in that key because the window is now part of what gets downloaded. This
    stage used to fetch the whole source and let `trim` cut it, which meant a sixty-second
    excerpt of a four-hour lecture transferred four hours; asking the site for the span
    instead is defect-for-defect the same tradeoff as asking it for audio on `--srt-only`.
    The consequence, stated plainly: moving `--start` re-downloads. It re-downloads the new
    window, which is the thing the user asked to look at, and not the source it came from.

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
        params=fetch_mod.cache_params(kind, start=start, end=end),
        artifacts=(info_path, known.path) if known else (info_path,),
    )
    if entry.hit and known is not None:
        log(f"fetch: cached ({known.path.name}, {known.title or known.id})")
        return known.path, entry.key, known.stem, known.title or known.path.name

    if start or end is not None:
        window = f"{media.format_timecode(start)} to " + (
            media.format_timecode(end) if end is not None else "the end"
        )
        log(f"fetching {cfg.url} ({kind}, only {window})")
    else:
        log(f"fetching {cfg.url} ({kind})")
    fetched = fetch_mod.fetch(cfg.url, work, kind=kind, progress=log, start=start, end=end)
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

    **ffmpeg's exit code is not evidence that a fragment exists.** A `--start` past the end
    of the source makes it write a few hundred bytes of container header, no audio at all,
    and exit 0. Committing that to the cache turned one bad timecode into a work directory
    that reported `trim: cached` and died in ffprobe on every subsequent run, forever. So
    the start is checked against the source's real duration first, the cut result is probed
    before the stage is committed, and a cached fragment that no longer probes is re-cut
    rather than served.
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
        if _probes_clean(dst):
            log(f"trim: cached ({window})")
            return dst, entry.key
        # A husk committed by a version that trusted ffmpeg's exit code. Recorded as the
        # miss it is, so the run summary does not claim a stage it is about to redo.
        cache.hits.remove(entry.name)
        cache.misses.append(entry.name)
        log(f"trim: the cached fragment for {window} is unreadable; cutting it again")

    if not cfg.dry_run:
        duration = media.probe(src).duration
        if start >= duration:
            raise media.MediaError(
                f"--start {media.format_timecode(start)} is at or past the end of "
                f"{src.name}, which is {media.format_timecode(duration)} long. "
                "There would be nothing left to transcribe."
            )

    media.trim(src, dst, start=start, end=end, dry_run=cfg.dry_run)
    if not cfg.dry_run and not _probes_clean(dst):
        size = dst.stat().st_size if dst.exists() else 0
        raise media.MediaError(
            f"the cut to {window} produced {dst.name} ({size} bytes), which ffprobe cannot "
            "read as media. The trim has NOT been cached, so fixing the window and running "
            "again re-cuts rather than serving this file."
        )
    cache.commit(entry)
    log(f"trimmed: {window} (stream copy)")
    return dst, entry.key


def _probes_clean(fragment: Path) -> bool:
    """Whether `fragment` is media ffprobe can read. See `_trim` for why this is asked."""
    try:
        media.probe(fragment)
    except (media.MediaError, ValueError):
        return False
    return True


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


def _prompt_echo_warnings(transcript: Transcript, opts: TranscribeOptions, log: Any) -> list[str]:
    """Say so when the transcript is the steering prompt rather than the speech.

    Without this a run that produced nothing but "Zadrži srpski jezik i latinično pismo."
    reported one cue, `lint_violations: []` and success, which is confident-looking garbage:
    the text is well-formed Serbian of a readable length, so nothing downstream objects to
    it. Checked in the pipeline and not in the engine so that every backend is covered
    (the cloud ones send the prompt too and have no retry), and against the transcript that
    is actually being used, including one served from the cache.
    """
    notes: list[str] = []
    retried = transcript.params.get("prompt_echo_retry")
    if retried:
        notes.append(
            f"the first decode returned the steering prompt as transcript text ({retried!r}); "
            "it was decoded again without the prompt, so this transcript is unsteered"
        )
    echo_n, echo_text = prompt_echoed(transcript.text, opts.initial_prompt)
    if echo_n:
        notes.append(
            f"this transcript still contains {echo_n} words of the steering prompt "
            f"({echo_text!r}). Check it against the audio: the clip may hold no speech, and "
            "`--prompt ''` decodes it without any prompt to echo"
        )
    for note in notes:
        log(f"warning: {note}")
    return notes


def _speechless_warnings(transcript: Transcript, log: Any) -> list[str]:
    """Say so when the speech-free gate removed text, and name what it removed.

    A shorter transcript must never be a silent one. The gate exists because Whisper
    invents confident text over music and titles (`engines/base.is_speechless`), and a user
    who is handed subtitles with a line quietly missing has no way to tell that from a
    recogniser that simply did not hear it. Read out of `params` rather than recomputed, so
    a run served from the cache reports exactly what the cold run did.
    """
    dropped = transcript.params.get("speechless_dropped") or []
    if not dropped:
        return []
    notes = [
        f"{len(dropped)} segment(s) held no speech and were removed from the subtitles. "
        "Whisper invents text over music, titles and applause, so this is usually right; "
        "check the audio if you expected words there. Removed: " + "; ".join(dropped)
    ]
    for note in notes:
        log(f"warning: {note}")
    return notes


def _no_cues_warning(cues: tuple[Cue, ...], transcript: Transcript, log: Any) -> list[str]:
    """Say so when the run produced nothing, rather than handing back an empty file.

    A warning and an empty `.srt`, deliberately, rather than an error. Two reasons. The
    secondary user story is a 353-episode batch, and raising here would abort the whole
    run over one episode that happens to be music; and an empty subtitle file is a valid
    one, so writing it keeps the render contract and the stage cache consistent with every
    other run. What must not happen is that it goes unremarked: an empty `.srt` looks
    identical whether the clip held no speech or the pipeline broke, and the difference is
    the entire question the user has. So it lands in the log, in `RunResult.warnings`, and
    in `--json`.
    """
    if cues:
        return []
    dropped = len(transcript.params.get("speechless_dropped") or [])
    why = (
        "every segment the recogniser produced was rejected as speech-free"
        if dropped
        else "the recogniser found no speech in it"
    )
    note = (
        f"no subtitles were produced: {why}. The .srt and .vtt were written and are empty. "
        "If you expected words here, check that the audio actually contains speech, and "
        "that --start/--end select the part that does"
    )
    log(f"warning: {note}")
    return [note]


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


def _edit(
    cfg: RunConfig,
    cache: cache_mod.StageCache,
    cues: tuple[Cue, ...],
    cues_key: str,
    log: Any,
) -> tuple[tuple[Cue, ...], str, dict[str, Any] | None]:
    """Apply the hand corrections the GUI's editor saved, as their own cached stage.

    The stage's input is `edits.json`, which no stage writes, and its key is the upstream
    key plus a digest of the corrections. `edits.py` has the full argument for why that is
    the only placement where a correction both survives a re-run and re-burns exactly once.

    A correction set recorded against a different transcript is reported and skipped, never
    re-pointed and never deleted: cue 41 of the old transcript is not cue 41 of the new one,
    and going back to the old model must make them line up again.

    A file that cannot be read at all is a different case and stops the run. Reading it as
    "no corrections" is what let a typo produce a run that reported success and burned the
    uncorrected words with nothing said; see `edits.load`.
    """
    if not cache.enabled:
        return cues, cues_key, None

    try:
        saved = edits_mod.load(cache.work)
    except edits_mod.EditFileError as exc:
        # On the same channel as the stale-key report below, because a GUI run shows the
        # user this log and not the traceback the CLI prints.
        log(f"edit: {exc}")
        raise
    if saved is None or not saved:
        return cues, cues_key, None

    if saved.base_key != cues_key:
        log(
            f"edit: {len(saved.texts)} hand correction(s) in {edits_mod.EDITS_NAME} were made "
            "against a different transcript and are NOT being applied. Open the editor again "
            "to redo them, or go back to the settings they were made under."
        )
        return (
            cues,
            cues_key,
            {"applied": [], "stale": len(saved.texts), "base_key": saved.base_key},
        )

    artifact = cache.work / edits_mod.ARTIFACT_NAME
    entry = cache.begin(
        "edit",
        input_hash=cues_key,
        # The digest, not the corrections themselves: a meta file carrying a paragraph of
        # Serbian per corrected cue stops being readable by hand.
        params={"edits": saved.digest()},
        artifacts=(artifact,),
    )
    if entry.hit:
        payload = read_json(artifact)
        edited = cues_from_dict(payload)
        applied = payload.get("applied", [])
        log(f"edit: cached ({len(applied)} hand correction(s))")
        return edited, entry.key, {"applied": applied, "stale": 0}

    edited, applied = edits_mod.apply_edits(cues, saved, cfg.cues)
    write_json(artifact, {**cues_to_dict(edited), "applied": applied})
    cache.commit(entry)
    log(f"edit: {len(applied)} hand correction(s) applied")
    return edited, entry.key, {"applied": applied, "stale": 0}


def _mux(
    cfg: RunConfig,
    cache: cache_mod.StageCache,
    *,
    src: Path,
    media_id: str,
    info: media.MediaInfo,
    burned: Path | None,
    burn_key: str,
    subs: Path,
    cues_key: str,
    out_dir: Path,
    stem: str,
    log: Any,
) -> Path | None:
    """`--soft-mux`: the same subtitles as a track the viewer can switch off.

    Which video the track goes onto is the whole decision. A source that has a picture gets
    it, because clean pixels plus a switchable track is the entire point of a soft track,
    and a run with the burn on therefore hands back two files that differ in exactly that.
    An audio-only input has no picture until the burn generates one, so there the burned
    canvas is the only candidate and the track rides along beside the rendered text.
    `--srt-only` asked for no video work at all and gets none.

    A stream copy, so it costs a second even on a long file; cached anyway, because the
    stage chain is what makes `--force cues` re-mux and `--style-preset` not.
    """
    if cfg.srt_only:
        log("soft-mux: skipped, --srt-only does no video work")
        return None

    if info.has_video:
        source, source_id, origin = src, media_id, "the source video"
    elif burned is not None:
        source, source_id, origin = burned, burn_key, "the burned canvas"
    else:
        log("soft-mux: skipped, this input has no video and the burn is turned off")
        return None

    # Matroska is the only one of the three that carries styled subtitles, and a WebM's
    # VP9/Opus streams do not belong in an MP4 either, so a webm source becomes an mkv.
    suffix = ".mkv" if source.suffix.lower() in {".mkv", ".webm"} else ".mp4"
    dst = out_dir / f"{stem}.softsubs{suffix}"

    entry = cache.begin(
        "mux",
        input_hash=cues_key,
        params={"source": source_id, "origin": origin, "container": suffix},
        artifacts=(dst,),
    )
    if entry.hit:
        log(f"soft-mux: cached ({dst.name})")
        return dst

    burn_mod.soft_mux(source, subs, dst, dry_run=cfg.dry_run)
    cache.commit(entry)
    log(f"muxed: {dst.name} (a switchable track on {origin})")
    return dst


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
) -> tuple[Path, str]:
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
        return dst, entry.key

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
    return dst, entry.key
