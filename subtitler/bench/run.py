"""The benchmark matrix runner: denoiser x engine x clip, plus the optional `--fix` axis.

What this file is for is comparison, so everything in it exists to make two cells
comparable to each other and to a run from last month.

**Every cell runs in its own process.** Peak RSS is one of the numbers being collected, and
`getrusage` reports a high-water mark that never comes back down, so a second cell measured
in the same interpreter inherits the first cell's peak and reports it as its own. A fresh
process per cell is the only way that number means anything. It costs a model load per cell,
which is honest: the wall clock then includes it, and the RTF from the transcript does not.

**The stage cache is shared, on purpose.** Cells run clip-outermost, then denoiser, then
engine, so `extract` runs once per clip and `denoise` once per (clip, preset) no matter how
many engines follow. That ordering is the whole reason `denoise` is a separate stage from
`extract`; see the header of `cache.py`. The work directory is deliberately outside the
timestamped result directory so a second run reuses it.

**A cell that fails is a result.** `groq-turbo` currently answers `organization_restricted`
on this account. That is recorded in `results.json` as a failed cell with its error, because
"we could not measure it" is the finding, and a matrix that aborts on the first cloud error
would report nothing at all.

Ground truth is Phase 8's job. This runner reads `benchmarks/references/<clip>.txt` if it is
there and scores against it, and if it is not there it says so and emits the reference-free
metrics only. It never invents one. `bench report` recomputes every metric from the kept
transcripts, so when Phase 8 lands a reference, the WER for a run from today can be filled
in without re-transcribing anything.
"""

from __future__ import annotations

import json
import multiprocessing
import os
import platform as platform_mod
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from subtitler import media
from subtitler.bench import metrics as metrics_mod
from subtitler.bench import report as report_mod
from subtitler.cues import CueConfig
from subtitler.model import Cue

__all__ = [
    "DEFAULT_CLIPS",
    "BenchConfig",
    "CellSpec",
    "expand",
    "load_reference",
    "reference_meta",
    "rescore",
    "resolve_clips",
    "run_matrix",
]

# The two checked-in fixtures, used when no clip directory has been populated. Both are
# archive audio; PRD open question 3 notes that neither represents the primary user story
# (one speaker on camera in a noisy room), and that clip does not exist yet.
DEFAULT_CLIPS: tuple[str, ...] = ("fixtures/gozba-sample.mp3", "fixtures/uvod-u-pravo.m4a")

MEDIA_SUFFIXES = frozenset(
    {".mp3", ".m4a", ".wav", ".flac", ".ogg", ".opus", ".mp4", ".mov", ".mkv", ".webm"}
)

# Local engines only. Cloud costs money and leaves the machine, so it joins the matrix when
# it is asked for by name and never by default.
LOCAL_ENGINES: tuple[str, ...] = ("mlx", "faster-whisper")

# A cell that has not finished in an hour is hung, not slow: the longest clip here is under
# three minutes and the slowest supported path is CPU int8.
CELL_TIMEOUT_S = 3600.0

# Recorded in env.json. Anything whose version can change a transcript or a score.
TRACKED_PACKAGES: tuple[str, ...] = (
    "faster-whisper",
    "ctranslate2",
    "mlx-whisper",
    "mlx",
    "groq",
    "jiwer",
    "litellm",
    "huggingface-hub",
    "numpy",
    "torch",
)


@dataclass(frozen=True, slots=True)
class CellSpec:
    """One cell of the matrix. Everything that can change the output is a field here."""

    clip: Path
    denoise: str = "none"
    engine: str = "faster-whisper"
    model: str = "large-v3"
    device: str = "auto"
    batch_size: int = 0
    language: str = "sr"
    fix: bool = False
    fix_model: str = ""
    force: str | None = None

    @property
    def clip_id(self) -> str:
        return self.clip.stem

    @property
    def cell_id(self) -> str:
        """Stable, filesystem-safe, and readable in that order.

        It names the transcript file and the row in the report, so it has to survive being
        a filename on a case-insensitive volume and still be greppable by hand.
        """
        parts = [self.clip_id, self.denoise, self.engine.replace("/", "-"), self.model]
        if self.batch_size:
            parts.append(f"b{self.batch_size}")
        parts.append("fix" if self.fix else "nofix")
        return "__".join(parts)


@dataclass(frozen=True, slots=True)
class BenchConfig:
    clips: tuple[Path, ...]
    denoisers: tuple[str, ...] = tuple(media.DENOISE_FILTERS)
    engines: tuple[str, ...] = ("faster-whisper",)
    model: str = "large-v3"
    device: str = "auto"
    batch_size: int = 0
    language: str = "sr"
    fix_axis: bool = False
    fix_model: str = ""
    out_root: Path = Path("benchmarks/results")
    references: Path = Path("benchmarks/references")
    work: Path = Path("benchmarks/.work")
    cues: CueConfig = field(default_factory=CueConfig)
    allow_dirty: bool = False
    force: str | None = None


def expand(cfg: BenchConfig) -> list[CellSpec]:
    """The matrix, in the order it must run in.

    Clip outermost, then denoiser, then engine: that is what lets `extract` be cached per
    clip and `denoise` per (clip, preset) across every engine that follows. The cache keeps
    one slot per stage, so any other nesting re-runs the denoiser for every engine.
    """
    cells: list[CellSpec] = []
    for clip in cfg.clips:
        for denoise in cfg.denoisers:
            for engine in cfg.engines:
                base = CellSpec(
                    clip=clip,
                    denoise=denoise,
                    engine=engine,
                    model=cfg.model,
                    device=cfg.device,
                    batch_size=cfg.batch_size,
                    language=cfg.language,
                    fix_model=cfg.fix_model,
                    force=cfg.force,
                )
                cells.append(base)
                if cfg.fix_axis:
                    # The corrected cell chains off the same cached transcript, so the
                    # `--fix` axis costs LLM calls and nothing else.
                    cells.append(replace(base, fix=True))
    return cells


# --------------------------------------------------------------------------------------
# Inputs
# --------------------------------------------------------------------------------------


def resolve_clips(spec: str | None, *, root: Path) -> tuple[Path, ...]:
    """A directory, a comma-separated list of files, or the default fixtures.

    `benchmarks/clips/` is empty in a fresh checkout because the interesting clips are too
    large to commit, so an unpopulated directory falls back to the two checked-in fixtures
    rather than reporting an empty matrix.

    A clip inside the repository is named relative to it. `results.json` is a committed
    artifact, and `fixtures/gozba-sample.mp3` means the same thing in someone else's
    checkout while `/home/someone/Projects/.../fixtures/gozba-sample.mp3` does not.
    """
    if spec:
        candidates = [Path(p.strip()) for p in spec.split(",") if p.strip()]
        if len(candidates) == 1 and candidates[0].is_dir():
            found = _media_in(candidates[0])
            if not found:
                raise ValueError(f"no media files in {candidates[0]}")
            return _relative_to(found, root)
        missing = [str(p) for p in candidates if not p.exists()]
        if missing:
            raise ValueError(f"clip not found: {', '.join(missing)}")
        return _relative_to(tuple(candidates), root)

    default_dir = root / "benchmarks" / "clips"
    found = _media_in(default_dir) if default_dir.is_dir() else ()
    if found:
        return _relative_to(found, root)
    return _relative_to(tuple(root / p for p in DEFAULT_CLIPS), root)


def _media_in(directory: Path) -> tuple[Path, ...]:
    return tuple(
        sorted(p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in MEDIA_SUFFIXES)
    )


def _relative_to(paths: Sequence[Path], root: Path) -> tuple[Path, ...]:
    out = []
    for path in paths:
        try:
            out.append(path.resolve().relative_to(root.resolve()))
        except ValueError:  # outside the repository: an absolute path is the only honest one
            out.append(path)
    return tuple(out)


def load_reference(clip_id: str, references: Path) -> str | None:
    """The adjudicated transcript for a clip, or None. Phase 8 is what fills this in."""
    path = references / f"{clip_id}.txt"
    if not path.exists():
        return None
    text = path.read_text(encoding="utf-8").strip()
    return text or None


def reference_meta(clip_id: str, references: Path) -> dict[str, Any]:
    """Provenance for a reference, written next to it whether or not one exists.

    `human_verified` is false in every case this phase can produce, and the report keys off
    it: a WER against an unverified reference measures agreement between models, not
    correctness, and it is labelled provisional everywhere it appears. When Phase 8 lands an
    adjudicated transcript it owns this file and sets the field honestly.
    """
    path = references / f"{clip_id}.txt"
    existing: dict[str, Any] = {}
    meta_path = references / f"{clip_id}.meta.json"
    if meta_path.exists():
        try:
            existing = json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            existing = {}

    if not path.exists():
        return {
            "clip": clip_id,
            "reference": None,
            "status": "absent",
            "human_verified": False,
            "adjudicated": False,
            "source": existing.get("source", ""),
            "note": (
                "No reference transcript. Reference-free metrics only; WER is not reported "
                "for this clip. Phase 8 (LLM adjudication) produces this file."
            ),
        }

    text = path.read_text(encoding="utf-8")
    return {
        "clip": clip_id,
        "reference": path.name,
        "status": "present",
        # Never inferred: the harness cannot know whether a human read it, so it only ever
        # carries forward what the file already claimed.
        "human_verified": bool(existing.get("human_verified", False)),
        "adjudicated": bool(existing.get("adjudicated", False)),
        "source": existing.get("source", "unknown; recorded when the reference was added"),
        "words": len(text.split()),
        "characters": len(text),
        "note": existing.get("note", ""),
    }


# --------------------------------------------------------------------------------------
# Environment and provenance
# --------------------------------------------------------------------------------------


def git_state(repo: Path) -> dict[str, Any]:
    """Commit, branch and dirtiness. A benchmark without a SHA is not reproducible."""

    def _git(*args: str) -> str | None:
        try:
            proc = subprocess.run(
                ["git", "-C", str(repo), *args],
                capture_output=True,
                text=True,
                check=False,
            )
        except (OSError, subprocess.SubprocessError):
            return None
        return proc.stdout.strip() if proc.returncode == 0 else None

    sha = _git("rev-parse", "HEAD")
    if sha is None:
        return {"sha": None, "branch": None, "dirty": None, "note": "not a git checkout"}
    status = _git("status", "--porcelain")
    return {
        "sha": sha,
        "branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": bool(status),
        "dirty_paths": status.splitlines() if status else [],
    }


def package_versions(names: Sequence[str] = TRACKED_PACKAGES) -> dict[str, str | None]:
    from importlib.metadata import PackageNotFoundError, version

    out: dict[str, str | None] = {}
    for name in names:
        try:
            out[name] = version(name)
        except PackageNotFoundError:
            out[name] = None
    return out


def _total_ram_bytes() -> int | None:
    try:
        return int(os.sysconf("SC_PAGE_SIZE")) * int(os.sysconf("SC_PHYS_PAGES"))
    except (ValueError, OSError, AttributeError):
        return None


def _cpu_model() -> str:
    """A useful CPU name on Linux, and whatever the platform module knows elsewhere."""
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        try:
            for line in cpuinfo.read_text(encoding="utf-8", errors="replace").splitlines():
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
        except OSError:
            pass
    return platform_mod.processor() or platform_mod.machine()


def collect_env() -> dict[str, Any]:
    """`doctor --json` plus the host facts a number is only comparable within.

    `diagnose` already answers the GPU and CUDA questions out of process, so this does not
    dlopen several hundred megabytes of CUDA into the benchmark's own interpreter.
    """
    from subtitler.doctor import detect_platform, diagnose

    plat = detect_platform()
    statuses = diagnose(plat)
    return {
        "doctor": {
            "platform": {
                "system": plat.system,
                "machine": plat.machine,
                "distro_id": plat.distro_id,
                "distro_like": plat.distro_like,
                "rosetta": plat.rosetta,
                "package_manager": plat.package_manager,
            },
            "deps": [s.to_dict() for s in statuses],
        },
        "host": {
            "os": platform_mod.platform(),
            "python": sys.version.split()[0],
            "cpu": _cpu_model(),
            "cpu_count": os.cpu_count(),
            "ram_bytes": _total_ram_bytes(),
        },
        "packages": package_versions(),
    }


# --------------------------------------------------------------------------------------
# One cell
# --------------------------------------------------------------------------------------


def _peak_rss_mb() -> float:
    """This process's high-water RSS. Linux reports kilobytes, macOS bytes.

    Platform branch via `doctor.detect_platform`, per non-negotiable 5: nothing outside
    `doctor.py` and `engines/mlx.py` calls `platform.system()` itself, so the macOS side of
    this stays exercisable from a Linux test.
    """
    import resource

    from subtitler.doctor import detect_platform

    raw = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return raw / (1024 * 1024) if detect_platform().is_macos else raw / 1024


def execute_cell(spec: CellSpec, *, work: Path) -> dict[str, Any]:
    """Run one pipeline and return everything the metrics need, as plain JSON.

    Runs with `srt_only`, so no video is encoded: the matrix measures transcription and cue
    layout, and burning 40 canvases would dominate the wall clock while measuring ffmpeg.

    Never raises. An engine that cannot run here (`organization_restricted`, missing
    weights, no CUDA) returns `ok: False` with the reason, so one dead cell does not take
    the matrix with it.
    """
    from subtitler.model import Transcript
    from subtitler.pipeline import RunConfig, run_pipeline

    fix_cfg = None
    if spec.fix:
        from subtitler import postedit

        fix_cfg = postedit.FixConfig(model=spec.fix_model or postedit.DEFAULT_MODEL)

    cfg = RunConfig(
        input=spec.clip,
        out_dir=work,
        engine=spec.engine,
        model=spec.model,
        device=spec.device,
        batch_size=spec.batch_size,
        language=spec.language,
        denoise=spec.denoise,
        burn=False,
        srt_only=True,
        fix=fix_cfg,
        force=spec.force,
    )

    started = time.monotonic()
    try:
        result = run_pipeline(cfg, log=lambda _message: None)
    except Exception as exc:  # every engine failure mode, including EngineUnavailable
        return {
            "ok": False,
            "error": f"{type(exc).__name__}: {exc}",
            "wall_s": time.monotonic() - started,
            "peak_rss_mb": _peak_rss_mb(),
        }
    wall = time.monotonic() - started

    transcript: Transcript | None = result.transcript
    return {
        "ok": True,
        "error": "",
        "wall_s": wall,
        "peak_rss_mb": _peak_rss_mb(),
        "engine": result.engine,
        "cached_stages": list(result.cached),
        "audio_s": transcript.duration if transcript else 0.0,
        "decode_s": transcript.runtime_s if transcript else 0.0,
        "rtf": transcript.rtf if transcript else 0.0,
        "language": transcript.language if transcript else "",
        "model": transcript.model if transcript else spec.model,
        "model_revision": transcript.model_revision if transcript else "",
        "engine_params": dict(transcript.params) if transcript else {},
        "segments": len(transcript.segments) if transcript else 0,
        "fix_report": result.fix,
        "text": " ".join(cue.text for cue in result.cues),
        "cues": [
            {"index": c.index, "start": c.start, "end": c.end, "lines": list(c.lines)}
            for c in result.cues
        ],
        "lint_violations": len(result.lint),
    }


def _cell_entrypoint(spec: CellSpec, work: str, out_path: str) -> None:
    """The child process. Writes JSON to a file rather than a Queue.

    A Queue would need the parent to drain it before joining, and a child that dies between
    `put` and the join deadlocks the parent on a full pipe. A file has neither failure mode:
    if it is not there afterwards, the child died, and the exit code says so.
    """
    payload = execute_cell(spec, work=Path(work))
    Path(out_path).write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def run_cell(spec: CellSpec, *, work: Path, timeout: float = CELL_TIMEOUT_S) -> dict[str, Any]:
    """Run one cell in a fresh process, so its peak RSS is its own.

    `spawn` rather than `fork`: a forked child inherits the parent's already-allocated heap
    and its RSS starts at whatever the parent had reached, which is precisely the
    measurement error this exists to avoid. It is also the only start method macOS supports
    safely.
    """
    ctx = multiprocessing.get_context("spawn")
    with tempfile.TemporaryDirectory(prefix="subtitler-bench-") as tmp:
        out_path = Path(tmp) / "cell.json"
        proc = ctx.Process(target=_cell_entrypoint, args=(spec, str(work), str(out_path)))
        proc.start()
        proc.join(timeout)
        if proc.is_alive():
            proc.kill()
            proc.join()
            return {"ok": False, "error": f"timed out after {timeout:.0f}s"}
        if not out_path.exists():
            return {
                "ok": False,
                "error": f"the worker process exited with code {proc.exitcode} and wrote nothing",
            }
        return json.loads(out_path.read_text(encoding="utf-8"))


# --------------------------------------------------------------------------------------
# Scoring a finished cell
# --------------------------------------------------------------------------------------


def _cues_from_payload(payload: dict[str, Any]) -> tuple[Cue, ...]:
    return tuple(
        Cue(index=c["index"], start=c["start"], end=c["end"], lines=tuple(c["lines"]))
        for c in payload.get("cues", [])
    )


def score_cell(
    payload: dict[str, Any],
    *,
    reference: str | None,
    cues: CueConfig,
) -> dict[str, Any]:
    """Attach every metric to a finished cell. Pure: no I/O, no model, no clock."""
    if not payload.get("ok"):
        return payload

    text = payload.get("text", "")
    params = payload.get("engine_params", {}) or {}
    scored = dict(payload)
    scored["cue_stats"] = metrics_mod.cue_stats(_cues_from_payload(payload), cues).to_dict()
    scored["hallucination"] = metrics_mod.hallucination(
        text,
        repetition_collapsed=params.get("repetition_collapsed"),
        silence_dropped=params.get("silence_dropped"),
    ).to_dict()

    if reference and text:
        try:
            scored["reference_score"] = metrics_mod.score(reference, text).to_dict()
        except (ImportError, ValueError) as exc:
            scored["reference_score"] = None
            scored["reference_error"] = str(exc)
    else:
        scored["reference_score"] = None
    return scored


def attach_fix_delta(records: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """For every corrected cell, how much text the correction pass actually changed.

    PRD open question 4 asks whether `--fix` improves WER or hurts it, and that question
    needs a reference to answer. This is the half that does not: the word-level distance
    between the corrected cell and the identical uncorrected one. It says how much the model
    rewrote, never whether the rewrite was right, and the report labels it that way.
    """
    by_key = {
        (r.get("cell_id", "").replace("__fix", "__nofix")): r
        for r in records
        if r.get("ok") and not r.get("fix")
    }
    out = []
    for record in records:
        row = dict(record)
        if row.get("ok") and row.get("fix"):
            baseline = by_key.get(row.get("cell_id", "").replace("__fix", "__nofix"))
            if baseline and baseline.get("text") and row.get("text"):
                try:
                    row["fix_change_rate"] = metrics_mod.score(baseline["text"], row["text"]).wer
                except (ImportError, ValueError):
                    row["fix_change_rate"] = None
        out.append(row)
    return out


# --------------------------------------------------------------------------------------
# The matrix
# --------------------------------------------------------------------------------------


def run_dir_name(now: datetime | None = None) -> str:
    """UTC, ISO 8601, with the colons that a filesystem should not have to carry."""
    stamp = (now or datetime.now(UTC)).strftime("%Y-%m-%dT%H-%M-%SZ")
    return stamp


def run_matrix(
    cfg: BenchConfig,
    *,
    repo: Path,
    log: Callable[[str], None] = print,
    runner: Callable[[CellSpec], dict[str, Any]] | None = None,
    env_collector: Callable[[], dict[str, Any]] = collect_env,
    now: datetime | None = None,
) -> Path:
    """Run every cell, write the run directory, and return its path.

    Refuses to start on a dirty tree without `--allow-dirty`: a result whose SHA does not
    describe the code that produced it is worse than no result, because it looks
    reproducible.
    """
    git = git_state(repo)
    if git.get("dirty") and not cfg.allow_dirty:
        raise ValueError(
            "the working tree is dirty, so this run could not be reproduced from its SHA.\n"
            "  commit first, or pass --allow-dirty to record the run as unreproducible.\n"
            "  changed: " + ", ".join(git.get("dirty_paths", [])[:10])
        )

    cells = expand(cfg)
    run_dir = cfg.out_root / run_dir_name(now)
    transcripts = run_dir / "transcripts"
    transcripts.mkdir(parents=True, exist_ok=True)
    cfg.work.mkdir(parents=True, exist_ok=True)
    cfg.references.mkdir(parents=True, exist_ok=True)

    execute = runner or (lambda spec: run_cell(spec, work=cfg.work))

    references: dict[str, str | None] = {}
    meta: dict[str, dict[str, Any]] = {}
    for clip in cfg.clips:
        clip_id = clip.stem
        references[clip_id] = load_reference(clip_id, cfg.references)
        meta[clip_id] = reference_meta(clip_id, cfg.references)
        (cfg.references / f"{clip_id}.meta.json").write_text(
            json.dumps(meta[clip_id], ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        if references[clip_id] is None:
            log(f"no reference for {clip_id}: reference-free metrics only, no WER")

    records: list[dict[str, Any]] = []
    for i, spec in enumerate(cells, start=1):
        log(f"[{i}/{len(cells)}] {spec.cell_id}")
        payload = execute(spec)
        record = {
            "cell_id": spec.cell_id,
            "clip": str(spec.clip),
            "clip_id": spec.clip_id,
            "denoise": spec.denoise,
            "engine_requested": spec.engine,
            "model_requested": spec.model,
            "device_requested": spec.device,
            "batch_size": spec.batch_size,
            "fix": spec.fix,
            "fix_model": spec.fix_model if spec.fix else "",
            **score_cell(payload, reference=references.get(spec.clip_id), cues=cfg.cues),
        }
        records.append(record)
        _write_transcript(transcripts, record)
        if record.get("ok"):
            log(
                f"    {record['segments']} segments, rtf {record.get('rtf', 0):.3f}, "
                f"{record['wall_s']:.1f}s wall, {record['peak_rss_mb']:.0f} MB peak"
            )
        else:
            log(f"    FAILED: {record.get('error', '')}")

    records = attach_fix_delta(records)
    payload = {
        "schema_version": 1,
        "created_utc": (now or datetime.now(UTC)).isoformat(),
        "git": git,
        "allow_dirty": cfg.allow_dirty,
        "config": {
            "clips": [str(c) for c in cfg.clips],
            "denoisers": list(cfg.denoisers),
            "engines": list(cfg.engines),
            "model": cfg.model,
            "device": cfg.device,
            "batch_size": cfg.batch_size,
            "language": cfg.language,
            "fix_axis": cfg.fix_axis,
            "fix_model": cfg.fix_model,
            "cues": {
                "max_line": cfg.cues.max_line,
                "max_lines": cfg.cues.max_lines,
                "min_dur": cfg.cues.min_dur,
                "max_dur": cfg.cues.max_dur,
                "max_cps": cfg.cues.max_cps,
            },
        },
        "references": meta,
        "results": records,
    }
    _write_run(run_dir, payload, env_collector())
    log(f"wrote {run_dir}")
    return run_dir


def _write_transcript(transcripts: Path, record: dict[str, Any]) -> None:
    """Keep the hypothesis, so a metric can be recomputed without re-running the model.

    Both forms: the flat text WER is computed from, and the SRT the cue statistics are
    computed from. Together they are enough for `bench report` to rebuild every number in
    `results.json` from scratch.
    """
    if not record.get("ok"):
        return
    from subtitler.render import write_srt

    (transcripts / f"{record['cell_id']}.txt").write_text(
        record.get("text", "") + "\n", encoding="utf-8"
    )
    cues = _cues_from_payload(record)
    if cues:
        write_srt(transcripts / f"{record['cell_id']}.srt", cues)


def _write_run(run_dir: Path, payload: dict[str, Any], env: dict[str, Any]) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "results.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (run_dir / "env.json").write_text(
        json.dumps(env, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (run_dir / "report.md").write_text(report_mod.render(payload), encoding="utf-8")


def rescore(run_dir: Path, *, references: Path, log: Callable[[str], None] = print) -> Path:
    """Recompute every metric in a finished run from its kept transcripts.

    This is the Phase 8 seam. An adjudicated reference landing next month does not require
    re-transcribing anything: point `bench report` at a run from today and the WER columns
    fill in from `transcripts/`, against the reference that exists now.
    """
    results_path = run_dir / "results.json"
    if not results_path.exists():
        raise ValueError(f"{results_path} does not exist; is that a benchmark run directory?")
    payload = json.loads(results_path.read_text(encoding="utf-8"))
    cue_cfg = CueConfig(**payload.get("config", {}).get("cues", {}))

    clip_ids = {r.get("clip_id", "") for r in payload.get("results", [])}
    refs = {cid: load_reference(cid, references) for cid in clip_ids if cid}
    payload["references"] = {cid: reference_meta(cid, references) for cid in sorted(refs)}
    for clip_id, text in sorted(refs.items()):
        if text is None:
            log(f"no reference for {clip_id}: reference-free metrics only, no WER")

    rescored = []
    for record in payload.get("results", []):
        restored = _restore_from_disk(run_dir, record)
        rescored.append(
            {
                **restored,
                **score_cell(restored, reference=refs.get(record.get("clip_id")), cues=cue_cfg),
            }
        )
    payload["results"] = attach_fix_delta(rescored)
    payload["rescored_utc"] = datetime.now(UTC).isoformat()

    results_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (run_dir / "report.md").write_text(report_mod.render(payload), encoding="utf-8")
    log(f"rewrote {run_dir / 'report.md'}")
    return run_dir / "report.md"


def _restore_from_disk(run_dir: Path, record: dict[str, Any]) -> dict[str, Any]:
    """Prefer the kept transcript over the copy inside results.json.

    They are the same today. They stop being the same the moment someone hand-corrects a
    transcript to see what it would score, which is a thing worth being able to do.
    """
    if not record.get("ok"):
        return record
    restored = dict(record)
    text_path = run_dir / "transcripts" / f"{record['cell_id']}.txt"
    if text_path.exists():
        restored["text"] = text_path.read_text(encoding="utf-8").strip()
    srt_path = run_dir / "transcripts" / f"{record['cell_id']}.srt"
    if srt_path.exists():
        from subtitler.render import read_subtitles

        restored["cues"] = [
            {"index": c.index, "start": c.start, "end": c.end, "lines": list(c.lines)}
            for c in read_subtitles(srt_path)
        ]
    return restored


def latest_run(out_root: Path) -> Path | None:
    """The newest run directory. The timestamp format sorts lexicographically on purpose."""
    if not out_root.is_dir():
        return None
    runs = sorted(p for p in out_root.iterdir() if p.is_dir() and (p / "results.json").exists())
    return runs[-1] if runs else None


def available_local_engines(model: str, device: str) -> tuple[str, ...]:
    """The local engines that could actually run here, in preference order."""
    from subtitler.engines import available_engines

    probed = available_engines(model=model, device=device)
    return tuple(name for name in LOCAL_ENGINES if probed.get(name) and probed[name].ok)


def parse_axis(spec: str | None, *, valid: Iterable[str], label: str) -> tuple[str, ...] | None:
    """A comma-separated axis restriction, validated against what exists."""
    if not spec:
        return None
    wanted = tuple(part.strip() for part in spec.split(",") if part.strip())
    known = tuple(valid)
    unknown = [w for w in wanted if w not in known]
    if unknown:
        raise ValueError(f"unknown {label}: {', '.join(unknown)}; choose from {', '.join(known)}")
    return wanted
