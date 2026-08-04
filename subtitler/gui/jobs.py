"""Background work and its log.

A 45-minute file is minutes of transcription even on a GPU, so nothing the user starts may
run inside the request that started it. Jobs run on a worker while the page polls, and the
pipeline's `log` callback is the progress display: every line it emits is appended here and
handed to the browser on the next poll.

The thread is behind `spawn`, the same way `doctor.Probe` is behind an argument: tests pass
`spawn=lambda work: work()` and the whole job machinery runs inline and deterministically,
with no sleeping, no polling and no thread to leak into the next test.
"""

from __future__ import annotations

import threading
import time
import traceback
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

Log = Callable[[str], None]
Work = Callable[[Log], dict[str, Any]]
Spawn = Callable[[Callable[[], None]], None]


class Busy(RuntimeError):
    """One job at a time. Two transcriptions at once on a laptop help nobody."""


# --------------------------------------------------------------------------------------
# Which stage a log line belongs to
# --------------------------------------------------------------------------------------
#
# `run_pipeline` does not report a stage before it starts one; it reports what it did once
# it is done, and a stage that runs cold (extract, cues) says nothing at all. So the stage
# shown in the UI is derived from the last line that names one, and the tick marks in front
# of it are inferred from the order below. That is honest, and it is the only thing possible
# without adding a progress protocol to the pipeline for the GUI's benefit.

STAGES: tuple[str, ...] = (
    "probe",
    "extract",
    "denoise",
    "transcribe",
    "cues",
    "fix",
    "render",
    "burn",
)

# First match wins, so the specific prefixes come before the ones they start with:
# "denoise" before "engine", and "transcrib" catches both "transcribe: cached" and
# "transcribed: 42 segments".
_STAGE_PREFIXES: tuple[tuple[str, str], ...] = (
    ("input:", "probe"),
    ("extract", "extract"),
    ("denoise", "denoise"),
    ("engine:", "transcribe"),
    ("batched decoding", "transcribe"),
    ("transcrib", "transcribe"),
    ("cues:", "cues"),
    ("fix:", "fix"),
    ("wrote:", "render"),
    ("lint:", "render"),
    ("burn", "burn"),
)


def stage_for_line(line: str) -> str | None:
    lowered = line.strip().lower()
    for prefix, stage in _STAGE_PREFIXES:
        if lowered.startswith(prefix):
            return stage
    return None


def stage_of(lines: list[str]) -> str | None:
    """The furthest stage any line has reached, not merely the last one recognised.

    Reading backwards would flap: `lint:` maps to render and is emitted after `wrote:`,
    but a burn line follows both, and a warning printed by a later stage must never move
    the display backwards.
    """
    best: int | None = None
    for line in lines:
        stage = stage_for_line(line)
        if stage is None:
            continue
        index = STAGES.index(stage)
        if best is None or index > best:
            best = index
    return None if best is None else STAGES[best]


# --------------------------------------------------------------------------------------
# Jobs
# --------------------------------------------------------------------------------------

RUNNING, DONE, ERROR = "running", "done", "error"


@dataclass
class Job:
    id: str
    kind: str  # "run" | "download"
    label: str
    started: float
    # Which stages this particular run will go through. A chip for `burn` that never
    # lights up on a `--srt-only` run reads as a step that failed, so the caller narrows
    # the list to what the config will actually do.
    stages: tuple[str, ...] = STAGES
    status: str = RUNNING
    lines: list[str] = field(default_factory=list)
    result: dict[str, Any] | None = None
    error: str | None = None
    detail: str = ""
    finished: float | None = None
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def append(self, message: str) -> None:
        text = str(message).rstrip("\n")
        with self._lock:
            for line in text.splitlines() or [""]:
                self.lines.append(line)

    def snapshot(self, since: int = 0, *, now: float) -> dict[str, Any]:
        with self._lock:
            lines = self.lines[max(0, since) :]
            total = len(self.lines)
            status, result, error, detail = self.status, self.result, self.error, self.detail
            finished = self.finished
            log_stage = stage_of(self.lines) if self.kind == "run" else None
        return {
            "id": self.id,
            "kind": self.kind,
            "label": self.label,
            "status": status,
            "lines": lines,
            "next": total,
            "stage": log_stage,
            "stages": list(self.stages),
            "elapsed_s": round((finished if finished is not None else now) - self.started, 1),
            "result": result,
            "error": error,
            "detail": detail,
        }


class JobManager:
    """One job at a time, kept after it finishes so the page can still read the result."""

    def __init__(self, *, spawn: Spawn | None = None, clock: Callable[[], float] = time.monotonic):
        self._spawn = spawn or thread_spawn
        self._clock = clock
        self._jobs: dict[str, Job] = {}
        self._current: Job | None = None
        self._counter = 0
        self._lock = threading.Lock()

    def start(self, kind: str, label: str, work: Work, *, stages: tuple[str, ...] = STAGES) -> Job:
        with self._lock:
            if self._current is not None and self._current.status == RUNNING:
                raise Busy(f"{self._current.label} is still running")
            self._counter += 1
            job = Job(
                id=f"j{self._counter}",
                kind=kind,
                label=label,
                started=self._clock(),
                stages=stages,
            )
            self._jobs[job.id] = job
            self._current = job

        def run() -> None:
            try:
                result = work(job.append)
            except BaseException as exc:  # a worker must never die silently
                job.error = f"{type(exc).__name__}: {exc}"
                job.detail = traceback.format_exc()
                job.status = ERROR
                job.append(f"error: {job.error}")
            else:
                job.result = result
                job.status = DONE
            finally:
                job.finished = self._clock()

        self._spawn(run)
        return job

    @property
    def current(self) -> Job | None:
        return self._current

    def get(self, job_id: str) -> Job | None:
        return self._jobs.get(job_id)

    def snapshot(self, job_id: str | None = None, since: int = 0) -> dict[str, Any] | None:
        job = self.get(job_id) if job_id else self._current
        return None if job is None else job.snapshot(since, now=self._clock())


def thread_spawn(work: Callable[[], None]) -> None:
    threading.Thread(target=work, daemon=True, name="subtitler-job").start()
