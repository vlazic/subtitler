"""What a front end knows, with no widget and no socket in it.

There are two views over this project now: the browser page in `app.py` and the native Tk
window in `window.py`. Neither is allowed to know anything the other does not, so
everything between "the user has filled the form in" and "the pipeline has produced cues"
lives here, above `forms` (payload to `RunConfig`), `edits` (corrections and their quality
report) and `jobs` (a worker and its log).

The split matters most for the Tk side. A window is the hardest thing in this repo to test
without a display, and both CI runners are headless, so the rule is that `window.py`
contains layout and event bindings and nothing else: every decision it makes is a call
into this module, which `tests/test_desktop.py` drives with no display at all.

`Session` is deliberately a poller rather than a callback sink. Tk owns its main loop and a
widget may only be touched from the thread that created it, so the worker writes into a
`Job` behind its lock and the window reads whole snapshots from `poll()` on an `after()`
timer. That is the same shape the browser page gets from `/api/job`, minus HTTP.
"""

from __future__ import annotations

import shutil
import subprocess
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from subtitler import doctor, media, models
from subtitler import edits as edits_mod
from subtitler.cues import CueConfig
from subtitler.gui import files, forms, jobs
from subtitler.gui.jobs import JobManager
from subtitler.model import Cue
from subtitler.pipeline import RunConfig, RunResult, output_dir, run_pipeline, work_dir

# The phases a window can be in. `REVIEW` is the one that earns the editor: the run stopped
# at the subtitle files on purpose and the burn has not happened yet.
IDLE, RUNNING, REVIEW, DONE, FAILED = "idle", "running", "review", "done", "failed"

TK_MISSING = (
    "This Python was built without tkinter, so the browser interface is opening instead. "
    "The Python `make setup` installs through uv has it, as does the python.org installer; "
    "for a Homebrew python@3.12 the fix is `brew install python-tk@3.12`, and for a Debian "
    "or Ubuntu system Python it is `sudo apt install python3-tk`."
)

NO_DISPLAY = "There is no display to open a window on, so the browser interface is opening instead."

# The same question `doctor` asks, asked by the same function, so the panel and the command
# can never disagree about whether this machine has a native window.
tk_available = doctor.tk_importable


def default_opener(argv: Sequence[str]) -> None:
    """Hand a path to the desktop's file manager and forget about it."""
    subprocess.Popen(list(argv), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


@dataclass(frozen=True, slots=True)
class Refusal:
    """Why something was not started, named so a window can point at the control.

    The same shape `forms.FormError` has, minus the exception: a refusal here is an
    ordinary outcome the user is expected to hit (an empty box, a busy run), not a fault.
    """

    field: str
    message: str

    def __str__(self) -> str:
        return f"{self.field}: {self.message}" if self.field else self.message


# --------------------------------------------------------------------------------------
# Corrections
# --------------------------------------------------------------------------------------


def stage_edits(cfg: RunConfig, base_key: str, texts: Mapping[int, str]) -> Path | None:
    """Write the hand corrections where the pipeline reads them, or remove a stale set.

    Shared by both front ends rather than written twice, because the *clearing* half is the
    part that is easy to forget and impossible to notice: a correction set left over from a
    previous pass would otherwise be applied to a run the user made no corrections in, and
    the only evidence would be words in the video nobody typed this time.
    """
    work = work_dir(cfg)
    if not texts:
        edits_mod.clear(work)
        return None
    edit_set = edits_mod.build(base_key, {str(index): text for index, text in texts.items()})
    return edits_mod.save(work, edit_set)


@dataclass
class EditorModel:
    """The cue list under correction: what the run produced, and what it now says.

    Every cue is kept as the `Cue` the run produced, not merely as its text, because
    `edits.relayout` returns a cue *unchanged* when the corrected text matches what it
    already says, and that is what preserves the line break the splitter chose from real
    word timings. A model that only remembered strings would re-wrap every cue the moment
    the user clicked on it, quietly replacing good breaks with synthesized ones.
    """

    limits: CueConfig
    base_key: str
    order: tuple[int, ...]
    origin: dict[int, Cue]
    rows: dict[int, dict[str, Any]]
    texts: dict[int, str]

    @classmethod
    def from_cues(cls, cues: Sequence[Cue], limits: CueConfig | None, base_key: str) -> EditorModel:
        cfg = limits or CueConfig()
        return cls(
            limits=cfg,
            base_key=base_key,
            order=tuple(cue.index for cue in cues),
            origin={cue.index: cue for cue in cues},
            rows={cue.index: edits_mod.cue_report(cue, cfg) for cue in cues},
            texts={cue.index: edits_mod.normalize(cue.text) for cue in cues},
        )

    def __len__(self) -> int:
        return len(self.order)

    def row(self, index: int) -> dict[str, Any]:
        return self.rows[index]

    def set_text(self, index: int, text: str) -> dict[str, Any]:
        """Re-wrap and re-lint one cue, and hand back the row the window should draw.

        The wrapping goes through `edits.relayout` and so through `cues.wrap_edited`, which
        is the only thing in this project allowed to choose a line break. A window that
        split the string itself would be a second set of rules about Serbian clitics kept
        in sync by hope.
        """
        cue = self.origin[index]
        wanted = edits_mod.normalize(text)
        self.texts[index] = wanted
        row = edits_mod.cue_report(edits_mod.relayout(cue, text, self.limits), self.limits)
        self.rows[index] = row
        return row

    def revert(self, index: int) -> dict[str, Any]:
        return self.set_text(index, self.origin[index].text)

    def is_edited(self, index: int) -> bool:
        return self.texts[index] != edits_mod.normalize(self.origin[index].text)

    def edits(self) -> dict[int, str]:
        """Only the cues whose text actually changed, which is what `edits.json` records.

        Sending every cue would work and would be wrong twice over: the digest that keys
        the `edit` stage would change whenever anything upstream re-wrapped, and a file
        meant to be readable by hand would carry the whole transcript.
        """
        return {index: self.texts[index] for index in self.order if self.is_edited(index)}

    def blank(self) -> list[int]:
        """Cues the user emptied. Refused before the burn, not after.

        `edits.build` raises on these too, but it raises after the corrections have been
        collected and the second run is about to start; catching them here lets the window
        select the offending cue instead of showing a message about one.
        """
        return [index for index in self.order if not self.texts[index]]

    def flagged(self) -> list[int]:
        return [index for index in self.order if self.rows[index]["problems"]]

    def summary(self) -> str:
        """The one line above the list: how much there is, and how much of it is wrong."""
        total = len(self.order)
        bad = len(self.flagged())
        edited = len(self.edits())
        parts = [f"{total} cue{'s' if total != 1 else ''}"]
        parts.append(f"{bad} flagged" if bad else "nothing flagged")
        if edited:
            parts.append(f"{edited} corrected")
        return ", ".join(parts)


# --------------------------------------------------------------------------------------
# Hearing a cue
# --------------------------------------------------------------------------------------


class Player:
    """Plays the span of audio one cue covers, one at a time.

    Reading speed is the one thing in the quality report a human cannot judge by eye, so
    the editor plays the cue rather than only counting its characters. The argv is built by
    `media.play_span_cmd` and not here: non-negotiable 1 says every ffmpeg invocation comes
    out of a function with a command-construction test, and ffplay is ffmpeg.

    Both the spawner and the lookup are injectable so the whole thing is exercised without
    ffplay, a sound card or a display.
    """

    def __init__(
        self,
        *,
        spawn: Callable[[Sequence[str]], Any] | None = None,
        which: Callable[[str], str | None] | None = None,
    ) -> None:
        self._spawn = spawn or (
            lambda argv: subprocess.Popen(
                list(argv), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )
        )
        self._which = which or shutil.which
        self._proc: Any = None

    @property
    def available(self) -> bool:
        return self._which(media.PLAYER) is not None

    def play(self, path: Path, start: float, end: float) -> str:
        """Start playback, returning "" or the sentence to show the user."""
        if not self.available:
            return f"{media.PLAYER} is not installed, so a cue cannot be played here"
        self.stop()
        try:
            self._proc = self._spawn(media.play_span_cmd(path, start=start, end=end))
        except OSError as exc:
            return f"could not play that cue ({exc})"
        return ""

    def stop(self) -> None:
        proc = self._proc
        self._proc = None
        if proc is None:
            return
        try:
            if proc.poll() is None:
                proc.terminate()
        except OSError:
            pass


# --------------------------------------------------------------------------------------
# The session
# --------------------------------------------------------------------------------------


class Session:
    """One window's worth of state: the run, its log, the editor, and the result."""

    def __init__(
        self,
        *,
        runner: Callable[..., RunResult] | None = None,
        spawn: jobs.Spawn | None = None,
        plat: doctor.Platform | None = None,
        opener: Callable[[Sequence[str]], None] | None = None,
        diagnose: Callable[[doctor.Platform], list[doctor.DepStatus]] | None = None,
        downloader: Callable[..., Path] | None = None,
    ) -> None:
        self.plat = plat or doctor.detect_platform()
        self._runner = runner or run_pipeline
        self._opener = opener or default_opener
        self._diagnose = diagnose or (lambda p: doctor.diagnose(p))
        self._download = downloader or models.download
        self._spawn = spawn or jobs.thread_spawn
        self.jobs = JobManager(spawn=self._spawn)
        self.phase = IDLE
        self.editor: EditorModel | None = None
        self.result: RunResult | None = None
        self.error = ""
        self.detail = ""
        self.command = ""
        self.out_dir: Path | None = None
        self.stages: tuple[str, ...] = ()
        self.label = ""
        self.doctor_report: dict[str, Any] | None = None
        self.doctor_running = False
        self._cfg: RunConfig | None = None
        self._values: dict[str, Any] = {}
        self._job: jobs.Job | None = None
        self._seen = 0
        # Whether the job being followed still owes the session an ending.
        self._pending = False
        # The weights being fetched, and the last measurement of how much of them landed.
        self._download_spec: models.ModelSpec | None = None
        self._usage: tuple[float, int] = (0.0, 0)
        # Which kind of job has just stopped and has not been drawn yet, popped by
        # `take_finished`. A window cannot ask "has the phase changed" instead: two runs in
        # a row both end in DONE, and the second one would repaint nothing.
        self._finished_kind: str | None = None

    @property
    def busy(self) -> bool:
        """Whether anything at all is in flight, of either kind.

        A model download counts as much as a run does: two transcriptions at once help
        nobody on a laptop, and starting one while several gigabytes of weights are still
        arriving is the same mistake wearing a different hat. The phase alone is not
        enough, because a download never enters `RUNNING` - that phase belongs to the
        pipeline - and the job manager is the thing that actually knows.
        """
        if self.phase == RUNNING:
            return True
        current = self.jobs.current
        return current is not None and current.status == jobs.RUNNING

    @property
    def media_path(self) -> Path | None:
        """The audio the cues were made from: the trimmed fragment, or the download.

        `RunResult.input` and not the box the user typed in, because a `--start/--end` run
        transcribed a fragment and every cue timestamp is relative to that fragment.
        """
        result = self.result
        if result is None:
            return None
        path = Path(result.input)
        return path if path.is_file() else None

    def start(self, values: Mapping[str, Any]) -> Refusal | None:
        """Validate the form and start the first pass. `None` means it is running."""
        try:
            cfg = forms.build_config(values)
        except forms.FormError as exc:
            return Refusal(exc.field, exc.message)
        self._values = dict(values)
        self.editor = None
        return self._launch(cfg)

    def approve(self) -> Refusal | None:
        """Save the corrections and run the second half: render, burn, and any soft mux."""
        if self.editor is None:
            return Refusal("", "there is nothing to approve yet")
        blank = self.editor.blank()
        if blank:
            return Refusal("cues", f"cue {blank[0]} is empty; a cue must say something")
        settings = dict(self._values)
        settings["review"] = False
        try:
            cfg = forms.build_config(settings)
        except forms.FormError as exc:
            return Refusal(exc.field, exc.message)
        try:
            stage_edits(cfg, self.editor.base_key, self.editor.edits())
        except (edits_mod.EditError, media.MediaError) as exc:
            return Refusal("", str(exc))
        return self._launch(cfg)

    def _launch(self, cfg: RunConfig) -> Refusal | None:
        command = forms.command_line(cfg)
        runner = self._runner

        def work(log: jobs.Log) -> dict[str, Any]:
            log(f"$ {command}")
            # The `RunResult` is handed back as itself rather than as `to_dict()`. The
            # browser app has no choice but to flatten it; a window shares the process with
            # the pipeline, and flattening here would throw away the very `Cue` objects the
            # editor is about to lay out.
            return {"run": runner(cfg, log=log)}

        try:
            job = self.jobs.start("run", forms.job_label(cfg), work, stages=forms.stages_for(cfg))
        except jobs.Busy as exc:
            return Refusal("", str(exc))

        self._cfg = cfg
        self._adopt(job)
        self.command = command
        self.label = forms.job_label(cfg)
        self.out_dir = output_dir(cfg)
        self.stages = forms.stages_for(cfg)
        self.phase = RUNNING
        self.error = ""
        self.detail = ""
        self.result = None
        return None

    def _adopt(self, job: jobs.Job) -> None:
        """Follow a newly started job: its log from the top, and its ending undrawn."""
        self._job = job
        self._seen = 0
        self._pending = True
        self._finished_kind = None
        # Cleared here rather than when a download ends, so that the finished download's
        # own bar can still read 100% while its result is on screen.
        self._download_spec = None

    def poll(self) -> dict[str, Any] | None:
        """The next slice of the running job, plus the phase it leaves the session in.

        Called from the Tk main loop's timer and from nowhere else. This is the single
        point where anything the worker produced crosses back to the thread that owns the
        widgets, which is what keeps the rule "no widget is touched off the main thread"
        checkable by reading one function.
        """
        if self._job is None:
            return None
        snapshot = self.jobs.snapshot(self._job.id, self._seen)
        if snapshot is None:
            return None
        self._seen = snapshot["next"]
        if snapshot["status"] != jobs.RUNNING and self._pending:
            self._pending = False
            self._finish(snapshot)
        snapshot["phase"] = self.phase
        return snapshot

    def take_finished(self) -> str | None:
        """The kind of job that just ended, once, or `None`.

        A handshake rather than a comparison of phases: two runs in a row both end in
        `DONE`, so a window that redrew on "the phase changed" would draw the first result
        and never the second.
        """
        kind, self._finished_kind = self._finished_kind, None
        return kind

    def _finish(self, snapshot: Mapping[str, Any]) -> None:
        kind = str(snapshot.get("kind") or "run")
        self._finished_kind = kind
        if snapshot["status"] == jobs.ERROR:
            self.error = str(snapshot.get("error") or f"the {kind} failed")
            self.detail = str(snapshot.get("detail") or "")
            # A failed *download* leaves the session where it was on purpose: a set of
            # corrections waiting for approval must survive somebody clicking Download on
            # the wrong model, and the message reaches them through `error` either way.
            if kind == "run":
                self.phase = FAILED
            return
        self.error = ""
        self.detail = ""
        if kind != "run":
            return
        result = (snapshot.get("result") or {}).get("run")
        self.result = result
        cfg = self._cfg
        if result is not None and cfg is not None and cfg.review:
            self.editor = EditorModel.from_cues(result.cues, cfg.cues, result.cues_key)
            self.phase = REVIEW
        else:
            self.phase = DONE

    # ------------------------------------------------------------------ the doctor panel

    def start_doctor(self, *, refresh: bool = False) -> bool:
        """Kick off the dependency check on a worker. `True` if one is now running.

        Off the job manager on purpose. `doctor` shells out a dozen times and takes several
        seconds, and it must be possible to read it *while* a transcription is going: a
        panel that refused to open because the machine was busy would be closed to the user
        at exactly the moment they went looking for it. It also must not block the first
        paint, which is what running it inline would do.
        """
        if refresh:
            self.doctor_report = None
        if self.doctor_report is not None or self.doctor_running:
            return self.doctor_running
        self.doctor_running = True
        self._spawn(self._run_doctor)
        return self.doctor_running

    def _run_doctor(self) -> None:
        try:
            statuses = self._diagnose(self.plat)
            self.doctor_report = {
                "platform": self.plat.describe(),
                "text": doctor.render(statuses, self.plat),
                "blocking": [s.dep.key for s in statuses if s.blocking],
            }
        except Exception as exc:  # a broken check must not wedge the panel shut
            self.doctor_report = {
                "platform": self.plat.describe(),
                "text": f"the dependency check itself failed: {exc}",
                "blocking": [],
            }
        finally:
            self.doctor_running = False

    # ------------------------------------------------------------------ the weights

    @property
    def backend(self) -> str:
        """Which engine's weights this machine would actually use."""
        return "mlx" if self.plat.is_apple_silicon else "faster-whisper"

    def model_rows(self) -> list[dict[str, Any]]:
        """The models for this machine, and whether each is already on disk.

        `EngineUnavailable` says "the large-v3 weights are not downloaded" and names a
        terminal command, which is a dead end for somebody who was given an icon. The
        window offers the download instead.
        """
        rows: list[dict[str, Any]] = []
        for spec in models.specs_for_backend(self.backend):
            path = models.local_path(spec)
            rows.append(
                {
                    "name": spec.name,
                    "backend": spec.backend,
                    "size": spec.size_label,
                    "cached": path is not None,
                    "path": str(path) if path else "",
                }
            )
        return rows

    def download_model(self, name: str) -> Refusal | None:
        """Fetch one set of weights, with `models.download`'s progress into the same log.

        Through the job manager, unlike the doctor check: this one is minutes of network
        and gigabytes of disk, and it is exactly the thing that must not happen twice at
        once or underneath a run.
        """
        try:
            spec = models.resolve(name, self.backend)
        except models.ModelNotFound as exc:
            return Refusal("model", str(exc))

        downloader = self._download

        def work(log: jobs.Log) -> dict[str, Any]:
            path = downloader(spec, progress=log)
            log(f"downloaded: {path}")
            return {"model": str(path)}

        try:
            job = self.jobs.start(
                "download", f"{spec.key} ({spec.size_label})", work, stages=("download",)
            )
        except jobs.Busy as exc:
            return Refusal("", str(exc))
        self._adopt(job)
        self._download_spec = spec
        self._usage = (0.0, 0)
        self.label = job.label
        self.stages = job.stages
        self.error = ""
        self.detail = ""
        return None

    def download_fraction(self) -> float | None:
        """How much of the weights are on disk, from 0 to 1, or `None` if not downloading.

        Measured rather than reported, because there is nothing to report: `snapshot_
        download` draws its own bars on a terminal nobody launched this from, and its
        `progress` callback in `models.py` emits three lines and no numbers. The size of
        the cache directory against `approx_bytes` is the only honest signal available, and
        on a three gigabyte download it is the difference between a bar and a frozen window.

        Recomputed at most once a second: the timer runs four times as often and this walks
        a directory tree.
        """
        spec = self._download_spec
        if spec is None:
            return None
        now = time.monotonic()
        when, size = self._usage
        if now - when >= 1.0:
            size = models.disk_usage(spec)
            self._usage = (now, size)
        return min(size / spec.approx_bytes, 1.0) if spec.approx_bytes else None

    # ------------------------------------------------------------------ results

    def outputs(self) -> list[tuple[str, Path]]:
        """The files this run produced, labelled, in the order a user cares about them."""
        result = self.result
        if result is None:
            return []
        found: list[tuple[str, Path]] = []
        for label, value in (
            ("video", result.muxed or result.video),
            ("subtitles", result.srt),
            ("web subtitles", result.vtt),
        ):
            if value is not None and Path(value).exists():
                found.append((label, Path(value)))
        return found

    @property
    def warnings(self) -> list[str]:
        """What the finished run wants read before anybody trusts the text.

        Carried through to the front end rather than left in the log, because the one that
        exists says the transcript may be the steering prompt rather than the speech, and a
        line scrolled off the top of a log is not a warning. The browser page puts these
        above the cue count for the same reason.
        """
        result = self.result
        if result is None:
            return []
        return [str(w) for w in (getattr(result, "warnings", None) or [])]

    def reveal(self, path: Path) -> str:
        """Put a finished file in front of the user, or say why that did not work."""
        argv = files.reveal_command(self.plat, path, is_dir=path.is_dir())
        try:
            self._opener(argv)
        except OSError as exc:
            # xdg-open is not on every Linux box, and failing to open a folder must not
            # read like a failure of the run that filled it.
            return f"could not open the file manager ({exc}); the folder is {path}"
        return ""

    def reveal_label(self) -> str:
        return "Show in Finder" if self.plat.is_macos else "Open folder"
