"""Routing and JSON, with no socket anywhere in it.

`GuiApp.handle()` is a function from (method, path, query, body) to a `Response`, so the
whole API is exercised in tests by calling it, and `server.py` is left with nothing but
bytes on a wire. Every effect the app can have on the machine arrives as an argument:

    runner    the pipeline (default `pipeline.run_pipeline`)
    spawn     how a job gets a worker (default a daemon thread)
    opener    how a folder gets shown to the user (default `subprocess.Popen`)
    plat      what machine this is (default `doctor.detect_platform()`)

so a test can drive a complete run, a doctor report and a Finder reveal without a display,
a GPU, or a Mac.
"""

from __future__ import annotations

import json
import mimetypes
import subprocess
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from subtitler import __version__, doctor, media, models
from subtitler import edits as edits_mod
from subtitler.gui import files, forms, jobs
from subtitler.gui.jobs import JobManager
from subtitler.model import Cue
from subtitler.pipeline import output_dir, run_pipeline, work_dir

STATIC_DIR = Path(__file__).resolve().parent / "static"

# One range response is capped rather than sized to the request, so that scrubbing a
# 90-minute lecture never reads the whole file into this process's memory. Answering with
# fewer bytes than were asked for is what the range machinery is for; the browser simply
# asks again for the rest.
RANGE_CHUNK = 4 * 1024 * 1024

Opener = Callable[[Sequence[str]], None]


@dataclass(frozen=True, slots=True)
class Response:
    status: int
    body: bytes
    content_type: str = "application/json; charset=utf-8"
    # Extra headers, which only the media route needs: a browser will not seek inside an
    # <audio> element unless the server answers ranges.
    headers: tuple[tuple[str, str], ...] = ()


def _json(payload: Any, status: int = 200) -> Response:
    body = json.dumps(payload, ensure_ascii=False, default=str).encode("utf-8")
    return Response(status, body)


def _error(message: str, status: int = 400, **extra: Any) -> Response:
    return _json({"error": message, **extra}, status)


def _default_opener(argv: Sequence[str]) -> None:
    subprocess.Popen(list(argv), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


class GuiApp:
    def __init__(
        self,
        *,
        token: str = "",
        plat: doctor.Platform | None = None,
        home: Path | None = None,
        runner: Callable[..., Any] | None = None,
        spawn: jobs.Spawn | None = None,
        opener: Opener | None = None,
        diagnose: Callable[[doctor.Platform], list[doctor.DepStatus]] | None = None,
    ) -> None:
        self.token = token
        self.plat = plat or doctor.detect_platform()
        self.home = home or Path.home()
        self._runner = runner or run_pipeline
        self._spawn = spawn or jobs.thread_spawn
        self._opener = opener or _default_opener
        self._diagnose = diagnose or (lambda p: doctor.diagnose(p))
        self.jobs = JobManager(spawn=self._spawn)
        self._doctor_report: dict[str, Any] | None = None
        self._doctor_running = False
        # The media the last review run actually transcribed: the trimmed fragment when
        # there was one, the downloaded file for a URL. Remembered rather than accepted as
        # a parameter, so `/api/media` cannot be turned into "read any file on this disk".
        self._review_media: Path | None = None

    # ---------------------------------------------------------------- dispatch

    def handle(
        self,
        method: str,
        path: str,
        query: Mapping[str, str] | None = None,
        body: bytes = b"",
        *,
        token: str | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> Response:
        query = query or {}
        headers = headers or {}

        if not path.startswith("/api/"):
            return self._static(path)

        # The server binds to loopback, but "only this machine" is not "only this user's
        # browser": any page the user visits can POST to 127.0.0.1. The token is handed to
        # the page in the URL that was opened for it and required on every API call, so a
        # drive-by request cannot list the home directory or start a run.
        if self.token and token != self.token:
            return _error("bad or missing token; reopen the link subtitler printed", 403)

        route = (method.upper(), path)
        if route == ("GET", "/api/options"):
            return self._options()
        if route == ("GET", "/api/files"):
            return self._files(query)
        if route == ("GET", "/api/doctor"):
            return self._doctor(query)
        if route == ("GET", "/api/models"):
            return self._models()
        if route == ("POST", "/api/models/download"):
            return self._download(_body(body))
        if route == ("POST", "/api/run"):
            return self._run(_body(body))
        if route == ("POST", "/api/preview"):
            return self._preview(_body(body))
        if route == ("POST", "/api/cues/check"):
            return self._check(_body(body))
        if route == ("POST", "/api/burn"):
            return self._approve(_body(body))
        if route == ("GET", "/api/media"):
            return self._media(headers)
        if route == ("GET", "/api/job"):
            return self._job(query)
        if route == ("POST", "/api/reveal"):
            return self._reveal(_body(body))
        return _error(f"no route for {method} {path}", 404)

    # ---------------------------------------------------------------- static

    def _static(self, path: str) -> Response:
        name = "index.html" if path in {"", "/", "/index.html"} else path.lstrip("/")
        target = (STATIC_DIR / name).resolve()
        if not target.is_file() or STATIC_DIR not in target.parents:
            return Response(404, b"not found", "text/plain; charset=utf-8")
        kind = mimetypes.guess_type(target.name)[0] or "application/octet-stream"
        if kind.startswith("text/") or kind.endswith("javascript"):
            kind += "; charset=utf-8"
        return Response(200, target.read_bytes(), kind)

    # ---------------------------------------------------------------- routes

    def _options(self) -> Response:
        payload = forms.options()
        payload.update(
            version=__version__,
            platform={
                "system": self.plat.system,
                "machine": self.plat.machine,
                "label": self.plat.describe(),
                "is_macos": self.plat.is_macos,
                "apple_silicon": self.plat.is_apple_silicon,
            },
            home=str(self.home),
            places=[p.to_dict() for p in files.places(self.plat, self.home)],
            reveal_label="Show in Finder" if self.plat.is_macos else "Open folder",
        )
        return _json(payload)

    def _files(self, query: Mapping[str, str]) -> Response:
        raw = query.get("path") or str(self.home)
        try:
            listing = files.list_dir(
                Path(raw),
                media_only=query.get("all") not in {"1", "true"},
                show_hidden=query.get("hidden") in {"1", "true"},
            )
        except (NotADirectoryError, PermissionError, OSError) as exc:
            return _error(str(exc), 400)
        return _json(listing)

    def _doctor(self, query: Mapping[str, str]) -> Response:
        """Kicked off in the background, because it shells out a dozen times.

        The whole point of putting `doctor` in the window is the user who will not run a
        terminal command to find out their ffmpeg has no libass. Blocking the first paint
        for several seconds to tell them so would trade one annoyance for another.
        """
        if query.get("refresh") in {"1", "true"}:
            self._doctor_report = None
        if self._doctor_report is None and not self._doctor_running:
            self._doctor_running = True
            self._spawn(self._refresh_doctor)
        # Re-read after spawning rather than before: an inline `spawn` (which is how the
        # tests run) has already finished by this point, and answering "not ready" to a
        # report that is sitting right there would make the page wait for nothing.
        if self._doctor_report is not None:
            return _json({"ready": True, **self._doctor_report})
        return _json({"ready": False})

    def _refresh_doctor(self) -> None:
        try:
            statuses = self._diagnose(self.plat)
            self._doctor_report = {
                "platform": self.plat.describe(),
                "text": doctor.render(statuses, self.plat),
                "deps": [s.to_dict() for s in statuses],
                "blocking": [s.dep.key for s in statuses if s.blocking],
            }
        except Exception as exc:  # a broken check must not wedge the panel
            self._doctor_report = {
                "platform": self.plat.describe(),
                "text": f"the dependency check itself failed: {exc}",
                "deps": [],
                "blocking": [],
            }
        finally:
            self._doctor_running = False

    def _models(self) -> Response:
        """What weights this machine would use, and whether they are on disk.

        `EngineUnavailable` says "the large-v3 weights are not downloaded" and names a
        terminal command, which is a dead end for this audience. The page offers the
        download instead, through `models.download`'s progress callback.
        """
        backend = "mlx" if self.plat.is_apple_silicon else "faster-whisper"
        specs = []
        for spec in models.specs_for_backend(backend):
            path = models.local_path(spec)
            specs.append(
                {
                    "name": spec.name,
                    "backend": spec.backend,
                    "size": spec.size_label,
                    "cached": path is not None,
                    "path": str(path) if path else None,
                }
            )
        return _json({"backend": backend, "cache_root": str(models.cache_root()), "models": specs})

    def _download(self, payload: Mapping[str, Any]) -> Response:
        backend = str(payload.get("backend") or "").strip() or (
            "mlx" if self.plat.is_apple_silicon else "faster-whisper"
        )
        name = str(payload.get("name") or "large-v3").strip()
        try:
            spec = models.resolve(name, backend)
        except models.ModelNotFound as exc:
            return _error(str(exc), 400)

        def work(log: jobs.Log) -> dict[str, Any]:
            path = models.download(spec, progress=log)
            return {"name": spec.name, "backend": spec.backend, "path": str(path)}

        return self._start("download", f"downloading {spec.key} ({spec.size_label})", work)

    def _preview(self, payload: Mapping[str, Any]) -> Response:
        """Validate the form and echo the command line, without starting anything."""
        try:
            cfg = forms.build_config(payload)
        except forms.FormError as exc:
            return _json({"ok": False, **exc.to_dict()})
        return _json({"ok": True, "command": forms.command_line(cfg), "argv": forms.to_argv(cfg)})

    def _run(self, payload: Mapping[str, Any]) -> Response:
        try:
            cfg = forms.build_config(payload)
        except forms.FormError as exc:
            return _json({"ok": False, **exc.to_dict()}, 400)
        return self._start_run(cfg)

    def _start_run(self, cfg: Any) -> Response:
        runner = self._runner
        command = forms.command_line(cfg)
        # `output_dir` and not `cfg.input.parent`: a URL run has no input path, and
        # reaching for one is the AttributeError this route used to raise on every link.
        out_dir = str(output_dir(cfg))

        def work(log: jobs.Log) -> dict[str, Any]:
            log(f"$ {command}")
            result = runner(cfg, log=log)
            summary = result.to_dict()
            summary["out_dir"] = str(
                Path(summary["srt"]).parent if summary.get("srt") else Path(out_dir)
            )
            if cfg.review:
                summary["review"] = self._review_payload(cfg, result)
            return summary

        return self._start(
            "run",
            forms.job_label(cfg),
            work,
            stages=forms.stages_for(cfg),
            command=command,
            out_dir=out_dir,
        )

    # ---------------------------------------------------------------- the editor

    def _review_payload(self, cfg: Any, result: Any) -> dict[str, Any]:
        """Everything the editor needs to open, computed once when the run stops.

        The per-cue quality report is built by the same `edits.cue_report` the live check
        route uses, so the marks a cue carries when the page opens and the marks it carries
        after a keystroke are produced by one function rather than two that can drift.
        """
        media_path = Path(result.input) if getattr(result, "input", None) else None
        self._review_media = media_path if media_path and media_path.is_file() else None
        return {
            "base_key": getattr(result, "cues_key", ""),
            "cues": edits_mod.cue_reports(result.cues, cfg.cues),
            "limits": {
                "max_line": cfg.cues.max_line,
                "max_lines": cfg.cues.max_lines,
                "min_dur": cfg.cues.min_dur,
                "max_dur": cfg.cues.max_dur,
                "max_cps": cfg.cues.max_cps,
            },
            "edited": (result.edits or {}).get("applied", []),
            "stale": (result.edits or {}).get("stale", 0),
            "media": str(media_path) if self._review_media else None,
        }

    def _check(self, payload: Mapping[str, Any]) -> Response:
        """Re-wrap and re-lint cues as they are typed.

        The wrapping happens here rather than in JavaScript because `cues.wrap_edited` is
        the only thing allowed to choose a line break: a second implementation in the page
        would be a second set of rules about Serbian clitics, kept in sync by hope.
        """
        try:
            cfg = forms.cue_config(payload)
        except forms.FormError as exc:
            return _json({"ok": False, **exc.to_dict()}, 400)

        rows = payload.get("cues")
        if not isinstance(rows, list):
            return _error("expected a list of cues to check")

        out: list[dict[str, Any]] = []
        for row in rows[:500]:
            if not isinstance(row, Mapping):
                continue
            try:
                index = int(row.get("index", 0))
                start = float(row.get("start", 0.0))
                end = float(row.get("end", 0.0))
            except (TypeError, ValueError):
                continue
            text = str(row.get("text") or "")
            lines = row.get("lines")
            if isinstance(lines, list) and lines and not text.strip():
                cue = Cue(index=index, start=start, end=end, lines=tuple(str(x) for x in lines))
            else:
                cue = edits_mod.relayout(
                    Cue(index=index, start=start, end=end, lines=()), text, cfg
                )
            out.append(edits_mod.cue_report(cue, cfg))
        return _json({"ok": True, "cues": out})

    def _approve(self, payload: Mapping[str, Any]) -> Response:
        """Save the corrections and run the second half: render, burn, and any soft mux.

        The corrections are written to `edits.json` in the work directory rather than
        carried in the request that starts the job, because that file is the pipeline's
        input and outlives the browser tab. See `edits.py`.
        """
        settings = {k: v for k, v in payload.items() if k not in {"edits", "base_key"}}
        settings["review"] = False
        try:
            cfg = forms.build_config(settings)
        except forms.FormError as exc:
            return _json({"ok": False, **exc.to_dict()}, 400)

        try:
            edit_set = edits_mod.build(payload.get("base_key", ""), payload.get("edits") or {})
        except edits_mod.EditError as exc:
            return _error(str(exc))

        try:
            work = work_dir(cfg)
        except media.MediaError as exc:
            return _error(str(exc))

        if edit_set:
            edits_mod.save(work, edit_set)
        else:
            # Nothing was corrected this time, so a set left over from a previous pass
            # must not be applied behind the user's back.
            edits_mod.clear(work)
        return self._start_run(cfg)

    def _media(self, headers: Mapping[str, str]) -> Response:
        """The audio the cues were made from, so a cue can be heard rather than counted.

        Range requests are answered because a browser will not seek inside an <audio>
        element without them, and seeking is the entire feature: the editor plays the span
        of the cue being typed, which is the only way to judge "too fast to read".
        """
        target = self._review_media
        if target is None or not target.is_file():
            return _error("no media to play; run a review first", 404)

        size = target.stat().st_size
        kind = mimetypes.guess_type(target.name)[0] or "application/octet-stream"
        span = _parse_range(headers.get("range", ""), size)
        if span is None:
            return Response(200, target.read_bytes(), kind, (("Accept-Ranges", "bytes"),))

        start, end = span
        with target.open("rb") as handle:
            handle.seek(start)
            chunk = handle.read(end - start + 1)
        return Response(
            206,
            chunk,
            kind,
            (
                ("Accept-Ranges", "bytes"),
                ("Content-Range", f"bytes {start}-{start + len(chunk) - 1}/{size}"),
            ),
        )

    def _start(
        self,
        kind: str,
        label: str,
        work: jobs.Work,
        *,
        stages: tuple[str, ...] = jobs.STAGES,
        **extra: Any,
    ) -> Response:
        try:
            job = self.jobs.start(kind, label, work, stages=stages)
        except jobs.Busy as exc:
            return _error(str(exc), 409)
        snapshot = job.snapshot(0, now=job.started)
        return _json({"ok": True, **snapshot, **extra})

    def _job(self, query: Mapping[str, str]) -> Response:
        try:
            since = int(query.get("since", "0"))
        except ValueError:
            since = 0
        snapshot = self.jobs.snapshot(query.get("id") or None, since)
        if snapshot is None:
            return _error("no such job", 404)
        return _json(snapshot)

    def _reveal(self, payload: Mapping[str, Any]) -> Response:
        raw = str(payload.get("path") or "").strip()
        if not raw:
            return _error("nothing to show", 400)
        target = Path(raw).expanduser()
        if not target.exists():
            return _error(f"no such path: {target}", 400)
        argv = files.reveal_command(self.plat, target, is_dir=target.is_dir())
        try:
            self._opener(argv)
        except OSError as exc:
            # xdg-open is not installed on every Linux box, and a failure to open a folder
            # must not read like a failure of the run that produced it.
            return _error(f"could not open the file manager ({exc}); the folder is {target}", 500)
        return _json({"ok": True, "argv": list(argv)})


def _parse_range(header: str, size: int) -> tuple[int, int] | None:
    """`bytes=START-END` to an inclusive, clamped, capped span. None means "send it all".

    Only the single-range form is handled, which is the only one a media element sends.
    Anything unparseable falls back to the whole file rather than a 416, because a player
    that gets bytes is a player that works.
    """
    value = (header or "").strip().lower()
    if not value.startswith("bytes=") or "," in value or size <= 0:
        return None
    first, _, last = value[len("bytes=") :].partition("-")
    try:
        if first == "":
            # A suffix range: the last N bytes.
            length = int(last)
            start, end = max(size - length, 0), size - 1
        else:
            start = int(first)
            end = int(last) if last else size - 1
    except ValueError:
        return None
    if start >= size or start < 0 or end < start:
        return None
    end = min(end, size - 1, start + RANGE_CHUNK - 1)
    return start, end


def _body(raw: bytes) -> dict[str, Any]:
    if not raw:
        return {}
    try:
        parsed = json.loads(raw.decode("utf-8"))
    except (ValueError, UnicodeDecodeError):
        return {}
    return parsed if isinstance(parsed, dict) else {}
