"""Downloading media from a URL, through yt-dlp.

yt-dlp is an **optional extra**, not a core dependency, for the same reason LiteLLM is: a
run over a local file must not pay for it and must not require it to be installed. It is
pure Python, so non-negotiable 6 still holds, and it is imported lazily so `--help` and
every file-based run cost nothing.

Two things this module is careful about:

* **What it asks the site for.** `--srt-only` never renders a pixel, so asking YouTube for
  1080p to produce a text file spends the user's bandwidth on nothing. `kind="audio"` asks
  for an audio track and `kind="video"` for a muxed mp4.
* **Where it puts it.** The caller passes the directory, and the pipeline passes its work
  directory. Nothing lands in the user's CWD (non-negotiable 4).

Errors are the other half of the job. A private video, a region block, a dead network and
a yt-dlp too old for a site's current layout are four different problems with four
different fixes, and all four arrive from yt-dlp as one `DownloadError` with a traceback
behind it. `_explain` turns each into a sentence naming the fix.
"""

from __future__ import annotations

import importlib.util
import json
import re
import time
import unicodedata
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from subtitler import cache as cache_mod

__all__ = [
    "FORMATS",
    "FetchError",
    "Fetched",
    "available",
    "cache_params",
    "fetch",
    "is_url",
    "normalize_url",
    "options",
    "read_info",
    "slugify",
    "url_id",
    "work_stem",
    "write_info",
]

INSTALL_HINT = "uv sync --extra fetch"

# http/https only. A local path never starts with one, so `run` can tell a URL from a file
# without touching the filesystem and without guessing about a file that does not exist yet.
_URL_RE = re.compile(r"^https?://", re.IGNORECASE)

# Query parameters that identify the sharer rather than the media. A share link pasted
# twice from two different places must not download the same video twice.
_NOISE_PARAMS = ("si", "feature", "pp")
_NOISE_PREFIXES = ("utm_",)

# The format selector per shape. 1080p is the ceiling on purpose: the burn re-encodes the
# video anyway, and a 4K source costs minutes of download and encode for a subtitle overlay
# nobody will inspect at that scale. The fallbacks matter as much as the first choice,
# because a site that offers neither an mp4 nor a separate m4a must still work.
FORMATS: dict[str, str] = {
    "audio": "bestaudio[ext=m4a]/bestaudio/best",
    "video": "bestvideo[height<=1080][ext=mp4]+bestaudio[ext=m4a]/best[height<=1080]/best",
}

# The download's basename inside the work directory. The extension is whatever the site
# served, and is recorded in `fetch.json` rather than assumed, because forcing a container
# would mean re-encoding audio this pipeline is about to resample to 16 kHz mono anyway.
DOWNLOAD_STEM = "fetch"
INFO_NAME = "fetch.json"

# At most one progress line per this many seconds. yt-dlp calls its hook several times a
# second; unthrottled that is hundreds of lines in a terminal and hundreds of events in a
# GUI's log stream, for a number that only needs to move.
PROGRESS_INTERVAL_S = 1.0

Progress = Callable[[str], None]


class FetchError(RuntimeError):
    """The download did not happen, and the message says what the user can do about it."""


@dataclass(frozen=True, slots=True)
class Fetched:
    """What came back, and enough about it to name the outputs without asking again."""

    path: Path
    url: str
    id: str = ""
    title: str = ""
    duration: float | None = None
    extractor: str = ""
    kind: str = "video"

    @property
    def stem(self) -> str:
        """A filesystem-safe name for the outputs derived from this download."""
        return slugify(self.title) or slugify(self.id) or work_stem(self.url)

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "url": self.url,
            "id": self.id,
            "title": self.title,
            "duration": self.duration,
            "extractor": self.extractor,
            "kind": self.kind,
        }


# --------------------------------------------------------------------------------------
# Pure helpers. No network, no yt-dlp, testable everywhere.
# --------------------------------------------------------------------------------------


def is_url(text: str | Path) -> bool:
    return bool(_URL_RE.match(str(text).strip()))


def normalize_url(url: str) -> str:
    """Strip the parts of a URL that identify the sharer rather than the media.

    Only tracking parameters go. `t`, `list` and `start` are left alone: they change which
    media or which part of it is meant, and dropping them would make two different requests
    share one cache entry.
    """
    from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

    parts = urlsplit(url.strip())
    kept = [
        (key, value)
        for key, value in parse_qsl(parts.query, keep_blank_values=True)
        if key not in _NOISE_PARAMS and not key.startswith(_NOISE_PREFIXES)
    ]
    path = parts.path.rstrip("/") or parts.path
    return urlunsplit((parts.scheme.lower(), parts.netloc.lower(), path, urlencode(kept), ""))


def url_id(url: str) -> str:
    """The `fetch` stage's input hash: the normalized URL, hashed like a file's content."""
    return cache_mod.text_id(normalize_url(url))


def work_stem(url: str) -> str:
    """The work directory's name for a URL run.

    Derived from the URL rather than from the video's title or id on purpose: the work
    directory has to be known *before* anything is downloaded, and resolving a title costs
    a network round trip that a warm run must not pay. The friendly name is used for the
    output files, where it is available from the cached `fetch.json`.
    """
    return f"url-{url_id(url)}"


def slugify(text: str, *, limit: int = 80) -> str:
    """A filesystem-safe stem. Keeps letters and digits, including ČĆĐŠŽ."""
    cleaned = unicodedata.normalize("NFC", (text or "").strip())
    cleaned = re.sub(r"[^\w\s-]", "", cleaned, flags=re.UNICODE)
    cleaned = re.sub(r"[\s_-]+", "-", cleaned).strip("-.")
    return cleaned[:limit].strip("-.")


def write_info(path: Path, fetched: Fetched) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(fetched.to_dict(), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


def read_info(path: Path) -> Fetched | None:
    """The cached record, or None if it is absent or unreadable.

    None rather than an exception: a corrupt sidecar has to degrade to downloading again,
    exactly like a corrupt stage meta degrades to re-running the stage.
    """
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(data, dict) or not data.get("path"):
        return None
    return Fetched(
        path=Path(data["path"]),
        url=str(data.get("url", "")),
        id=str(data.get("id", "")),
        title=str(data.get("title", "")),
        duration=data.get("duration"),
        extractor=str(data.get("extractor", "")),
        kind=str(data.get("kind", "video")),
    )


def cache_params(kind: str) -> dict[str, Any]:
    """Everything that changes what lands on disk, and nothing that does not.

    The format selector is in here as well as the shape name, so that changing the selector
    in a future release re-downloads rather than serving a file chosen by the old one. The
    output directory is not: moving a work directory does not change what was downloaded.
    """
    return {"kind": kind, "format": FORMATS[kind]}


def available() -> bool:
    """Whether yt-dlp can be imported, without importing it."""
    return importlib.util.find_spec("yt_dlp") is not None


# --------------------------------------------------------------------------------------
# The download
# --------------------------------------------------------------------------------------


def _import_yt_dlp() -> Any:
    try:
        import yt_dlp
    except ImportError as exc:
        raise FetchError(f"downloading a URL needs yt-dlp.\n  fix: {INSTALL_HINT}") from exc
    return yt_dlp


def _explain(message: str) -> str:
    """Turn yt-dlp's message into one sentence a user can act on.

    Matched on substrings rather than on exception types, because yt-dlp raises one
    `DownloadError` for every one of these and puts the distinguishing detail in the text.

    Order matters. A dead network reaches yt-dlp as "Unable to download webpage", the same
    opening as a page that changed shape, so the transport causes are tested before the
    extractor ones. Getting that backwards told a user with no wifi to upgrade yt-dlp.
    """
    low = message.lower()
    for markers, sentence in _EXPLANATIONS:
        if any(marker in low for marker in markers):
            return sentence
    return message.strip().splitlines()[-1] if message.strip() else "the download failed."


# (substrings, the sentence). First match wins, so the order is part of the behaviour.
_EXPLANATIONS: tuple[tuple[tuple[str, ...], str], ...] = (
    (
        (
            "failed to resolve",
            "name or service not known",
            "nodename nor servname",
            "name resolution",
            "no route to host",
            "network is unreachable",
            "connection refused",
            "connection reset",
            "timed out",
            "urlopen error",
        ),
        "the network is not reachable from here.",
    ),
    (
        ("private video", "sign in if you've been granted access"),
        "that video is private. Nothing can download it without an account that can see it.",
    ),
    (
        ("members-only", "join this channel"),
        "that video is members-only, so a public request cannot reach it.",
    ),
    (
        (
            "available in your country",
            "available from your location",
            "geo restrict",
            "geo-restrict",
        ),
        "that video is blocked in this region.",
    ),
    (
        ("video unavailable", "has been removed", "is not available", "http error 404"),
        "that video is unavailable: removed, never public, or the URL is wrong.",
    ),
    (
        ("age-restricted", "age restricted", "confirm your age"),
        "that video is age-restricted and needs a signed-in session yt-dlp does not have.",
    ),
    (
        ("sign in to confirm", "not a bot", "cookies", "http error 403"),
        (
            "the site asked for a signed-in session before serving this video, and there is "
            "not one here."
        ),
    ),
    (
        ("http error 429", "too many requests", "rate-limit", "rate limit"),
        "the site is rate-limiting this machine. Wait a few minutes and try again.",
    ),
    (
        ("unsupported url", "no video formats found", "no suitable format"),
        "yt-dlp found nothing downloadable at that URL.",
    ),
    (
        ("unable to extract", "unable to download webpage", "failed to parse", "extractor"),
        (
            "yt-dlp could not read that page, which usually means the site changed and this "
            f"yt-dlp is too old.\n  fix: {INSTALL_HINT} --upgrade-package yt-dlp"
        ),
    ),
)


def _progress_reporter(progress: Progress | None) -> tuple[Any, Any]:
    """A yt-dlp progress hook plus a logger, both routed into one string callback.

    `models.download` takes the same `progress: Callable[[str], None]`, so a GUI that can
    already stream a model download can stream this one without learning a second shape.
    """
    say: Progress = progress or (lambda _message: None)
    state = {"last": 0.0}

    def hook(event: dict[str, Any]) -> None:
        status = event.get("status")
        if status == "finished":
            say(f"downloaded: {Path(str(event.get('filename', '?'))).name}")
            return
        if status == "error":
            say("download failed")
            return
        now = time.monotonic()
        if now - state["last"] < PROGRESS_INTERVAL_S:
            return
        state["last"] = now
        total = event.get("total_bytes") or event.get("total_bytes_estimate")
        done = event.get("downloaded_bytes") or 0
        speed = event.get("speed")
        pieces = [f"{done / 1e6:.1f} MB"]
        if total:
            pieces.insert(0, f"{100 * done / total:.0f}%")
            pieces[-1] = f"{done / 1e6:.1f}/{total / 1e6:.1f} MB"
        if speed:
            pieces.append(f"{speed / 1e6:.1f} MB/s")
        say("fetching: " + " ".join(pieces))

    class Logger:
        """yt-dlp writes to stdout by default, and `run --json` owns stdout."""

        def debug(self, message: str) -> None:
            if message.startswith("[debug] "):
                return
            if any(mark in message for mark in ("[download] Destination", "Merging formats")):
                say(message.strip())

        def info(self, message: str) -> None:
            say(message.strip())

        def warning(self, message: str) -> None:
            say(f"yt-dlp: {message.strip()}")

        def error(self, message: str) -> None:  # raised as an exception too; do not duplicate
            pass

    return hook, Logger()


def options(dst_dir: Path, *, kind: str, progress: Progress | None = None) -> dict[str, Any]:
    """The yt-dlp options dict. Split out so a test can assert on it without a network."""
    if kind not in FORMATS:
        raise FetchError(f"unknown fetch kind {kind!r}; choose from {sorted(FORMATS)}")
    hook, logger = _progress_reporter(progress)
    opts: dict[str, Any] = {
        "format": FORMATS[kind],
        # An absolute template, and deliberately no `paths`: yt-dlp ignores `paths` when the
        # template is absolute and says so in a warning, and the two saying the same thing
        # twice is one of them being wrong later. Partial files and the pre-merge streams
        # land beside it, which is the work directory, which is the point.
        "outtmpl": str(dst_dir / f"{DOWNLOAD_STEM}.%(ext)s"),
        "noplaylist": True,
        "quiet": True,
        "no_warnings": False,
        "noprogress": True,
        "logger": logger,
        "progress_hooks": [hook],
        "retries": 3,
        "overwrites": True,
    }
    if kind == "video":
        # One file out, whatever the site split it into. Without this a DASH source lands
        # as two files and the pipeline would transcribe the audio of one and burn onto
        # the other.
        opts["merge_output_format"] = "mp4"
    return opts


def fetch(
    url: str,
    dst_dir: Path,
    *,
    kind: str = "video",
    progress: Progress | None = None,
) -> Fetched:
    """Download `url` into `dst_dir` and return what landed there.

    `kind` is "video" when the run will produce a video and "audio" when it will not.
    Raises `FetchError` with an actionable sentence for everything a user can fix.
    """
    yt_dlp = _import_yt_dlp()
    dst_dir = Path(dst_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)
    opts = options(dst_dir, kind=kind, progress=progress)

    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(url, download=True)
    except FetchError:
        raise
    except Exception as exc:  # yt-dlp raises DownloadError, ExtractorError and OSError alike
        raise FetchError(f"could not download {url}: {_explain(str(exc))}") from exc

    if not isinstance(info, dict):
        raise FetchError(f"could not download {url}: yt-dlp returned nothing to download.")
    # A playlist URL with noplaylist set still comes back wrapped on some extractors.
    if info.get("_type") == "playlist":
        entries = [entry for entry in (info.get("entries") or []) if isinstance(entry, dict)]
        if not entries:
            raise FetchError(f"could not download {url}: that URL holds no playable media.")
        info = entries[0]

    path = _downloaded_path(info, dst_dir)
    return Fetched(
        path=path,
        url=url,
        id=str(info.get("id") or ""),
        title=str(info.get("title") or ""),
        duration=_as_float(info.get("duration")),
        extractor=str(info.get("extractor_key") or info.get("extractor") or ""),
        kind=kind,
    )


def _downloaded_path(info: dict[str, Any], dst_dir: Path) -> Path:
    """Where the file actually landed.

    yt-dlp records this in `requested_downloads`, which is authoritative after any
    post-processing (a merge changes the extension). The glob is the fallback for older
    versions and for extractors that do not fill it in; it excludes the partial files and
    the sidecar so it cannot return one of those.
    """
    for entry in info.get("requested_downloads") or []:
        candidate = entry.get("filepath") or entry.get("_filename")
        if candidate and Path(candidate).exists():
            return Path(candidate)
    candidate = info.get("filepath") or info.get("_filename")
    if candidate and Path(candidate).exists():
        return Path(candidate)
    found = sorted(
        item
        for item in dst_dir.glob(f"{DOWNLOAD_STEM}.*")
        if item.is_file() and item.suffix not in {".part", ".ytdl", ".json"}
    )
    if found:
        return found[0]
    raise FetchError(
        "yt-dlp reported success but no media file was written. "
        f"Look in {dst_dir} and try again with --force fetch."
    )


def _as_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
