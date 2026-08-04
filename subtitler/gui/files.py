"""The file picker's model, and the one place the GUI branches on macOS.

A browser cannot hand a page the real path of a file the user picked, and uploading a 3 GB
video to a server running on the same machine is absurd, so the picker is served: the page
asks for a directory listing and the user clicks through their own disk. Everything here is
therefore a pure function of a path plus a `Platform`, which is what keeps the mac side
(`~/Movies`, `open -R`) testable from Linux exactly as `doctor.py` is.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from subtitler.doctor import Platform

AUDIO_SUFFIXES = frozenset(
    {".mp3", ".m4a", ".wav", ".flac", ".ogg", ".oga", ".opus", ".aac", ".wma", ".aiff", ".aif"}
)
VIDEO_SUFFIXES = frozenset(
    {".mp4", ".m4v", ".mkv", ".mov", ".avi", ".webm", ".mpg", ".mpeg", ".ts", ".wmv", ".flv"}
)
MEDIA_SUFFIXES = AUDIO_SUFFIXES | VIDEO_SUFFIXES

SUBTITLE_SUFFIXES = frozenset({".srt", ".vtt"})


def is_media(path: Path) -> bool:
    return path.suffix.lower() in MEDIA_SUFFIXES


# --------------------------------------------------------------------------------------
# Shortcuts
# --------------------------------------------------------------------------------------

# The only difference between the two platforms in this file, and the reason it is a
# constant rather than an `if` inside a loop: macOS calls it Movies, freedesktop calls it
# Videos, and a shortcut that points at a folder the user does not have is worse than no
# shortcut at all, so both lists are filtered by existence below.
_MAC_FOLDERS = ("Desktop", "Downloads", "Movies", "Music", "Documents")
_XDG_FOLDERS = ("Desktop", "Downloads", "Videos", "Music", "Documents")


@dataclass(frozen=True, slots=True)
class Place:
    name: str
    path: Path

    def to_dict(self) -> dict[str, str]:
        return {"name": self.name, "path": str(self.path)}


def places(plat: Platform, home: Path) -> list[Place]:
    """Home plus whichever of the usual media folders actually exist."""
    names = _MAC_FOLDERS if plat.is_macos else _XDG_FOLDERS
    found = [Place("Home", home)]
    found.extend(Place(name, home / name) for name in names if (home / name).is_dir())
    return found


# --------------------------------------------------------------------------------------
# Listing
# --------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Entry:
    name: str
    path: str
    is_dir: bool
    kind: str  # "folder" | "audio" | "video" | "subtitle" | "file"
    size: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "path": self.path,
            "is_dir": self.is_dir,
            "kind": self.kind,
            "size": self.size,
        }


def _kind(path: Path, is_dir: bool) -> str:
    if is_dir:
        return "folder"
    suffix = path.suffix.lower()
    if suffix in AUDIO_SUFFIXES:
        return "audio"
    if suffix in VIDEO_SUFFIXES:
        return "video"
    if suffix in SUBTITLE_SUFFIXES:
        return "subtitle"
    return "file"


def list_dir(path: Path, *, media_only: bool = True, show_hidden: bool = False) -> dict[str, Any]:
    """One directory, folders first, then files, both case-insensitively by name.

    Unreadable children are skipped rather than raised on: a home directory with one
    permission-denied folder in it must still list, or the picker is unusable on any real
    machine.
    """
    path = path.expanduser()
    if not path.is_dir():
        raise NotADirectoryError(f"not a folder: {path}")

    folders: list[Entry] = []
    files: list[Entry] = []
    try:
        children = sorted(path.iterdir(), key=lambda p: p.name.lower())
    except OSError as exc:
        raise PermissionError(f"cannot read {path}: {exc}") from exc

    for child in children:
        if not show_hidden and child.name.startswith("."):
            continue
        try:
            is_dir = child.is_dir()
            size = 0 if is_dir else child.stat().st_size
        except OSError:
            continue
        kind = _kind(child, is_dir)
        if is_dir:
            folders.append(Entry(child.name, str(child), True, kind, 0))
        elif not media_only or kind in {"audio", "video"}:
            files.append(Entry(child.name, str(child), False, kind, size))

    parent = str(path.parent) if path.parent != path else None
    return {
        "path": str(path),
        "parent": parent,
        "crumbs": [{"name": p.name or str(p), "path": str(p)} for p in reversed(path.parents)]
        + [{"name": path.name or str(path), "path": str(path)}],
        "entries": [e.to_dict() for e in (*folders, *files)],
    }


# --------------------------------------------------------------------------------------
# Showing the result to the user
# --------------------------------------------------------------------------------------


def reveal_command(plat: Platform, path: Path, *, is_dir: bool) -> Sequence[str]:
    """The argv that puts the finished file in front of the user.

    macOS is the primary target and it is the one that can do this properly: `open -R`
    opens Finder with the file *selected*, which is what "show me the subtitles" means.
    Nothing on freedesktop reveals a file, so Linux opens the containing folder instead.
    `is_dir` is passed in rather than probed so this stays a pure function and both
    branches are testable from either machine.
    """
    if plat.is_macos:
        return ["open", str(path)] if is_dir else ["open", "-R", str(path)]
    return ["xdg-open", str(path if is_dir else path.parent)]


def open_command(plat: Platform, path: Path) -> Sequence[str]:
    """Open a file with whatever the desktop thinks should play it."""
    return ["open", str(path)] if plat.is_macos else ["xdg-open", str(path)]
