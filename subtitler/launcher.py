"""Something to double-click.

`subtitler gui` from a terminal does not solve the problem a GUI exists for: the person
this is built for does not open a terminal. `subtitler install-app` is run once, by
whoever sets the machine up, and afterwards there is an icon in Applications or in the app
menu.

Everything here is a pure function of a `Platform` and a few paths, so the macOS half is
built and inspected from Linux exactly as `doctor.py`'s is (non-negotiable 5). `plan()`
decides what the files are; `install()` writes them and does nothing else.

**The bundle is not signed.** It cannot be: signing needs an Apple Developer certificate,
and notarising needs to upload the thing to Apple. So the first launch shows *"Subtitler
cannot be opened because it is from an unidentified developer"*, and the user has to
right-click the icon and choose Open, which offers the same dialog with an Open button on
it. That is once per install, and `install()` prints it. Pretending otherwise would put
the friend in front of a dialog nobody warned them about.

**A Finder-launched app has almost no PATH.** launchd hands it `/usr/bin:/bin:/usr/sbin:
/sbin`, which does not include `/opt/homebrew/bin`, so ffmpeg is missing from a bundle that
works perfectly from a shell. The bundle's executable puts the two Homebrew prefixes back.
"""

from __future__ import annotations

import shlex
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from subtitler import __version__, icon
from subtitler.doctor import Platform

APP_NAME = "Subtitler"
BUNDLE_ID = "com.github.vlazic.subtitler"
DESKTOP_NAME = "subtitler.desktop"
ICON_STEM = "subtitler"
COMMENT = "Audio or video in, burned-in subtitles out"

# Where a Finder-launched app's log goes. `~/Library/Logs` is the documented user location
# and Console.app lists it, which makes "it opened and vanished" a question with an answer.
MAC_LOG = "$HOME/Library/Logs/Subtitler.log"


class LauncherError(RuntimeError):
    """This platform has no launcher this project knows how to write."""


@dataclass(frozen=True, slots=True)
class Artifact:
    path: Path
    content: bytes
    executable: bool = False


@dataclass(frozen=True, slots=True)
class Plan:
    """What `install-app` will write, and what the user must be told afterwards."""

    target: Path
    artifacts: tuple[Artifact, ...]
    notes: tuple[str, ...] = ()
    directories: tuple[Path, ...] = ()

    def paths(self) -> list[Path]:
        return [a.path for a in self.artifacts]


# --------------------------------------------------------------------------------------
# What the launcher runs
# --------------------------------------------------------------------------------------


def launch_argv(
    *,
    executable: Path | None = None,
    exists: Callable[[Path], bool] = Path.exists,
) -> list[str]:
    """The absolute command the icon runs, resolved once at install time.

    Absolute on purpose. A `.desktop` entry and a `.app` bundle are both launched with a
    PATH the user never set, so `subtitler` as a bare word works from a shell and fails
    from an icon. The console script installed next to the running interpreter is
    preferred; a checkout run straight out of `uv run` may not have one, and there
    `-m subtitler.cli` is the same entry point by another road.

    **`absolute()` and never `resolve()`.** A virtualenv's `bin/python3` is a symlink to
    the interpreter it was made from, so resolving it walks straight out of the
    environment: on this machine it produced
    `~/.local/share/uv/python/cpython-3.12.11-.../bin/python3.12`, which has no subtitler
    installed, and the entry `install-app` wrote could not have run. Nothing here needs a
    canonical path; it needs the path whose neighbours are this project's packages.
    """
    python = Path(executable or sys.executable).expanduser().absolute()
    script = python.parent / "subtitler"
    if exists(script):
        return [str(script), "gui"]
    return [str(python), "-m", "subtitler.cli", "gui"]


def _quoted(argv: Sequence[str]) -> str:
    return " ".join(shlex.quote(part) for part in argv)


# --------------------------------------------------------------------------------------
# Linux
# --------------------------------------------------------------------------------------


def desktop_entry(argv: Sequence[str], *, icon_name: str = ICON_STEM) -> str:
    """A freedesktop Desktop Entry. `desktop-file-validate` passes it clean.

    `Terminal=false` is the whole point of the exercise. `StartupWMClass` matches the
    class Tk gives its toplevel (the executable's basename, capitalised) so the window
    docks against this launcher instead of appearing as a second, nameless icon.

    Two things the validator refused before it was run against the real file:

    * **`Version` is the version of the *specification*, not of this program**, and only
      the values the spec has actually had are accepted. `1.5` is not one of them.
    * **Exactly one main category.** `AudioVideo`, `Audio`, `Video` and `Utility` are all
      main categories, and a file naming four of them puts the application in the menu
      four times. `AudioVideoEditing` is an additional category, which is what the rest of
      the description belongs in.
    """
    return (
        "[Desktop Entry]\n"
        "Type=Application\n"
        "Version=1.0\n"
        f"Name={APP_NAME}\n"
        "GenericName=Subtitle maker\n"
        f"Comment={COMMENT}\n"
        f"Exec={_quoted(argv)}\n"
        f"Icon={icon_name}\n"
        "Terminal=false\n"
        "Categories=AudioVideo;AudioVideoEditing;\n"
        "Keywords=subtitles;srt;transcribe;whisper;captions;\n"
        "StartupNotify=true\n"
        "StartupWMClass=Subtitler\n"
    )


def _linux_plan(home: Path, argv: Sequence[str]) -> Plan:
    apps = home / ".local" / "share" / "applications"
    icons = home / ".local" / "share" / "icons" / "hicolor"
    artifacts = [
        Artifact(apps / DESKTOP_NAME, desktop_entry(argv).encode("utf-8")),
    ]
    artifacts.extend(
        Artifact(icons / f"{size}x{size}" / "apps" / f"{ICON_STEM}.png", icon.png_bytes(size))
        for size in icon.PNG_SIZES
    )
    return Plan(
        target=apps / DESKTOP_NAME,
        artifacts=tuple(artifacts),
        notes=(
            "It is in your applications menu as Subtitler. If your desktop caches the "
            "menu, `update-desktop-database ~/.local/share/applications` refreshes it.",
        ),
    )


# --------------------------------------------------------------------------------------
# macOS
# --------------------------------------------------------------------------------------


def info_plist(*, version: str = __version__) -> str:
    """`Contents/Info.plist`, with the keys Apple documents as required for an app bundle.

    `CFBundlePackageType`, `CFBundleExecutable`, `CFBundleIdentifier`, `CFBundleName`,
    `CFBundleVersion` and `CFBundleInfoDictionaryVersion` are the required set;
    `CFBundleIconFile` is what puts the icon on it, and `NSHighResolutionCapable` is what
    stops the whole window being drawn at 1x and scaled up on a Retina display, which is
    the most visible thing a hand-written bundle gets wrong.
    """
    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" '
        '"http://www.apple.com/DTDs/PropertyList-1.0.dtd">\n'
        '<plist version="1.0">\n'
        "<dict>\n"
        "\t<key>CFBundleInfoDictionaryVersion</key>\n\t<string>6.0</string>\n"
        "\t<key>CFBundlePackageType</key>\n\t<string>APPL</string>\n"
        "\t<key>CFBundleSignature</key>\n\t<string>????</string>\n"
        f"\t<key>CFBundleName</key>\n\t<string>{APP_NAME}</string>\n"
        f"\t<key>CFBundleDisplayName</key>\n\t<string>{APP_NAME}</string>\n"
        f"\t<key>CFBundleExecutable</key>\n\t<string>{APP_NAME}</string>\n"
        f"\t<key>CFBundleIdentifier</key>\n\t<string>{BUNDLE_ID}</string>\n"
        f"\t<key>CFBundleIconFile</key>\n\t<string>{APP_NAME}.icns</string>\n"
        f"\t<key>CFBundleVersion</key>\n\t<string>{version}</string>\n"
        f"\t<key>CFBundleShortVersionString</key>\n\t<string>{version}</string>\n"
        "\t<key>LSMinimumSystemVersion</key>\n\t<string>11.0</string>\n"
        "\t<key>LSApplicationCategoryType</key>\n"
        "\t<string>public.app-category.video</string>\n"
        "\t<key>NSHighResolutionCapable</key>\n\t<true/>\n"
        "</dict>\n"
        "</plist>\n"
    )


def bundle_script(argv: Sequence[str]) -> str:
    """`Contents/MacOS/Subtitler`: the whole executable of the bundle.

    Three things it has to do that a bare `exec` would not:

    * **Put Homebrew back on PATH.** Finder launches through launchd, whose PATH is
      `/usr/bin:/bin:/usr/sbin:/sbin`. ffmpeg lives in `/opt/homebrew/bin` on Apple
      Silicon and `/usr/local/bin` on Intel, so without this the app starts and every run
      fails with "ffmpeg not found" on a machine where ffmpeg plainly works.
    * **Keep a log.** An app launched from Finder has no stdout anyone can see.
    * **Say something when it fails.** Otherwise the icon bounces once and nothing happens,
      which is indistinguishable from a broken install. `osascript` is in the base system.
    """
    return (
        "#!/bin/sh\n"
        "# Generated by `subtitler install-app`. Re-run that command to update it.\n"
        'PATH="/opt/homebrew/bin:/usr/local/bin:$PATH"\n'
        "export PATH\n"
        f'LOG="{MAC_LOG}"\n'
        'mkdir -p "$(dirname "$LOG")"\n'
        "{\n"
        '  echo "=== $(date) ==="\n'
        f"  {_quoted(argv)}\n"
        '} >>"$LOG" 2>&1\n'
        "status=$?\n"
        'if [ "$status" -ne 0 ]; then\n'
        '  osascript -e "display alert \\"Subtitler could not start\\" message \\"It '
        'exited with status $status. The details are in $LOG.\\"" >/dev/null 2>&1\n'
        "fi\n"
        'exit "$status"\n'
    )


def _mac_plan(home: Path, argv: Sequence[str]) -> Plan:
    """`~/Applications/Subtitler.app`, laid out the way Apple documents a bundle.

    The user's own `~/Applications` rather than `/Applications`: writing to the latter
    needs an administrator password, and the point of this command is that it is run once
    by somebody who is not enjoying it.
    """
    app = home / "Applications" / f"{APP_NAME}.app"
    contents = app / "Contents"
    return Plan(
        target=app,
        directories=(contents / "MacOS", contents / "Resources"),
        artifacts=(
            Artifact(contents / "Info.plist", info_plist().encode("utf-8")),
            # Four characters, no newline: the classic type/creator pair, which Launch
            # Services still reads and which costs nothing to get right.
            Artifact(contents / "PkgInfo", b"APPL????"),
            Artifact(
                contents / "MacOS" / APP_NAME, bundle_script(argv).encode("utf-8"), executable=True
            ),
            Artifact(contents / "Resources" / f"{APP_NAME}.icns", icon.icns_bytes()),
        ),
        notes=(
            "macOS will refuse to open it the first time: the bundle is not signed by an "
            'identified developer, so you get "Subtitler cannot be opened". Right-click '
            "(or Control-click) the icon, choose Open, and confirm. That is once, ever.",
            f"If it opens and closes again, the reason is in {MAC_LOG}.",
        ),
    )


# --------------------------------------------------------------------------------------
# Planning and installing
# --------------------------------------------------------------------------------------


def plan(plat: Platform, *, home: Path, argv: Sequence[str] | None = None) -> Plan:
    """What would be written for this machine. Writes nothing."""
    command = list(argv) if argv is not None else launch_argv()
    if plat.is_macos:
        return _mac_plan(home, command)
    if plat.system == "Linux":
        return _linux_plan(home, command)
    raise LauncherError(
        f"install-app knows macOS and Linux; this is {plat.system}. "
        "Run `subtitler gui` from a terminal instead."
    )


def install(plan_: Plan) -> list[Path]:
    """Write the plan. Overwrites, because re-running it is how an upgrade is applied."""
    for directory in plan_.directories:
        directory.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for artifact in plan_.artifacts:
        artifact.path.parent.mkdir(parents=True, exist_ok=True)
        artifact.path.write_bytes(artifact.content)
        if artifact.executable:
            artifact.path.chmod(0o755)
        written.append(artifact.path)
    return written
