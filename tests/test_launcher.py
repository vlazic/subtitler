"""The double-clickable launcher, and the icon it carries.

The macOS half is the point of this file. There is no Mac here, so the `.app` bundle is
built against a faked `Platform` and inspected byte by byte: the structure Apple documents,
the keys `Info.plist` is required to have, and the two things a Finder-launched app needs
that a shell-launched one does not (a PATH with Homebrew in it, and somewhere for its output
to go). Launching it is the one thing that cannot be tested from here, and is stated as
unverified in the README rather than implied to work.
"""

from __future__ import annotations

import plistlib
import struct
import zlib
from pathlib import Path

import pytest

from subtitler import icon, launcher
from subtitler.doctor import Platform

MAC = Platform(system="Darwin", machine="arm64", brew_prefix=Path("/opt/homebrew"))
LINUX = Platform(system="Linux", machine="x86_64", distro_id="pop", distro_like="ubuntu debian")
WINDOWS = Platform(system="Windows", machine="AMD64")

ARGV = ["/home/friend/.local/bin/subtitler", "gui"]


# --------------------------------------------------------------------------------------
# What the icon runs
# --------------------------------------------------------------------------------------


class TestLaunchArgv:
    def test_it_prefers_the_console_script_next_to_the_interpreter(self) -> None:
        argv = launcher.launch_argv(
            executable=Path("/opt/venv/bin/python3"), exists=lambda p: p.name == "subtitler"
        )
        assert argv == ["/opt/venv/bin/subtitler", "gui"]

    def test_a_checkout_with_no_console_script_still_gets_a_working_command(self) -> None:
        argv = launcher.launch_argv(
            executable=Path("/opt/venv/bin/python3"), exists=lambda p: False
        )
        assert argv == ["/opt/venv/bin/python3", "-m", "subtitler.cli", "gui"]

    def test_the_command_is_absolute(self) -> None:
        """A `.desktop` entry and a `.app` bundle are launched with a PATH the user never
        set, so a bare `subtitler` works from a shell and fails from an icon."""
        for argv in (
            launcher.launch_argv(executable=Path("/opt/venv/bin/python3"), exists=lambda p: True),
            launcher.launch_argv(executable=Path("/opt/venv/bin/python3"), exists=lambda p: False),
        ):
            assert Path(argv[0]).is_absolute()


# --------------------------------------------------------------------------------------
# Linux
# --------------------------------------------------------------------------------------


class TestDesktopEntry:
    def entry(self) -> dict[str, str]:
        text = launcher.desktop_entry(ARGV)
        assert text.startswith("[Desktop Entry]\n")
        return dict(line.split("=", 1) for line in text.splitlines()[1:] if line and "=" in line)

    def test_it_carries_the_keys_the_specification_requires(self) -> None:
        keys = self.entry()
        assert keys["Type"] == "Application"
        assert keys["Name"] == "Subtitler"
        assert keys["Exec"] == "/home/friend/.local/bin/subtitler gui"
        assert keys["Icon"] == "subtitler"

    def test_it_never_opens_a_terminal(self) -> None:
        """The whole exercise is for the person who does not open one."""
        assert self.entry()["Terminal"] == "false"

    def test_a_path_with_a_space_in_it_is_quoted(self) -> None:
        """An unquoted Exec would launch `/home/my` with `files/bin/subtitler` as an
        argument, and the icon would do nothing at all."""
        keys = dict(
            line.split("=", 1)
            for line in launcher.desktop_entry(["/home/my files/bin/subtitler", "gui"]).splitlines()
            if "=" in line
        )
        assert keys["Exec"] == "'/home/my files/bin/subtitler' gui"

    def test_the_plan_writes_the_entry_and_the_icon_theme(self, tmp_path: Path) -> None:
        plan = launcher.plan(LINUX, home=tmp_path, argv=ARGV)
        launcher.install(plan)
        entry = tmp_path / ".local" / "share" / "applications" / "subtitler.desktop"
        assert entry.is_file()
        assert plan.target == entry
        for size in icon.PNG_SIZES:
            png = tmp_path / ".local/share/icons/hicolor" / f"{size}x{size}" / "apps/subtitler.png"
            assert png.is_file()


# --------------------------------------------------------------------------------------
# macOS
# --------------------------------------------------------------------------------------


class TestBundle:
    def built(self, tmp_path: Path) -> Path:
        launcher.install(launcher.plan(MAC, home=tmp_path, argv=ARGV))
        return tmp_path / "Applications" / "Subtitler.app"

    def test_the_layout_is_the_one_apple_documents(self, tmp_path: Path) -> None:
        app = self.built(tmp_path)
        assert (app / "Contents" / "Info.plist").is_file()
        assert (app / "Contents" / "MacOS" / "Subtitler").is_file()
        assert (app / "Contents" / "Resources" / "Subtitler.icns").is_file()
        assert (app / "Contents" / "PkgInfo").read_bytes() == b"APPL????"

    def test_the_plist_parses_and_carries_the_required_keys(self, tmp_path: Path) -> None:
        """Parsed with `plistlib` rather than grepped: a plist this project hand-writes and
        never opens on a Mac has to be proven well-formed here or nowhere."""
        data = plistlib.loads((self.built(tmp_path) / "Contents" / "Info.plist").read_bytes())
        assert data["CFBundlePackageType"] == "APPL"
        assert data["CFBundleExecutable"] == "Subtitler"
        assert data["CFBundleIdentifier"] == launcher.BUNDLE_ID
        assert data["CFBundleInfoDictionaryVersion"] == "6.0"
        assert data["CFBundleVersion"]
        assert data["CFBundleIconFile"] == "Subtitler.icns"

    def test_the_executable_named_by_the_plist_is_the_one_that_exists(self, tmp_path: Path) -> None:
        """Launch Services runs `Contents/MacOS/<CFBundleExecutable>` and nothing else; a
        mismatch is an app that bounces once and dies with no message anywhere."""
        app = self.built(tmp_path)
        data = plistlib.loads((app / "Contents" / "Info.plist").read_bytes())
        assert (app / "Contents" / "MacOS" / data["CFBundleExecutable"]).is_file()

    def test_the_executable_is_executable(self, tmp_path: Path) -> None:
        script = self.built(tmp_path) / "Contents" / "MacOS" / "Subtitler"
        assert script.stat().st_mode & 0o111

    def test_it_puts_homebrew_back_on_the_path(self) -> None:
        """launchd hands a Finder-launched app `/usr/bin:/bin:/usr/sbin:/sbin`. ffmpeg is
        in `/opt/homebrew/bin`, so without this the bundle starts and every run fails with
        "ffmpeg not found" on a machine where ffmpeg plainly works from a shell."""
        script = launcher.bundle_script(ARGV)
        assert "/opt/homebrew/bin" in script
        assert "/usr/local/bin" in script

    def test_it_keeps_a_log_and_says_something_when_it_fails(self) -> None:
        """An app launched from Finder has no stdout anyone can see, so a crash is an icon
        that bounces once. `osascript` is in the base system."""
        script = launcher.bundle_script(ARGV)
        assert "Library/Logs/Subtitler.log" in script
        assert "osascript" in script

    def test_the_script_starts_with_a_shebang_and_runs_the_command(self) -> None:
        script = launcher.bundle_script(ARGV)
        assert script.startswith("#!/bin/sh\n")
        assert "/home/friend/.local/bin/subtitler gui" in script

    def test_the_gatekeeper_consequence_is_stated_and_not_glossed_over(self) -> None:
        """An unsigned bundle is refused on first launch. The user has to be told, or the
        one dialog they cannot get past is the first thing this project shows them."""
        notes = " ".join(launcher.plan(MAC, home=Path("/Users/friend"), argv=ARGV).notes)
        assert "not signed" in notes
        assert "Right-click" in notes or "right-click" in notes


class TestUnsupported:
    def test_a_platform_with_no_launcher_says_so_instead_of_writing_nonsense(self) -> None:
        with pytest.raises(launcher.LauncherError) as exc:
            launcher.plan(WINDOWS, home=Path("/home/x"), argv=ARGV)
        assert "Windows" in str(exc.value)


# --------------------------------------------------------------------------------------
# The icon
# --------------------------------------------------------------------------------------


class TestIcon:
    def test_the_png_is_a_valid_png_of_the_size_asked_for(self) -> None:
        data = icon.png_bytes(64)
        assert data.startswith(b"\x89PNG\r\n\x1a\n")
        _length, kind = struct.unpack(">I4s", data[8:16])
        assert kind == b"IHDR"
        width, height, depth, colour = struct.unpack(">IIBB", data[16 : 16 + 10])
        assert (width, height, depth, colour) == (64, 64, 8, 6)
        assert data.endswith(b"IEND\xae\x42\x60\x82")

    def test_every_chunk_checksum_is_right(self) -> None:
        """A PNG with a bad CRC is rejected outright by strict decoders, and the encoder
        here is hand-written."""
        data = icon.png_bytes(32)
        offset = 8
        seen = []
        while offset < len(data):
            (length,) = struct.unpack(">I", data[offset : offset + 4])
            kind = data[offset + 4 : offset + 8]
            payload = data[offset + 8 : offset + 8 + length]
            (crc,) = struct.unpack(">I", data[offset + 8 + length : offset + 12 + length])
            assert crc == zlib.crc32(kind + payload) & 0xFFFFFFFF
            seen.append(kind)
            offset += 12 + length
        assert seen == [b"IHDR", b"IDAT", b"IEND"]

    def test_the_icon_is_not_a_blank_square(self) -> None:
        """The drawing is analytic, so an off-by-one in the coverage function would produce
        a plausible file with nothing in it."""
        raw = icon.pixels(64)
        colours = {raw[i : i + 4] for i in range(0, len(raw), 4)}
        assert len(colours) > 20
        # A transparent corner and an opaque middle: the rounded rectangle is really round.
        assert raw[3] == 0
        middle = (32 * 64 + 32) * 4
        assert raw[middle + 3] == 255

    def test_the_icns_header_declares_its_own_length(self) -> None:
        """The header is a total byte count including itself, and a wrong one makes the
        whole file unreadable to Finder."""
        data = icon.icns_bytes()
        assert data[:4] == b"icns"
        assert struct.unpack(">I", data[4:8])[0] == len(data)

    def test_every_icns_entry_is_a_declared_type_holding_a_png(self) -> None:
        data = icon.icns_bytes()
        offset, seen = 8, []
        while offset < len(data):
            kind = data[offset : offset + 4]
            (length,) = struct.unpack(">I", data[offset + 4 : offset + 8])
            payload = data[offset + 8 : offset + length]
            assert payload.startswith(b"\x89PNG\r\n\x1a\n"), kind
            seen.append(kind)
            offset += length
        assert seen == [kind for kind, _ in icon.ICNS_TYPES]
