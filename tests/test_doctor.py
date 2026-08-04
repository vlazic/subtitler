"""Both OS branches, exercised on whichever machine happens to run the tests.

macOS is the primary target and the maintainer develops on Linux, so every mac-specific
decision is reachable here through a faked `Platform` and `Probe`. If a check ever reads
the real machine directly, these tests stop meaning anything.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from subtitler.doctor import (
    DEPS,
    FAIL,
    OK,
    SKIP,
    WARN,
    Platform,
    Probe,
    check_ffmpeg,
    check_groq,
    check_libass,
    check_local_engine,
    check_rosetta,
    diagnose,
    has_encoder,
    has_filter,
    install_plan,
    render,
)

FFMPEG_4 = "ffmpeg version 4.4.2-0ubuntu0.22.04.1 Copyright (c) 2000-2021"
FFMPEG_8 = "ffmpeg version 8.0 Copyright (c) 2000-2026 the FFmpeg developers"
FFMPEG_3 = "ffmpeg version 3.4.11-0ubuntu0.1 Copyright (c) 2000-2020"


def _filter_help(*names: str) -> dict[tuple[str, ...], str]:
    """ffmpeg answers `-h filter=NAME` with "Filter NAME" or "Unknown filter 'NAME'."."""
    return {
        ("ffmpeg", "-hide_banner", "-h", f"filter={n}"): f"Filter {n}\n  Some description.\n"
        for n in names
    }


def _encoder_help(*names: str) -> dict[tuple[str, ...], str]:
    return {
        ("ffmpeg", "-hide_banner", "-h", f"encoder={n}"): f"Encoder {n} [desc]:\n" for n in names
    }


CAPS_OK = {
    **_filter_help("ass", "subtitles", "afftdn", "arnndn", "anlmdn"),
    **_encoder_help("libx264", "aac"),
}


def fake_probe(
    *,
    present: set[str] | None = None,
    outputs: dict[tuple[str, ...], str] | None = None,
    modules: set[str] | None = None,
    env: dict[str, str] | None = None,
    python: tuple[int, int] = (3, 12),
) -> Probe:
    present = present if present is not None else set()
    outputs = outputs or {}
    modules = modules or set()

    class _Probe(Probe):
        def output(self, cmd, *, timeout: int = 30):
            return outputs.get(tuple(cmd))

    return _Probe(
        which=lambda name: f"/usr/bin/{name}" if name in present else None,
        env=env or {},
        module_available=lambda name: name in modules,
        python_version=python,
    )


MAC = Platform(system="Darwin", machine="arm64", brew_prefix=Path("/opt/homebrew"))
MAC_ROSETTA = Platform(
    system="Darwin", machine="x86_64", brew_prefix=Path("/usr/local"), rosetta=True
)
POP = Platform(system="Linux", machine="x86_64", distro_id="pop", distro_like="ubuntu debian")
UBUNTU = Platform(system="Linux", machine="x86_64", distro_id="ubuntu", distro_like="debian")
ARCH = Platform(system="Linux", machine="x86_64", distro_id="arch", distro_like="")


class TestPlatform:
    def test_pop_os_is_debian_like(self) -> None:
        """The dev machine is ID=pop with ID_LIKE="ubuntu debian". Matching only on ID
        would fail on the very first machine this ever runs on."""
        assert POP.is_debian_like
        assert POP.package_manager == "apt"

    def test_ubuntu_is_debian_like(self) -> None:
        assert UBUNTU.is_debian_like

    def test_arch_has_no_package_manager(self) -> None:
        assert not ARCH.is_debian_like
        assert ARCH.package_manager is None

    def test_mac_uses_brew_only_when_a_prefix_was_found(self) -> None:
        assert MAC.package_manager == "brew"
        assert Platform(system="Darwin", machine="arm64").package_manager is None

    def test_apple_silicon_detection(self) -> None:
        assert MAC.is_apple_silicon
        assert not MAC_ROSETTA.is_apple_silicon
        assert not POP.is_apple_silicon

    def test_describe_mentions_rosetta_and_prefix(self) -> None:
        text = MAC_ROSETTA.describe()
        assert "rosetta" in text
        assert "/usr/local" in text


class TestFfmpegVersion:
    def test_accepts_4_4(self) -> None:
        probe = fake_probe(present={"ffmpeg"}, outputs={("ffmpeg", "-version"): FFMPEG_4})
        result = check_ffmpeg(POP, probe)
        assert result.status == OK
        assert result.version == "4.4"

    def test_accepts_homebrew_8(self) -> None:
        probe = fake_probe(present={"ffmpeg"}, outputs={("ffmpeg", "-version"): FFMPEG_8})
        assert check_ffmpeg(MAC, probe).status == OK

    def test_rejects_below_the_floor(self) -> None:
        probe = fake_probe(present={"ffmpeg"}, outputs={("ffmpeg", "-version"): FFMPEG_3})
        result = check_ffmpeg(POP, probe)
        assert result.status == FAIL
        assert "minimum" in result.detail

    def test_missing_binary(self) -> None:
        assert check_ffmpeg(POP, fake_probe()).status == FAIL


class TestFfmpegCapabilities:
    def test_libass_present(self) -> None:
        probe = fake_probe(present={"ffmpeg"}, outputs=CAPS_OK)
        assert check_libass(MAC, probe).status == OK

    def test_libass_missing_is_a_hard_fail(self) -> None:
        """An ffmpeg without libass installs fine and then fails at burn time."""
        probe = fake_probe(present={"ffmpeg"}, outputs=_filter_help("afftdn"))
        result = check_libass(MAC, probe)
        assert result.status == FAIL
        assert "ass" in result.detail

    def test_capability_checks_skip_when_ffmpeg_is_absent(self) -> None:
        """Reporting 'libass missing' when ffmpeg itself is missing is noise."""
        assert check_libass(POP, fake_probe()).status == SKIP

    def test_probe_asks_about_one_filter_at_a_time(self) -> None:
        """Regression: parsing the `-filters` table with a regex tuned against ffmpeg 4.4
        reported `ass` as missing on Homebrew's 8.1, where the filter was plainly present.
        The column layout is not stable across releases; `-h filter=NAME` is."""
        assert has_filter(fake_probe(outputs=_filter_help("ass")), "ass")
        assert not has_filter(
            fake_probe(
                outputs={("ffmpeg", "-hide_banner", "-h", "filter=ass"): "Unknown filter 'ass'."}
            ),
            "ass",
        )
        assert not has_filter(fake_probe(), "ass")  # ffmpeg could not run at all

    def test_encoder_probe(self) -> None:
        assert has_encoder(fake_probe(outputs=_encoder_help("libx264")), "libx264")
        assert not has_encoder(
            fake_probe(
                outputs={
                    ("ffmpeg", "-hide_banner", "-h", "encoder=libx264"): (
                        "Codec 'libx264' is not recognized by FFmpeg."
                    )
                }
            ),
            "libx264",
        )


class TestRosetta:
    def test_native_arm_is_fine(self) -> None:
        assert check_rosetta(MAC, fake_probe()).status == OK

    def test_translated_python_warns(self) -> None:
        result = check_rosetta(MAC_ROSETTA, fake_probe())
        assert result.status == WARN
        assert "mlx" in result.detail

    def test_skipped_on_linux(self) -> None:
        assert check_rosetta(POP, fake_probe()).status == SKIP


class TestEngines:
    def test_apple_silicon_wants_mlx(self) -> None:
        assert check_local_engine(MAC, fake_probe(modules={"mlx_whisper"})).status == OK
        missing = check_local_engine(MAC, fake_probe(modules={"faster_whisper"}))
        assert missing.status == WARN
        assert "mlx-whisper" in missing.detail

    def test_linux_wants_faster_whisper(self) -> None:
        assert check_local_engine(POP, fake_probe(modules={"faster_whisper"})).status == OK
        missing = check_local_engine(POP, fake_probe(modules={"mlx_whisper"}))
        assert missing.status == WARN
        assert "faster-whisper" in missing.detail

    def test_groq_needs_both_package_and_key(self) -> None:
        assert check_groq(POP, fake_probe(modules={"groq"}, env={"GROQ_API_KEY": "x"})).status == OK
        assert check_groq(POP, fake_probe(modules={"groq"})).status == WARN
        assert check_groq(POP, fake_probe(env={"GROQ_API_KEY": "x"})).status == WARN

    def test_groq_accepts_the_key_pool(self) -> None:
        probe = fake_probe(modules={"groq"}, env={"GROQ_API_KEYS": "a,b,c"})
        assert check_groq(POP, probe).status == OK


class TestFixes:
    def _statuses(self, plat: Platform):
        return diagnose(plat, fake_probe(), DEPS)

    def test_mac_fixes_use_brew_without_sudo(self) -> None:
        """brew under sudo breaks the installation in ways that are tedious to undo."""
        commands = install_plan(self._statuses(MAC), MAC)
        assert commands
        assert all(cmd[0] == "brew" for cmd in commands)
        assert not any("sudo" in cmd for cmd in commands)

    def test_linux_fixes_use_apt_with_sudo(self) -> None:
        commands = install_plan(self._statuses(POP), POP)
        assert commands
        assert all(cmd[0] == "sudo" for cmd in commands)

    def test_commands_are_deduplicated(self) -> None:
        """ffmpeg, ffprobe, libass and the encoders are all one package."""
        commands = install_plan(self._statuses(POP), POP)
        assert len(commands) == len({tuple(c) for c in commands})

    def test_distro_without_a_package_manager_gets_no_plan(self) -> None:
        assert install_plan(self._statuses(ARCH), ARCH) == []

    def test_uv_falls_back_to_the_installer_script_on_linux(self) -> None:
        """There is no apt package for uv, so the fix has to be the curl installer."""
        uv_dep = next(d for d in DEPS if d.key == "uv")
        assert "astral.sh" in uv_dep.fix_for(POP)
        assert uv_dep.fix_for(MAC) == "brew install uv"


class TestDiagnose:
    def test_a_healthy_linux_box_reports_no_blockers(self) -> None:
        probe = fake_probe(
            present={"ffmpeg", "ffprobe", "uv", "fc-match"},
            outputs={
                ("ffmpeg", "-version"): FFMPEG_4,
                **CAPS_OK,
                ("uv", "--version"): "uv 0.8.15",
            },
            modules={"faster_whisper", "groq"},
            env={"GROQ_API_KEY": "x"},
        )
        statuses = diagnose(POP, probe, DEPS)
        assert not [s for s in statuses if s.blocking]

    def test_a_bare_mac_reports_ffmpeg_as_blocking(self) -> None:
        statuses = diagnose(MAC, fake_probe(), DEPS)
        blocking = {s.dep.key for s in statuses if s.blocking}
        assert "ffmpeg" in blocking
        assert "uv" in blocking

    def test_render_is_readable_and_names_the_platform(self) -> None:
        text = render(diagnose(MAC_ROSETTA, fake_probe(), DEPS), MAC_ROSETTA)
        assert "Darwin" in text
        assert "rosetta" in text
        assert "brew install ffmpeg" in text

    @pytest.mark.parametrize("plat", [MAC, MAC_ROSETTA, POP, UBUNTU, ARCH])
    def test_every_platform_renders_without_raising(self, plat: Platform) -> None:
        render(diagnose(plat, fake_probe(), DEPS), plat)
