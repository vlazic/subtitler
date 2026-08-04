"""Both OS branches, exercised on whichever machine happens to run the tests.

macOS is the primary target and the maintainer develops on Linux, so every mac-specific
decision is reachable here through a faked `Platform` and `Probe`. If a check ever reads
the real machine directly, these tests stop meaning anything.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from subtitler.doctor import (
    CUDA_PROBE_ARGV,
    DEPS,
    FAIL,
    NVIDIA_SMI_QUERY,
    OK,
    SKIP,
    WARN,
    Platform,
    Probe,
    check_cuda_runtime,
    check_ffmpeg,
    check_gpu,
    check_groq,
    check_libass,
    check_local_engine,
    check_rosetta,
    check_tkinter,
    diagnose,
    has_encoder,
    has_filter,
    install_plan,
    render,
    tk_importable,
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
    tk: bool = True,
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
        # Faked like everything else, and defaulted to present, so that the rest of this
        # file reports the same thing on a machine with Tk and a machine without it.
        tk_importable=lambda: tk,
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


def cuda_probe_output(**fields) -> dict[tuple[str, ...], str]:
    """What `subtitler.engines.faster.cuda_report()` prints, as the probe would see it."""
    report = {
        "ctranslate2": "4.8.1",
        "devices": 1,
        "packages": ["cublas", "cudnn"],
        "cublas12": True,
        "cudnn9": True,
        "usable": True,
        "error": "",
    }
    report.update(fields)
    # CTranslate2 logs to stderr and `Probe.output` concatenates both streams, so the real
    # thing is never a bare JSON document. Reproduce that here or the parser is untested.
    return {
        CUDA_PROBE_ARGV: "[info] Using CUDA allocator: cuda_malloc_async\n" + json.dumps(report)
    }


SMI_3090 = {NVIDIA_SMI_QUERY: "NVIDIA GeForce RTX 3090, 580.159.03, 24576\n"}


class TestGpu:
    """Every one of these must come out `n/a` or `warn`, never `fail`.

    CI has no GPU on either runner and the primary target is a Mac. A GPU check that can
    block `subtitler doctor` is a check that turns both of those machines red.
    """

    def test_a_real_card_is_reported_with_its_vram(self) -> None:
        result = check_gpu(POP, fake_probe(present={"nvidia-smi"}, outputs=SMI_3090))
        assert result.status == OK
        assert "580.159.03" in result.version
        assert "RTX 3090" in result.detail
        assert "24576" in result.detail

    def test_no_driver_is_not_a_failure(self) -> None:
        result = check_gpu(POP, fake_probe())
        assert result.status == SKIP
        assert "CPU" in result.detail

    def test_macos_is_skipped_without_running_anything(self) -> None:
        assert check_gpu(MAC, fake_probe(present={"nvidia-smi"})).status == SKIP
        assert check_cuda_runtime(MAC, fake_probe(present={"nvidia-smi"})).status == SKIP

    def test_driver_tools_present_but_no_card(self) -> None:
        probe = fake_probe(present={"nvidia-smi"}, outputs={NVIDIA_SMI_QUERY: "\n"})
        assert check_gpu(POP, probe).status == SKIP

    def test_a_working_runtime_reports_float16(self) -> None:
        probe = fake_probe(
            present={"nvidia-smi"},
            outputs={**SMI_3090, **cuda_probe_output()},
            modules={"ctranslate2"},
        )
        result = check_cuda_runtime(POP, probe)
        assert result.status == OK
        assert "float16" in result.detail

    def test_the_cuda_11_vs_12_trap_warns_and_names_the_library(self) -> None:
        """The whole reason this check exists. A machine can have a current driver and a
        system CUDA toolkit that is 11.x, and CTranslate2's wheels want 12. The symptom is
        `libcublas.so.12: cannot open shared object file` at model load time, minutes into
        a run, with nothing pointing at `uv sync --extra cuda`."""
        probe = fake_probe(
            present={"nvidia-smi"},
            outputs={**SMI_3090, **cuda_probe_output(cublas12=False, usable=False)},
            modules={"ctranslate2"},
        )
        result = check_cuda_runtime(POP, probe)
        assert result.status == WARN
        assert "libcublas.so.12" in result.detail
        assert "CPU" in result.detail
        cuda_dep = next(d for d in DEPS if d.key == "cuda")
        assert "--extra cuda" in cuda_dep.fix_for(POP)

    def test_a_gpu_the_engine_cannot_see_is_not_a_warning(self) -> None:
        """A card the driver shows but CTranslate2 does not is n/a, not a problem: an
        integrated display adapter is the common case and nothing is broken."""
        probe = fake_probe(
            present={"nvidia-smi"},
            outputs={**SMI_3090, **cuda_probe_output(devices=0, usable=False)},
            modules={"ctranslate2"},
        )
        assert check_cuda_runtime(POP, probe).status == SKIP

    def test_no_faster_whisper_means_nothing_to_report(self) -> None:
        probe = fake_probe(present={"nvidia-smi"}, outputs=SMI_3090)
        assert check_cuda_runtime(POP, probe).status == SKIP

    def test_a_probe_that_says_nothing_warns_rather_than_claiming_cuda(self) -> None:
        probe = fake_probe(
            present={"nvidia-smi"},
            outputs={**SMI_3090, CUDA_PROBE_ARGV: "Traceback (most recent call last):\n"},
            modules={"ctranslate2"},
        )
        assert check_cuda_runtime(POP, probe).status == WARN

    def test_neither_gpu_check_can_ever_block(self) -> None:
        for plat in (MAC, MAC_ROSETTA, POP, UBUNTU, ARCH):
            statuses = diagnose(plat, fake_probe(present={"nvidia-smi"}), DEPS)
            blocking = {s.dep.key for s in statuses if s.blocking}
            assert "gpu" not in blocking
            assert "cuda" not in blocking


class TestNativeWindow:
    """The one dependency whose absence is a downgrade rather than a failure.

    Without Tk, `subtitler gui` opens the browser page instead of the desktop window, so
    everything still works. That is why this check may never turn `doctor` red.

    It names the formula anyway. The documented setup goes through uv, whose interpreters
    bundle Tk, so a machine reaching this branch got its Python somewhere else - a Homebrew
    `python@3.12` or a Debian system Python - and the unhandled symptom there is
    `ImportError: No module named '_tkinter'`, which names nothing anybody can act on.
    """

    def test_an_interpreter_with_tk_is_simply_ok(self) -> None:
        assert check_tkinter(MAC, fake_probe(tk=True)).status == OK

    def test_a_homebrew_python_without_tk_warns_and_is_never_blocking(self) -> None:
        statuses = diagnose(MAC, fake_probe(tk=False), DEPS)
        tk_status = next(s for s in statuses if s.dep.key == "tkinter")
        assert tk_status.result.status == WARN
        assert tk_status.blocking is False

    def test_the_warning_names_the_command_for_this_platform(self) -> None:
        """A user reading it has no window, so the sentence has to be self-contained and
        has to be right for the machine they are on."""
        tk_dep = next(d for d in DEPS if d.key == "tkinter")
        assert tk_dep.fix_for(MAC) == "brew install python-tk@3.12"
        assert tk_dep.fix_for(POP) == "sudo apt install -y python3-tk"
        # No package manager at all still gets an instruction rather than an empty line.
        assert "Tk" in tk_dep.fix_for(ARCH)

    def test_the_report_says_what_happens_instead_of_the_window(self) -> None:
        text = render(diagnose(MAC, fake_probe(tk=False), DEPS), MAC)
        assert "brew install python-tk@3.12" in text
        assert "browser" in text

    def test_a_missing_tk_never_sends_anyone_to_the_installer(self) -> None:
        """`doctor` prints "N required dependencies missing" only for blockers, and a
        machine whose only complaint is Tk has a perfectly working install."""
        healthy = fake_probe(
            present={"ffmpeg", "ffprobe", "uv", "fc-match"},
            outputs={
                ("ffmpeg", "-version"): FFMPEG_4,
                **CAPS_OK,
                ("uv", "--version"): "uv 0.8.15",
            },
            modules={"faster_whisper", "groq"},
            env={"GROQ_API_KEY": "x"},
            tk=False,
        )
        statuses = diagnose(POP, healthy, DEPS)
        assert not [s for s in statuses if s.blocking]
        assert "all required dependencies present" in render(statuses, POP)

    def test_the_real_probe_answers_by_importing_rather_than_by_finding_the_spec(self) -> None:
        """`find_spec("tkinter")` says yes on a build with no `_tkinter`, which is exactly
        the build the question exists for. Asserted against the real import."""
        expected = True
        try:
            import tkinter  # noqa: F401
        except Exception:
            expected = False
        assert tk_importable() is expected
        assert Probe().tk_importable() is expected


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
