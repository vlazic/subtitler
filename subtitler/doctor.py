"""Dependency detection and installation, for macOS and Debian-family Linux.

The design constraint that shapes this whole file: **the primary target is macOS and the
maintainer develops on Linux**, so the mac branch must be exercisable without a Mac. Every
fact about the machine is read through `Platform` (what we are) and `Probe` (what we can
find), both of which tests replace with fakes. Nothing here calls `subprocess` or
`platform.system()` directly at check time.
"""

from __future__ import annotations

import importlib.util
import os
import platform as _platform
import re
import shutil
import subprocess
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from subtitler.media import MIN_FFMPEG


def _module_available(name: str) -> bool:
    """importlib raises on some namespace-package shapes; a missing module is not an error."""
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


# Homebrew's prefix differs by architecture and we never hardcode it, but we do need
# somewhere to look when `brew` is not on PATH at all.
_BREW_CANDIDATES = (Path("/opt/homebrew"), Path("/usr/local"))

OK, WARN, FAIL, SKIP = "ok", "warn", "fail", "skip"

_MARK = {OK: "ok  ", WARN: "warn", FAIL: "MISS", SKIP: "n/a "}


# --------------------------------------------------------------------------------------
# What we are, and how we look
# --------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Platform:
    system: str  # "Darwin" | "Linux" | ...
    machine: str  # "arm64" | "x86_64" | ...
    distro_id: str = ""
    distro_like: str = ""
    brew_prefix: Path | None = None
    rosetta: bool = False

    @property
    def is_macos(self) -> bool:
        return self.system == "Darwin"

    @property
    def is_apple_silicon(self) -> bool:
        return self.is_macos and self.machine in {"arm64", "aarch64"}

    @property
    def is_debian_like(self) -> bool:
        """Match ID and ID_LIKE both.

        The dev machine is Pop!_OS, which is `ID=pop` with `ID_LIKE="ubuntu debian"`.
        Keying only on `ID == "ubuntu"` fails on the very first machine this runs on.
        """
        haystack = f"{self.distro_id} {self.distro_like}".lower()
        return any(name in haystack.split() for name in ("debian", "ubuntu"))

    @property
    def package_manager(self) -> str | None:
        if self.is_macos:
            return "brew" if self.brew_prefix else None
        return "apt" if self.is_debian_like else None

    def describe(self) -> str:
        bits = [self.system, self.machine]
        if self.distro_id:
            bits.append(self.distro_id)
        if self.brew_prefix:
            bits.append(f"brew:{self.brew_prefix}")
        if self.rosetta:
            bits.append("rosetta")
        return " ".join(bits)


@dataclass(frozen=True, slots=True)
class Probe:
    """The only channel through which a check may learn anything about the machine."""

    # Plain function defaults are safe here only because of `slots=True`: a slotted
    # dataclass has no class attribute for the descriptor protocol to bind against, so
    # these stay functions rather than turning into bound methods.
    which: Callable[[str], str | None] = shutil.which
    env: Mapping[str, str] = field(default_factory=lambda: os.environ)
    module_available: Callable[[str], bool] = _module_available
    python_version: tuple[int, int] = field(default_factory=lambda: sys.version_info[:2])

    def output(self, cmd: Sequence[str], *, timeout: int = 30) -> str | None:
        """Run a command and return stdout+stderr, or None if it could not run."""
        try:
            proc = subprocess.run(
                list(cmd), capture_output=True, text=True, check=False, timeout=timeout
            )
        except (OSError, subprocess.SubprocessError):
            return None
        return (proc.stdout or "") + (proc.stderr or "")


def detect_platform(probe: Probe | None = None) -> Platform:
    probe = probe or Probe()
    system = _platform.system()
    machine = _platform.machine()

    if system == "Darwin":
        return Platform(
            system=system,
            machine=machine,
            brew_prefix=_brew_prefix(probe),
            rosetta=_under_rosetta(probe, machine),
        )

    distro_id, distro_like = _os_release()
    return Platform(system=system, machine=machine, distro_id=distro_id, distro_like=distro_like)


def _os_release(path: Path = Path("/etc/os-release")) -> tuple[str, str]:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return "", ""
    values: dict[str, str] = {}
    for line in text.splitlines():
        key, _, value = line.partition("=")
        values[key.strip()] = value.strip().strip('"')
    return values.get("ID", ""), values.get("ID_LIKE", "")


def _brew_prefix(probe: Probe) -> Path | None:
    """Ask brew where it lives; never assume.

    Apple Silicon puts it at /opt/homebrew and Intel at /usr/local, and a Python running
    under Rosetta on an Apple Silicon Mac sees the Intel one. Asking is the only correct
    answer. When `brew` is not on PATH we still probe both, so `doctor` can tell the user
    that Homebrew exists and their shell just never ran `brew shellenv`.
    """
    if probe.which("brew"):
        out = (probe.output(["brew", "--prefix"]) or "").strip().splitlines()
        if out and out[0].startswith("/"):
            return Path(out[0])
    for candidate in _BREW_CANDIDATES:
        if (candidate / "bin" / "brew").exists():
            return candidate
    return None


def _under_rosetta(probe: Probe, machine: str) -> bool:
    """An arm64 Mac running an x86_64 Python reports x86_64 and translates silently.

    Worth catching loudly: mlx-whisper will not work, and `brew --prefix` points at the
    Intel prefix, so the user gets a confusing mix of half-working advice.
    """
    if machine not in {"x86_64", "i386"}:
        return False
    out = probe.output(["sysctl", "-n", "sysctl.proc_translated"])
    return bool(out and out.strip().startswith("1"))


# --------------------------------------------------------------------------------------
# The dependency table
# --------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class CheckResult:
    status: str
    version: str = ""
    detail: str = ""


@dataclass(frozen=True, slots=True)
class Dep:
    key: str
    label: str
    required: bool
    why: str
    check: Callable[[Platform, Probe], CheckResult]
    brew: str | None = None
    apt: str | None = None
    manual: str = ""

    def fix_for(self, plat: Platform) -> str:
        manager = plat.package_manager
        if manager == "brew" and self.brew:
            return f"brew install {self.brew}"
        if manager == "apt" and self.apt:
            return f"sudo apt install -y {self.apt}"
        return self.manual


@dataclass(frozen=True, slots=True)
class DepStatus:
    dep: Dep
    result: CheckResult
    fix: str

    @property
    def blocking(self) -> bool:
        return self.dep.required and self.result.status == FAIL

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.dep.key,
            "label": self.dep.label,
            "required": self.dep.required,
            "status": self.result.status,
            "version": self.result.version,
            "detail": self.result.detail,
            "fix": self.fix,
        }


# ---- individual checks ----------------------------------------------------------------


def _ffmpeg_version(probe: Probe) -> tuple[int, int] | None:
    out = probe.output(["ffmpeg", "-version"])
    if not out:
        return None
    match = re.search(r"ffmpeg version n?(\d+)\.(\d+)", out)
    return (int(match.group(1)), int(match.group(2))) if match else None


def check_ffmpeg(plat: Platform, probe: Probe) -> CheckResult:
    if not probe.which("ffmpeg"):
        return CheckResult(FAIL, detail="not on PATH")
    version = _ffmpeg_version(probe)
    if version is None:
        return CheckResult(WARN, detail="installed, but the version could not be parsed")
    label = f"{version[0]}.{version[1]}"
    if version < MIN_FFMPEG:
        return CheckResult(FAIL, label, f"below the {MIN_FFMPEG[0]}.{MIN_FFMPEG[1]} minimum")
    return CheckResult(OK, label)


def check_ffprobe(plat: Platform, probe: Probe) -> CheckResult:
    return CheckResult(OK) if probe.which("ffprobe") else CheckResult(FAIL, detail="not on PATH")


def has_filter(probe: Probe, name: str) -> bool:
    """Ask ffmpeg about one filter rather than parsing the `-filters` table.

    The table's column layout is not stable across releases: a regex tuned against the
    4.4 output reported `ass` as missing on Homebrew's 8.1 even though the filter was
    plainly there, which the macOS CI job surfaced as a phantom required-dependency
    failure. `-h filter=NAME` prints "Filter NAME" or "Unknown filter 'NAME'." on every
    version, so there is nothing to keep in sync.
    """
    out = probe.output(["ffmpeg", "-hide_banner", "-h", f"filter={name}"]) or ""
    return out.lstrip().startswith(f"Filter {name}")


def has_encoder(probe: Probe, name: str) -> bool:
    out = probe.output(["ffmpeg", "-hide_banner", "-h", f"encoder={name}"]) or ""
    return out.lstrip().startswith(f"Encoder {name}")


def _partition(probe: Probe, names: Sequence[str], test) -> tuple[list[str], list[str]]:
    found, missing = [], []
    for name in names:
        (found if test(probe, name) else missing).append(name)
    return found, missing


def check_libass(plat: Platform, probe: Probe) -> CheckResult:
    if not probe.which("ffmpeg"):
        return CheckResult(SKIP, detail="ffmpeg is missing")
    _, missing = _partition(probe, ["ass", "subtitles"], has_filter)
    if missing:
        return CheckResult(FAIL, detail=f"ffmpeg built without: {', '.join(missing)}")
    return CheckResult(OK, detail="ass, subtitles")


def check_encoders(plat: Platform, probe: Probe) -> CheckResult:
    if not probe.which("ffmpeg"):
        return CheckResult(SKIP, detail="ffmpeg is missing")
    _, missing = _partition(probe, ["libx264", "aac"], has_encoder)
    if missing:
        return CheckResult(FAIL, detail=f"ffmpeg built without: {', '.join(missing)}")
    return CheckResult(OK, detail="libx264, aac")


def check_denoise_filters(plat: Platform, probe: Probe) -> CheckResult:
    if not probe.which("ffmpeg"):
        return CheckResult(SKIP, detail="ffmpeg is missing")
    found, missing = _partition(probe, ["afftdn", "arnndn", "anlmdn"], has_filter)
    if missing:
        return CheckResult(WARN, detail=f"unavailable: {', '.join(missing)}")
    return CheckResult(OK, detail=", ".join(found))


def check_uv(plat: Platform, probe: Probe) -> CheckResult:
    if not probe.which("uv"):
        return CheckResult(FAIL, detail="not on PATH")
    out = (probe.output(["uv", "--version"]) or "").strip()
    return CheckResult(OK, out.replace("uv ", "").split()[0] if out else "")


def check_python(plat: Platform, probe: Probe) -> CheckResult:
    major, minor = probe.python_version
    label = f"{major}.{minor}"
    if (major, minor) < (3, 12):
        return CheckResult(FAIL, label, "3.12 or newer is required")
    return CheckResult(OK, label)


def check_fontconfig(plat: Platform, probe: Probe) -> CheckResult:
    if plat.is_macos:
        # libass uses CoreText on macOS, and the font ships inside the package anyway.
        return CheckResult(SKIP, detail="not used on macOS")
    if not probe.which("fc-match"):
        return CheckResult(WARN, detail="fontconfig missing; only affects --font overrides")
    return CheckResult(OK)


def check_xcode_clt(plat: Platform, probe: Probe) -> CheckResult:
    if not plat.is_macos:
        return CheckResult(SKIP, detail="macOS only")
    out = probe.output(["xcode-select", "-p"])
    if not out or not out.strip().startswith("/"):
        return CheckResult(WARN, detail="command line tools not installed")
    return CheckResult(OK, detail=out.strip().splitlines()[0])


def check_rosetta(plat: Platform, probe: Probe) -> CheckResult:
    if not plat.is_macos:
        return CheckResult(SKIP, detail="macOS only")
    if plat.rosetta:
        return CheckResult(
            WARN,
            detail="this Python is running under Rosetta; mlx will not work",
        )
    return CheckResult(OK, detail=plat.machine)


def check_local_engine(plat: Platform, probe: Probe) -> CheckResult:
    """The engine that should be the default on this machine."""
    if plat.is_apple_silicon:
        if probe.module_available("mlx_whisper"):
            return CheckResult(OK, detail="mlx-whisper")
        return CheckResult(WARN, detail="mlx-whisper is not installed")
    if probe.module_available("faster_whisper"):
        return CheckResult(OK, detail="faster-whisper")
    return CheckResult(WARN, detail="faster-whisper is not installed")


def check_groq(plat: Platform, probe: Probe) -> CheckResult:
    has_pkg = probe.module_available("groq")
    has_key = bool(probe.env.get("GROQ_API_KEY") or probe.env.get("GROQ_API_KEYS"))
    if has_pkg and has_key:
        return CheckResult(OK)
    missing = []
    if not has_pkg:
        missing.append("the groq package is not installed")
    if not has_key:
        missing.append("no API key is set")
    return CheckResult(WARN, detail="; ".join(missing))


DEPS: tuple[Dep, ...] = (
    Dep(
        key="ffmpeg",
        label="ffmpeg",
        required=True,
        why="every stage: probe, extract, denoise, burn-in",
        check=check_ffmpeg,
        brew="ffmpeg",
        apt="ffmpeg",
    ),
    Dep(
        key="ffprobe",
        label="ffprobe",
        required=True,
        why="reads duration and stream layout",
        check=check_ffprobe,
        brew="ffmpeg",
        apt="ffmpeg",
    ),
    Dep(
        key="libass",
        label="ffmpeg libass",
        required=True,
        why="renders the subtitles into the video",
        check=check_libass,
        # Homebrew's regular `ffmpeg` bottle is built WITHOUT libass; `ffmpeg-full` is the
        # one that has it. `brew install ffmpeg` is what everyone types, and it produces an
        # ffmpeg that cannot burn in subtitles, so the fix has to name the right formula.
        brew="ffmpeg-full",
        apt="ffmpeg",
        manual="install an ffmpeg built with libass",
    ),
    Dep(
        key="encoders",
        label="ffmpeg encoders",
        required=True,
        why="libx264 and aac produce the output file",
        check=check_encoders,
        brew="ffmpeg",
        apt="ffmpeg",
        manual="reinstall ffmpeg with libx264 and aac support",
    ),
    Dep(
        key="denoise",
        label="ffmpeg denoise",
        required=False,
        why="the --denoise presets",
        check=check_denoise_filters,
        brew="ffmpeg",
        apt="ffmpeg",
    ),
    Dep(
        key="uv",
        label="uv",
        required=True,
        why="manages the virtualenv and the engine extras",
        check=check_uv,
        brew="uv",
        # There is no apt package for uv, so the fix has to be the installer script.
        apt=None,
        manual="curl -LsSf https://astral.sh/uv/install.sh | sh",
    ),
    Dep(
        key="python",
        label="python",
        required=True,
        why="3.12 or newer",
        check=check_python,
        manual="uv python install 3.12",
    ),
    Dep(
        key="fontconfig",
        label="fontconfig",
        required=False,
        why="resolving a system font passed to --font",
        check=check_fontconfig,
        apt="fontconfig",
    ),
    Dep(
        key="xcode-clt",
        label="Xcode CLT",
        required=False,
        why="building any dependency that has no wheel",
        check=check_xcode_clt,
        manual="xcode-select --install",
    ),
    Dep(
        key="architecture",
        label="architecture",
        required=False,
        why="mlx needs a native arm64 Python",
        check=check_rosetta,
        manual="uv python install 3.12   # installs a native arm64 interpreter",
    ),
    Dep(
        key="engine-local",
        label="local engine",
        required=False,
        why="transcription without uploading anything",
        check=check_local_engine,
        manual="uv sync --extra mlx     # on Apple Silicon\n"
        "       uv sync --extra local   # everywhere else",
    ),
    Dep(
        key="engine-cloud",
        label="cloud engine",
        required=False,
        why="the Groq comparison baseline",
        check=check_groq,
        manual="uv sync --extra cloud, then set GROQ_API_KEY in .env",
    ),
)


# --------------------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------------------


def diagnose(
    plat: Platform | None = None,
    probe: Probe | None = None,
    deps: Sequence[Dep] = DEPS,
) -> list[DepStatus]:
    probe = probe or Probe()
    plat = plat or detect_platform(probe)
    out = []
    for dep in deps:
        result = dep.check(plat, probe)
        fix = dep.fix_for(plat) if result.status in {FAIL, WARN} else ""
        out.append(DepStatus(dep=dep, result=result, fix=fix))
    return out


def render(statuses: Sequence[DepStatus], plat: Platform) -> str:
    lines = [f"platform: {plat.describe()}"]
    manager = plat.package_manager
    if manager is None:
        lines.append("no supported package manager found; fixes below are manual instructions")
    lines.append("")

    width = max(len(s.dep.label) for s in statuses)
    for status in statuses:
        mark = _MARK[status.result.status]
        detail = status.result.detail
        version = status.result.version
        suffix = " ".join(part for part in (version, f"({detail})" if detail else "") if part)
        lines.append(f"  {mark}  {status.dep.label:<{width}}  {suffix}".rstrip())
        if status.fix:
            for i, fix_line in enumerate(status.fix.splitlines()):
                lines.append(f"      {'fix:' if i == 0 else '    '} {fix_line.strip()}")

    blocking = [s for s in statuses if s.blocking]
    lines.append("")
    if blocking:
        lines.append(
            f"{len(blocking)} required dependencies missing. Run: subtitler doctor --install"
        )
    else:
        lines.append("all required dependencies present")
    return "\n".join(lines)


def install_plan(statuses: Sequence[DepStatus], plat: Platform) -> list[list[str]]:
    """Concrete argv commands, de-duplicated and ordered.

    `sudo` is used only for apt, never for brew: running brew under sudo breaks the
    installation in ways that are tedious to undo.
    """
    manager = plat.package_manager
    if manager is None:
        return []

    commands: list[list[str]] = []
    seen: set[tuple[str, ...]] = set()
    for status in statuses:
        if status.result.status not in {FAIL, WARN}:
            continue
        dep = status.dep
        if manager == "brew" and dep.brew:
            cmd = ["brew", "install", dep.brew]
        elif manager == "apt" and dep.apt:
            cmd = ["sudo", "apt-get", "install", "-y", dep.apt]
        else:
            continue  # no package-manager path; the manual instruction was already printed
        key = tuple(cmd)
        if key not in seen:
            seen.add(key)
            commands.append(cmd)
    return commands
