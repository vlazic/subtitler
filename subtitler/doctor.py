"""Dependency detection and installation, for macOS and Debian-family Linux.

The design constraint that shapes this whole file: **the primary target is macOS and the
maintainer develops on Linux**, so the mac branch must be exercisable without a Mac. Every
fact about the machine is read through `Platform` (what we are) and `Probe` (what we can
find), both of which tests replace with fakes. Nothing here calls `subprocess` or
`platform.system()` directly at check time.
"""

from __future__ import annotations

import importlib.util
import json
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


def tk_importable() -> bool:
    """Whether this interpreter can build a native window.

    A real import and **not** `_module_available("tkinter")`. `tkinter` is a pure-Python
    package that ships with every CPython source tree, so `find_spec` finds it on exactly
    the builds this question exists for; what a Tk-less build lacks is the compiled
    `_tkinter` extension that package imports on its first line.

    Importing it opens no display and creates no window - only `Tk()` connects to one - so
    this is safe on a headless box and on both CI runners.
    """
    try:
        import tkinter  # noqa: F401
    except Exception:
        # Deliberately not just ImportError: a half-installed Tcl/Tk fails inside the
        # extension's own initialisation, and the answer for the caller is the same.
        return False
    return True


_TK_IMPORTABLE = tk_importable


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
    # Its own entry rather than `module_available("tkinter")`, because the two answer
    # different questions on the interpreter that matters. See `tk_importable`. The
    # default is spelled through a module-level alias so the class attribute of the same
    # name cannot be mistaken for a recursive reference to itself.
    tk_importable: Callable[[], bool] = _TK_IMPORTABLE

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


def check_tkinter(plat: Platform, probe: Probe) -> CheckResult:
    """Whether `subtitler gui` opens the native window or falls back to the browser page.

    This is a **warning and never a failure**, and the reason is the whole shape of the GUI:
    a machine without Tk still gets a working interface, just the browser one. Nothing is
    broken, so nothing here may turn `doctor` red or send anyone to `--install`.

    Tk is a property of the interpreter *build*, not of this project, and there is nothing
    a dependency list could do about it: `_tkinter` is a C extension compiled into the
    interpreter against the system Tcl/Tk, so it has no PyPI package, and every toolkit
    that does (PySide6, wxPython, pywebview) is a compiled wheel and therefore forbidden by
    non-negotiable 6. The answer is to detect it and say the right sentence.

    Verified rather than assumed, because the numbers decide how loud this should be:
    `.python-version` pins 3.12 and `make setup` goes through uv, whose
    python-build-standalone distributions bundle Tk 8.6 on macOS and Linux both, and the
    python.org installer ships it too. So the documented setup already has a window, on the
    primary target included. What is left is a Homebrew `python@3.12` (which does not pull
    `python-tk@3.12`) and a Debian or Ubuntu system Python: real machines, reached by
    bypassing the documented setup, and each one formula away from a native window.
    """
    if probe.tk_importable():
        return CheckResult(OK, detail="the native window is available")
    return CheckResult(
        WARN,
        detail="this Python has no tkinter, so `subtitler gui` opens the browser page instead",
    )


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


# --------------------------------------------------------------------------------------
# The GPU, which most machines running this do not have
# --------------------------------------------------------------------------------------
#
# Both checks below degrade to `n/a`, never to a failure. macOS is the primary target and
# has no CUDA at all; both CI runners are CPU-only. A GPU report that can fail CI is a
# report nobody can keep green.

NVIDIA_SMI_QUERY = (
    "nvidia-smi",
    "--query-gpu=name,driver_version,memory.total",
    "--format=csv,noheader,nounits",
)

# Run out-of-process on purpose: answering "can CTranslate2 use this device" means
# dlopening several hundred megabytes of CUDA libraries with RTLD_GLOBAL, and `doctor`
# should not do that to itself. Going through `Probe.output` also keeps it fakeable.
CUDA_PROBE_ARGV = (
    sys.executable,
    "-c",
    "import json; from subtitler.engines.faster import cuda_report; print(json.dumps(cuda_report()))",
)


def _no_nvidia(plat: Platform, probe: Probe) -> CheckResult | None:
    """The shared "there is nothing to report here" answer, or None to carry on."""
    if plat.is_macos:
        return CheckResult(SKIP, detail="no CUDA on macOS; mlx uses the Apple GPU instead")
    if not probe.which("nvidia-smi"):
        return CheckResult(SKIP, detail="no NVIDIA driver found; transcription runs on the CPU")
    return None


def check_gpu(plat: Platform, probe: Probe) -> CheckResult:
    """Which NVIDIA GPU this is, and how much VRAM it has."""
    absent = _no_nvidia(plat, probe)
    if absent is not None:
        return absent
    out = probe.output(list(NVIDIA_SMI_QUERY))
    line = (out or "").strip().splitlines()[0] if (out or "").strip() else ""
    parts = [p.strip() for p in line.split(",")]
    if len(parts) < 3:
        return CheckResult(SKIP, detail="nvidia-smi is installed but reported no GPU")
    name, driver, memory = parts[0], parts[1], parts[2]
    return CheckResult(OK, f"driver {driver}", f"{name}, {memory} MiB")


def _parse_json_line(text: str) -> dict[str, Any] | None:
    """CTranslate2 logs to stderr, and `Probe.output` concatenates both streams.

    So the payload is whichever line parses, not the first line and not the whole blob.
    """
    for line in reversed(text.splitlines()):
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            parsed = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def check_cuda_runtime(plat: Platform, probe: Probe) -> CheckResult:
    """Whether faster-whisper will actually decode on the GPU here.

    The trap this exists to name: CTranslate2's wheels link against CUDA **12**, and a
    machine can have a perfectly good driver plus a system toolkit that is 11.x. The
    driver is not the problem, the runtime libraries are, and the symptom is
    `libcublas.so.12: cannot open shared object file` at model load time with no hint that
    `uv sync --extra cuda` is the fix.
    """
    absent = _no_nvidia(plat, probe)
    if absent is not None:
        return absent
    if not probe.module_available("ctranslate2"):
        return CheckResult(SKIP, detail="faster-whisper is not installed")

    report = _parse_json_line(probe.output(list(CUDA_PROBE_ARGV), timeout=180) or "")
    if report is None:
        return CheckResult(WARN, detail="the CUDA probe produced no answer")
    if report.get("error"):
        return CheckResult(WARN, detail=str(report["error"]))
    if not report.get("devices"):
        return CheckResult(SKIP, detail="CTranslate2 sees no CUDA device")
    if report.get("usable"):
        return CheckResult(OK, detail="CTranslate2 will decode on the GPU in float16")

    missing = [
        name
        for name, key in (("libcublas.so.12", "cublas12"), ("libcudnn.so.9", "cudnn9"))
        if not report.get(key)
    ]
    detail = f"a GPU is present but {', '.join(missing) or 'the CUDA runtime'} will not load"
    return CheckResult(WARN, detail=f"{detail}; transcription falls back to the CPU")


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
        key="tkinter",
        label="native window",
        # Never required. Without it `subtitler gui` opens the browser page, which is a
        # working GUI, so a missing Tk is a downgrade and not a broken install.
        required=False,
        why="the desktop window; without it `subtitler gui` opens the browser page",
        check=check_tkinter,
        # Versioned on purpose: `python-tk@3.12` is the formula that adds Tk to Homebrew's
        # `python@3.12`, which is the interpreter `.python-version` pins.
        brew="python-tk@3.12",
        apt="python3-tk",
        manual="install your Python distribution's Tk bindings, "
        "or use the uv-managed Python `make setup` installs, which bundles them",
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
        key="gpu",
        label="nvidia gpu",
        required=False,
        why="faster-whisper decodes about 16x faster on CUDA than on the CPU",
        check=check_gpu,
    ),
    Dep(
        key="cuda",
        label="cuda runtime",
        required=False,
        why="CTranslate2 needs CUDA 12 libraries, whatever the system toolkit is",
        check=check_cuda_runtime,
        manual="uv sync --extra local --extra cuda",
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
