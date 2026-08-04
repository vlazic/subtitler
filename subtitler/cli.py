"""Command line entrypoint.

This file wires argparse to handlers and does nothing else. All logic lives in the
modules the handlers call, so the CLI surface stays testable and the subcommands stay
independently implementable.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from subtitler import __version__

# Subcommands land phase by phase. Anything still stubbed exits with a clear message
# naming the phase rather than an AttributeError or a half-run pipeline.
_PENDING = {
    "models": "Phase 3",
    "bench": "Phase 7",
}


def _not_yet(command: str) -> int:
    phase = _PENDING.get(command, "a later phase")
    print(f"subtitler {command}: not implemented yet (lands in {phase})", file=sys.stderr)
    return 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="subtitler",
        description="Audio or video in, burned-in subtitles out.",
    )
    parser.add_argument("--version", action="version", version=f"subtitler {__version__}")
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="repeat for more detail",
    )

    sub = parser.add_subparsers(dest="command", metavar="COMMAND")

    p_run = sub.add_parser("run", help="transcribe a file and write subtitles")
    p_run.add_argument("input", help="audio or video file")
    p_run.add_argument("-o", "--out", help="output directory (default: alongside the input)")
    p_run.add_argument(
        "--engine",
        default="auto",
        help="auto | mlx | faster-whisper | groq | groq-turbo",
    )
    p_run.add_argument("--model", default="large-v3")
    p_run.add_argument("--device", default="auto", help="auto | cpu | cuda")
    p_run.add_argument(
        "--lang",
        default="sr",
        help="pinned language code. 'auto' is supported but not recommended",
    )
    p_run.add_argument("--prompt", help="override the default steering prompt")
    p_run.add_argument("--prompt-file", help="read the steering prompt from a file")
    p_run.add_argument(
        "--denoise",
        default="none",
        choices=["none", "afftdn", "arnndn", "anlmdn", "speech"],
        help="ffmpeg-based denoise preset (default: none)",
    )
    p_run.add_argument("--burn", action=argparse.BooleanOptionalAction, default=True)
    p_run.add_argument("--soft-mux", action="store_true", help="also mux a subtitle track")
    p_run.add_argument("--srt-only", action="store_true", help="skip all video work")
    p_run.add_argument("--canvas", default="1280x720", help="canvas size for audio-only input")
    p_run.add_argument("--canvas-color", default="0x101010")
    p_run.add_argument(
        "--style-preset",
        default="outline",
        choices=["outline", "box", "minimal"],
    )
    p_run.add_argument("--font", help="system font family (default: the bundled Noto Sans)")
    p_run.add_argument("--font-size", type=int)
    p_run.add_argument("--max-line", type=int, default=42, help="characters per line")
    p_run.add_argument("--max-lines", type=int, default=2)
    p_run.add_argument("--min-dur", type=float, default=1.0, help="seconds")
    p_run.add_argument("--max-dur", type=float, default=7.0, help="seconds")
    p_run.add_argument("--max-cps", type=float, default=20.0, help="characters per second")
    p_run.add_argument("--fix", action="store_true", help="run the LLM correction pass")
    p_run.add_argument("--fix-model", default="anthropic/claude-sonnet-5", help="LiteLLM model id")
    p_run.add_argument("--auto-download", action="store_true", help="fetch a missing model")
    p_run.add_argument("--force", nargs="?", const="all", help="invalidate the cache from a stage")
    p_run.add_argument("--dry-run", action="store_true", help="print commands, execute nothing")
    p_run.add_argument("--json", action="store_true", help="machine-readable summary on stdout")

    p_doctor = sub.add_parser("doctor", help="check and install system dependencies")
    p_doctor.add_argument("--install", action="store_true", help="install what is missing")
    p_doctor.add_argument("--yes", action="store_true", help="do not prompt before installing")
    p_doctor.add_argument("--json", action="store_true")

    p_models = sub.add_parser("models", help="manage the local model cache")
    p_models.add_argument("action", choices=["list", "download", "path", "rm"])
    p_models.add_argument("name", nargs="?", default="large-v3")

    p_burn = sub.add_parser("burn", help="burn existing subtitles into a video")
    p_burn.add_argument("video")
    p_burn.add_argument("subs")
    p_burn.add_argument("-o", "--out", required=True)
    p_burn.add_argument(
        "--style-preset",
        default="outline",
        choices=["outline", "box", "minimal"],
    )
    p_burn.add_argument("--preview", action="store_true", help="render sample stills instead")

    p_lint = sub.add_parser("lint", help="check cue length, duration and reading speed")
    p_lint.add_argument("subs")
    p_lint.add_argument("--max-line", type=int, default=42)
    p_lint.add_argument("--max-lines", type=int, default=2)
    p_lint.add_argument("--min-dur", type=float, default=1.0)
    p_lint.add_argument("--max-dur", type=float, default=7.0)
    p_lint.add_argument("--max-cps", type=float, default=20.0)

    p_convert = sub.add_parser("convert", help="convert between verbose_json, srt and vtt")
    p_convert.add_argument("input")
    p_convert.add_argument("-o", "--out", required=True)

    p_bench = sub.add_parser("bench", help="engine and denoiser quality matrix")
    p_bench.add_argument("action", choices=["run", "report", "agents"])
    p_bench.add_argument("target", nargs="?", help="run directory, for report and agents")
    p_bench.add_argument("--clips", default="benchmarks/clips")
    p_bench.add_argument("--out", default="benchmarks/results")
    p_bench.add_argument("--denoise", default=None, help="restrict the denoiser axis")
    p_bench.add_argument("--allow-dirty", action="store_true")

    return parser


def _cmd_run(args: argparse.Namespace) -> int:
    from subtitler.config import load_dotenv
    from subtitler.cues import CueConfig
    from subtitler.pipeline import RunConfig, run_pipeline

    load_dotenv()

    prompt = args.prompt
    if args.prompt_file:
        prompt = Path(args.prompt_file).read_text(encoding="utf-8").strip()

    cfg = RunConfig(
        input=Path(args.input),
        out_dir=Path(args.out) if args.out else None,
        engine=args.engine,
        model=args.model,
        device=args.device,
        language=args.lang,
        prompt=prompt,
        denoise=args.denoise,
        burn=args.burn,
        soft_mux=args.soft_mux,
        srt_only=args.srt_only,
        canvas=args.canvas,
        canvas_color=args.canvas_color,
        style_preset=args.style_preset,
        font=args.font,
        font_size=args.font_size,
        cues=CueConfig(
            max_line=args.max_line,
            max_lines=args.max_lines,
            min_dur=args.min_dur,
            max_dur=args.max_dur,
            max_cps=args.max_cps,
        ),
        dry_run=args.dry_run,
        verbose=args.verbose,
    )

    quiet = args.json

    def log(message: str) -> None:
        if not quiet:
            print(message, file=sys.stderr)

    result = run_pipeline(cfg, log=log)
    if args.json:
        print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
    return 0


def _cmd_doctor(args: argparse.Namespace) -> int:
    import subprocess

    from subtitler.config import load_dotenv
    from subtitler.doctor import detect_platform, diagnose, install_plan, render

    load_dotenv()
    plat = detect_platform()
    statuses = diagnose(plat)

    if args.json:
        print(
            json.dumps(
                {
                    "platform": {
                        "system": plat.system,
                        "machine": plat.machine,
                        "distro_id": plat.distro_id,
                        "distro_like": plat.distro_like,
                        "brew_prefix": str(plat.brew_prefix) if plat.brew_prefix else None,
                        "rosetta": plat.rosetta,
                        "package_manager": plat.package_manager,
                    },
                    "deps": [s.to_dict() for s in statuses],
                },
                indent=2,
            )
        )
        return 1 if any(s.blocking for s in statuses) else 0

    print(render(statuses, plat))

    if not args.install:
        return 1 if any(s.blocking for s in statuses) else 0

    commands = install_plan(statuses, plat)
    if not commands:
        print("\nnothing to install through a package manager here.", file=sys.stderr)
        return 1 if any(s.blocking for s in statuses) else 0

    print("\nwill run:")
    for cmd in commands:
        print("  " + " ".join(cmd))
    if not args.yes:
        try:
            if input("\nproceed? [y/N] ").strip().lower() not in {"y", "yes"}:
                print("aborted", file=sys.stderr)
                return 1
        except EOFError:
            print("not a tty; pass --yes to install non-interactively", file=sys.stderr)
            return 1

    for cmd in commands:
        print(f"\n+ {' '.join(cmd)}", file=sys.stderr)
        proc = subprocess.run(cmd, check=False)
        if proc.returncode != 0:
            print(f"failed: {' '.join(cmd)}", file=sys.stderr)
            return proc.returncode

    # Re-check, so the exit code reflects reality rather than the attempt.
    print("\nre-checking...\n", file=sys.stderr)
    statuses = diagnose(plat)
    print(render(statuses, plat))
    return 1 if any(s.blocking for s in statuses) else 0


def _cmd_lint(args: argparse.Namespace) -> int:
    from subtitler.cues import CueConfig, lint_cues
    from subtitler.render import read_subtitles

    cues = read_subtitles(Path(args.subs))
    problems = lint_cues(
        cues,
        CueConfig(
            max_line=args.max_line,
            max_lines=args.max_lines,
            min_dur=args.min_dur,
            max_dur=args.max_dur,
            max_cps=args.max_cps,
        ),
    )
    for problem in problems:
        print(problem)
    if problems:
        print(f"\n{len(problems)} violations across {len(cues)} cues", file=sys.stderr)
        return 1
    print(f"{len(cues)} cues, no violations", file=sys.stderr)
    return 0


def _cmd_burn(args: argparse.Namespace) -> int:
    from subtitler.burn import burn
    from subtitler.media import probe
    from subtitler.pipeline import DEFAULT_CANVAS
    from subtitler.render import read_subtitles

    video = Path(args.video)
    info = probe(video)
    cues = read_subtitles(Path(args.subs))
    width = info.width or DEFAULT_CANVAS[0]
    height = info.height or DEFAULT_CANVAS[1]

    burn(
        cues,
        Path(args.out),
        video=video if info.has_video else None,
        audio=None if info.has_video else video,
        width=width,
        height=height,
        duration=info.duration,
        style_preset=args.style_preset,
    )
    print(f"wrote {args.out}", file=sys.stderr)
    return 0


def _cmd_convert(args: argparse.Namespace) -> int:
    from subtitler.cues import segments_to_cues
    from subtitler.model import read_json
    from subtitler.render import read_subtitles, write_srt, write_vtt

    src, dst = Path(args.input), Path(args.out)

    if src.suffix.lower() == ".json":
        # A Whisper verbose_json response, from any provider: the shape is the same.
        from subtitler.engines.base import TranscribeOptions
        from subtitler.engines.groq import parse_verbose_json

        transcript = parse_verbose_json(read_json(src), opts=TranscribeOptions())
        cues = segments_to_cues(transcript.segments)
    else:
        cues = read_subtitles(src)

    if dst.suffix.lower() == ".srt":
        write_srt(dst, cues)
    elif dst.suffix.lower() == ".vtt":
        write_vtt(dst, cues)
    else:
        print(f"unsupported output extension: {dst.suffix}", file=sys.stderr)
        return 2
    print(f"wrote {dst} ({len(cues)} cues)", file=sys.stderr)
    return 0


_HANDLERS = {
    "run": _cmd_run,
    "doctor": _cmd_doctor,
    "lint": _cmd_lint,
    "burn": _cmd_burn,
    "convert": _cmd_convert,
}


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if not args.command:
        parser.print_help()
        return 0

    handler = _HANDLERS.get(args.command)
    if handler is None:
        return _not_yet(args.command)

    try:
        return handler(args)
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
