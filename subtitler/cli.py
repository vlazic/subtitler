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

# Imported at module scope only for the flag defaults. Both are cheap: LiteLLM is imported
# lazily inside `postedit` and faster-whisper inside `engines.faster`, so `--help` and a run
# without `--fix` never pay for either.
from subtitler import __version__, postedit
from subtitler.engines import faster as engines_faster

# Subcommands land phase by phase. Anything still stubbed exits with a clear message
# naming the phase rather than an AttributeError or a half-run pipeline. Empty now that
# `bench` has landed; kept because the next unimplemented surface should reuse it rather
# than invent a second way to say the same thing. `bench agents` is Phase 8 and says so
# itself, since the other two bench actions do work.
_PENDING: dict[str, str] = {}


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

    p_run = sub.add_parser("run", help="transcribe a file or a URL and write subtitles")
    p_run.add_argument(
        "input",
        help=(
            "audio or video file, or an http(s) URL to fetch with yt-dlp "
            "(needs `uv sync --extra fetch`)"
        ),
    )
    p_run.add_argument(
        "-o",
        "--out",
        help="output directory (default: alongside the input; required for a URL)",
    )
    p_run.add_argument(
        "--start",
        metavar="TIME",
        help="keep only from this point on: SS, MM:SS or HH:MM:SS. Cue times start at zero",
    )
    p_run.add_argument(
        "--end",
        metavar="TIME",
        help="keep only up to this point, same formats as --start",
    )
    p_run.add_argument(
        "--engine",
        default="auto",
        help="auto | mlx | faster-whisper | groq | groq-turbo",
    )
    p_run.add_argument("--model", default="large-v3")
    p_run.add_argument("--device", default="auto", help="auto | cpu | cuda")
    p_run.add_argument(
        "--batch-size",
        type=int,
        default=0,
        metavar="N",
        help=(
            "decode N chunks at a time on CUDA (faster-whisper only; 0 = sequential, the "
            f"default). {engines_faster.DEFAULT_BATCH_SIZE} is about 3x faster on a 24 GB "
            "card, at the cost of the steering prompt, which batched decoding echoes into "
            "the transcript"
        ),
    )
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
    p_run.add_argument(
        "--fix-model",
        default=postedit.DEFAULT_MODEL,
        help="any LiteLLM model id, e.g. openai/gpt-4o, groq/..., ollama/llama3.1",
    )
    p_run.add_argument(
        "--fix-prompt",
        default=postedit.DEFAULT_PROMPT,
        help="prompt name from prompts/ (postedit, gozba) or a path to a .md file",
    )
    p_run.add_argument(
        "--fix-temperature",
        type=float,
        default=None,
        help=(
            "forward a sampling temperature. Omitted by default on purpose: current Claude "
            "models reject temperature/top_p/top_k with a 400"
        ),
    )
    p_run.add_argument("--fix-batch", type=int, default=postedit.DEFAULT_BATCH_SIZE)
    p_run.add_argument("--fix-workers", type=int, default=postedit.DEFAULT_WORKERS)
    p_run.add_argument(
        "--fix-markup",
        default="strip",
        choices=["strip", "html"],
        help="what to do with markdown the model emits anyway (html gives <b>/<i>)",
    )
    p_run.add_argument(
        "--drop-intro-phrases",
        metavar="FILE",
        help="drop cues containing any phrase in FILE, one per line. Off by default",
    )
    p_run.add_argument("--auto-download", action="store_true", help="fetch a missing model")
    p_run.add_argument(
        "--force",
        nargs="?",
        const="all",
        metavar="STAGE",
        help=(
            "invalidate the stage cache from a stage onwards "
            "(fetch, trim, extract, denoise, transcribe, cues, fix, burn); "
            "bare --force means all"
        ),
    )
    p_run.add_argument("--dry-run", action="store_true", help="print commands, execute nothing")
    p_run.add_argument("--json", action="store_true", help="machine-readable summary on stdout")

    p_gui = sub.add_parser("gui", help="open the graphical interface in your browser")
    p_gui.add_argument("--port", type=int, default=0, help="0 (the default) picks a free port")
    p_gui.add_argument(
        "--host",
        default="127.0.0.1",
        help="loopback by default; anything else exposes your files to the network",
    )
    p_gui.add_argument(
        "--no-browser",
        action="store_true",
        help="print the address instead of opening a browser",
    )

    # `run URL` covers the common case on its own, so this exists for the other one:
    # keeping the media. Downloading once and then iterating locally with `run` costs the
    # site nothing and works offline, and it is also how you look at what actually came
    # down before spending an hour of GPU on it.
    p_fetch = sub.add_parser("fetch", help="download a URL with yt-dlp, without transcribing")
    p_fetch.add_argument("url")
    p_fetch.add_argument("-o", "--out", required=True, help="output directory")
    p_fetch.add_argument(
        "--audio-only",
        action="store_true",
        help="fetch just the audio track, which is all a transcript needs",
    )

    p_doctor = sub.add_parser("doctor", help="check and install system dependencies")
    p_doctor.add_argument("--install", action="store_true", help="install what is missing")
    p_doctor.add_argument("--yes", action="store_true", help="do not prompt before installing")
    p_doctor.add_argument("--json", action="store_true")

    p_models = sub.add_parser("models", help="manage the local model cache")
    p_models.add_argument("action", choices=["list", "download", "path", "rm"])
    p_models.add_argument("name", nargs="?", default="large-v3")
    p_models.add_argument(
        "--backend",
        choices=["mlx", "faster-whisper"],
        help="default: whichever this machine would use",
    )
    p_models.add_argument("--all", action="store_true", help="list every backend")

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
    p_bench.add_argument(
        "--clips",
        default=None,
        help=(
            "a directory or a comma-separated list of files. Default: benchmarks/clips if "
            "it holds media, otherwise the two checked-in fixtures"
        ),
    )
    p_bench.add_argument("--out", default="benchmarks/results")
    p_bench.add_argument("--references", default="benchmarks/references")
    p_bench.add_argument(
        "--work",
        default="benchmarks/.work",
        help="shared stage cache for the matrix; kept across runs on purpose",
    )
    p_bench.add_argument("--denoise", default=None, help="restrict the denoiser axis")
    p_bench.add_argument(
        "--engine",
        default=None,
        help="restrict the engine axis. Default: every local engine usable here",
    )
    p_bench.add_argument("--model", default="large-v3")
    p_bench.add_argument("--device", default="auto", help="auto | cpu | cuda")
    p_bench.add_argument("--batch-size", type=int, default=0, metavar="N")
    p_bench.add_argument("--lang", default="sr", help="pinned language code")
    p_bench.add_argument(
        "--fix",
        action="store_true",
        help="add the correction pass as a second axis, so each cell is measured on and off",
    )
    p_bench.add_argument("--fix-model", default=postedit.DEFAULT_MODEL)
    p_bench.add_argument(
        "--force",
        nargs="?",
        const="all",
        metavar="STAGE",
        help="invalidate the shared stage cache from a stage onwards",
    )
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

    fix = None
    if args.fix:
        fix = postedit.FixConfig(
            model=args.fix_model,
            prompt=args.fix_prompt,
            batch_size=args.fix_batch,
            workers=args.fix_workers,
            temperature=args.fix_temperature,
            markup=args.fix_markup,
            drop_intro_phrases=(Path(args.drop_intro_phrases) if args.drop_intro_phrases else None),
        )
    elif args.drop_intro_phrases:
        print("--drop-intro-phrases has no effect without --fix", file=sys.stderr)

    cfg = RunConfig.from_source(
        args.input,
        start=args.start,
        end=args.end,
        out_dir=Path(args.out) if args.out else None,
        engine=args.engine,
        model=args.model,
        device=args.device,
        batch_size=args.batch_size,
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
        fix=fix,
        force=args.force,
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


def _cmd_fetch(args: argparse.Namespace) -> int:
    from subtitler import fetch as fetch_mod

    if not fetch_mod.is_url(args.url):
        print(f"not an http(s) URL: {args.url}", file=sys.stderr)
        return 2

    out = Path(args.out).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    kind = "audio" if args.audio_only else "video"
    got = fetch_mod.fetch(
        args.url, out, kind=kind, progress=lambda message: print(message, file=sys.stderr)
    )

    # yt-dlp writes to a fixed stem so the pipeline's cache can name the artifact. A file
    # the user keeps deserves the video's own name instead.
    named = out / f"{got.stem}{got.path.suffix}"
    if named != got.path:
        got.path.replace(named)
    print(named)
    return 0


def _cmd_gui(args: argparse.Namespace) -> int:
    from subtitler.config import load_dotenv
    from subtitler.gui.server import serve

    # Same as `run`: the correction pass reads its API key from .env, and the GUI offers
    # the same checkbox, so the key has to be loaded before the server starts.
    load_dotenv()
    return serve(host=args.host, port=args.port, open_browser=not args.no_browser)


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


def _cmd_models(args: argparse.Namespace) -> int:
    from subtitler import models
    from subtitler.engines import is_apple_silicon

    # The backend whose weights this machine would actually use.
    backend = args.backend or ("mlx" if is_apple_silicon() else "faster-whisper")

    if args.action == "list":
        print(models.render_list(None if args.all else backend))
        return 0

    try:
        spec = models.resolve(args.name, backend)
    except models.ModelNotFound as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if args.action == "path":
        path = models.local_path(spec)
        print(path or models.cache_root())
        return 0 if path else 1

    if args.action == "rm":
        removed = models.remove(spec)
        print(f"{'removed' if removed else 'nothing to remove for'} {spec.key}", file=sys.stderr)
        return 0

    # download
    if models.local_path(spec) is not None:
        print(f"{spec.key} is already cached at {models.local_path(spec)}", file=sys.stderr)
        return 0
    models.download(spec, progress=lambda msg: print(msg, file=sys.stderr))
    return 0


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
    from subtitler.burn import burn, preview
    from subtitler.media import probe
    from subtitler.pipeline import DEFAULT_CANVAS
    from subtitler.render import read_subtitles

    video = Path(args.video)
    info = probe(video)
    cues = read_subtitles(Path(args.subs))
    width = info.width or DEFAULT_CANVAS[0]
    height = info.height or DEFAULT_CANVAS[1]

    if args.preview:
        stills = preview(
            cues,
            Path(args.out),
            video=video if info.has_video else None,
            audio=None if info.has_video else video,
            width=width,
            height=height,
        )
        for path in stills:
            print(path)
        print(f"\n{len(stills)} stills written. Pick one and pass --style-preset.", file=sys.stderr)
        return 0

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


def _cmd_bench(args: argparse.Namespace) -> int:
    from subtitler import media
    from subtitler.bench import run as bench
    from subtitler.config import load_dotenv

    load_dotenv()
    root = Path.cwd()
    out_root = Path(args.out)
    references = Path(args.references)

    def log(message: str) -> None:
        print(message, file=sys.stderr)

    if args.action == "agents":
        # Phase 8: LLM adjudication of reference transcripts. The seam it plugs into is
        # `bench report`, which recomputes every metric from the kept transcripts, so an
        # adjudicated reference scores a run that already happened without re-transcribing.
        print("subtitler bench agents: not implemented yet (lands in Phase 8)", file=sys.stderr)
        return 2

    if args.action == "report":
        target = Path(args.target) if args.target else bench.latest_run(out_root)
        if target is None:
            print(f"no benchmark runs under {out_root}", file=sys.stderr)
            return 1
        bench.rescore(target, references=references, log=log)
        print((target / "report.md").read_text(encoding="utf-8"))
        return 0

    clips = bench.resolve_clips(args.clips, root=root)
    denoisers = bench.parse_axis(
        args.denoise, valid=media.DENOISE_FILTERS, label="denoiser"
    ) or tuple(media.DENOISE_FILTERS)
    engines = bench.parse_axis(
        args.engine, valid=("mlx", "faster-whisper", "groq", "groq-turbo"), label="engine"
    )
    if engines is None:
        engines = bench.available_local_engines(args.model, args.device)
        if not engines:
            print(
                "no local engine is usable here; run `subtitler doctor`, or name one with --engine",
                file=sys.stderr,
            )
            return 1
        log(f"engine axis: {', '.join(engines)} (auto-detected)")

    cfg = bench.BenchConfig(
        clips=clips,
        denoisers=denoisers,
        engines=engines,
        model=args.model,
        device=args.device,
        batch_size=args.batch_size,
        language=args.lang,
        fix_axis=args.fix,
        fix_model=args.fix_model,
        out_root=out_root,
        references=references,
        work=Path(args.work),
        allow_dirty=args.allow_dirty,
        force=args.force,
    )
    run_dir = bench.run_matrix(cfg, repo=root, log=log)
    print(run_dir)
    return 0


_HANDLERS = {
    "run": _cmd_run,
    "fetch": _cmd_fetch,
    "bench": _cmd_bench,
    "gui": _cmd_gui,
    "doctor": _cmd_doctor,
    "models": _cmd_models,
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
