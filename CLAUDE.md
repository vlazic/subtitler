# subtitler

Audio or video in, burned-in subtitles out. Local-first. **macOS Apple Silicon is the
primary target**, Linux is the development platform.

## Start here

1. `docs/PRD.md` for what this is and what "done" means.
2. `docs/architecture.md` for the pipeline stages and the data model.
3. `docs/prior-art.md` for what was salvaged from `gozba2`, what was dropped, and the list
   of bugs that were fixed on the way. Do not reintroduce them.

## Non-negotiables

1. **Never invent an ffmpeg command outside `media.py` or `burn.py`.** Every invocation is
   built by a function that has a command-construction test. `subprocess.run(list, shell=False)`
   only. No f-string shell lines.
2. **`language` is always pinned, never auto-detected.** Whisper reads Serbian as `hr` or
   `bs` on short or quiet clips. `--lang auto` exists and warns loudly.
3. **The Serbian steering prompt is verbatim** and is not "improved" without a benchmark run
   showing it helps. It lives as one constant in `engines/base.py`.
4. **Nothing is written into the user's CWD.** The code this replaced wrote `temp_audio.wav`
   into whatever directory it was run from and collided on concurrent runs. Use
   `tempfile.TemporaryDirectory` or the stage cache under the output directory.
5. **macOS branches must be unit-testable on Linux.** Anything platform-specific goes behind
   the injectable `Platform` object in `doctor.py` or inside `engines/mlx.py`. Nothing else
   may call `platform.system()` directly.
6. **Zero compiled dependencies.** No autotools, no `cargo install`, no vendored C. If a
   feature needs one, it does not ship. Denoising uses ffmpeg filters for exactly this reason.
7. **Fonts are bundled, never resolved from the system by default.** No font ships on both
   macOS and stock Ubuntu, so a system font name silently renders differently per machine.
8. **`cues.py` owns line layout, libass never wraps.** The generated `.ass` sets
   `WrapStyle: 2` and emits explicit `\N` breaks.
9. **No benchmark leaderboard number ever comes from an LLM.** Agents classify and judge;
   metrics are computed by `bench/metrics.py`.
10. Run `make test && make lint` before committing. Semantic commit messages.

## Layout

| Path | What |
|---|---|
| `subtitler/cli.py` | argparse entrypoint, subcommands only, no logic |
| `subtitler/pipeline.py` | stage orchestration and the stage cache |
| `subtitler/media.py` | ffprobe/ffmpeg: probe, extract, denoise, split long inputs |
| `subtitler/model.py` | `Word`, `Segment`, `Cue`, `Transcript` and their JSON |
| `subtitler/engines/` | one adapter per transcription backend behind `base.Engine` |
| `subtitler/cues.py` | segment to display-cue conversion. The quality-critical file |
| `subtitler/render.py` | SRT and VTT writers plus the validator |
| `subtitler/burn.py` | `.ass` generation and the three ffmpeg burn commands |
| `subtitler/postedit.py` | optional LiteLLM correction pass |
| `subtitler/doctor.py` | dependency detection and install |
| `subtitler/bench/` | benchmark matrix, Serbian normalization, metrics, report |

## Gotchas that cost time before

- ffmpeg 4.4 (Ubuntu 22.04) vs 7.x/8.x (Homebrew): `-vsync` vs `-fps_mode`, and `-shortest`
  against an infinite `lavfi` source is unreliable on 4.x. Always pass an explicit `d=` and
  `-t` from ffprobe.
- **Name every filter option**, including the filename: `ass=f=subs.ass:fontsdir=fonts`.
  When a filter is missing and you pass a positional argument, ffmpeg reports
  `No option name near '...'` instead of `No such filter: 'ass'`, because option parsing
  fails before the name is resolved. That misleading error cost real time once already.
- **Do not trust that a Homebrew ffmpeg has libass.** The macOS CI runner's ffmpeg 8.1.2
  does not carry the `ass` filter. `doctor` treats libass as a required dependency for
  exactly this reason, and CI prints `which -a ffmpeg` plus the configuration line so the
  build in use is always identifiable.
- `-pix_fmt yuv420p` is mandatory on every encode. Without it QuickTime and Safari may
  refuse the file, and the target user is on a Mac.
- ASS colors are `&HAABBGGRR`, byte order reversed from RGBA. Use `rgba_to_ass()`.
- Filter path escaping is avoided entirely: write `subs.ass` into a temp dir and run ffmpeg
  with `cwd` set there. Do not try to escape a real path into a filtergraph.
- `/etc/os-release` on this dev box is `ID=pop`, `ID_LIKE="ubuntu debian"`. Match on both.
- Current Claude models reject `temperature`/`top_p`/`top_k` with a 400. `postedit.py` must
  not send sampling parameters unless the user asks explicitly.
