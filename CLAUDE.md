# subtitler

Audio or video in, burned-in subtitles out. Local-first. **macOS Apple Silicon is the
primary target**, Linux is the development platform.

## Start here

1. `docs/PRD.md` for what this is and what "done" means.
2. `docs/architecture.md` for the pipeline stages, the cache key chain and the data model.
3. `docs/STATUS.md` for where the project actually stands: the current state, the findings
   that cost real time, and the outstanding work as a checklist. Read it before starting
   anything, and update it when you finish.
4. `docs/prior-art.md` for what was salvaged from `gozba2`, what was dropped, and the list
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
| `subtitler/pipeline.py` | stage orchestration |
| `subtitler/cache.py` | the content-addressed stage cache and `--force` |
| `subtitler/media.py` | ffprobe/ffmpeg: probe, extract, denoise, split long inputs |
| `subtitler/model.py` | `Word`, `Segment`, `Cue`, `Transcript` and their JSON |
| `subtitler/engines/` | one adapter per transcription backend behind `base.Engine` |
| `subtitler/cues.py` | segment to display-cue conversion. The quality-critical file |
| `subtitler/render.py` | SRT and VTT writers plus the validator |
| `subtitler/burn.py` | `.ass` generation and the three ffmpeg burn commands |
| `subtitler/postedit.py` | optional LiteLLM correction pass |
| `subtitler/edits.py` | hand corrections from the GUI editor: the artifact, and the `edit` stage |
| `subtitler/doctor.py` | dependency detection and install |
| `subtitler/gui/` | both UIs: `forms` (pure), `files`, `jobs`, `session` shared; `window` native, `app`+`server`+`static` browser |
| `subtitler/launcher.py` | `install-app`: the `.app` bundle and the `.desktop` entry |
| `subtitler/icon.py` | the icon, drawn and encoded as PNG and ICNS in pure Python |
| `subtitler/bench/` | benchmark matrix, Serbian normalization, metrics, report, reference adjudication, and `review`: the human pass over the spans it flagged |

## Gotchas that cost time before

- ffmpeg 4.4 (Ubuntu 22.04) vs 7.x/8.x (Homebrew): `-vsync` vs `-fps_mode`, and `-shortest`
  against an infinite `lavfi` source is unreliable on 4.x. Always pass an explicit `d=` and
  `-t` from ffprobe.
- **Name every filter option**, including the filename: `ass=f=subs.ass:fontsdir=fonts`.
  When a filter is missing and you pass a positional argument, ffmpeg reports
  `No option name near '...'` instead of `No such filter: 'ass'`, because option parsing
  fails before the name is resolved. That misleading error cost real time once already.
- **On macOS the formula is `ffmpeg-full`, not `ffmpeg`.** Homebrew's regular `ffmpeg`
  bottle is built without libass (verified on the macos-14 runner: the configuration line
  has `--enable-libx264` and no `--enable-libass`), so burn-in cannot work with it. This
  is the single most important macOS setup fact in the project, because `brew install
  ffmpeg` is what everyone types. `doctor` names the right formula; CI installs it and
  hard-fails if the resulting binary still lacks libass.
- `-pix_fmt yuv420p` is mandatory on every encode. Without it QuickTime and Safari may
  refuse the file, and the target user is on a Mac.
- ASS colors are `&HAABBGGRR`, byte order reversed from RGBA. Use `rgba_to_ass()`.
- Filter path escaping is avoided entirely: write `subs.ass` into a temp dir and run ffmpeg
  with `cwd` set there. Do not try to escape a real path into a filtergraph. **`arnndn=m=`
  takes a path too**, and gets the identical treatment in `media.denoise_audio`: the
  bundled model is copied to `rnnoise.rnnn` in a temp cwd. Any future filter that takes a
  filename goes the same way.
- **A stage's cache params must contain everything that changes its output and nothing
  that does not.** Too little serves a stale artifact; too much means the cache never
  hits. `transcribe` keys on `engine.describe()`, not on `--model`, because int8 on CPU and
  float16 on CUDA are different transcripts from the same model name.
- Denoise is a **separate ffmpeg pass** from extraction, not an extra `-af` on it. Folding
  them back together would put the denoiser in the extraction's cache key, so changing
  `--denoise` would demux a 3 GB source again instead of filtering the WAV already on disk.
- `/etc/os-release` on this dev box is `ID=pop`, `ID_LIKE="ubuntu debian"`. Match on both.
- **`nvidia.cublas.lib` has no `__file__`.** The pip CUDA packages are namespace packages,
  so `Path(module.__file__ or "").parent` is `Path(".")` and the CUDA preload looked for
  every library in the current working directory. It reported "nothing to preload" on a
  machine with all of them installed and had never loaded one. Use `__path__`.
- **The system CUDA toolkit is 11.5 and CTranslate2 wants 12.** Not a driver problem: 580
  supports 12 and 13. `uv sync --extra cuda` puts the cu12 libraries in the venv and
  `engines/faster.py` opens them `RTLD_GLOBAL` before CTranslate2 looks.
- **Batched decoding cannot carry `initial_prompt`.** `generate_segment_batched` passes it
  as `previous_tokens` for every window in the file, with no `prompt_reset_since` to move
  past it the way the sequential path has. On a 54-minute episode the model echoed the
  Serbian steering prompt back as transcript text and lost 15% of the speech. `--batch-size`
  therefore drops the prompt and says so; do not "fix" that by passing it through.
- Current Claude models reject `temperature`/`top_p`/`top_k` with a 400. `postedit.py` must
  not send sampling parameters unless the user asks explicitly.
- **LiteLLM's `num_retries` needs `tenacity`, and LiteLLM does not depend on it.** Without
  it the first retryable error becomes `tenacity import failed`, which hid a plain missing
  `ANTHROPIC_API_KEY` behind a package name. It is in the `fix` extra for that reason.
- Any text that did not come out of the splitter has to be re-wrapped through
  `cues.wrap_edited` (which goes through `wrap_words`), never the greedy `wrap_text`.
  Wrapping with the latter stranded the clitic "se" at the start of a line, which is the
  exact break `cues.CLITICS` forbids. Two callers now: the correction pass and the GUI's
  hand editor. A cue whose text came back unchanged keeps its original break: the splitter
  chose it from real word timings, and nothing downstream can do better.
- **Hand corrections are nobody's stage artifact.** Writing them into `cues.json` means the
  next run recomputes that stage against an unchanged key and silently overwrites them;
  putting them in the `cues` key means every keystroke invalidates a transcript-derived
  artifact. They live in `edits.json`, read as the input of the `edit` stage, which records
  the cues key they were made against so a moved transcript reports and skips them instead
  of re-pointing them at whatever now holds that index. See `subtitler/edits.py`.
- **There are two GUIs: the native window is the default and the browser page is the
  fallback.** `cli._cmd_gui` picks. The original reasoning rejected tkinter outright and was
  half right: Tk is a property of the *interpreter build*, and Homebrew's `python@3.12`
  really does not pull `python-tk@3.12`. But `.python-version` pins 3.12 and `make setup`
  goes through uv, whose python-build-standalone interpreters bundle Tk 8.6 on macOS and
  Linux both, so the documented setup has a window everywhere, and the browser page's own
  failure is unfixable by any argument about imports: the friend still has to open a
  terminal and type `subtitler gui`, which is the thing a GUI exists to spare them.
  `install-app` gives them an icon, and an icon has to open a window. What is left for the
  browser page is a Python obtained some other way and a machine with no display, and it
  keeps working for both. **Do not try to add a toolkit as a dependency**: `_tkinter` is
  compiled into the interpreter and has no PyPI package, and PySide6, wxPython and
  pywebview are all compiled wheels, which non-negotiable 6 forbids.
- **`doctor`'s `tkinter` check is a warning and must never become a failure.** A machine
  without Tk still gets a working GUI, so nothing is broken; turning it red would send
  someone to `--install` over a downgrade. It names the per-platform formula because
  `ImportError: No module named '_tkinter'` names nothing anybody can act on.
- **Neither CI runner has a display and macOS has no Xvfb**, so `window.py` contains layout
  and event bindings and nothing else: every decision it makes is a call into
  `gui/session.py`, which `tests/test_desktop.py` drives with no display at all. The one
  thing a headless test cannot check is that the widgets built match the options the form
  accepts, so `window.CONTROLS` is declared, tested against `forms`, and asserted against
  the real widgets in `Window.__init__`.
- **`--device cuda` used to skip the CUDA preload.** `resolve_device()` reaches
  `preload_cuda_libraries()` only through `_cuda_usable()`, which it consults on `auto`
  alone, so asking for cuda by name handed CTranslate2 an unprepared loader and died with
  `libcublas.so.12 is not found` at the first decoded window, on a box where `doctor` had
  just reported CUDA usable. `_load()` preloads whenever the resolved device is cuda.
- **The steering prompt is exposed on the first 30-second window of every decode, on both
  paths.** `generate_segments` seeds `all_tokens` with `initial_prompt` and only raises
  `prompt_reset_since` past it at the *end* of the first window, so "the sequential path
  resets it" was never protection for window one. Two sightings: `--denoise arnndn` on
  `fixtures/uvod-u-pravo.m4a` replaced the first fifty words of the lecture with the tail
  of `SERBIAN_PROMPT`, and a 10-second YouTube fragment of titles and music came back as
  nothing but `Zadrži srpski jezik i latinično pismo.` because there was no second window
  to reset into. Both engines now decode once more without the prompt when
  `bench/metrics.prompt_echo` fires on the result, and the pipeline warns either way. Do
  not swap that trigger for a duration threshold: it would miss the long-file sighting and
  strip the steering from every legitimate short clip.
  `bench/metrics.prompt_echo` is the one detector for this, because nothing else notices:
  the text repeats nothing, has no filler word in it and reads like Serbian.
- **A key pool needs every key tried, not one drawn at random.** `groq.py` used
  `random.choice` per attempt and gave up on the first non-retryable error, so one restricted
  key out of two failed a different random half of the benchmark's cloud cells on every run.
  Shuffle once, then give each key a turn.
- **`benchmarks/references/` is a consensus pseudo-reference, and the WER table inherits its
  shape.** It was adjudicated by an LLM from the transcripts of the very engines it scores,
  so an engine that sits in the middle of that consensus is measured partly against its own
  output, and an error every engine made identically is invisible to it and to any number
  derived from it. `bench.agents` writes that caveat into each `meta.json`, `report.py`
  prints it above the leaderboard, and every WER carries `*` while `human_verified` is false.
  Do not quote a number from that table without the qualifier, and **do not raise
  `human_verified` by hand.** `subtitler bench review` is the supported way: it merges the 44
  rows of `benchmarks/references/review-queue.md` into ~35 stops, plays each span, writes the
  answer into `benchmarks/references/<clip>.txt`, saves after every answer so it is
  resumable, and flips the flag per clip only once that clip has no unresolved span left.
  Nothing in the queue has been settled yet. `--fix` cells and prompt-echoing cells never
  vote in an adjudication, for the same reason: the first would make the reference agree with
  the correction pass under test.
- `<b>` and `<i>` are markup, not width. `cues.display_len` is what `lint` measures, so
  `--fix-markup html` does not report violations on lines that read fine.
