# subtitler

Give it a video or an audio file. Get back `.srt`, `.vtt`, and a copy of the video with the
subtitles burned in.

Runs locally by default: no API key, nothing uploaded. Tuned for Serbian, but the language
is a flag, so it works for anything Whisper supports.

```bash
subtitler run ~/Videos/govor.mp4
# -> govor.srt  govor.vtt  govor.subbed.mp4
```

## Why this exists

Whisper gives you one caption per speech segment, which can run 5 to 25 seconds of unbroken
text. That is fine for a scrolling transcript on a web page and unreadable burned into a
video. The interesting part of this tool is not the transcription, it is turning those
segments into cues a human can actually read: split at sentence and clause boundaries,
balanced across at most two lines, never longer than 42 characters per line, never faster
than 20 characters per second.

## Install

Requires [uv](https://docs.astral.sh/uv/) and an ffmpeg built with **libass**.

> **macOS:** use `brew install ffmpeg-full`, not `brew install ffmpeg`. Homebrew's
> regular `ffmpeg` bottle is built without libass, and without libass subtitles
> cannot be burned into the video at all. `subtitler doctor` checks for this and
> tells you the right formula.

```bash
git clone https://github.com/vlazic/subtitler && cd subtitler
make setup          # picks the right extras for your platform, then runs the doctor
make install-deps   # only if the doctor reports something missing
make models         # downloads Whisper large-v3 (about 3 GB, one time)
```

`make setup` resolves to `uv sync --extra mlx --extra cloud --extra dev` on macOS and
`--extra local --extra cloud --extra dev` on Linux, because `mlx-whisper` has no Linux wheels.

## The window, for people who do not use a terminal

```bash
subtitler gui        # or: make gui
```

That starts a small server on your own machine and opens your browser at it. Nothing is
uploaded and nothing leaves the computer: the page is talking to a program running on it,
and the address it opens (`http://127.0.0.1:...`) is not reachable from anywhere else.
Leave the terminal window open while you use it, and press Ctrl-C when you are done.

The page covers the whole job:

- **Pick a file** by browsing your own disk, starting from Desktop, Downloads and
  Movies (Videos on Linux). It filters to audio and video, with a switch to show everything.
- **Choose what to make**: a video with the subtitles burned in plus the `.srt`/`.vtt`, or
  the subtitle files alone. Pick the look (outline, box, minimal) and where to save.
- **Choose how to transcribe**: engine, model, CPU or GPU, language, and the denoise
  preset. If the model has not been downloaded yet, the page says so and offers a button
  that downloads it with a progress log, instead of telling you to run a command.
- **Shape the subtitles**: characters per line, lines per subtitle, shortest and longest
  duration, reading speed. Everything else, including the steering prompt, the canvas, the
  fonts, `--force` and the LLM correction pass, is under "Everything else".
- **Watch it run.** The transcription happens on a worker, so the page stays responsive
  through a 45-minute file. It shows which stage is running and streams the same log the
  command line prints, then lists the files it wrote with a button that opens the folder
  (Show in Finder on macOS).
- **Check my computer** runs the same checks as `subtitler doctor` and shows what is
  missing along with the exact command to fix it. This is the point of the button: on macOS
  `brew install ffmpeg` gives you an ffmpeg without libass, and burn-in silently cannot
  work. The window tells you before you waste a transcription on it.

Every screen prints the equivalent command line, so anything you set up by clicking can be
repeated, scripted, or pasted into a bug report.

There is nothing to install for it. The GUI is stdlib `http.server` and one HTML file, with
no packaging step, no bundled JavaScript, and no compiled dependency, which is also why it
is a browser page rather than a Tk window: whether `tkinter` exists at all depends on how
your Python was built, and Homebrew's `python@3.12` does not ship it.

`--port N` pins the port (the default picks a free one) and `--no-browser` just prints the
address. Do not pass `--host` unless you mean it: the page can read and write files
anywhere your user can, and binding to anything but loopback hands that to your network.

## Usage

```bash
subtitler gui                                 # the browser interface (see above)
subtitler run INPUT.mp4                       # transcribe, shape cues, burn in
subtitler run INPUT.m4a --canvas 1920x1080    # audio-only input gets a video canvas
subtitler run INPUT.mp4 --srt-only            # sidecar files, no video work
subtitler run INPUT.mp4 --style-preset box    # outline | box | minimal
subtitler run INPUT.mp4 --engine groq         # cloud instead of local
subtitler run INPUT.mp3 --batch-size 16       # NVIDIA GPU: ~3x again, see below
subtitler run INPUT.mp4 --denoise speech      # none | afftdn | arnndn | anlmdn | speech
subtitler run INPUT.mp4 --fix                 # optional LLM correction pass
subtitler run INPUT.mp4 --fix --fix-model openai/gpt-4o
subtitler run INPUT.mp4 --force transcribe    # ignore the cache from a stage onwards

subtitler doctor [--install]                  # check and install system dependencies
subtitler models list | download | path | rm
subtitler burn VIDEO SUBS -o OUT [--preview]  # re-style without re-transcribing
subtitler lint SUBS.srt                       # check cue length, duration, reading speed
subtitler bench run | report                  # engine and denoiser quality matrix
```

## Re-runs are free

Every expensive stage is cached in `<output>/.subtitler/<name>/`, keyed on the content of
its input and the parameters that produced it. Run the same command twice and the second
run reads the cache: 84 seconds becomes 0.5, and the `.srt`, `.vtt` and `.mp4` come out
byte for byte the same.

The keys chain, so changing one thing re-does only what depends on it:

| Change | What re-runs |
|---|---|
| `--style-preset box` | the burn |
| `--max-line 30` | cues, then the burn |
| `--denoise afftdn` | denoise, transcribe, cues, burn (the extraction is reused) |
| `--fix-model` or an edit to the prompt file | the correction pass, then the burn |
| a different `--model` or device | transcribe onwards |
| the input file's bytes | everything |

`--force` ignores the cache: bare `--force` for all of it, or `--force transcribe` for one
stage and everything after it. Deleting the `.subtitler` directory is always safe.

Large inputs are fingerprinted by sampling (length, plus the first, middle and last
megabyte) rather than by hashing in full, because reading 3 GB to decide whether to skip
work defeats the point. See the note in `subtitler/cache.py` for the tradeoff.

## Denoising

Off by default, and it is not the centrepiece: on this project's material all five
denoisers produced roughly 95% identical Serbian text. It is a benchmarked, pluggable
stage, and every preset is a built-in ffmpeg filter, so there is nothing to compile.

| Preset | Filter | For |
|---|---|---|
| `none` | | the default |
| `afftdn` | `afftdn=nf=-25` | steady broadband hiss |
| `arnndn` | `arnndn` + bundled RNNoise weights | speech recorded with a fan or an air conditioner |
| `anlmdn` | `anlmdn` | non-local means; slower, gentler |
| `speech` | highpass, `afftdn`, `loudnorm` | rumble plus hiss plus uneven levels |

`arnndn` needs a model file, so one ships in `subtitler/assets/rnnoise/` (see the README
there for provenance and licence). Nothing is downloaded at denoise time.

## The correction pass

`--fix` sends the cue text to an LLM to fix spelling, grammar and mangled proper nouns. Off
by default, and the only part of this tool that leaves your machine or costs money.

```bash
uv sync --extra fix
subtitler run INPUT.mp3 --fix                          # anthropic/claude-sonnet-5
subtitler run INPUT.mp3 --fix --fix-model openai/gpt-4o
subtitler run INPUT.mp3 --fix --fix-model ollama/llama3.1   # local, no key
```

It corrects **text only**. Timestamps are never sent to the model, so they cannot come back
wrong: run the same file with and without `--fix` and every `-->` line is byte for byte the
same. Cues go out in batches of 40 (`--fix-batch`) over 4 threads (`--fix-workers`) as a
numbered JSON list, and a reply is accepted only if its length and its index set match the
request exactly. A batch that fails that check is discarded, the original cues are kept, and
the warning names the batch and its cue range. If every batch fails, the run stops with an
error rather than writing an uncorrected file and exiting 0.

`--fix-model` takes any [LiteLLM](https://docs.litellm.ai/docs/providers) model string, so
OpenAI, Groq, Anthropic, Ollama and the rest work with no code change. **No sampling
parameter is sent unless you pass `--fix-temperature`**, because current Claude models
reject `temperature`, `top_p` and `top_k` with a 400 and LiteLLM forwards whatever it is
given.

The prompt is a file, not a string in the source:

| Flag | Effect |
|---|---|
| `--fix-prompt postedit` | the default: grammar, source language and script preserved, no markdown |
| `--fix-prompt gozba` | the Serbian philosophy-show variant: **bold** for figures, *italics* for titles |
| `--fix-prompt ./mine.md` | any file. Everything above the first `---` is notes; the rest is the prompt |
| `--fix-markup html` | keep the emphasis as `<b>`/`<i>` instead of stripping it |
| `--drop-intro-phrases FILE` | drop cues containing any phrase in FILE, one per line |

Markdown is stripped by default because the default output is burned into video through
libass, which renders `<b>` as three literal characters. `--fix-markup html` is for the
sidecar `.vtt` case; its tags do not count against the 42-character line limit, in the
wrapper or in `lint`.

`--drop-intro-phrases` removes only the cues that match, and matching ignores case, quotes
and markup, so one line in the file covers every spelling of a station ident.

## Engines

| Engine | Where | Notes |
|---|---|---|
| `mlx` | macOS Apple Silicon | default on a Mac. `mlx-community/whisper-large-v3-mlx` |
| `faster-whisper` | Linux, Intel Mac, any CPU | default elsewhere. CUDA when available, see below |
| `groq` | anywhere with a key | `whisper-large-v3` and `-turbo`, for comparison or long files |

The engine is chosen automatically from the platform. Asking for one explicitly that is not
installed is a hard error with the exact `uv sync` command to fix it, never a silent fallback.

## NVIDIA GPUs

If you have one, use it. On a 54-minute Serbian episode, `faster-whisper` large-v3 on an
RTX 3090 is **17x** faster than the same model on a 16-thread CPU, and **51x** with
`--batch-size 16`. 47 minutes becomes 2 minutes 54, or 55 seconds.

```bash
uv sync --extra local --extra cuda   # the --extra cuda is the whole trick
subtitler doctor                     # confirms the GPU and the runtime
subtitler run gozba.mp3              # CUDA is picked automatically when it works
```

Nothing else changes: the device is chosen automatically, `--device cpu` forces the old
path, and a CUDA runtime that turns out to be unusable degrades to the CPU mid-run and says
so in the transcript's `cuda_fallback`.

### The CUDA 11 vs 12 trap

CTranslate2's wheels link against CUDA **12**. Your driver is almost certainly new enough,
and that is not what is being checked. What matters is the CUDA *runtime libraries* on the
machine, and a distribution that ships `nvcc` 11.5 has `libcublas.so.11`, which does not
satisfy a link against `libcublas.so.12`. The symptom is this, minutes into a run:

```
Library libcublas.so.12 is not found or cannot be loaded
```

The fix is `--extra cuda`, which pulls `nvidia-cublas-cu12` and `nvidia-cudnn-cu12` into
the venv as ordinary Python packages. Nothing is installed system-wide and the 11.5 toolkit
is left alone. `engines/faster.py` opens those libraries with `RTLD_GLOBAL` before
CTranslate2 starts, so they are already registered under their sonames when it looks. On a
machine with no GPU there is nothing to preload and the whole thing is a no-op.

`subtitler doctor` reports both halves separately, so a failure is attributable:

```
  ok    nvidia gpu       driver 580.159.03 (NVIDIA GeForce RTX 3090, 24576 MiB)
  ok    cuda runtime     (CTranslate2 will decode on the GPU in float16)
```

Without `--extra cuda` the second line becomes a warning naming `libcublas.so.12` and the
`uv sync` that fixes it. Neither line can ever fail the doctor: on a Mac and on a CPU-only
box they read `n/a`.

### `--batch-size`, and what it costs

`--batch-size N` decodes N VAD chunks at once instead of walking the file sequentially. It
is off by default, and turning it on is a real trade rather than free speed:

**Batched decoding cannot use the steering prompt.** faster-whisper prepends
`initial_prompt` to *every* window in batched mode, and the model starts writing the prompt
into the transcript. On a 54-minute Serbian episode it echoed "Zadrži srpski jezik i
latinično pismo..." back dozens of times and came out 900 words (15%) short. So the engine
drops the prompt when batching, prints a line saying it did, and records
`"initial_prompt": false` in the transcript params. Without the prompt the batched
transcript is 6231 words against the sequential 6186, with no echo.

For Serbian the prompt is worth keeping, so the default stays sequential. For a bulk
back-catalogue where 4 hours versus 13 hours decides whether the job happens at all, batch.

`16` is the recommended value on a 24 GB card. `32` is 6% faster and peaks at 22.5 GB,
which leaves no room for a desktop session on the same GPU.

### Measured

RTX 3090, large-v3, Serbian. CPU is int8 on 16 threads, GPU is float16.

| Input | Config | RTF | Wall clock |
|---|---|---|---|
| 109 s fixture | CPU int8 | 0.677 | 79 s |
| 109 s fixture | CUDA float16 | 0.042 | 7.5 s |
| 109 s fixture | CUDA float16, `--batch-size 16` | 0.021 | 5.2 s |
| 54 min episode | CPU int8 | 0.869 | 46 min 52 s |
| 54 min episode | CUDA float16 | 0.051 | 2 min 54 s |
| 54 min episode | CUDA float16, `--batch-size 16` | 0.015 | 55 s |

Long files are *worse* than short ones on the CPU (0.87 against 0.68) and no worse on the
GPU, so the speedup grows with the thing you actually want to transcribe.

The 353-episode, 260-hour `gozba` archive that motivated this: **9.4 days** on the CPU,
**13 hours** on the GPU, **4 hours** batched.

**The GPU transcript is not a degraded one.** On the 109 s fixture, CPU int8 and CUDA
float16 produce byte-identical cue *text*; the only difference is timestamps, and by at
most 40 ms on 6 of 20 cues. Over the whole 54-minute episode they agree on 96.9% of words
(6148 against 6168), which is ordinary int8-versus-float16 drift and not lost content.

A larger beam does not help and was not adopted: `beam_size` 8 was 7% slower than 5 on the
fixture and 17% slower on the episode, for a transcript no closer to the reference. Nor
does batch 32: 6% faster than 16 and 22.5 GB of VRAM against under 16.

## Verified on

| Platform | Engine | Status |
|---|---|---|
| Pop!_OS 22.04, ffmpeg 4.4.2 | groq | end to end, including burn-in |
| Pop!_OS 22.04, ffmpeg 4.4.2 | faster-whisper large-v3, CPU int8 | 109s of Serbian at RTF 0.68 |
| Pop!_OS 22.04, ffmpeg 4.4.2 | stage cache | 83.9s cold, 0.52s warm, all three outputs byte-identical |
| Pop!_OS 22.04, ffmpeg 4.4.2 | all five denoise presets | each runs and changes the audio |
| Pop!_OS 22.04, ffmpeg 4.4.2 | `--fix` against `openai/gpt-4o` | 20 Serbian cues, 5 corrected, every timestamp byte-identical |
| Any | `--fix` against Anthropic or Groq | not verified: no key for either on this machine |
| ubuntu-latest CI | faster-whisper tiny | transcribe, burn-in, hostile paths, diacritics |
| macOS 14 Apple Silicon CI, ffmpeg-full 8.1.2 | mlx tiny | transcribe (RTF 0.40), burn-in, hostile paths, diacritics |
| Pop!_OS 22.04, RTX 3090 | faster-whisper large-v3, CUDA float16 | 109s at RTF 0.042 and 54min at RTF 0.051, GPU at 85-100% throughout |
| Pop!_OS 22.04, RTX 3090 | the same, `--batch-size 16` | 54min at RTF 0.015, under 16 GB VRAM |
| Pop!_OS 22.04, RTX 3090 | CPU int8 against CUDA float16 | identical cue text on the 109s fixture, 96.9% word agreement over 54 minutes |
| Pop!_OS 22.04, no `--extra cuda` | the CUDA-less fallback | doctor warns and names `libcublas.so.12`, the run decodes on the CPU, exit 0 |
| Pop!_OS 22.04, Chrome | `subtitler gui` | picked a file, set the options and started a run from the page; 109s of Serbian burned in, 21 cues, 12s |
| ubuntu-latest CI | `subtitler gui` | binds, serves the page, refuses an untokened call, reports dependencies, completes a faster-whisper run through the API, headless |
| macOS 14 Apple Silicon CI | `subtitler gui` | the same, reporting `Darwin arm64 brew:/opt/homebrew` and `Show in Finder`, transcribing on mlx |
| Any Mac | `subtitler gui` reveal in Finder | not verified on hardware: `open -R` is covered by a test with a faked `Platform`, never by a Mac |

Both CI runners render `ČĆĐŠŽ čćđšž` identically from the bundled font, which is what the
bundling is for. CI transcribes with `tiny`, which is fast and cheap and produces poor
Serbian: it proves the path works, not that the output is good. Quality is the benchmark's
job, not CI's.

This table is updated only when CI or a human actually runs it there. Nothing is listed as
working because it ought to.

## License

MIT. Bundled Noto Sans is SIL OFL 1.1 (see `subtitler/assets/fonts/`). The bundled RNNoise
weights carry no copyright claim upstream (see `subtitler/assets/rnnoise/`); RNNoise itself
is Xiph.Org's under BSD 3-clause and reaches this project only through ffmpeg's `arnndn`
filter. Whisper weights are MIT. ffmpeg is invoked as a subprocess and is not linked.
