# subtitler

Give it a video or an audio file, or a URL. Get back `.srt`, `.vtt`, and a copy of the video
with the subtitles burned in.

Runs locally by default: no API key, nothing uploaded. Tuned for Serbian, but the language
is a flag, so it works for anything Whisper supports.

```bash
subtitler run ~/Videos/govor.mp4
# -> govor.srt  govor.vtt  govor.subbed.mp4

subtitler run "https://www.youtube.com/watch?v=..." --start 10:00 --end 13:00 -o ~/out
# -> the three minutes you asked for, subtitled, with cue times starting at 00:00:00
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

`make setup` resolves to `uv sync --extra mlx --extra cloud --extra fetch --extra dev` on
macOS and `--extra local --extra cloud --extra fetch --extra dev` on Linux, because
`mlx-whisper` has no Linux wheels.

Optional extras, none of which a plain local run needs:

| Extra | For | Install |
|---|---|---|
| `fetch` | downloading a URL, through yt-dlp | `uv sync --extra fetch` |
| `fix` | the LLM correction pass | `uv sync --extra fix` |
| `cuda` | the CUDA 12 libraries CTranslate2 wants | `uv sync --extra cuda` |

Each is imported lazily and each says its own install line if you use the feature without
it, so nothing here is a prerequisite for anything else.

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
- **Or paste a link**, with a start and stop time beside it. Same yt-dlp path the command
  line uses. A link has no folder of its own to write beside, so the page asks for one
  before it will start; that is non-negotiable 4, said while the button is still unpressed.
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
  command line prints (including yt-dlp's download progress), then lists the files it wrote
  with a button that opens the folder (Show in Finder on macOS).
- **Read the subtitles before the video is made.** On by default, and the section below is
  about what it does to the cache.
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

## Checking the subtitles before the video is made

Recognition gets names, foreign words and proper nouns wrong, and the only way to know is
to read them. With **"Let me read and correct the subtitles before the video is made"**
ticked (it is by default), the run stops as soon as the cues exist, before anything is
encoded, and the page turns into an editor:

- every cue with its **start, end and duration**, because "too fast to read" is not a
  judgement anyone can make from a character count;
- a **Listen** button that plays that cue's span of the source audio, so you can hear what
  was actually said instead of guessing;
- the **lines as they will be rendered**, each with the number of characters that decides
  whether it fits;
- and a **live quality check**: as you type, the cue is re-broken and re-checked, and any
  cue that breaks a rule is highlighted with the reason in the same words `subtitler lint`
  would use. Long line, too short, too long, too fast. Markup (`<b>`, `<i>`) is not
  counted, because a player renders it as weight and not as width.

Nothing is re-checked in the browser: the page sends the text back and the server answers
with `cues.wrap_edited` and `cues.lint_cues`, the same two functions the burn and the
`lint` command use. A second implementation in JavaScript would be a second set of rules
about Serbian clitics, kept in sync by hope.

Then **"Looks right, make the video"** burns it. On the command line the same stop is
`subtitler run FILE --review`, which writes the `.srt` and `.vtt` and stops.

**Timings are never edited, only text.** The clock came from real word timings and is what
reading speed is measured against; a mouse that could drag it would put the one number a
human cannot judge by eye under the mouse. Cues are never added or removed either, which is
what keeps a correction addressable across re-runs.

### What the editor does to the cache, and what happens when the transcript changes

Corrections go into `<output>/.subtitler/<name>/edits.json`, which is **nobody's stage
artifact**. That is the entire design, and the two obvious alternatives are both wrong:

- write them into `cues.json` and the next run recomputes that stage, the key still
  matches, and `segments_to_cues` overwrites every correction without a word;
- put them in the `cues` stage's key and they survive, but now every keystroke invalidates
  a transcript-derived artifact that has nothing to do with them.

So they are read as the input of a stage of their own, `edit`, sitting between `cues`/`fix`
and `burn`. Its key is the upstream key plus a digest of the corrections, which means the
burn re-runs exactly when the text it would render changed, and never because the editor
was opened and closed again.

`edits.json` records the key of the cues it was written against. When you change the model,
the denoiser, the cue shape, or pass `--force transcribe`, that key moves, and the
corrections are **reported and skipped rather than applied**:

```
edit: 2 hand correction(s) in edits.json were made against a different transcript and are
NOT being applied. Open the editor again to redo them, or go back to the settings they
were made under.
```

Cue 41 of the old transcript is not cue 41 of the new one, and quietly re-pointing them at
whatever now holds that index is the worst available outcome. Nothing is deleted either:
switch back to the settings they were made under and they line up again and apply. Opening
the editor on the new transcript and correcting it there is the way forward.

A run where nobody has ever opened the editor writes no `edit` stage at all and has exactly
the cache keys it had before this existed.

## Usage

```bash
subtitler gui                                 # the browser interface (see above)
subtitler run INPUT.mp4                       # transcribe, shape cues, burn in
subtitler run INPUT.m4a --canvas 1920x1080    # audio-only input gets a video canvas
subtitler run INPUT.mp4 --srt-only            # sidecar files, no video work
subtitler run INPUT.mp4 --review              # write the subtitles and stop before the video
subtitler run INPUT.mp4 --soft-mux            # also a switchable subtitle track
subtitler run INPUT.mp4 --style-preset box    # outline | box | minimal
subtitler run INPUT.mp4 --engine groq         # cloud instead of local
subtitler run INPUT.mp3 --batch-size 16       # NVIDIA GPU: ~3x again, see below
subtitler run INPUT.mp4 --denoise speech      # none | afftdn | arnndn | anlmdn | speech
subtitler run INPUT.mp4 --fix                 # optional LLM correction pass
subtitler run INPUT.mp4 --fix --fix-model openai/gpt-4o
subtitler run INPUT.mp4 --force transcribe    # ignore the cache from a stage onwards

subtitler run URL -o DIR                      # fetch with yt-dlp, then the usual pipeline
subtitler run URL -o DIR --srt-only           # fetches the audio track only
subtitler run IN --start 10:00 --end 13:00    # keep a fragment; works on files and URLs
subtitler fetch URL -o DIR [--audio-only]     # download and stop

subtitler doctor [--install]                  # check and install system dependencies
subtitler models list | download | path | rm
subtitler burn VIDEO SUBS -o OUT [--preview]  # re-style without re-transcribing
subtitler lint SUBS.srt                       # check cue length, duration, reading speed
subtitler bench run | report                  # engine x denoiser x clip matrix, see below
```

## Fetching a URL

Any `http(s)` argument to `run` is a URL, and anything else is a path; there is no flag to
remember. It goes through [yt-dlp](https://github.com/yt-dlp/yt-dlp), so it is not YouTube
only, and it lives behind `uv sync --extra fetch` because a run over a local file has no
business importing it.

```bash
subtitler run "https://www.youtube.com/watch?v=..." -o ~/subs
```

- **`-o` is required for a URL.** A file input has a directory of its own to write beside;
  a URL has none, and this project never writes into the directory you happened to run it
  from. The download itself goes into `<output>/.subtitler/url-<hash>/`.
- **`--srt-only` downloads the audio track only.** On the clip in the table below that is
  3.7 MB instead of 20.3 MB. Otherwise it asks for an mp4 up to 1080p, since the burn
  re-encodes anyway and 4K would cost minutes for an overlay nobody inspects at that scale.
- **The outputs are named from the video's title**, slugified, diacritics intact.
- `subtitler fetch URL -o DIR` downloads and stops, if you would rather keep the file and
  iterate on it offline.

Failures that you can do something about arrive as a sentence rather than a traceback: a
private video, a region block, a dead network, an age gate, a rate limit, or a yt-dlp too
old for a site that changed under it (which names `--upgrade-package yt-dlp`).

**You are responsible for what you point it at.** Downloading and republishing someone
else's video is a question about copyright and about the site's terms of service, and this
tool answers neither. It will fetch whatever yt-dlp can reach; whether you have the right
to do that, and the right to publish what comes out, is yours to know.

## Keeping a fragment

`--start` and `--end` take `SS`, `MM:SS` or `HH:MM:SS` and work on files and URLs alike.

```bash
subtitler run lecture.mp4 --start 10:00 --end 13:00
```

Two things this gets right on purpose:

- **The cue timestamps are relative to the fragment.** Keep 10:00 to 13:00 and the first
  cue reads `00:00:00`, not `00:10:00`. That is not arithmetic applied afterwards: the cut
  happens before the audio is extracted, so everything downstream simply sees a file that
  begins where you asked.
- **The burn uses the fragment**, so the exported video is three minutes long. Burning onto
  the original would give you the full-length source carrying subtitles that match three
  minutes of it.

The cut is a stream copy, so it costs about a second rather than a re-encode. The price of
that is the one every stream copy pays: video can only be cut at a keyframe, so the
fragment may begin up to one keyframe interval early (1.2 seconds on the clip below). Only
the boundary is approximate; nothing inside the fragment is misaligned, because the
transcript is made from the fragment itself.

## A switchable subtitle track

`--soft-mux` adds the subtitles to the video as a track the viewer can turn off, without
re-encoding anything. It takes about a second and writes `NAME.softsubs.mp4` beside the
rest.

Which video the track goes onto is the only decision here. A source that already has a
picture gets it, because clean pixels plus a switchable track is the entire point of a soft
track: a run with `--soft-mux` on top of the default burn therefore hands back two files
that differ in exactly that. An audio-only input has no picture until the burn generates
one, so there the track rides along beside the rendered text on the generated canvas.
`--srt-only` asked for no video work at all and gets none; the run says so and carries on.

```bash
$ subtitler run talk.mp4 --soft-mux
...
muxed: talk.softsubs.mp4 (a switchable track on the source video)

$ ffprobe -v error -select_streams s -show_entries stream=index,codec_name:stream_tags=language \
    -of csv talk.softsubs.mp4
stream,2,mov_text,srp
```

MP4 carries `mov_text`, which drops all styling; a `.mkv` or `.webm` source is muxed to
Matroska instead, which carries ASS. The track is keyed on the same text the burn is, so a
correction typed in the editor re-muxes as well as re-burns.

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
| a correction typed in the editor | the burn (and the soft-muxed track), nothing above it |
| a different `--model` or device | transcribe onwards |
| `--start` or `--end` | the cut, then everything after it. **Not the download** |
| `--srt-only` on a URL that was fetched as video | the download, as audio this time |
| the input file's bytes | everything |

Stages, in the order `--force` treats as "and everything after": `fetch`, `trim`,
`extract`, `denoise`, `transcribe`, `cues`, `fix`, `edit`, `burn`, `mux`.

`--force` ignores the cache: bare `--force` for all of it, or `--force transcribe` for one
stage and everything after it. Deleting the `.subtitler` directory is always safe.

A download is the one stage that cannot be content-addressed, because there is nothing to
hash until it has happened. Its key is the URL plus which shape was asked for, and nothing
about the remote file: checking whether an upload changed would cost a network round trip
on every warm run, and a warm run is meant to be free and offline. So if the uploader
replaces the video behind a URL that did not change, `--force fetch` is how you say so.

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

## The benchmark

```bash
uv sync --extra bench
subtitler bench run                       # every local engine x every denoiser x every clip
subtitler bench run --engine faster-whisper,groq-turbo --device cuda --fix
subtitler bench report [RUN]              # recompute every metric from the kept transcripts
```

Each cell writes to `benchmarks/results/<UTC timestamp>/`: `results.json` (one record per
cell), `transcripts/` (every hypothesis, as text and as SRT), `env.json` (`doctor --json`
plus the OS, CPU, RAM, GPU and library versions) and a generated `report.md`. The git SHA is
recorded and a dirty tree is refused without `--allow-dirty`, because a result whose commit
does not describe the code that produced it is worse than no result.

**Cells run one per process.** Peak RSS is one of the numbers, and `getrusage` reports a
high-water mark that never comes down, so a second cell measured in the same interpreter
would inherit the first one's peak. They share the stage cache instead, clip-outermost, so
each clip is extracted once and each denoiser runs once no matter how many engines follow.

**Two kinds of number, and the report keeps them apart.** WER, WER_folded, CER and the
substitution/insertion/deletion split need a reference transcript. Realtime factor, wall
clock, peak memory, cue count, reading speed and the hallucination heuristics do not.
`benchmarks/references/` is currently empty, so **no WER is reported at all**: the harness
says so and emits the reference-free half rather than inventing a ground truth. Adjudicated
references are Phase 8. When one lands, `subtitler bench report` scores a run from months
ago against it without re-transcribing anything.

Text is normalized identically on both sides before scoring: NFC, Serbian Cyrillic to Latin
through a hand-written table, lowercase, punctuation to spaces, whitespace collapsed.
**WER_folded** repeats the score with `č ć` folded to `c`, `đ` to `dj`, `š` to `s` and `ž`
to `z`. The gap between the two is the most useful single number for Serbian: it separates
hearing the wrong word from writing `c` where `č` belongs. Digits and abbreviations are
deliberately left alone in v1, so `20` scores against `dvadeset` as an error; that inflates
every engine's number equally and the report says so next to the table.

What the first full matrix (30 cells on an RTX 3090, `benchmarks/results/`) actually found:

- **`--denoise arnndn` can lose speech.** On the 164 s law lecture it produced 232 words
  against 280 for every other preset, and opened with `Koristi ispravna imena za ljude,
  knjige, filozofske škole itd.` where the first fifty words belong: the tail of the Serbian
  steering prompt, echoed back as transcript. Sequential decoding, not batched. Only
  faster-whisper did this; both Groq models transcribed the same denoised audio normally.
  `metrics.prompt_echo` exists because of that cell, and the report calls it out by name.
- **The other three denoisers change almost nothing**, which is what the prior art said and
  the reason denoising is off by default.
- **Speed, on this box**: `groq-turbo` RTF 0.006 to 0.015, `groq` 0.012 to 0.018, local
  `large-v3` on CUDA float16 0.041 to 0.047 at 3.4 GB peak RSS. Whether the local transcript
  is *better* is exactly the question no reference can answer yet.

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
| Pop!_OS 22.04, RTX 3090 | `subtitler bench run`, 30 cells | 2 clips x 5 denoisers x faster-whisper/groq/groq-turbo, every cell ok, 4 minutes |
| Pop!_OS 22.04, RTX 3090 | the `--fix` axis, 12 cells | `openai/gpt-4o` rewrote 2.0% to 15.4% of the words, depending on the clip |
| Pop!_OS 22.04 | `subtitler bench report` | recomputed every metric from the kept transcripts, no model reloaded |
| Any | WER, CER, WER_folded on real clips | not verified: no reference transcript exists yet. The code path is unit-tested, nothing more |
| Pop!_OS 22.04, Chrome | `subtitler gui` | picked a file, set the options and started a run from the page; 109s of Serbian burned in, 21 cues, 12s |
| ubuntu-latest CI | `subtitler gui` | binds, serves the page, refuses an untokened call, reports dependencies, completes a faster-whisper run through the API, headless |
| macOS 14 Apple Silicon CI | `subtitler gui` | the same, reporting `Darwin arm64 brew:/opt/homebrew` and `Show in Finder`, transcribing on mlx |
| Any Mac | `subtitler gui` reveal in Finder | not verified on hardware: `open -R` is covered by a test with a faked `Platform`, never by a Mac |
| Pop!_OS 22.04, RTX 3090 | `run URL --start 1:00 --end 1:45` | 23.3s cold, end to end: 20.3 MB fetched, cut, transcribed, burned. See the note below |
| Pop!_OS 22.04, RTX 3090 | the same command again | 0.36s, every stage cached, no network |
| Pop!_OS 22.04, RTX 3090 | the same with `--start 1:10` | 17.6s, `fetch: cached`. Moving the window re-cuts, it does not re-download |
| Pop!_OS 22.04, RTX 3090 | the same with `--srt-only` | fetched 3.7 MB of audio instead of 20.3 MB of video |
| Pop!_OS 22.04, ffmpeg 4.4.2 | `run FILE --start 0:20 --end 0:50` | 30.0s of the Serbian fixture, burned output exactly 30.0s |
| Pop!_OS 22.04, no `--extra fetch` | a URL run | `error: downloading a URL needs yt-dlp. fix: uv sync --extra fetch`, and the whole suite still passes |
| Pop!_OS 22.04 | yt-dlp failure messages | unavailable, 404, and a dead DNS resolver each produce one sentence, verified against real yt-dlp output |
| Pop!_OS 22.04, Chrome | the cue editor, end to end | picked the Serbian fixture in the page, landed in the editor after 8s, corrected two cues, heard cue 2 through the Listen button (a `206` off `/api/media`), approved, and read the corrected text off a frame pulled out of the burned mp4 |
| Pop!_OS 22.04, Chrome | the live quality check | a deliberately unreadable correction was marked `3 lines (max 2)` and `28.9 chars/sec exceeds 20.0` as it was typed, and libass rendered exactly the three lines the wrapper chose |
| Pop!_OS 22.04, RTX 3090 | corrections against the stage cache | survived a `--force cues`, left the burn cached on a second approval, re-burned when the text changed, and were reported and skipped after `--denoise afftdn` moved the transcript |
| Pop!_OS 22.04, ffmpeg 4.4.2 | `--soft-mux` | `ffprobe` reports stream 2 as `mov_text` tagged `language=srp`, and extracting it back out returns the corrected text |
| Pop!_OS 22.04, Chrome | a link and a trim window from the page | pasted a YouTube URL with 0:05 to 0:20, watched yt-dlp's progress in the log, landed in the editor with 4 cues starting at `00:00.000`, approved, burned |

The URL row above is [What's Up: June 2026 Skywatching Tips from
NASA](https://www.youtube.com/watch?v=NtiKxO8xIbY), 229 seconds, from NASA JPL's official
channel, which is US government work and therefore public domain. Keeping 1:00 to 1:45 gave
a 46.2-second fragment (1.2 seconds of keyframe slop at the head, as documented above), the
first cue reads `00:00:00,000`, and `ffprobe` puts the burned mp4 at 46.23 seconds, which
is the fragment and not the 229-second source.

CI never downloads anything. yt-dlp is stubbed in `tests/test_fetch.py` the way LiteLLM is
in `tests/test_postedit.py`, and the trim is asserted as an argv list. A test that hits
YouTube is slow, rate-limited and fails for reasons that have nothing to do with this code.

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
