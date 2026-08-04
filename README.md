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

## Usage

```bash
subtitler run INPUT.mp4                       # transcribe, shape cues, burn in
subtitler run INPUT.m4a --canvas 1920x1080    # audio-only input gets a video canvas
subtitler run INPUT.mp4 --srt-only            # sidecar files, no video work
subtitler run INPUT.mp4 --style-preset box    # outline | box | minimal
subtitler run INPUT.mp4 --engine groq         # cloud instead of local
subtitler run INPUT.mp4 --denoise speech      # none | afftdn | arnndn | anlmdn | speech
subtitler run INPUT.mp4 --fix                 # optional LLM correction pass
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

## Engines

| Engine | Where | Notes |
|---|---|---|
| `mlx` | macOS Apple Silicon | default on a Mac. `mlx-community/whisper-large-v3-mlx` |
| `faster-whisper` | Linux, Intel Mac, any CPU | default elsewhere. CUDA when available |
| `groq` | anywhere with a key | `whisper-large-v3` and `-turbo`, for comparison or long files |

The engine is chosen automatically from the platform. Asking for one explicitly that is not
installed is a hard error with the exact `uv sync` command to fix it, never a silent fallback.

## Verified on

| Platform | Engine | Status |
|---|---|---|
| Pop!_OS 22.04, ffmpeg 4.4.2 | groq | end to end, including burn-in |
| Pop!_OS 22.04, ffmpeg 4.4.2 | faster-whisper large-v3, CPU int8 | 109s of Serbian at RTF 0.68 |
| Pop!_OS 22.04, ffmpeg 4.4.2 | stage cache | 83.9s cold, 0.52s warm, all three outputs byte-identical |
| Pop!_OS 22.04, ffmpeg 4.4.2 | all five denoise presets | each runs and changes the audio |
| ubuntu-latest CI | faster-whisper tiny | transcribe, burn-in, hostile paths, diacritics |
| macOS 14 Apple Silicon CI, ffmpeg-full 8.1.2 | mlx tiny | transcribe (RTF 0.40), burn-in, hostile paths, diacritics |
| Any | faster-whisper on CUDA | not verified: this dev box has CUDA 13 and CTranslate2 wants 12 |

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
