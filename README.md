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
subtitler run INPUT.mp4 --fix                 # optional LLM correction pass

subtitler doctor [--install]                  # check and install system dependencies
subtitler models list | download | path | rm
subtitler burn VIDEO SUBS -o OUT [--preview]  # re-style without re-transcribing
subtitler lint SUBS.srt                       # check cue length, duration, reading speed
subtitler bench run | report                  # engine and denoiser quality matrix
```

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
| Pop!_OS 22.04, ffmpeg 4.4.2 | groq | end to end: transcribe, cues, srt/vtt, burn-in, diacritics |
| ubuntu-latest CI, ffmpeg 7.x | n/a | doctor, tests, burn-in, hostile paths, diacritics |
| macOS 14 Apple Silicon CI, ffmpeg-full 8.1.2 | n/a | doctor, tests, burn-in, hostile paths, diacritics |
| Pop!_OS 22.04 | faster-whisper | not yet verified (lands in Phase 3) |
| macOS 14 Apple Silicon | mlx | not yet verified (lands in Phase 3) |

Both CI runners render `ČĆĐŠŽ čćđšž` identically from the bundled font, which is what the
bundling is for. Transcription on a real Mac is still unverified: CI proves the path runs,
not that mlx is any good at Serbian. That is what the benchmark is for.

This table is updated only when CI or a human actually runs it there. Nothing is listed as
working because it ought to.

## License

MIT. Bundled Noto Sans is SIL OFL 1.1 (see `subtitler/assets/fonts/`). Whisper weights are
MIT. ffmpeg is invoked as a subprocess and is not linked.
