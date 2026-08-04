# PRD: subtitler

Status: draft, Phase 0. Amend this document when a benchmark result contradicts it.

## 1. Goal

Give a video or audio file containing speech, produce `.srt`, `.vtt`, and a subtitled
`.mp4`, on macOS Apple Silicon or Linux, with one command, without sending anything to a
cloud service by default.

## 2. Users and stories

**Primary. The friend.** He recorded himself speaking Serbian on video and wants subtitles
burned in so the clip plays on social media. He is on a Mac. He should need one command, no
configuration, and no API key. Success is a watchable file.

**Secondary. The archive.** The `gozba` radio-show workflow: long Serbian audio, a sidecar
`.vtt` for the web player, optionally an LLM correction pass with domain formatting. 353
episodes exist, 26 have transcripts. This is the workload the prior bash pipeline served.

**Tertiary. The maintainer.** Deciding whether local `large-v3` actually beats
`groq/whisper-large-v3-turbo` on Serbian, with numbers rather than by opening `meld`.

## 3. Non-goals

Translation. Speaker diarization. Live or streaming input. A GUI. Windows. A hosted service.
Docker or CUDA container images (the prior art had these and they were dead weight).
Languages other than Serbian as a *tuned* target: others work, they are just not tuned.

## 4. Product surface

| Command | Produces |
|---|---|
| `subtitler run INPUT` | `<stem>.srt`, `<stem>.vtt`, `<stem>.subbed.mp4` |
| `subtitler doctor [--install]` | dependency report; exit 1 if anything required is missing |
| `subtitler models list/download/path/rm` | model cache management |
| `subtitler burn VIDEO SUBS` | re-style without re-transcribing; `--preview` gives stills |
| `subtitler lint SUBS` | cue-quality violations, exit 1 if any |
| `subtitler convert IN -o OUT` | verbose_json, srt, and vtt in any direction |
| `subtitler bench run/report/agents` | engine x denoiser quality matrix |

## 5. Acceptance criteria

Each is a test, not an aspiration.

1. **Zero-config first run.** On a clean machine, after `subtitler doctor --install` and
   `subtitler models download`, `subtitler run fixtures/gozba-sample.mp3` produces all three
   artifacts with no manual step and no traceback.
2. **Diacritics render on both platforms.** Burned-in `č ć đ š ž` matches a checked-in
   reference PNG within tolerance on both the `ubuntu-latest` and `macos-14` CI runners.
   This is the test that catches a font regression; nothing else will.
3. **Cue quality is enforced, not hoped for.** No cue exceeds 42 characters per line, 2
   lines, 7.0 seconds, or 20 characters per second. No cue is shorter than 1.0 second.
   `subtitler lint` exits non-zero on any violation and runs on every produced file.
4. **Local beats cloud, or the default changes.** The default local engine achieves
   WER <= 15% normalized against `benchmarks/references/`, and beats
   `groq/whisper-large-v3-turbo` on the same clips. If it does not, the default flips to
   whatever wins and this document is amended with the number. The 15% figure is a target,
   not a prediction: there is no credible published Serbian `large-v3` number to anchor on.
5. **Faster than real time on a Mac.** A 10-minute video completes end to end on an
   M-series Mac with RTF < 1.0.
6. **Re-runs are free and byte-identical.** Running the same command twice hits the stage
   cache; the second run finishes in under 2 seconds and produces identical output.
7. **Nasty filenames work.** Burning onto a path containing a space, a colon, and a single
   quote succeeds. This is the acceptance test for the filter-path strategy.

## 6. Decisions carried over from the prior art

Each of these is a conclusion from reading the existing code and its outputs, not a guess.

- **Denoising is off by default.** All five working denoisers in `gozba2/py/test.py`
  produced roughly 95% identical Serbian text on both sample runs, and the last script
  written in that experiment kept only one of them. Denoising stays as a pluggable,
  benchmarked stage. It is not the centerpiece, and it does not justify a compiled
  dependency.
- **Cue splitting is the actual differentiator.** Every prior tool did naive one segment to
  one cue. Readable burned-in subtitles need real splitting, wrapping, and reading-speed
  limits, and no prior tool had any of the three.
- **Cloud is a baseline and a fallback, not the default.** The friend's video should never
  need to leave his laptop.
- **No compiled dependencies.** The vendored `rnnoise` checkout is x86-64 ELF with dangling
  symlinks into `/usr/share/automake-1.16`. ffmpeg ships `arnndn`, which is the same
  algorithm, on both platforms.
- **Fonts are bundled.** No font ships on both macOS and stock Ubuntu, so any system font
  name renders differently per machine.

## 7. Platform matrix

| Platform | Engine | Denoise | Burn-in | Status |
|---|---|---|---|---|
| macOS 14+, Apple Silicon | mlx | ffmpeg filters | libass via Homebrew ffmpeg | primary target |
| macOS, Intel | faster-whisper (CPU) | same | same | supported, untuned |
| Ubuntu/Debian/Pop!_OS, CPU | faster-whisper (int8) | same | same | dev platform |
| Ubuntu/Debian/Pop!_OS, CUDA | faster-whisper (float16) | same | same | dev platform |
| Windows | none | n/a | n/a | out of scope |

Minimum ffmpeg is 4.4 (what Ubuntu 22.04 ships). Homebrew ships 7.x/8.x. Every flag used
must be in the intersection of the two.

## 8. Deferred to v2

Soft-mux subtitle tracks into MKV with chapters. Batch and watch modes. `--translate`.
Snapping cue boundaries to shot changes. Automatic Whisper prompt tuning. A waveform canvas
for audio-only input instead of a flat colour.

## 9. Open questions

1. Is 42 characters per line right for a 9:16 social crop? A vertical video needs closer to
   28. Should the splitter be aspect-ratio aware? Decide before Phase 4.
2. Should the friend's default style preset be `box` rather than `outline`? Decide from the
   `--preview` stills, not from theory.
3. The benchmark has no clip representing the primary user story: a single speaker on camera
   in a noisy room. Both existing fixtures are archive audio. Does that clip exist, or does
   it need recording first?
4. Does `--fix` improve WER or hurt it? An LLM that "corrects" grammar changes words the
   speaker actually said. Measured as its own axis; no assumption either way.
