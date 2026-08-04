# Prior art

This project consolidates several earlier experiments. This file records what was taken,
what was dropped, and the bugs found on the way, so none of it gets re-litigated or
reintroduced.

## Sources

| Source | What it was | Verdict |
|---|---|---|
| `gozba2/emisije/transcribe-audio.sh` | ~380 lines of bash: denoise, downsample, Groq Whisper with a Serbian steering prompt, JSON to VTT via a Deno script fetched from GitHub at runtime, chunked GPT-4o post-edit, markdown fixup, intro removal, VTT validation | The pipeline **shape** is the blueprint. The implementation is replaced. |
| `gozba/transcribe-audio.sh` | An earlier, simpler revision of the same script | Superseded. Its LLM step passed the whole VTT in one request with no chunking. |
| `gozba2/py/test.py` | A six-denoiser benchmark | Algorithms ported into `bench/` with bugs fixed. Not runtime code. |
| `gozba2/py/groq_to_vtt.py` | Groq call plus JSON to VTT plus the only burn-in command that exists anywhere | Both reimplemented. The burn recipe is the starting point for `burn.py`. |
| `gozba2/py/denoise_and_encode.py` | Video in, denoised video out, spectral subtraction only | The ffmpeg extract and remux shape is reused; the Python denoiser is not. |
| `record-audio/record_audio/transcribe.py` | faster-whisper transcription with hard-won production constants | The single most reusable file. Constants and helpers lifted directly. |
| `json-verbose-to-vtt-converter/main.ts` | Deno tool: Whisper verbose_json to VTT plus validation | Reimplemented in Python so the runtime `deno run <url>` dependency disappears. |

## Kept

- The pipeline shape: denoise, 16 kHz mono, Whisper with a pinned language and a steering
  prompt, cues, optional LLM correction, render, burn.
- The verbatim Serbian steering prompt.
- VTT validation as a real step rather than an assumption.
- The `GROQ_API_KEYS` comma-separated key pool with random per-request selection (minus the
  original's habit of echoing the first 15 characters of the key to stderr).
- From `record-audio`: `SILENT_PEAK_DBFS = -60.0` (Whisper hallucinates on silence, so
  silent spans are dropped rather than filtered afterwards), `MAX_REPEATS = 6` and
  `MAX_REPEAT_PHRASE_TOKENS = 8` (repetition-loop collapse), a fixed random seed,
  `preload_cuda_libraries()` for the CUDA 12 vs 13 wheel mismatch, and the integer-milliseconds
  SRT clock.
- The two Serbian test fixtures and a real Groq `verbose_json` response as a converter golden.

## Dropped

- **The Docker and Makefile stack.** `nvidia/cuda` base image, `ppa:deadsnakes`, cu121 torch,
  `nvidia-container-toolkit`, `nvidia-ctk`, `systemctl restart docker`. All Linux and NVIDIA
  specific; all irrelevant when the target is a Mac.
- **DeepFilterNet.** It never worked (see bug 7 below) and its macOS wheels are unreliable.
- **The vendored `rnnoise` C checkout.** x86-64 ELF binaries plus dangling symlinks into
  `/usr/share/automake-1.16`. Dead on Apple Silicon. ffmpeg's `arnndn` filter is the same
  algorithm and ships on both platforms.
- **`nnnoiseless`.** Works, but needs `cargo install`, which is a compiled dependency the
  friend should not have to acquire. Same reason: `arnndn` covers it.
- **`meld`-based comparison.** Replaced by WER/CER.
- **The Deno runtime dependency** for VTT conversion and validation.
- **`remove_intro.py`** as a default step. It is gozba-specific (it matches five spellings
  of one radio station's ident). It survives as an opt-in `--drop-intro-phrases FILE`.

## Empirical findings

Read out of the committed outputs rather than assumed:

1. **Denoiser choice barely matters.** In `gozba2/py/denoised_output-2/` and
   `denoised_output_gozba/`, all five working denoisers produced roughly 95% identical
   Serbian text. RNNoise's higher word count is a duplicated opening phrase, not extra
   content. noisereduce-conservative had the cleanest segmentation. The author's own final
   script kept only spectral subtraction and dropped the rest.
2. **`whisper-large-v3-turbo` makes clear Serbian word errors.** Visible in the committed
   VTTs: `povenuli` for `pomenuli`, `državne gane` for `državne organe`. This is the
   hypothesis the benchmark tests: full `large-v3` should beat turbo by enough to justify
   the local path.
3. **Cue length is the real problem.** Every produced VTT has one cue per Whisper segment,
   5 to 25 seconds each, unwrapped. Fine for the scrolling transcript player, unusable
   burned into video.

## Bugs found, and their fixes

1. `groq_to_vtt.py::format_timestamp` uses `f"{seconds:06.3f}"`, so `59.9996` renders as
   `60.000`, producing the invalid timestamp `00:00:60.000`. **Fix:** convert to integer
   milliseconds first, as in `record-audio`'s `srt_clock`.
2. `groq_to_vtt.py::json_to_vtt` seeds the list with `"WEBVTT\n"` and joins with `"\n"`
   while each cue also begins with `"\n"`, giving two blank lines after the header.
   **Fix:** build cue blocks and `"\n\n".join(["WEBVTT", *blocks])`.
3. `test.py` Wiener filter: `noise_frames = int(self.sr / D.shape[1])` divides the sample
   rate by the frame **count**, not the hop length, so it profiles about 4 frames instead
   of the 1 second the comment claims. **Fix:** `int(1.0 * sr / hop_length)`.
4. `test.py` and `denoise_and_encode.py` spectral subtraction: `int(sr / 1024)` is the same
   class of bug (1024 is `n_fft`, not `hop_length`). At 22050 Hz it profiles about 0.24 s.
   **Fix:** `int(noise_seconds * sr / hop_length)`.
5. `denoise_and_encode.py` extracts stereo (`-ac 2`) and then lets `librosa.load` silently
   downmix to mono, so the stereo extraction is wasted work.
6. `denoise_and_encode.py` uses fixed temp filenames in the current working directory, which
   collide on concurrent runs, and its `except` block cleans up names that may be unbound if
   the failure happened early. **Fix:** `tempfile.TemporaryDirectory`.
7. `test.py` DeepFilterNet never worked: `enhance()` was passed a file path where it expects
   a tensor, and `output_dir`/`pad` are not in its signature. Proof: no
   `deepfilternet_output.wav` exists in either output directory while all five other
   algorithms produced files. Recorded so it is not retried blind.
8. `transcribe-audio.sh::process_audio` checks `$?` after a different command than the one
   it means to test, and `set -o errexit` makes the branch dead anyway.
9. `transcribe-audio.sh::split_vtt_file` chunks by counting blank lines with `awk`, not by
   counting cues, so it can split in the middle of a cue. **Fix:** batch over parsed cue
   objects.
10. `json-verbose-to-vtt-converter/main.ts::formatTime` uses
    `Math.floor((seconds % 1) * 1000)`, truncating rather than rounding milliseconds.
11. `remove_intro.py` drops everything before the matched intro phrase, and its `WEBVTT`
    re-insertion inspects a line that may be a timestamp.
12. The `groq_to_vtt.py` burn-in recipe has three problems: it relies on `-shortest` against
    an infinite `lavfi` source, which is unreliable on ffmpeg 4.x; it omits
    `-pix_fmt yuv420p`, so the output may not play in QuickTime or Safari, which matters
    when the target user is on a Mac; and it names `FontName=Arial`, which does not exist on
    stock Ubuntu and silently falls back to a font with different metrics, so the same
    subtitle file lays out differently on different machines.
13. `bash-helpers.sh` calls `error()` in both gozba repos without ever defining it.
