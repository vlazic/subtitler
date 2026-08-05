# Architecture

Short on purpose. Every module in this project carries a docstring that argues for its own
design, and those are the detail; this page is the map that says which one to open. Read it
alongside `docs/PRD.md` (what "done" means) and `docs/STATUS.md` (what is left).

## The pipeline

One run is a sequence of stages. Square brackets are conditional.

```
[fetch] -> [trim] -> probe -> extract -> [denoise] -> transcribe -> cues -> [fix]
        -> [edit] -> render -> [burn] -> [mux]
```

`subtitler/pipeline.py` sequences them and owns the work directory, and does nothing else:
every stage is a function that lives elsewhere and can be tested without a pipeline.

| Stage | Module | What it does, and the one thing worth knowing |
|---|---|---|
| `fetch` | `fetch.py` | yt-dlp downloads a URL. It asks the site for the **window**, so a 60-second excerpt of a four-hour lecture transfers 60 seconds. Optional extra; a local-file run never imports it |
| `trim` | `media.py` | Cuts the fragment out *first*, so every later stage sees a file that begins at the window and cue timestamps need no offset arithmetic. A stream copy. ffmpeg's exit code is not evidence a fragment exists, so the result is probed before it is cached |
| `probe` | `media.py` | ffprobe. Run **after** the trim: a stream copy lands on a keyframe, so the fragment's real duration is a fact, not `end - start` |
| `extract` | `media.py` | 16 kHz mono WAV, which is what every recognizer wants |
| `denoise` | `media.py` | Five ffmpeg filter presets. A **separate pass** from extraction, so changing the preset does not re-demux a 3 GB source. Off by default: measured, it barely helps |
| `transcribe` | `engines/` | One adapter per backend behind `base.Engine`. Shared post-decode hygiene lives in `engines/base.py`, not per adapter |
| `cues` | `cues.py` | Segments to display cues: split, wrap, merge, over word-level timings. The quality-critical file, and the one that separates a readable subtitle from a wall of text |
| `fix` | `postedit.py` | Optional LiteLLM correction pass. Text only, with index validation, so a model cannot move a timestamp. The only stage that costs money, which is why its caching matters most |
| `edit` | `edits.py` | Hand corrections from the editor. Its input is `edits.json`, a file no stage writes |
| `render` | `render.py` | SRT and VTT, plus the validator. **Not cached**, deliberately: re-deriving it every run is what keeps "the second run is byte-identical" a fact that gets re-established rather than an artifact nobody touched |
| `burn` | `burn.py` | A real `.ass` file and three ffmpeg commands. `cues.py` owns line layout and libass never wraps |
| `mux` | `burn.py` | `--soft-mux`: the same subtitles as a switchable track |

Two surfaces drive that pipeline and neither contains a second copy of it:
`subtitler/cli.py` (argparse, subcommands only, no logic) and `subtitler/gui/` (a native
Tk window by default, a browser page as the fallback, both over `gui/session.py`).

## The cache key chain

`subtitler/cache.py` is the authority here and explains every entry; the shape is:

```
fetch      <- the URL, the shape asked for, and the window
trim       <- the source file's content id (or fetch's key)
extract    <- the source file's content id (or trim's key)
denoise    <- extract's key
transcribe <- the audio stage's key (denoise if denoising, else extract)
cues       <- transcribe's key
fix        <- cues' key
edit       <- fix's key (or cues') + a digest of the hand corrections
burn       <- the text stage's key + the source content id
mux        <- the text stage's key + the id of the video the track is attached to
```

Each stage writes `<stage>.meta.json` beside its artifact, recording the key, the input
hash and the exact parameters. A stage is skipped when the key recomputed from *this* run
matches the one on disk **and** every artifact it claims to have written still exists.

Three properties follow from chaining rather than hashing the command line:

- A change invalidates exactly what is downstream of it. `--style-preset` re-burns without
  re-transcribing; `--denoise` re-runs the denoiser without re-extracting.
- `--force <stage>` invalidates that stage and everything after it, because a new
  transcript makes cues computed from the old one meaningless.
- The rule a new stage must obey: **its params contain everything that can change its
  output and nothing that cannot.** Too little serves a stale artifact, too much means the
  cache never hits. `transcribe` keys on `engine.describe()` rather than on `--model`,
  because int8 on CPU and float16 on CUDA are different transcripts from the same name.

`commit()` is separate from `begin()` so a crash halfway through leaves a stage that misses
next time rather than one claiming a result it never wrote. There is **one slot per stage,
not one per key**: `denoise.wav` is overwritten when the preset changes, which keeps a work
directory human-readable at the cost of not caching both sides of an A/B.

## The data model

`subtitler/model.py`. Four frozen dataclasses, and every stage artifact is JSON so the
cache can be inspected by hand.

```
Transcript ── segments ──> Segment ── words ──> Word
                                                   \
                                            (cues.py)
                                                     \
                                                      v
                                                     Cue
```

- **`Word`** is `start`, `end`, `text`, `prob`. Word-level timing is the contract of this
  project: `cues.py` needs it to pick split points, and every engine can supply it. When
  one does not, `synthesize_words` distributes the span by character length, which is a
  degraded fallback and never a fatal error.
- **`Segment`** is one utterance as the recognizer emitted it, plus the decoder's own
  diagnostics: `no_speech_prob`, `avg_logprob`, `compression_ratio`. Those are what the
  speech-free gate in `engines/base.py` reads; `docs/STATUS.md` records what they are
  actually worth, measured.
- **`Cue`** is one *displayed* subtitle, and its `lines` are already wrapped. Nothing
  downstream re-wraps: anything that edits cue text has to go back through
  `cues.wrap_edited`.
- **`Transcript`** carries the segments plus the engine, model, pinned revision, runtime
  and `params`. `params` is also how an engine reports what it did to its own output
  (`prompt_echo_retry`, `silence_dropped`, `speechless_dropped`), because an engine has no
  log channel and the pipeline has to be able to warn about a transcript served from the
  cache.

`RunResult` in `pipeline.py` is what a run hands back. Note that `lint` and `warnings` are
different channels on purpose: a `lint` entry says a cue is hard to read, while a
`warnings` entry says the text may not be what anybody said.
