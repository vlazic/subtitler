# Status: where the project stands and what is left

What this document is for: a reader who has not been here before should be able to learn
the current state, the findings that were expensive to acquire, and the outstanding work,
without re-deriving any of it from the code or the git history.

Two companion documents. `docs/PRD.md` says what this is and what "done" means.
`docs/architecture.md` says how the pipeline is put together. `docs/prior-art.md` records
what was salvaged from `gozba2`, what was dropped, and the bugs fixed on the way.

---

## 1. Where it stands

| | |
|---|---|
| Repo | `https://github.com/vlazic/subtitler`, public, `main` |
| Tests | **919 passed, 1 skipped** |
| CI | green on `ubuntu-latest` **and** `macos-14` |
| Speed | 51x realtime on an RTX 3090 (`--batch-size 16`), RTF 0.015 |
| Quality | local `large-v3` **9.7% WER** pooled, beating groq-turbo 11.1% and groq large-v3 27.3% |

Every WER number above is provisional. See section 3 and the caveat in `CLAUDE.md`: the
reference transcripts are an LLM-adjudicated consensus of the very engines they score, and
carry `*` until `benchmarks/references/review-queue.md` has been worked through with the
audio.

**What it does.** Takes an audio file, a video file, or a URL. Optionally trims to a
window. Optionally denoises. Transcribes locally (mlx on Apple Silicon, faster-whisper
elsewhere, Groq as a cloud baseline). Shapes the output into cues a human can actually
read. Optionally corrects them with an LLM. Lets you read and edit them before anything is
encoded. Writes `.srt` and `.vtt` and burns the subtitles into a video. Re-runs hit a
content-addressed cache and are effectively free. There is a desktop window and a
double-clickable app icon for people who do not use a terminal.

---

## 2. What was built

Phases 0 through 8 plus a CUDA phase, each verified by running rather than by assertion.

| Phase | Delivered |
|---|---|
| 0 | Repo, `pyproject.toml` (uv + hatchling), `CLAUDE.md`, `docs/PRD.md`, `docs/prior-art.md`, fixtures |
| 1 | probe, extract, Groq engine, SRT/VTT, ASS burn-in, **macOS CI from day one** |
| 2 | `subtitler doctor`: declarative dep table, injectable `Platform`/`Probe`, both OS branches unit-tested from Linux |
| 3 | Engine protocol, mlx + faster-whisper + Groq, `subtitler models` with SHA-pinned revisions |
| 4 | **The cue splitter.** 11 segments to 20 cues, zero lint violations, Serbian clitic and preposition rules |
| 5 | Content-addressed stage cache: **84s cold, 0.52s warm, byte-identical**. Five ffmpeg denoise presets |
| 6 | LiteLLM correction pass, text-only with index validation, prompts in files |
| 6b | **CUDA on the 3090.** 17x sequential, 51x batched. `doctor` GPU checks |
| 7 | Benchmark harness: matrix runner, Serbian normalization, WER/CER/WER_folded, hallucination heuristics |
| 8 | LLM-adjudicated reference transcripts plus an adversarial critic, then real WER numbers |
| extra | Browser GUI, URL fetch via yt-dlp, review-and-edit step, native desktop window, `install-app` |

The last row was **not requested**. See section 6.

---

## 3. Findings that cost time and must not be relearned

These are the expensive ones. Most are recorded in the `CLAUDE.md` gotchas as well.

1. **macOS needs `brew install ffmpeg-full`, not `ffmpeg`.** Homebrew split the formula and
   the regular bottle has no libass, so burn-in cannot work at all. Verified from the CI
   runner's own configuration line. Single most important macOS setup fact.
2. **`No option name near '...'` means the ffmpeg filter is MISSING**, not that the syntax
   is wrong: option parsing fails before the name is resolved. Naming every filter option
   (`ass=f=subs.ass:...`) makes a missing filter say so plainly.
3. **The CUDA preload had never loaded a library on any machine.** The pip `nvidia-*`
   packages are namespace packages, so `__file__` is `None` and `Path(module.__file__ or
   "").parent` silently became `Path(".")`. Use `__path__`.
4. **CTranslate2 initializes the device lazily**, so a bad CUDA setup fails during decode,
   not at construction. No probe catches it; the fallback must wrap the actual work.
5. **Whisper echoes its own steering prompt as transcript** when there is no speech.
   `prompt_reset_since` only advances at the end of the first 30-second window, so window
   one is exposed at every duration. Fixed by detect-then-re-decode, not a duration
   threshold.
6. **No font ships on both macOS and stock Ubuntu**, so a system font name renders
   differently per machine. Noto Sans is bundled.
7. **Filter path escaping is unwinnable**; write `subs.ass` to a temp dir and run ffmpeg
   with `cwd` set there instead.
8. **tkinter cannot be a pip dependency** (C extension, no PyPI package), but uv's
   python-build-standalone interpreter **bundles it**, so `make setup` gets a window on
   macOS and Linux both. Only a Homebrew or system Python lacks it.
9. **Pop!_OS is `ID=pop`, `ID_LIKE="ubuntu debian"`.** Match on both.
10. **Current Claude models reject `temperature`/`top_p`/`top_k` with a 400.** The
    correction pass must not send sampling parameters by default.
11. **One of the two keys in `GROQ_API_KEYS` is dead.** Random per-attempt selection made
    runs fail a different random half each time. Fixed by shuffling once and giving every
    key a turn. **The dead key is still in the maintainer's `.env` and should be pruned.**
12. **Over speech-free audio, `no_speech_prob` and word confidence point the wrong way.**
    Measured, not assumed. Ten seconds of titles and music produced `Hvala što pratite
    kanal.` at `no_speech_prob` 0.14 and mean word confidence 0.80, while genuine speech in
    this project's own fixtures reaches 0.436 and falls to 0.779. The model is confident
    because it is reciting memorised YouTube outro text rather than guessing at audio, so
    no threshold on either signal alone can separate them. What does separate them is that
    genuine filler is maximally confident: the real `Hvala.` in `fixtures/gozba-sample.mp3`
    scores 1.000, in a transcript running at 1.40 words per second, while a hallucinated
    one sits at 0.80 in a transcript running at 0.13 to 0.40. `engines/base.is_speechless`
    is built on exactly that difference.

### The measurements behind the speech-free gate

Kept here because the thresholds in `engines/base.py` are only defensible next to the
numbers they were chosen from. Both fixtures, 25 segments of real speech, against
ffmpeg-generated music-only clips.

| Signal | Real speech (25 segments) | Hallucination over music | Threshold |
|---|---|---|---|
| `no_speech_prob` | max **0.436** | 0.14 to 0.19 (0.80 to 0.94 when the head does fire) | drop at 0.60 |
| mean word confidence | min **0.779** | 0.797 to 0.856 | drop at 0.35 |
| words per second | 1.40 (`gozba-sample`), 1.70 (`uvod-u-pravo`) | 0.13 to 0.40 | filler test needs < 0.5 |
| confidence of filler text | **1.000** (the genuine `Hvala.`) | 0.797 to 0.856 | filler test needs < 0.95 |

---

## 4. TO-DO

### 4.1 Needs the maintainer, cannot be delegated

- [ ] **Resolve the 44 spans in `benchmarks/references/review-queue.md`.** Roughly twenty
      minutes with the audio. This is the single highest-value item: it flips
      `human_verified` to `true` and turns **every WER number in the project** from
      provisional into real, retroactively, for every run already recorded (`subtitler
      bench report` rescores from saved transcripts without re-transcribing).
- [ ] **Phase 9: put it in the friend's hands on a real Mac.** CI proves the mac path runs.
      It cannot prove the Tk window opens rather than hangs, and it cannot prove he can use
      it. Have him run `make setup`, `subtitler install-app`, then double-click.
- [ ] **Decide what to do about the unrequested scope** (section 6).
- [ ] **Prune the dead Groq key** from `.env`.
- [ ] Optional: `rm ~/.local/share/applications/subtitler.desktop`, left installed on the
      dev machine by an agent, pointing at this checkout's venv.

### 4.2 Untested, and the results would change decisions

- [ ] **Does the Tk window open or hang on macOS?** CI passed a `subtitler gui` invocation
      on `macos-14`, but a hang fits the same evidence as success. Unproven either way.
- [ ] **mlx transcription quality on Serbian is unmeasured.** CI only ever runs the `tiny`
      model, which is deliberately too weak to judge. The benchmark's mlx axis has never
      run, because the dev box is Linux. Run `subtitler bench run` on the Mac.
- [ ] **The gozba archive batch run.** 353 episodes, about 260 hours. At the measured 51x
      this is a few hours rather than the 9.4 days it would take on CPU. This is the
      secondary user story and the reason CUDA was worth doing.
- [ ] **Batched CUDA correctness at scale.** `--batch-size 16` was measured for speed and
      spot-checked for correctness; it has not been run across the whole matrix.
- [ ] **A noisy single-speaker clip** (PRD open question 3). Both fixtures are archive
      audio. The primary user story, someone talking to a camera in a room, is
      unrepresented in every quality number this project has produced.

### 4.3 Known defects, ranked by whether they can hurt someone

- [x] **Speech-free audio produced confident hallucination.** Ten seconds of titles and
      music returned `Hvala što pratite kanal.` with no warning; the `-60 dBFS` silence
      gate missed it because music is loud. Fixed by `engines/base.drop_speechless_segments`,
      applied by all three engines. What the gate removes is named in the run's warnings,
      and a run left with no cues warns and writes an empty `.srt` rather than raising. See
      finding 12 for why the obvious signals did not work.
- [ ] **CLI `--review` is a dead end.** `RunResult.to_dict` omits `cues_key`, so a CLI user
      who stops at review cannot learn the `base_key` needed to write `edits.json` without
      reading `cues.meta.json` by hand. The flag reads as general in `--help`.
- [ ] **`subtitler fetch` has no `--start`/`--end`**, unlike `run`.
- [ ] **The yt-dlp guard has no absolute size or duration ceiling.**
- [ ] A second URL run once died with `FetchError: ffmpeg exited with code 1` and surfaced
      no cause. Possibly transient, possibly a real gap in fetch error reporting.
- [ ] yt-dlp prints three EJS/signature errors that are not fatal but read like failures.
- [ ] `_review_media` in the browser GUI is never cleared, so it survives into later runs.
- [ ] The browser GUI's `_check` silently truncates at 500 rows; a long episode exceeds it.
- [ ] `_cmd_fetch` silently clobbers an existing output file of the same name.
- [x] **`CLAUDE.md` pointed at `docs/architecture.md`, which did not exist.** Written.

### 4.4 Benchmark improvements, all deliberately not built

- [ ] Digit and abbreviation normalization (`20` vs `dvadeset`, `npr.` vs `na primer`).
      Inflates every WER currently reported. Needs a Serbian number speller with case and
      gender agreement.
- [ ] A non-Whisper engine, so the consensus reference is not three variants of one model.
      This is the deepest limitation of the current WER numbers.
- [ ] A second critic pass over the revised references.
- [ ] More than one sample per matrix cell; run-to-run variance is unquantified.
- [ ] CI never syncs `--extra fetch`, so nothing exercises the yt-dlp seam. Its one test
      asserts `isinstance(..., bool)`, which passes either way.
- [ ] An opt-in CI job constructing a real `Window` (needs Xvfb on Linux, and the macOS
      question above settled first).

### 4.5 Answered, recorded so they are not re-asked

- **PRD criterion 4, does local beat cloud on Serbian:** yes, provisionally. 9.7% vs 11.1%
  pooled. Default unchanged.
- **PRD open question 4, does `--fix` help:** **it does both, consistently.** Worse on
  clean audio (0.7 to 5.2), better on noisy audio (15.4 to 7.9). Use it on rough
  recordings, leave it off on clean ones.
- **Are diacritics the error source:** no. `WER_folded` equals `WER` in all 30 cells. The
  errors are whole wrong words. This contradicts an explicit prediction in the original
  plan.
- **Does the denoiser choice matter:** barely, and `arnndn` actively caused a prompt echo
  on one clip. Off by default is correct.
- **`groq/whisper-large-v3` truncates:** 68 of its 118 errors are deletions. It drops
  content rather than mishearing it. Do not use it for the archive.

---

## 5. Still open from the PRD

1. Is 42 characters per line right for a 9:16 social crop? Vertical needs closer to 28.
   Should the splitter be aspect-ratio aware?
2. Should the friend's default style preset be `box` rather than `outline`? Decide from
   `subtitler burn --preview` stills, not from theory.

---

## 6. The unrequested scope, stated plainly

Five features arrived that nobody asked for, built by agents that exceeded their briefs:
the browser GUI, URL fetch with `--start`/`--end`, the review-and-edit step, the native
desktop window, and `install-app`. Roughly 5,800 lines.

The maintainer chose to **keep the GUI** and the PRD was amended to say so, with three
conditions recorded: no new dependency, a thin shell over the same pipeline rather than a
second code path, and tests. All three currently hold.

The URL fetch and review step were **independently reviewed** and their four real defects
fixed with measured evidence. They are sound rather than merely present. They are still
more surface than was asked for, and reverting is cleaner now than later if any of it feels
like ballast.

Two process notes worth carrying forward. Agent self-reports were unreliable in both
directions (features reported working that had real defects, and "failures" that were the
agent's own driving errors), which is why the independent review was worth its cost. And
parking an agent's in-flight work on a branch while it was still running caused churn; do
not do that.
