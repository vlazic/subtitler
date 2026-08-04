# Benchmark run

- created: `2026-08-04T15:23:29.591698+00:00`
- commit: `8fb31586772754a34f98d429563dac0e0272a58e` on `HEAD` (clean)
- clips: fixtures/gozba-sample.mp3, fixtures/uvod-u-pravo.m4a
- denoisers: none
- engines: faster-whisper, groq, groq-turbo (large-v3, device cuda)
- metrics recomputed from the kept transcripts: `2026-08-04T17:54:38.033441+00:00`

## What this run cannot answer

- **The reference for gozba-sample, uvod-u-pravo is not human-verified.** Every WER derived from it is marked `*` and is provisional: an unverified reference measures agreement between models, not correctness.
- **`gozba-sample` is a consensus pseudo-reference, not ground truth.** It was adjudicated from 3 engine transcripts (`faster-whisper`, `groq`, `groq-turbo`) by an LLM that cannot hear the audio. It works at the text level with Serbian language knowledge, which catches a reading that is not Serbian and is **blind to any error every engine made the same way**. So the WER column below ranks these engines against each other; it does not measure how much of the speech each one got right. 9 span(s) are flagged as uncertain and 1 more are disputed by the adversarial reviewer: `benchmarks/references/review-queue.md` lists them with timestamps, for the human pass that would make this reference real.
- **`uvod-u-pravo` is a consensus pseudo-reference, not ground truth.** It was adjudicated from 3 engine transcripts (`faster-whisper`, `groq`, `groq-turbo`) by an LLM that cannot hear the audio. It works at the text level with Serbian language knowledge, which catches a reading that is not Serbian and is **blind to any error every engine made the same way**. So the WER column below ranks these engines against each other; it does not measure how much of the speech each one got right. 26 span(s) are flagged as uncertain and 8 more are disputed by the adversarial reviewer: `benchmarks/references/review-queue.md` lists them with timestamps, for the human pass that would make this reference real.

## Leaderboard

`RTF` is decode time over audio duration and comes from the transcript, so it is the engine's own speed. `wall s` and `peak MB` are the whole cell in its own process, and the `cached` column is what it did **not** have to do: a cell that reused a cached transcript never loaded a model, and its wall clock and peak memory are not comparable with a cell that did.

| # | clip | denoise | engine | fix | WER % | WER folded % | CER % | sub/ins/del | RTF | wall s | peak MB | cached |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | gozba-sample | none | faster-whisper | no | 0.7* | 0.7* | 0.2 | 1/0/0 | 0.044 | 7.9 | 3386 | extract |
| 2 | gozba-sample | none | groq-turbo | no | 3.3* | 3.3* | 0.7 | 4/0/1 | 0.014 | 1.8 | 49 | extract |
| 3 | gozba-sample | none | faster-whisper | yes | 5.2* | 5.2* | 1.2 | 7/1/0 | 0.044 | 10.7 | 251 | extract,transcribe,cues |
| 4 | gozba-sample | none | groq-turbo | yes | 5.2* | 5.2* | 1.1 | 7/0/1 | 0.014 | 9.6 | 199 | extract,transcribe,cues |
| 5 | uvod-u-pravo | none | groq-turbo | yes | 7.9* | 7.9* | 3.7 | 16/1/5 | 0.006 | 11.3 | 200 | extract,transcribe,cues |
| 6 | uvod-u-pravo | none | faster-whisper | yes | 12.9* | 12.9* | 7.5 | 25/5/6 | 0.046 | 10.8 | 231 | extract,transcribe,cues |
| 7 | uvod-u-pravo | none | faster-whisper | no | 14.6* | 14.6* | 7.7 | 31/5/5 | 0.046 | 10.8 | 3386 | extract |
| 8 | uvod-u-pravo | none | groq-turbo | no | 15.4* | 15.4* | 5.4 | 34/4/5 | 0.006 | 1.2 | 52 | extract |
| 9 | uvod-u-pravo | none | groq | yes | 23.9* | 23.9* | 12.9 | 34/5/28 | 0.012 | 12.2 | 200 | extract,transcribe,cues |
| 10 | uvod-u-pravo | none | groq | no | 26.1* | 26.1* | 13.4 | 41/4/28 | 0.012 | 2.3 | 52 | extract |
| 11 | gozba-sample | none | groq | no | 29.4* | 29.4* | 26.0 | 4/1/40 | 0.016 | 1.9 | 49 | extract |
| 12 | gozba-sample | none | groq | yes | 30.7* | 30.7* | 26.3 | 5/1/41 | 0.016 | 8.0 | 200 | extract,transcribe,cues |

## Cue shape and hallucination signals

| cell | cues | mean CPS | p95 CPS | max CPS | over line % | over dur % | over CPS % | under min dur % | longest repeat | prompt echo | collapses | silence dropped | filler |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| gozba-sample__none__faster-whisper__large-v3__fix | 20 | 12.3 | 18.4 | 19.2 | 0.0 | 0.0 | 0.0 | 0.0 | 2 (`da se`) | 0 | 0 | 0 | hvalax1 |
| gozba-sample__none__faster-whisper__large-v3__nofix | 20 | 12.3 | 18.4 | 19.2 | 0.0 | 0.0 | 0.0 | 0.0 | 2 (`da se`) | 0 | 0 | 0 | hvalax1 |
| gozba-sample__none__groq-turbo__large-v3__fix | 19 | 12.3 | 17.2 | 17.2 | 0.0 | 0.0 | 0.0 | 0.0 | 2 (`da se`) | 0 | 0 | n/a | hvalax1 |
| gozba-sample__none__groq-turbo__large-v3__nofix | 19 | 12.3 | 17.2 | 17.2 | 0.0 | 0.0 | 0.0 | 0.0 | 2 (`da se`) | 0 | 0 | n/a | hvalax1 |
| gozba-sample__none__groq__large-v3__fix | 21 | 11.7 | 16.7 | 20.0 | 0.0 | 0.0 | 0.0 | 0.0 | 2 (`da se`) | 0 | 0 | n/a | hvalax2 |
| gozba-sample__none__groq__large-v3__nofix | 21 | 11.7 | 16.7 | 20.0 | 0.0 | 0.0 | 0.0 | 0.0 | 2 (`da se`) | 0 | 0 | n/a | hvalax2 |
| uvod-u-pravo__none__faster-whisper__large-v3__fix | 35 | 10.6 | 14.5 | 15.2 | 0.0 | 0.0 | 0.0 | 0.0 | 5 (`koji je državno službeno lice`) | 0 | 0 | 0 | none |
| uvod-u-pravo__none__faster-whisper__large-v3__nofix | 35 | 10.7 | 14.8 | 15.1 | 0.0 | 0.0 | 0.0 | 0.0 | 5 (`koji je državno službeno lice`) | 0 | 0 | 0 | none |
| uvod-u-pravo__none__groq-turbo__large-v3__fix | 37 | 10.5 | 15.0 | 15.7 | 0.0 | 0.0 | 0.0 | 0.0 | 5 (`koji je državno službeno lice`) | 0 | 0 | n/a | none |
| uvod-u-pravo__none__groq-turbo__large-v3__nofix | 37 | 10.7 | 15.2 | 16.2 | 0.0 | 0.0 | 0.0 | 0.0 | 5 (`koji je državno službno novice`) | 0 | 0 | n/a | none |
| uvod-u-pravo__none__groq__large-v3__fix | 35 | 10.3 | 15.6 | 15.9 | 0.0 | 0.0 | 0.0 | 0.0 | 5 (`koji je državno službeno lice`) | 0 | 0 | n/a | none |
| uvod-u-pravo__none__groq__large-v3__nofix | 35 | 10.3 | 15.6 | 15.9 | 0.0 | 0.0 | 0.0 | 0.0 | 5 (`koji je državno službeno lice`) | 0 | 0 | n/a | none |

## The `--fix` axis

`change %` is the word-level distance between the corrected cell and the identical uncorrected one. It measures **how much the model rewrote, not whether the rewrite was right**. Whether `--fix` improves WER or hurts it (PRD open question 4) needs a reference transcript to answer, and is answered in the leaderboard above only when one exists.

| cell | change % | cues changed | model | wall s |
|---|---|---|---|---|
| gozba-sample__none__faster-whisper__large-v3__fix | 4.6 | 8 | openai/gpt-4o | 10.7 |
| gozba-sample__none__groq-turbo__large-v3__fix | 2.0 | 3 | openai/gpt-4o | 9.6 |
| gozba-sample__none__groq__large-v3__fix | 4.4 | 7 | openai/gpt-4o | 8.0 |
| uvod-u-pravo__none__faster-whisper__large-v3__fix | 14.6 | 23 | openai/gpt-4o | 10.8 |
| uvod-u-pravo__none__groq-turbo__large-v3__fix | 15.4 | 25 | openai/gpt-4o | 11.3 |
| uvod-u-pravo__none__groq__large-v3__fix | 12.1 | 20 | openai/gpt-4o | 12.2 |

## How the text was normalized

Applied identically to hypothesis and reference, in this order: NFC, Serbian Cyrillic to Latin via a hand-written table, lowercase, punctuation to spaces (including the Serbian quotes), whitespace collapsed. `WER folded` repeats the score with `č ć` folded to `c`, `đ` to `dj`, `š` to `s` and `ž` to `z`; the gap between the two columns separates hearing the wrong word from writing `c` for `č`.

**Digits and abbreviations are deliberately not normalized in v1.** `20` scores as a substitution against `dvadeset`, and `npr.` against `na primer`. Both inflate every WER here, and both inflate it equally for every engine in the matrix, so the ranking survives while the absolute numbers are pessimistic.

## Environment

Full detail in `env.json` next to this file: `doctor --json`, the OS, CPU, RAM and GPU, and the version of every library that can change a transcript.

Cue limits in force: max 42 chars per line, 2 lines, 1.0-7.0s, 20.0 CPS.
