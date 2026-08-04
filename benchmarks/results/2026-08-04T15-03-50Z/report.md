# Benchmark run

- created: `2026-08-04T15:04:58.810407+00:00`
- commit: `77a00173c4c1dbf5e6679514ac63278ddaaf9c05` on `main` (clean)
- clips: fixtures/gozba-sample.mp3, fixtures/uvod-u-pravo.m4a
- denoisers: none
- engines: faster-whisper, groq-turbo (large-v3, device cuda)
- metrics recomputed from the kept transcripts: `2026-08-04T15:06:15.500704+00:00`

## What this run cannot answer

- **No reference transcript for gozba-sample, uvod-u-pravo.** WER, CER and the error decomposition are therefore not reported for those clips: this run measures shape, speed and hallucination signals only. Phase 8 (LLM adjudication of reference transcripts) is what fills that gap; nothing here invents one.
- **The leaderboard below is ordered by realtime factor, not by quality.** Speed is not accuracy. Nothing in this run ranks transcription quality.

## Leaderboard (by speed: no reference exists)

`RTF` is decode time over audio duration and comes from the transcript, so it is the engine's own speed. `wall s` and `peak MB` are the whole cell in its own process, and the `cached` column is what it did **not** have to do: a cell that reused a cached transcript never loaded a model, and its wall clock and peak memory are not comparable with a cell that did.

| # | clip | denoise | engine | fix | WER % | WER folded % | CER % | sub/ins/del | RTF | wall s | peak MB | cached |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | uvod-u-pravo | none | groq-turbo | no | n/a | n/a | n/a | n/a | 0.007 | 1.3 | 51 | extract |
| 2 | uvod-u-pravo | none | groq-turbo | yes | n/a | n/a | n/a | n/a | 0.007 | 13.1 | 199 | extract,transcribe,cues |
| 3 | gozba-sample | none | groq-turbo | no | n/a | n/a | n/a | n/a | 0.010 | 1.3 | 50 | extract |
| 4 | gozba-sample | none | groq-turbo | yes | n/a | n/a | n/a | n/a | 0.010 | 9.8 | 199 | extract,transcribe,cues |
| 5 | gozba-sample | none | faster-whisper | no | n/a | n/a | n/a | n/a | 0.041 | 7.5 | 3386 | extract |
| 6 | gozba-sample | none | faster-whisper | yes | n/a | n/a | n/a | n/a | 0.041 | 11.1 | 249 | extract,transcribe,cues |
| 7 | uvod-u-pravo | none | faster-whisper | no | n/a | n/a | n/a | n/a | 0.044 | 10.4 | 3386 | extract |
| 8 | uvod-u-pravo | none | faster-whisper | yes | n/a | n/a | n/a | n/a | 0.044 | 10.8 | 232 | extract,transcribe,cues |

## Cue shape and hallucination signals

| cell | cues | mean CPS | p95 CPS | max CPS | over line % | over dur % | over CPS % | under min dur % | longest repeat | prompt echo | collapses | silence dropped | filler |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| gozba-sample__none__faster-whisper__large-v3__fix | 20 | 12.3 | 18.4 | 19.2 | 0.0 | 0.0 | 0.0 | 0.0 | 2 (`da se`) | 0 | 0 | 0 | hvalax1 |
| gozba-sample__none__faster-whisper__large-v3__nofix | 20 | 12.3 | 18.4 | 19.2 | 0.0 | 0.0 | 0.0 | 0.0 | 2 (`da se`) | 0 | 0 | 0 | hvalax1 |
| gozba-sample__none__groq-turbo__large-v3__fix | 19 | 12.3 | 17.2 | 17.2 | 0.0 | 0.0 | 0.0 | 0.0 | 2 (`da se`) | 0 | 0 | n/a | hvalax1 |
| gozba-sample__none__groq-turbo__large-v3__nofix | 19 | 12.3 | 17.2 | 17.2 | 0.0 | 0.0 | 0.0 | 0.0 | 2 (`da se`) | 0 | 0 | n/a | hvalax1 |
| uvod-u-pravo__none__faster-whisper__large-v3__fix | 35 | 10.7 | 14.5 | 15.2 | 0.0 | 0.0 | 0.0 | 0.0 | 5 (`koji je državno službeno lice`) | 0 | 0 | 0 | none |
| uvod-u-pravo__none__faster-whisper__large-v3__nofix | 35 | 10.7 | 14.8 | 15.1 | 0.0 | 0.0 | 0.0 | 0.0 | 5 (`koji je državno službeno lice`) | 0 | 0 | 0 | none |
| uvod-u-pravo__none__groq-turbo__large-v3__fix | 37 | 10.7 | 15.7 | 17.7 | 0.0 | 0.0 | 0.0 | 0.0 | 5 (`koji je državno službeno lice`) | 0 | 0 | n/a | none |
| uvod-u-pravo__none__groq-turbo__large-v3__nofix | 37 | 10.7 | 15.2 | 16.2 | 0.0 | 0.0 | 0.0 | 0.0 | 5 (`koji je državno službno novice`) | 0 | 0 | n/a | none |

## The `--fix` axis

`change %` is the word-level distance between the corrected cell and the identical uncorrected one. It measures **how much the model rewrote, not whether the rewrite was right**. Whether `--fix` improves WER or hurts it (PRD open question 4) needs a reference transcript to answer, and is answered in the leaderboard above only when one exists.

| cell | change % | cues changed | model | wall s |
|---|---|---|---|---|
| gozba-sample__none__faster-whisper__large-v3__fix | 3.9 | 6 | openai/gpt-4o | 11.1 |
| gozba-sample__none__groq-turbo__large-v3__fix | 1.3 | 2 | openai/gpt-4o | 9.8 |
| uvod-u-pravo__none__faster-whisper__large-v3__fix | 15.4 | 21 | openai/gpt-4o | 10.8 |
| uvod-u-pravo__none__groq-turbo__large-v3__fix | 15.1 | 24 | openai/gpt-4o | 13.1 |

## How the text was normalized

Applied identically to hypothesis and reference, in this order: NFC, Serbian Cyrillic to Latin via a hand-written table, lowercase, punctuation to spaces (including the Serbian quotes), whitespace collapsed. `WER folded` repeats the score with `č ć` folded to `c`, `đ` to `dj`, `š` to `s` and `ž` to `z`; the gap between the two columns separates hearing the wrong word from writing `c` for `č`.

**Digits and abbreviations are deliberately not normalized in v1.** `20` scores as a substitution against `dvadeset`, and `npr.` against `na primer`. Both inflate every WER here, and both inflate it equally for every engine in the matrix, so the ranking survives while the absolute numbers are pessimistic.

## Environment

Full detail in `env.json` next to this file: `doctor --json`, the OS, CPU, RAM and GPU, and the version of every library that can change a transcript.

Cue limits in force: max 42 chars per line, 2 lines, 1.0-7.0s, 20.0 CPS.
