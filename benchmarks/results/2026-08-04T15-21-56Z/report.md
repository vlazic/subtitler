# Benchmark run

- created: `2026-08-04T15:23:29.591698+00:00`
- commit: `8fb31586772754a34f98d429563dac0e0272a58e` on `HEAD` (clean)
- clips: fixtures/gozba-sample.mp3, fixtures/uvod-u-pravo.m4a
- denoisers: none
- engines: faster-whisper, groq, groq-turbo (large-v3, device cuda)

## What this run cannot answer

- **No reference transcript for gozba-sample, uvod-u-pravo.** WER, CER and the error decomposition are therefore not reported for those clips: this run measures shape, speed and hallucination signals only. Phase 8 (LLM adjudication of reference transcripts) is what fills that gap; nothing here invents one.
- **The leaderboard below is ordered by realtime factor, not by quality.** Speed is not accuracy. Nothing in this run ranks transcription quality.

## Leaderboard (by speed: no reference exists)

`RTF` is decode time over audio duration and comes from the transcript, so it is the engine's own speed. `wall s` and `peak MB` are the whole cell in its own process, and the `cached` column is what it did **not** have to do: a cell that reused a cached transcript never loaded a model, and its wall clock and peak memory are not comparable with a cell that did.

| # | clip | denoise | engine | fix | WER % | WER folded % | CER % | sub/ins/del | RTF | wall s | peak MB | cached |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | uvod-u-pravo | none | groq-turbo | no | n/a | n/a | n/a | n/a | 0.006 | 1.2 | 52 | extract |
| 2 | uvod-u-pravo | none | groq-turbo | yes | n/a | n/a | n/a | n/a | 0.006 | 11.3 | 200 | extract,transcribe,cues |
| 3 | uvod-u-pravo | none | groq | no | n/a | n/a | n/a | n/a | 0.012 | 2.3 | 52 | extract |
| 4 | uvod-u-pravo | none | groq | yes | n/a | n/a | n/a | n/a | 0.012 | 12.2 | 200 | extract,transcribe,cues |
| 5 | gozba-sample | none | groq-turbo | no | n/a | n/a | n/a | n/a | 0.014 | 1.8 | 49 | extract |
| 6 | gozba-sample | none | groq-turbo | yes | n/a | n/a | n/a | n/a | 0.014 | 9.6 | 199 | extract,transcribe,cues |
| 7 | gozba-sample | none | groq | no | n/a | n/a | n/a | n/a | 0.016 | 1.9 | 49 | extract |
| 8 | gozba-sample | none | groq | yes | n/a | n/a | n/a | n/a | 0.016 | 8.0 | 200 | extract,transcribe,cues |
| 9 | gozba-sample | none | faster-whisper | no | n/a | n/a | n/a | n/a | 0.044 | 7.9 | 3386 | extract |
| 10 | gozba-sample | none | faster-whisper | yes | n/a | n/a | n/a | n/a | 0.044 | 10.7 | 251 | extract,transcribe,cues |
| 11 | uvod-u-pravo | none | faster-whisper | no | n/a | n/a | n/a | n/a | 0.046 | 10.8 | 3386 | extract |
| 12 | uvod-u-pravo | none | faster-whisper | yes | n/a | n/a | n/a | n/a | 0.046 | 10.8 | 231 | extract,transcribe,cues |

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
