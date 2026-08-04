# Benchmark run

- created: `2026-08-04T15:00:31.450111+00:00`
- commit: `3dc527d689eabbc513e16dd5ffc5498b215e0626` on `main` (clean)
- clips: /home/vlazic/Projects/github.com/vlazic/subtitler/fixtures/gozba-sample.mp3, /home/vlazic/Projects/github.com/vlazic/subtitler/fixtures/uvod-u-pravo.m4a
- denoisers: none, afftdn, arnndn, anlmdn, speech
- engines: faster-whisper (large-v3, device cuda)
- metrics recomputed from the kept transcripts: `2026-08-04T15:06:15.637388+00:00`

## What this run cannot answer

- **No reference transcript for gozba-sample, uvod-u-pravo.** WER, CER and the error decomposition are therefore not reported for those clips: this run measures shape, speed and hallucination signals only. Phase 8 (LLM adjudication of reference transcripts) is what fills that gap; nothing here invents one.
- **The leaderboard below is ordered by realtime factor, not by quality.** Speed is not accuracy. Nothing in this run ranks transcription quality.
- **No cloud engine was in the matrix**, so this run says nothing about PRD acceptance criterion 4 (local versus `groq/whisper-large-v3-turbo`).
- **The `--fix` axis was not run**, so PRD open question 4 (does the correction pass improve WER or hurt it) is untouched here.

## Leaderboard (by speed: no reference exists)

`RTF` is decode time over audio duration and comes from the transcript, so it is the engine's own speed. `wall s` and `peak MB` are the whole cell in its own process, and the `cached` column is what it did **not** have to do: a cell that reused a cached transcript never loaded a model, and its wall clock and peak memory are not comparable with a cell that did.

| # | clip | denoise | engine | fix | WER % | WER folded % | CER % | sub/ins/del | RTF | wall s | peak MB | cached |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | uvod-u-pravo | arnndn | faster-whisper | no | n/a | n/a | n/a | n/a | 0.038 | 10.9 | 3386 | extract |
| 2 | gozba-sample | speech | faster-whisper | no | n/a | n/a | n/a | n/a | 0.041 | 9.2 | 3385 | extract |
| 3 | gozba-sample | arnndn | faster-whisper | no | n/a | n/a | n/a | n/a | 0.042 | 8.6 | 3386 | extract |
| 4 | gozba-sample | none | faster-whisper | no | n/a | n/a | n/a | n/a | 0.042 | 7.8 | 3385 | - |
| 5 | gozba-sample | anlmdn | faster-whisper | no | n/a | n/a | n/a | n/a | 0.042 | 8.4 | 3385 | extract |
| 6 | gozba-sample | afftdn | faster-whisper | no | n/a | n/a | n/a | n/a | 0.042 | 7.8 | 3385 | extract |
| 7 | uvod-u-pravo | none | faster-whisper | no | n/a | n/a | n/a | n/a | 0.045 | 10.6 | 3385 | - |
| 8 | uvod-u-pravo | afftdn | faster-whisper | no | n/a | n/a | n/a | n/a | 0.047 | 11.1 | 3385 | extract |
| 9 | uvod-u-pravo | speech | faster-whisper | no | n/a | n/a | n/a | n/a | 0.047 | 13.4 | 3385 | extract |
| 10 | uvod-u-pravo | anlmdn | faster-whisper | no | n/a | n/a | n/a | n/a | 0.047 | 11.9 | 3386 | extract |

## Cue shape and hallucination signals

**1 cell(s) echoed the Serbian steering prompt back as transcript text**: uvod-u-pravo__arnndn__faster-whisper__large-v3__nofix. That is decoder output standing where speech should be, so the affected transcript is missing whatever was said there. Worth reading before trusting any other number in that row.

| cell | cues | mean CPS | p95 CPS | max CPS | over line % | over dur % | over CPS % | under min dur % | longest repeat | prompt echo | collapses | silence dropped | filler |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| gozba-sample__afftdn__faster-whisper__large-v3__nofix | 21 | 11.9 | 18.3 | 18.5 | 0.0 | 0.0 | 0.0 | 0.0 | 2 (`da se`) | 0 | 0 | 0 | hvalax1 |
| gozba-sample__anlmdn__faster-whisper__large-v3__nofix | 20 | 12.3 | 18.4 | 19.2 | 0.0 | 0.0 | 0.0 | 0.0 | 2 (`da se`) | 0 | 0 | 0 | hvalax1 |
| gozba-sample__arnndn__faster-whisper__large-v3__nofix | 20 | 12.2 | 18.1 | 18.5 | 0.0 | 0.0 | 0.0 | 0.0 | 2 (`da se`) | 0 | 0 | 0 | hvalax1 |
| gozba-sample__none__faster-whisper__large-v3__nofix | 20 | 12.3 | 18.4 | 19.2 | 0.0 | 0.0 | 0.0 | 0.0 | 2 (`da se`) | 0 | 0 | 0 | hvalax1 |
| gozba-sample__speech__faster-whisper__large-v3__nofix | 21 | 11.9 | 18.4 | 18.9 | 0.0 | 0.0 | 0.0 | 0.0 | 2 (`da se`) | 0 | 0 | 0 | hvalax1 |
| uvod-u-pravo__afftdn__faster-whisper__large-v3__nofix | 37 | 10.9 | 15.6 | 16.4 | 0.0 | 0.0 | 0.0 | 0.0 | 5 (`koji je državno službeno lice`) | 0 | 0 | 0 | none |
| uvod-u-pravo__anlmdn__faster-whisper__large-v3__nofix | 36 | 10.7 | 15.1 | 15.6 | 0.0 | 0.0 | 0.0 | 0.0 | 5 (`koji je državno službeno lice`) | 0 | 0 | 0 | none |
| uvod-u-pravo__arnndn__faster-whisper__large-v3__nofix | 29 | 20.3 | 16.2 | 290.0 | 0.0 | 0.0 | 3.4 | 3.4 | 5 (`koji je državu službe novice`) | **9** (`koristi ispravna imena za ljude knjige filozofske škole itd`) | 0 | 0 | none |
| uvod-u-pravo__none__faster-whisper__large-v3__nofix | 35 | 10.7 | 14.8 | 15.1 | 0.0 | 0.0 | 0.0 | 0.0 | 5 (`koji je državno službeno lice`) | 0 | 0 | 0 | none |
| uvod-u-pravo__speech__faster-whisper__large-v3__nofix | 36 | 10.8 | 15.5 | 16.1 | 0.0 | 0.0 | 0.0 | 0.0 | 5 (`koji je državno službeno lice`) | 0 | 0 | 0 | none |

## How the text was normalized

Applied identically to hypothesis and reference, in this order: NFC, Serbian Cyrillic to Latin via a hand-written table, lowercase, punctuation to spaces (including the Serbian quotes), whitespace collapsed. `WER folded` repeats the score with `č ć` folded to `c`, `đ` to `dj`, `š` to `s` and `ž` to `z`; the gap between the two columns separates hearing the wrong word from writing `c` for `č`.

**Digits and abbreviations are deliberately not normalized in v1.** `20` scores as a substitution against `dvadeset`, and `npr.` against `na primer`. Both inflate every WER here, and both inflate it equally for every engine in the matrix, so the ranking survives while the absolute numbers are pessimistic.

## Environment

Full detail in `env.json` next to this file: `doctor --json`, the OS, CPU, RAM and GPU, and the version of every library that can change a transcript.

Cue limits in force: max 42 chars per line, 2 lines, 1.0-7.0s, 20.0 CPS.
