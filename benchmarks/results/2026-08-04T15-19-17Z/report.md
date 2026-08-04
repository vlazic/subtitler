# Benchmark run

- created: `2026-08-04T15:21:39.866633+00:00`
- commit: `8fb31586772754a34f98d429563dac0e0272a58e` on `HEAD` (clean)
- clips: fixtures/gozba-sample.mp3, fixtures/uvod-u-pravo.m4a
- denoisers: none, afftdn, arnndn, anlmdn, speech
- engines: faster-whisper, groq, groq-turbo (large-v3, device cuda)

## What this run cannot answer

- **No reference transcript for gozba-sample, uvod-u-pravo.** WER, CER and the error decomposition are therefore not reported for those clips: this run measures shape, speed and hallucination signals only. Phase 8 (LLM adjudication of reference transcripts) is what fills that gap; nothing here invents one.
- **The leaderboard below is ordered by realtime factor, not by quality.** Speed is not accuracy. Nothing in this run ranks transcription quality.
- **The `--fix` axis was not run**, so PRD open question 4 (does the correction pass improve WER or hurt it) is untouched here.

## Leaderboard (by speed: no reference exists)

`RTF` is decode time over audio duration and comes from the transcript, so it is the engine's own speed. `wall s` and `peak MB` are the whole cell in its own process, and the `cached` column is what it did **not** have to do: a cell that reused a cached transcript never loaded a model, and its wall clock and peak memory are not comparable with a cell that did.

| # | clip | denoise | engine | fix | WER % | WER folded % | CER % | sub/ins/del | RTF | wall s | peak MB | cached |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | uvod-u-pravo | arnndn | groq-turbo | no | n/a | n/a | n/a | n/a | 0.006 | 1.2 | 52 | extract,denoise |
| 2 | uvod-u-pravo | afftdn | groq-turbo | no | n/a | n/a | n/a | n/a | 0.006 | 1.3 | 52 | extract,denoise |
| 3 | uvod-u-pravo | anlmdn | groq-turbo | no | n/a | n/a | n/a | n/a | 0.007 | 1.3 | 52 | extract,denoise |
| 4 | uvod-u-pravo | speech | groq-turbo | no | n/a | n/a | n/a | n/a | 0.007 | 1.3 | 52 | extract,denoise |
| 5 | gozba-sample | arnndn | groq-turbo | no | n/a | n/a | n/a | n/a | 0.009 | 1.1 | 49 | extract,denoise |
| 6 | gozba-sample | speech | groq-turbo | no | n/a | n/a | n/a | n/a | 0.009 | 1.2 | 50 | extract,denoise |
| 7 | uvod-u-pravo | none | groq-turbo | no | n/a | n/a | n/a | n/a | 0.009 | 1.7 | 53 | extract |
| 8 | gozba-sample | none | groq-turbo | no | n/a | n/a | n/a | n/a | 0.010 | 1.3 | 51 | extract |
| 9 | gozba-sample | anlmdn | groq-turbo | no | n/a | n/a | n/a | n/a | 0.011 | 1.3 | 50 | extract,denoise |
| 10 | gozba-sample | afftdn | groq-turbo | no | n/a | n/a | n/a | n/a | 0.013 | 1.6 | 50 | extract,denoise |
| 11 | uvod-u-pravo | speech | groq | no | n/a | n/a | n/a | n/a | 0.013 | 2.4 | 51 | extract,denoise |
| 12 | uvod-u-pravo | arnndn | groq | no | n/a | n/a | n/a | n/a | 0.014 | 2.6 | 51 | extract,denoise |
| 13 | gozba-sample | speech | groq | no | n/a | n/a | n/a | n/a | 0.014 | 1.7 | 49 | extract,denoise |
| 14 | gozba-sample | afftdn | groq | no | n/a | n/a | n/a | n/a | 0.014 | 1.8 | 49 | extract,denoise |
| 15 | uvod-u-pravo | none | groq | no | n/a | n/a | n/a | n/a | 0.015 | 2.7 | 51 | extract |
| 16 | uvod-u-pravo | afftdn | groq | no | n/a | n/a | n/a | n/a | 0.016 | 2.8 | 52 | extract,denoise |
| 17 | gozba-sample | anlmdn | groq | no | n/a | n/a | n/a | n/a | 0.017 | 2.0 | 50 | extract,denoise |
| 18 | uvod-u-pravo | anlmdn | groq | no | n/a | n/a | n/a | n/a | 0.017 | 2.9 | 52 | extract,denoise |
| 19 | gozba-sample | none | groq | no | n/a | n/a | n/a | n/a | 0.018 | 2.4 | 52 | extract |
| 20 | gozba-sample | arnndn | groq | no | n/a | n/a | n/a | n/a | 0.020 | 2.4 | 50 | extract,denoise |
| 21 | uvod-u-pravo | arnndn | faster-whisper | no | n/a | n/a | n/a | n/a | 0.038 | 11.1 | 3385 | extract |
| 22 | gozba-sample | arnndn | faster-whisper | no | n/a | n/a | n/a | n/a | 0.041 | 8.6 | 3386 | extract |
| 23 | gozba-sample | anlmdn | faster-whisper | no | n/a | n/a | n/a | n/a | 0.041 | 8.3 | 3386 | extract |
| 24 | gozba-sample | afftdn | faster-whisper | no | n/a | n/a | n/a | n/a | 0.042 | 7.8 | 3386 | extract |
| 25 | gozba-sample | none | faster-whisper | no | n/a | n/a | n/a | n/a | 0.042 | 7.9 | 3387 | - |
| 26 | gozba-sample | speech | faster-whisper | no | n/a | n/a | n/a | n/a | 0.044 | 9.4 | 3386 | extract |
| 27 | uvod-u-pravo | none | faster-whisper | no | n/a | n/a | n/a | n/a | 0.045 | 10.6 | 3386 | - |
| 28 | uvod-u-pravo | anlmdn | faster-whisper | no | n/a | n/a | n/a | n/a | 0.045 | 11.6 | 3386 | extract |
| 29 | uvod-u-pravo | speech | faster-whisper | no | n/a | n/a | n/a | n/a | 0.045 | 13.2 | 3385 | extract |
| 30 | uvod-u-pravo | afftdn | faster-whisper | no | n/a | n/a | n/a | n/a | 0.046 | 11.1 | 3386 | extract |

## Cue shape and hallucination signals

**1 cell(s) echoed the Serbian steering prompt back as transcript text**: uvod-u-pravo__arnndn__faster-whisper__large-v3__nofix. That is decoder output standing where speech should be, so the affected transcript is missing whatever was said there. Worth reading before trusting any other number in that row.

| cell | cues | mean CPS | p95 CPS | max CPS | over line % | over dur % | over CPS % | under min dur % | longest repeat | prompt echo | collapses | silence dropped | filler |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| gozba-sample__afftdn__faster-whisper__large-v3__nofix | 21 | 11.9 | 18.3 | 18.5 | 0.0 | 0.0 | 0.0 | 0.0 | 2 (`da se`) | 0 | 0 | 0 | hvalax1 |
| gozba-sample__afftdn__groq-turbo__large-v3__nofix | 20 | 12.1 | 16.6 | 17.4 | 0.0 | 0.0 | 0.0 | 0.0 | 2 (`da se`) | 0 | 0 | n/a | hvalax1 |
| gozba-sample__afftdn__groq__large-v3__nofix | 17 | 13.1 | 20.0 | 20.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0 | 0 | 0 | n/a | hvalax2 |
| gozba-sample__anlmdn__faster-whisper__large-v3__nofix | 20 | 12.3 | 18.4 | 19.2 | 0.0 | 0.0 | 0.0 | 0.0 | 2 (`da se`) | 0 | 0 | 0 | hvalax1 |
| gozba-sample__anlmdn__groq-turbo__large-v3__nofix | 19 | 12.3 | 17.2 | 17.2 | 0.0 | 0.0 | 0.0 | 0.0 | 2 (`da se`) | 0 | 0 | n/a | hvalax1 |
| gozba-sample__anlmdn__groq__large-v3__nofix | 21 | 11.7 | 16.7 | 20.0 | 0.0 | 0.0 | 0.0 | 0.0 | 2 (`da se`) | 0 | 0 | n/a | hvalax2 |
| gozba-sample__arnndn__faster-whisper__large-v3__nofix | 20 | 12.2 | 18.1 | 18.5 | 0.0 | 0.0 | 0.0 | 0.0 | 2 (`da se`) | 0 | 0 | 0 | hvalax1 |
| gozba-sample__arnndn__groq-turbo__large-v3__nofix | 20 | 12.1 | 16.6 | 17.6 | 0.0 | 0.0 | 0.0 | 0.0 | 2 (`da se`) | 0 | 0 | n/a | hvalax1 |
| gozba-sample__arnndn__groq__large-v3__nofix | 14 | 12.1 | 20.0 | 20.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0 | 0 | 0 | n/a | hvalax2 |
| gozba-sample__none__faster-whisper__large-v3__nofix | 20 | 12.3 | 18.4 | 19.2 | 0.0 | 0.0 | 0.0 | 0.0 | 2 (`da se`) | 0 | 0 | 0 | hvalax1 |
| gozba-sample__none__groq-turbo__large-v3__nofix | 19 | 12.3 | 17.2 | 17.2 | 0.0 | 0.0 | 0.0 | 0.0 | 2 (`da se`) | 0 | 0 | n/a | hvalax1 |
| gozba-sample__none__groq__large-v3__nofix | 21 | 11.7 | 16.7 | 20.0 | 0.0 | 0.0 | 0.0 | 0.0 | 2 (`da se`) | 0 | 0 | n/a | hvalax2 |
| gozba-sample__speech__faster-whisper__large-v3__nofix | 21 | 11.9 | 18.4 | 18.9 | 0.0 | 0.0 | 0.0 | 0.0 | 2 (`da se`) | 0 | 0 | 0 | hvalax1 |
| gozba-sample__speech__groq-turbo__large-v3__nofix | 19 | 12.1 | 17.4 | 17.4 | 0.0 | 0.0 | 0.0 | 0.0 | 2 (`da se`) | 0 | 0 | n/a | hvalax1 |
| gozba-sample__speech__groq__large-v3__nofix | 19 | 13.3 | 20.0 | 20.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0 | 0 | 0 | n/a | hvalax2 |
| uvod-u-pravo__afftdn__faster-whisper__large-v3__nofix | 37 | 10.9 | 15.6 | 16.4 | 0.0 | 0.0 | 0.0 | 0.0 | 5 (`koji je državno službeno lice`) | 0 | 0 | 0 | none |
| uvod-u-pravo__afftdn__groq-turbo__large-v3__nofix | 35 | 10.6 | 13.2 | 16.2 | 0.0 | 0.0 | 0.0 | 0.0 | 6 (`koji je državno službeno u lice`) | 0 | 0 | n/a | none |
| uvod-u-pravo__afftdn__groq__large-v3__nofix | 32 | 10.2 | 14.7 | 16.9 | 0.0 | 0.0 | 0.0 | 0.0 | 5 (`koji je državno službeno lice`) | 0 | 0 | n/a | none |
| uvod-u-pravo__anlmdn__faster-whisper__large-v3__nofix | 36 | 10.7 | 15.1 | 15.6 | 0.0 | 0.0 | 0.0 | 0.0 | 5 (`koji je državno službeno lice`) | 0 | 0 | 0 | none |
| uvod-u-pravo__anlmdn__groq-turbo__large-v3__nofix | 37 | 10.7 | 15.2 | 16.2 | 0.0 | 0.0 | 0.0 | 0.0 | 5 (`koji je državno službno novice`) | 0 | 0 | n/a | none |
| uvod-u-pravo__anlmdn__groq__large-v3__nofix | 35 | 10.3 | 15.6 | 15.9 | 0.0 | 0.0 | 0.0 | 0.0 | 5 (`koji je državno službeno lice`) | 0 | 0 | n/a | none |
| uvod-u-pravo__arnndn__faster-whisper__large-v3__nofix | 29 | 20.3 | 16.2 | 290.0 | 0.0 | 0.0 | 3.4 | 3.4 | 5 (`koji je državu službe novice`) | **9** (`koristi ispravna imena za ljude knjige filozofske škole itd`) | 0 | 0 | none |
| uvod-u-pravo__arnndn__groq-turbo__large-v3__nofix | 36 | 10.5 | 15.2 | 16.4 | 0.0 | 0.0 | 0.0 | 2.8 | 5 (`koji je državno spužbeno lice`) | 0 | 0 | n/a | none |
| uvod-u-pravo__arnndn__groq__large-v3__nofix | 38 | 10.1 | 15.0 | 17.6 | 0.0 | 0.0 | 0.0 | 0.0 | 5 (`koji je državu službe novice`) | 0 | 0 | n/a | none |
| uvod-u-pravo__none__faster-whisper__large-v3__nofix | 35 | 10.7 | 14.8 | 15.1 | 0.0 | 0.0 | 0.0 | 0.0 | 5 (`koji je državno službeno lice`) | 0 | 0 | 0 | none |
| uvod-u-pravo__none__groq-turbo__large-v3__nofix | 37 | 10.7 | 15.2 | 16.2 | 0.0 | 0.0 | 0.0 | 0.0 | 5 (`koji je državno službno novice`) | 0 | 0 | n/a | none |
| uvod-u-pravo__none__groq__large-v3__nofix | 34 | 10.4 | 15.9 | 17.1 | 0.0 | 0.0 | 0.0 | 0.0 | 5 (`koji je državno službeno lice`) | 0 | 0 | n/a | none |
| uvod-u-pravo__speech__faster-whisper__large-v3__nofix | 36 | 10.8 | 15.5 | 16.1 | 0.0 | 0.0 | 0.0 | 0.0 | 5 (`koji je državno službeno lice`) | 0 | 0 | 0 | none |
| uvod-u-pravo__speech__groq-turbo__large-v3__nofix | 35 | 10.5 | 13.0 | 16.2 | 0.0 | 0.0 | 0.0 | 0.0 | 5 (`koji je državno službno nulice`) | 0 | 0 | n/a | none |
| uvod-u-pravo__speech__groq__large-v3__nofix | 35 | 10.3 | 16.3 | 17.1 | 0.0 | 0.0 | 0.0 | 0.0 | 5 (`koji je državno službeno lice`) | 0 | 0 | n/a | none |

## How the text was normalized

Applied identically to hypothesis and reference, in this order: NFC, Serbian Cyrillic to Latin via a hand-written table, lowercase, punctuation to spaces (including the Serbian quotes), whitespace collapsed. `WER folded` repeats the score with `č ć` folded to `c`, `đ` to `dj`, `š` to `s` and `ž` to `z`; the gap between the two columns separates hearing the wrong word from writing `c` for `č`.

**Digits and abbreviations are deliberately not normalized in v1.** `20` scores as a substitution against `dvadeset`, and `npr.` against `na primer`. Both inflate every WER here, and both inflate it equally for every engine in the matrix, so the ranking survives while the absolute numbers are pessimistic.

## Environment

Full detail in `env.json` next to this file: `doctor --json`, the OS, CPU, RAM and GPU, and the version of every library that can change a transcript.

Cue limits in force: max 42 chars per line, 2 lines, 1.0-7.0s, 20.0 CPS.
