# Benchmark run

- created: `2026-08-04T15:21:39.866633+00:00`
- commit: `8fb31586772754a34f98d429563dac0e0272a58e` on `HEAD` (clean)
- clips: fixtures/gozba-sample.mp3, fixtures/uvod-u-pravo.m4a
- denoisers: none, afftdn, arnndn, anlmdn, speech
- engines: faster-whisper, groq, groq-turbo (large-v3, device cuda)
- metrics recomputed from the kept transcripts: `2026-08-04T17:54:38.284060+00:00`

## What this run cannot answer

- **The reference for gozba-sample, uvod-u-pravo is not human-verified.** Every WER derived from it is marked `*` and is provisional: an unverified reference measures agreement between models, not correctness.
- **`gozba-sample` is a consensus pseudo-reference, not ground truth.** It was adjudicated from 3 engine transcripts (`faster-whisper`, `groq`, `groq-turbo`) by an LLM that cannot hear the audio. It works at the text level with Serbian language knowledge, which catches a reading that is not Serbian and is **blind to any error every engine made the same way**. So the WER column below ranks these engines against each other; it does not measure how much of the speech each one got right. 9 span(s) are flagged as uncertain and 1 more are disputed by the adversarial reviewer: `benchmarks/references/review-queue.md` lists them with timestamps, for the human pass that would make this reference real.
- **`uvod-u-pravo` is a consensus pseudo-reference, not ground truth.** It was adjudicated from 3 engine transcripts (`faster-whisper`, `groq`, `groq-turbo`) by an LLM that cannot hear the audio. It works at the text level with Serbian language knowledge, which catches a reading that is not Serbian and is **blind to any error every engine made the same way**. So the WER column below ranks these engines against each other; it does not measure how much of the speech each one got right. 26 span(s) are flagged as uncertain and 8 more are disputed by the adversarial reviewer: `benchmarks/references/review-queue.md` lists them with timestamps, for the human pass that would make this reference real.
- **The `--fix` axis was not run**, so PRD open question 4 (does the correction pass improve WER or hurt it) is untouched here.

## Leaderboard

`RTF` is decode time over audio duration and comes from the transcript, so it is the engine's own speed. `wall s` and `peak MB` are the whole cell in its own process, and the `cached` column is what it did **not** have to do: a cell that reused a cached transcript never loaded a model, and its wall clock and peak memory are not comparable with a cell that did.

| # | clip | denoise | engine | fix | WER % | WER folded % | CER % | sub/ins/del | RTF | wall s | peak MB | cached |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | gozba-sample | none | faster-whisper | no | 0.7* | 0.7* | 0.2 | 1/0/0 | 0.042 | 7.9 | 3387 | - |
| 2 | gozba-sample | anlmdn | faster-whisper | no | 0.7* | 0.7* | 0.2 | 1/0/0 | 0.041 | 8.3 | 3386 | extract |
| 3 | gozba-sample | arnndn | faster-whisper | no | 1.3* | 1.3* | 0.3 | 2/0/0 | 0.041 | 8.6 | 3386 | extract |
| 4 | gozba-sample | none | groq-turbo | no | 3.3* | 3.3* | 0.7 | 4/0/1 | 0.010 | 1.3 | 51 | extract |
| 5 | gozba-sample | afftdn | groq-turbo | no | 3.3* | 3.3* | 0.6 | 4/0/1 | 0.013 | 1.6 | 50 | extract,denoise |
| 6 | gozba-sample | anlmdn | groq-turbo | no | 3.3* | 3.3* | 0.7 | 4/0/1 | 0.011 | 1.3 | 50 | extract,denoise |
| 7 | gozba-sample | arnndn | groq-turbo | no | 3.9* | 3.9* | 0.8 | 5/0/1 | 0.009 | 1.1 | 49 | extract,denoise |
| 8 | gozba-sample | speech | groq-turbo | no | 3.9* | 3.9* | 0.9 | 5/0/1 | 0.009 | 1.2 | 50 | extract,denoise |
| 9 | gozba-sample | afftdn | faster-whisper | no | 5.9* | 5.9* | 2.5 | 5/0/4 | 0.042 | 7.8 | 3386 | extract |
| 10 | gozba-sample | speech | faster-whisper | no | 5.9* | 5.9* | 2.3 | 5/0/4 | 0.044 | 9.4 | 3386 | extract |
| 11 | uvod-u-pravo | none | faster-whisper | no | 14.6* | 14.6* | 7.7 | 31/5/5 | 0.045 | 10.6 | 3386 | - |
| 12 | uvod-u-pravo | anlmdn | faster-whisper | no | 15.0* | 15.0* | 7.8 | 31/6/5 | 0.045 | 11.6 | 3386 | extract |
| 13 | uvod-u-pravo | anlmdn | groq-turbo | no | 15.0* | 15.0* | 5.4 | 33/4/5 | 0.007 | 1.3 | 52 | extract,denoise |
| 14 | uvod-u-pravo | none | groq-turbo | no | 15.4* | 15.4* | 5.4 | 34/4/5 | 0.009 | 1.7 | 53 | extract |
| 15 | uvod-u-pravo | speech | faster-whisper | no | 16.4* | 16.4* | 7.3 | 29/10/7 | 0.045 | 13.2 | 3385 | extract |
| 16 | uvod-u-pravo | afftdn | groq-turbo | no | 18.2* | 18.2* | 8.7 | 33/7/11 | 0.006 | 1.3 | 52 | extract,denoise |
| 17 | uvod-u-pravo | afftdn | faster-whisper | no | 19.3* | 19.3* | 8.1 | 37/11/6 | 0.046 | 11.1 | 3386 | extract |
| 18 | uvod-u-pravo | speech | groq-turbo | no | 20.0* | 20.0* | 8.5 | 39/7/10 | 0.007 | 1.3 | 52 | extract,denoise |
| 19 | uvod-u-pravo | speech | groq | no | 20.7* | 20.7* | 8.9 | 41/7/10 | 0.013 | 2.4 | 51 | extract,denoise |
| 20 | uvod-u-pravo | afftdn | groq | no | 23.2* | 23.2* | 15.0 | 32/5/28 | 0.016 | 2.8 | 52 | extract,denoise |
| 21 | uvod-u-pravo | none | groq | no | 23.9* | 23.9* | 11.4 | 43/5/19 | 0.015 | 2.7 | 51 | extract |
| 22 | uvod-u-pravo | anlmdn | groq | no | 26.1* | 26.1* | 13.4 | 41/4/28 | 0.017 | 2.9 | 52 | extract,denoise |
| 23 | gozba-sample | none | groq | no | 29.4* | 29.4* | 26.0 | 4/1/40 | 0.018 | 2.4 | 52 | extract |
| 24 | gozba-sample | anlmdn | groq | no | 29.4* | 29.4* | 26.0 | 4/1/40 | 0.017 | 2.0 | 50 | extract,denoise |
| 25 | gozba-sample | afftdn | groq | no | 30.1* | 30.1* | 24.0 | 3/1/42 | 0.014 | 1.8 | 49 | extract,denoise |
| 26 | gozba-sample | speech | groq | no | 32.0* | 32.0* | 24.4 | 5/1/43 | 0.014 | 1.7 | 49 | extract,denoise |
| 27 | uvod-u-pravo | arnndn | groq-turbo | no | 33.6* | 33.6* | 15.0 | 66/12/16 | 0.006 | 1.2 | 52 | extract,denoise |
| 28 | uvod-u-pravo | arnndn | groq | no | 35.0* | 35.0* | 16.7 | 71/9/18 | 0.014 | 2.6 | 51 | extract,denoise |
| 29 | uvod-u-pravo | arnndn | faster-whisper | no | 42.5* | 42.5* | 30.8 | 51/10/58 | 0.038 | 11.1 | 3385 | extract |
| 30 | gozba-sample | arnndn | groq | no | 54.9* | 54.9* | 47.7 | 4/1/79 | 0.020 | 2.4 | 50 | extract,denoise |

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
| gozba-sample__speech__groq__large-v3__nofix | 19 | 13.3 | 20.0 | 20.0 | 0.0 | 0.0 | 5.3 | 0.0 | 0 | 0 | 0 | n/a | hvalax2 |
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
