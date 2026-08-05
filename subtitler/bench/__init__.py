"""Benchmark harness: matrix runner, Serbian normalization, metrics, report, adjudication.

No number this package reports ever comes from an LLM (non-negotiable 9). `agents` uses one
to *adjudicate a reference transcript* out of several engine transcripts, which is text; the
scoring itself is arithmetic over that text and lives in `metrics.py`.

Five modules, in dependency order:

| Module | What |
|---|---|
| `normalize` | Serbian text folding: Cyrillic to Latin, case, punctuation, diacritics |
| `metrics` | WER, WER_folded, CER, cue shape, hallucination heuristics |
| `report` | a payload dict in, `report.md` out. Pure, and jiwer-free |
| `agents` | the reference-adjudication manifest, its schemas and its merge. No model client |
| `review` | the human pass over a reference: the only thing that raises `human_verified` |
| `run` | the denoiser x engine x clip matrix, one process per cell |

`run` is imported lazily by the CLI: it pulls in the whole pipeline, and `subtitler --help`
should not pay for that.
"""

from __future__ import annotations

__all__: list[str] = ["agents", "metrics", "normalize", "report", "run"]
