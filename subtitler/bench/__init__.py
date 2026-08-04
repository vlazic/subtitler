"""Benchmark harness: matrix runner, Serbian normalization, metrics, report.

No number this package reports ever comes from an LLM (non-negotiable 9). Phase 8 may use an
agent to *adjudicate a reference transcript*; the scoring itself is arithmetic over strings
and lives in `metrics.py`.

Four modules, in dependency order:

| Module | What |
|---|---|
| `normalize` | Serbian text folding: Cyrillic to Latin, case, punctuation, diacritics |
| `metrics` | WER, WER_folded, CER, cue shape, hallucination heuristics |
| `report` | a payload dict in, `report.md` out. Pure, and jiwer-free |
| `run` | the denoiser x engine x clip matrix, one process per cell |

`run` is imported lazily by the CLI: it pulls in the whole pipeline, and `subtitler --help`
should not pay for that.
"""

from __future__ import annotations

__all__: list[str] = ["metrics", "normalize", "report", "run"]
