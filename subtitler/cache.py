"""The content-addressed stage cache.

Every stage writes `<stage>.meta.json` next to its artifact. The meta records the key, the
hash of the stage's input, and the exact parameters that produced it. A stage is skipped
when the key recomputed from *this* run matches the key on disk and every artifact it
claims to have written still exists.

The key is a chain, not a flat hash of the command line:

    fetch      <- the URL, and nothing else that is knowable offline
    trim       <- the source file's content id (or fetch's key)
    extract    <- the source file's content id (or trim's key)
    denoise    <- extract's key
    transcribe <- the audio stage's key (denoise if denoising, else extract)
    cues       <- transcribe's key
    fix        <- cues' key          (Phase 6; the seam is here, unused)
    burn       <- cues' key + the source content id

Chaining means a change anywhere invalidates exactly the stages downstream of it and
nothing else. Switching `--style-preset` re-burns without re-transcribing; switching
`--denoise` re-runs the denoiser without re-extracting audio from a 3 GB video.

`fetch` is the one stage that cannot be content-addressed, because there is nothing to hash
until it has already happened. Its key is the normalized URL plus which shape was asked for
(audio for `--srt-only`, video otherwise), and deliberately nothing about the remote file:
finding out whether the upload changed costs a network round trip, and paying for one on
every warm run would make a re-run neither free nor offline. The consequence, stated
plainly: if the uploader replaces the video behind a URL, this cache serves the old
download until `--force fetch`. That is the same tradeoff `content_id` makes for large
files, for the same reason.

`trim` sits between the source and the extraction rather than anywhere later, and that
placement is the feature. Cutting first means the audio the recognizer sees *starts* at the
fragment, so its cue timestamps come out relative to the fragment with no arithmetic
anywhere downstream, and the burn re-encodes the fragment rather than the full-length
source. Keying it on the source's content id plus the two timecodes is also what makes
changing `--start` re-cut without re-downloading: `fetch` is upstream of it and its own key
never mentions a timecode.

Splitting `denoise` out of `extract` (they were one ffmpeg invocation before) is what makes
that last case work, and it is also what makes the Phase 7 engine x denoiser matrix extract
each clip once instead of once per denoiser. The extra pass costs a second on a 16 kHz mono
WAV, which is nothing next to demuxing the source again.

**One slot per stage, not one per key.** `denoise.wav` is overwritten when the preset
changes, rather than kept alongside `denoise-<key>.wav`. So alternating between two
settings re-does the work each time instead of finding both cached. That is deliberate: a
work directory holding one WAV and one JSON per stage can be read by a human, while a
content-addressed heap of them accumulates gigabytes of stale audio that nothing ever
deletes. The case this cache exists for is running the same command twice, and switching a
knob once and going forward, not oscillating.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1

# The order stages run in. `--force <stage>` invalidates the named stage and everything
# after it, so this tuple is the definition of "after".
STAGE_ORDER: tuple[str, ...] = (
    "fetch",
    "trim",
    "extract",
    "denoise",
    "transcribe",
    "cues",
    "fix",
    "burn",
)

# 64 bits of key. A collision needs about 2**32 distinct stage inputs in one work
# directory; the short form is what makes a hand-inspected meta file readable.
KEY_LEN = 16

# Files at or under this size are hashed whole. 8 MiB of sha256 is a few milliseconds.
FULL_HASH_MAX_BYTES = 8 * 1024 * 1024
SAMPLE_BYTES = 1024 * 1024


class CacheError(ValueError):
    """The cache was asked for something it cannot do, e.g. `--force nonsense`."""


def content_id(path: Path) -> str:
    """A cheap, stable identifier for a file's content.

    Full sha256 of a 3 GB video reads 3 GB. On a laptop that is roughly ten seconds, which
    would eat the whole budget for the "a re-run finishes in under 2 seconds" criterion
    before any stage had even been consulted. So large files are sampled: the byte length,
    plus the first, middle and last megabyte. That is 3 MiB of reading no matter how big
    the file is. Small files are hashed in full, because at that size there is no reason
    not to be exact.

    The tradeoff, stated plainly: for a large file, an edit that changes only bytes outside
    the three sampled windows *and* leaves the total length unchanged is invisible here,
    and the cache would be served stale. For media that is close to impossible (any
    re-encode or re-export changes the length) and `--force` is the escape hatch.

    Sampling is chosen over the cheaper size+mtime+path because mtime is wrong in both
    directions: a plain `cp` gives byte-identical content a new mtime and would miss the
    cache for no reason, and a restored backup or a `touch -r` can carry an old mtime onto
    new content and would serve a stale one. Content is the thing the cache is keyed on, so
    content is what gets read.
    """
    stat = path.stat()
    digest = hashlib.sha256()
    digest.update(f"size={stat.st_size}\n".encode())
    with path.open("rb") as handle:
        if stat.st_size <= FULL_HASH_MAX_BYTES:
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                digest.update(chunk)
        else:
            for offset in (0, stat.st_size // 2, stat.st_size - SAMPLE_BYTES):
                handle.seek(offset)
                digest.update(handle.read(SAMPLE_BYTES))
    return digest.hexdigest()[:KEY_LEN]


def text_id(text: str) -> str:
    """The same shape of id as `content_id`, for an input that is not a file yet.

    A URL is the case this exists for: the `fetch` stage has to be keyed before anything
    has been downloaded, so it is keyed on what the user typed rather than on bytes.
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:KEY_LEN]


def stage_key(name: str, *, input_hash: str, params: Mapping[str, Any]) -> str:
    """sha256 over the stage name, its input's hash and its parameters, truncated.

    `sort_keys` is what makes this stable: two runs that build the same params dict in a
    different insertion order must produce the same key, or the cache never hits.
    """
    payload = json.dumps(
        {"stage": name, "input": input_hash, "params": params},
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
        default=str,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:KEY_LEN]


def invalidated_from(force: str | None) -> frozenset[str]:
    """Which stages `--force` invalidates.

    `--force` with no value means everything. `--force transcribe` means transcribe and
    every stage after it, because a new transcript makes the cues computed from the old one
    meaningless. Invalidating only the named stage would leave the pipeline internally
    inconsistent, which is a far worse failure than doing a little extra work.
    """
    if force is None:
        return frozenset()
    name = force.strip().lower()
    if name in {"all", "*"}:
        return frozenset(STAGE_ORDER)
    if name not in STAGE_ORDER:
        raise CacheError(f"unknown stage {force!r}; choose from: {', '.join(STAGE_ORDER)}, all")
    return frozenset(STAGE_ORDER[STAGE_ORDER.index(name) :])


@dataclass(frozen=True, slots=True)
class Entry:
    """One consulted stage. `hit` means the artifacts on disk are already correct."""

    name: str
    key: str
    hit: bool
    input_hash: str
    params: dict[str, Any]
    artifacts: tuple[Path, ...]
    reason: str = ""


@dataclass(slots=True)
class StageCache:
    """Consult with `begin()`, and on a miss run the stage and call `commit()`.

    `commit()` is deliberately separate from `begin()`: the meta file is written only after
    the stage has actually produced its artifacts, so a crash halfway through leaves a
    stage that misses next time rather than one that claims a result it never wrote.
    """

    work: Path
    forced: frozenset[str] = frozenset()
    enabled: bool = True
    hits: list[str] = field(default_factory=list)
    misses: list[str] = field(default_factory=list)

    def meta_path(self, name: str) -> Path:
        return self.work / f"{name}.meta.json"

    def begin(
        self,
        name: str,
        *,
        input_hash: str,
        params: Mapping[str, Any],
        artifacts: Sequence[Path],
    ) -> Entry:
        key = stage_key(name, input_hash=input_hash, params=params)
        payload = dict(params)
        made = tuple(artifacts)

        reason = self._miss_reason(name, key, made)
        hit = reason == ""
        (self.hits if hit else self.misses).append(name)
        return Entry(
            name=name,
            key=key,
            hit=hit,
            input_hash=input_hash,
            params=payload,
            artifacts=made,
            reason=reason,
        )

    def _miss_reason(self, name: str, key: str, artifacts: Sequence[Path]) -> str:
        """Empty string means a hit. Anything else is the human-readable reason it missed."""
        if not self.enabled:
            return "cache disabled"
        if name in self.forced:
            return "forced"
        meta = self.meta_path(name)
        if not meta.exists():
            return "no cached run"
        try:
            data = json.loads(meta.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return "unreadable meta"
        if data.get("schema_version") != SCHEMA_VERSION:
            return "meta schema changed"
        if data.get("key") != key:
            return "inputs or parameters changed"
        missing = [p.name for p in artifacts if not p.exists()]
        if missing:
            return f"missing artifact: {', '.join(missing)}"
        return ""

    def commit(self, entry: Entry) -> None:
        if not self.enabled:
            return
        self.work.mkdir(parents=True, exist_ok=True)
        self.meta_path(entry.name).write_text(
            json.dumps(
                {
                    "schema_version": SCHEMA_VERSION,
                    "stage": entry.name,
                    "key": entry.key,
                    "input": entry.input_hash,
                    "params": entry.params,
                    "artifacts": [str(p) for p in entry.artifacts],
                },
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
                default=str,
            )
            + "\n",
            encoding="utf-8",
        )
