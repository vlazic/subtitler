"""The content-addressed stage cache.

These exercise the key algebra directly rather than through the pipeline: a cache bug
shows up as "the second run was slow" or, far worse, "the second run used a stale
transcript", and neither is visible from an end-to-end assertion on the output files.
"""

from __future__ import annotations

import json
import os

import pytest

from subtitler.cache import (
    FULL_HASH_MAX_BYTES,
    KEY_LEN,
    STAGE_ORDER,
    CacheError,
    StageCache,
    content_id,
    invalidated_from,
    stage_key,
    text_id,
)


class TestTextId:
    """A URL cannot be content-addressed before it has been downloaded."""

    def test_same_text_same_id(self):
        assert text_id("https://example.com/a") == text_id("https://example.com/a")

    def test_different_text_different_id(self):
        assert text_id("https://example.com/a") != text_id("https://example.com/b")

    def test_it_is_the_same_shape_as_a_content_id(self):
        """Both end up in the `input` field of a stage meta, so a human reading one cannot
        tell which kind it is and should not have to."""
        assert len(text_id("x")) == KEY_LEN


class TestContentId:
    def test_same_bytes_same_id(self, tmp_path):
        a, b = tmp_path / "a.bin", tmp_path / "b.bin"
        a.write_bytes(b"hello world")
        b.write_bytes(b"hello world")
        assert content_id(a) == content_id(b)

    def test_different_bytes_different_id(self, tmp_path):
        a, b = tmp_path / "a.bin", tmp_path / "b.bin"
        a.write_bytes(b"hello world")
        b.write_bytes(b"hello worlt")
        assert content_id(a) != content_id(b)

    def test_id_is_truncated(self, tmp_path):
        path = tmp_path / "a.bin"
        path.write_bytes(b"x")
        assert len(content_id(path)) == KEY_LEN

    def test_mtime_does_not_affect_the_id(self, tmp_path):
        """A plain `cp` gives byte-identical content a new mtime.

        This is the regression that rules out size+mtime+path as the fingerprint: keying on
        mtime would miss the cache after a copy, restore or `touch`, and the user would sit
        through a 75-second transcription of a file that had not changed.
        """
        path = tmp_path / "a.bin"
        path.write_bytes(b"hello world")
        before = content_id(path)
        os.utime(path, (1_000_000, 1_000_000))
        assert content_id(path) == before

    def test_large_file_is_sampled_not_read_whole(self, tmp_path):
        """Over the threshold, only three windows plus the length are read.

        The tradeoff this documents: an edit confined to the unsampled middle of a large
        file, leaving the byte length unchanged, is invisible. `--force` is the escape
        hatch. Asserting it here means nobody later "fixes" the sampling by accident and
        silently makes hashing a 3 GB video cost ten seconds again.
        """
        path = tmp_path / "big.bin"
        size = FULL_HASH_MAX_BYTES + (4 << 20)
        path.write_bytes(b"\0" * size)
        before = content_id(path)

        with path.open("r+b") as handle:
            # A quarter of the way in: outside the first, middle and last megabyte.
            handle.seek(size // 4)
            handle.write(b"CHANGED")
        assert content_id(path) == before

        # A length change is always caught, and every real re-encode changes the length.
        with path.open("ab") as handle:
            handle.write(b"!")
        assert content_id(path) != before

    def test_small_file_is_hashed_in_full(self, tmp_path):
        path = tmp_path / "small.bin"
        path.write_bytes(b"\0" * 4096)
        before = content_id(path)
        with path.open("r+b") as handle:
            handle.seek(2048)
            handle.write(b"CHANGED")
        assert content_id(path) != before


class TestStageKey:
    def test_param_order_does_not_change_the_key(self):
        """Two runs that build the same params in a different insertion order must agree.

        Without sort_keys the cache would never hit for a stage whose params dict is built
        conditionally, and the failure would look like "caching does not work" rather than
        like a serialization bug.
        """
        a = stage_key("cues", input_hash="abc", params={"max_line": 42, "max_dur": 7.0})
        b = stage_key("cues", input_hash="abc", params={"max_dur": 7.0, "max_line": 42})
        assert a == b

    def test_stage_name_is_part_of_the_key(self):
        args = {"input_hash": "abc", "params": {}}
        assert stage_key("cues", **args) != stage_key("transcribe", **args)

    def test_input_hash_is_part_of_the_key(self):
        assert stage_key("cues", input_hash="a", params={}) != stage_key(
            "cues", input_hash="b", params={}
        )

    def test_params_are_part_of_the_key(self):
        assert stage_key("cues", input_hash="a", params={"max_line": 42}) != stage_key(
            "cues", input_hash="a", params={"max_line": 30}
        )

    def test_non_json_values_do_not_raise(self, tmp_path):
        """Params carry Paths and None. A TypeError here would abort the whole run."""
        assert stage_key("burn", input_hash="a", params={"font": None, "dir": tmp_path})


class TestInvalidatedFrom:
    def test_none_invalidates_nothing(self):
        assert invalidated_from(None) == frozenset()

    def test_all_invalidates_everything(self):
        assert invalidated_from("all") == frozenset(STAGE_ORDER)

    def test_a_stage_invalidates_itself_and_everything_after(self):
        """`--force transcribe` must take the cues with it.

        Keeping cues computed from the previous transcript would leave the run internally
        inconsistent: subtitles that do not match the words in transcribe.json.
        """
        assert invalidated_from("transcribe") == frozenset({"transcribe", "cues", "fix", "burn"})

    def test_the_first_stage_invalidates_everything(self):
        assert invalidated_from("fetch") == frozenset(STAGE_ORDER)

    def test_forcing_the_trim_does_not_re_download(self):
        """The point of putting `fetch` first.

        A user fixing a badly chosen `--start` on a 400 MB download must re-cut, not
        re-fetch. `--force trim` is the manual version of the same rule the keys enforce.
        """
        assert "fetch" not in invalidated_from("trim")

    def test_the_extraction_is_downstream_of_the_cut(self):
        """Trimming after extraction would give the recognizer the whole file and leave
        every cue offset by the start time. The order here is what forbids that."""
        assert STAGE_ORDER.index("trim") < STAGE_ORDER.index("extract")
        assert STAGE_ORDER.index("fetch") < STAGE_ORDER.index("trim")

    def test_forcing_the_extraction_still_takes_everything_after_it(self):
        assert invalidated_from("extract") == frozenset(
            {"extract", "denoise", "transcribe", "cues", "fix", "burn"}
        )

    def test_the_last_stage_invalidates_only_itself(self):
        assert invalidated_from("burn") == frozenset({"burn"})

    def test_unknown_stage_names_the_choices(self):
        with pytest.raises(CacheError, match="transcribe"):
            invalidated_from("transcript")

    def test_fix_is_a_known_stage(self):
        """Phase 6 lands `fix` between cues and render. The seam is reserved now so that
        adding it does not change what `--force cues` already means."""
        assert "fix" in STAGE_ORDER
        assert STAGE_ORDER.index("cues") < STAGE_ORDER.index("fix")


def artifact(tmp_path, name="out.json", body="{}"):
    path = tmp_path / name
    path.write_text(body, encoding="utf-8")
    return path


class TestStageCache:
    def test_first_look_is_a_miss(self, tmp_path):
        cache = StageCache(tmp_path)
        entry = cache.begin("cues", input_hash="a", params={}, artifacts=())
        assert not entry.hit
        assert entry.reason == "no cached run"

    def test_committed_stage_hits_next_time(self, tmp_path):
        cache = StageCache(tmp_path)
        out = artifact(tmp_path)
        entry = cache.begin("cues", input_hash="a", params={"x": 1}, artifacts=(out,))
        cache.commit(entry)

        again = StageCache(tmp_path).begin(
            "cues", input_hash="a", params={"x": 1}, artifacts=(out,)
        )
        assert again.hit
        assert again.key == entry.key

    def test_changed_params_miss(self, tmp_path):
        cache = StageCache(tmp_path)
        out = artifact(tmp_path)
        cache.commit(cache.begin("cues", input_hash="a", params={"x": 1}, artifacts=(out,)))

        again = StageCache(tmp_path).begin(
            "cues", input_hash="a", params={"x": 2}, artifacts=(out,)
        )
        assert not again.hit
        assert again.reason == "inputs or parameters changed"

    def test_changed_input_hash_misses(self, tmp_path):
        """This is the chain working: a new upstream key must invalidate downstream."""
        cache = StageCache(tmp_path)
        out = artifact(tmp_path)
        cache.commit(cache.begin("cues", input_hash="a", params={}, artifacts=(out,)))

        again = StageCache(tmp_path).begin("cues", input_hash="b", params={}, artifacts=(out,))
        assert not again.hit

    def test_a_deleted_artifact_misses(self, tmp_path):
        """The meta may be valid while the file it describes is gone.

        `rm work/extract.wav` has to mean "extract again", not "the meta says it is fine".
        """
        cache = StageCache(tmp_path)
        out = artifact(tmp_path)
        cache.commit(cache.begin("cues", input_hash="a", params={}, artifacts=(out,)))
        out.unlink()

        again = StageCache(tmp_path).begin("cues", input_hash="a", params={}, artifacts=(out,))
        assert not again.hit
        assert "missing artifact" in again.reason

    def test_uncommitted_stage_misses(self, tmp_path):
        """A crash between running a stage and committing it must not leave a false hit.

        This is why `commit()` is a separate call rather than something `begin()` arranges:
        the meta lands only after the artifacts do.
        """
        cache = StageCache(tmp_path)
        out = artifact(tmp_path)
        cache.begin("cues", input_hash="a", params={}, artifacts=(out,))

        again = StageCache(tmp_path).begin("cues", input_hash="a", params={}, artifacts=(out,))
        assert not again.hit

    def test_forced_stage_misses_even_when_valid(self, tmp_path):
        out = artifact(tmp_path)
        StageCache(tmp_path).commit(
            StageCache(tmp_path).begin("cues", input_hash="a", params={}, artifacts=(out,))
        )
        forced = StageCache(tmp_path, forced=invalidated_from("cues"))
        entry = forced.begin("cues", input_hash="a", params={}, artifacts=(out,))
        assert not entry.hit
        assert entry.reason == "forced"

    def test_disabled_cache_never_hits_and_never_writes(self, tmp_path):
        """A dry run prints commands instead of running them.

        If it read the cache it would report stages as done that it never did, and if it
        wrote one it would claim artifacts that do not exist.
        """
        cache = StageCache(tmp_path, enabled=False)
        out = artifact(tmp_path)
        entry = cache.begin("cues", input_hash="a", params={}, artifacts=(out,))
        assert not entry.hit
        cache.commit(entry)
        assert not (tmp_path / "cues.meta.json").exists()

    def test_corrupt_meta_misses_instead_of_raising(self, tmp_path):
        """A truncated meta from an interrupted write must degrade to a re-run."""
        (tmp_path / "cues.meta.json").write_text("{not json", encoding="utf-8")
        entry = StageCache(tmp_path).begin("cues", input_hash="a", params={}, artifacts=())
        assert not entry.hit
        assert entry.reason == "unreadable meta"

    def test_schema_bump_misses(self, tmp_path):
        out = artifact(tmp_path)
        cache = StageCache(tmp_path)
        cache.commit(cache.begin("cues", input_hash="a", params={}, artifacts=(out,)))
        meta = tmp_path / "cues.meta.json"
        data = json.loads(meta.read_text(encoding="utf-8"))
        data["schema_version"] = 999
        meta.write_text(json.dumps(data), encoding="utf-8")

        entry = StageCache(tmp_path).begin("cues", input_hash="a", params={}, artifacts=(out,))
        assert not entry.hit

    def test_meta_records_the_input_hash_and_the_params(self, tmp_path):
        """The meta has to be readable by a human debugging a stale-cache report."""
        cache = StageCache(tmp_path)
        out = artifact(tmp_path)
        cache.commit(
            cache.begin("cues", input_hash="src123", params={"max_line": 42}, artifacts=(out,))
        )
        data = json.loads((tmp_path / "cues.meta.json").read_text(encoding="utf-8"))
        assert data["stage"] == "cues"
        assert data["input"] == "src123"
        assert data["params"] == {"max_line": 42}
        assert data["artifacts"] == [str(out)]

    def test_hits_and_misses_are_recorded(self, tmp_path):
        out = artifact(tmp_path)
        cache = StageCache(tmp_path)
        cache.commit(cache.begin("cues", input_hash="a", params={}, artifacts=(out,)))
        assert cache.misses == ["cues"]

        second = StageCache(tmp_path)
        second.begin("cues", input_hash="a", params={}, artifacts=(out,))
        assert second.hits == ["cues"]
