"""Fetching a URL, without a network.

Nothing here touches YouTube, and that is deliberate: a test that downloads is slow,
rate-limited and fails for reasons that have nothing to do with this code. yt-dlp is
stubbed the way LiteLLM is in `test_postedit.py`, so what is under test is the part this
project owns: which format is asked for, where the file is put, what reaches the progress
callback, and whether a failure a user can act on arrives as a sentence.

The real download is verified by hand and recorded in the README's "Verified on" table.
"""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import pytest

from subtitler import cache as cache_mod
from subtitler import fetch
from subtitler.fetch import (
    FORMATS,
    Fetched,
    FetchError,
    is_url,
    normalize_url,
    options,
    read_info,
    slugify,
    url_id,
    work_stem,
    write_info,
)

URL = "https://www.youtube.com/watch?v=aaaaaaaaaaa"


class FakeYoutubeDL:
    """Stands in for `yt_dlp.YoutubeDL`. Writes a file instead of downloading one."""

    def __init__(self, opts):
        self.opts = opts
        self.calls: list[str] = []

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def extract_info(self, url, download=True):
        self.calls.append(url)
        # The template is `<dir>/fetch.%(ext)s`; produce what a real run would.
        ext = self.opts.get("merge_output_format") or "m4a"
        path = Path(self.opts["outtmpl"].replace("%(ext)s", ext))
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"not really a video")
        for hook in self.opts.get("progress_hooks", []):
            hook(
                {"status": "downloading", "downloaded_bytes": 5_000_000, "total_bytes": 10_000_000}
            )
            hook({"status": "finished", "filename": str(path)})
        return {
            "id": "aaaaaaaaaaa",
            "title": "Neki Naslov: sa čćžšđ",
            "duration": 42.5,
            "extractor_key": "Youtube",
            "requested_downloads": [{"filepath": str(path)}],
        }


@pytest.fixture
def fake_yt_dlp(monkeypatch):
    module = types.SimpleNamespace(YoutubeDL=FakeYoutubeDL)
    monkeypatch.setitem(sys.modules, "yt_dlp", module)
    return FakeYoutubeDL


class TestIsUrl:
    """`run` decides between a path and a URL from this and nothing else."""

    @pytest.mark.parametrize(
        "text",
        [
            "https://www.youtube.com/watch?v=x",
            "http://example.com/a.mp4",
            "HTTPS://EXAMPLE.COM/a",
            "  https://example.com/a  ",
        ],
    )
    def test_urls(self, text):
        assert is_url(text)

    @pytest.mark.parametrize(
        "text",
        [
            "video.mp4",
            "/home/me/video.mp4",
            "~/Videos/a.mkv",
            "ftp://example.com/a.mp4",
            "C:/videos/a.mp4",
            "",
        ],
    )
    def test_not_urls(self, text):
        assert not is_url(text)

    def test_a_path_is_never_probed_on_disk(self, tmp_path):
        """Detection must not depend on the file existing, or a typo in a filename would
        be silently sent to yt-dlp as a URL."""
        assert not is_url(tmp_path / "does-not-exist.mp4")


class TestUrlIdentity:
    def test_tracking_parameters_do_not_change_the_key(self):
        """The share button appends `si=`. Pasting the share link and the plain link must
        not download the same video twice."""
        assert url_id(URL) == url_id(URL + "&si=Ab12Cd")

    def test_a_timestamp_parameter_does_change_the_key(self):
        """`t` and `list` change which media is meant, so they stay in the key."""
        assert url_id(URL) != url_id(URL + "&t=90")

    def test_the_video_id_still_matters(self):
        assert url_id(URL) != url_id(URL.replace("aaaaaaaaaaa", "bbbbbbbbbbb"))

    def test_the_id_is_the_shape_of_a_content_id(self):
        assert len(url_id(URL)) == cache_mod.KEY_LEN

    def test_work_stem_is_stable_and_filesystem_safe(self):
        stem = work_stem(URL)
        assert stem == work_stem(URL)
        assert "/" not in stem and ":" not in stem and "?" not in stem

    def test_a_trailing_slash_is_not_a_different_video(self):
        assert normalize_url("https://example.com/a/") == normalize_url("https://example.com/a")


class TestSlugify:
    def test_serbian_diacritics_survive(self):
        """The output files are named from the video's title, and this project's material
        is Serbian. Stripping to ASCII would mangle every one of them."""
        assert slugify("Gozba: peti deo čćžšđ") == "Gozba-peti-deo-čćžšđ"

    def test_path_separators_cannot_appear(self):
        assert "/" not in slugify("a/b/c")

    def test_an_empty_title_gives_an_empty_stem(self):
        assert slugify("   ") == ""

    def test_a_title_that_is_all_punctuation_does_not_become_a_dotfile(self):
        assert not slugify("...").startswith(".")


class TestFormatSelection:
    def test_srt_only_asks_for_audio_not_1080p(self, tmp_path):
        """The bandwidth criterion. A run that produces a text file must not download the
        pixels it will never look at."""
        opts = options(tmp_path, kind="audio")
        assert "bestaudio" in opts["format"]
        assert "bestvideo" not in opts["format"]
        assert "merge_output_format" not in opts

    def test_a_burn_run_asks_for_video_and_one_merged_file(self, tmp_path):
        """A DASH source arrives as two files. Without the merge the pipeline would
        transcribe one and burn onto the other."""
        opts = options(tmp_path, kind="video")
        assert "bestvideo" in opts["format"]
        assert opts["merge_output_format"] == "mp4"

    def test_an_unknown_kind_is_refused(self, tmp_path):
        with pytest.raises(FetchError, match="audio"):
            options(tmp_path, kind="1080p")

    def test_nothing_is_written_outside_the_given_directory(self, tmp_path):
        """Non-negotiable 4: the download goes where the caller says, never into the CWD."""
        opts = options(tmp_path, kind="video")
        assert opts["outtmpl"].startswith(str(tmp_path))
        # `paths` alongside an absolute template is ignored by yt-dlp, with a warning.
        assert "paths" not in opts

    def test_a_playlist_url_fetches_one_video(self, tmp_path):
        """`--start`/`--end` are about one video. Pointing at a watch URL that happens to
        carry `&list=` must not download the whole playlist."""
        assert options(tmp_path, kind="video")["noplaylist"] is True

    def test_yt_dlp_does_not_write_to_stdout(self, tmp_path):
        """`run --json` owns stdout. yt-dlp prints there unless it is given a logger."""
        opts = options(tmp_path, kind="video")
        assert opts["quiet"] is True
        assert opts["logger"] is not None


class TestFetch:
    def test_the_file_lands_in_the_given_directory(self, fake_yt_dlp, tmp_path):
        got = fetch.fetch(URL, tmp_path, kind="video")
        assert got.path.parent == tmp_path
        assert got.path.exists()

    def test_the_metadata_comes_back_for_naming_the_outputs(self, fake_yt_dlp, tmp_path):
        got = fetch.fetch(URL, tmp_path, kind="video")
        assert got.id == "aaaaaaaaaaa"
        assert got.duration == 42.5
        assert got.stem == "Neki-Naslov-sa-čćžšđ"

    def test_progress_reaches_the_caller(self, fake_yt_dlp, tmp_path):
        """`models.download` takes the same `progress` callback, so a GUI that streams one
        can stream the other without learning a second shape."""
        lines: list[str] = []
        fetch.fetch(URL, tmp_path, kind="video", progress=lines.append)
        assert any("50%" in line for line in lines)
        assert any("downloaded" in line for line in lines)

    def test_progress_is_optional(self, fake_yt_dlp, tmp_path):
        assert fetch.fetch(URL, tmp_path, kind="video").path.exists()

    def test_a_playlist_wrapper_is_unwrapped(self, fake_yt_dlp, tmp_path, monkeypatch):
        """Some extractors return a playlist envelope even with noplaylist set."""
        real = FakeYoutubeDL.extract_info

        def wrapped(self, url, download=True):
            return {"_type": "playlist", "entries": [real(self, url, download)]}

        monkeypatch.setattr(FakeYoutubeDL, "extract_info", wrapped)
        assert fetch.fetch(URL, tmp_path, kind="video").id == "aaaaaaaaaaa"

    def test_an_empty_playlist_is_a_sentence(self, fake_yt_dlp, tmp_path, monkeypatch):
        monkeypatch.setattr(
            FakeYoutubeDL,
            "extract_info",
            lambda self, url, download=True: {"_type": "playlist", "entries": []},
        )
        with pytest.raises(FetchError, match="no playable media"):
            fetch.fetch(URL, tmp_path, kind="video")

    def test_success_with_no_file_is_not_reported_as_success(
        self, fake_yt_dlp, tmp_path, monkeypatch
    ):
        """The one failure mode that would otherwise surface as ffprobe complaining about
        a file that is not there."""
        monkeypatch.setattr(
            FakeYoutubeDL, "extract_info", lambda self, url, download=True: {"id": "x"}
        )
        with pytest.raises(FetchError, match="no media file"):
            fetch.fetch(URL, tmp_path, kind="video")


class TestErrorsAreSentences:
    """Every one of these is something a user can do something about.

    A traceback out of yt-dlp names a Python module; these name the situation.
    """

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("ERROR: [youtube] x: Private video. Sign in if you've been", "private"),
            ("ERROR: The uploader has not made this video available in your country", "region"),
            ("ERROR: [youtube] x: Video unavailable", "unavailable"),
            ("ERROR: Sign in to confirm you're not a bot", "signed-in session"),
            ("ERROR: Unsupported URL: https://example.com/x", "nothing downloadable"),
            ("ERROR: [youtube] x: Unable to extract player response", "too old"),
            ("ERROR: HTTP Error 429: Too Many Requests", "rate-limiting"),
            # The three below are verbatim from yt-dlp 2026.7.4 on this machine, not
            # invented. Both of the last two open with "Unable to download webpage", which
            # is also how a stale extractor announces itself, and getting that order wrong
            # told a user with no network to upgrade yt-dlp.
            (
                "ERROR: [generic] x: Unable to download webpage: HTTPSConnection("
                "host='nope.invalid', port=443): Failed to resolve 'nope.invalid' "
                "([Errno -2] Name or service not known)",
                "network is not reachable",
            ),
            (
                "ERROR: Unable to download webpage: <urlopen error [Errno -3] "
                "Temporary failure in name resolution>",
                "network is not reachable",
            ),
            (
                "ERROR: [generic] nothing-here: Unable to download webpage: "
                "HTTP Error 404: Not Found (caused by <HTTPError 404: Not Found>)",
                "unavailable",
            ),
        ],
    )
    def test_each_common_failure_names_the_situation(
        self, fake_yt_dlp, tmp_path, monkeypatch, raw, expected
    ):
        def boom(self, url, download=True):
            raise RuntimeError(raw)

        monkeypatch.setattr(FakeYoutubeDL, "extract_info", boom)
        with pytest.raises(FetchError, match=expected):
            fetch.fetch(URL, tmp_path, kind="video")

    def test_a_stale_yt_dlp_is_told_how_to_update(self, fake_yt_dlp, tmp_path, monkeypatch):
        """The single most common cause of a download that used to work: sites change, and
        an extractor from six months ago no longer reads the page."""
        monkeypatch.setattr(
            FakeYoutubeDL,
            "extract_info",
            lambda self, url, download=True: (_ for _ in ()).throw(
                RuntimeError("ERROR: Unable to extract initial data")
            ),
        )
        with pytest.raises(FetchError, match="upgrade-package yt-dlp"):
            fetch.fetch(URL, tmp_path, kind="video")

    def test_an_unrecognised_error_still_says_something(self, fake_yt_dlp, tmp_path, monkeypatch):
        monkeypatch.setattr(
            FakeYoutubeDL,
            "extract_info",
            lambda self, url, download=True: (_ for _ in ()).throw(RuntimeError("ERROR: nonsense")),
        )
        with pytest.raises(FetchError, match="nonsense"):
            fetch.fetch(URL, tmp_path, kind="video")

    def test_a_missing_yt_dlp_names_the_install_line(self, monkeypatch, tmp_path):
        """Same shape as a missing LiteLLM: the fix is one command, so print the command."""
        import builtins

        real = builtins.__import__

        def blocked(name, *args, **kwargs):
            if name == "yt_dlp":
                raise ImportError("no module named yt_dlp")
            return real(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", blocked)
        monkeypatch.delitem(sys.modules, "yt_dlp", raising=False)
        with pytest.raises(FetchError, match="--extra fetch"):
            fetch.fetch(URL, tmp_path, kind="video")


class TestInfoSidecar:
    """`fetch.json` is what makes a warm run know the download's name and extension."""

    def test_round_trip(self, tmp_path):
        got = Fetched(path=tmp_path / "fetch.webm", url=URL, id="x", title="T", duration=3.0)
        write_info(tmp_path / "fetch.json", got)
        assert read_info(tmp_path / "fetch.json") == got

    def test_a_missing_record_is_none_not_an_exception(self, tmp_path):
        assert read_info(tmp_path / "nope.json") is None

    def test_a_corrupt_record_degrades_to_none(self, tmp_path):
        """An interrupted write must mean "download again", not "crash on a warm run"."""
        (tmp_path / "fetch.json").write_text("{truncated", encoding="utf-8")
        assert read_info(tmp_path / "fetch.json") is None

    def test_a_record_without_a_path_is_none(self, tmp_path):
        (tmp_path / "fetch.json").write_text(json.dumps({"url": URL}), encoding="utf-8")
        assert read_info(tmp_path / "fetch.json") is None


class TestCacheParams:
    def test_the_shape_is_part_of_the_key(self):
        """Switching from `--srt-only` to a burn has to fetch the video, because the audio
        already on disk has no pixels in it."""
        assert fetch.cache_params("audio") != fetch.cache_params("video")

    def test_the_format_selector_is_in_the_key(self):
        """Changing the selector in a future release must re-download rather than serve a
        file chosen by the old one."""
        assert fetch.cache_params("video")["format"] == FORMATS["video"]

    def test_the_directory_is_not_in_the_key(self):
        """Moving a work directory does not change what was downloaded."""
        assert "path" not in fetch.cache_params("video")
        assert "dir" not in fetch.cache_params("video")
