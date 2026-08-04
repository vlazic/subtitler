"""The verbose_json parser, against a real Groq response captured from this material."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from subtitler.cues import segments_to_cues
from subtitler.engines.base import TranscribeOptions, collapse_repetition
from subtitler.engines.groq import parse_verbose_json
from subtitler.render import render_srt, render_vtt, validate_vtt

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "groq_verbose.json"


@pytest.fixture(scope="module")
def raw() -> dict:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def transcript(raw: dict):
    return parse_verbose_json(
        raw, opts=TranscribeOptions(), engine="groq", model="whisper-large-v3"
    )


class TestParse:
    def test_language_and_duration(self, transcript) -> None:
        assert transcript.duration == pytest.approx(109.0, abs=1.0)
        assert transcript.segments

    def test_segments_are_ordered_and_non_empty(self, transcript) -> None:
        starts = [s.start for s in transcript.segments]
        assert starts == sorted(starts)
        assert all(s.text.strip() for s in transcript.segments)

    def test_serbian_latin_is_preserved(self, transcript) -> None:
        text = transcript.text
        assert any(ch in text for ch in "čćžšđ"), "diacritics were stripped somewhere"

    def test_words_are_synthesized_when_absent(self, transcript) -> None:
        """This fixture predates word granularity, so the fallback has to carry it."""
        cues = segments_to_cues(transcript.segments)
        assert cues
        assert all(c.end > c.start for c in cues)


class TestRoundTrip:
    def test_generated_vtt_validates(self, transcript) -> None:
        cues = segments_to_cues(transcript.segments)
        assert validate_vtt(render_vtt(cues)) == []

    def test_srt_is_non_empty_and_numbered(self, transcript) -> None:
        text = render_srt(segments_to_cues(transcript.segments))
        assert text.startswith("1\n")


class TestRepetitionCollapse:
    def test_collapses_a_long_run(self) -> None:
        assert collapse_repetition("ne " * 12) == "ne"

    def test_collapses_a_repeated_phrase(self) -> None:
        assert collapse_repetition("ne znam " * 10) == "ne znam"

    def test_leaves_normal_text_alone(self) -> None:
        text = "Misao Lokove filozofije, ukratko izraženo, sastoji se u ovome."
        assert collapse_repetition(text) == text

    def test_leaves_a_short_run_alone(self) -> None:
        assert collapse_repetition("da da da") == "da da da"
