"""The LLM correction pass, without an API key.

Every model call here is a stub. What is under test is not whether a model writes good
Serbian, it is the three properties the bash pipeline this replaces got wrong:

  1. timestamps cannot change, because they are never sent;
  2. a reply that does not match its request is discarded, and the batch is named;
  3. no sampling parameter is sent unless the user asked for one.

The last is not a style preference. Current Claude models (Opus 5, Sonnet 5, Opus 4.7/4.8)
return a 400 for `temperature`, `top_p` or `top_k`, and LiteLLM forwards whatever it is
handed, so a default temperature would make the default model unusable.
"""

from __future__ import annotations

import json
import sys
import types

import pytest

from subtitler import postedit
from subtitler.cues import CueConfig
from subtitler.model import Cue
from subtitler.postedit import (
    BatchRejected,
    FixConfig,
    FixError,
    batch_cues,
    drop_intro_cues,
    fix_cues,
    load_intro_phrases,
    load_prompt,
    parse_reply,
    render_request,
    strip_markdown,
)


def cue(index: int, text: str, start: float | None = None, end: float | None = None) -> Cue:
    start = index * 2.0 if start is None else start
    end = start + 1.7 if end is None else end
    return Cue(index=index, start=start, end=end, lines=(text,))


def cues(*texts: str) -> tuple[Cue, ...]:
    return tuple(cue(i, t) for i, t in enumerate(texts, start=1))


def echo(_system: str, user: str) -> str:
    """A model that returns every item unchanged. The identity case must still validate."""
    return json.dumps(json.loads(user), ensure_ascii=False)


def rewriting(mapping: dict[int, str]):
    """A model that replaces the text of the given indices and passes the rest through."""

    def complete(_system: str, user: str) -> str:
        items = json.loads(user)
        return json.dumps(
            [{"i": it["i"], "text": mapping.get(it["i"], it["text"])} for it in items],
            ensure_ascii=False,
        )

    return complete


def constant(reply: str):
    def complete(_system: str, _user: str) -> str:
        return reply

    return complete


# --------------------------------------------------------------------------------------
# The acceptance criterion
# --------------------------------------------------------------------------------------


class TestTimestampsNeverMove:
    def test_correcting_text_leaves_every_timestamp_identical(self):
        """The Phase 6 acceptance criterion, as an assertion.

        The pipeline this replaces asked the model to "keep the timestamps, just correct
        the text" with the clock inside the payload, and then needed a validator and two
        repair scripts to find out when it had not. Here the clock is never in the payload.
        """
        original = cues("prvi red", "drugi red", "treci red")
        fixed, report = fix_cues(
            original,
            FixConfig(),
            complete=rewriting({1: "Prvi red.", 2: "Drugi red.", 3: "Treći red."}),
            log=lambda _m: None,
        )
        assert [c.text for c in fixed] == ["Prvi red.", "Drugi red.", "Treći red."]
        assert [(c.index, c.start, c.end) for c in fixed] == [
            (c.index, c.start, c.end) for c in original
        ]
        assert report.changed == 3

    def test_the_request_carries_no_timestamp(self):
        """A timestamp that is never sent cannot come back wrong."""
        payload = render_request(cues("zdravo", "svete"))
        assert json.loads(payload) == [
            {"i": 1, "text": "zdravo"},
            {"i": 2, "text": "svete"},
        ]
        assert "start" not in payload and "-->" not in payload

    def test_a_partly_rejected_run_still_leaves_every_timestamp_alone(self):
        """The discarded half keeps its originals; the corrected half keeps its clock."""
        original = cues("a", "b", "c", "d")
        fixed, _ = fix_cues(
            original,
            FixConfig(batch_size=2, workers=1),
            complete=lambda s, u: (
                "nope" if json.loads(u)[0]["i"] == 1 else rewriting({3: "C.", 4: "D."})(s, u)
            ),
            log=lambda _m: None,
        )
        assert [(c.index, c.start, c.end) for c in fixed] == [
            (c.index, c.start, c.end) for c in original
        ]
        assert [c.text for c in fixed] == ["a", "b", "C.", "D."]


# --------------------------------------------------------------------------------------
# Rule 1: sampling parameters
# --------------------------------------------------------------------------------------


class FakeLiteLLM(types.ModuleType):
    """A stand-in `litellm` module, so this runs on CI where the fix extra is not synced."""

    def __init__(self, content: str = "[]"):
        super().__init__("litellm")
        self.calls: list[dict] = []
        self.suppress_debug_info = False
        self._content = content

    def completion(self, **kwargs):
        self.calls.append(kwargs)
        message = types.SimpleNamespace(content=self._content)
        return types.SimpleNamespace(choices=[types.SimpleNamespace(message=message)])


@pytest.fixture
def fake_litellm(monkeypatch):
    module = FakeLiteLLM()
    monkeypatch.setitem(sys.modules, "litellm", module)
    return module


class TestSamplingParameters:
    def test_no_sampling_parameter_is_sent_by_default(self, fake_litellm):
        """Regression: Claude Opus 5 / Sonnet 5 return a 400 for temperature/top_p/top_k.

        LiteLLM forwards every sampling parameter it is given, so the kwargs dict has to be
        built empty rather than built with defaults and pruned.
        """
        postedit._litellm_complete("system", "user", FixConfig())
        sent = fake_litellm.calls[0]
        assert "temperature" not in sent
        assert "top_p" not in sent
        assert "top_k" not in sent

    def test_the_default_model_is_the_anthropic_one(self, fake_litellm):
        postedit._litellm_complete("system", "user", FixConfig())
        assert fake_litellm.calls[0]["model"] == "anthropic/claude-sonnet-5"

    def test_an_explicit_temperature_is_forwarded(self, fake_litellm):
        postedit._litellm_complete("system", "user", FixConfig(temperature=0.2))
        assert fake_litellm.calls[0]["temperature"] == 0.2

    def test_a_temperature_of_zero_is_forwarded_not_treated_as_unset(self, fake_litellm):
        """`if cfg.temperature:` would silently drop 0.0, the one value people ask for."""
        postedit._litellm_complete("system", "user", FixConfig(temperature=0.0))
        assert fake_litellm.calls[0]["temperature"] == 0.0

    def test_any_litellm_model_string_passes_through_untouched(self, fake_litellm):
        """`--fix-model` must reach openai, groq and ollama without a code change."""
        for model in ("openai/gpt-4o", "groq/llama-3.3-70b-versatile", "ollama/llama3.1"):
            postedit._litellm_complete("system", "user", FixConfig(model=model))
        assert [c["model"] for c in fake_litellm.calls] == [
            "openai/gpt-4o",
            "groq/llama-3.3-70b-versatile",
            "ollama/llama3.1",
        ]

    def test_a_missing_litellm_is_reported_once_with_the_install_line(self, monkeypatch):
        """Regression: the import failure used to arrive once per batch, as "every batch
        failed", which reads like a model problem for what is a one-line install.
        """
        import builtins

        real = builtins.__import__

        def blocked(name, *args, **kwargs):
            if name == "litellm":
                raise ImportError("no module named litellm")
            return real(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", blocked)
        monkeypatch.delitem(sys.modules, "litellm", raising=False)
        with pytest.raises(FixError, match="--extra fix"):
            fix_cues(cues("a"), FixConfig(), log=lambda _m: None)

    def test_the_system_and_user_messages_are_two_separate_turns(self, fake_litellm):
        postedit._litellm_complete("SYS", "USR", FixConfig())
        assert fake_litellm.calls[0]["messages"] == [
            {"role": "system", "content": "SYS"},
            {"role": "user", "content": "USR"},
        ]


# --------------------------------------------------------------------------------------
# Rule 2: validation
# --------------------------------------------------------------------------------------


class TestValidation:
    def test_a_reply_with_the_wrong_count_is_rejected(self):
        batch = cues("a", "b", "c")
        with pytest.raises(BatchRejected, match="2 items, expected 3"):
            parse_reply(json.dumps([{"i": 1, "text": "a"}, {"i": 2, "text": "b"}]), batch)

    def test_a_reply_with_the_wrong_indices_is_rejected(self):
        """Same count, renumbered. This is the failure the old validator could not name.

        A model that returns 1..3 for a batch of cues 41..43 produces a file where the
        correction was applied to the wrong lines, and every timestamp still looks fine.
        """
        batch = (cue(41, "a"), cue(42, "b"), cue(43, "c"))
        reply = json.dumps([{"i": i, "text": "x"} for i in (1, 2, 3)])
        with pytest.raises(BatchRejected, match="index set does not match"):
            parse_reply(reply, batch)

    def test_a_reply_that_is_not_json_is_rejected(self):
        with pytest.raises(BatchRejected, match="not JSON"):
            parse_reply("Sure! Here are the corrections:", cues("a"))

    def test_a_reply_that_is_an_object_is_rejected(self):
        with pytest.raises(BatchRejected, match="expected a JSON array"):
            parse_reply(json.dumps({"1": "a"}), cues("a"))

    def test_an_empty_reply_is_rejected(self):
        with pytest.raises(BatchRejected, match="empty reply"):
            parse_reply("   ", cues("a"))

    def test_an_emptied_text_is_rejected(self):
        """A model that returns "" for a cue would delete a subtitle, not correct it."""
        with pytest.raises(BatchRejected, match="is empty"):
            parse_reply(json.dumps([{"i": 1, "text": "  "}]), cues("a"))

    def test_a_non_string_text_is_rejected(self):
        with pytest.raises(BatchRejected, match="is a int"):
            parse_reply(json.dumps([{"i": 1, "text": 7}]), cues("a"))

    def test_a_bare_string_list_is_rejected(self):
        """Positional-only output cannot be checked against the index set, so it is not accepted."""
        with pytest.raises(BatchRejected, match="not an object"):
            parse_reply(json.dumps(["a"]), cues("a"))

    def test_a_fenced_reply_is_accepted(self):
        """The prompt says no code fence; models add one anyway, and it is free to strip."""
        reply = "```json\n" + json.dumps([{"i": 1, "text": "ok"}]) + "\n```"
        assert parse_reply(reply, cues("a")) == {1: "ok"}

    def test_a_valid_reply_round_trips(self):
        assert parse_reply(json.dumps([{"i": 1, "text": "ok"}]), cues("a")) == {1: "ok"}


class TestRejectedBatches:
    def test_a_bad_batch_keeps_its_originals_and_the_others_are_still_corrected(self):
        """One batch failing must not cost the whole run, and must not go unreported."""
        original = cues("a", "b", "c", "d")
        calls = {"n": 0}

        def flaky(_system: str, user: str) -> str:
            calls["n"] += 1
            items = json.loads(user)
            if items[0]["i"] == 1:
                return "I cannot help with that."
            return json.dumps([{"i": it["i"], "text": it["text"].upper()} for it in items])

        messages: list[str] = []
        fixed, report = fix_cues(
            original,
            FixConfig(batch_size=2, workers=1),
            complete=flaky,
            log=messages.append,
        )
        assert [c.text for c in fixed] == ["a", "b", "C", "D"]
        assert len(report.rejected) == 1
        assert report.changed == 2

    def test_the_warning_names_the_batch_and_its_cue_range(self):
        """The old repair scripts could only say "broken". This says which cues."""
        messages: list[str] = []
        fix_cues(
            cues(*[f"r{i}" for i in range(1, 7)]),
            FixConfig(batch_size=2, workers=1),
            complete=lambda s, u: "garbage" if json.loads(u)[0]["i"] == 3 else echo(s, u),
            log=messages.append,
        )
        discarded = [m for m in messages if "discarded" in m]
        assert len(discarded) == 1
        assert "batch 2/3" in discarded[0]
        assert "cues 3-4" in discarded[0]

    def test_a_rejected_batch_is_retried_once_before_being_discarded(self):
        """One retry, not a loop: the caller is billed per attempt."""
        attempts = {"n": 0}

        def unreliable(system: str, user: str) -> str:
            attempts["n"] += 1
            if attempts["n"] == 1:
                return "sorry"
            return echo(system, user)

        _, report = fix_cues(cues("a", "b"), FixConfig(), complete=unreliable, log=lambda _m: None)
        assert attempts["n"] == 2
        assert report.rejected == []

    def test_a_provider_error_is_not_retried(self):
        """Auth and quota errors do not get better on a second identical request."""
        attempts = {"n": 0}

        def dead(_system: str, _user: str) -> str:
            attempts["n"] += 1
            raise ConnectionError("AuthenticationError: no key")

        with pytest.raises(FixError, match="every batch failed"):
            fix_cues(cues("a"), FixConfig(), complete=dead, log=lambda _m: None)
        assert attempts["n"] == 1

    def test_every_batch_failing_raises_rather_than_silently_writing_the_original(self):
        """A run that reports success having corrected nothing is the worst outcome.

        It is indistinguishable from a working run in the output file, so a missing API key
        would produce an uncorrected transcript and a green exit code.
        """
        with pytest.raises(FixError, match="every batch failed"):
            fix_cues(
                cues("a", "b", "c"),
                FixConfig(batch_size=1),
                complete=constant("not json"),
                log=lambda _m: None,
            )


class TestBatching:
    def test_batches_are_whole_cues(self):
        """The bug: `awk` chunked the VTT by counting blank lines, splitting cues in half."""
        batched = batch_cues(cues(*[f"r{i}" for i in range(1, 8)]), 3)
        assert [len(b) for b in batched] == [3, 3, 1]
        assert [c.index for b in batched for c in b] == list(range(1, 8))

    def test_a_batch_size_below_one_is_rejected_rather_than_looping_forever(self):
        with pytest.raises(FixError, match="at least 1"):
            batch_cues(cues("a"), 0)

    def test_every_cue_appears_in_exactly_one_batch(self):
        source = cues(*[f"r{i}" for i in range(1, 101)])
        batched = batch_cues(source, 40)
        assert sum(len(b) for b in batched) == 100
        assert len({c.index for b in batched for c in b}) == 100

    def test_batches_are_reassembled_in_order_regardless_of_completion_order(self):
        """The pool returns whichever batch finishes first; the cue list must not reorder."""
        import time

        def slow_first(_system: str, user: str) -> str:
            items = json.loads(user)
            if items[0]["i"] == 1:
                time.sleep(0.05)
            return json.dumps([{"i": it["i"], "text": it["text"].upper()} for it in items])

        fixed, _ = fix_cues(
            cues(*[f"r{i}" for i in range(1, 13)]),
            FixConfig(batch_size=2, workers=4),
            complete=slow_first,
            log=lambda _m: None,
        )
        assert [c.index for c in fixed] == list(range(1, 13))
        assert [c.text for c in fixed] == [f"R{i}" for i in range(1, 13)]

    def test_no_cues_means_no_model_call(self):
        def explode(_s, _u):  # pragma: no cover - must never run
            raise AssertionError("called the model for an empty cue list")

        fixed, report = fix_cues((), FixConfig(), complete=explode)
        assert fixed == () and report.batches == 0


# --------------------------------------------------------------------------------------
# Rule 3: the prompt is a file
# --------------------------------------------------------------------------------------


class TestPrompts:
    def test_the_default_prompt_loads_and_says_nothing_about_serbian(self):
        """`postedit.md` is the generic one. The domain variant is opt-in."""
        text = load_prompt("postedit")
        assert "serbian" not in text.lower() and "srpski" not in text.lower()
        assert "language" in text.lower()

    def test_the_gozba_prompt_carries_the_domain_formatting(self):
        text = load_prompt("gozba")
        assert "**bold**" in text and "*italic" in text
        assert "Latin script" in text

    def test_every_prompt_carries_the_contract_the_validator_enforces(self):
        """A prompt file that could drop the contract would silently reject every batch."""
        for name in ("postedit", "gozba"):
            text = load_prompt(name)
            assert "JSON array" in text
            assert '{"i": <integer>, "text":' in text

    def test_the_editor_facing_preamble_is_not_sent_to_the_model(self):
        assert "Edit freely below this line" not in load_prompt("postedit")

    def test_a_prompt_can_be_a_path(self, tmp_path):
        path = tmp_path / "custom.md"
        path.write_text("# notes\n\n---\n\nBe terse.\n", encoding="utf-8")
        assert "Be terse." in load_prompt(str(path))

    def test_a_file_without_a_rule_is_used_whole(self, tmp_path):
        path = tmp_path / "plain.md"
        path.write_text("Be terse.\n", encoding="utf-8")
        assert load_prompt(str(path)).startswith("Be terse.")

    def test_an_unknown_prompt_names_the_ones_that_exist(self):
        with pytest.raises(FixError, match="postedit"):
            load_prompt("nonexistent-prompt")

    def test_the_prompt_content_is_in_the_cache_key(self, tmp_path):
        """Editing a prompt file must re-run the pass; recording only its name would not."""
        a, b = tmp_path / "a.md", tmp_path / "b.md"
        a.write_text("Correct the text.", encoding="utf-8")
        b.write_text("Correct the text harder.", encoding="utf-8")
        key_a = postedit.cache_params(FixConfig(prompt=str(a)), CueConfig())["prompt_sha"]
        key_b = postedit.cache_params(FixConfig(prompt=str(b)), CueConfig())["prompt_sha"]
        assert key_a != key_b

    def test_worker_count_is_not_in_the_cache_key(self):
        """Thread count cannot change the output, so it must not re-bill the whole pass."""
        one = postedit.cache_params(FixConfig(workers=1), CueConfig())
        eight = postedit.cache_params(FixConfig(workers=8), CueConfig())
        assert one == eight


# --------------------------------------------------------------------------------------
# strip_markdown, ported from gozba2/emisije/replace_markdown_formatting.py
# --------------------------------------------------------------------------------------


class TestStripMarkdown:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("**Aristotel** je rekao", "Aristotel je rekao"),
            ("*Gozba* je dijalog", "Gozba je dijalog"),
            ("**Platon** i njegova *Gozba*", "Platon i njegova Gozba"),
            ("bez ikakvog markdowna", "bez ikakvog markdowna"),
            ("`kod` u tekstu", "kod u tekstu"),
            ("# Naslov", "Naslov"),
            ("- stavka", "stavka"),
        ],
    )
    def test_emphasis_is_removed_by_default(self, raw, expected):
        """The default output is burned in through libass, which renders <b> literally."""
        assert strip_markdown(raw) == expected

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("**Aristotel**", "<b>Aristotel</b>"),
            ("*Gozba*", "<i>Gozba</i>"),
            ("**Platon** i *Gozba*", "<b>Platon</b> i <i>Gozba</i>"),
        ],
    )
    def test_html_mode_reproduces_the_original_scripts_behaviour(self, raw, expected):
        assert strip_markdown(raw, html=True) == expected

    def test_bold_is_never_seen_as_two_italics(self):
        """The ported bug class: `**x**` matched by the single-asterisk rule as `*` + `*x*`."""
        assert strip_markdown("**x**", html=True) == "<b>x</b>"
        assert "<i>" not in strip_markdown("**x**", html=True)

    def test_a_lone_asterisk_is_left_alone(self):
        """Unpaired punctuation must not eat the rest of the cue."""
        assert strip_markdown("2 * 3 = 6") == "2 * 3 = 6"

    def test_diacritics_survive(self):
        assert strip_markdown("**Šta je čovek?**") == "Šta je čovek?"

    def test_the_pass_applies_it_to_what_the_model_returns(self):
        """Told not to emit markdown, models emit it anyway. That is why this is not optional."""
        fixed, _ = fix_cues(
            cues("aristotel je rekao"),
            FixConfig(),
            complete=rewriting({1: "**Aristotel** je rekao"}),
            log=lambda _m: None,
        )
        assert fixed[0].text == "Aristotel je rekao"

    def test_html_markup_mode_reaches_the_cue(self):
        fixed, _ = fix_cues(
            cues("aristotel"),
            FixConfig(markup="html"),
            complete=rewriting({1: "**Aristotel**"}),
            log=lambda _m: None,
        )
        assert fixed[0].text == "<b>Aristotel</b>"


# --------------------------------------------------------------------------------------
# drop_intro_cues, ported from gozba2/emisije/remove_intro.py
# --------------------------------------------------------------------------------------

INTRO = "Poštovani slušaoci, slušate Gozbu Radio Beograda 2."


class TestDropIntroPhrases:
    def test_only_the_matching_cue_is_dropped(self):
        """Regression for the ported bug: the original deleted everything *before* it too.

        One ident line 40 seconds into a show silently threw away the first 40 seconds.
        """
        source = (cue(1, "Dobar dan."), cue(2, INTRO), cue(3, "Danas govorimo o Platonu."))
        kept = drop_intro_cues(source, [INTRO])
        assert [c.text for c in kept] == ["Dobar dan.", "Danas govorimo o Platonu."]

    def test_the_survivors_keep_their_timestamps(self):
        source = (cue(1, "a"), cue(2, INTRO), cue(3, "b"))
        kept = drop_intro_cues(source, [INTRO])
        assert [(c.start, c.end) for c in kept] == [
            (source[0].start, source[0].end),
            (source[2].start, source[2].end),
        ]

    def test_the_survivors_are_renumbered_sequentially(self):
        """An SRT with a gap in its cue numbers is not an SRT some players will read."""
        source = (cue(1, INTRO), cue(2, "a"), cue(3, "b"))
        assert [c.index for c in drop_intro_cues(source, [INTRO])] == [1, 2]

    @pytest.mark.parametrize(
        "variant",
        [
            "Poštovani slušaoci, slušate Gozbu Radio Beograda 2.",
            "poštovani slušaoci, slušate gozbu radio beograda 2.",
            'Poštovani slušaoci, slušate "Gozbu" Radio Beograda 2.',
            "Poštovani slušaoci, slušate <i>Gozbu</i> Radio Beograda 2.",
            "Poštovani slušaoci, slušate *Gozbu* Radio Beograda 2.",
        ],
    )
    def test_one_phrase_matches_every_spelling_the_original_listed_separately(self, variant):
        """`remove_intro.py` listed these five as five phrases. Normalisation collapses them."""
        assert drop_intro_cues((cue(1, variant),), [INTRO]) == ()

    def test_a_non_matching_cue_is_untouched(self):
        source = (cue(1, "Nema uvoda ovde."),)
        assert drop_intro_cues(source, [INTRO]) == source

    def test_no_phrases_means_no_change(self):
        source = cues("a", "b")
        assert drop_intro_cues(source, []) == source

    def test_the_phrase_file_ignores_blanks_and_comments(self, tmp_path):
        path = tmp_path / "phrases.txt"
        path.write_text(f"# the station ident\n\n{INTRO}\n\n", encoding="utf-8")
        assert load_intro_phrases(path) == (INTRO,)

    def test_it_is_off_unless_a_file_is_given(self):
        fixed, report = fix_cues(
            (cue(1, INTRO), cue(2, "a")),
            FixConfig(),
            complete=echo,
            log=lambda _m: None,
        )
        assert len(fixed) == 2 and report.dropped == 0

    def test_the_pass_applies_it_when_a_file_is_given(self, tmp_path):
        path = tmp_path / "phrases.txt"
        path.write_text(INTRO + "\n", encoding="utf-8")
        fixed, report = fix_cues(
            (cue(1, INTRO), cue(2, "a")),
            FixConfig(drop_intro_phrases=path),
            complete=echo,
            log=lambda _m: None,
        )
        assert [c.text for c in fixed] == ["a"] and report.dropped == 1


# --------------------------------------------------------------------------------------
# Re-layout
# --------------------------------------------------------------------------------------


class TestRelayout:
    def test_a_correction_is_re_wrapped_to_the_line_budget(self):
        long = "Ovo je znatno duza recenica koja mora da se prelomi u dva reda"
        fixed, _ = fix_cues(
            cues("kratko"),
            FixConfig(),
            cue_config=CueConfig(max_line=32),
            complete=rewriting({1: long}),
            log=lambda _m: None,
        )
        assert len(fixed[0].lines) == 2
        assert all(len(line) <= 32 for line in fixed[0].lines)

    def test_a_correction_that_grows_past_the_line_budget_loses_no_words(self):
        """Regression: `wrap_text` truncates to max_lines, which would delete the tail.

        Unreachable in the normal pipeline because the splitter guarantees a chunk fits,
        and very reachable here, where the model decides how long the text is.
        """
        long = " ".join(f"rec{i}" for i in range(40))
        fixed, _ = fix_cues(
            cues("kratko"),
            FixConfig(),
            cue_config=CueConfig(max_line=20, max_lines=2),
            complete=rewriting({1: long}),
            log=lambda _m: None,
        )
        assert " ".join(fixed[0].lines) == long

    def test_an_unchanged_text_keeps_the_splitters_original_line_break(self):
        """Regression, found on the real Serbian fixture against gpt-4o.

        Eight cues came back with their text byte-identical and their line break moved,
        because every cue was re-wrapped unconditionally. The splitter's break used real
        word timings that cannot be recovered here, so re-deriving it can only be worse.
        """
        original = (
            Cue(
                index=1,
                start=20.9,
                end=26.9,
                lines=("Propisuje se kao put saznanja s jedne", "strane iskustvo i posmatranje,"),
            ),
        )
        fixed, report = fix_cues(original, FixConfig(), complete=echo, log=lambda _m: None)
        assert fixed == original
        assert report.changed == 0

    def test_a_re_wrap_obeys_the_clitic_rule(self):
        """Regression, found on the real Serbian fixture against gpt-4o.

        Wrapping the corrected text with the plain greedy `wrap_text` produced

            Da se opšta predstava, da / se ono što je istinito,

        stranding the clitic "se" at the start of line two. The correction goes through
        `cues.wrap_words` over synthesized word timings so the clitic and preposition rules
        still apply.
        """
        source = (Cue(index=1, start=11.6, end=16.1, lines=("x",)),)
        text = "Da se opšta predstava, da se ono što je istinito,"
        fixed, _ = fix_cues(
            source,
            FixConfig(),
            cue_config=CueConfig(max_line=42),
            complete=rewriting({1: text}),
            log=lambda _m: None,
        )
        assert " ".join(fixed[0].lines) == text
        assert not fixed[0].lines[-1].startswith("se ")

    def test_a_re_wrap_does_not_strand_a_preposition(self):
        source = (Cue(index=1, start=0.0, end=5.0, lines=("x",)),)
        text = "Propisuje se kao put saznanja s jedne strane iskustvo,"
        fixed, _ = fix_cues(
            source,
            FixConfig(),
            cue_config=CueConfig(max_line=42),
            complete=rewriting({1: text}),
            log=lambda _m: None,
        )
        assert " ".join(fixed[0].lines) == text
        assert not fixed[0].lines[0].endswith(" s")

    def test_html_tags_do_not_count_towards_the_line_budget(self):
        """Regression, found on the real gozba fixture.

        `<i>` and `</i>` are seven characters of nothing on a web player, which is the only
        place `--fix-markup html` output goes. Measuring them against `max_line` pushed a
        two-line cue onto three lines for emphasis that occupies no width. The break is
        chosen on the markup-free text and applied to the marked-up one.
        """
        text = "koja je napisao u svojoj *Istoriji filozofije*, zapocinjemo danasnju emisiju."
        source = (Cue(index=1, start=48.7, end=53.2, lines=("x",)),)
        html, _ = fix_cues(
            source,
            FixConfig(markup="html"),
            cue_config=CueConfig(max_line=42),
            complete=rewriting({1: text}),
            log=lambda _m: None,
        )
        plain, _ = fix_cues(
            source,
            FixConfig(markup="strip"),
            cue_config=CueConfig(max_line=42),
            complete=rewriting({1: text}),
            log=lambda _m: None,
        )
        assert len(html[0].lines) == len(plain[0].lines) == 2
        assert "<i>Istoriji" in html[0].text and "filozofije</i>," in html[0].text

    def test_lint_does_not_count_markup_as_width(self):
        """Regression, found on the real gozba fixture: three false violations.

        `lint` measured `<b>Johna Lockea</b>` as 45 characters on a line whose reader sees
        38, so the opt-in html path exited non-zero on cues that were within every limit.
        """
        from subtitler.cues import lint_cues

        marked = Cue(
            index=1,
            start=75.8,
            end=78.1,
            lines=("Razgovaramo o filozofiji <b>Johna Lockea</b>,",),
        )
        assert len(marked.lines[0]) > 42
        assert lint_cues((marked,), CueConfig()) == []

    def test_lint_still_catches_a_genuinely_long_line(self):
        from subtitler.cues import lint_cues

        long = Cue(index=1, start=0.0, end=5.0, lines=("x" * 43,))
        assert any("chars (max 42)" in p for p in lint_cues((long,), CueConfig()))

    def test_a_markup_span_may_cross_the_line_break(self):
        """Words per line are preserved, so an emphasis that spans the break survives it."""
        text = "prva rec *druga rec treca rec cetvrta rec* peta rec i sesta rec ovde"
        fixed, _ = fix_cues(
            (Cue(index=1, start=0.0, end=5.0, lines=("x",)),),
            FixConfig(markup="html"),
            cue_config=CueConfig(max_line=34),
            complete=rewriting({1: text}),
            log=lambda _m: None,
        )
        joined = " ".join(fixed[0].lines)
        assert joined.count("<i>") == 1 and joined.count("</i>") == 1
        assert "*" not in joined

    def test_an_unchanged_cue_is_not_counted_as_changed(self):
        _, report = fix_cues(cues("a", "b"), FixConfig(), complete=echo, log=lambda _m: None)
        assert report.changed == 0
