"""Serbian normalization.

Every WER in the benchmark is a function of this file, so it is tested at the character
level rather than through a score. A normalizer bug does not raise: it quietly moves a
number, and a leaderboard built on it ranks the wrong engine first.
"""

from __future__ import annotations

from subtitler.bench.normalize import (
    CYRILLIC_TO_LATIN,
    fold_diacritics,
    normalize,
    to_latin,
    tokens,
)


class TestToLatin:
    def test_every_serbian_letter_has_a_mapping(self):
        """All 30, no more. A Russian or Macedonian letter must not be silently mapped."""
        assert len(CYRILLIC_TO_LATIN) == 30
        assert "ы" not in CYRILLIC_TO_LATIN
        assert "ѕ" not in CYRILLIC_TO_LATIN

    def test_plain_word(self):
        assert to_latin("реченица") == "rečenica"
        assert to_latin("добар дан") == "dobar dan"

    def test_diacritics(self):
        assert to_latin("Ђорђе шешељ жаба ћирилица чаша") == "Đorđe šešelj žaba ćirilica čaša"

    def test_digraphs_lowercase(self):
        assert to_latin("љубав") == "ljubav"
        assert to_latin("њива") == "njiva"
        assert to_latin("џак") == "džak"

    def test_digraph_title_case(self):
        """A capitalized name keeps the second letter lowercase: Његош, not NJegoš."""
        assert to_latin("Његош") == "Njegoš"
        assert to_latin("Љубав") == "Ljubav"
        assert to_latin("Џак") == "Džak"

    def test_digraph_in_shouted_text(self):
        """Uppercase context makes the whole digraph uppercase."""
        assert to_latin("ЊЕГОШ") == "NJEGOŠ"
        assert to_latin("ЉУБАВ") == "LJUBAV"

    def test_latin_input_passes_through(self):
        assert to_latin("već napisano latinicom") == "već napisano latinicom"

    def test_non_serbian_characters_are_untouched(self):
        assert to_latin("тест 42 (x) ы") == "test 42 (x) ы"

    def test_mixed_scripts_in_one_string(self):
        assert to_latin("ОК, идемо") == "OK, idemo"


class TestFold:
    def test_folds_every_diacritic(self):
        assert fold_diacritics("čćđšž") == "ccdjsz"

    def test_folds_uppercase_too(self):
        assert fold_diacritics("ČĆĐŠŽ") == "CCDJSZ"

    def test_leaves_everything_else(self):
        assert fold_diacritics("dobar dan 42") == "dobar dan 42"


class TestNormalize:
    def test_the_full_pipeline(self):
        assert normalize("Добар Дан, свете!") == "dobar dan svete"

    def test_serbian_quotes_are_stripped(self):
        assert normalize("„Реч“ и »друга«") == "reč i druga"

    def test_ellipsis_and_dashes(self):
        assert normalize("Па... то је — тако") == "pa to je tako"

    def test_punctuation_becomes_a_space_not_nothing(self):
        """Deleting would glue two words into one that matches neither side."""
        assert normalize("reč,druga") == "reč druga"
        assert normalize("crno-beli") == "crno beli"

    def test_whitespace_is_collapsed(self):
        assert normalize("  dva \n\t razmaka  ") == "dva razmaka"

    def test_nfc_makes_decomposed_input_identical(self):
        """`č` typed as c + combining caron must score as the same word."""
        composed = "reč"
        decomposed = "reč"
        assert composed != decomposed
        assert normalize(composed) == normalize(decomposed)

    def test_both_scripts_normalize_to_the_same_string(self):
        assert normalize("Ђорђе је дошао.") == normalize("Đorđe je došao.")

    def test_is_idempotent(self):
        once = normalize("Његош, „песник“!")
        assert normalize(once) == once

    def test_fold_changes_only_the_diacritics(self):
        assert normalize("Чаша", fold=True) == "casa"
        assert normalize("Ђорђе", fold=True) == "djordje"
        assert normalize("Чаша") == "čaša"

    def test_digits_are_left_alone(self):
        """Documented v1 behaviour: `20` does not become `dvadeset` and inflates WER."""
        assert normalize("имао је 20 година") == "imao je 20 godina"

    def test_empty_input(self):
        assert normalize("") == ""
        assert normalize("   ...  ") == ""


class TestTokens:
    def test_splits_normalized_text(self):
        assert tokens("Добар дан, свете!") == ["dobar", "dan", "svete"]

    def test_empty_text_is_no_tokens(self):
        assert tokens("") == []
        assert tokens("!!!") == []

    def test_fold_is_passed_through(self):
        assert tokens("Чаша", fold=True) == ["casa"]
