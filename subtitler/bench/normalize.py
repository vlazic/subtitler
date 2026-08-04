"""Serbian text normalization, applied identically to hypothesis and reference.

Scoring raw Whisper output against a reference measures typography as much as recognition.
A pipeline is therefore fixed here and used for every number in `metrics.py`:

    NFC  ->  Cyrillic to Latin  ->  lowercase  ->  strip punctuation  ->  collapse spaces

Three decisions worth defending:

**Cyrillic to Latin, not the other way.** Serbian is digraphic, and the two scripts are not
symmetric. Latin to Cyrillic is lossy in exactly the place that matters: `nj` in `injekcija`
is two letters, not `њ`, and no table can tell those apart without a lexicon. Cyrillic to
Latin is a pure one-to-one (or one-to-two) mapping with no ambiguity at all, so that is the
direction everything is folded into.

**A hand-written table, not a library.** It is 30 lines, it is auditable at a glance, and it
carries no dependency into a package that has almost none. `cyrtranslit` and friends also
carry Macedonian, Bulgarian and Russian tables that would silently change how an unexpected
character scores.

**Digits and abbreviations are left alone in v1.** "20" against "dvadeset" scores as a
substitution, and so does "npr." against "na primer". Both inflate WER, and both inflate it
identically for every engine in the matrix, so the ranking survives even though the absolute
number is pessimistic. Fixing it properly needs a Serbian number speller with case and
gender agreement, which is a project of its own. The report says so, out loud, next to the
numbers.

`fold_diacritics` is the second half of the story. Serbian Whisper output confuses two very
different things: hearing the wrong word, and writing `c` where `č` belongs. Folding the
diacritics away and rescoring separates them, and the gap between the two scores is the most
useful single number this harness produces for Serbian.
"""

from __future__ import annotations

import re
import unicodedata

__all__ = [
    "CYRILLIC_TO_LATIN",
    "DIACRITIC_FOLD",
    "fold_diacritics",
    "normalize",
    "to_latin",
    "tokens",
]

# The Serbian Cyrillic alphabet, in alphabetical order. Only the 30 letters Serbian uses:
# a character from another Cyrillic alphabet is deliberately left untouched so it shows up
# in a diff instead of being quietly mapped to something plausible.
CYRILLIC_TO_LATIN: dict[str, str] = {
    "а": "a",
    "б": "b",
    "в": "v",
    "г": "g",
    "д": "d",
    "ђ": "đ",
    "е": "e",
    "ж": "ž",
    "з": "z",
    "и": "i",
    "ј": "j",
    "к": "k",
    "л": "l",
    "љ": "lj",
    "м": "m",
    "н": "n",
    "њ": "nj",
    "о": "o",
    "п": "p",
    "р": "r",
    "с": "s",
    "т": "t",
    "ћ": "ć",
    "у": "u",
    "ф": "f",
    "х": "h",
    "ц": "c",
    "ч": "č",
    "џ": "dž",
    "ш": "š",
}

# Latin letters that carry a Serbian diacritic, and what they fold to. `đ` becomes `dj`
# rather than `d` because that is how it is actually typed on a keyboard without the
# character, which is the confusion being measured.
DIACRITIC_FOLD: dict[str, str] = {
    "č": "c",
    "ć": "c",
    "đ": "dj",
    "š": "s",
    "ž": "z",
}

_CYRILLIC_UPPER = {c.upper(): latin for c, latin in CYRILLIC_TO_LATIN.items()}

_WHITESPACE_RE = re.compile(r"\s+")


def _is_upper_cyrillic(ch: str) -> bool:
    return ch in _CYRILLIC_UPPER


def to_latin(text: str) -> str:
    """Serbian Cyrillic to Latin, preserving case, including for the digraphs.

    `Њ` becomes `Nj` in `Његош` and `NJ` in `ЊЕГОШ`. The rule is the next character: an
    uppercase Cyrillic letter after the digraph means the word is being shouted, anything
    else means it is a name. Everything that is not a Serbian Cyrillic letter passes
    through untouched, so a Latin input is already its own answer.
    """
    out: list[str] = []
    for i, ch in enumerate(text):
        lower = CYRILLIC_TO_LATIN.get(ch)
        if lower is not None:
            out.append(lower)
            continue
        upper = _CYRILLIC_UPPER.get(ch)
        if upper is None:
            out.append(ch)
            continue
        shouted = len(upper) > 1 and i + 1 < len(text) and _is_upper_cyrillic(text[i + 1])
        out.append(upper.upper() if len(upper) == 1 or shouted else upper.capitalize())
    return "".join(out)


def fold_diacritics(text: str) -> str:
    """`č ć` to `c`, `đ` to `dj`, `š` to `s`, `ž` to `z`, in both cases.

    Used only for WER_folded. Applied after `normalize`, which has already lowercased, but
    the uppercase forms are handled anyway so the function is usable on its own.
    """
    out: list[str] = []
    for ch in text:
        folded = DIACRITIC_FOLD.get(ch)
        if folded is not None:
            out.append(folded)
            continue
        upper = DIACRITIC_FOLD.get(ch.lower())
        out.append(upper.upper() if upper is not None else ch)
    return "".join(out)


def _strip_punctuation(text: str) -> str:
    """Every Unicode punctuation mark becomes a space, not nothing.

    Deleting instead would glue `reč,druga` into one token that matches nothing, and would
    turn `crno-beli` into a word neither side wrote. A space splits both the same way on
    both sides of the comparison, which is all a WER normalizer has to guarantee. This also
    covers the Serbian quotes (`„ “ ” » «`) and the ellipsis, which are the marks Whisper
    emits most and a human transcript almost never does.
    """
    return "".join(" " if unicodedata.category(ch).startswith("P") else ch for ch in text)


def normalize(text: str, *, fold: bool = False) -> str:
    """The full pipeline. `fold=True` additionally removes the Serbian diacritics.

    Order is not negotiable: the script conversion has to happen before lowercasing (the
    digraph case rule reads uppercase context), and punctuation has to go before the
    whitespace collapse (it is replaced by spaces, which then need collapsing).
    """
    text = unicodedata.normalize("NFC", text)
    text = to_latin(text)
    text = text.lower()
    text = _strip_punctuation(text)
    if fold:
        text = fold_diacritics(text)
    return _WHITESPACE_RE.sub(" ", text).strip()


def tokens(text: str, *, fold: bool = False) -> list[str]:
    """Normalized whitespace-separated tokens. Empty text gives an empty list."""
    normalized = normalize(text, fold=fold)
    return normalized.split() if normalized else []
