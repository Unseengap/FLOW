"""Morphology Map — Word family clusters with systematic fiber offsets.

Maps word families ("run"/"running"/"ran"/"runner") as geometric clusters
with systematic fiber offsets.  Each inflection form is a small geometric
displacement from the base form, encoded in the causal and logical fibers.

Inflection geometry
-------------------
  Base form   → position P on M(t)
  Past tense  → P + δ_past  (shift along τ-axis in causal fiber)
  Progressive → P + δ_prog  (shift toward continuity in probabilistic fiber)
  Plural      → P + δ_plural (shift in logical fiber — quantification)
  Comparative → P + δ_comp  (shift in similarity fiber — degree)
  Negative    → P + δ_neg   (flip in logical fiber dim 80)

The MorphologyMap does not store inflected forms explicitly — it
computes them on demand from the base form's geometry and suffix rules.

No weights.  No tokens.  No training.  Morphology IS geometric displacement.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np

# ── Re-use dimension slices from syntax_geometry ──────────────────────────
from .syntax_geometry import (
    _SIM_START, _SIM_END,
    _CAUSAL_START, _CAUSAL_END,
    _LOGICAL_START, _LOGICAL_END,
    _PROB_START, _PROB_END,
)


class Inflection(Enum):
    """Inflection types with their geometric displacement directions."""
    BASE = "base"
    PAST = "past"               # -ed, irregular past
    PROGRESSIVE = "progressive" # -ing
    PERFECT = "perfect"         # has + past participle
    PLURAL = "plural"           # -s/-es
    SINGULAR = "singular"       # base noun form
    COMPARATIVE = "comparative" # -er / more
    SUPERLATIVE = "superlative" # -est / most
    NEGATIVE = "negative"       # un-/in-/dis-/not
    GERUND = "gerund"           # -ing (nominal use)
    THIRD_PERSON = "third_person"  # -s (verb)
    POSSESSIVE = "possessive"   # 's


@dataclass
class WordForm:
    """An inflected word form with its geometric relationship to the base.

    Attributes
    ----------
    surface     : the surface string ("running", "ran", "runs")
    base        : the base/lemma form ("run")
    inflection  : the inflection type
    offset      : 104D geometric offset from base position
    confidence  : confidence in the morphological analysis
    """
    surface: str
    base: str
    inflection: Inflection
    offset: np.ndarray
    confidence: float = 0.8

    def __repr__(self) -> str:
        return (
            f"WordForm('{self.surface}', base='{self.base}', "
            f"{self.inflection.value}, conf={self.confidence:.2f})"
        )


class MorphologyMap:
    """Maps word families via geometric offsets in the 104D fiber bundle.

    The map uses suffix patterns to identify inflection types and computes
    systematic geometric offsets.  These offsets encode tense, number, and
    degree as small displacements in the appropriate fibers.

    Parameters
    ----------
    dim : int
        Manifold dimensionality (default 104).
    """

    def __init__(self, dim: int = 104) -> None:
        self.dim = dim
        self._offset_cache: Dict[Inflection, np.ndarray] = {}
        self._build_offset_table()

        # Irregular verb forms (common ones)
        self._irregular_past: Dict[str, str] = {
            "be": "was", "is": "was", "are": "were", "am": "was",
            "have": "had", "has": "had", "do": "did", "does": "did",
            "go": "went", "come": "came", "make": "made", "take": "took",
            "give": "gave", "get": "got", "know": "knew", "think": "thought",
            "see": "saw", "say": "said", "run": "ran", "find": "found",
            "tell": "told", "feel": "felt", "become": "became",
            "leave": "left", "put": "put", "keep": "kept",
            "begin": "began", "show": "showed", "hear": "heard",
            "write": "wrote", "stand": "stood", "lose": "lost",
            "bring": "brought", "hold": "held", "lead": "led",
            "read": "read", "grow": "grew", "draw": "drew",
            "set": "set", "cut": "cut", "hit": "hit",
            "drive": "drove", "rise": "rose", "fall": "fell",
            "break": "broke", "build": "built", "send": "sent",
            "spend": "spent", "mean": "meant", "catch": "caught",
            "teach": "taught", "buy": "bought", "fight": "fought",
            "wear": "wore", "choose": "chose", "speak": "spoke",
            "eat": "ate", "sit": "sat", "lie": "lay",
            "sing": "sang", "win": "won", "fly": "flew",
            "throw": "threw", "sleep": "slept", "pay": "paid",
        }

        # Reverse mapping: past → base
        self._past_to_base: Dict[str, str] = {v: k for k, v in self._irregular_past.items()}

        # Irregular plurals
        self._irregular_plural: Dict[str, str] = {
            "child": "children", "person": "people", "man": "men",
            "woman": "women", "mouse": "mice", "tooth": "teeth",
            "foot": "feet", "goose": "geese", "ox": "oxen",
            "fish": "fish", "sheep": "sheep", "deer": "deer",
            "series": "series", "species": "species",
            "datum": "data", "criterion": "criteria",
            "phenomenon": "phenomena", "analysis": "analyses",
            "thesis": "theses", "crisis": "crises",
            "hypothesis": "hypotheses",
        }

    # ── Public API ─────────────────────────────────────────────────── #

    def analyse(self, word: str) -> WordForm:
        """Analyse a word and return its base form + inflection type.

        Parameters
        ----------
        word : the surface form to analyse

        Returns
        -------
        WordForm with base, inflection, and geometric offset.
        """
        word_lower = word.lower().strip()

        # Check irregular past
        if word_lower in self._past_to_base:
            base = self._past_to_base[word_lower]
            return WordForm(
                surface=word_lower,
                base=base,
                inflection=Inflection.PAST,
                offset=self._offset_cache[Inflection.PAST],
                confidence=0.95,
            )

        # Check irregular plural
        for singular, plural in self._irregular_plural.items():
            if word_lower == plural and plural != singular:
                return WordForm(
                    surface=word_lower,
                    base=singular,
                    inflection=Inflection.PLURAL,
                    offset=self._offset_cache[Inflection.PLURAL],
                    confidence=0.95,
                )

        # Regular suffix analysis
        return self._analyse_regular(word_lower)

    def inflect(self, base: str, inflection: Inflection) -> str:
        """Generate an inflected surface form from a base word.

        Parameters
        ----------
        base       : the base/lemma form
        inflection : desired inflection type

        Returns
        -------
        The inflected surface string.
        """
        base_lower = base.lower().strip()

        if inflection == Inflection.BASE:
            return base_lower

        if inflection == Inflection.PAST:
            return self._make_past(base_lower)

        if inflection == Inflection.PROGRESSIVE or inflection == Inflection.GERUND:
            return self._make_progressive(base_lower)

        if inflection == Inflection.PLURAL:
            return self._make_plural(base_lower)

        if inflection == Inflection.THIRD_PERSON:
            return self._make_third_person(base_lower)

        if inflection == Inflection.COMPARATIVE:
            return self._make_comparative(base_lower)

        if inflection == Inflection.SUPERLATIVE:
            return self._make_superlative(base_lower)

        if inflection == Inflection.NEGATIVE:
            return self._make_negative(base_lower)

        if inflection == Inflection.POSSESSIVE:
            return self._make_possessive(base_lower)

        return base_lower

    def get_offset(self, inflection: Inflection) -> np.ndarray:
        """Return the 104D geometric offset for an inflection type."""
        return self._offset_cache.get(inflection, np.zeros(self.dim))

    def word_family(self, base: str) -> Dict[Inflection, str]:
        """Generate all common inflected forms of a base word.

        Returns a dictionary mapping inflection types to surface forms.
        """
        family = {Inflection.BASE: base}
        for infl in [
            Inflection.PAST, Inflection.PROGRESSIVE,
            Inflection.PLURAL, Inflection.THIRD_PERSON,
            Inflection.COMPARATIVE, Inflection.SUPERLATIVE,
            Inflection.NEGATIVE, Inflection.GERUND,
        ]:
            form = self.inflect(base, infl)
            if form != base:
                family[infl] = form
        return family

    # ── Geometric offset table ─────────────────────────────────────── #

    def _build_offset_table(self) -> None:
        """Build the geometric displacement vectors for each inflection.

        Offsets are small, structured displacements in specific fibers:
        - Tense → causal fiber (τ-axis shift)
        - Number → logical fiber (quantification dims)
        - Degree → similarity fiber (magnitude scaling)
        - Negation → logical fiber (dim 80 flip)
        """
        dim = self.dim

        # PAST: shift along τ (causal dim 0 = dim 64) toward earlier time
        past = np.zeros(dim)
        past[_CAUSAL_START] = -0.1  # earlier τ
        past[_CAUSAL_START + 1] = 0.05  # observational layer shift
        self._offset_cache[Inflection.PAST] = past

        # PROGRESSIVE: shift in probabilistic fiber (continuity)
        prog = np.zeros(dim)
        prog[_PROB_START:_PROB_START + 4] = 0.05  # continuity signal
        prog[_CAUSAL_START] = 0.02  # slight forward-looking τ
        self._offset_cache[Inflection.PROGRESSIVE] = prog
        self._offset_cache[Inflection.GERUND] = prog.copy()

        # PERFECT: past + completion signal
        perfect = np.zeros(dim)
        perfect[_CAUSAL_START] = -0.08  # past-oriented
        perfect[_PROB_START + 4:_PROB_START + 8] = 0.05  # completion signal
        self._offset_cache[Inflection.PERFECT] = perfect

        # PLURAL: quantification shift in logical fiber
        plural = np.zeros(dim)
        plural[_LOGICAL_START + 1] = 0.1  # universal quantifier direction
        plural[_LOGICAL_START + 2] = 0.05  # existential support
        self._offset_cache[Inflection.PLURAL] = plural

        # SINGULAR: neutral (base form for nouns)
        self._offset_cache[Inflection.SINGULAR] = np.zeros(dim)

        # THIRD_PERSON: verb agreement marker
        third = np.zeros(dim)
        third[_LOGICAL_START + 4] = 0.05  # agreement marker in logical fiber
        self._offset_cache[Inflection.THIRD_PERSON] = third

        # COMPARATIVE: degree shift in similarity fiber
        comp = np.zeros(dim)
        comp[_SIM_START + 32:_SIM_START + 36] = 0.06  # adjective quadrant boost
        self._offset_cache[Inflection.COMPARATIVE] = comp

        # SUPERLATIVE: stronger degree shift
        superl = np.zeros(dim)
        superl[_SIM_START + 32:_SIM_START + 36] = 0.12  # stronger boost
        self._offset_cache[Inflection.SUPERLATIVE] = superl

        # NEGATIVE: flip negation dimension in logical fiber
        neg = np.zeros(dim)
        neg[_LOGICAL_START] = -0.2  # negation dim flip
        neg[_LOGICAL_START + 3] = 0.1  # negative quantifier support
        self._offset_cache[Inflection.NEGATIVE] = neg

        # POSSESSIVE: slight relational shift
        poss = np.zeros(dim)
        poss[_CAUSAL_START + 2] = 0.05  # possessive relational signal
        self._offset_cache[Inflection.POSSESSIVE] = poss

        # BASE: zero offset
        self._offset_cache[Inflection.BASE] = np.zeros(dim)

    # ── Regular suffix analysis ────────────────────────────────────── #

    def _analyse_regular(self, word: str) -> WordForm:
        """Analyse regular inflections via suffix patterns."""

        # Progressive / Gerund: -ing
        if word.endswith("ing") and len(word) > 4:
            base = self._strip_ing(word)
            return WordForm(
                surface=word, base=base,
                inflection=Inflection.PROGRESSIVE,
                offset=self._offset_cache[Inflection.PROGRESSIVE],
                confidence=0.8,
            )

        # Past tense: -ed
        if word.endswith("ed") and len(word) > 3:
            base = self._strip_ed(word)
            return WordForm(
                surface=word, base=base,
                inflection=Inflection.PAST,
                offset=self._offset_cache[Inflection.PAST],
                confidence=0.75,
            )

        # Superlative: -est
        if word.endswith("est") and len(word) > 4:
            base = self._strip_est(word)
            return WordForm(
                surface=word, base=base,
                inflection=Inflection.SUPERLATIVE,
                offset=self._offset_cache[Inflection.SUPERLATIVE],
                confidence=0.7,
            )

        # Comparative: -er (careful: not all -er words are comparatives)
        if word.endswith("er") and len(word) > 3 and not word.endswith("ter"):
            base = self._strip_er(word)
            return WordForm(
                surface=word, base=base,
                inflection=Inflection.COMPARATIVE,
                offset=self._offset_cache[Inflection.COMPARATIVE],
                confidence=0.6,
            )

        # Plural: -s/-es (careful heuristic)
        if word.endswith("es") and len(word) > 3 and word[-3] in "sxzh":
            base = word[:-2]
            return WordForm(
                surface=word, base=base,
                inflection=Inflection.PLURAL,
                offset=self._offset_cache[Inflection.PLURAL],
                confidence=0.7,
            )
        if word.endswith("ies") and len(word) > 4:
            base = word[:-3] + "y"
            return WordForm(
                surface=word, base=base,
                inflection=Inflection.PLURAL,
                offset=self._offset_cache[Inflection.PLURAL],
                confidence=0.7,
            )
        if word.endswith("s") and not word.endswith("ss") and len(word) > 3:
            base = word[:-1]
            return WordForm(
                surface=word, base=base,
                inflection=Inflection.PLURAL,
                offset=self._offset_cache[Inflection.PLURAL],
                confidence=0.5,
            )

        # Base form
        return WordForm(
            surface=word, base=word,
            inflection=Inflection.BASE,
            offset=self._offset_cache[Inflection.BASE],
            confidence=0.9,
        )

    # ── Inflection generation ──────────────────────────────────────── #

    def _make_past(self, base: str) -> str:
        if base in self._irregular_past:
            return self._irregular_past[base]
        if base.endswith("e"):
            return base + "d"
        if base.endswith("y") and len(base) > 2 and base[-2] not in "aeiou":
            return base[:-1] + "ied"
        if (len(base) >= 3 and base[-1] not in "aeiouwxy"
                and base[-2] in "aeiou" and base[-3] not in "aeiou"):
            return base + base[-1] + "ed"
        return base + "ed"

    def _make_progressive(self, base: str) -> str:
        if base.endswith("ie"):
            return base[:-2] + "ying"
        if base.endswith("e") and not base.endswith("ee"):
            return base[:-1] + "ing"
        if (len(base) >= 3 and base[-1] not in "aeiouwxy"
                and base[-2] in "aeiou" and base[-3] not in "aeiou"):
            return base + base[-1] + "ing"
        return base + "ing"

    def _make_plural(self, base: str) -> str:
        if base in self._irregular_plural:
            return self._irregular_plural[base]
        if base.endswith(("s", "x", "z", "sh", "ch")):
            return base + "es"
        if base.endswith("y") and len(base) > 1 and base[-2] not in "aeiou":
            return base[:-1] + "ies"
        if base.endswith("f"):
            return base[:-1] + "ves"
        if base.endswith("fe"):
            return base[:-2] + "ves"
        return base + "s"

    def _make_third_person(self, base: str) -> str:
        if base.endswith(("s", "x", "z", "sh", "ch")):
            return base + "es"
        if base.endswith("y") and len(base) > 1 and base[-2] not in "aeiou":
            return base[:-1] + "ies"
        return base + "s"

    def _make_comparative(self, base: str) -> str:
        if len(base) <= 6:
            if base.endswith("e"):
                return base + "r"
            if base.endswith("y") and len(base) > 2 and base[-2] not in "aeiou":
                return base[:-1] + "ier"
            return base + "er"
        return "more " + base

    def _make_superlative(self, base: str) -> str:
        if len(base) <= 6:
            if base.endswith("e"):
                return base + "st"
            if base.endswith("y") and len(base) > 2 and base[-2] not in "aeiou":
                return base[:-1] + "iest"
            return base + "est"
        return "most " + base

    def _make_negative(self, base: str) -> str:
        # Common prefix negations
        if base.startswith(("im", "in", "ir", "il", "un", "dis", "non")):
            return base  # already negative
        if base[0] in "bmp":
            return "im" + base
        if base[0] == "l":
            return "il" + base
        if base[0] == "r":
            return "ir" + base
        return "un" + base

    def _make_possessive(self, base: str) -> str:
        if base.endswith("s"):
            return base + "'"
        return base + "'s"

    # ── Suffix stripping helpers ───────────────────────────────────── #

    def _strip_ing(self, word: str) -> str:
        stem = word[:-3]
        # running → run (doubled consonant)
        if len(stem) >= 2 and stem[-1] == stem[-2] and stem[-1] not in "aeiou":
            return stem[:-1]
        # making → make (dropped -e)
        if len(stem) >= 2 and stem[-1] not in "aeiou" and stem[-2] in "aeiou":
            with_e = stem + "e"
            return with_e
        return stem

    def _strip_ed(self, word: str) -> str:
        stem = word[:-2]
        if not stem:
            return word
        # stopped → stop (doubled consonant)
        if len(stem) >= 2 and stem[-1] == stem[-2] and stem[-1] not in "aeiou":
            return stem[:-1]
        # loved → love (dropped -e added back)
        if word.endswith("ed") and not word.endswith("eed"):
            with_e = stem + "e"
            return with_e
        return stem

    def _strip_er(self, word: str) -> str:
        stem = word[:-2]
        if not stem:
            return word
        if stem.endswith("i") and len(stem) > 1:
            return stem[:-1] + "y"
        return stem

    def _strip_est(self, word: str) -> str:
        stem = word[:-3]
        if not stem:
            return word
        if stem.endswith("i") and len(stem) > 1:
            return stem[:-1] + "y"
        return stem
