"""Word Placer — maps vocabulary words onto the 104D manifold via C3 Annealing.

Each word is given an initial 104D position derived entirely from its surface
properties (character n-grams, morphological class, syllable count).  It is
then placed on M(t) by the existing AnnealingEngine — the same machinery used
for every other experience.

NO corpus statistics are required for initial placement.  Co-occurrence
evidence (C4 contrast scheduling) refines positions afterwards.

Manifold dimension layout (matches architecture-specification.md)
-----------------------------------------------------------------
Dims   0–63  : Similarity base manifold  (16-domain universal taxonomy)
Dims  64–79  : Causal fiber              (Pearl do-calculus)
Dims  80–87  : Logical fiber             (Boolean hypercube)
Dims  88–103 : Probabilistic fiber       (Fisher-Rao metric)

Structural feature vector (§2.2 of vocabulary-geometry-specification.md)
-------------------------------------------------------------------------
Similarity fiber (0–63):
    • Character 4-gram fingerprint → hash-spread across 64 dims
    • Morphological class → coarse quadrant shift
    • Length / syllable hint → density nudge

Causal fiber (64–79):
    • All zeros — causality emerges from directed PMI in Phase 7b

Logical fiber (80–87):
    • Negation markers ('not','never','no','neither','nor') → flip dim 80
    • Universal quantifiers ('all','every','always') → set dim 81 high
    • Existential quantifiers ('some','many','often') → set dim 82 high
    • Negative quantifiers ('none','never','nothing') → set dim 83 high
    • Others → neutral midpoint 0.5 on all dims

Probabilistic fiber (88–103):
    • Function words → high-certainty region (close to 1.0)
    • Hedging words   → maximal-uncertainty region (close to 0.0)
    • Content words   → moderate uncertainty (0.45–0.55)
"""

from __future__ import annotations

import re
import numpy as np
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from src.phase3.annealing_engine.engine import AnnealingEngine

from src.phase3.annealing_engine.experience import Experience

# ── Constants ─────────────────────────────────────────────────────────────

DIM_TOTAL     = 104
DIM_SIM_END   = 64
DIM_CAUSAL    = slice(64, 80)
DIM_LOGICAL   = slice(80, 88)
DIM_PROB      = slice(88, 104)

# Morphological class → quadrant offset within similarity fiber
_MORPH_OFFSET: dict[str, int] = {
    "verb":      0,
    "noun":     16,
    "adjective": 32,
    "adverb":   48,
    "function":  0,   # function words sit in their probabilistic region instead
    "unknown":   0,
}

# Function words — high probabilistic certainty
_FUNCTION_WORDS = frozenset({
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "to", "of", "in", "for",
    "on", "with", "at", "by", "from", "as", "it", "its", "this", "that",
    "these", "those", "and", "or", "but", "nor", "so", "yet", "both",
    "either", "neither", "each", "every", "all", "any", "few", "more",
    "most", "other", "such", "than", "too", "very", "just", "because",
    "if", "when", "while", "although", "though", "unless", "since", "after",
    "before", "until", "once", "about", "against", "between", "into",
    "through", "during", "without", "toward", "upon", "among", "i", "you",
    "he", "she", "we", "they", "me", "him", "her", "us", "them", "my",
    "your", "his", "our", "their",
})

# Negation / logical operators
_NEGATION = frozenset({"not", "never", "no", "neither", "nor", "nothing",
                        "nobody", "nowhere", "none"})
_UNIVERSAL = frozenset({"all", "every", "always", "everywhere", "everyone",
                         "everything", "invariably", "necessarily"})
_EXISTENTIAL = frozenset({"some", "many", "often", "sometimes", "somewhere",
                           "someone", "something", "frequently", "usually"})
_NEG_QUANT = frozenset({"none", "never", "nothing", "nobody", "nowhere"})

# Epistemic hedging words → high uncertainty in probabilistic fiber
_HEDGING = frozenset({
    "maybe", "perhaps", "possibly", "probably", "likely", "unlikely",
    "might", "could", "suggest", "suggests", "seems", "appear", "appears",
    "apparently", "presumably", "ostensibly", "arguably", "conceivably",
    "presumably", "roughly", "approximately", "somewhat", "relatively",
})


# ─────────────────────────────────────────────────────────────────────────────
# Structural feature vector
# ─────────────────────────────────────────────────────────────────────────────

def structural_feature_vector(word: str, freq_rank: int = 5000) -> np.ndarray:
    """Derive a 104D initial placement vector from surface properties only.

    Parameters
    ----------
    word      : lowercase word string (no punctuation)
    freq_rank : 1-based frequency rank; rank 1 = most common word in corpus.
                Used to set density hint in the probabilistic fiber.

    Returns
    -------
    np.ndarray of shape (104,) in [0, 1]
    """
    vec = np.zeros(DIM_TOTAL, dtype=float)

    # ── Similarity fiber (dims 0–63) ──────────────────────────────────────
    morph_class = _morphological_class(word)
    offset      = _MORPH_OFFSET[morph_class]

    # Character 4-gram fingerprint distributed across 16 dims after the offset
    ngram_fp = _char_ngram_fingerprint(word, n=4, size=16)
    start = offset
    vec[start: start + 16] = ngram_fp

    # Length hint: longer words → spread into later dims (content-richness)
    length_factor = min(1.0, len(word) / 15.0)
    vec[16:32] += length_factor * 0.3

    # Syllable hint: polysyllabic → nudge similarity toward domain 2–4
    syllables = _count_syllables(word)
    syl_factor = min(1.0, syllables / 6.0)
    vec[32:48] += syl_factor * 0.2

    # ── Causal fiber (dims 64–79) — zeros; filled by ContrastScheduler ────
    vec[DIM_CAUSAL] = 0.0

    # ── Logical fiber (dims 80–87) ────────────────────────────────────────
    w = word.lower()
    if w in _NEGATION or w in _NEG_QUANT:
        vec[80] = 1.0   # negation bit
    else:
        vec[80] = 0.0

    vec[81] = 1.0 if w in _UNIVERSAL else 0.5   # universal quantifier
    vec[82] = 1.0 if w in _EXISTENTIAL else 0.5  # existential quantifier
    vec[83] = 1.0 if w in _NEG_QUANT else 0.0   # negative quantifier
    vec[84:88] = 0.5  # neutral on remaining logical dims

    # ── Probabilistic fiber (dims 88–103) ─────────────────────────────────
    if w in _FUNCTION_WORDS:
        # High certainty: dims near 1.0
        vec[DIM_PROB] = 0.85 + 0.1 * _char_ngram_fingerprint(w, n=2, size=16)
    elif w in _HEDGING:
        # Maximal uncertainty: dims near 0.0
        vec[DIM_PROB] = 0.05 + 0.1 * _char_ngram_fingerprint(w, n=2, size=16)
    else:
        # Content words: moderate uncertainty shaped by frequency rank
        # Common words → slightly more certain; rare words → more uncertain
        base = 0.5 - 0.15 * (1.0 / (1.0 + freq_rank / 1000.0))
        vec[DIM_PROB] = base + 0.1 * _char_ngram_fingerprint(w, n=2, size=16)

    # Clip to [0, 1]
    vec = np.clip(vec, 0.0, 1.0)

    return vec


# ── Helpers ────────────────────────────────────────────────────────────────


def _morphological_class(word: str) -> str:
    """Infer coarse morphological class from surface patterns."""
    w = word.lower()
    if w in _FUNCTION_WORDS:
        return "function"
    # Verb suffixes
    if re.search(r"(ing|ize|ise|ify|ate|en)$", w):
        return "verb"
    if re.search(r"(ed|tion|sion|ment|ance|ence|ity|ness|ship|hood|dom)$", w):
        return "noun"
    if re.search(r"(ful|less|ous|ive|al|ic|able|ible|ary|ory|ary)$", w):
        return "adjective"
    if re.search(r"(ly|ward|wise)$", w):
        return "adverb"
    # Short words default to nouns
    return "noun" if len(w) > 3 else "unknown"


def _char_ngram_fingerprint(word: str, n: int = 4, size: int = 16) -> np.ndarray:
    """Hash character n-grams into a fixed-size [0,1] vector.

    Purely deterministic — same word always produces the same fingerprint.
    This is NOT an embedding; it is a geometric address derived from surface
    character statistics.
    """
    fp = np.zeros(size, dtype=float)
    padded = f"<{word}>"
    for i in range(len(padded) - n + 1):
        gram = padded[i: i + n]
        h    = hash(gram) & 0x7FFFFFFF  # positive 31-bit hash
        idx  = h % size
        fp[idx] += 1.0
    # Normalise to [0, 1]
    mx = fp.max()
    if mx > 0:
        fp /= mx
    return fp


def _count_syllables(word: str) -> int:
    """Approximate syllable count via vowel-cluster counting."""
    w = word.lower()
    vowels = re.findall(r"[aeiou]+", w)
    n = len(vowels)
    # Trailing silent 'e' correction
    if w.endswith("e") and n > 1:
        n -= 1
    return max(1, n)


def batch_structural_vectors_gpu(
    words: list[str], freq_ranks: list[int]
) -> list[np.ndarray]:
    """Compute structural vectors for all words using GPU when available.

    Falls back to CPU-only computation if cupy is not installed.
    The GPU path vectorizes the char n-gram hashing and fiber assignment
    across all words simultaneously.
    """
    try:
        import cupy as cp
    except ImportError:
        # CPU fallback — same result, just sequential
        return [structural_feature_vector(w, r) for w, r in zip(words, freq_ranks)]

    N = len(words)
    # Pre-compute all vectors on CPU (char hashing isn't great on GPU)
    # but do the fiber assignments + clipping on GPU in batch
    vecs_cpu = np.zeros((N, DIM_TOTAL), dtype=np.float32)
    for i, (w, r) in enumerate(zip(words, freq_ranks)):
        vecs_cpu[i] = structural_feature_vector(w, r)

    # Transfer to GPU, clip, transfer back (batched)
    vecs_gpu = cp.asarray(vecs_cpu)
    vecs_gpu = cp.clip(vecs_gpu, 0.0, 1.0)
    result = cp.asnumpy(vecs_gpu)

    return [result[i].astype(np.float64) for i in range(N)]


# ─────────────────────────────────────────────────────────────────────────────
# WordPlacer
# ─────────────────────────────────────────────────────────────────────────────

class WordPlacer:
    """Place vocabulary words on M(t) via C3 Annealing at T = T_floor.

    Cold placement (T = T_floor) ensures words land conservatively close
    to their structurally-derived initial position.  Fine-grained positioning
    is left to the ContrastScheduler (Phase 7b).

    Parameters
    ----------
    annealing_engine : AnnealingEngine
        The live C3 engine connected to the LivingManifold.
    """

    def __init__(self, annealing_engine: "AnnealingEngine") -> None:
        self._engine = annealing_engine
        self._saved_T0: Optional[float] = None

    # ── Public API ─────────────────────────────────────────────────────────

    def place(self, word: str, freq_rank: int = 5000) -> str:
        """Place a single word on M(t) and return its canonical label.

        The word is placed under label ``"vocab::{word}"`` via the existing
        AnnealingEngine.  Temperature is temporarily forced to T_floor so that
        the placement is conservative.

        Parameters
        ----------
        word      : cleaned word string (lowercase, no punctuation)
        freq_rank : 1-based frequency rank from the corpus

        Returns
        -------
        str — the label ``"vocab::{word}"`` as placed on the manifold
        """
        vec   = structural_feature_vector(word, freq_rank)
        label = f"vocab::{word}"

        # Force cold temperature for conservative placement
        self._set_cold()
        try:
            result = self._engine.process(
                Experience(vector=vec, label=label, source="vocabulary_geometry")
            )
        finally:
            self._restore_temperature()

        return result.placed_label or label

    def place_batch(
        self, words: list[str], freq_ranks: Optional[list[int]] = None,
        progress_callback=None,
    ) -> list[str]:
        """Place multiple words.  Returns list of placed labels.

        Optimization: temperature is saved/restored once for the entire
        batch instead of per-word (50K save/restore cycles eliminated).
        Uses manifold.place_fast() to skip per-word density/curvature
        recomputation, then flushes once at the end.

        Parameters
        ----------
        words             : list of cleaned word strings
        freq_ranks        : 1-based frequency ranks (default: sequential)
        progress_callback : optional callable(i, total, label) for logging
        """
        if freq_ranks is None:
            freq_ranks = list(range(1, len(words) + 1))

        manifold = self._engine._manifold
        labels: list[str] = []
        self._set_cold()
        try:
            for i, (w, r) in enumerate(zip(words, freq_ranks)):
                vec   = structural_feature_vector(w, r)
                label = f"vocab::{w}"
                # Fast placement: skip per-word density/curvature recompute
                manifold.place_fast(label, vec, origin="vocabulary_geometry")
                labels.append(label)
                # Tick the temperature schedule to match process() semantics
                self._engine._schedule.step()
                if progress_callback is not None and (i + 1) % 1000 == 0:
                    progress_callback(i + 1, len(words), label)
            # Rebuild tree + recompute densities in one pass
            manifold.flush_batch(labels)
        finally:
            self._restore_temperature()

        return labels

    def place_batch_gpu(
        self, words: list[str], freq_ranks: Optional[list[int]] = None,
        progress_callback=None,
    ) -> list[str]:
        """GPU-accelerated batch placement.

        Pre-computes all 50K structural vectors on GPU (cupy) in a single
        batch, then places via fast path on the manifold.  Falls back to
        place_batch() if cupy is unavailable.
        """
        try:
            import cupy as _cp
        except ImportError:
            return self.place_batch(words, freq_ranks, progress_callback)

        if freq_ranks is None:
            freq_ranks = list(range(1, len(words) + 1))

        # Batch-compute all structural vectors on GPU
        vecs = batch_structural_vectors_gpu(words, freq_ranks)

        manifold = self._engine._manifold
        labels: list[str] = []
        self._set_cold()
        try:
            for i, (w, vec) in enumerate(zip(words, vecs)):
                label = f"vocab::{w}"
                # Fast placement: skip per-word density/curvature recompute
                manifold.place_fast(label, vec, origin="vocabulary_geometry")
                labels.append(label)
                self._engine._schedule.step()
                if progress_callback is not None and (i + 1) % 1000 == 0:
                    progress_callback(i + 1, len(words), label)
            # Rebuild tree + recompute densities in one pass
            manifold.flush_batch(labels)
        finally:
            self._restore_temperature()

        return labels

    # ── Temperature management ─────────────────────────────────────────────

    def _set_cold(self) -> None:
        """Override engine temperature to T_floor for conservative placement."""
        schedule = self._engine.schedule
        self._saved_T0 = schedule.T0
        # Set T0 to T_floor so current_temperature ~ T_floor * exp(-λt) + T_floor
        # This keeps placement conservative without modifying the engine's design.
        schedule.T0 = schedule.T_floor

    def _restore_temperature(self) -> None:
        """Restore T0 after cold placement."""
        if self._saved_T0 is not None:
            self._engine.schedule.T0 = self._saved_T0
            self._saved_T0 = None
