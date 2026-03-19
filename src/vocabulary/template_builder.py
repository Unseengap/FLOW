"""Template Builder — constructs ExpressionEntry objects from manifold word positions.

Generates three levels of vocabulary entries (§4.4 of the spec):

  Level 1 — Word entries     (~50,000): single vocabulary words
  Level 2 — Phrase entries   (~35,000): nearby-word combinations
  Level 3 — Sentence frames  (~15,000): full frames with {} slots

Each entry's wave_profile is the weighted centroid of its constituent
word positions on M(t) — formula §2.4:

    Ψ_T = Σᵢ ωᵢ · P(wᵢ)  /  ‖Σᵢ ωᵢ · P(wᵢ)‖

Where:
    ωᵢ = 1 − ρ(P(wᵢ))   (low-density words contribute more)

Structural metadata derivation (§2.5):
    register         : mean probabilistic fiber norm
    rhythm           : word count: ≤4→short, 5–9→medium, ≥10→long
    uncertainty_fit  : mean amplitude at probabilistic fiber dims 88–103
    causal_strength  : mean signed component on causal dims 64–79
    hedging          : neighbour of uncertainty/possibility seed concepts
"""

from __future__ import annotations

import re
import numpy as np
from typing import List, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from src.phase2.living_manifold.manifold import LivingManifold

from src.phase1.expression.matcher import ExpressionEntry
from src.phase1.expression.wave import WAVE_DIM
from src.vocabulary.cooccurrence import CoOccurrenceMatrix

# ── Optional GPU acceleration via cupy ────────────────────────────────────────
try:
    import cupy as cp
    _HAS_CUPY = True
except ImportError:
    _HAS_CUPY = False

# ── Constants ─────────────────────────────────────────────────────────────────

DIM_CAUSAL = slice(64, 80)   # causal fiber
DIM_LOGICAL = slice(80, 88)  # logical fiber
DIM_PROB = slice(88, 104)    # probabilistic fiber

# Seed concept labels that mark "uncertainty" and "possibility" regions
_UNCERTAINTY_SEEDS = frozenset({
    "prob::maximal_uncertainty",
    "prob::high_uncertainty",
    "causal::correlation",
    "uncertainty",
    "possibility",
})

# Minimum geodesic radius for phrase combination (calibrated in Phase 7c)
# This is overridden by TemplateBuilder.calibrate_phrase_radius() — see §9.4
_DEFAULT_PHRASE_RADIUS = 0.35

# Sentence frame skeletons for Level 3 generation
# {} slots are filled by cluster centroids found on the manifold
_SENTENCE_FRAMES: List[Tuple[str, dict]] = [
    # Causal frames
    ("The {} leads to {}.",
     {"causal_strength": 0.85, "logical_strength": 0.5, "rhythm": "medium",
      "uncertainty_fit": 0.2, "register": "formal"}),
    ("When {} occurs, {} follows.",
     {"causal_strength": 0.9, "logical_strength": 0.6, "rhythm": "medium",
      "uncertainty_fit": 0.2, "register": "neutral"}),
    ("The {} enables {} through {}.",
     {"causal_strength": 0.8, "logical_strength": 0.5, "rhythm": "long",
      "uncertainty_fit": 0.25, "register": "formal"}),
    ("As {} increases, {} tends to {}.",
     {"causal_strength": 0.85, "logical_strength": 0.55, "rhythm": "long",
      "uncertainty_fit": 0.3, "register": "neutral"}),
    ("The relationship between {} and {} shows that {}.",
     {"causal_strength": 0.7, "logical_strength": 0.6, "rhythm": "long",
      "uncertainty_fit": 0.35, "register": "formal"}),
    # Descriptive frames
    ("The {} is defined by its {}.",
     {"causal_strength": 0.2, "logical_strength": 0.6, "rhythm": "medium",
      "uncertainty_fit": 0.3, "register": "formal"}),
    ("A key property of {} is {}.",
     {"causal_strength": 0.2, "logical_strength": 0.65, "rhythm": "medium",
      "uncertainty_fit": 0.25, "register": "formal"}),
    ("The {} exhibits {} behaviour under {}.",
     {"causal_strength": 0.4, "logical_strength": 0.5, "rhythm": "long",
      "uncertainty_fit": 0.4, "register": "formal"}),
    # Uncertainty frames
    ("Evidence suggests that {} may {}.",
     {"causal_strength": 0.4, "logical_strength": 0.4, "rhythm": "medium",
      "uncertainty_fit": 0.75, "register": "formal", "hedging": True}),
    ("It remains unclear whether {} or {}.",
     {"causal_strength": 0.2, "logical_strength": 0.5, "rhythm": "medium",
      "uncertainty_fit": 0.85, "register": "neutral", "hedging": True}),
    ("The {} is consistent with {}.",
     {"causal_strength": 0.45, "logical_strength": 0.55, "rhythm": "medium",
      "uncertainty_fit": 0.6, "register": "formal", "hedging": True}),
    # Logical frames
    ("If {} then {} must follow from {}.",
     {"causal_strength": 0.5, "logical_strength": 0.9, "rhythm": "long",
      "uncertainty_fit": 0.15, "register": "formal"}),
    ("Both {} and {} depend on {}.",
     {"causal_strength": 0.3, "logical_strength": 0.8, "rhythm": "medium",
      "uncertainty_fit": 0.2, "register": "formal"}),
    ("The absence of {} implies {}.",
     {"causal_strength": 0.6, "logical_strength": 0.8, "rhythm": "medium",
      "uncertainty_fit": 0.2, "register": "formal"}),
    # Temporal / sequential frames
    ("Initially {}, then {} transforms into {}.",
     {"causal_strength": 0.75, "logical_strength": 0.6, "rhythm": "long",
      "uncertainty_fit": 0.25, "register": "neutral"}),
    ("The process begins with {} and culminates in {}.",
     {"causal_strength": 0.8, "logical_strength": 0.55, "rhythm": "long",
      "uncertainty_fit": 0.2, "register": "formal"}),
]


# ─────────────────────────────────────────────────────────────────────────────
# Wave profile composition
# ─────────────────────────────────────────────────────────────────────────────

def compose_wave_profile(
    manifold: "LivingManifold",
    labels: List[str],
) -> np.ndarray:
    """Compute the wave profile for a multi-word entry (§2.4).

    Ψ_T = Σᵢ ωᵢ · P(wᵢ)  /  ‖Σᵢ ωᵢ · P(wᵢ)‖
    ωᵢ  = 1 − ρ(P(wᵢ))  (low-density words contribute more)

    Words not on the manifold are skipped.
    """
    vecs    = []
    weights = []

    for label in labels:
        try:
            pos     = manifold.position(label)
            density = float(manifold.density(pos))
            w       = max(0.01, 1.0 - density)
            vecs.append(pos[:WAVE_DIM])
            weights.append(w)
        except (KeyError, ValueError):
            continue

    if not vecs:
        return np.ones(WAVE_DIM, dtype=float) / np.sqrt(WAVE_DIM)

    arr = np.stack(vecs)                         # (n, WAVE_DIM)
    wts = np.array(weights, dtype=float)
    wts /= wts.sum()
    centroid = wts @ arr                          # (WAVE_DIM,)

    norm = np.linalg.norm(centroid)
    if norm < 1e-12:
        return np.ones(WAVE_DIM, dtype=float) / np.sqrt(WAVE_DIM)
    return centroid / norm


# ─────────────────────────────────────────────────────────────────────────────
# Metadata derivation (§2.5)
# ─────────────────────────────────────────────────────────────────────────────

def _derive_register(manifold: "LivingManifold", label: str) -> str:
    """Derive register from mean probabilistic fiber norm."""
    try:
        pos  = manifold.position(label)
        norm = float(np.mean(pos[88:104]))
        if norm > 0.65:
            return "formal"
        if norm < 0.35:
            return "casual"
        return "neutral"
    except (KeyError, ValueError):
        return "neutral"


def _derive_uncertainty_fit(manifold: "LivingManifold", label: str) -> float:
    """Mean amplitude in probabilistic fiber dims 88–103."""
    try:
        pos = manifold.position(label)
        return float(np.mean(pos[88:104]))
    except (KeyError, ValueError):
        return 0.5


def _derive_causal_strength(manifold: "LivingManifold", label: str) -> float:
    """Mean signed component on causal dims 64–79."""
    try:
        pos = manifold.position(label)
        return float(np.mean(pos[64:80]))
    except (KeyError, ValueError):
        return 0.0


def _derive_hedging(
    manifold: "LivingManifold",
    label: str,
    k: int = 5,
) -> bool:
    """Return True if the word's nearest neighbours include uncertainty seeds."""
    try:
        pos = manifold.position(label)
        neighbors = manifold.nearest(pos, k=k)
        for (nlabel, _pos_vec) in neighbors:
            if nlabel in _UNCERTAINTY_SEEDS:
                return True
        return False
    except (KeyError, ValueError):
        return False


def _rhythm_from_text(text: str) -> str:
    """Derive rhythm class from word count."""
    n = len(text.split())
    if n <= 4:
        return "short"
    if n <= 9:
        return "medium"
    return "long"


# ─────────────────────────────────────────────────────────────────────────────
# TemplateBuilder
# ─────────────────────────────────────────────────────────────────────────────

class TemplateBuilder:
    """Construct ExpressionEntry objects from word positions on M(t).

    Three levels of abstraction are generated (§4.4):
      Level 1 — single words
      Level 2 — short phrases (adjacent + nearby words)
      Level 3 — sentence frames with {} slots

    Parameters
    ----------
    manifold : LivingManifold
        Fully populated manifold (must contain vocab::* points).
    phrase_radius : float | None
        Geodesic radius threshold for Level 2 phrase combination.
        If None, calibrate_phrase_radius() must be called first or a
        default of _DEFAULT_PHRASE_RADIUS is used.
    max_level1 : int
        Maximum Level 1 entries to generate.  Default 50_000.
    max_level2 : int
        Maximum Level 2 entries to generate.  Default 35_000.
    max_level3 : int
        Maximum Level 3 entries to generate.  Default 15_000.
    """

    def __init__(
        self,
        manifold: "LivingManifold",
        phrase_radius: Optional[float] = None,
        max_level1: int = 50_000,
        max_level2: int = 35_000,
        max_level3: int = 15_000,
    ) -> None:
        self._manifold     = manifold
        self._phrase_radius = phrase_radius or _DEFAULT_PHRASE_RADIUS
        self.max_level1    = max_level1
        self.max_level2    = max_level2
        self.max_level3    = max_level3

    # ── Public API ─────────────────────────────────────────────────────────

    def calibrate_phrase_radius(self) -> float:
        """Calibrate the phrase-combination radius from manifold density (§9.4).

        Computes the mean pairwise distance within the top-10 densest clusters
        on the similarity fiber, then sets phrase_radius = 0.5 × that mean.

        Returns the calibrated radius.
        """
        vocab_labels = [
            l for l in self._manifold._points if l.startswith("vocab::")
        ]
        if len(vocab_labels) < 20:
            return self._phrase_radius

        # Sample densities
        densities = []
        for label in vocab_labels:
            try:
                pos = self._manifold.position(label)
                d   = float(self._manifold.density(pos))
                densities.append((d, label))
            except (KeyError, ValueError):
                continue

        densities.sort(key=lambda x: -x[0])
        top_labels = [l for _, l in densities[:50]]

        # Mean pairwise distance in the top-50 dense labels
        positions = []
        for label in top_labels:
            try:
                positions.append(self._manifold.position(label))
            except (KeyError, ValueError):
                continue

        if len(positions) < 2:
            return self._phrase_radius

        dists = []
        for i in range(min(len(positions), 20)):
            for j in range(i + 1, min(len(positions), 20)):
                d = float(np.linalg.norm(positions[i] - positions[j]))
                dists.append(d)

        if not dists:
            return self._phrase_radius

        mean_dist = float(np.mean(dists))
        self._phrase_radius = max(0.05, 0.5 * mean_dist)
        return self._phrase_radius

    def build(self, matrix: Optional[CoOccurrenceMatrix] = None) -> List[ExpressionEntry]:
        """Build all three levels of vocabulary entries.

        Parameters
        ----------
        matrix : CoOccurrenceMatrix | None
            If provided, co-occurrence statistics are used to weight Level 2
            phrase selection (higher PMI → phrase candidates prioritised).
            If None, proximity on the manifold alone determines phrases.

        Returns
        -------
        List[ExpressionEntry] — all generated entries, deduplicated.
        """
        entries: List[ExpressionEntry] = []
        seen_texts = set()

        # ── Level 1 ────────────────────────────────────────────────────────
        l1 = self._build_level1()
        for e in l1:
            if e.text not in seen_texts:
                seen_texts.add(e.text)
                entries.append(e)

        # ── Level 2 ────────────────────────────────────────────────────────
        l2 = self._build_level2(matrix)
        for e in l2:
            if e.text not in seen_texts:
                seen_texts.add(e.text)
                entries.append(e)

        # ── Level 3 ────────────────────────────────────────────────────────
        l3 = self._build_level3()
        for e in l3:
            if e.text not in seen_texts:
                seen_texts.add(e.text)
                entries.append(e)

        return entries

    # ── Level 1 — single words (batch-vectorized) ──────────────────────────

    def _build_level1(self) -> List[ExpressionEntry]:
        """Generate one ExpressionEntry per vocabulary word on M(t).

        Uses batch-vectorized numpy (or cupy on GPU) to compute all wave
        profiles and metadata in a single pass — avoids 50K+ per-word loops.
        Falls back to per-word iteration only when batch fails.
        """
        vocab_labels = [
            l for l in self._manifold._points if l.startswith("vocab::")
        ]
        vocab_labels = vocab_labels[: self.max_level1]
        if not vocab_labels:
            return []

        # ── Gather all positions + densities into arrays ───────────────
        valid_labels = []
        valid_words  = []
        positions    = []
        densities    = []

        for label in vocab_labels:
            word = label[len("vocab::")+0:]
            if not word:
                continue
            try:
                pos = self._manifold.position(label)
                d   = float(self._manifold.density(pos))
                valid_labels.append(label)
                valid_words.append(word)
                positions.append(pos)
                densities.append(d)
            except (KeyError, ValueError):
                continue

        if not positions:
            return []

        N = len(positions)

        # ── Batch compute wave profiles ────────────────────────────────
        pos_arr = np.stack(positions)               # (N, 104)
        den_arr = np.array(densities, dtype=float)  # (N,)

        # Use GPU if cupy is available
        if _HAS_CUPY:
            pos_gpu = cp.asarray(pos_arr[:, :WAVE_DIM])  # (N, WAVE_DIM)
            norms   = cp.linalg.norm(pos_gpu, axis=1, keepdims=True)
            norms   = cp.maximum(norms, 1e-12)
            waves   = cp.asnumpy(pos_gpu / norms)         # (N, WAVE_DIM)
        else:
            wave_arr = pos_arr[:, :WAVE_DIM]              # (N, WAVE_DIM)
            norms    = np.linalg.norm(wave_arr, axis=1, keepdims=True)
            norms    = np.maximum(norms, 1e-12)
            waves    = wave_arr / norms                    # (N, WAVE_DIM)

        # ── Batch compute metadata from array slicing ──────────────────
        prob_fiber   = pos_arr[:, 88:104]             # (N, 16)
        causal_fiber = pos_arr[:, 64:80]              # (N, 16)

        mean_prob    = prob_fiber.mean(axis=1)        # (N,)
        mean_causal  = causal_fiber.mean(axis=1)      # (N,)
        mean_unc     = mean_prob                      # same computation

        # Vectorized register: formal > 0.65, casual < 0.35, else neutral
        registers = np.where(
            mean_prob > 0.65, "formal",
            np.where(mean_prob < 0.35, "casual", "neutral")
        )

        # ── Build entries (fast — no manifold lookups in this loop) ────
        entries = []
        for i in range(N):
            entry = ExpressionEntry(
                text             = valid_words[i],
                wave_profile     = waves[i],
                register         = str(registers[i]),
                rhythm           = "short",
                uncertainty_fit  = float(mean_unc[i]),
                causal_strength  = float(mean_causal[i]),
                hedging          = _derive_hedging(self._manifold, valid_labels[i]),
            )
            entries.append(entry)

        return entries

    # ── Level 2 — short phrases (batch kNN) ──────────────────────────────

    def _build_level2(
        self, matrix: Optional[CoOccurrenceMatrix] = None
    ) -> List[ExpressionEntry]:
        """Generate phrase entries from nearby word combinations (§4.4).

        Uses batch-vectorized kNN via cKDTree (or FAISS-GPU when available)
        to find all nearby pairs at once instead of per-word nearest() calls.
        """
        vocab_labels = [
            l for l in self._manifold._points if l.startswith("vocab::")
        ]
        if len(vocab_labels) < 2:
            return []

        # ── Gather all positions into a matrix ────────────────────────────
        label_list = []
        pos_list   = []
        for label in vocab_labels:
            try:
                pos = self._manifold.position(label)
                label_list.append(label)
                pos_list.append(pos)
            except (KeyError, ValueError):
                continue

        if len(pos_list) < 2:
            return []

        pos_arr = np.stack(pos_list)  # (V, 104)

        # ── Batch kNN: find 6 nearest neighbours for all words at once ────
        from scipy.spatial import cKDTree
        tree = cKDTree(pos_arr)
        dists_all, idxs_all = tree.query(pos_arr, k=7)  # k+1 because self is included

        # ── Build phrase entries from nearby pairs ─────────────────────────
        entries   = []
        seen_pair = set()
        count     = 0

        for i in range(len(label_list)):
            if count >= self.max_level2:
                break
            label_a = label_list[i]
            word_a  = label_a[len("vocab::"):]

            for j_idx in range(1, min(7, len(idxs_all[i]))):  # skip self at idx 0
                if count >= self.max_level2:
                    break
                j    = idxs_all[i][j_idx]
                dist = dists_all[i][j_idx]
                if dist > self._phrase_radius:
                    continue

                label_b = label_list[j]
                if not label_b.startswith("vocab::"):
                    continue

                pair_key = tuple(sorted([label_a, label_b]))
                if pair_key in seen_pair:
                    continue
                seen_pair.add(pair_key)

                word_b = label_b[len("vocab::"):]
                if len(word_a) <= len(word_b):
                    phrase = f"{word_a} {word_b}"
                    labels = [label_a, label_b]
                else:
                    phrase = f"{word_b} {word_a}"
                    labels = [label_b, label_a]

                try:
                    wave = compose_wave_profile(self._manifold, labels)
                    entry = ExpressionEntry(
                        text             = phrase,
                        wave_profile     = wave,
                        register         = _derive_register(self._manifold, labels[0]),
                        rhythm           = "short",
                        uncertainty_fit  = float(np.mean([
                            _derive_uncertainty_fit(self._manifold, l) for l in labels
                        ])),
                        causal_strength  = float(np.mean([
                            _derive_causal_strength(self._manifold, l) for l in labels
                        ])),
                        hedging          = any(
                            _derive_hedging(self._manifold, l) for l in labels
                        ),
                    )
                    entries.append(entry)
                    count += 1
                except (KeyError, ValueError):
                    continue

        return entries

    # ── Level 3 — sentence frames ──────────────────────────────────────────

    def _build_level3(self) -> List[ExpressionEntry]:
        """Generate sentence frame entries with {} slots (§4.4 Level 3).

        Slot wave profiles are derived from high-density clusters on the
        similarity fiber.  If fewer vocab words than slots exist, frames
        are still generated using the available words.
        """
        entries       = []
        # Find words to represent each frame slot
        slot_words    = self._dense_vocab_sample(n=20)

        for text_template, feature_hints in _SENTENCE_FRAMES:
            if len(entries) >= self.max_level3:
                break

            # Count {} slots in the template
            n_slots = text_template.count("{}")

            # Use slot_words in round-robin for labels
            slot_labels = [
                f"vocab::{slot_words[i % len(slot_words)]}"
                for i in range(n_slots)
            ] if slot_words else []

            # Compute wave profile from slot labels
            if slot_labels:
                wave = compose_wave_profile(self._manifold, slot_labels)
            else:
                # Fallback: derive from feature hints
                wave = self._wave_from_hints(feature_hints)

            # Merge feature hints with geometry-derived values
            if slot_labels:
                causal  = float(np.mean([
                    _derive_causal_strength(self._manifold, l)
                    for l in slot_labels if self._on_manifold(l)
                ] or [feature_hints.get("causal_strength", 0.0)]))
                unc_fit = float(np.mean([
                    _derive_uncertainty_fit(self._manifold, l)
                    for l in slot_labels if self._on_manifold(l)
                ] or [feature_hints.get("uncertainty_fit", 0.5)]))
            else:
                causal  = float(feature_hints.get("causal_strength", 0.0))
                unc_fit = float(feature_hints.get("uncertainty_fit", 0.5))

            entry = ExpressionEntry(
                text             = text_template,
                wave_profile     = wave,
                register         = feature_hints.get("register", "neutral"),
                rhythm           = feature_hints.get("rhythm", "medium"),
                uncertainty_fit  = unc_fit,
                causal_strength  = causal,
                hedging          = bool(feature_hints.get("hedging", False)),
            )
            entries.append(entry)

        return entries

    # ── Helper methods ─────────────────────────────────────────────────────

    def _dense_vocab_sample(self, n: int = 20) -> List[str]:
        """Return up to *n* high-density vocabulary word strings."""
        vocab_labels = [
            l for l in self._manifold._points if l.startswith("vocab::")
        ]
        if not vocab_labels:
            return []

        densities = []
        for label in vocab_labels:
            try:
                pos = self._manifold.position(label)
                d   = float(self._manifold.density(pos))
                densities.append((d, label[len("vocab::"):]))
            except (KeyError, ValueError):
                continue

        densities.sort(key=lambda x: -x[0])
        return [word for _, word in densities[:n]]

    def _wave_from_hints(self, hints: dict) -> np.ndarray:
        """Fallback wave profile built from feature hints (like the old matcher)."""
        profile = np.zeros(WAVE_DIM, dtype=float)
        rng = np.random.default_rng(hash(str(sorted(hints.items()))) % 2**31)
        causal  = float(hints.get("causal_strength", 0.0))
        unc_fit = float(hints.get("uncertainty_fit", 0.5))
        hedging = bool(hints.get("hedging", False))
        profile[0:16]  = causal  * (0.5 + 0.5 * rng.uniform(size=16))
        profile[32:48] = unc_fit * (0.5 + 0.5 * rng.uniform(size=16))
        if hedging:
            profile[32:48] *= 1.5
        norm = np.linalg.norm(profile)
        if norm < 1e-12:
            return np.ones(WAVE_DIM, dtype=float) / np.sqrt(WAVE_DIM)
        return profile / norm

    def _on_manifold(self, label: str) -> bool:
        """Return True if *label* is registered on the manifold."""
        try:
            self._manifold.position(label)
            return True
        except (KeyError, ValueError):
            return False
