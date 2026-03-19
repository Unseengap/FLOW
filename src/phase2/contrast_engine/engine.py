"""Contrast Engine — places concepts via same/different relational judgments.

The core operation is elegantly simple:

  INPUT:  A pair (E₁, E₂) and a judgment J ∈ {same, different}
  OUTPUT: Two displacement vectors applied to the Living Manifold

  same      → pull P₁ and P₂ closer by factor α
  different → push P₁ and P₂ apart  by factor β

Over many judgments, the geometry self-organises so that things judged same
cluster together and things judged different remain separated — without any
labels, categories, or supervised signal.

The engine also maintains a PersistenceDiagram and periodically proposes
structural corrections back to the manifold based on which clusters are
genuinely persistent vs. transient.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Tuple

import numpy as np

from src.phase2.living_manifold.manifold import LivingManifold
from .persistence import PersistenceDiagram


class JudgmentType(Enum):
    """The two possible relational judgments."""
    SAME = "same"
    DIFFERENT = "different"


@dataclass
class ContrastPair:
    """A single contrast judgment to be applied.

    Attributes
    ----------
    label_a, label_b : concept labels already on the manifold
    judgment         : SAME or DIFFERENT
    strength         : float in (0, 1] scaling α/β (default 1.0)
    """

    label_a: str
    label_b: str
    judgment: JudgmentType
    strength: float = 1.0

    def __post_init__(self) -> None:
        if not (0.0 < self.strength <= 1.0):
            raise ValueError(f"strength must be in (0,1], got {self.strength}")


@dataclass
class ContrastResult:
    """Outcome of a single contrast operation.

    Attributes
    ----------
    label_a, label_b    : concept labels
    judgment            : applied judgment
    distance_before     : geodesic distance before the operation
    distance_after      : geodesic distance after the operation
    delta_a, delta_b    : displacement vectors applied to each concept
    n_affected_a        : points affected by the deformation at A
    n_affected_b        : points affected by the deformation at B
    """

    label_a: str
    label_b: str
    judgment: JudgmentType
    distance_before: float
    distance_after: float
    delta_a: np.ndarray
    delta_b: np.ndarray
    n_affected_a: int
    n_affected_b: int

    @property
    def distance_change(self) -> float:
        return self.distance_after - self.distance_before

    @property
    def moved_correct_direction(self) -> bool:
        """True if distance changed in the right direction for the judgment."""
        if self.judgment == JudgmentType.SAME:
            return self.distance_change < 0  # should decrease
        return self.distance_change > 0  # should increase


class ContrastEngine:
    """Place and refine concepts using same/different relational judgments.

    Parameters
    ----------
    manifold : LivingManifold
        The living manifold that will be deformed.
    alpha    : float
        Attraction coefficient for SAME judgments (~0.1).
    beta     : float
        Repulsion coefficient for DIFFERENT judgments (~0.1).
    correction_interval : int
        Apply persistent homology corrections every N judgments.
    """

    def __init__(
        self,
        manifold: LivingManifold,
        alpha: float = 0.1,
        beta: float = 0.1,
        correction_interval: int = 50,
    ) -> None:
        self._manifold = manifold
        self.alpha = alpha
        self.beta = beta
        self.correction_interval = correction_interval
        self._diagram = PersistenceDiagram()
        self._n_judgments: int = 0
        self._history: List[ContrastResult] = []

    # ------------------------------------------------------------------ #
    # Core operation                                                       #
    # ------------------------------------------------------------------ #

    def judge(
        self,
        label_a: str,
        label_b: str,
        judgment: JudgmentType,
        strength: float = 1.0,
    ) -> ContrastResult:
        """Apply a single SAME or DIFFERENT judgment between two concepts.

        Both concepts must already exist on the manifold.

        Parameters
        ----------
        label_a, label_b : existing concept labels
        judgment         : SAME or DIFFERENT
        strength         : scales the displacement magnitude in (0, 1]

        Returns
        -------
        ContrastResult with before/after distances and displacement vectors.
        """
        pa = self._manifold.position(label_a)
        pb = self._manifold.position(label_b)

        dist_before = float(np.linalg.norm(pb - pa))

        delta_a, delta_b = self._compute_displacements(pa, pb, judgment, strength)

        n_a = self._manifold.deform_local(label_a, delta_a)
        n_b = self._manifold.deform_local(label_b, delta_b)

        pa_new = self._manifold.position(label_a)
        pb_new = self._manifold.position(label_b)
        dist_after = float(np.linalg.norm(pb_new - pa_new))

        self._manifold.update_density(label_a)
        self._manifold.update_density(label_b)

        result = ContrastResult(
            label_a=label_a,
            label_b=label_b,
            judgment=judgment,
            distance_before=dist_before,
            distance_after=dist_after,
            delta_a=delta_a,
            delta_b=delta_b,
            n_affected_a=n_a,
            n_affected_b=n_b,
        )

        self._diagram.record(label_a, label_b, dist_after, self._manifold.t)
        self._history.append(result)
        self._n_judgments += 1

        if self._n_judgments % self.correction_interval == 0:
            self.apply_structural_corrections()

        return result

    def judge_batch(
        self, pairs: List[ContrastPair]
    ) -> List[ContrastResult]:
        """Apply a list of contrast judgments in sequence.

        Returns a list of ContrastResult in the same order.
        """
        return [
            self.judge(
                p.label_a, p.label_b, p.judgment, p.strength
            )
            for p in pairs
        ]

    def judge_fast(
        self,
        label_a: str,
        label_b: str,
        judgment: JudgmentType,
        strength: float = 1.0,
    ) -> int:
        """Lightweight contrast judgment — skips density updates & persistence.

        Applies the same displacement logic as ``judge()`` but omits the
        per-pair density recomputation, persistence diagram recording, and
        structural-correction checks.  Callers should call
        ``manifold.force_rebuild_tree()`` and refresh densities in bulk
        after processing a large batch of fast judgments.

        Returns
        -------
        int — 1 if applied, 0 if skipped (coincident points).
        """
        pa = self._manifold.position(label_a)
        pb = self._manifold.position(label_b)

        delta_a, delta_b = self._compute_displacements(pa, pb, judgment, strength)

        # Skip if both deltas are zero (coincident points with SAME judgment)
        if np.dot(delta_a, delta_a) < 1e-24 and np.dot(delta_b, delta_b) < 1e-24:
            return 0

        self._manifold.deform_local(label_a, delta_a)
        self._manifold.deform_local(label_b, delta_b)

        self._n_judgments += 1
        return 1

    # ------------------------------------------------------------------ #
    # Displacement computation                                             #
    # ------------------------------------------------------------------ #

    def _compute_displacements(
        self,
        pa: np.ndarray,
        pb: np.ndarray,
        judgment: JudgmentType,
        strength: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute displacement vectors for both points.

        For SAME:
          Both points move toward each other by α·strength·(distance/2).
          direction of A's move = unit(B - A)
          direction of B's move = unit(A - B)

        For DIFFERENT:
          Both points move away from each other by β·strength·(distance/2).
          direction of A's move = unit(A - B)  (away from B)
          direction of B's move = unit(B - A)  (away from A)
        """
        diff = pb - pa  # vector from A to B
        dist = float(np.linalg.norm(diff))

        if dist < 1e-12:
            # Points are coincident — use a random perturbation for DIFFERENT
            if judgment == JudgmentType.DIFFERENT:
                rng = np.random.default_rng(seed=hash(str(pa.tobytes())) % (2**32))
                direction = rng.standard_normal(pa.shape)
                direction /= np.linalg.norm(direction) + 1e-12
            else:
                return np.zeros_like(pa), np.zeros_like(pb)
        else:
            direction = diff / dist  # unit vector from A → B

        move_dist = dist / 2.0  # each point moves half the distance

        if judgment == JudgmentType.SAME:
            # Attraction: A moves toward B, B moves toward A
            magnitude = self.alpha * strength * move_dist
            delta_a = +magnitude * direction   # toward B
            delta_b = -magnitude * direction   # toward A
        else:
            # Repulsion: A moves away from B, B moves away from A
            magnitude = self.beta * strength * move_dist
            delta_a = -magnitude * direction   # away from B
            delta_b = +magnitude * direction   # away from A

        return delta_a, delta_b

    # ------------------------------------------------------------------ #
    # Structural corrections via persistent homology                       #
    # ------------------------------------------------------------------ #

    def apply_structural_corrections(
        self, min_lifetime: float = 2.0
    ) -> int:
        """Apply corrections suggested by the persistence diagram.

        Corrections that suggest "tighten" are applied as SAME judgments
        with low strength; "separate" corrections as DIFFERENT judgments.

        Returns the number of corrections applied.
        """
        corrections = self._diagram.cluster_corrections(min_lifetime)
        applied = 0
        for corr in corrections:
            la, lb = corr["label_a"], corr["label_b"]
            strength = float(corr["strength"]) * 0.3  # soft corrections

            # Only apply if both labels exist on the manifold
            if la not in self._manifold.labels or lb not in self._manifold.labels:
                continue

            judgment = (
                JudgmentType.SAME
                if corr["type"] == "tighten"
                else JudgmentType.DIFFERENT
            )
            try:
                self.judge(la, lb, judgment, strength=max(strength, 0.01))
                applied += 1
            except (KeyError, ValueError):
                continue

        return applied

    # ------------------------------------------------------------------ #
    # Self-supervised pair generation                                      #
    # ------------------------------------------------------------------ #

    def generate_temporal_pairs(
        self,
        sequence: List[str],
        window: int = 3,
    ) -> List[ContrastPair]:
        """Generate SAME pairs from temporal proximity.

        Concepts that appear close together in *sequence* are assumed to
        be related and get a SAME judgment.

        Parameters
        ----------
        sequence : ordered list of concept labels
        window   : concepts within this distance in the sequence are paired

        Returns
        -------
        List of ContrastPairs with SAME judgment.
        """
        pairs: List[ContrastPair] = []
        for i, la in enumerate(sequence):
            for j in range(i + 1, min(i + window + 1, len(sequence))):
                lb = sequence[j]
                if la != lb:
                    # Closer in sequence = stronger signal
                    distance_in_seq = j - i
                    strength = 1.0 / distance_in_seq
                    pairs.append(ContrastPair(la, lb, JudgmentType.SAME, strength=strength))
        return pairs

    def generate_contrast_pairs(
        self,
        group_a: List[str],
        group_b: List[str],
    ) -> List[ContrastPair]:
        """Generate DIFFERENT pairs between two groups.

        Every element of group_a is paired with every element of group_b
        as a DIFFERENT judgment.
        """
        pairs: List[ContrastPair] = []
        for la in group_a:
            for lb in group_b:
                pairs.append(ContrastPair(la, lb, JudgmentType.DIFFERENT))
        return pairs

    # ------------------------------------------------------------------ #
    # Properties / summary                                                 #
    # ------------------------------------------------------------------ #

    @property
    def n_judgments(self) -> int:
        return self._n_judgments

    @property
    def persistence_diagram(self) -> PersistenceDiagram:
        return self._diagram

    @property
    def history(self) -> List[ContrastResult]:
        return list(self._history)

    def correct_direction_rate(self) -> float:
        """Fraction of judgments that moved concepts in the correct direction."""
        if not self._history:
            return 0.0
        correct = sum(1 for r in self._history if r.moved_correct_direction)
        return correct / len(self._history)

    def summary(self) -> str:
        rate = self.correct_direction_rate() * 100
        lines = [
            "═══ Contrast Engine ═════════════════════════════════════════",
            f"  Total judgments    : {self.n_judgments}",
            f"  α (attraction)     : {self.alpha}",
            f"  β (repulsion)      : {self.beta}",
            f"  Correct direction  : {rate:.1f}%",
            f"  Persistence pairs  : {len(self._diagram)}",
            f"  Manifold writes    : {self._manifold.n_writes}",
            "═════════════════════════════════════════════════════════════",
        ]
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"ContrastEngine(judgments={self.n_judgments}, "
            f"α={self.alpha}, β={self.beta})"
        )
