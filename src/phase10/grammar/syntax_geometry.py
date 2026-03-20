"""Syntax Geometry — S-V-O ordering and role assignment from causal fiber.

Encodes syntactic structure as geometric operations on the 104D manifold
vectors embedded in WavePoints.  No grammar rules are hand-coded — the
causal fiber (dims 64–79) determines which concept acts as subject, verb,
or object; the similarity fiber quadrant (dims 0–63) identifies the
morphological class (verb/noun/adj/adv); and the logical fiber (dims 80–87)
encodes quantification and negation.

Grammar-as-geometry rules
-------------------------
  Subject → Verb:      causal_direction(S, V) > 0  (subject causally precedes verb)
  Verb → Object:       causal_direction(V, O) > 0  (verb causally precedes object)
  Modifier → Head:     ‖modifier − head‖ < phrase_radius  (closeness = modification)
  Causal ordering:     τ(subject) ≤ τ(verb) ≤ τ(object)

No weights.  No tokens.  No training.  Syntax IS geometry.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Tuple

import numpy as np

# ── Manifold dimension slices ────────────────────────────────────────────
_SIM_START, _SIM_END = 0, 64
_CAUSAL_START, _CAUSAL_END = 64, 80
_LOGICAL_START, _LOGICAL_END = 80, 88
_PROB_START, _PROB_END = 88, 104

# Similarity fiber quadrant offsets (from WordPlacer morphological classification)
_VERB_QUAD = 0       # dims 0–15
_NOUN_QUAD = 16      # dims 16–31
_ADJ_QUAD = 32       # dims 32–47
_ADV_QUAD = 48       # dims 48–63

# Causal asymmetry coefficient (from architecture spec)
_GAMMA = 2.0

# Phrase modification radius — within this distance, words modify each other
_PHRASE_RADIUS = 1.5


class SyntacticRole(Enum):
    """Syntactic role assigned to a concept by the geometry."""
    SUBJECT = "subject"
    VERB = "verb"
    OBJECT = "object"
    MODIFIER = "modifier"     # adjective/adverb modifying a head
    CONNECTOR = "connector"   # causal/logical connective
    TOPIC = "topic"           # high-amplitude thematic anchor
    COMPLEMENT = "complement" # additional information (low causal, medium amplitude)


@dataclass
class RoleAssignment:
    """A concept with its assigned syntactic role.

    Attributes
    ----------
    label       : human-readable concept label
    role        : syntactic role (subject, verb, object, modifier, etc.)
    vector      : 104D manifold position
    amplitude   : wave amplitude (meaning centrality)
    tau         : causal time coordinate
    morph_class : inferred morphological class (verb/noun/adj/adv/function)
    causal_strength : mean absolute value of causal fiber
    confidence  : role assignment confidence ∈ [0, 1]
    """
    label: str
    role: SyntacticRole
    vector: np.ndarray
    amplitude: float
    tau: float
    morph_class: str = "noun"
    causal_strength: float = 0.0
    confidence: float = 0.5

    def __repr__(self) -> str:
        return (
            f"RoleAssignment('{self.label}', {self.role.value}, "
            f"morph={self.morph_class}, amp={self.amplitude:.3f}, "
            f"tau={self.tau:.2f}, conf={self.confidence:.2f})"
        )


class SyntaxGeometry:
    """Assigns syntactic roles to wave concepts using manifold geometry.

    This class operates entirely within C7's boundary — it receives only
    the 104D vectors, amplitudes, labels, and tau values embedded in
    WavePoints.  It never accesses the manifold directly.

    The causal fiber (dims 64–79) drives ordering: concepts with earlier
    τ and stronger causal outflow become subjects; concepts with later τ
    and causal inflow become objects; concepts in the verb quadrant of
    the similarity fiber become verbs.

    Parameters
    ----------
    phrase_radius : float
        Maximum distance for a modifier to attach to its head.
    causal_threshold : float
        Minimum causal fiber magnitude to count as a causal participant.
    """

    def __init__(
        self,
        phrase_radius: float = _PHRASE_RADIUS,
        causal_threshold: float = 0.1,
    ) -> None:
        self.phrase_radius = phrase_radius
        self.causal_threshold = causal_threshold

    # ── Public API ─────────────────────────────────────────────────── #

    def assign_roles(
        self,
        vectors: List[np.ndarray],
        labels: List[str],
        amplitudes: List[float],
        taus: List[float],
    ) -> List[RoleAssignment]:
        """Assign syntactic roles to a list of concepts from a WaveSegment.

        Parameters
        ----------
        vectors    : list of 104D manifold position vectors
        labels     : human-readable concept labels
        amplitudes : wave amplitudes (meaning centrality)
        taus       : causal time coordinates

        Returns
        -------
        List of RoleAssignment ordered by natural speaking order
        (subject → verb → object → modifiers).
        """
        if not vectors:
            return []

        n = len(vectors)
        assignments: List[RoleAssignment] = []

        for i in range(n):
            vec = vectors[i]
            morph = self._infer_morph_class(vec)
            causal_str = self._causal_strength(vec)
            assignments.append(RoleAssignment(
                label=labels[i] if i < len(labels) else f"concept_{i}",
                role=SyntacticRole.COMPLEMENT,  # default; refined below
                vector=vec,
                amplitude=amplitudes[i] if i < len(amplitudes) else 0.0,
                tau=taus[i] if i < len(taus) else 0.5,
                morph_class=morph,
                causal_strength=causal_str,
                confidence=0.0,
            ))

        # ── Phase 1: Assign primary roles by morphology + causal position ──
        self._assign_primary_roles(assignments)

        # ── Phase 2: Assign modifiers ──────────────────────────────────────
        self._assign_modifiers(assignments)

        # ── Phase 3: Order by natural speaking sequence ────────────────────
        ordered = self._order_for_speech(assignments)

        return ordered

    def causal_direction(self, vec_a: np.ndarray, vec_b: np.ndarray) -> float:
        """Compute causal direction from A to B.

        Returns a positive value if A causally precedes B (A → B),
        negative if B causally precedes A (B → A), near zero if no
        clear causal ordering.

        Uses the asymmetric causal metric with γ=2.0:
        forward distance is cheaper than backward distance.
        """
        causal_a = vec_a[_CAUSAL_START:_CAUSAL_END]
        causal_b = vec_b[_CAUSAL_START:_CAUSAL_END]

        # τ-axis is the first causal dimension (dim 64)
        tau_diff = causal_b[0] - causal_a[0]

        # Asymmetric distance: forward (τ increasing) costs 1×, backward costs γ×
        forward_cost = np.sum(np.maximum(causal_b - causal_a, 0) ** 2)
        backward_cost = _GAMMA * np.sum(np.maximum(causal_a - causal_b, 0) ** 2)

        # Net direction: positive means A → B is the natural causal flow
        return float(forward_cost - backward_cost + tau_diff)

    def concept_distance(self, vec_a: np.ndarray, vec_b: np.ndarray) -> float:
        """Euclidean distance in the full 104D space."""
        return float(np.linalg.norm(vec_a - vec_b))

    def similarity_distance(self, vec_a: np.ndarray, vec_b: np.ndarray) -> float:
        """Distance in the similarity fiber only (dims 0–63)."""
        return float(np.linalg.norm(
            vec_a[_SIM_START:_SIM_END] - vec_b[_SIM_START:_SIM_END]
        ))

    # ── Internal role assignment ───────────────────────────────────── #

    def _infer_morph_class(self, vec: np.ndarray) -> str:
        """Infer morphological class from the similarity fiber quadrant.

        Words are placed in the similarity fiber with quadrant offsets:
          - Verbs:      dims  0–15 (strongest energy)
          - Nouns:      dims 16–31
          - Adjectives: dims 32–47
          - Adverbs:    dims 48–63

        We determine which quadrant has the highest mean energy.
        """
        sim = vec[_SIM_START:_SIM_END]
        quadrant_energies = {
            "verb": float(np.mean(np.abs(sim[_VERB_QUAD:_VERB_QUAD + 16]))),
            "noun": float(np.mean(np.abs(sim[_NOUN_QUAD:_NOUN_QUAD + 16]))),
            "adjective": float(np.mean(np.abs(sim[_ADJ_QUAD:_ADJ_QUAD + 16]))),
            "adverb": float(np.mean(np.abs(sim[_ADV_QUAD:_ADV_QUAD + 16]))),
        }

        # If all energies are very low, it's a function word
        max_energy = max(quadrant_energies.values())
        if max_energy < 0.01:
            return "function"

        return max(quadrant_energies, key=quadrant_energies.get)

    def _causal_strength(self, vec: np.ndarray) -> float:
        """Mean absolute value of the causal fiber — how causally active."""
        causal = vec[_CAUSAL_START:_CAUSAL_END]
        return float(np.mean(np.abs(causal)))

    def _logical_features(self, vec: np.ndarray) -> dict:
        """Extract logical fiber features for grammar decisions."""
        logical = vec[_LOGICAL_START:_LOGICAL_END]
        return {
            "negation": float(logical[0]) if len(logical) > 0 else 0.0,
            "universal": float(logical[1]) if len(logical) > 1 else 0.0,
            "existential": float(logical[2]) if len(logical) > 2 else 0.0,
            "negative_quant": float(logical[3]) if len(logical) > 3 else 0.0,
        }

    def _probabilistic_features(self, vec: np.ndarray) -> dict:
        """Extract probabilistic fiber features for register decisions."""
        prob = vec[_PROB_START:_PROB_END]
        mean_prob = float(np.mean(prob))
        return {
            "certainty": mean_prob,
            "register": "formal" if mean_prob > 0.65 else ("casual" if mean_prob < 0.35 else "neutral"),
        }

    def _assign_primary_roles(self, assignments: List[RoleAssignment]) -> None:
        """Assign subject/verb/object based on morph class + causal position.

        Strategy:
        1. Verbs: highest-amplitude concept classified as verb
        2. Subject: highest-amplitude noun/topic with earliest τ and causal outflow
        3. Object: highest-amplitude noun with latest τ and causal inflow
        4. Remaining: complement
        """
        if not assignments:
            return

        # ── Find the best verb candidate ──────────────────────────────
        verb_candidates = [
            a for a in assignments if a.morph_class == "verb"
        ]
        # If no verb by morphology, pick the highest causal-strength concept
        if not verb_candidates:
            verb_candidates = sorted(
                assignments,
                key=lambda a: a.causal_strength,
                reverse=True,
            )

        if verb_candidates:
            best_verb = max(verb_candidates, key=lambda a: a.amplitude)
            best_verb.role = SyntacticRole.VERB
            best_verb.confidence = min(1.0, best_verb.causal_strength + 0.3)

        # ── Assign subject and object from remaining nouns ────────────
        remaining = [a for a in assignments if a.role != SyntacticRole.VERB]
        nouns = [a for a in remaining if a.morph_class in ("noun", "function")]

        if not nouns:
            nouns = remaining  # fallback: use whatever's left

        if nouns:
            # Subject: earliest τ, highest amplitude → the initiator
            subject_score = lambda a: -a.tau + a.amplitude
            subj = max(nouns, key=subject_score)
            subj.role = SyntacticRole.SUBJECT
            subj.confidence = min(1.0, subj.amplitude + 0.2)

            # Object: latest τ among remaining, highest amplitude → the receiver
            obj_candidates = [a for a in nouns if a is not subj]
            if obj_candidates:
                obj_score = lambda a: a.tau + a.amplitude * 0.5
                obj = max(obj_candidates, key=obj_score)
                obj.role = SyntacticRole.OBJECT
                obj.confidence = min(1.0, obj.amplitude + 0.1)

        # ── High-amplitude topic anchor ───────────────────────────────
        for a in assignments:
            if a.role == SyntacticRole.COMPLEMENT and a.amplitude > 0.7:
                a.role = SyntacticRole.TOPIC
                a.confidence = a.amplitude

    def _assign_modifiers(self, assignments: List[RoleAssignment]) -> None:
        """Assign modifier roles to adjectives/adverbs near their heads."""
        heads = [a for a in assignments
                 if a.role in (SyntacticRole.SUBJECT, SyntacticRole.VERB,
                               SyntacticRole.OBJECT, SyntacticRole.TOPIC)]

        for a in assignments:
            if a.role != SyntacticRole.COMPLEMENT:
                continue
            if a.morph_class not in ("adjective", "adverb"):
                continue

            # Find nearest head within phrase radius
            nearest_head = None
            nearest_dist = float("inf")
            for h in heads:
                d = self.similarity_distance(a.vector, h.vector)
                if d < nearest_dist and d < self.phrase_radius:
                    nearest_dist = d
                    nearest_head = h

            if nearest_head is not None:
                a.role = SyntacticRole.MODIFIER
                a.confidence = float(1.0 - nearest_dist / self.phrase_radius)

    def _order_for_speech(
        self, assignments: List[RoleAssignment]
    ) -> List[RoleAssignment]:
        """Order assignments for natural English sentence construction.

        Order: Subject → Modifiers-of-subject → Verb → Modifiers-of-verb
               → Object → Modifiers-of-object → Topic → Complement

        Within each group, order by τ (causal time).
        """
        role_priority = {
            SyntacticRole.SUBJECT: 0,
            SyntacticRole.MODIFIER: 1,   # will be re-sorted near their head
            SyntacticRole.VERB: 2,
            SyntacticRole.OBJECT: 3,
            SyntacticRole.TOPIC: 4,
            SyntacticRole.CONNECTOR: 5,
            SyntacticRole.COMPLEMENT: 6,
        }

        return sorted(
            assignments,
            key=lambda a: (role_priority.get(a.role, 9), a.tau),
        )
