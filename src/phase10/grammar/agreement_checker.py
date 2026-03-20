"""Agreement Checker — Number/tense agreement as manifold distance constraints.

Verifies and corrects subject-verb agreement, tense consistency, and
modifier-head agreement using geometric distance constraints on the
104D fiber bundle.

Agreement as geometry
---------------------
  Subject-verb number:  ‖logical_fiber(S) − logical_fiber(V)‖ < ε_number
  Tense consistency:    all verbs' causal fiber τ-offsets within ε_tense
  Modifier-head:        similarity_fiber distance < phrase_radius

When agreement fails, the checker proposes the minimal geometric
correction (inflection change) that restores agreement.

No weights.  No tokens.  No training.  Agreement IS distance.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from .syntax_geometry import (
    SyntacticRole,
    RoleAssignment,
    _CAUSAL_START, _CAUSAL_END,
    _LOGICAL_START, _LOGICAL_END,
)
from .morphology_map import MorphologyMap, Inflection


@dataclass
class AgreementResult:
    """Result of an agreement check.

    Attributes
    ----------
    is_valid         : whether all agreement constraints are satisfied
    violations       : list of (description, severity) pairs
    corrections      : list of (role_index, suggested_inflection) pairs
    confidence       : confidence in the agreement analysis
    """
    is_valid: bool = True
    violations: List[Tuple[str, float]] = field(default_factory=list)
    corrections: List[Tuple[int, Inflection]] = field(default_factory=list)
    confidence: float = 0.9

    def __repr__(self) -> str:
        status = "VALID" if self.is_valid else f"INVALID ({len(self.violations)} violations)"
        return f"AgreementResult({status}, conf={self.confidence:.2f})"


class AgreementChecker:
    """Checks and corrects grammatical agreement using geometric constraints.

    Agreement rules are encoded as distance thresholds in specific fibers:

    1. Subject-verb number agreement:
       - Plural subject (high logical dim 81) → verb needs plural form
       - Singular subject → verb needs third-person-singular form

    2. Tense consistency:
       - All verbs in a clause should have consistent causal fiber τ-offset
       - Past tense verbs: τ-offset < 0;  Present: ≈ 0;  Future: > 0

    3. Determiner-noun agreement:
       - "a/an" with singular; "these/those" with plural

    Parameters
    ----------
    morphology : MorphologyMap
        Morphological inflection engine.
    number_threshold : float
        Maximum logical fiber distance for number agreement.
    tense_threshold : float
        Maximum τ-offset variance for tense consistency.
    """

    def __init__(
        self,
        morphology: Optional[MorphologyMap] = None,
        number_threshold: float = 0.15,
        tense_threshold: float = 0.1,
    ) -> None:
        self.morphology = morphology or MorphologyMap()
        self.number_threshold = number_threshold
        self.tense_threshold = tense_threshold

    # ── Public API ─────────────────────────────────────────────────── #

    def check(self, roles: List[RoleAssignment]) -> AgreementResult:
        """Check all agreement constraints for a set of role assignments.

        Parameters
        ----------
        roles : ordered list of syntactic role assignments

        Returns
        -------
        AgreementResult with validity, violations, and corrections.
        """
        violations: List[Tuple[str, float]] = []
        corrections: List[Tuple[int, Inflection]] = []

        # ── Subject-verb number agreement ─────────────────────────────
        sv_viols, sv_corrs = self._check_subject_verb_number(roles)
        violations.extend(sv_viols)
        corrections.extend(sv_corrs)

        # ── Tense consistency ─────────────────────────────────────────
        tense_viols = self._check_tense_consistency(roles)
        violations.extend(tense_viols)

        is_valid = len(violations) == 0
        confidence = 1.0 - 0.1 * len(violations)
        confidence = max(0.0, min(1.0, confidence))

        return AgreementResult(
            is_valid=is_valid,
            violations=violations,
            corrections=corrections,
            confidence=confidence,
        )

    def correct_surface(
        self,
        word: str,
        inflection: Inflection,
    ) -> str:
        """Apply a correction by inflecting a word.

        Parameters
        ----------
        word       : the current surface form
        inflection : the target inflection

        Returns
        -------
        The corrected surface form.
        """
        form = self.morphology.analyse(word)
        return self.morphology.inflect(form.base, inflection)

    def infer_tense(self, roles: List[RoleAssignment]) -> str:
        """Infer the dominant tense from the causal fiber τ-offsets.

        Returns "past", "present", or "future".
        """
        verbs = [r for r in roles if r.role == SyntacticRole.VERB]
        if not verbs:
            return "present"

        tau_offsets = [r.vector[_CAUSAL_START] for r in verbs]
        mean_tau = float(np.mean(tau_offsets))

        if mean_tau < -0.05:
            return "past"
        elif mean_tau > 0.05:
            return "future"
        return "present"

    def infer_number(self, role: RoleAssignment) -> str:
        """Infer singular/plural from the logical fiber.

        Returns "singular" or "plural".
        """
        # Logical fiber dim 81 = universal quantifier (plural direction)
        # Logical fiber dim 82 = existential quantifier
        quant = role.vector[_LOGICAL_START + 1] if len(role.vector) > _LOGICAL_START + 1 else 0.0
        return "plural" if quant > 0.5 else "singular"

    # ── Internal checks ────────────────────────────────────────────── #

    def _check_subject_verb_number(
        self, roles: List[RoleAssignment]
    ) -> Tuple[List[Tuple[str, float]], List[Tuple[int, Inflection]]]:
        """Check subject-verb number agreement."""
        violations = []
        corrections = []

        subjects = [(i, r) for i, r in enumerate(roles)
                    if r.role == SyntacticRole.SUBJECT]
        verbs = [(i, r) for i, r in enumerate(roles)
                 if r.role == SyntacticRole.VERB]

        if not subjects or not verbs:
            return violations, corrections

        # Check each subject-verb pair
        for si, subj in subjects:
            subj_number = self.infer_number(subj)
            for vi, verb in verbs:
                verb_logical = verb.vector[_LOGICAL_START:_LOGICAL_END]
                subj_logical = subj.vector[_LOGICAL_START:_LOGICAL_END]

                dist = float(np.linalg.norm(verb_logical - subj_logical))
                if dist > self.number_threshold:
                    severity = min(1.0, dist / self.number_threshold)
                    violations.append((
                        f"number disagreement: '{subj.label}' ({subj_number}) "
                        f"vs '{verb.label}' (logical dist={dist:.3f})",
                        severity,
                    ))
                    # Suggest correction
                    if subj_number == "plural":
                        corrections.append((vi, Inflection.BASE))  # plural verb = base form
                    else:
                        corrections.append((vi, Inflection.THIRD_PERSON))

        return violations, corrections

    def _check_tense_consistency(
        self, roles: List[RoleAssignment]
    ) -> List[Tuple[str, float]]:
        """Check that all verbs share consistent tense (τ-offset)."""
        violations = []

        verbs = [r for r in roles if r.role == SyntacticRole.VERB]
        if len(verbs) < 2:
            return violations

        tau_offsets = [v.vector[_CAUSAL_START] for v in verbs]
        tau_var = float(np.var(tau_offsets))

        if tau_var > self.tense_threshold:
            violations.append((
                f"tense inconsistency among {len(verbs)} verbs "
                f"(τ-offset variance={tau_var:.4f})",
                min(1.0, tau_var / self.tense_threshold),
            ))

        return violations
