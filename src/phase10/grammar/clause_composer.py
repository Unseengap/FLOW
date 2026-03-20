"""Clause Composer — Main + subordinate clause composition via fiber bundle.

Composes multi-clause sentences by interpreting the wave segment structure
as a hierarchical clause tree.  The causal fiber determines clause ordering
(cause-clause before effect-clause), the logical fiber determines clause
type (conditional, conjunctive, contrastive), and amplitude determines
clause importance (main vs subordinate).

Clause types derived geometrically
-----------------------------------
  MAIN:         High amplitude, strong causal outflow
  SUBORDINATE:  Lower amplitude, contained within main's fiber section
  CONDITIONAL:  Logical fiber shows implication (IF...THEN)
  CONTRASTIVE:  Contrast signal between adjacent segments (ALTHOUGH/WHILE)
  CAUSAL:       Strong causal direction between segments (BECAUSE/THEREFORE)
  TEMPORAL:     τ-ordered with significant gap (BEFORE/AFTER/WHEN)
  RELATIVE:     Modifier with shared referent (WHICH/THAT)

No weights.  No tokens.  No training.  Clauses emerge from geometry.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Tuple

import numpy as np

from .syntax_geometry import (
    SyntaxGeometry,
    SyntacticRole,
    RoleAssignment,
    _CAUSAL_START, _CAUSAL_END,
    _LOGICAL_START, _LOGICAL_END,
    _PROB_START, _PROB_END,
)


class ClauseType(Enum):
    """Type of clause, derived from geometric relationships."""
    MAIN = "main"
    SUBORDINATE = "subordinate"
    CONDITIONAL = "conditional"       # if...then
    CONTRASTIVE = "contrastive"       # although / while / however
    CAUSAL = "causal"                 # because / therefore / since
    TEMPORAL = "temporal"             # when / before / after
    RELATIVE = "relative"            # which / that
    ADDITIVE = "additive"            # and / also / furthermore


@dataclass
class Clause:
    """A single syntactic clause with its roles and type.

    Attributes
    ----------
    clause_type : the geometric clause type
    roles       : ordered list of role assignments (S-V-O-Mod)
    amplitude   : mean amplitude of the clause's concepts
    tau_range   : (min_tau, max_tau) — causal time span
    coherence   : how tightly the clause's concepts cluster
    confidence  : confidence in the clause type assignment
    subordinate_to : index of parent clause (None if main)
    """
    clause_type: ClauseType
    roles: List[RoleAssignment]
    amplitude: float = 0.5
    tau_range: Tuple[float, float] = (0.0, 1.0)
    coherence: float = 0.5
    confidence: float = 0.5
    subordinate_to: Optional[int] = None

    @property
    def is_main(self) -> bool:
        return self.clause_type == ClauseType.MAIN

    @property
    def has_subject(self) -> bool:
        return any(r.role == SyntacticRole.SUBJECT for r in self.roles)

    @property
    def has_verb(self) -> bool:
        return any(r.role == SyntacticRole.VERB for r in self.roles)

    @property
    def subject(self) -> Optional[RoleAssignment]:
        for r in self.roles:
            if r.role == SyntacticRole.SUBJECT:
                return r
        return None

    @property
    def verb(self) -> Optional[RoleAssignment]:
        for r in self.roles:
            if r.role == SyntacticRole.VERB:
                return r
        return None

    @property
    def object(self) -> Optional[RoleAssignment]:
        for r in self.roles:
            if r.role == SyntacticRole.OBJECT:
                return r
        return None

    def __repr__(self) -> str:
        role_strs = [f"{r.label}({r.role.value})" for r in self.roles[:4]]
        return (
            f"Clause({self.clause_type.value}, "
            f"roles=[{', '.join(role_strs)}], "
            f"amp={self.amplitude:.2f})"
        )


@dataclass
class SentencePlan:
    """A planned sentence consisting of one or more composed clauses.

    Attributes
    ----------
    clauses     : ordered list of clauses (main first, subordinates follow)
    connectives : connective words/phrases between clauses
    complexity  : sentence complexity score (1=simple, 2=compound, 3=complex)
    confidence  : overall plan confidence
    """
    clauses: List[Clause] = field(default_factory=list)
    connectives: List[str] = field(default_factory=list)
    complexity: int = 1
    confidence: float = 0.5

    @property
    def is_compound(self) -> bool:
        """Multiple main clauses joined by coordinating conjunction."""
        main_count = sum(1 for c in self.clauses if c.is_main)
        return main_count >= 2

    @property
    def is_complex(self) -> bool:
        """Main clause + one or more subordinate clauses."""
        has_main = any(c.is_main for c in self.clauses)
        has_sub = any(not c.is_main for c in self.clauses)
        return has_main and has_sub

    def __repr__(self) -> str:
        types = [c.clause_type.value for c in self.clauses]
        return f"SentencePlan({types}, complexity={self.complexity})"


class ClauseComposer:
    """Composes multi-clause sentences from wave segment role assignments.

    The composer analyses relationships between segments and role groups
    to determine clause boundaries, types, and connectives — all derived
    from the geometric structure of the 104D vectors.

    Parameters
    ----------
    syntax : SyntaxGeometry
        Role assignment engine (dependency injected for reuse).
    causal_threshold : float
        Minimum causal direction magnitude to trigger a causal clause.
    contrast_threshold : float
        Minimum contrast signal to trigger a contrastive clause.
    tau_gap_threshold : float
        Minimum τ gap between groups to trigger a temporal clause.
    """

    def __init__(
        self,
        syntax: Optional[SyntaxGeometry] = None,
        causal_threshold: float = 0.15,
        contrast_threshold: float = 0.3,
        tau_gap_threshold: float = 0.2,
    ) -> None:
        self.syntax = syntax or SyntaxGeometry()
        self.causal_threshold = causal_threshold
        self.contrast_threshold = contrast_threshold
        self.tau_gap_threshold = tau_gap_threshold

    # ── Public API ─────────────────────────────────────────────────── #

    def compose(
        self,
        role_groups: List[List[RoleAssignment]],
    ) -> SentencePlan:
        """Compose a sentence plan from one or more role groups.

        Each role group typically comes from a single WaveSegment's
        role assignments.  The composer determines relationships between
        groups and constructs a clause tree.

        Parameters
        ----------
        role_groups : list of role assignment lists (one per segment/sub-segment)

        Returns
        -------
        SentencePlan with one or more clauses and connectives.
        """
        if not role_groups:
            return SentencePlan(clauses=[], complexity=0, confidence=0.0)

        clauses: List[Clause] = []
        connectives: List[str] = []

        # ── Build a clause from each role group ───────────────────────
        for i, roles in enumerate(role_groups):
            if not roles:
                continue
            clause = self._build_clause(roles)
            clauses.append(clause)

        if not clauses:
            return SentencePlan(clauses=[], complexity=0, confidence=0.0)

        # ── Determine clause relationships ────────────────────────────
        if len(clauses) == 1:
            clauses[0].clause_type = ClauseType.MAIN
            complexity = 1
        else:
            self._assign_clause_types(clauses)
            connectives = self._select_connectives(clauses)
            complexity = self._compute_complexity(clauses)

        confidence = float(np.mean([c.confidence for c in clauses]))

        return SentencePlan(
            clauses=clauses,
            connectives=connectives,
            complexity=complexity,
            confidence=confidence,
        )

    def compose_single(
        self, roles: List[RoleAssignment]
    ) -> SentencePlan:
        """Compose a sentence plan from a single set of role assignments.

        Convenience wrapper for the common case of one segment.
        """
        return self.compose([roles])

    # ── Internal clause construction ──────────────────────────────── #

    def _build_clause(self, roles: List[RoleAssignment]) -> Clause:
        """Build a clause from a set of role assignments."""
        if not roles:
            return Clause(
                clause_type=ClauseType.MAIN,
                roles=[],
                amplitude=0.0,
                confidence=0.0,
            )

        amplitudes = [r.amplitude for r in roles]
        taus = [r.tau for r in roles]
        vecs = [r.vector for r in roles]

        mean_amp = float(np.mean(amplitudes))
        tau_range = (min(taus), max(taus))

        # Coherence: how tightly clustered
        if len(vecs) > 1:
            centroid = np.mean(vecs, axis=0)
            dists = [float(np.linalg.norm(v - centroid)) for v in vecs]
            coherence = float(np.exp(-np.mean(dists) / 2.0))
        else:
            coherence = 1.0

        confidence = float(np.mean([r.confidence for r in roles]))

        return Clause(
            clause_type=ClauseType.MAIN,  # refined later
            roles=roles,
            amplitude=mean_amp,
            tau_range=tau_range,
            coherence=coherence,
            confidence=confidence,
        )

    def _assign_clause_types(self, clauses: List[Clause]) -> None:
        """Determine clause types based on inter-clause geometric relationships.

        The highest-amplitude clause is the main clause.  Others are typed
        based on their geometric relationship to the main clause.
        """
        if not clauses:
            return

        # The highest-amplitude clause is the main clause
        main_idx = max(range(len(clauses)), key=lambda i: clauses[i].amplitude)
        clauses[main_idx].clause_type = ClauseType.MAIN

        for i, clause in enumerate(clauses):
            if i == main_idx:
                continue

            # Determine relationship type to the main clause
            rel_type = self._classify_relationship(clauses[main_idx], clause)
            clause.clause_type = rel_type
            clause.subordinate_to = main_idx

    def _classify_relationship(self, main: Clause, other: Clause) -> ClauseType:
        """Classify the relationship between two clauses geometrically.

        Uses causal fiber direction, logical fiber features, amplitude
        ratio, and τ gap to determine clause type.
        """
        # Compute inter-clause causal direction
        main_centroid = self._clause_centroid(main)
        other_centroid = self._clause_centroid(other)

        causal_dir = self.syntax.causal_direction(main_centroid, other_centroid)

        # Logical fiber analysis
        main_logical = main_centroid[_LOGICAL_START:_LOGICAL_END]
        other_logical = other_centroid[_LOGICAL_START:_LOGICAL_END]
        logical_diff = float(np.linalg.norm(main_logical - other_logical))

        # τ gap
        tau_gap = abs(other.tau_range[0] - main.tau_range[1])

        # Contrast signal: difference in probabilistic fiber
        main_prob = main_centroid[_PROB_START:_PROB_END]
        other_prob = other_centroid[_PROB_START:_PROB_END]
        contrast = float(np.linalg.norm(main_prob - other_prob))

        # ── Decision tree ─────────────────────────────────────────────
        # Strong causal direction → CAUSAL clause
        if abs(causal_dir) > self.causal_threshold:
            return ClauseType.CAUSAL

        # High contrast signal → CONTRASTIVE clause
        if contrast > self.contrast_threshold:
            return ClauseType.CONTRASTIVE

        # Significant τ gap → TEMPORAL clause
        if tau_gap > self.tau_gap_threshold:
            return ClauseType.TEMPORAL

        # High logical difference → CONDITIONAL clause
        if logical_diff > 0.5:
            return ClauseType.CONDITIONAL

        # Similar amplitude → ADDITIVE (coordinate) clause
        amp_ratio = min(main.amplitude, other.amplitude) / (max(main.amplitude, other.amplitude) + 1e-12)
        if amp_ratio > 0.7:
            return ClauseType.ADDITIVE

        # Default: subordinate
        return ClauseType.SUBORDINATE

    def _clause_centroid(self, clause: Clause) -> np.ndarray:
        """Amplitude-weighted centroid of a clause's role vectors."""
        if not clause.roles:
            return np.zeros(104)
        vecs = np.stack([r.vector for r in clause.roles])
        amps = np.array([r.amplitude for r in clause.roles])
        total = amps.sum() + 1e-12
        return (vecs.T @ amps / total)

    def _select_connectives(self, clauses: List[Clause]) -> List[str]:
        """Select connective words/phrases between clauses.

        Returns a list of n-1 connectives for n clauses.
        """
        connectives: List[str] = []

        for i in range(1, len(clauses)):
            clause = clauses[i]
            conn = self._connective_for_type(clause.clause_type, clause)
            connectives.append(conn)

        return connectives

    def _connective_for_type(self, ctype: ClauseType, clause: Clause) -> str:
        """Select the most appropriate connective for a clause type.

        The selection is deterministic based on the clause's geometric
        properties (τ range, amplitude) to avoid random variation.
        """
        # Use a hash of the clause's tau range for deterministic variety
        variety_key = int(abs(hash(
            f"{clause.tau_range[0]:.3f}{clause.tau_range[1]:.3f}"
        ))) % 3

        connective_map = {
            ClauseType.CAUSAL: ["because", "since", "as a result of"],
            ClauseType.CONTRASTIVE: ["although", "while", "whereas"],
            ClauseType.TEMPORAL: ["when", "after", "before"],
            ClauseType.CONDITIONAL: ["if", "provided that", "given that"],
            ClauseType.ADDITIVE: ["and", "furthermore", "additionally"],
            ClauseType.RELATIVE: ["which", "that", "where"],
            ClauseType.SUBORDINATE: ["whereby", "such that", "in which"],
            ClauseType.MAIN: ["moreover", "also", "in addition"],
        }

        options = connective_map.get(ctype, ["and"])
        return options[variety_key % len(options)]

    def _compute_complexity(self, clauses: List[Clause]) -> int:
        """Compute sentence complexity from clause structure.

        1 = simple (one clause)
        2 = compound (two main clauses with coordinating conjunction)
        3 = complex (main + subordinate)
        """
        if len(clauses) <= 1:
            return 1

        main_count = sum(1 for c in clauses if c.is_main)
        sub_count = sum(1 for c in clauses if not c.is_main)

        if main_count >= 2:
            return 2  # compound
        if sub_count >= 1:
            return 3  # complex
        return 1
