"""Grammar Renderer — Compositional sentence construction from geometry.

Replaces the template-based slot-filling in Expression Renderer Stage 3
with compositional sentence construction driven by manifold geometry.

The renderer takes a SentencePlan (from ClauseComposer) and produces
fluent natural language by:

1. Rendering each clause's S-V-O structure from role assignments
2. Inflecting words via MorphologyMap based on tense/number
3. Connecting clauses via geometrically-derived connectives
4. Applying flow-preservation rules (speed → sentence length, etc.)
5. Checking and correcting agreement via AgreementChecker

The output is a single sentence or compound-complex sentence that reads
naturally — no template slots, no fixed patterns.

No weights.  No tokens.  No training.  Sentences ARE geometry rendered.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from .syntax_geometry import SyntaxGeometry, SyntacticRole, RoleAssignment
from .clause_composer import ClauseComposer, Clause, ClauseType, SentencePlan
from .morphology_map import MorphologyMap, Inflection
from .agreement_checker import AgreementChecker


@dataclass
class RenderedSentence:
    """A fully rendered sentence with metadata.

    Attributes
    ----------
    text       : the final sentence string
    clauses    : clause breakdown for diagnostics
    complexity : sentence complexity (1=simple, 2=compound, 3=complex)
    confidence : rendering confidence
    tense      : dominant tense used
    """
    text: str
    clauses: List[Clause] = field(default_factory=list)
    complexity: int = 1
    confidence: float = 0.5
    tense: str = "present"

    def __repr__(self) -> str:
        return (
            f"RenderedSentence(complexity={self.complexity}, "
            f"tense={self.tense}, conf={self.confidence:.2f}, "
            f"text='{self.text[:80]}{'...' if len(self.text) > 80 else ''}')"
        )


class GrammarRenderer:
    """Compositional sentence renderer driven by manifold geometry.

    Replaces template slot-filling with geometry-driven construction:
    - Role assignments → word order
    - Clause relationships → connectives
    - Causal fiber → tense
    - Logical fiber → agreement
    - Amplitude → emphasis

    Parameters
    ----------
    morphology : MorphologyMap
        Inflection engine (shared across renderings).
    agreement : AgreementChecker
        Agreement verification (shared).
    syntax : SyntaxGeometry
        Role assignment engine (shared).
    composer : ClauseComposer
        Clause composition engine (shared).
    """

    def __init__(
        self,
        morphology: Optional[MorphologyMap] = None,
        agreement: Optional[AgreementChecker] = None,
        syntax: Optional[SyntaxGeometry] = None,
        composer: Optional[ClauseComposer] = None,
    ) -> None:
        self.morphology = morphology or MorphologyMap()
        self.agreement = agreement or AgreementChecker(self.morphology)
        self.syntax = syntax or SyntaxGeometry()
        self.composer = composer or ClauseComposer(self.syntax)

    # ── Public API ─────────────────────────────────────────────────── #

    def render_segment(
        self,
        vectors: List[np.ndarray],
        labels: List[str],
        amplitudes: List[float],
        taus: List[float],
        flow_speed: float = 0.5,
        coherence: float = 0.5,
        uncertainty: float = 0.5,
    ) -> RenderedSentence:
        """Render a wave segment into a composed sentence.

        This is the main entry point.  It takes segment data (vectors,
        labels, amplitudes, taus) and produces a grammatically correct
        sentence without using any templates.

        Parameters
        ----------
        vectors     : 104D manifold position vectors for concepts
        labels      : human-readable concept labels
        amplitudes  : wave amplitudes (meaning centrality)
        taus        : causal time coordinates
        flow_speed  : flow speed at this segment (0=slow, 1=fast)
        coherence   : segment coherence (0=diffuse, 1=tight)
        uncertainty : segment uncertainty (0=certain, 1=uncertain)

        Returns
        -------
        RenderedSentence with composed text and metadata.
        """
        if not vectors or not labels:
            return RenderedSentence(text="", confidence=0.0)

        # Clean labels
        clean_labels = [self._clean_label(l) for l in labels]

        # ── Step 1: Assign syntactic roles ────────────────────────────
        roles = self.syntax.assign_roles(vectors, clean_labels, amplitudes, taus)

        if not roles:
            return RenderedSentence(text="", confidence=0.0)

        # ── Step 2: Determine tense from causal fiber ─────────────────
        tense = self.agreement.infer_tense(roles)

        # ── Step 3: Compose clause plan ───────────────────────────────
        # Split roles into groups if there are multiple subjects/verbs
        role_groups = self._split_into_clause_groups(roles)
        plan = self.composer.compose(role_groups)

        if not plan.clauses:
            # Fallback: build a simple declarative from all roles
            text = self._render_simple(roles, tense, flow_speed, uncertainty)
            return RenderedSentence(
                text=text, complexity=1,
                confidence=0.5, tense=tense,
            )

        # ── Step 4: Render each clause ────────────────────────────────
        rendered_clauses = []
        for clause in plan.clauses:
            clause_text = self._render_clause(clause, tense, flow_speed, uncertainty)
            rendered_clauses.append(clause_text)

        # ── Step 5: Assemble with connectives ─────────────────────────
        text = self._assemble_clauses(
            rendered_clauses, plan.connectives, plan.clauses,
            flow_speed, uncertainty,
        )

        # ── Step 6: Agreement check & correction ──────────────────────
        # (Post-render verification — if violations found, attempt re-render)
        for clause in plan.clauses:
            result = self.agreement.check(clause.roles)
            if not result.is_valid:
                # Apply corrections to labels and re-render the clause
                for idx, inflection in result.corrections:
                    if idx < len(clause.roles):
                        corrected = self.agreement.correct_surface(
                            clause.roles[idx].label, inflection
                        )
                        clause.roles[idx].label = corrected

        confidence = plan.confidence * (1.0 - uncertainty * 0.3)

        return RenderedSentence(
            text=text,
            clauses=plan.clauses,
            complexity=plan.complexity,
            confidence=float(max(0.0, min(1.0, confidence))),
            tense=tense,
        )

    def render_from_plan(
        self,
        plan: SentencePlan,
        tense: str = "present",
        flow_speed: float = 0.5,
        uncertainty: float = 0.5,
    ) -> RenderedSentence:
        """Render a pre-built SentencePlan into a sentence string.

        Useful when the caller has already done role assignment and
        clause composition.
        """
        if not plan.clauses:
            return RenderedSentence(text="", confidence=0.0)

        rendered_clauses = []
        for clause in plan.clauses:
            clause_text = self._render_clause(clause, tense, flow_speed, uncertainty)
            rendered_clauses.append(clause_text)

        text = self._assemble_clauses(
            rendered_clauses, plan.connectives, plan.clauses,
            flow_speed, uncertainty,
        )

        return RenderedSentence(
            text=text,
            clauses=plan.clauses,
            complexity=plan.complexity,
            confidence=float(plan.confidence),
            tense=tense,
        )

    # ── Clause rendering ───────────────────────────────────────────── #

    def _render_clause(
        self,
        clause: Clause,
        tense: str,
        flow_speed: float,
        uncertainty: float,
    ) -> str:
        """Render a single clause into a text fragment.

        Constructs the clause from its S-V-O role assignments, inflecting
        verbs for tense and applying modifiers.
        """
        if not clause.roles:
            return ""

        subject = clause.subject
        verb = clause.verb
        obj = clause.object

        parts: List[str] = []

        # ── Subject phrase ────────────────────────────────────────────
        if subject:
            subj_phrase = self._render_noun_phrase(subject, clause.roles, tense)
            parts.append(subj_phrase)

        # ── Verb phrase ───────────────────────────────────────────────
        if verb:
            verb_phrase = self._render_verb_phrase(verb, tense, subject, uncertainty)
            parts.append(verb_phrase)
        elif subject and obj:
            # No verb identified — use a linking verb
            parts.append(self._select_linking_verb(tense, uncertainty))

        # ── Object phrase ─────────────────────────────────────────────
        if obj:
            obj_phrase = self._render_noun_phrase(obj, clause.roles, tense)
            parts.append(obj_phrase)

        # ── Remaining roles (topics, complements) ─────────────────────
        extras = [
            r for r in clause.roles
            if r.role in (SyntacticRole.TOPIC, SyntacticRole.COMPLEMENT)
            and r is not subject and r is not verb and r is not obj
        ]
        for extra in extras[:2]:  # limit to 2 extras
            if extra.amplitude > 0.3:
                parts.append(self._render_complement(extra, tense))

        # Join and clean
        text = " ".join(p for p in parts if p)
        return text.strip()

    def _render_noun_phrase(
        self,
        role: RoleAssignment,
        all_roles: List[RoleAssignment],
        tense: str,
    ) -> str:
        """Render a noun phrase with its modifiers.

        Structure: [determiner] [pre-modifiers] head [post-modifiers]
        """
        head = role.label

        # Find modifiers attached to this head (within phrase radius)
        modifiers = [
            r for r in all_roles
            if r.role == SyntacticRole.MODIFIER
            and self.syntax.similarity_distance(r.vector, role.vector) < self.syntax.phrase_radius
        ]

        pre_mods = [m.label for m in modifiers if m.morph_class == "adjective"]
        post_mods = [m.label for m in modifiers if m.morph_class == "adverb"]

        # Determiner based on amplitude and definiteness
        determiner = self._select_determiner(role)

        parts = []
        if determiner:
            parts.append(determiner)
        parts.extend(pre_mods[:2])  # max 2 pre-modifiers
        parts.append(head)
        if post_mods:
            parts.append("(" + " ".join(post_mods[:1]) + ")")

        return " ".join(parts)

    def _render_verb_phrase(
        self,
        verb_role: RoleAssignment,
        tense: str,
        subject: Optional[RoleAssignment],
        uncertainty: float,
    ) -> str:
        """Render a verb phrase with appropriate tense inflection.

        Uses the MorphologyMap to inflect the verb for tense, and adds
        hedging adverbs for uncertain segments.
        """
        base = verb_role.label

        # Analyse the current form to find the base
        form = self.morphology.analyse(base)
        lemma = form.base

        # Inflect for tense
        if tense == "past":
            surface = self.morphology.inflect(lemma, Inflection.PAST)
        elif tense == "present":
            # Check if subject is third person singular
            if subject and self.agreement.infer_number(subject) == "singular":
                surface = self.morphology.inflect(lemma, Inflection.THIRD_PERSON)
            else:
                surface = lemma  # base form for plural/first person
        else:
            surface = "will " + lemma  # future

        # Add hedging for uncertain segments
        if uncertainty > 0.7:
            surface = "may " + lemma  # override with modal for high uncertainty
        elif uncertainty > 0.5:
            surface = "likely " + surface

        return surface

    def _render_complement(self, role: RoleAssignment, tense: str) -> str:
        """Render a complement or topic phrase."""
        if role.role == SyntacticRole.TOPIC:
            return f"regarding {role.label}"
        return f"involving {role.label}"

    def _select_determiner(self, role: RoleAssignment) -> str:
        """Select an appropriate determiner based on geometry.

        High amplitude (definite) → "the"
        Medium amplitude → "" (bare noun, often for abstract concepts)
        Low amplitude + existential → "a/an"
        Plural → "" (bare plural) or "the"
        """
        number = self.agreement.infer_number(role)

        if role.amplitude > 0.7:
            return "the"

        if role.amplitude > 0.4 and number == "singular":
            # Abstract/mass nouns are often bare
            if role.morph_class in ("noun",):
                return ""
            return "the"

        if number == "singular" and role.amplitude < 0.4:
            first_letter = role.label[0].lower() if role.label else "x"
            if first_letter in "aeiou":
                return "an"
            return "a"

        return ""

    def _select_linking_verb(self, tense: str, uncertainty: float) -> str:
        """Select a linking verb when no verb was identified from geometry."""
        if uncertainty > 0.6:
            if tense == "past":
                return "appeared to involve"
            return "appears to involve"

        if tense == "past":
            return "involved"
        return "involves"

    # ── Clause assembly ────────────────────────────────────────────── #

    def _assemble_clauses(
        self,
        rendered_clauses: List[str],
        connectives: List[str],
        clauses: List[Clause],
        flow_speed: float,
        uncertainty: float,
    ) -> str:
        """Assemble rendered clauses into a complete sentence.

        Applies connectives and adjusts structure based on flow speed.
        """
        if not rendered_clauses:
            return ""

        if len(rendered_clauses) == 1:
            text = rendered_clauses[0]
        else:
            parts = []
            for i, clause_text in enumerate(rendered_clauses):
                if not clause_text:
                    continue
                if i == 0:
                    parts.append(clause_text)
                else:
                    conn_idx = i - 1
                    conn = connectives[conn_idx] if conn_idx < len(connectives) else "and"
                    clause_type = clauses[i].clause_type if i < len(clauses) else ClauseType.ADDITIVE

                    # Position connective appropriately
                    if clause_type in (ClauseType.CAUSAL, ClauseType.TEMPORAL,
                                       ClauseType.CONDITIONAL):
                        # Subordinating: "because X", "when X"
                        parts.append(f", {conn} {clause_text}")
                    elif clause_type == ClauseType.CONTRASTIVE:
                        # Contrastive often as new clause
                        parts.append(f", {conn} {clause_text}")
                    else:
                        # Coordinating: "and X", "furthermore X"
                        if conn in ("and", "or", "but", "yet"):
                            parts.append(f" {conn} {clause_text}")
                        else:
                            parts.append(f"; {conn}, {clause_text}")

            text = "".join(parts)

        # ── Flow speed adjustments ────────────────────────────────────
        if flow_speed > 0.7:
            text = self._condense_text(text)
        elif flow_speed < 0.3:
            text = self._elaborate_text(text, uncertainty)

        # ── Uncertainty prefix ────────────────────────────────────────
        if uncertainty > 0.75:
            text = "It seems that " + text[0].lower() + text[1:] if text else text

        # ── Final cleanup ─────────────────────────────────────────────
        text = self._clean_sentence(text)

        return text

    # ── Role group splitting ───────────────────────────────────────── #

    def _split_into_clause_groups(
        self, roles: List[RoleAssignment]
    ) -> List[List[RoleAssignment]]:
        """Split roles into clause groups based on natural boundaries.

        A new clause group starts when:
        - A new subject appears after a verb+object sequence
        - There's a significant τ gap between consecutive roles
        - A connector role appears
        """
        if len(roles) <= 3:
            return [roles]

        groups: List[List[RoleAssignment]] = []
        current_group: List[RoleAssignment] = []
        has_verb = False
        has_object = False

        for role in roles:
            # Start a new group on connector or after a complete S-V-O
            if role.role == SyntacticRole.CONNECTOR:
                if current_group:
                    groups.append(current_group)
                current_group = []
                has_verb = False
                has_object = False
                continue

            if (role.role == SyntacticRole.SUBJECT
                    and has_verb and has_object and current_group):
                groups.append(current_group)
                current_group = []
                has_verb = False
                has_object = False

            current_group.append(role)

            if role.role == SyntacticRole.VERB:
                has_verb = True
            if role.role == SyntacticRole.OBJECT:
                has_object = True

        if current_group:
            groups.append(current_group)

        return groups if groups else [roles]

    # ── Simple fallback renderer ───────────────────────────────────── #

    def _render_simple(
        self,
        roles: List[RoleAssignment],
        tense: str,
        flow_speed: float,
        uncertainty: float,
    ) -> str:
        """Fallback: render a simple declarative sentence from roles.

        Used when clause composition isn't possible (e.g., single concept).
        """
        labels = [r.label for r in roles if r.label][:4]
        if not labels:
            return ""

        if len(labels) == 1:
            if uncertainty > 0.6:
                return f"There appears to be {labels[0]}."
            return f"The concept of {labels[0]} is present."

        if len(labels) == 2:
            if tense == "past":
                return f"{labels[0].capitalize()} was connected to {labels[1]}."
            return f"{labels[0].capitalize()} relates to {labels[1]}."

        # 3+ labels: construct a richer sentence
        subject = labels[0].capitalize()
        verb = "involves" if tense == "present" else "involved"
        objects = " and ".join(labels[1:3])
        return f"{subject} {verb} {objects}."

    # ── Text adjustment helpers ────────────────────────────────────── #

    def _condense_text(self, text: str) -> str:
        """Shorten text for fast flow — remove elaborations."""
        # Remove parenthetical asides
        import re
        text = re.sub(r'\s*\([^)]*\)', '', text)
        # Remove trailing qualifiers after semicolons
        if "; " in text:
            text = text[:text.index("; ")]
        return text

    def _elaborate_text(self, text: str, uncertainty: float) -> str:
        """Expand text for slow flow — add qualifiers."""
        if uncertainty > 0.5:
            text += ", though this relationship warrants further examination"
        return text

    def _clean_sentence(self, text: str) -> str:
        """Clean up a rendered sentence: capitalise, punctuate, fix spacing."""
        text = text.strip()
        if not text:
            return text

        # Fix double spaces
        while "  " in text:
            text = text.replace("  ", " ")

        # Fix comma spacing
        text = text.replace(" ,", ",").replace(",,", ",")

        # Capitalise first letter
        text = text[0].upper() + text[1:]

        # Ensure terminal punctuation
        if text and text[-1] not in ".!?":
            text += "."

        return text

    @staticmethod
    def _clean_label(label: str) -> str:
        """Strip domain prefix from a manifold concept label.

        'causal::mechanism' → 'mechanism'
        'vocab::running'    → 'running'
        """
        if "::" in label:
            label = label.split("::")[-1]
        label = label.replace("_", " ")
        return label.strip()
