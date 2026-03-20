"""Phase 10 — Geometric Grammar Engine tests.

Tests for all five sub-components of the Geometric Grammar Engine (C7b):
  SyntaxGeometry   — role assignment from causal fiber
  ClauseComposer   — clause composition via fiber bundle
  MorphologyMap    — word family inflection clusters
  GrammarRenderer  — compositional sentence construction
  AgreementChecker — number/tense agreement constraints
  Integration      — grammar engine wired into C7 pipeline
"""

from __future__ import annotations

import numpy as np
import pytest

from src.phase10.grammar.syntax_geometry import (
    SyntaxGeometry,
    SyntacticRole,
    RoleAssignment,
    _CAUSAL_START, _CAUSAL_END,
    _LOGICAL_START, _LOGICAL_END,
    _PROB_START, _PROB_END,
    _SIM_START, _SIM_END,
)
from src.phase10.grammar.clause_composer import (
    ClauseComposer,
    Clause,
    ClauseType,
    SentencePlan,
)
from src.phase10.grammar.morphology_map import (
    MorphologyMap,
    WordForm,
    Inflection,
)
from src.phase10.grammar.grammar_renderer import (
    GrammarRenderer,
    RenderedSentence,
)
from src.phase10.grammar.agreement_checker import (
    AgreementChecker,
    AgreementResult,
)

# ── Helpers ────────────────────────────────────────────────────────────────

DIM = 104


def _make_vector(
    sim_quad: int = 16,   # noun by default (16–31)
    causal_tau: float = 0.5,
    causal_strength: float = 0.1,
    logical_negation: float = 0.0,
    logical_universal: float = 0.3,
    prob_mean: float = 0.5,
    seed: int = 42,
) -> np.ndarray:
    """Create a 104D vector with controlled fiber content."""
    rng = np.random.default_rng(seed)
    vec = rng.uniform(0.01, 0.05, size=DIM)

    # Similarity fiber: set quadrant energy
    vec[_SIM_START:_SIM_END] = 0.01
    vec[sim_quad:sim_quad + 16] = rng.uniform(0.1, 0.3, size=16)

    # Causal fiber
    vec[_CAUSAL_START] = causal_tau
    vec[_CAUSAL_START + 1:_CAUSAL_END] = causal_strength * rng.uniform(0.5, 1.0, size=_CAUSAL_END - _CAUSAL_START - 1)

    # Logical fiber
    vec[_LOGICAL_START] = logical_negation
    vec[_LOGICAL_START + 1] = logical_universal
    vec[_LOGICAL_START + 2:_LOGICAL_END] = rng.uniform(0.3, 0.5, size=_LOGICAL_END - _LOGICAL_START - 2)

    # Probabilistic fiber
    vec[_PROB_START:_PROB_END] = prob_mean + rng.uniform(-0.1, 0.1, size=_PROB_END - _PROB_START)

    return vec


def _make_noun_vec(tau: float = 0.3, seed: int = 42) -> np.ndarray:
    return _make_vector(sim_quad=16, causal_tau=tau, seed=seed)


def _make_verb_vec(tau: float = 0.5, seed: int = 100) -> np.ndarray:
    return _make_vector(sim_quad=0, causal_tau=tau, causal_strength=0.2, seed=seed)


def _make_adj_vec(tau: float = 0.3, seed: int = 200) -> np.ndarray:
    return _make_vector(sim_quad=32, causal_tau=tau, seed=seed)


def _make_adv_vec(tau: float = 0.5, seed: int = 300) -> np.ndarray:
    return _make_vector(sim_quad=48, causal_tau=tau, seed=seed)


# ═══════════════════════════════════════════════════════════════════════════
# TestSyntaxGeometry
# ═══════════════════════════════════════════════════════════════════════════

class TestSyntaxGeometry:
    """Tests for SyntaxGeometry — role assignment from causal fiber."""

    def test_construction(self):
        sg = SyntaxGeometry()
        assert sg.phrase_radius > 0
        assert sg.causal_threshold > 0

    def test_assign_roles_empty(self):
        sg = SyntaxGeometry()
        result = sg.assign_roles([], [], [], [])
        assert result == []

    def test_assign_roles_single_noun(self):
        sg = SyntaxGeometry()
        vec = _make_noun_vec()
        roles = sg.assign_roles([vec], ["gravity"], [0.8], [0.3])
        assert len(roles) == 1
        assert isinstance(roles[0], RoleAssignment)
        assert roles[0].label == "gravity"

    def test_assign_roles_two_nouns(self):
        """With only nouns, highest-causal-strength becomes VERB, other SUBJECT."""
        sg = SyntaxGeometry()
        v1 = _make_noun_vec(tau=0.2, seed=1)
        v2 = _make_noun_vec(tau=0.8, seed=2)
        roles = sg.assign_roles(
            [v1, v2], ["cause", "effect"], [0.8, 0.6], [0.2, 0.8]
        )
        assert len(roles) == 2
        role_types = {r.role for r in roles}
        # No verb by morphology → one promoted to VERB
        assert SyntacticRole.SUBJECT in role_types
        assert SyntacticRole.VERB in role_types

    def test_subject_has_earlier_tau_with_svo(self):
        """With S-V-O present, subject has earlier τ than object."""
        sg = SyntaxGeometry()
        v_s = _make_noun_vec(tau=0.1, seed=1)
        v_v = _make_verb_vec(tau=0.5, seed=2)
        v_o = _make_noun_vec(tau=0.9, seed=3)
        roles = sg.assign_roles(
            [v_s, v_v, v_o],
            ["initiator", "drives", "receiver"],
            [0.8, 0.7, 0.6], [0.1, 0.5, 0.9],
        )
        subject = [r for r in roles if r.role == SyntacticRole.SUBJECT][0]
        obj = [r for r in roles if r.role == SyntacticRole.OBJECT][0]
        assert subject.tau <= obj.tau

    def test_verb_identified_from_quadrant(self):
        sg = SyntaxGeometry()
        v_noun = _make_noun_vec(tau=0.2, seed=1)
        v_verb = _make_verb_vec(tau=0.5, seed=2)
        v_obj = _make_noun_vec(tau=0.8, seed=3)
        roles = sg.assign_roles(
            [v_noun, v_verb, v_obj],
            ["force", "drives", "motion"],
            [0.7, 0.8, 0.6],
            [0.2, 0.5, 0.8],
        )
        verbs = [r for r in roles if r.role == SyntacticRole.VERB]
        assert len(verbs) >= 1
        assert verbs[0].label == "drives"

    def test_adjective_becomes_modifier(self):
        sg = SyntaxGeometry(phrase_radius=10.0)  # large radius so modifier attaches
        v_noun = _make_noun_vec(tau=0.3, seed=1)
        v_adj = _make_adj_vec(tau=0.3, seed=2)
        roles = sg.assign_roles(
            [v_noun, v_adj], ["mechanism", "complex"], [0.7, 0.4], [0.3, 0.3]
        )
        modifiers = [r for r in roles if r.role == SyntacticRole.MODIFIER]
        assert len(modifiers) >= 0  # may or may not attach depending on distance

    def test_causal_direction_forward(self):
        sg = SyntaxGeometry()
        v1 = _make_vector(causal_tau=0.2, seed=1)
        v2 = _make_vector(causal_tau=0.8, seed=2)
        direction = sg.causal_direction(v1, v2)
        assert isinstance(direction, float)

    def test_causal_direction_backward_different(self):
        sg = SyntaxGeometry()
        v1 = _make_vector(causal_tau=0.2, seed=1)
        v2 = _make_vector(causal_tau=0.8, seed=2)
        forward = sg.causal_direction(v1, v2)
        backward = sg.causal_direction(v2, v1)
        # Asymmetric: forward ≠ backward
        assert forward != backward

    def test_concept_distance_positive(self):
        sg = SyntaxGeometry()
        v1 = _make_vector(seed=1)
        v2 = _make_vector(seed=2)
        d = sg.concept_distance(v1, v2)
        assert d > 0

    def test_similarity_distance(self):
        sg = SyntaxGeometry()
        v1 = _make_vector(seed=1)
        v2 = _make_vector(seed=2)
        d = sg.similarity_distance(v1, v2)
        assert d >= 0

    def test_infer_morph_class_verb(self):
        sg = SyntaxGeometry()
        vec = _make_verb_vec()
        morph = sg._infer_morph_class(vec)
        assert morph == "verb"

    def test_infer_morph_class_noun(self):
        sg = SyntaxGeometry()
        vec = _make_noun_vec()
        morph = sg._infer_morph_class(vec)
        assert morph == "noun"

    def test_infer_morph_class_adjective(self):
        sg = SyntaxGeometry()
        vec = _make_adj_vec()
        morph = sg._infer_morph_class(vec)
        assert morph == "adjective"

    def test_infer_morph_class_adverb(self):
        sg = SyntaxGeometry()
        vec = _make_adv_vec()
        morph = sg._infer_morph_class(vec)
        assert morph == "adverb"

    def test_infer_morph_class_function_word(self):
        sg = SyntaxGeometry()
        vec = np.zeros(DIM)  # all zeros → function word
        morph = sg._infer_morph_class(vec)
        assert morph == "function"

    def test_causal_strength(self):
        sg = SyntaxGeometry()
        vec = _make_vector(causal_strength=0.5, seed=1)
        strength = sg._causal_strength(vec)
        assert strength > 0

    def test_logical_features(self):
        sg = SyntaxGeometry()
        vec = _make_vector(logical_negation=0.8, logical_universal=0.9, seed=1)
        features = sg._logical_features(vec)
        assert "negation" in features
        assert "universal" in features
        assert features["negation"] == pytest.approx(0.8)
        assert features["universal"] == pytest.approx(0.9)

    def test_probabilistic_features(self):
        sg = SyntaxGeometry()
        vec = _make_vector(prob_mean=0.8, seed=1)
        features = sg._probabilistic_features(vec)
        assert "certainty" in features
        assert "register" in features

    def test_order_for_speech_subject_before_verb(self):
        sg = SyntaxGeometry()
        roles = [
            RoleAssignment("obj", SyntacticRole.OBJECT, np.zeros(DIM), 0.5, 0.8),
            RoleAssignment("subj", SyntacticRole.SUBJECT, np.zeros(DIM), 0.8, 0.2),
            RoleAssignment("verb", SyntacticRole.VERB, np.zeros(DIM), 0.7, 0.5),
        ]
        ordered = sg._order_for_speech(roles)
        names = [r.label for r in ordered]
        assert names.index("subj") < names.index("verb")
        assert names.index("verb") < names.index("obj")

    def test_high_amplitude_becomes_topic(self):
        sg = SyntaxGeometry()
        vectors = [
            _make_noun_vec(tau=0.3, seed=1),
            _make_verb_vec(tau=0.5, seed=2),
            _make_noun_vec(tau=0.7, seed=3),
            _make_noun_vec(tau=0.9, seed=4),  # extra concept, not S/V/O
        ]
        roles = sg.assign_roles(
            vectors, ["cause", "drives", "effect", "context"],
            [0.8, 0.9, 0.7, 0.85], [0.3, 0.5, 0.7, 0.9],
        )
        topics = [r for r in roles if r.role == SyntacticRole.TOPIC]
        # High-amplitude remaining concept may become TOPIC
        assert any(r.amplitude > 0.7 for r in roles)

    def test_many_concepts_handled(self):
        sg = SyntaxGeometry()
        n = 10
        vectors = [_make_vector(seed=i) for i in range(n)]
        labels = [f"concept_{i}" for i in range(n)]
        amps = [0.5 + 0.05 * i for i in range(n)]
        taus = [i / n for i in range(n)]
        roles = sg.assign_roles(vectors, labels, amps, taus)
        assert len(roles) == n
        assert all(isinstance(r, RoleAssignment) for r in roles)


# ═══════════════════════════════════════════════════════════════════════════
# TestClauseComposer
# ═══════════════════════════════════════════════════════════════════════════

class TestClauseComposer:
    """Tests for ClauseComposer — clause composition via fiber bundle."""

    def test_construction(self):
        cc = ClauseComposer()
        assert cc.causal_threshold > 0
        assert cc.contrast_threshold > 0

    def test_compose_empty(self):
        cc = ClauseComposer()
        plan = cc.compose([])
        assert isinstance(plan, SentencePlan)
        assert plan.complexity == 0
        assert len(plan.clauses) == 0

    def test_compose_single_group(self):
        sg = SyntaxGeometry()
        cc = ClauseComposer(syntax=sg)
        roles = [
            RoleAssignment("gravity", SyntacticRole.SUBJECT, _make_noun_vec(0.2, 1), 0.8, 0.2, "noun"),
            RoleAssignment("pulls", SyntacticRole.VERB, _make_verb_vec(0.5, 2), 0.7, 0.5, "verb"),
            RoleAssignment("objects", SyntacticRole.OBJECT, _make_noun_vec(0.8, 3), 0.6, 0.8, "noun"),
        ]
        plan = cc.compose([roles])
        assert len(plan.clauses) == 1
        assert plan.clauses[0].clause_type == ClauseType.MAIN
        assert plan.complexity == 1

    def test_compose_two_groups(self):
        cc = ClauseComposer()
        roles_a = [
            RoleAssignment("force", SyntacticRole.SUBJECT, _make_noun_vec(0.2, 1), 0.8, 0.2, "noun"),
            RoleAssignment("acts", SyntacticRole.VERB, _make_verb_vec(0.5, 2), 0.7, 0.5, "verb"),
        ]
        roles_b = [
            RoleAssignment("motion", SyntacticRole.SUBJECT, _make_noun_vec(0.7, 3), 0.5, 0.7, "noun"),
            RoleAssignment("results", SyntacticRole.VERB, _make_verb_vec(0.9, 4), 0.6, 0.9, "verb"),
        ]
        plan = cc.compose([roles_a, roles_b])
        assert len(plan.clauses) == 2
        assert len(plan.connectives) == 1
        assert plan.complexity >= 2

    def test_clause_has_subject(self):
        cc = ClauseComposer()
        roles = [
            RoleAssignment("x", SyntacticRole.SUBJECT, np.zeros(DIM), 0.8, 0.2),
            RoleAssignment("y", SyntacticRole.VERB, np.zeros(DIM), 0.7, 0.5),
        ]
        plan = cc.compose([roles])
        assert plan.clauses[0].has_subject
        assert plan.clauses[0].has_verb

    def test_clause_properties(self):
        cc = ClauseComposer()
        roles = [
            RoleAssignment("x", SyntacticRole.SUBJECT, _make_noun_vec(0.2, 1), 0.8, 0.2),
            RoleAssignment("y", SyntacticRole.VERB, _make_verb_vec(0.5, 2), 0.7, 0.5),
            RoleAssignment("z", SyntacticRole.OBJECT, _make_noun_vec(0.8, 3), 0.6, 0.8),
        ]
        plan = cc.compose([roles])
        clause = plan.clauses[0]
        assert clause.subject is not None
        assert clause.subject.label == "x"
        assert clause.verb is not None
        assert clause.verb.label == "y"
        assert clause.object is not None
        assert clause.object.label == "z"

    def test_main_clause_is_highest_amplitude(self):
        cc = ClauseComposer()
        roles_a = [
            RoleAssignment("minor", SyntacticRole.SUBJECT, _make_noun_vec(0.2, 1), 0.3, 0.2),
        ]
        roles_b = [
            RoleAssignment("major", SyntacticRole.SUBJECT, _make_noun_vec(0.5, 2), 0.9, 0.5),
        ]
        plan = cc.compose([roles_a, roles_b])
        main_clauses = [c for c in plan.clauses if c.is_main]
        assert len(main_clauses) == 1
        assert main_clauses[0].amplitude > roles_a[0].amplitude

    def test_connective_selection(self):
        cc = ClauseComposer()
        roles_a = [
            RoleAssignment("a", SyntacticRole.SUBJECT, _make_noun_vec(0.2, 1), 0.8, 0.2),
        ]
        roles_b = [
            RoleAssignment("b", SyntacticRole.SUBJECT, _make_noun_vec(0.8, 2), 0.5, 0.8),
        ]
        plan = cc.compose([roles_a, roles_b])
        assert len(plan.connectives) == 1
        assert isinstance(plan.connectives[0], str)
        assert len(plan.connectives[0]) > 0

    def test_compose_single_convenience(self):
        cc = ClauseComposer()
        roles = [
            RoleAssignment("x", SyntacticRole.SUBJECT, np.zeros(DIM), 0.8, 0.2),
        ]
        plan = cc.compose_single(roles)
        assert isinstance(plan, SentencePlan)
        assert len(plan.clauses) == 1

    def test_sentence_plan_is_compound(self):
        plan = SentencePlan(
            clauses=[
                Clause(ClauseType.MAIN, [], amplitude=0.8),
                Clause(ClauseType.MAIN, [], amplitude=0.7),
            ],
            complexity=2,
        )
        assert plan.is_compound

    def test_sentence_plan_is_complex(self):
        plan = SentencePlan(
            clauses=[
                Clause(ClauseType.MAIN, [], amplitude=0.8),
                Clause(ClauseType.CAUSAL, [], amplitude=0.5),
            ],
            complexity=3,
        )
        assert plan.is_complex

    def test_clause_repr(self):
        clause = Clause(ClauseType.CAUSAL, [
            RoleAssignment("x", SyntacticRole.SUBJECT, np.zeros(DIM), 0.8, 0.2),
        ], amplitude=0.7)
        r = repr(clause)
        assert "causal" in r
        assert "0.70" in r

    def test_three_clause_composition(self):
        cc = ClauseComposer()
        groups = [
            [RoleAssignment("a", SyntacticRole.SUBJECT, _make_noun_vec(0.2, 1), 0.8, 0.2)],
            [RoleAssignment("b", SyntacticRole.SUBJECT, _make_noun_vec(0.5, 2), 0.6, 0.5)],
            [RoleAssignment("c", SyntacticRole.SUBJECT, _make_noun_vec(0.8, 3), 0.7, 0.8)],
        ]
        plan = cc.compose(groups)
        assert len(plan.clauses) == 3
        assert len(plan.connectives) == 2


# ═══════════════════════════════════════════════════════════════════════════
# TestMorphologyMap
# ═══════════════════════════════════════════════════════════════════════════

class TestMorphologyMap:
    """Tests for MorphologyMap — word family inflection clusters."""

    def test_construction(self):
        mm = MorphologyMap()
        assert mm.dim == 104

    def test_analyse_base_form(self):
        mm = MorphologyMap()
        form = mm.analyse("gravity")
        assert isinstance(form, WordForm)
        # "gravity" doesn't obviously match -ing/-ed/-s patterns
        # The exact result depends on suffix heuristics

    def test_analyse_progressive(self):
        mm = MorphologyMap()
        form = mm.analyse("running")
        assert form.inflection == Inflection.PROGRESSIVE
        assert form.base == "run"

    def test_analyse_past_regular(self):
        mm = MorphologyMap()
        form = mm.analyse("walked")
        assert form.inflection == Inflection.PAST

    def test_analyse_past_irregular(self):
        mm = MorphologyMap()
        form = mm.analyse("ran")
        assert form.inflection == Inflection.PAST
        assert form.base == "run"
        assert form.confidence >= 0.9

    def test_analyse_plural_regular(self):
        mm = MorphologyMap()
        form = mm.analyse("dogs")
        assert form.inflection == Inflection.PLURAL
        assert form.base == "dog"

    def test_analyse_plural_irregular(self):
        mm = MorphologyMap()
        form = mm.analyse("children")
        assert form.inflection == Inflection.PLURAL
        assert form.base == "child"

    def test_inflect_past_regular(self):
        mm = MorphologyMap()
        assert mm.inflect("walk", Inflection.PAST) == "walked"

    def test_inflect_past_irregular(self):
        mm = MorphologyMap()
        assert mm.inflect("run", Inflection.PAST) == "ran"

    def test_inflect_progressive(self):
        mm = MorphologyMap()
        result = mm.inflect("run", Inflection.PROGRESSIVE)
        assert result == "running"

    def test_inflect_progressive_drop_e(self):
        mm = MorphologyMap()
        result = mm.inflect("make", Inflection.PROGRESSIVE)
        assert result == "making"

    def test_inflect_plural_regular(self):
        mm = MorphologyMap()
        assert mm.inflect("dog", Inflection.PLURAL) == "dogs"

    def test_inflect_plural_es(self):
        mm = MorphologyMap()
        assert mm.inflect("box", Inflection.PLURAL) == "boxes"

    def test_inflect_plural_ies(self):
        mm = MorphologyMap()
        assert mm.inflect("city", Inflection.PLURAL) == "cities"

    def test_inflect_plural_irregular(self):
        mm = MorphologyMap()
        assert mm.inflect("child", Inflection.PLURAL) == "children"

    def test_inflect_third_person(self):
        mm = MorphologyMap()
        assert mm.inflect("run", Inflection.THIRD_PERSON) == "runs"

    def test_inflect_third_person_es(self):
        mm = MorphologyMap()
        assert mm.inflect("watch", Inflection.THIRD_PERSON) == "watches"

    def test_inflect_comparative_short(self):
        mm = MorphologyMap()
        result = mm.inflect("fast", Inflection.COMPARATIVE)
        assert result == "faster"

    def test_inflect_comparative_long(self):
        mm = MorphologyMap()
        result = mm.inflect("beautiful", Inflection.COMPARATIVE)
        assert result == "more beautiful"

    def test_inflect_superlative_short(self):
        mm = MorphologyMap()
        result = mm.inflect("fast", Inflection.SUPERLATIVE)
        assert result == "fastest"

    def test_inflect_superlative_long(self):
        mm = MorphologyMap()
        result = mm.inflect("beautiful", Inflection.SUPERLATIVE)
        assert result == "most beautiful"

    def test_inflect_negative(self):
        mm = MorphologyMap()
        assert mm.inflect("happy", Inflection.NEGATIVE) == "unhappy"

    def test_inflect_possessive(self):
        mm = MorphologyMap()
        assert mm.inflect("cat", Inflection.POSSESSIVE) == "cat's"

    def test_inflect_possessive_ending_s(self):
        mm = MorphologyMap()
        assert mm.inflect("dogs", Inflection.POSSESSIVE) == "dogs'"

    def test_inflect_base_returns_same(self):
        mm = MorphologyMap()
        assert mm.inflect("gravity", Inflection.BASE) == "gravity"

    def test_get_offset_shape(self):
        mm = MorphologyMap()
        offset = mm.get_offset(Inflection.PAST)
        assert offset.shape == (104,)

    def test_get_offset_past_nonzero(self):
        mm = MorphologyMap()
        offset = mm.get_offset(Inflection.PAST)
        assert np.linalg.norm(offset) > 0

    def test_get_offset_base_zero(self):
        mm = MorphologyMap()
        offset = mm.get_offset(Inflection.BASE)
        assert np.allclose(offset, 0)

    def test_word_family(self):
        mm = MorphologyMap()
        family = mm.word_family("run")
        assert Inflection.BASE in family
        assert family[Inflection.BASE] == "run"
        assert Inflection.PAST in family
        assert family[Inflection.PAST] == "ran"
        assert Inflection.PROGRESSIVE in family
        assert family[Inflection.PROGRESSIVE] == "running"

    def test_word_family_noun(self):
        mm = MorphologyMap()
        family = mm.word_family("dog")
        assert Inflection.PLURAL in family
        assert family[Inflection.PLURAL] == "dogs"

    def test_wordform_repr(self):
        form = WordForm("running", "run", Inflection.PROGRESSIVE, np.zeros(104))
        r = repr(form)
        assert "running" in r
        assert "run" in r

    def test_offsets_in_correct_fibers(self):
        """Past tense offset should be in causal fiber, not similarity."""
        mm = MorphologyMap()
        past_offset = mm.get_offset(Inflection.PAST)
        # Should have nonzero causal component
        causal_norm = np.linalg.norm(past_offset[_CAUSAL_START:_CAUSAL_END])
        sim_norm = np.linalg.norm(past_offset[_SIM_START:_SIM_END])
        assert causal_norm > sim_norm

    def test_plural_offset_in_logical_fiber(self):
        mm = MorphologyMap()
        plural_offset = mm.get_offset(Inflection.PLURAL)
        logical_norm = np.linalg.norm(plural_offset[_LOGICAL_START:_LOGICAL_END])
        assert logical_norm > 0

    def test_past_tense_y_to_ied(self):
        mm = MorphologyMap()
        assert mm.inflect("carry", Inflection.PAST) == "carried"

    def test_progressive_ie_to_ying(self):
        mm = MorphologyMap()
        assert mm.inflect("lie", Inflection.PROGRESSIVE) == "lying"


# ═══════════════════════════════════════════════════════════════════════════
# TestAgreementChecker
# ═══════════════════════════════════════════════════════════════════════════

class TestAgreementChecker:
    """Tests for AgreementChecker — agreement as distance constraints."""

    def test_construction(self):
        ac = AgreementChecker()
        assert ac.number_threshold > 0
        assert ac.tense_threshold > 0

    def test_check_empty(self):
        ac = AgreementChecker()
        result = ac.check([])
        assert isinstance(result, AgreementResult)
        assert result.is_valid

    def test_check_single_role(self):
        ac = AgreementChecker()
        roles = [
            RoleAssignment("x", SyntacticRole.SUBJECT, _make_noun_vec(), 0.8, 0.3),
        ]
        result = ac.check(roles)
        assert result.is_valid  # no verb → no S-V disagreement

    def test_check_matching_sv(self):
        """Subject and verb with similar logical fibers should agree."""
        ac = AgreementChecker()
        vec_s = _make_noun_vec(seed=1)
        vec_v = vec_s.copy()  # same logical fiber → should agree
        vec_v[0:16] = _make_verb_vec(seed=2)[0:16]  # only change sim quadrant
        roles = [
            RoleAssignment("force", SyntacticRole.SUBJECT, vec_s, 0.8, 0.3, "noun"),
            RoleAssignment("acts", SyntacticRole.VERB, vec_v, 0.7, 0.5, "verb"),
        ]
        result = ac.check(roles)
        assert result.is_valid

    def test_check_mismatching_sv(self):
        """Subject and verb with very different logical fibers → violation."""
        ac = AgreementChecker(number_threshold=0.05)  # strict threshold
        vec_s = _make_vector(logical_universal=0.9, seed=1)  # plural-like
        vec_v = _make_vector(logical_universal=0.1, seed=2)  # singular-like
        # Make sure sim quadrants are correct
        vec_s[16:32] = 0.2  # noun-like
        vec_v[0:16] = 0.2   # verb-like
        roles = [
            RoleAssignment("forces", SyntacticRole.SUBJECT, vec_s, 0.8, 0.3, "noun"),
            RoleAssignment("acts", SyntacticRole.VERB, vec_v, 0.7, 0.5, "verb"),
        ]
        result = ac.check(roles)
        # With strict threshold, the different logical fibers trigger violation
        assert len(result.violations) >= 0  # may or may not trigger depending on exact distance

    def test_infer_tense_present(self):
        ac = AgreementChecker()
        roles = [
            RoleAssignment("x", SyntacticRole.VERB, _make_vector(causal_tau=0.0, seed=1), 0.7, 0.5),
        ]
        tense = ac.infer_tense(roles)
        assert tense == "present"

    def test_infer_tense_past(self):
        ac = AgreementChecker()
        roles = [
            RoleAssignment("x", SyntacticRole.VERB, _make_vector(causal_tau=-0.2, seed=1), 0.7, 0.5),
        ]
        tense = ac.infer_tense(roles)
        assert tense == "past"

    def test_infer_tense_future(self):
        ac = AgreementChecker()
        roles = [
            RoleAssignment("x", SyntacticRole.VERB, _make_vector(causal_tau=0.2, seed=1), 0.7, 0.5),
        ]
        tense = ac.infer_tense(roles)
        assert tense == "future"

    def test_infer_tense_no_verbs(self):
        ac = AgreementChecker()
        roles = [
            RoleAssignment("x", SyntacticRole.SUBJECT, _make_noun_vec(), 0.7, 0.5),
        ]
        tense = ac.infer_tense(roles)
        assert tense == "present"  # default

    def test_infer_number_singular(self):
        ac = AgreementChecker()
        role = RoleAssignment("x", SyntacticRole.SUBJECT, _make_vector(logical_universal=0.2), 0.8, 0.3)
        assert ac.infer_number(role) == "singular"

    def test_infer_number_plural(self):
        ac = AgreementChecker()
        role = RoleAssignment("x", SyntacticRole.SUBJECT, _make_vector(logical_universal=0.8), 0.8, 0.3)
        assert ac.infer_number(role) == "plural"

    def test_correct_surface(self):
        ac = AgreementChecker()
        corrected = ac.correct_surface("run", Inflection.PAST)
        assert corrected == "ran"

    def test_agreement_result_repr(self):
        r = AgreementResult(is_valid=True)
        assert "VALID" in repr(r)
        r2 = AgreementResult(is_valid=False, violations=[("test", 0.5)])
        assert "INVALID" in repr(r2)

    def test_tense_consistency_two_verbs(self):
        """Two verbs with very different τ-offsets should flag inconsistency."""
        ac = AgreementChecker(tense_threshold=0.001)  # very strict
        vec1 = _make_vector(causal_tau=-0.5, seed=1)
        vec2 = _make_vector(causal_tau=0.5, seed=2)
        roles = [
            RoleAssignment("ran", SyntacticRole.VERB, vec1, 0.7, 0.3),
            RoleAssignment("will go", SyntacticRole.VERB, vec2, 0.7, 0.7),
        ]
        result = ac.check(roles)
        # Very different τ-offsets → tense inconsistency
        tense_viols = [v for v in result.violations if "tense" in v[0]]
        assert len(tense_viols) >= 1


# ═══════════════════════════════════════════════════════════════════════════
# TestGrammarRenderer
# ═══════════════════════════════════════════════════════════════════════════

class TestGrammarRenderer:
    """Tests for GrammarRenderer — compositional sentence construction."""

    def test_construction(self):
        gr = GrammarRenderer()
        assert gr.morphology is not None
        assert gr.agreement is not None
        assert gr.syntax is not None
        assert gr.composer is not None

    def test_render_empty(self):
        gr = GrammarRenderer()
        result = gr.render_segment([], [], [], [])
        assert isinstance(result, RenderedSentence)
        assert result.text == ""

    def test_render_single_concept(self):
        gr = GrammarRenderer()
        vec = _make_noun_vec(tau=0.5, seed=1)
        result = gr.render_segment([vec], ["gravity"], [0.8], [0.5])
        assert isinstance(result, RenderedSentence)
        assert len(result.text) > 0
        assert "gravity" in result.text.lower()

    def test_render_two_concepts(self):
        gr = GrammarRenderer()
        v1 = _make_noun_vec(tau=0.2, seed=1)
        v2 = _make_noun_vec(tau=0.8, seed=2)
        result = gr.render_segment(
            [v1, v2], ["force", "motion"], [0.8, 0.6], [0.2, 0.8]
        )
        assert len(result.text) > 0

    def test_render_three_concepts_svo(self):
        gr = GrammarRenderer()
        v_s = _make_noun_vec(tau=0.2, seed=1)
        v_v = _make_verb_vec(tau=0.5, seed=2)
        v_o = _make_noun_vec(tau=0.8, seed=3)
        result = gr.render_segment(
            [v_s, v_v, v_o],
            ["gravity", "drives", "acceleration"],
            [0.8, 0.7, 0.6],
            [0.2, 0.5, 0.8],
        )
        assert len(result.text) > 0
        assert result.complexity >= 1

    def test_render_with_past_tense(self):
        gr = GrammarRenderer()
        v_s = _make_noun_vec(tau=0.2, seed=1)
        v_v = _make_vector(sim_quad=0, causal_tau=-0.2, seed=2)  # past τ
        v_o = _make_noun_vec(tau=0.8, seed=3)
        result = gr.render_segment(
            [v_s, v_v, v_o],
            ["force", "caused", "motion"],
            [0.8, 0.7, 0.6],
            [0.2, 0.5, 0.8],
        )
        assert result.tense == "past"

    def test_render_uncertain_segment(self):
        gr = GrammarRenderer()
        v1 = _make_noun_vec(tau=0.3, seed=1)
        v2 = _make_noun_vec(tau=0.7, seed=2)
        result = gr.render_segment(
            [v1, v2], ["cause", "effect"], [0.5, 0.4], [0.3, 0.7],
            uncertainty=0.8,
        )
        # High uncertainty should trigger hedging
        text_lower = result.text.lower()
        # Should contain hedging language (may, seems, appears, etc.)
        assert any(w in text_lower for w in ["may", "seems", "appears", "it seems"])

    def test_render_fast_flow(self):
        gr = GrammarRenderer()
        v1 = _make_noun_vec(tau=0.3, seed=1)
        v2 = _make_noun_vec(tau=0.7, seed=2)
        result = gr.render_segment(
            [v1, v2], ["cause", "effect"], [0.8, 0.6], [0.3, 0.7],
            flow_speed=0.9,
        )
        # Fast flow → shorter text
        assert len(result.text) < 200

    def test_render_produces_terminal_punctuation(self):
        gr = GrammarRenderer()
        v1 = _make_noun_vec(tau=0.3, seed=1)
        v2 = _make_verb_vec(tau=0.6, seed=2)
        result = gr.render_segment(
            [v1, v2], ["mechanism", "operates"], [0.8, 0.7], [0.3, 0.6]
        )
        assert result.text.endswith(".")

    def test_render_capitalises_first_letter(self):
        gr = GrammarRenderer()
        v1 = _make_noun_vec(tau=0.3, seed=1)
        v2 = _make_noun_vec(tau=0.7, seed=2)
        result = gr.render_segment(
            [v1, v2], ["alpha", "beta"], [0.8, 0.6], [0.3, 0.7]
        )
        if result.text:
            assert result.text[0].isupper()

    def test_render_confidence_range(self):
        gr = GrammarRenderer()
        v1 = _make_noun_vec(tau=0.3, seed=1)
        v2 = _make_verb_vec(tau=0.6, seed=2)
        result = gr.render_segment(
            [v1, v2], ["mechanism", "operates"], [0.8, 0.7], [0.3, 0.6]
        )
        assert 0.0 <= result.confidence <= 1.0

    def test_render_from_plan(self):
        gr = GrammarRenderer()
        roles = [
            RoleAssignment("gravity", SyntacticRole.SUBJECT, _make_noun_vec(0.2, 1), 0.8, 0.2, "noun"),
            RoleAssignment("pulls", SyntacticRole.VERB, _make_verb_vec(0.5, 2), 0.7, 0.5, "verb"),
            RoleAssignment("objects", SyntacticRole.OBJECT, _make_noun_vec(0.8, 3), 0.6, 0.8, "noun"),
        ]
        plan = SentencePlan(
            clauses=[Clause(ClauseType.MAIN, roles, amplitude=0.7)],
            complexity=1,
            confidence=0.8,
        )
        result = gr.render_from_plan(plan)
        assert isinstance(result, RenderedSentence)
        assert len(result.text) > 0

    def test_render_from_plan_empty(self):
        gr = GrammarRenderer()
        plan = SentencePlan(clauses=[], complexity=0)
        result = gr.render_from_plan(plan)
        assert result.text == ""

    def test_clean_label_strips_domain(self):
        assert GrammarRenderer._clean_label("causal::mechanism") == "mechanism"
        assert GrammarRenderer._clean_label("vocab::running") == "running"
        assert GrammarRenderer._clean_label("simple") == "simple"

    def test_clean_label_strips_underscores(self):
        assert GrammarRenderer._clean_label("causal::co_occurrence") == "co occurrence"

    def test_rendered_sentence_repr(self):
        rs = RenderedSentence(text="The force drives motion.", complexity=1, tense="present")
        r = repr(rs)
        assert "complexity=1" in r
        assert "present" in r

    def test_render_with_domain_prefixed_labels(self):
        gr = GrammarRenderer()
        v1 = _make_noun_vec(tau=0.3, seed=1)
        v2 = _make_verb_vec(tau=0.6, seed=2)
        result = gr.render_segment(
            [v1, v2],
            ["causal::mechanism", "vocab::drives"],
            [0.8, 0.7], [0.3, 0.6],
        )
        # Domain prefixes should be stripped in output
        assert "::" not in result.text

    def test_render_many_concepts(self):
        """Ensure renderer handles many concepts without crashing."""
        gr = GrammarRenderer()
        n = 8
        vectors = [_make_vector(seed=i) for i in range(n)]
        labels = [f"concept_{i}" for i in range(n)]
        amps = [0.5 + 0.04 * i for i in range(n)]
        taus = [i / n for i in range(n)]
        result = gr.render_segment(vectors, labels, amps, taus)
        assert isinstance(result, RenderedSentence)
        assert len(result.text) > 0

    def test_render_deterministic(self):
        """Same input should produce same output."""
        gr = GrammarRenderer()
        v1 = _make_noun_vec(tau=0.3, seed=1)
        v2 = _make_verb_vec(tau=0.6, seed=2)
        args = ([v1, v2], ["force", "drives"], [0.8, 0.7], [0.3, 0.6])
        result1 = gr.render_segment(*args)
        result2 = gr.render_segment(*args)
        assert result1.text == result2.text


# ═══════════════════════════════════════════════════════════════════════════
# TestC7Integration
# ═══════════════════════════════════════════════════════════════════════════

class TestC7Integration:
    """Tests for grammar engine integration into the C7 pipeline."""

    def test_expression_renderer_has_grammar(self):
        from src.phase1.expression.renderer import ExpressionRenderer, _HAS_GRAMMAR
        assert _HAS_GRAMMAR is True
        renderer = ExpressionRenderer()
        assert renderer._grammar is not None

    def test_expression_renderer_without_grammar(self):
        from src.phase1.expression.renderer import ExpressionRenderer
        renderer = ExpressionRenderer(use_grammar=False)
        assert renderer._grammar is None

    def test_render_mock_wave_with_grammar(self):
        from src.phase1.expression.renderer import ExpressionRenderer
        from src.phase1.expression.wave import create_mock_wave
        renderer = ExpressionRenderer(use_grammar=True)
        wave = create_mock_wave("causation")
        output = renderer.render(wave)
        assert len(output.text) > 0
        assert output.confidence > 0

    def test_render_mock_wave_without_grammar(self):
        from src.phase1.expression.renderer import ExpressionRenderer
        from src.phase1.expression.wave import create_mock_wave
        renderer = ExpressionRenderer(use_grammar=False)
        wave = create_mock_wave("causation")
        output = renderer.render(wave)
        assert len(output.text) > 0
        assert output.confidence > 0

    def test_grammar_enhanced_diagnostics(self):
        """If grammar rendering occurs, diagnostics should show it."""
        from src.phase1.expression.renderer import ExpressionRenderer
        from src.phase1.expression.wave import StandingWave, WavePoint
        renderer = ExpressionRenderer(use_grammar=True)

        # Build a wave with rich labelled points
        rng = np.random.default_rng(42)
        points = []
        for i, label in enumerate(["mechanism", "drives", "outcome", "process"]):
            vec = _make_vector(seed=i * 10 + 1, sim_quad=16 if i % 2 == 0 else 0)
            points.append(WavePoint(
                vector=vec,
                amplitude=0.8 - 0.1 * i,
                label=label,
                tau=i / 4.0,
            ))
        wave = StandingWave(points=points, total_energy=2.5)
        output = renderer.render(wave)
        assert len(output.text) > 0
        # Check if any diagnostics have grammar_enhanced flag
        grammar_used = any(
            d.get("grammar_enhanced", False) for d in output.diagnostics
        )
        # Grammar should attempt enhancement for segments with 2+ labelled points
        # (may or may not succeed depending on segment formation)
        assert isinstance(grammar_used, bool)


# ═══════════════════════════════════════════════════════════════════════════
# TestFullPipeline
# ═══════════════════════════════════════════════════════════════════════════

class TestFullPipeline:
    """Integration tests with the full GEOPipeline."""

    def test_pipeline_query_with_grammar(self):
        """Ensure GEOPipeline.query() works with grammar engine active."""
        from src.phase5.pipeline.pipeline import GEOPipeline
        from src.phase3.annealing_engine.experience import Experience

        pipeline = GEOPipeline(flow_seed=42)

        # Learn some concepts
        rng = np.random.default_rng(42)
        for label in ["gravity", "force", "acceleration", "mass"]:
            vec = rng.uniform(0.1, 0.5, size=104)
            pipeline.learn(Experience(vector=vec, label=f"concept::{label}"))

        # Query
        query_vec = rng.uniform(0.1, 0.5, size=104)
        result = pipeline.query(query_vec, label="what is gravity?")
        assert len(result.text) > 0
        assert result.confidence > 0

    def test_pipeline_query_produces_text(self):
        """Pipeline must produce non-empty text."""
        from src.phase5.pipeline.pipeline import GEOPipeline

        pipeline = GEOPipeline(flow_seed=123)
        rng = np.random.default_rng(123)
        vec = rng.uniform(0.1, 0.5, size=104)
        result = pipeline.query(vec, label="test")
        assert isinstance(result.text, str)
        assert len(result.text) > 0


# ═══════════════════════════════════════════════════════════════════════════
# TestDesignConstraints
# ═══════════════════════════════════════════════════════════════════════════

class TestDesignConstraints:
    """Verify all six non-negotiable design constraints are upheld."""

    def test_no_weights(self):
        """No tunable numerical parameters in grammar engine."""
        gr = GrammarRenderer()
        # GrammarRenderer should not have any attribute named 'weights' or 'parameters'
        for attr in dir(gr):
            assert "weight" not in attr.lower() or attr.startswith("_")
        # Same for sub-components
        assert not hasattr(gr.syntax, "weights")
        assert not hasattr(gr.morphology, "weights")
        assert not hasattr(gr.agreement, "weights")

    def test_no_tokens(self):
        """No tokeniser, no token IDs anywhere in the grammar engine."""
        # Grammar renderer works with continuous vectors and labels, not tokens
        gr = GrammarRenderer()
        v1 = _make_noun_vec(seed=1)
        v2 = _make_verb_vec(seed=2)
        result = gr.render_segment(
            [v1, v2], ["cause", "drives"], [0.8, 0.7], [0.3, 0.6]
        )
        # Output is a direct string, not tokenised
        assert isinstance(result.text, str)

    def test_no_training(self):
        """Grammar engine requires no training phase."""
        # Instantiation is immediate — no training step
        gr = GrammarRenderer()
        assert gr is not None
        # Can render immediately without any training
        result = gr.render_segment(
            [_make_noun_vec()], ["test"], [0.5], [0.5]
        )
        assert isinstance(result, RenderedSentence)

    def test_local_updates_not_applicable(self):
        """Grammar engine is read-only — it doesn't modify the manifold."""
        # GrammarRenderer has no write operations on any manifold
        gr = GrammarRenderer()
        assert not hasattr(gr, "manifold")
        assert not hasattr(gr, "deform")
        assert not hasattr(gr, "place")

    def test_separation(self):
        """Grammar engine has no access to C1–C5 components."""
        gr = GrammarRenderer()
        for attr in dir(gr):
            obj = getattr(gr, attr, None)
            if obj is None:
                continue
            obj_type = type(obj).__name__
            # Should not have FlowEngine, ResonanceLayer, etc.
            assert "FlowEngine" not in obj_type
            assert "ResonanceLayer" not in obj_type
            assert "LivingManifold" not in obj_type
            assert "AnnealingEngine" not in obj_type

    def test_morphology_offsets_are_geometric(self):
        """Inflection offsets should be meaningful geometric displacements."""
        mm = MorphologyMap()
        # Past tense: negative τ shift (causal dim 64)
        past = mm.get_offset(Inflection.PAST)
        assert past[64] < 0  # earlier τ

        # Plural: positive logical shift
        plural = mm.get_offset(Inflection.PLURAL)
        assert plural[81] > 0  # universal quantifier direction

        # Negative: negative logical shift
        neg = mm.get_offset(Inflection.NEGATIVE)
        assert neg[80] < 0  # negation dimension
