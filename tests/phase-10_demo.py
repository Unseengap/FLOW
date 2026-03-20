#!/usr/bin/env python3
"""Phase 10 — Geometric Grammar Engine demo.

Demonstrates the five sub-components of C7b (Geometric Grammar Engine)
and shows grammar-enhanced output from the full GEOPipeline.

Components exercised:
  SyntaxGeometry   — role assignment from causal fiber
  ClauseComposer   — clause composition via fiber bundle
  MorphologyMap    — word family inflection clusters
  GrammarRenderer  — compositional sentence construction
  AgreementChecker — agreement as distance constraints
  C7 Integration   — grammar engine wired into ExpressionRenderer
  Full Pipeline    — GEOPipeline.query() with grammar enhancement
"""

from __future__ import annotations

import sys
import time
import numpy as np

sys.path.insert(0, ".")

from src.phase10.grammar.syntax_geometry import SyntaxGeometry, SyntacticRole
from src.phase10.grammar.clause_composer import ClauseComposer, ClauseType
from src.phase10.grammar.morphology_map import MorphologyMap, Inflection
from src.phase10.grammar.grammar_renderer import GrammarRenderer, RenderedSentence
from src.phase10.grammar.agreement_checker import AgreementChecker

DIM = 104
_SIM_START, _SIM_END = 0, 64
_CAUSAL_START, _CAUSAL_END = 64, 80
_LOGICAL_START, _LOGICAL_END = 80, 88
_PROB_START, _PROB_END = 88, 104


def _make_vector(sim_quad=16, causal_tau=0.5, causal_strength=0.1,
                 logical_neg=0.0, logical_univ=0.3, seed=42):
    rng = np.random.default_rng(seed)
    vec = rng.uniform(0.01, 0.05, size=DIM)
    vec[_SIM_START:_SIM_END] = 0.01
    vec[sim_quad:sim_quad + 16] = rng.uniform(0.1, 0.3, size=16)
    vec[_CAUSAL_START] = causal_tau
    vec[_CAUSAL_START + 1:_CAUSAL_END] = causal_strength * rng.uniform(0.5, 1.0, size=15)
    vec[_LOGICAL_START] = logical_neg
    vec[_LOGICAL_START + 1] = logical_univ
    vec[_PROB_START:_PROB_END] = 0.5 + rng.uniform(-0.1, 0.1, size=16)
    return vec


def _noun(tau, seed):
    return _make_vector(sim_quad=16, causal_tau=tau, seed=seed)


def _verb(tau, seed):
    return _make_vector(sim_quad=0, causal_tau=tau, causal_strength=0.2, seed=seed)


def _adj(tau, seed):
    return _make_vector(sim_quad=32, causal_tau=tau, seed=seed)


# ═══════════════════════════════════════════════════════════════════════════

def demo_syntax_geometry():
    print("=" * 70)
    print("1. SyntaxGeometry — Role Assignment from Causal Fiber")
    print("=" * 70)
    sg = SyntaxGeometry()

    vectors = [_noun(0.2, 1), _verb(0.5, 2), _noun(0.8, 3)]
    labels = ["gravity", "drives", "acceleration"]
    amps = [0.8, 0.7, 0.6]
    taus = [0.2, 0.5, 0.8]

    roles = sg.assign_roles(vectors, labels, amps, taus)
    for r in roles:
        morph = r.morph_class or "?"
        print(f"  {r.label:<20s} → {r.role.value:<12s} (morph={morph}, τ={r.tau:.2f}, amp={r.amplitude:.2f})")

    # Show causal direction
    d_fwd = sg.causal_direction(vectors[0], vectors[2])
    d_bwd = sg.causal_direction(vectors[2], vectors[0])
    print(f"\n  Causal direction gravity→acceleration: {d_fwd:+.3f}")
    print(f"  Causal direction acceleration→gravity: {d_bwd:+.3f}")
    print()


def demo_clause_composer():
    print("=" * 70)
    print("2. ClauseComposer — Clause Composition via Fiber Bundle")
    print("=" * 70)
    sg = SyntaxGeometry()
    cc = ClauseComposer(syntax=sg)

    # Two clauses
    group_a = sg.assign_roles(
        [_noun(0.2, 1), _verb(0.5, 2), _noun(0.8, 3)],
        ["force", "drives", "motion"], [0.8, 0.7, 0.6], [0.2, 0.5, 0.8])
    group_b = sg.assign_roles(
        [_noun(0.7, 4), _verb(0.9, 5)],
        ["resistance", "opposes"], [0.5, 0.6], [0.7, 0.9])

    plan = cc.compose([group_a, group_b])
    print(f"  Clauses: {len(plan.clauses)}")
    print(f"  Complexity: {plan.complexity}")
    print(f"  Compound: {plan.is_compound}")
    print(f"  Connectives: {plan.connectives}")
    for i, cl in enumerate(plan.clauses):
        roles_str = ", ".join(f"{r.label}={r.role.value}" for r in cl.roles)
        print(f"  Clause {i}: {cl.clause_type.value} [amp={cl.amplitude:.2f}] — {roles_str}")
    print()


def demo_morphology_map():
    print("=" * 70)
    print("3. MorphologyMap — Word Family Inflection Clusters")
    print("=" * 70)
    mm = MorphologyMap()

    # Word families
    for base in ["run", "drive", "force", "mechanism"]:
        family = mm.word_family(base)
        items = [f"{infl.value}: {form}" for infl, form in sorted(family.items(), key=lambda x: x[0].value)]
        print(f"  {base:<15s} → {', '.join(items)}")

    # Irregular analysis
    print("\n  Analysis:")
    for word in ["running", "drove", "children", "fastest", "unhappy"]:
        form = mm.analyse(word)
        print(f"    {word:<15s} → base={form.base}, inflection={form.inflection.value}, conf={form.confidence:.2f}")

    # Geometric offsets
    print("\n  Geometric offsets (L2 norm by fiber):")
    for infl in [Inflection.PAST, Inflection.PLURAL, Inflection.NEGATIVE, Inflection.PROGRESSIVE]:
        offset = mm.get_offset(infl)
        sim_n = np.linalg.norm(offset[_SIM_START:_SIM_END])
        caus_n = np.linalg.norm(offset[_CAUSAL_START:_CAUSAL_END])
        log_n = np.linalg.norm(offset[_LOGICAL_START:_LOGICAL_END])
        prob_n = np.linalg.norm(offset[_PROB_START:_PROB_END])
        print(f"    {infl.value:<15s} sim={sim_n:.3f}  causal={caus_n:.3f}  logical={log_n:.3f}  prob={prob_n:.3f}")
    print()


def demo_agreement_checker():
    print("=" * 70)
    print("4. AgreementChecker — Agreement as Distance Constraints")
    print("=" * 70)
    ac = AgreementChecker()

    # Matching S-V pair
    v_s = _noun(0.3, 1)
    v_v = v_s.copy()
    v_v[0:16] = _verb(0.5, 2)[0:16]

    from src.phase10.grammar.syntax_geometry import RoleAssignment
    roles = [
        RoleAssignment("force", SyntacticRole.SUBJECT, v_s, 0.8, 0.3, "noun"),
        RoleAssignment("acts", SyntacticRole.VERB, v_v, 0.7, 0.5, "verb"),
    ]
    result = ac.check(roles)
    print(f"  Matching S-V: valid={result.is_valid}, violations={result.violations}")

    # Tense inference
    for tau_val, label in [(-0.3, "past τ"), (0.0, "present τ"), (0.3, "future τ")]:
        v = _make_vector(causal_tau=tau_val, seed=10)
        roles_t = [RoleAssignment("test", SyntacticRole.VERB, v, 0.7, 0.5)]
        tense = ac.infer_tense(roles_t)
        print(f"  τ = {tau_val:+.1f} → tense = {tense}")

    # Number inference
    for univ, label in [(0.2, "singular"), (0.8, "plural")]:
        v = _make_vector(logical_univ=univ, seed=20)
        role = RoleAssignment("test", SyntacticRole.SUBJECT, v, 0.8, 0.3)
        number = ac.infer_number(role)
        print(f"  Universal quantifier = {univ:.1f} → number = {number}")
    print()


def demo_grammar_renderer():
    print("=" * 70)
    print("5. GrammarRenderer — Compositional Sentence Construction")
    print("=" * 70)
    gr = GrammarRenderer()

    # Basic S-V-O
    result = gr.render_segment(
        [_noun(0.2, 1), _verb(0.5, 2), _noun(0.8, 3)],
        ["gravity", "drives", "acceleration"],
        [0.8, 0.7, 0.6], [0.2, 0.5, 0.8],
    )
    print(f"  S-V-O:        {result.text}")
    print(f"                 tense={result.tense}, complexity={result.complexity}, conf={result.confidence:.2f}")

    # Past tense
    result2 = gr.render_segment(
        [_noun(0.2, 1), _make_vector(sim_quad=0, causal_tau=-0.2, seed=2), _noun(0.8, 3)],
        ["force", "caused", "motion"],
        [0.8, 0.7, 0.6], [0.2, 0.5, 0.8],
    )
    print(f"  Past tense:   {result2.text}")
    print(f"                 tense={result2.tense}")

    # Uncertain
    result3 = gr.render_segment(
        [_noun(0.3, 1), _noun(0.7, 2)],
        ["cause", "effect"],
        [0.5, 0.4], [0.3, 0.7],
        uncertainty=0.8,
    )
    print(f"  Uncertain:    {result3.text}")

    # Multi-concept
    n = 5
    vecs = [_make_vector(seed=i * 10) for i in range(n)]
    labels = ["mechanism", "drives", "outcome", "through", "interaction"]
    result4 = gr.render_segment(vecs, labels, [0.7] * n, [i / n for i in range(n)])
    print(f"  Multi-concept: {result4.text}")
    print(f"                 complexity={result4.complexity}")

    # With domain prefixes
    result5 = gr.render_segment(
        [_noun(0.3, 1), _verb(0.6, 2)],
        ["causal::mechanism", "vocab::operates"],
        [0.8, 0.7], [0.3, 0.6],
    )
    print(f"  Domain-clean: {result5.text}")
    print()


def demo_c7_integration():
    print("=" * 70)
    print("6. C7 Integration — Grammar Engine in ExpressionRenderer")
    print("=" * 70)
    from src.phase1.expression.renderer import ExpressionRenderer, _HAS_GRAMMAR
    from src.phase1.expression.wave import create_mock_wave, StandingWave, WavePoint

    print(f"  Grammar module available: {_HAS_GRAMMAR}")

    # With grammar
    renderer_on = ExpressionRenderer(use_grammar=True)
    print(f"  Grammar attached: {renderer_on._grammar is not None}")

    # Without grammar
    renderer_off = ExpressionRenderer(use_grammar=False)

    wave = create_mock_wave("causation")
    out_on = renderer_on.render(wave)
    out_off = renderer_off.render(wave)
    print(f"\n  Template-only output:  {out_off.text[:100]}...")
    print(f"  Grammar output:        {out_on.text[:100]}...")
    print(f"  Template confidence:   {out_off.confidence:.3f}")
    print(f"  Grammar confidence:    {out_on.confidence:.3f}")

    # Rich wave with labelled points
    rng = np.random.default_rng(42)
    points = []
    for i, label in enumerate(["mechanism", "drives", "outcome", "interaction", "process"]):
        vec = _make_vector(seed=i * 10 + 1, sim_quad=16 if i % 2 == 0 else 0)
        points.append(WavePoint(vector=vec, amplitude=0.8 - 0.08 * i, label=label, tau=i / 5.0))
    rich_wave = StandingWave(points=points, total_energy=3.0)

    rich_on = renderer_on.render(rich_wave)
    rich_off = renderer_off.render(rich_wave)
    print(f"\n  Rich wave (template):  {rich_off.text[:100]}...")
    print(f"  Rich wave (grammar):   {rich_on.text[:100]}...")
    print()


def demo_full_pipeline():
    print("=" * 70)
    print("7. Full Pipeline — GEOPipeline.query() with Grammar Enhancement")
    print("=" * 70)
    from src.phase5.pipeline.pipeline import GEOPipeline
    from src.phase3.annealing_engine.experience import Experience

    pipeline = GEOPipeline(flow_seed=42)

    # Learn concepts
    rng = np.random.default_rng(42)
    concepts = ["gravity", "mass", "force", "acceleration", "energy",
                "momentum", "velocity", "distance", "time", "work"]
    for label in concepts:
        vec = rng.uniform(0.1, 0.5, size=104)
        pipeline.learn(Experience(vector=vec, label=f"concept::{label}"))

    print(f"  Concepts on manifold: {pipeline.n_concepts}")
    print(f"  Temperature: {pipeline.temperature:.4f}")

    # Query
    t0 = time.perf_counter()
    query_vec = rng.uniform(0.1, 0.5, size=104)
    result = pipeline.query(query_vec, label="what governs motion?")
    elapsed = time.perf_counter() - t0

    print(f"\n  Query: 'what governs motion?'")
    print(f"  Output: {result.text}")
    print(f"  Confidence: {result.confidence:.3f}")
    print(f"  Wave confidence: {result.wave_confidence:.3f}")
    print(f"  Steps: {result.n_steps}")
    print(f"  Termination: {result.termination_reason}")
    print(f"  Flow preserved: {result.flow_preserved}")
    print(f"  Elapsed: {elapsed:.3f}s")
    print()


# ═══════════════════════════════════════════════════════════════════════════

def main():
    print()
    print("╔══════════════════════════════════════════════════════════════════════╗")
    print("║        Phase 10 — Geometric Grammar Engine (C7b) Demo             ║")
    print("║    Compositional syntax derived from manifold geometry             ║")
    print("╚══════════════════════════════════════════════════════════════════════╝")
    print()

    t0 = time.perf_counter()

    demo_syntax_geometry()
    demo_clause_composer()
    demo_morphology_map()
    demo_agreement_checker()
    demo_grammar_renderer()
    demo_c7_integration()
    demo_full_pipeline()

    elapsed = time.perf_counter() - t0

    print("=" * 70)
    print(f"Phase 10 demo completed in {elapsed:.3f}s")
    print("  • No weight matrices, no tokenisers, no training")
    print("  • Grammar derived from 104D manifold geometry")
    print("  • S-V-O ordering from causal fiber direction")
    print("  • Inflections as geometric offsets in fiber bundle")
    print("  • Agreement as distance constraints in logical fiber")
    print("=" * 70)


if __name__ == "__main__":
    main()
