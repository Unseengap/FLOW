"""Phase 10 — Geometric Grammar Engine (C7b).

Replaces template-based rendering in C7 with compositional syntax
derived from manifold geometry.  Grammar rules are encoded as geometric
operations on the 104D fiber bundle:

  Subject → Verb:     causal_direction(subject_pos, verb_pos) > 0
  Verb → Object:      causal_direction(verb_pos, object_pos) > 0
  Modifier → Head:    distance(modifier_pos, head_pos) < phrase_radius
  Main → Subordinate: fiber_section(main) contains fiber_section(sub)

Sub-modules
-----------
syntax_geometry   — SyntaxGeometry: S-V-O ordering, role assignment from causal fiber
clause_composer   — ClauseComposer: composes main + subordinate clauses via fiber bundle
morphology_map    — MorphologyMap: word family clusters with systematic fiber offsets
grammar_renderer  — GrammarRenderer: compositional sentence construction (replaces slot-fill)
agreement_checker — AgreementChecker: number/tense agreement as distance constraints

No weights. No tokens. Grammar IS geometry.
"""

from .grammar.syntax_geometry import SyntaxGeometry, SyntacticRole, RoleAssignment
from .grammar.clause_composer import ClauseComposer, Clause, ClauseType, SentencePlan
from .grammar.morphology_map import MorphologyMap, WordForm, Inflection
from .grammar.grammar_renderer import GrammarRenderer
from .grammar.agreement_checker import AgreementChecker, AgreementResult

__all__ = [
    "SyntaxGeometry",
    "SyntacticRole",
    "RoleAssignment",
    "ClauseComposer",
    "Clause",
    "ClauseType",
    "SentencePlan",
    "MorphologyMap",
    "WordForm",
    "Inflection",
    "GrammarRenderer",
    "AgreementChecker",
    "AgreementResult",
]
