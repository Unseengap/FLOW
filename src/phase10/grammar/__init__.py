"""Phase 10 — Geometric Grammar sub-package.

Contains the five sub-components of the Geometric Grammar Engine:
  SyntaxGeometry   — role assignment from causal fiber
  ClauseComposer   — clause composition via fiber bundle sections
  MorphologyMap    — word family inflection clusters
  GrammarRenderer  — compositional sentence construction
  AgreementChecker — number/tense agreement constraints
"""

from .syntax_geometry import SyntaxGeometry, SyntacticRole, RoleAssignment
from .clause_composer import ClauseComposer, Clause, ClauseType, SentencePlan
from .morphology_map import MorphologyMap, WordForm, Inflection
from .grammar_renderer import GrammarRenderer
from .agreement_checker import AgreementChecker, AgreementResult

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
