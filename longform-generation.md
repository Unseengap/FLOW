# FLOW — Longform Generation Architecture

**Status:** Design specification — not yet implemented  
**Depends on:** Phases 1–8 (complete), Phase 7 vocabulary (scaling in progress)  
**Components involved:** C5 Flow Engine, C6 Resonance Layer, C7 Expression Renderer, new C8 Discourse Planner

---

## The Problem

FLOW currently generates **one paragraph per query** — a single trajectory through M(t), one standing wave Ψ, a few rendered sentences. This is sufficient for Q&A but cannot produce:

- Books (novels, textbooks, technical manuals)
- Research papers and essays
- Long-form reports and analyses
- Screenplays and scripts
- Multi-chapter tutorials and courses
- Legal documents and contracts
- Narrative journalism and long-read articles
- Code documentation spanning hundreds of pages

Longform generation requires **hierarchical structure**, **long-range coherence**, **working memory**, and **discourse planning** — none of which exist in the current pipeline.

---

## Why FLOW Has a Structural Advantage Over LLMs for Longform

### The context window problem (LLMs)

Every LLM has a finite context window:

| Model | Context window | Approximate pages |
|---|---|---|
| GPT-4o | 128K tokens | ~50 pages |
| Claude 3.5 | 200K tokens | ~80 pages |
| Gemini 1.5 Pro | 1M tokens | ~400 pages |

When generating page 350 of a novel, an LLM with a 128K context window **literally cannot see** anything before page 300. It maintains coherence through statistical patterns, not structural understanding. Characters drift, plot threads disappear, themes contradict.

### The manifold solution (FLOW)

FLOW's manifold M(t) has **no context window**. Every concept, every causal relationship, every similarity — all 104 dimensions — are simultaneously present at all times. When generating chapter 40, the system has the same geometric access to chapter 1's structure as chapter 39's.

This isn't a memory trick. The manifold IS the knowledge. A character's traits are geometric positions. A plot thread is a causal chain in the fiber (dims 64–79). A theme is a dense cluster in the similarity base (dims 0–63). They don't fade or get pushed out of a sliding window — they're permanently encoded in the shape of the space.

---

## Architecture: Hierarchical Trajectory Composition

The core insight: **longform text is trajectories at multiple scales on the same manifold.**

```
Level 0 — Work Plan       : 1 macro-trajectory across M(t)
                            visits thematic regions, establishes global arc
                            → document outline (parts / acts)

Level 1 — Chapter Plan    : N medium trajectories, each seeded by a
                            Level 0 waypoint → chapter outlines

Level 2 — Section Plan    : M short trajectories per chapter,
                            seeded by Level 1 waypoints → section flow

Level 3 — Sentence Gen    : current C5 → C6 → C7 pipeline
                            one trajectory per paragraph
                            → rendered natural language
```

```
          ┌─────────────────────────────────────────────┐
          │           Level 0: Work Plan                │
          │   slow macro-trajectory across M(t)         │
          │   ●━━━━━●━━━━━●━━━━━●━━━━━●━━━━━●          │
          │   Ch1   Ch2   Ch3   Ch4   Ch5   Ch6        │
          └─────┬─────┬─────┬─────┬─────┬─────┬────────┘
                │     │     │     │     │     │
          ┌─────▼──┐  │  ┌──▼──┐  │  ┌──▼──┐  │
          │Level 1 │  │  │ L1  │  │  │ L1  │  │  ...
          │Ch1 plan│  │  │Ch3  │  │  │Ch5  │  │
          │●━●━●━● │  │  │plan │  │  │plan │  │
          └┬─┬─┬─┬┘  │  └─────┘  │  └─────┘  │
           │ │ │ │    │           │            │
          ┌▼─▼─▼─▼─┐  │           │            │
          │Level 2  │  │           │            │
          │Sections │  │           │            │
          │per §    │  │           │            │
          └┬─┬─┬─┬─┘  │           │            │
           │ │ │ │    │           │            │
          ┌▼─▼─▼─▼─┐  │           │            │
          │Level 3  │  │           │            │
          │C5→C6→C7 │  │           │            │
          │sentences│  │           │            │
          └─────────┘  │           │            │
                       ...        ...          ...
```

Each level uses the **same Flow Engine** (C5) with different parameters:

| Level | Step size | Noise σ | Force balance | Trajectory length |
|---|---|---|---|---|
| 0 — Work | Large | Low | Gravity-dominant (thematic pull) | 5–50 waypoints |
| 1 — Chapter | Medium | Medium | Causal-dominant (plot progression) | 10–30 waypoints |
| 2 — Section | Small | Medium | Momentum-dominant (local coherence) | 5–15 waypoints |
| 3 — Sentence | Current | Current | All four forces balanced | 10–50 steps |

---

## New Component: C8 Discourse Planner

### Responsibility

Orchestrate multi-level trajectory composition. C8 takes a **seed intent** (what to write about) and produces an ordered sequence of Level 3 trajectory seeds that, when rendered through C5→C6→C7, produce coherent longform text.

### Interface (proposed)

```python
from src.discourse import DiscoursePlanner, DocumentPlan, ChapterPlan

planner = DiscoursePlanner(
    manifold=pipeline.manifold,
    flow_engine=pipeline._flow_engine,
)

# Generate a document plan from a seed concept
plan = planner.plan(
    seed=vec_104d,               # starting concept
    target_length="chapter",     # "paragraph" | "section" | "chapter" | "book"
    style="expository",          # "narrative" | "expository" | "instructional" | "analytical"
    constraints={                # optional structural constraints
        "must_visit": [vec_a, vec_b],    # concepts that must appear
        "must_avoid": [vec_c],           # concepts to exclude
        "causal_chain": [(cause, effect)], # required causal connections
    },
)

# plan.chapters : list[ChapterPlan]
# plan.outline  : str — human-readable outline
# plan.estimated_paragraphs : int

# Generate text from the plan
for chapter in plan.chapters:
    for section in chapter.sections:
        for paragraph_seed in section.paragraph_seeds:
            result = pipeline.query(paragraph_seed.vector, label=paragraph_seed.label)
            yield result.text
```

### Internal mechanics

1. **Level 0 — Macro trajectory**
   - Start at seed concept
   - Run C5 with large step size + low noise → visits major thematic regions
   - Each waypoint becomes a chapter seed
   - Causal fiber ensures cause→effect ordering across chapters

2. **Level 1 — Chapter planning**
   - For each Level 0 waypoint, run a medium-scale trajectory
   - Semantic gravity pulls toward related concepts
   - Contrast repulsion prevents repetition (already built into C5)
   - Each waypoint becomes a section seed

3. **Level 2 — Section planning**
   - For each Level 1 waypoint, run a short trajectory
   - Momentum force maintains local coherence
   - Each waypoint becomes a paragraph seed

4. **Level 3 — Paragraph generation**
   - Current C5→C6→C7 pipeline, unchanged
   - Standing wave Ψ → rendered text

---

## Working Memory: Trajectory History Layer

### Problem

When generating paragraph 500, the system needs to know what was already said — not to repeat it, and to maintain coherence.

### Solution: Temporary deformation layer

Each generated paragraph leaves a **trace** on the manifold — a small, temporary density increase at the trajectory's positions. This serves as working memory:

- **Anti-repetition**: subsequent trajectories are repelled from high-trace regions (contrast repulsion force, already in C5)
- **Coherence**: recent trace is denser, creating gravitational pull toward recent themes (semantic gravity force, already in C5)
- **Natural fading**: trace decays over time, preventing old content from over-constraining new generation

```python
class TrajectoryHistory:
    """Working memory as temporary manifold density."""

    def __init__(self, decay_rate: float = 0.995):
        self.traces: list[TraceEntry] = []
        self.decay_rate = decay_rate

    def record(self, trajectory: Trajectory, paragraph_index: int):
        """Record a trajectory as a density trace."""
        ...

    def density_at(self, position: np.ndarray) -> float:
        """Return accumulated trace density at a position."""
        ...

    def decay_step(self):
        """Decay all traces by decay_rate — old content fades."""
        ...
```

This requires **no new components** — it piggybacks on existing C5 forces:
- High trace density → semantic gravity pulls nearby → thematic coherence
- High trace density → contrast repulsion pushes away from exact repetition
- The balance between attraction and repulsion naturally produces **variation on a theme** rather than verbatim repetition or wild tangents

---

## Longform Generation Types

### Narrative fiction (novels, short stories)

```
Seed: character concepts + setting + conflict placed on M(t)

Level 0 trajectory: visits conflict → escalation → climax → resolution
  (causal fiber encodes plot causality: event A → event B → event C)

Level 1: each chapter develops around a trajectory waypoint
  characters = manifold concepts with mutual distances encoding relationships
  tension = curvature of the causal fiber (high κ = dramatic tension)

Level 2: sections within chapters
  dialogue = trajectories that oscillate between two character positions
  description = trajectories that spiral around a setting concept

Level 3: sentences via C5→C6→C7
```

**Unique advantage**: Character consistency is geometric. A character's traits are a position in 104D. Their position never drifts — it's a fixed point on the manifold. An LLM can forget a character's eye color by page 200. FLOW cannot.

### Expository writing (textbooks, technical docs, encyclopedias)

```
Seed: topic concept

Level 0: breadth-first sweep of the topic's neighborhood on M(t)
  each major concept cluster becomes a chapter

Level 1: depth-first dive into each cluster
  causal fiber orders topics: prerequisites before applications
  logical fiber ensures non-contradiction across chapters

Level 2: sections explain sub-concepts
  consistency gradient ensures related ideas are near each other

Level 3: sentences, definitions, examples
```

**Unique advantage**: The logical fiber (dims 80–87, Boolean hypercube) structurally prevents contradictions. If chapter 3 states X, and X is geometrically contradictory to Y, the contrast repulsion force will prevent chapter 15 from stating Y. No LLM has this guarantee.

### Instructional / how-to (tutorials, courses, manuals)

```
Seed: skill or procedure concept

Level 0: follows the causal chain of the skill
  prerequisite → foundation → technique → application → mastery
  (causal ancestry in the fiber naturally orders this)

Level 1: each stage unpacks into lessons
  novelty estimation calibrates difficulty progression:
  early chapters = high-density regions (familiar concepts)
  later chapters = lower-density regions (novel material)

Level 2: sections within lessons
  exercises = queries into UNKNOWN regions adjacent to taught concepts
  (the system literally generates questions about what it doesn't know)

Level 3: instructional prose
```

**Unique advantage**: Difficulty progression is automatic. The density gradient on M(t) naturally orders content from well-known (crystallized) to unfamiliar (flexible/unknown). The system can generate exercises by querying its own UNKNOWN regions — asking genuine questions, not fake ones.

### Analytical / research (papers, reports, legal analysis)

```
Seed: thesis or research question

Level 0: explores the thesis neighborhood
  causal fiber traces causes and effects
  probabilistic fiber (dims 88–103, Fisher-Rao) quantifies uncertainty

Level 1: each major finding or argument becomes a section
  evidence = crystallized regions (high density, high confidence)
  speculation = flexible regions (medium density, qualified language)
  unknowns = explicitly flagged via region classification

Level 2: sub-arguments with supporting structure

Level 3: precise analytical prose
  confidence scores from C7 modulate language:
  high confidence → "X causes Y"
  medium confidence → "X likely contributes to Y"
  low confidence → "The relationship between X and Y remains unclear"
```

**Unique advantage**: Epistemic honesty is structural. The system's confidence isn't a learned text pattern — it's a measurement of manifold density. When FLOW says "this is uncertain," the geometry is actually sparse there. When it says "this is well-established," the region is genuinely crystallized from extensive experience.

### Screenplays and dialogue-heavy formats

```
Seed: character set + premise

Level 0: dramatic arc (setup → confrontation → resolution)

Level 1: scenes — each is a trajectory between character positions
  distance between characters on M(t) = dramatic tension
  characters "moving closer" on the manifold = alliance forming
  characters "moving apart" = conflict escalating

Level 2: beats within scenes

Level 3: dialogue lines
  each character's voice = trajectories constrained to start from
  that character's position on M(t)
  distinct positions → distinct vocabulary selection → distinct voice
```

**Unique advantage**: Every character has a permanent geometric identity. Voice consistency is not a prompt instruction — it's a mathematical constraint. Character A at position $P_A$ will always select vocabulary from $P_A$'s neighborhood, producing consistently different language from character B at position $P_B$.

### Code documentation and technical reference

```
Seed: codebase structure fed as experience

Level 0: architecture overview
  each module = a concept cluster on M(t)
  dependencies = causal fiber connections

Level 1: module-by-module documentation
  ordered by causal dependency (foundations first)

Level 2: class/function level
  API surface = manifold positions
  relationships = geometric distances and causal directions

Level 3: docstrings, examples, explanations
```

**Unique advantage**: The documentation maintains structural consistency with the code because both are encoded in the same geometry. If module A depends on module B, the causal fiber ensures B is documented before A, every time, automatically.

---

## Coherence Guarantees

What makes FLOW's longform generation fundamentally different from LLMs:

### 1. Global consistency (no context window)

The entire manifold is available at every generation step. Page 1 and page 400 have equal geometric access to all concepts and relationships.

### 2. Causal ordering

The causal fiber (dims 64–79) with asymmetric metric (γ=2.0) ensures that effects never precede their causes in the generated text. This is structural — not a learned pattern that can fail.

### 3. Logical non-contradiction

The logical fiber (dims 80–87, Boolean hypercube with Hamming distance) actively repels the trajectory from contradicting earlier statements via the contrast repulsion force in C5.

### 4. Thematic gravity

Dense concept clusters pull trajectories toward them via semantic gravity (Force 1 in C5). Themes naturally persist and recur without explicit tracking.

### 5. Anti-repetition

The contrast repulsion force (Force 4 in C5) combined with trajectory history traces prevents verbatim repetition while allowing thematic recurrence.

### 6. Difficulty calibration

Region classification (CRYSTALLIZED → FLEXIBLE → UNKNOWN) automatically calibrates language complexity:
- Crystallized regions → confident, precise language
- Flexible regions → exploratory, qualified language
- Unknown regions → acknowledged uncertainty

---

## Estimated Complexity

| What | Lines of code | New components | Existing components reused |
|---|---|---|---|
| C8 Discourse Planner | ~500–800 | 1 (DiscoursePlanner) | C5 FlowEngine (parameterized) |
| Trajectory History | ~200–300 | 1 (TrajectoryHistory) | C5 forces (unmodified) |
| Multi-level orchestrator | ~300–400 | 1 (LongformGenerator) | C5, C6, C7 (unmodified) |
| Document assembly | ~200–300 | 1 (DocumentAssembler) | None |
| **Total** | **~1,200–1,800** | **4 new classes** | **C5, C6, C7 unchanged** |

All four forces in C5, the resonance layer C6, and the expression renderer C7 are reused without modification. The architecture was designed for this — hierarchical trajectory composition is mathematically natural on Riemannian manifolds.

---

## Prerequisites

1. **Large vocabulary** (Phase 7 scaling — in progress on Kaggle)
   - Current ~100 entries → need 50K–200K for literary prose
   - The Kaggle notebook targets 50K as a first milestone

2. **Richer C7 templates** (future Phase)
   - Sentence variety: compound, complex, interrogative, imperative
   - Paragraph transitions: "however", "furthermore", "as a result"
   - Discourse markers: "in chapter 3 we saw...", "returning to..."

3. **Entity tracking on M(t)** (future Phase)
   - Named entities as first-class manifold concepts
   - Pronoun resolution via geometric proximity

4. **Genre-specific rendering modes** (future Phase)
   - Academic: citations, hedging language, formal register
   - Narrative: past tense, dialogue formatting, scene breaks
   - Technical: code blocks, numbered steps, cross-references

---

## The Fundamental Claim

A 400-page book is not 400 pages of independent text. It is a **single coherent structure** — a shape in meaning-space with global constraints (theme, plot, logic) and local texture (sentences, word choice, rhythm).

FLOW stores knowledge as shape. A book plan is just a large-scale trajectory through that shape. The architecture doesn't need to be redesigned for longform — it needs to be **scaled up** to operate at multiple trajectory resolutions simultaneously.

The manifold already encodes:
- What concepts exist (positions)
- How concepts relate (distances)
- What causes what (causal fiber)
- What contradicts what (logical fiber)
- How certain each relationship is (probabilistic fiber)
- Where knowledge is strong vs. weak (density / region classification)

A book is a guided tour through this geometry. The tour guide (C8 Discourse Planner) is the only missing piece. The landscape (M(t)) and the walking mechanism (C5 SDE) are already built.
