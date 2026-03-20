# Contextual Action Grouping Engine — Architectural Specification

**Version:** 1.0  
**Date:** 2026-03-19  
**Status:** Design  
**Depends on:** C1–C7 (core FLOW), Motor/Action Fiber, Temporal Fiber, Social Fiber, Meta-cognitive Fiber  

---

## 0. The Problem

Current AI assistants execute instructions one at a time. They wait for explicit commands. They don't understand that when a human says "I'm heading to a meeting," a cascade of related actions naturally follows — grab notes, mute phone, check room number, walk there.

Humans don't think in isolated commands. They think in **action bundles** — groups of actions that common sense ties together based on context. These bundles change depending on:

- **Environment** — kitchen vs office vs car vs outdoors
- **Activity** — cooking vs coding vs driving vs sleeping
- **Directive** — "clean up" vs "get ready" vs "let's ship this"
- **Life state** — morning routine vs deadline crunch vs relaxing vs emergency
- **Social context** — alone vs with family vs in a meeting vs with strangers

A geometric agent should learn these groupings the same way humans do — through experience, not rules.

---

## 1. Core Concept: Action Bundles as Geometric Clusters

An **action bundle** is a set of actions that common sense groups together in a specific context. On the manifold, this is a **cluster of action-points in the Motor/Action fiber whose proximity is conditioned on the active context region.**

```
Action Bundle = {
    context:  region of M(t) that is currently active (environment + activity + state)
    actions:  cluster of action-points within that context's gravitational basin
    ordering: causal fiber direction among the action-points (what must happen first)
    urgency:  density gradient (crystallized bundles fire fast, flexible ones allow delay)
}
```

The key insight: **the same action lives in DIFFERENT bundles depending on context.**

```
"Lock the door"
  └─ Context: LEAVING HOME     → Bundle: {keys, wallet, phone, lock door, check stove}
  └─ Context: GOING TO SLEEP   → Bundle: {lock door, set alarm, brush teeth, lights off}
  └─ Context: EMERGENCY        → Bundle: {lock door, call 911, gather family}

Same action. Different bundle. Different ordering. Different urgency.
The geometry handles this because context CHANGES which region of M(t) is active,
and the action's fiber components shift accordingly.
```

---

## 2. Architecture

### 2.1 New Fiber Dimensions

The Contextual Action Grouping Engine requires these manifold fiber extensions:

| Fiber | Dims | Purpose |
|---|---|---|
| **Context Fiber** | 32D | Encodes environment, activity, life state as a continuous position |
| **Action Fiber** | 32D | Encodes actions, their preconditions, effects, and tool requirements |
| **Temporal Fiber** | 16D | Encodes urgency, sequence ordering, time-sensitivity, deadlines |
| **Social Fiber** | 16D | Encodes who is present, roles, social appropriateness of actions |

These extend the existing 104D manifold. The Context + Action fibers are the primary workspace for this engine.

### 2.2 Component: Context Recognizer (CR)

**Responsibility:** Observe all available signals and produce a continuous context vector on M(t).

```
INPUT SIGNALS:
  ┌──────────────────────────────────────────────────────┐
  │ Environment │ Location, device, sensors, time of day │
  │ Activity    │ What the user is currently doing       │
  │ Directive   │ What the user just said or commanded   │
  │ Life State  │ Routine phase, emotional state, energy │
  │ Social      │ Who else is present, relationships     │
  │ History     │ What happened in the last N minutes    │
  └──────────────────────────────────────────────────────┘
                            │
                            ▼
              ┌──────────────────────────┐
              │    CONTEXT RECOGNIZER    │
              │                          │
              │  Maps signals → context  │
              │  fiber position on M(t)  │
              │                          │
              │  Output: 32D context     │
              │  vector = WHERE we are   │
              │  in situation-space      │
              └──────────────────────────┘
```

**How it works geometrically:**

Each signal is an experience fed through C3 (Annealing Engine). The context fiber self-organizes so that:

- Similar situations cluster together (morning routines near each other)
- Different situations separate (work vs leisure in different regions)
- Transitions are geodesics (leaving-home is between at-home and commuting)

**The Context Recognizer does NOT classify.** It places the current moment as a point on the context fiber. Classification is a downstream consequence of proximity to crystallized regions.

```python
class ContextRecognizer:
    """Maps raw signals to a context position on the manifold."""
    
    def observe(self, signals: ContextSignals) -> np.ndarray:
        """
        Fuse all input signals into a unified context vector.
        
        Process:
          1. Each signal type → position in its sub-fiber
          2. Weighted combination based on signal strength/reliability
          3. Project onto context fiber of M(t)
          4. Return 32D context vector
          
        No classification. No labels. Just geometric position.
        """
        
    def context_shift_detected(self, prev: np.ndarray, curr: np.ndarray) -> bool:
        """
        Detect when context has changed enough to trigger bundle re-evaluation.
        Uses geodesic distance on context fiber — not a threshold on raw vectors.
        """
```

### 2.3 Component: Bundle Geometry Engine (BGE)

**Responsibility:** Given a context position, identify which actions naturally group together and in what order.

This is the core innovation. Action bundles are NOT stored as lists. They are **geometric basins on the action fiber that become active when the context fiber enters a specific region.**

```
HOW BUNDLES FORM (Learning):

  1. Human performs action A in context C
     → A placed on action fiber, linked to C on context fiber
     
  2. Human performs action B shortly after A in same context C
     → B placed near A on action fiber (temporal proximity = geometric proximity)
     → Causal fiber direction A→B if A was precondition for B
     
  3. After many experiences of {A, B, D, E} co-occurring in context C:
     → These actions CRYSTALLIZE into a cluster on the action fiber
     → The cluster has a BASIN OF ATTRACTION (semantic gravity pulls related actions in)
     → The causal fiber imposes an ORDERING within the cluster
     
  4. When context C is detected again:
     → The context fiber position activates the gravitational basin
     → All actions in the basin are "ready" — the bundle is loaded
     → Causal ordering determines which fires first
```

```
HOW BUNDLES ACTIVATE (Inference):

  Context Recognizer outputs context position P_ctx
       │
       ▼
  Bundle Geometry Engine queries M(t):
    "Which action-fiber clusters have strong coupling to P_ctx?"
       │
       ▼
  Coupling = cross-fiber resonance between context and action fibers
    coupling(P_ctx, cluster_k) = Σ_i  exp(-d(P_ctx, ctx_i)² / 2r²) · density(cluster_k)
    
    where ctx_i are the context-fiber positions that were active
    when cluster_k's actions were originally experienced
       │
       ▼
  Top-K bundles ranked by coupling strength
       │
       ▼
  Within each bundle, actions ordered by CAUSAL FIBER direction
       │
       ▼
  Urgency determined by TEMPORAL FIBER:
    - Crystallized temporal patterns → strong urgency (habitual)
    - Flexible temporal patterns → suggested but deferrable
    - Unknown temporal patterns → offered tentatively
```

```python
class BundleGeometryEngine:
    """Identifies and activates action bundles from manifold geometry."""
    
    def activate_bundles(self, context: np.ndarray, k: int = 3) -> list[ActionBundle]:
        """
        Given current context position, find the top-k action bundles.
        
        Each ActionBundle contains:
          - actions: ordered list of action-points (causal fiber ordering)
          - confidence: density of the cluster (crystallized = high confidence)  
          - urgency: temporal fiber gradient (time-sensitive actions rank higher)
          - flexibility: how rigid the bundle is (can actions be skipped/reordered?)
        """
        
    def extend_bundle(self, bundle: ActionBundle, new_action: Action) -> ActionBundle:
        """
        User performed an action not in the predicted bundle.
        Extend the bundle geometry — place the new action near the cluster.
        The manifold deforms locally to accommodate.
        """
        
    def split_bundle(self, bundle: ActionBundle, context_a: np.ndarray, 
                     context_b: np.ndarray) -> tuple[ActionBundle, ActionBundle]:
        """
        A bundle that was one cluster has diverged across two contexts.
        The contrast engine (C4) separates them: DIFFERENT judgment
        pulls the sub-bundles apart on the action fiber.
        """
```

### 2.4 Component: Common Sense Geometry (CSG)

**Responsibility:** Encode the "obvious" relationships that humans consider common sense — not as rules, but as geometric priors on the manifold that experience refines.

```
COMMON SENSE IS GEOMETRY:

  "You need shoes before you can go outside"
    → causal fiber: shoes → outside (directed edge, high density)
    
  "Don't yell in a library"
    → social fiber: library_context + loud_action = high logical fiber contradiction
    
  "Bring an umbrella when it's raining"  
    → context fiber: rain, action fiber: umbrella — high cross-fiber coupling
    
  "Eat before taking medication on a full stomach"
    → temporal fiber: eat → medication (ordering), causal fiber: sequence dependency
    
  "Turn off the oven when you leave the kitchen"
    → context fiber: leaving_kitchen, action fiber: turn_off_oven
    → coupling crystallized through safety-importance weighting
```

**How common sense is LEARNED, not programmed:**

```
Phase 1 — Seed Priors (Weak Geometric Bias):
  The seed geometry (C1) provides basic causal/logical structure.
  "Cause precedes effect" is geometric (causal fiber asymmetry).
  "Contradiction repels" is geometric (logical fiber opposition).
  These are not common sense — they're mathematical prerequisites for it.

Phase 2 — Experience Accumulation (C3 Annealing):
  The agent observes human action sequences in context.
  Each observation is an experience placed on M(t).
  Frequently co-occurring (context, action-group) pairs crystallize.
  
  Example: 1000 observations of humans putting on shoes before going outside
    → "shoes → outside" becomes a dense, crystallized causal chain
    → This IS the common sense. It's not a rule — it's geometric density.

Phase 3 — Contrast Refinement (C4):
  The agent observes violations and corrections.
  "You forgot your keys" → contrast judgment: keys SAME-AS leaving_home bundle
    → Keys get pulled into the leaving-home cluster
  "Don't do X in context Y" → contrast judgment: X DIFFERENT-FROM Y
    → X gets pushed away from Y's action basin  

Phase 4 — Generalization via Geometric Isomorphism:
  Once "shoes → outside" is crystallized, the PATTERN generalizes:
    → "gear_up → activity" has the same fiber structure
    → "preparation → execution" is geometrically isomorphic
    → New activity encountered? The agent already knows "prepare first"
      because the geometric PATTERN is learned, not the specific instance.
```

```python
class CommonSenseGeometry:
    """
    Common sense as geometric density on the manifold.
    Not rules. Not knowledge graphs. Experienced patterns 
    crystallized into manifold structure.
    """
    
    def plausibility(self, action: np.ndarray, context: np.ndarray) -> float:
        """
        How plausible is this action in this context?
        = cross-fiber coupling strength between action and context positions.
        
        High density coupling → "obvious" action (common sense)
        Zero density → "that doesn't make sense here"
        Negative logical fiber → "that contradicts the situation"
        """
    
    def missing_actions(self, completed: list[np.ndarray], 
                        bundle: ActionBundle) -> list[Action]:
        """
        What actions in the expected bundle haven't been done yet?
        The geometric difference between the activated bundle 
        and the completed-action trace.
        
        "You haven't locked the door yet" — because the leaving-home
        bundle has a crystallized action-point for locking that 
        hasn't been matched by any completed action.
        """
    
    def safety_critical(self, action: np.ndarray, context: np.ndarray) -> bool:
        """
        Is this a safety-critical action in this context?
        = action sits in a HIGH-DENSITY region of the causal fiber
          with STRONG downstream consequences.
        
        "Turn off the stove" when leaving → high causal density 
        toward fire/damage outcomes → safety critical = True
        """
```

---

## 3. The Full Loop: Observe → Group → Suggest → Learn

```
┌─────────────────────────────────────────────────────────────────────┐
│                                                                     │
│  ┌──────────────┐      ┌──────────────┐      ┌──────────────┐     │
│  │   OBSERVE     │      │    GROUP      │      │   SUGGEST     │     │
│  │               │      │               │      │               │     │
│  │ Environment   │─────▶│ Context →     │─────▶│ Present       │     │
│  │ User actions  │      │ Bundle lookup │      │ action bundle │     │
│  │ Directives    │      │ on M(t)       │      │ with ordering │     │
│  │ Life state    │      │               │      │ and urgency   │     │
│  │ Social scene  │      │               │      │               │     │
│  └──────────────┘      └──────────────┘      └──────┬───────┘     │
│                                                      │             │
│                                                      ▼             │
│                                               ┌──────────────┐     │
│                                               │   EXECUTE     │     │
│                                               │               │     │
│                                               │ Agent acts OR │     │
│                                               │ user acts     │     │
│                                               │ (observed)    │     │
│                                               └──────┬───────┘     │
│                                                      │             │
│                                                      ▼             │
│  ┌──────────────┐      ┌──────────────┐      ┌──────────────┐     │
│  │   LEARN       │◀─────│   COMPARE     │◀─────│   OUTCOME     │     │
│  │               │      │               │      │               │     │
│  │ Deform M(t)   │      │ Expected      │      │ What actually │     │
│  │ Strengthen or │      │ bundle vs     │      │ happened?     │     │
│  │ weaken bundle │      │ actual actions │      │ Success/fail? │     │
│  │ geometry      │      │               │      │               │     │
│  └──────────────┘      └──────────────┘      └──────────────┘     │
│                                                                     │
│                         ┌──────────────┐                           │
│                         │  ADAPT        │                           │
│                         │               │                           │
│                         │ If user added │                           │
│                         │ actions →     │                           │
│                         │ extend bundle │                           │
│                         │               │                           │
│                         │ If user       │                           │
│                         │ skipped →     │                           │
│                         │ weaken link   │                           │
│                         │               │                           │
│                         │ If user       │                           │
│                         │ reordered →   │                           │
│                         │ adjust causal │                           │
│                         │ fiber         │                           │
│                         └──────────────┘                           │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 4. Context Types and Bundle Examples

### 4.1 Environment-Driven Bundles

The agent senses (or is told) the physical/digital environment.

| Environment | Detected Via | Bundle Activated | Action Order (Causal Fiber) |
|---|---|---|---|
| **Kitchen** | Location sensor, appliance state | Cooking bundle | Prep → Cook → Plate → Clean |
| **Car** | Bluetooth connect, GPS moving | Driving bundle | Seatbelt → Mirror → Navigate → Drive |
| **Office desk** | Device connected, calendar active | Work bundle | Check email → Priority tasks → Deep work → Break |
| **Bed (nighttime)** | Time + device idle + lights dim | Sleep bundle | Lock doors → Set alarm → Lights off → Do not disturb |
| **Code editor** | IDE active, file types detected | Dev bundle | Read context → Write code → Test → Commit |
| **Grocery store** | Location + shopping list app open | Shopping bundle | Check list → Navigate aisles → Compare prices → Checkout |

### 4.2 Directive-Driven Bundles

The user says something, and the agent infers the action cascade.

| Directive | Context Fiber Activation | Bundle Triggered |
|---|---|---|
| "Let's ship this" | Work context + code project | Run tests → Fix failures → Update changelog → Tag release → Deploy → Notify team |
| "I'm heading out" | Home context + time of day | Keys → Wallet → Phone → Lock door → Check stove/lights |
| "Clean up" | Kitchen context | Clear plates → Wash dishes → Wipe counters → Take out trash |
| "Clean up" | Code context | Remove dead code → Organize imports → Run linter → Commit |
| "Get ready" | Morning + work day | Shower → Dress → Breakfast → Check calendar → Pack bag |
| "Get ready" | Evening + date night | Shower → Dress up → Restaurant booking → Directions |
| "Help me move" | Home context + packing signals | Pack boxes → Label → Load truck → Drive → Unload → Unpack |

**Notice:** "Clean up" triggers completely different bundles depending on context. The geometry handles this because the SAME directive occupies different cross-fiber positions when combined with different context vectors.

### 4.3 Life-State Bundles

Longer-term patterns that group high-level goals.

| Life State | Time Scale | Meta-Bundle |
|---|---|---|
| **Morning routine** | 30–60 min | Wake → Hygiene → Dress → Eat → Commute prep |
| **Deep focus work** | 2–4 hours | Block notifications → Set timer → Single task → Break cycle |
| **Winding down** | 1–2 hours | Reduce screen brightness → Light reading → Prepare tomorrow's list → Sleep prep |
| **Hosting guests** | 3–6 hours | Clean house → Prepare food → Set up space → Greet → Serve → Clean after |
| **Travel day** | Full day | Pack → Documents check → Transport → Check-in → Settle → Explore |
| **Deadline crunch** | Variable | Prioritize → Eliminate distractions → Execute → Review → Submit |

### 4.4 Social-Context Bundles

Who is present changes which bundles activate.

| Social Context | Bundle Modification |
|---|---|
| **Alone** | Full autonomy — all personal bundles available |
| **With children** | Safety actions get urgency boost; patience-requiring actions shift to background |
| **In a meeting** | Mute notifications; note-taking active; speaking-turn awareness |
| **With a stranger** | Social courtesy actions activated; private actions suppressed |
| **With close friend** | Relaxed ordering; more actions can be parallel; fewer formality constraints |

---

## 5. How the Agent Learns Bundle Geometry

### 5.1 Cold Start — Seed Bundles from Universal Patterns

On first activation, the agent has NO experience. The seed geometry (C1) provides only mathematical structure. But certain patterns are derivable from first principles:

```
From the CAUSAL FIBER alone:
  - "Preconditions must precede actions" (causality IS sequence)
  - "Preparation clusters before execution clusters" (causal asymmetry)
  - "Cleanup follows activity" (temporal fiber directionality)

From the LOGICAL FIBER alone:
  - "Contradictory actions don't co-occur" (you can't lock AND unlock)
  - "Mutually exclusive contexts suppress each other's bundles"
  
From the PROBABILISTIC FIBER alone:
  - "Low-confidence bundles are suggested, not executed"
  - "High-confidence bundles in safety contexts auto-execute"
```

These aren't "common sense" yet. They're **geometric priors** — structural biases that experience refines into actual common sense.

### 5.2 Learning from Observation

```
Day 1: Agent observes user

  08:00 — User wakes up, goes to bathroom, brushes teeth, showers
  08:30 — User goes to kitchen, makes coffee, eats breakfast
  09:00 — User sits at desk, opens laptop, checks email
  
  Agent places each (context, action, time) triple on M(t) via C3.
  No bundles yet — just scattered points in context×action space.
  
Day 7: Same pattern repeated 5× with minor variations

  Morning routine actions have CRYSTALLIZED into a dense cluster.
  Causal ordering has emerged from temporal fiber patterns.
  The "morning routine" bundle exists — not because anyone defined it,
  but because geometric density created a basin of attraction.
  
Day 30: Bundle is robust

  Agent can now PREDICT: "It's 8 AM on a weekday, you usually 
  make coffee next." The prediction is a trajectory through the
  crystallized morning-routine region of the action fiber.
  
  User skips coffee one day → density slightly decreases on that link.
  User adds "check weather" → new action placed in the bundle, 
  deforms local geometry to accommodate.
  
  The bundle is ALIVE. It grows and adapts with the user.
```

### 5.3 Learning from Correction

```
Agent suggests: "Should I set your alarm? You usually do at this time."
User: "No, I'm on vacation this week."

What happens on M(t):
  1. "Vacation" is a context-fiber position FAR from "workday"
  2. Contrast judgment: alarm DIFFERENT-FROM vacation context
     → alarm action pushed away from vacation region
  3. The alarm action remains crystallized in the workday bundle
     but now has EXPLICIT REPULSION from vacation context
  
  Next vacation, the agent does NOT suggest the alarm.
  The geometry remembers.
```

### 5.4 Learning from Failure

```
Agent groups "send email → close laptop → leave office"
User actually needed to: "send email → wait for reply → THEN close laptop"

What happens on M(t):
  1. The causal fiber between "send email" and "close laptop" was too short
  2. The "wait for reply" action gets placed BETWEEN them on the causal fiber
  3. The bundle geometry deforms: send → wait → close
  4. Temporal fiber captures: wait duration is variable (flexible region)
  
  Next time: agent suggests "send email" then WAITS,
  monitoring for the reply signal before suggesting "close laptop."
```

### 5.5 Generalization Across Users (Federated Bundle Geometry)

For actions that are truly universal (e.g., "lock door when leaving home"), multiple agents can share bundle geometry without sharing private data:

```
Agent A's bundle geometry for "leaving home": 
  {keys(0.95), wallet(0.92), phone(0.97), lock_door(0.99), check_stove(0.85)}
  
Agent B's bundle geometry for "leaving home":
  {keys(0.91), wallet(0.88), phone(0.93), lock_door(0.96), turn_off_lights(0.90)}
  
Geometric consensus (shared without raw data):
  → Union of action basins with density-weighted confidence
  → "Turn off lights" added to Agent A's bundle as low-density suggestion
  → Agent A can accept or reject — local geometry adapts
  
Privacy: Only manifold positions shared, never raw experiences.
Locality guarantee: Agent B's personal habits can't corrupt Agent A's geometry.
```

---

## 6. Activation Modes

The agent operates in three modes depending on confidence and context:

### 6.1 Autonomous Mode (Crystallized Bundles, Safe Context)

```
Trigger: Bundle density > 0.9 AND safety_critical = False AND past success rate > 95%

Agent executes the bundle without asking.

Example: 
  User says "goodnight" → Agent automatically:
    - Sets alarm for usual time
    - Enables Do Not Disturb
    - Dims smart lights
    - Locks doors
    
  All actions in a crystallized bundle with high historical success.
  No confirmation needed — the geometry is confident.
```

### 6.2 Suggestive Mode (Flexible Bundles, Normal Context)

```
Trigger: Bundle density 0.4–0.9 OR context is novel OR actions have side effects

Agent suggests the bundle and waits for confirmation.

Example:
  User opens code editor on Saturday (unusual — usually weekday only)
  Agent: "Looks like you're coding. Want me to run through your usual dev setup?"
  
  Bundle is known but context is slightly off (weekend).
  Suggest, don't execute. Learn from the response.
```

### 6.3 Observational Mode (Unknown/Sparse Geometry)

```
Trigger: Bundle density < 0.4 OR context is in UNKNOWN region OR first encounter

Agent watches silently and LEARNS.

Example:
  User starts a new hobby (woodworking). Agent has no bundle geometry for this.
  It observes: measure → cut → sand → assemble → finish
  Over time, a woodworking bundle crystallizes on the action fiber.
  
  No suggestions until geometry is dense enough.
  No interruptions. Just silent learning.
```

---

## 7. Integration with FLOW Pipeline

```
Existing FLOW:
  Experience → C3 (Annealing) → C2 (Manifold) → C5 (Flow) → C6 (Wave) → C7 (Language)

Extended FLOW with Action Grouping:

  Sensory Input ──────┐
  User Actions ───────┤
  User Directives ────┤
  Environment Data ───┤
  Social Signals ─────┘
           │
           ▼
  ┌─────────────────────┐
  │  CONTEXT RECOGNIZER  │  → Context fiber position on M(t)
  └──────────┬──────────┘
             │
             ▼
  ┌─────────────────────┐
  │  BUNDLE GEOMETRY     │  → Activated action bundles (ranked)
  │  ENGINE              │
  └──────────┬──────────┘
             │
             ├──────────────────────────────────────────┐
             │                                          │
             ▼                                          ▼
  ┌─────────────────────┐                   ┌─────────────────────┐
  │  ACTION EXECUTOR     │                   │  C5 FLOW ENGINE      │
  │  (Autonomous mode)   │                   │  (Language response)  │
  │                      │                   │                      │
  │  Executes high-      │                   │  Navigates M(t) to   │
  │  confidence bundles  │                   │  explain what it's   │
  │  directly            │                   │  doing and why       │
  └──────────┬──────────┘                   └──────────┬──────────┘
             │                                          │
             ▼                                          ▼
  ┌─────────────────────┐                   ┌─────────────────────┐
  │  OUTCOME OBSERVER    │                   │  C6 → C7             │
  │                      │                   │  Resonance → Speech  │
  │  Did it work?        │                   │                      │
  │  User satisfied?     │                   │  "I've set your      │
  │  Unexpected result?  │                   │   alarm and locked   │
  │                      │                   │   up because you     │
  │  Feed back to C3     │                   │   said goodnight"    │
  │  as new experience   │                   │                      │
  └──────────┬──────────┘                   └──────────────────────┘
             │
             ▼
  ┌─────────────────────┐
  │  COMMON SENSE        │
  │  GEOMETRY            │
  │                      │
  │  Plausibility check  │
  │  Missing action?     │
  │  Safety critical?    │
  │  Should I warn?      │
  └─────────────────────┘
```

---

## 8. Bundle Data Types

```python
@dataclass
class ContextSignals:
    """Raw input signals from all sources."""
    environment: np.ndarray      # Sensor/device/location features
    user_action: np.ndarray      # What the user just did
    directive: np.ndarray | None # What the user said (if anything)
    time_features: np.ndarray    # Time of day, day of week, season
    social_state: np.ndarray     # Who is present, relationship vectors
    history: list[np.ndarray]    # Recent action sequence (last N steps)

@dataclass  
class ActionBundle:
    """A context-activated group of related actions."""
    context_position: np.ndarray   # Where on context fiber this bundle lives
    actions: list[Action]          # Ordered action list (causal fiber ordering)
    density: float                 # How crystallized (0=novel, 1=habitual)
    urgency: float                 # Time-sensitivity (temporal fiber gradient)
    flexibility: float             # Can actions be skipped/reordered? (1−density)
    safety_flags: list[int]        # Which action indices are safety-critical
    
@dataclass
class Action:
    """A single action within a bundle."""
    label: str                     # Human-readable name
    position: np.ndarray           # Position on action fiber
    preconditions: list[str]       # Actions that must come before (causal fiber)
    effects: list[str]             # What changes after execution
    confidence: float              # Manifold density at this action-point
    context_coupling: float        # Cross-fiber resonance with current context
    
@dataclass
class BundleActivation:
    """Result of bundle activation for a given context."""
    bundles: list[ActionBundle]    # Ranked by coupling strength
    mode: str                      # "autonomous" | "suggestive" | "observational"  
    explanation_trajectory: Trajectory  # C5 trajectory explaining WHY these bundles
    missing_actions: list[Action]  # Actions expected but not yet completed
    warnings: list[str]            # Safety-critical actions that haven't fired
```

---

## 9. What This Enables

### 9.1 Proactive Intelligence

The agent doesn't wait for commands. It anticipates needs from context geometry:

```
Context: User picks up car keys at 8:47 AM on a Tuesday
  → Context fiber: "leaving for work" (crystallized pattern)
  → Bundle: commute preparation
  → Agent proactively: checks traffic, starts car warm-up, queues podcast

Context: User opens a blank document at 3 PM
  → Context fiber: "writing task" (moderate density)  
  → Bundle: writing preparation
  → Agent: pulls up relevant research notes, sets focus mode, hides notifications
```

### 9.2 Cascading Awareness

One action triggers a cascade of related preparations:

```
User: "I'm having 6 people over for dinner Saturday"
  
  CASCADE (each action triggers next via causal fiber):
  
  Thursday:  → Suggest grocery list (based on usual cooking patterns)
  Friday:    → Remind to clean living room (hosting bundle, -1 day)
  Saturday AM: → Start prep timeline (cooking bundle, scaled for 6)
  Saturday PM: → Set table suggestions (hosting bundle, social context)
               → Adjust thermostat (guests = more body heat)
               → Queue background music (social gathering context)
               → Estimated prep-to-serve timeline
  After:     → Cleanup bundle activates when guests leave
  
  Each step is a TRAJECTORY through the action fiber,
  ordered by the temporal and causal fibers.
  The agent learned this pattern from observing hosting events.
```

### 9.3 Adaptive Personalization

The same agent becomes completely different for different users:

```
User A (minimalist): Morning bundle → {coffee, check phone, leave}
  Geometry: sparse action fiber, few crystallized bundles, high flexibility
  
User B (structured): Morning bundle → {alarm, stretch, shower, skincare, 
  outfit choice, breakfast, vitamins, pack bag, check weather, leave}
  Geometry: dense action fiber, many crystallized bundles, rigid ordering
  
Same architecture. Different geometry. 
The manifold SHAPES ITSELF to each human.
```

### 9.4 Graceful Degradation

When context is ambiguous, bundles compete:

```
User picks up keys at 2 AM (unusual)
  
  Bundle candidates:
    - "Leaving for work" → coupling: 0.3 (wrong time)
    - "Emergency" → coupling: 0.5 (unusual hour deviation)
    - "Insomnia walk" → coupling: 0.2 (if user has this pattern)
    - "Unknown" → coupling: 0.7 (no strong match)
    
  Mode: OBSERVATIONAL (highest coupling is "Unknown")
  Agent watches silently. Whatever happens becomes a new experience.
  
  If this happens 3 more times → "2 AM walk" crystallizes as a new bundle.
```

### 9.5 Explainable Grouping

Every bundle activation has a trajectory audit trail:

```
User: "Why did you turn off the stove?"

Agent trajectory: 
  Context(leaving_kitchen) → Bundle(kitchen_safety) → 
  Action(check_stove, density=0.97) → Action(turn_off, safety_critical=True)
  
  "I turned off the stove because you left the kitchen and it was still on. 
   This is something that has been important in your past kitchen routines, 
   and the stove being on while the kitchen is empty is flagged as 
   safety-critical in your experience history."
   
The explanation is the trajectory. Every decision is geometrically traceable.
```

---

## 10. Constraints (What This System Must NOT Do)

```
1. NEVER execute safety-critical actions without crystallized geometry
   → New bundle with "delete all files" CANNOT be autonomous
   → Must accumulate MANY successful observations first

2. NEVER override explicit user commands
   → User says "leave the lights on" → bundle's "lights off" action is SUPPRESSED
   → The suppression deforms geometry: lights-off coupling to this context WEAKENS

3. NEVER share raw experience data between users  
   → Federated bundle geometry shares POSITIONS not EXPERIENCES
   → Locality guarantee: User A's personal patterns can't reach User B

4. NEVER act on UNKNOWN-region geometry in autonomous mode
   → Only CRYSTALLIZED bundles (density > 0.9) can auto-execute
   → New situations ALWAYS start in observational mode

5. NEVER persist sensitive actions in bundle geometry
   → Passwords, financial transactions, medical actions
   → These require fresh confirmation every time (density CAP at 0.5)

6. ALWAYS allow bundle interruption
   → User can stop any cascade at any point
   → Interruption is itself an experience that deforms the bundle
```

---

## 11. Implementation Roadmap

| Phase | Component | LOC | Depends On |
|---|---|---|---|
| **A** | ContextSignals data types + Context Fiber (32D extension) | ~400 | Core manifold |
| **B** | ContextRecognizer — signal fusion + context placement | ~800 | Phase A |
| **C** | Action Fiber (32D extension) + Action data types | ~400 | Core manifold |
| **D** | BundleGeometryEngine — activation, extension, splitting | ~1,200 | Phase A, C |
| **E** | CommonSenseGeometry — plausibility, missing actions, safety | ~800 | Phase D |
| **F** | Activation mode controller (autonomous/suggestive/observational) | ~600 | Phase D, E |
| **G** | Outcome observer + learning feedback loop | ~500 | Phase D, F |
| **H** | Cascade planner (multi-step temporal chains) | ~700 | Phase D, temporal fiber |
| **I** | Integration with C5/C6/C7 for explanation rendering | ~400 | All above |
| **J** | Test suite + demo scenarios | ~1,000 | All above |

**Total: ~6,800 lines. Zero weights. Zero rules. Pure geometric learning.**

---

## 12. The Deeper Point

This is not a feature. This is what separates a **tool** from a **mind.**

A tool waits for instructions. A mind anticipates, groups, prepares, and learns from the rhythm of life. The Contextual Action Grouping Engine doesn't add "smart automation" to FLOW — it gives FLOW the geometric substrate for **situated intelligence**: the ability to exist in a context, understand what that context demands, and act accordingly.

Every human has this. No AI system currently does. Transformers can't have it because their context is a finite token window that resets every conversation. FLOW's context is a manifold that crystallizes over a lifetime.

The agent doesn't follow a script. It grows a geometric understanding of what it means to be helpful in THIS life, for THIS person, in THIS moment.

That understanding is shape. And shape is all you need.
