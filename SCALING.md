# FLOW — Scaling & Deployment Strategy

**Status:** Active roadmap document  
**Audience:** Cofounders, contributors, infrastructure engineers  
**Last updated:** 2026-03-18

---

## 0. Where We Are Today

| Metric | Current value |
|---|---|
| Manifold dimension | 104D (fixed by design) |
| Points on M(t) | 81 seed + ~50 learned ≈ 137 |
| Vocabulary entries | 32 hand-crafted + ~68 Phase 7 = 100 |
| Total source | 10,265 lines across 54 files |
| Tests | 674/674 passing |
| Runtime hardware | MacBook CPU, 8GB RAM |
| Output quality | Proof-of-architecture (poor fluency) |

The architecture (C1–C7) is fully integrated and producing output. The bottleneck is **not compute** — it is manifold size (too few concepts) and vocabulary depth (too few expression entries). Scaling means growing these two things on real hardware.

---

## 1. Where Knowledge Lives — The Persistence Story

FLOW does not have "model weights" to checkpoint. Knowledge lives in four places, each requiring its own persistence strategy.

### 1.1  Knowledge Storage Map

```
┌──────────────────────────────────────────────────────────────────────┐
│                     WHERE KNOWLEDGE IS STORED                        │
├──────────────────────┬───────────────────────────────────────────────┤
│ Layer                │ What it contains                              │
├──────────────────────┼───────────────────────────────────────────────┤
│ 1. Seed Manifold M₀  │ Mathematical skeleton — causality, logic,     │
│    (IMMUTABLE)       │ probability, similarity. 81 archetypal points.│
│                      │ Derived from first principles. Never changes. │
│                      │ Recomputable in <0.01s. No need to persist.   │
├──────────────────────┼───────────────────────────────────────────────┤
│ 2. Living Manifold   │ All learned knowledge. Every concept ever     │
│    M(t)              │ placed, every deformation ever applied.       │
│    (MUTABLE)         │ THIS IS THE MODEL.                            │
│                      │                                               │
│    Contains:         │   _points:  Dict[str, np.ndarray(104,)]       │
│                      │   _state.deformation: accumulated φ(t)        │
│                      │   _state.density: ρ(t) per point              │
│                      │   _state.curvature: κ(t) per point            │
│                      │   _state.t: manifold time                     │
│                      │   _state.n_writes: write counter              │
│                      │   _geodesic: kNN graph (rebuildable)          │
│                      │   _kdtree: spatial index (rebuildable)        │
├──────────────────────┼───────────────────────────────────────────────┤
│ 3. Vocabulary Store  │ Expression entries for C7 rendering.          │
│    (APPEND-ONLY)     │ wave_profiles, texts, register, rhythm, etc.  │
│                      │ Currently serialised as .npz via              │
│                      │ VocabularyStore.save() / .load()              │
│                      │ This is the system's language capability.      │
├──────────────────────┼───────────────────────────────────────────────┤
│ 4. Temperature       │ Annealing schedule state: T₀, λ, T_floor,    │
│    Schedule          │ current time t. Small scalar state.           │
│    (RECOVERABLE)     │ Can be approximately reconstructed from       │
│                      │ n_writes if lost.                             │
└──────────────────────┴───────────────────────────────────────────────┘
```

### 1.2  What Must Be Saved vs What Can Be Rebuilt

```
MUST SAVE (knowledge is lost otherwise):
  ✦ _points          — label → 104D position vector for every concept
  ✦ _state.density   — label → float density per point
  ✦ Vocabulary .npz  — all ExpressionEntry wave profiles + texts

CAN REBUILD ON LOAD:
  ✦ M₀ seed geometry — deterministic derivation, <0.01s
  ✦ KD-tree           — rebuilt on first query from _points
  ✦ kNN geodesic graph — rebuilt lazily from _points
  ✦ Curvature κ(t)    — recomputable from kNN distances
  ✦ Composer / metrics — stateless, derived from geometry classes

NICE TO SAVE (avoids redundant recomputation):
  ✦ _state.deformation — accumulated displacement history per point
  ✦ _state.t, n_writes — manifold time / write counter
  ✦ Temperature schedule state (T₀, λ, T_floor, current_t)
```

### 1.3  Serialisation Format: ManifoldSnapshot

The manifold must be serialised as a single `.npz` file. No pickle, no Python object serialisation — pure numpy arrays, matching the pattern already established by `VocabularyStore`.

```python
# ── Save ──────────────────────────────────────────────────────────
ManifoldSnapshot.save(manifold, "manifold_v1.npz")

# Contents of manifold_v1.npz:
#   labels          : object array  (N,) — concept labels
#   positions       : float64       (N, 104) — current positions
#   densities       : float64       (N,) — ρ(t) per point
#   deformations    : float64       (N, 104) — accumulated φ(t)
#   curvatures      : float64       (N,) — κ(t) per point
#   manifold_time   : float64       (1,) — current t
#   n_writes        : uint64        (1,) — write counter
#   format_version  : uint32        (1,) — migration support
#   dimension       : uint32        (1,) — should always be 104

# ── Load ──────────────────────────────────────────────────────────
manifold = ManifoldSnapshot.load("manifold_v1.npz")
# Internally:
#   1. Recompute M₀ via SeedGeometryEngine().build()
#   2. Create LivingManifold(M₀)
#   3. Overwrite _points, _state from .npz arrays
#   4. KD-tree and geodesic graph rebuild lazily on first query
```

### 1.4  Size Estimates

| Manifold size | .npz file size | RAM at runtime |
|---|---|---|
| 137 points (today) | ~120 KB | ~2 MB |
| 10K points | ~8 MB | ~50 MB |
| 100K points | ~80 MB | ~500 MB |
| 1M points | ~800 MB | ~5 GB |
| 10M points | ~8 GB | ~50 GB |

Vocabulary store scales linearly:

| Entries | .npz file size | RAM at runtime |
|---|---|---|
| 100 (today) | ~21 KB | ~1 MB |
| 10K | ~4 MB | ~50 MB |
| 100K | ~40 MB | ~500 MB |
| 1M | ~400 MB | ~5 GB |

**Total model artifact at production scale (1M concepts + 100K vocab):**

```
manifold_v1.npz  ~800 MB
vocabulary.npz   ~40 MB
────────────────────────
Total             ~840 MB   ← this replaces a "model checkpoint"
```

Compare: GPT-2-small = 548 MB, LLaMA-7B = 13 GB, LLaMA-70B = 130 GB. A FLOW manifold with 1M concepts and 100K expression entries is smaller than GPT-2.

---

## 2. Platform Strategy: HuggingFace Hub

FLOW is not a PyTorch model, but HuggingFace Hub stores arbitrary model artifacts.

### 2.1  Repository Structure on HuggingFace

```
yourname/flow-geometric-manifold/
├── README.md                    # Model card (architecture, usage, constraints)
├── config.json                  # Pipeline hyperparameters (T₀, λ, T_floor, etc.)
├── manifold_v1.npz              # Full manifold state M(t)
├── vocabulary.npz               # C7 expression vocabulary
├── src/                         # Source code (pip-installable via git)
│   └── ... (full FLOW source tree)
└── requirements.txt             # numpy, scipy, networkx, pytest
```

### 2.2  HuggingFace Integration API (to be built)

```python
# ── Upload ─────────────────────────────────────────────────────────
from flow.hub import push_to_hub

pipeline = GEOPipeline()
# ... learn, contrast, build vocabulary ...
push_to_hub(pipeline, repo_id="yourname/flow-v1", vocabulary_path="vocabulary.npz")
# Saves: manifold_v1.npz, vocabulary.npz, config.json

# ── Download & reload ──────────────────────────────────────────────
from flow.hub import load_from_hub

pipeline = load_from_hub("yourname/flow-v1")
result = pipeline.query(vec, label="what causes gravity?")
print(result.text)
```

### 2.3  HuggingFace Spaces — Live Demo

A Gradio app on HF Spaces (free CPU tier) can serve FLOW at its current scale:

```python
import gradio as gr
from flow.hub import load_from_hub

pipeline = load_from_hub("yourname/flow-v1")

def answer(question: str) -> str:
    # Convert question to manifold query via nearest concept
    labels = pipeline.manifold.labels
    # Simple: use first concept matching any word in the question
    vec = pipeline.manifold.position(labels[0])  # placeholder — needs text→vector
    result = pipeline.query(vec, label=question)
    return result.text

demo = gr.Interface(fn=answer, inputs="text", outputs="text",
                    title="FLOW — Geometric Causal Reasoning",
                    description="Weight-free, token-free reasoning on a Riemannian manifold")
demo.launch()
```

### 2.4  What HuggingFace Cannot Do

HuggingFace's `transformers` pipeline API assumes tokenizer + model forward pass. FLOW uses neither. Integration options:

| HF Feature | Compatible? | Notes |
|---|---|---|
| Hub (file hosting) | **Yes** | .npz artifacts, code, model card |
| Spaces (Gradio) | **Yes** | Free CPU tier is sufficient |
| Datasets | **Yes** | Publish vocabulary.npz as a dataset |
| `transformers` library | **No** | No tokenizer, no forward() |
| `pipeline("text-generation")` | **No** | FLOW doesn't do token prediction |
| Inference API | **No** (custom) | Would need a custom inference handler |
| GGUF/ONNX export | **No** | Not a neural network |

---

## 3. Scaling Roadmap: MacBook → Kaggle → Cloud → Custom

### Phase A — Kaggle CPU (Immediate, Free)

**Hardware:** 4 CPU cores, 30 GB RAM, 9-hour sessions  
**Cost:** Free  
**What it enables:** 30× more RAM than a MacBook = room for a 100K-point manifold and 100K vocabulary entries

**Tasks to run on Kaggle:**

```
1. VOCABULARY GROWTH (highest impact)
   Feed a real corpus (Wikipedia, BookCorpus, OpenWebText) through
   the Phase 7 VocabularyBuilder pipeline.
   
   Target: 50,000–100,000 expression entries from 1M+ tokens.
   Estimated time: 2–4 hours on Kaggle CPU.
   Output: vocabulary_100k.npz (~40 MB)

2. MANIFOLD GROWTH
   Place 10K–50K concepts on M(t) via C3/C4.
   Source: vocabulary words + domain-specific concept lists (medical,
   legal, scientific, common-sense).
   
   Target: 50,000 concepts on the manifold.
   Estimated time: 1–3 hours on Kaggle CPU.
   Output: manifold_50k.npz (~40 MB)

3. EVALUATION AT SCALE
   Run PipelineEvaluator.run_suite() with 100+ queries across
   diverse domains. Collect coherence, causal, locality metrics.
   
   Target: Statistical confidence in output quality metrics.
```

**Kaggle kernel outline:**

```python
# ── Kaggle Notebook Cell 1: Install ──────────────────────────────
!git clone https://github.com/Unseengap/FLOW
!pip install numpy scipy networkx

# ── Cell 2: Build pipeline + feed corpus ──────────────────────────
import sys; sys.path.insert(0, "FLOW")
from src.phase5 import GEOPipeline
from src.vocabulary import VocabularyBuilder
from src.phase3.annealing_engine.experience import Experience

pipeline = GEOPipeline(T0=1.0, lambda_=0.01, T_floor=0.05)

# Feed a real corpus
from datasets import load_dataset
ds = load_dataset("wikipedia", "20220301.simple", split="train")

builder = VocabularyBuilder(
    pipeline.manifold, pipeline._annealing, pipeline._contrast_engine,
    window=5, min_count=10, v_max=50000
)
for i, article in enumerate(ds):
    builder.feed(article["text"])
    if i >= 50000:
        break

print(f"Tokens fed: {builder.n_tokens_fed}")

# ── Cell 3: Build & save ─────────────────────────────────────────
n = builder.build_and_save("/kaggle/working/vocabulary_large.npz")
print(f"Vocabulary entries built: {n}")

# Save manifold state
ManifoldSnapshot.save(pipeline.manifold, "/kaggle/working/manifold_large.npz")
```

### Phase B — Kaggle GPU (Optional, Targeted)

**Hardware:** NVIDIA T4 (16GB VRAM) or P100 (16GB VRAM) + 30 GB RAM  
**Cost:** Free (30 hrs/week GPU quota)  
**What it enables:** GPU-accelerated batch distance computation only

GPUs help FLOW in exactly **two** operations:

```
1. BATCH DISTANCE MATRICES (resonance accumulator)
   The O(n²) all-pairs excitation kernel in ResonanceAccumulator
   maps directly to GPU matrix operations.

   # CPU: 200 steps × 200 steps × 104D = 4.2M distance ops
   # GPU: single batched matmul — 50× faster

   import cupy as cp
   positions_gpu = cp.asarray(positions)  # (n_steps, 104)
   dists = cp.linalg.norm(
       positions_gpu[:, None, :] - positions_gpu[None, :, :], axis=-1
   )

2. BATCH KNN LOOKUP (manifold nearest-neighbour)
   At 50K+ points, kNN queries dominate runtime.
   FAISS-GPU can search 100K 104D vectors in <1ms.
   
   import faiss
   res = faiss.StandardGpuResources()
   index = faiss.GpuIndexFlatL2(res, 104)
   index.add(all_positions.astype('float32'))
   dists, idxs = index.search(query.reshape(1, -1).astype('float32'), k=10)
```

**What GPUs DO NOT help with:**

| Operation | Why GPU doesn't help |
|---|---|
| SDE integration (C5) | Inherently sequential — each step depends on previous |
| Geodesic graph (Dijkstra) | Graph algorithm — CPU-bound, irregular memory access |
| Expression rendering (C7) | String manipulation and template matching |
| Annealing loop (C3) | Sequential per-experience processing |
| Contrast engine (C4) | Sequential pairwise judgments |

### Phase C — Cloud Instance (AWS / GCP / Azure)

**Hardware:** 64-core CPU + 256 GB RAM + optional GPU  
**Cost:** ~$2–5/hour (CPU), ~$5–15/hour (GPU)  
**What it enables:** 1M+ point manifold, production-grade throughput

**Required code changes before cloud deployment:**

```
CRITICAL (must fix before >10K points):
┌─────────────────────────────────────────────────────────────────┐
│ 1. SPATIAL INDEXING                                              │
│    Replace: scipy.spatial.KDTree (rebuilt from scratch)          │
│    With:    scipy.spatial.cKDTree (C-accelerated, 10–50×)       │
│    Files:   src/phase2/living_manifold/manifold.py               │
│    Effort:  ~20 lines changed                                    │
├─────────────────────────────────────────────────────────────────┤
│ 2. VOCABULARY MATCHING                                           │
│    Replace: O(V) linear scan in ResonanceMatcher.match()        │
│    With:    FAISS IndexFlatIP for approximate nearest neighbor    │
│    Files:   src/phase1/expression/matcher.py                     │
│    Effort:  ~50 lines added                                      │
├─────────────────────────────────────────────────────────────────┤
│ 3. GEODESIC GRAPH                                                │
│    Replace: O(n²) full rebuild on every mutation                 │
│    With:    Incremental kNN update (add/remove edges locally)    │
│    Files:   src/phase2/living_manifold/geodesic.py               │
│    Effort:  ~100 lines rewritten                                 │
├─────────────────────────────────────────────────────────────────┤
│ 4. DEFORMATION RANGE QUERY                                       │
│    Replace: O(n) scan of all points in LocalDeformation.apply() │
│    With:    cKDTree.query_ball_point() pre-filter                │
│    Files:   src/phase2/living_manifold/deformation.py            │
│    Effort:  ~30 lines changed                                    │
└─────────────────────────────────────────────────────────────────┘

RECOMMENDED (improves throughput significantly):
┌─────────────────────────────────────────────────────────────────┐
│ 5. BATCH ANNEALING                                               │
│    Current: Sequential process(experience) — one at a time       │
│    Target:  Parallel batch placement with conflict resolution    │
│    Effort:  ~200 lines new code                                  │
├─────────────────────────────────────────────────────────────────┤
│ 6. MANIFOLD PERSISTENCE (ManifoldSnapshot)                       │
│    Current: No save/load for manifold state                      │
│    Target:  .npz serialisation matching VocabularyStore pattern  │
│    Effort:  ~150 lines new module                                │
├─────────────────────────────────────────────────────────────────┤
│ 7. MULTI-QUERY PARALLELISM                                       │
│    Current: Single-threaded query processing                     │
│    Target:  Concurrent reads via threading (manifold is mostly   │
│             read-only during query phase — readers don't conflict)│
│    Effort:  ~100 lines + read/write lock                         │
└─────────────────────────────────────────────────────────────────┘
```

### Phase D — Custom Compute (Long-Term Vision)

**The fundamental hardware mismatch:** current accelerators are matrix-multiplication engines. FLOW needs:

| Operation | Optimal hardware | Current best available |
|---|---|---|
| High-dimensional kNN | Spatial accelerators | FAISS on GPU (good enough) |
| Graph traversal | Neuromorphic (event-driven) | Multi-core CPU |
| Wave accumulation | Optical / analog compute | Batched GPU matmul |
| Local deformation | Spatial computing | FPGA / custom ASIC |
| SDE integration | Low-latency sequential | Single CPU core |

**Realistic near-term substitute (2026):**

A single AWS `c7a.16xlarge` (64 vCPU AMD EPYC, 128 GB RAM) with FAISS-CPU can handle:
- 1M manifold points
- 100K vocabulary entries
- ~50 queries/second
- Cost: ~$2.50/hour

For higher throughput, a `g5.xlarge` (NVIDIA A10G, 24GB VRAM) adds GPU-accelerated distance matrices:
- ~200 queries/second for short trajectories
- Cost: ~$1.00/hour (spot pricing)

---

## 4. What the O(n²) Bottlenecks Actually Look Like

The following operations currently have quadratic scaling. Here is what happens as the manifold grows:

### 4.1  Geodesic Graph Rebuild

```
Current: _rebuild_graph() computes all-pairs distances → O(n²)

n=100    :  10,000 distance ops   →  <1ms     ✓
n=1,000  :  1,000,000 ops        →  ~50ms    ✓
n=10,000 :  100,000,000 ops      →  ~5s      ⚠ (per mutation)
n=100,000:  10,000,000,000 ops   →  ~8 min   ✗ (per mutation)

Fix: Incremental graph update — only recompute edges for the
     mutated point + its k neighbours. O(k·n) per mutation.
     With k=8 and n=100K this is 800K ops instead of 10B.
```

### 4.2  Resonance Accumulator

```
Current: all-pairs excitation/harmonic kernel → O(n_steps²)

steps=50   :  2,500 kernel evaluations  →  <1ms    ✓
steps=200  :  40,000 evaluations        →  ~5ms    ✓
steps=1000 :  1,000,000 evaluations     →  ~200ms  ⚠
steps=5000 :  25,000,000 evaluations    →  ~5s     ✗

Fix: Excitation kernel has exponential decay — points beyond 3σ
     contribute ≈0. Use cKDTree range query to find only nearby
     steps. Reduces to O(n·k_effective) where k_effective << n.
```

### 4.3  Deformation Scan

```
Current: LocalDeformation.apply() iterates ALL manifold points → O(n)

n=100    :  100 distance checks   →  <1ms     ✓
n=10,000 :  10,000 checks         →  ~2ms     ✓
n=100,000:  100,000 checks        →  ~20ms    ⚠ (per deformation)
n=1,000,000: 1,000,000 checks     →  ~200ms   ✗ (per deformation)

Fix: cKDTree.query_ball_point(centre, 3*sigma) pre-filters.
     With locality_radius=1.0 and 3σ cutoff, typically 20–50
     points are in range regardless of total manifold size.
     Reduces to O(k_local) ≈ O(1) amortised.
```

---

## 5. Scaling Priority Order

What to do first, ranked by impact / effort ratio:

```
PRIORITY 1 — VOCABULARY GROWTH ON KAGGLE           [Impact: ★★★★★]
  Feed a real corpus, build 50K–100K entries.        [Effort: ★☆☆☆☆]
  This is the single largest quality improvement.
  No code changes needed — Phase 7 pipeline works.
  Just needs a bigger machine and a bigger corpus.

PRIORITY 2 — MANIFOLD PERSISTENCE                  [Impact: ★★★★★]
  Build ManifoldSnapshot.save() / .load().            [Effort: ★★☆☆☆]
  Without this, every run starts from 81 seed points.
  ~150 lines of new code, following VocabularyStore pattern.

PRIORITY 3 — FAISS VOCABULARY MATCHING              [Impact: ★★★★☆]
  Drop-in replacement for linear scan in matcher.     [Effort: ★☆☆☆☆]
  ~50 lines changed. Makes 100K vocab feasible.
  faiss-cpu is a pip install, no ML framework.

PRIORITY 4 — cKDTree + RANGE QUERY DEFORMATION     [Impact: ★★★☆☆]
  Replace KDTree with cKDTree everywhere.             [Effort: ★☆☆☆☆]
  Replace deformation full-scan with range query.
  ~50 lines changed across 2 files.

PRIORITY 5 — INCREMENTAL GEODESIC GRAPH            [Impact: ★★★☆☆]
  Stop rebuilding the full kNN graph on mutations.    [Effort: ★★★☆☆]
  ~100 lines rewritten.

PRIORITY 6 — HUGGINGFACE HUB INTEGRATION           [Impact: ★★★☆☆]
  push_to_hub() / load_from_hub() + Gradio Space.    [Effort: ★★☆☆☆]
  ~200 lines of new hub integration module.

PRIORITY 7 — MULTI-QUERY PARALLELISM               [Impact: ★★☆☆☆]
  Reader/writer lock on the manifold.                 [Effort: ★★★☆☆]
  Only matters when serving multiple users.
```

---

## 6. Dependency Map — What Unlocks What

```
                 ┌──────────────────────────┐
                 │  PRIORITY 2              │
                 │  Manifold Persistence    │
                 │  (ManifoldSnapshot)      │
                 └──────────┬───────────────┘
                            │ unlocks
        ┌───────────────────┼───────────────────┐
        ▼                   ▼                   ▼
 ┌──────────────┐  ┌───────────────┐  ┌────────────────┐
 │  PRIORITY 1  │  │  PRIORITY 6   │  │  Long-running  │
 │  Kaggle      │  │  HuggingFace  │  │  growth jobs   │
 │  Vocab Build │  │  Hub + Spaces │  │  (overnight)   │
 └──────┬───────┘  └───────────────┘  └────────────────┘
        │ unlocks
        ▼
 ┌──────────────┐
 │  PRIORITY 3  │
 │  FAISS index │  (matching 100K entries in <1ms)
 └──────┬───────┘
        │ unlocks
        ▼
 ┌──────────────────────────────────────────────┐
 │  PRIORITY 4 + 5                               │
 │  cKDTree + incremental geodesic               │
 │  (manifold with 100K+ points)                 │
 └──────┬───────────────────────────────────────┘
        │ unlocks
        ▼
 ┌──────────────────────────────────────────────┐
 │  Cloud deployment                             │
 │  1M concepts, 100K vocab, multi-user serving  │
 └──────────────────────────────────────────────┘
```

---

## 7. What "Production" Means for FLOW

FLOW is not a chatbot. It is a reasoning engine. Production criteria are different:

### 7.1  Minimum Viable Product (MVP)

```
✓  Manifold with 50K+ concepts covering 3+ real-world domains
✓  Vocabulary with 50K+ expression entries from real text
✓  Manifold persistence — save/load without data loss
✓  Sub-second query latency on a single CPU core
✓  Output text that is grammatically correct and topically relevant
✓  Hosted on HuggingFace Spaces with Gradio interface
✓  Downloadable manifold + vocabulary artifacts on HuggingFace Hub
```

### 7.2  Full Production

```
✓  1M+ concepts across all major knowledge domains
✓  100K+ vocabulary entries with diverse grammatical structures
✓  FAISS-indexed vocabulary matching (<1ms per segment)
✓  Incremental geodesic graph (no O(n²) rebuilds)
✓  Multi-user serving with read/write locking
✓  API endpoint (FastAPI or similar)
✓  Manifold versioning — track M(t) over time like git
✓  Continuous ingestion — new experiences absorbed in real-time
✓  Evaluation dashboard — coherence / causal / locality metrics live
```

### 7.3  What FLOW Replaces vs What It Doesn't

```
FLOW CAN REPLACE:
  ✦ Knowledge graph query engines (it IS a knowledge graph — geometric)
  ✦ Causal reasoning modules in hybrid AI systems
  ✦ Static embedding stores (Word2Vec / GloVe) — the manifold is dynamic
  ✦ Simple Q&A over structured domains

FLOW CANNOT REPLACE (today):
  ✦ Large language models for open-ended generation
  ✦ Translation systems
  ✦ Code generation
  ✦ Creative writing

FLOW'S UNIQUE VALUE:
  ✦ Zero catastrophic forgetting — old knowledge never moves
  ✦ One-shot concept placement — no retraining
  ✦ Fully interpretable reasoning — the trajectory IS the explanation
  ✦ Causal reasoning is structural, not statistical
  ✦ No training phase — growth IS the operating mode
  ✦ Entire "model" is <1 GB at 1M concepts
```

---

## 8. Cost Comparison

| Platform | Cost | Manifold scale | Vocab scale | Queries/sec |
|---|---|---|---|---|
| MacBook Air M1 | $0 | 1K | 1K | ~5 |
| Kaggle CPU (free) | $0 | 50K | 100K | ~10 |
| Kaggle GPU (free) | $0 (30h/wk) | 100K | 100K | ~50 |
| HF Spaces (free CPU) | $0 | 10K | 50K | ~3 (demo) |
| AWS c7a.4xlarge | $0.60/hr | 500K | 100K | ~50 |
| AWS c7a.16xlarge | $2.50/hr | 1M+ | 500K | ~200 |
| AWS g5.xlarge (GPU) | $1.00/hr spot | 1M+ | 500K | ~500 |

**The key insight: FLOW is extremely cheap to serve.** A 1M-concept manifold fits in 5 GB of RAM. There are no GPU requirements for inference — GPUs only accelerate batch vocabulary building and distance matrices. A $0.60/hour CPU instance can serve production traffic.

Compare: serving LLaMA-70B requires ~$5–10/hour on A100 GPUs.

---

## 9. Implementation Checklist

### Immediate (this week)

- [ ] Build `ManifoldSnapshot` class (`src/persistence/snapshot.py`)
  - [ ] `save(manifold, path)` → .npz with all mutable state
  - [ ] `load(path)` → reconstituted `LivingManifold`
  - [ ] Round-trip test: save → load → validate positions match
- [ ] Add `faiss-cpu` to requirements.txt (optional dependency)
- [ ] Create Kaggle notebook for large-scale vocabulary building
- [ ] Push FLOW repo + artifacts to HuggingFace Hub

### Next sprint

- [ ] Replace `KDTree` with `cKDTree` in `manifold.py`
- [ ] Add FAISS index to `ResonanceMatcher` for >1K vocabulary
- [ ] Pre-filter `LocalDeformation.apply()` with `cKDTree.query_ball_point()`
- [ ] Build Gradio demo for HF Spaces
- [ ] Run first large-scale vocabulary build on Kaggle (target: 50K entries)

### Following sprint

- [ ] Incremental geodesic graph (no full rebuild)
- [ ] `GEOPipeline.save()` / `.load()` convenience methods
- [ ] Manifold versioning (save checkpoints at intervals)
- [ ] Multi-query thread safety (reader/writer lock)
- [ ] Cloud deployment playbook (Dockerfile + FastAPI wrapper)

---

## 10. Architecture Constraints on Scaling

All scaling work MUST preserve these six constraints:

| Constraint | Scaling implication |
|---|---|
| **NO WEIGHTS** | FAISS is a spatial index, not a learned model. cKDTree is a data structure. No violation. |
| **NO TOKENS** | Vocabulary entries are geometric wave profiles, not token IDs. FAISS searches in continuous space. |
| **NO TRAINING** | Large-scale vocabulary building is continuous C3/C4 operation — the same growth mode, just more data. |
| **LOCAL UPDATES** | cKDTree range queries enforce locality more precisely than brute-force scan. Improvement, not violation. |
| **CAUSALITY FIRST** | Unchanged — causal fiber structure is preserved at any scale. |
| **SEPARATION** | FAISS index is internal to C7 matcher. ManifoldSnapshot is a persistence utility. No component boundary crossed. |

**Forbidden at any scale:**
- No PyTorch, TensorFlow, or JAX
- No tokenizer
- No gradient computation
- No loss function
- No attention mechanism
- No embedding lookup table (the manifold is the embedding)
