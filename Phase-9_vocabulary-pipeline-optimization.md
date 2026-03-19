# Phase 9 — Vocabulary Pipeline Optimization

**Completed**: March 19, 2026  
**Status**: ✅ 721/722 tests passing (0 regressions; 1 skipped — optional FAISS)

---

## Scope

Phase 9 addresses the performance bottleneck in the Phase 7 `VocabularyBuilder.build_and_save()` pipeline, which was taking 2–4 hours on Kaggle CPU for 50K vocabulary words. Three source-level optimizations and a GPU acceleration path were added, along with a restructured Kaggle notebook that breaks the opaque single-cell build into 6 observable, debuggable stages.

**No behavioral changes.** All optimizations produce identical geometric results — the manifold is shaped the same way, just faster.

---

## Problem Statement

The `build_and_save()` call executes 5 internal steps sequentially in a single opaque function. Profiling identified three dominant bottlenecks:

| Step | Operation | Root Cause | Time Share |
|---|---|---|---|
| Step 2: `WordPlacer.place_batch()` | 50K sequential C3 placements | Temperature save/restore on **every word** (50K cycles); structural vector computed per-word in Python | ~40% |
| Step 3: `ContrastScheduler.run_passes()` | 100K+ C4 judgments × 3 passes | Per-pair `try/except` manifold lookup for O(V²) pairs; no progress visibility | ~45% |
| Step 5: `TemplateBuilder._build_level1()` | 50K hedging derivations | Per-word `manifold.nearest(k=5)` call — 50K individual kNN searches | ~10% |

The remaining ~5% (PMI matrix build, phrase radius calibration, save) was already fast.

---

## What Was Changed

### 1. `src/vocabulary/word_placer.py` — Batch Temperature + GPU Path

**Before:** `place_batch()` was a list comprehension calling `place()` per word, each of which saved T₀, set T_floor, placed, then restored T₀.  50K temperature save/restore cycles.

**After:**
- `place_batch()` wraps the entire batch in a single `_set_cold()` / `_restore_temperature()` pair
- Added `progress_callback` parameter for logging every 1000 words
- New `place_batch_gpu()` method that pre-computes all 50K structural vectors on GPU via cupy, then places sequentially (graceful fallback to CPU)
- New `batch_structural_vectors_gpu()` helper function

**Risk:** Zero — temperature value is identical throughout the batch (forced to T_floor). The per-word save/restore was pure overhead.

### 2. `src/vocabulary/contrast_scheduler.py` — Pre-Built Label Set

**Before:** `run()` called `_on_manifold(manifold, label)` for every pair, which did a `try/except` around `manifold.position(label)` — O(V²) exception-driven existence checks.

**After:**
- Pre-builds `manifold_labels = set(manifold.labels)` once at the start of each `run()` call
- All membership checks are `label in manifold_labels` — O(1) set lookup instead of try/except
- `_apply_causal_bias()` accepts the pre-built set to avoid redundant lookups
- Added `progress_callback` to both `run()` and `run_passes()` for per-batch visibility

**Risk:** Zero — the set contains exactly the same labels as would succeed in the try/except path.

### 3. `src/vocabulary/template_builder.py` — Batch Hedging via cKDTree

**Before:** `_build_level1()` called `_derive_hedging(manifold, label)` per word, each doing `manifold.nearest(pos, k=5)` — 50K individual kNN searches through the full manifold.

**After:**
- New `_batch_derive_hedging(manifold, labels)` builds a single `cKDTree` from all manifold points and queries all 50K positions in one batch call
- `_build_level1()` calls the batch function once instead of 50K individual calls
- Original `_derive_hedging()` preserved for backward compatibility (used by Level 2/3)

**Risk:** Zero — identical results, verified by tests. The cKDTree batch query returns the same neighbors as individual `manifold.nearest()` calls.

---

## Notebook Restructuring

### Before: 1 opaque cell

Cell 14 called `builder.build_and_save(VOCAB_PATH)` — a single function that could run for 2 hours with no output, no way to tell if it was hanging, and no error isolation.

### After: 6 staged cells with full visibility

| Cell | Stage | What It Shows |
|---|---|---|
| 5A | Build PMI matrix | Vocabulary size, PMI pair counts, top-10 strongest pairs, contrast workload preview |
| 5B | Place words on M(t) | GPU/CPU detection, progress every 1000 words (rate, temperature, last label), spot-check placements |
| 5C | Contrast refinement | Per-batch progress (rate, % complete), per-pass completion log, correct-direction rate, sample distance check |
| 5D | Calibrate phrase radius | Calibrated radius value, vocab count on manifold |
| 5E | Build ExpressionEntry objects | Per-level timing and counts, sample entries, register distribution, hedging count |
| 5F | Save to .npz | File size, entry count, full timing summary table |

Each cell has:
- `try/except` with `traceback.print_exc()` — errors are caught and shown, not silently swallowed
- Timing (`time.time()`) at start and end
- Progress callbacks that log periodically so you can see the system is alive
- Sanity checks (spot-check position lookups, distance calculations, sample entries)

---

## Estimated Speedup

| Optimization | Mechanism | Expected Impact |
|---|---|---|
| Batch temperature | Eliminate 50K save/restore cycles | ~10-15% of total |
| Pre-built label set | O(1) set lookup vs O(V²) try/except | ~20-30% of contrast step |
| Batch hedging cKDTree | 1 batch query vs 50K individual kNN | ~90% of Level 1 build |
| GPU structural vectors | cupy batch compute (when available) | ~5-10% of placement step |

Combined estimate: **30–50% total runtime reduction** on CPU; additional gains on GPU.

Additional parameter-level speedups available without code changes:
- Reduce `N_CONTRAST_PASSES` from 3 to 1 (first pass does ~80% of work)
- Raise `TAU_SAME` from 1.0 to 1.5 (fewer SAME pairs, strongest preserved)
- Lower `V_MAX` from 50K to 30K (long tail contributes marginal expressiveness)

---

## Files Modified

| File | Change | Lines |
|---|---|---|
| `src/vocabulary/word_placer.py` | `place_batch()` single temp wrap + callback; `place_batch_gpu()` new method; `batch_structural_vectors_gpu()` new function | +65 |
| `src/vocabulary/contrast_scheduler.py` | Pre-built label set; progress callbacks on `run()`, `run_passes()`, `_apply_causal_bias()` | +25, -15 |
| `src/vocabulary/template_builder.py` | `_batch_derive_hedging()` new function; `_build_level1()` uses batch hedging | +75 |
| `notebooks/kaggle_vocabulary_growth.ipynb` | 1 build cell → 6 staged cells (5A–5F) with logging, error handling, progress callbacks | restructured |

---

## Design Constraints Upheld

| Constraint | Status | Evidence |
|---|---|---|
| NO WEIGHTS | ✅ | No tunable parameters added; optimizations are data-structure level |
| NO TOKENS | ✅ | No tokeniser changes; words remain geometric events |
| NO TRAINING | ✅ | C3/C4 placement loop unchanged; just fewer Python overhead cycles |
| LOCAL UPDATES | ✅ | Gaussian deformation kernel unchanged; cKDTree is a spatial index |
| CAUSALITY FIRST | ✅ | Causal fiber nudges unchanged; label set optimization is read-only |
| SEPARATION | ✅ | No cross-component changes; optimizations are within Group A |

---

## Test Results

```
721 passed, 1 skipped in 11.84s
```

- 1 skip: optional FAISS GPU test (no FAISS installed locally)
- 0 failures, 0 regressions
- All Phase 1–8 tests remain green
- Phase 7 tests (word_placer, contrast_scheduler, template_builder) all pass with optimized code
