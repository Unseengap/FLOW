# FLOW Saved Models

This directory contains saved FLOW model checkpoints that are committed to git.

## FLOW-V1-Base

The constitutional base model built by `notebooks/kaggle_base_model.ipynb`.

| File | Contents |
|------|----------|
| `FLOW-V1-Base_manifold.npz` | Full manifold state — axioms, schemas, vocabulary geometry |
| `FLOW-V1-Base_vocab.npz` | Expression vocabulary entries for C7 rendering |

### Loading

```python
from src.phase5 import GEOPipeline
pipeline = GEOPipeline.load(
    'models/FLOW-V1-Base_manifold.npz',
    vocabulary_path='models/FLOW-V1-Base_vocab.npz',
    flow_seed=42
)
```
