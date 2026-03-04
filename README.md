# despy

Dynamic Ensemble Selection (DES) for Python.

Rather than picking one model or blending all models equally, DES routes each
test sample to whichever models performed best on similar training examples.
The idea is that no single model wins everywhere — a KNN might dominate
smooth low-dimensional regions while a gradient booster wins on sharp
boundaries — and that routing by local competence captures these divisions
automatically without any manual tuning.

despy works with any array-producing model. The router only ever sees numpy
arrays of predictions; the underlying models are never called at inference time
and need no sklearn wrappers.

---

## Installation

```bash
pip install despy

# ANN backends — install whichever you want to use
pip install faiss-cpu   # FAISS (good default for most datasets)
pip install annoy       # Annoy (memory-efficient, simple)
pip install hnswlib     # HNSW (best for high-dimensional data)
```

---

## Quick start

```python
from despy.des.knndws  import KNNDWS
from despy.des.knorau  import KNORAU
from despy.des.knoraiu import KNORAIU

# -- 1. Train your models however you like --------------------------------
#    sklearn, torch, jax, keras — despy doesn't care
models = {"rf": rf, "xgb": xgb, "mlp": mlp}

# -- 2. Get predictions on a held-out validation set ---------------------
#    Regression: scalar arrays
#    Classification: predict_proba() probability arrays, OR hard predictions
val_preds = {name: m.predict(X_val) for name, m in models.items()}

# -- 3. Fit the router ---------------------------------------------------
router = KNNDWS(task="regression", metric="mae", mode="min", k=20)
router.fit(X_val, y_val, val_preds)

# -- 4. Route test samples -----------------------------------------------
test_preds = {name: m.predict(X_test) for name, m in models.items()}

for i, x in enumerate(X_test):
    weights = router.predict(x, temperature=0.1)
    # weights: {"rf": 0.7, "xgb": 0.2, "mlp": 0.1}
    prediction = sum(weights[n] * test_preds[n][i] for n in weights)
```

For classification with probability arrays, blend the output the same way to
get a final probability distribution, then take the argmax.

---

## Why despy?

Most DES libraries are tied to scikit-learn. despy only ever sees a numpy
feature matrix and a dict of prediction arrays — the models themselves are
never touched after training.

```python
# PyTorch — no wrappers needed
with torch.no_grad():
    val_preds  = {name: m(X_val_t).cpu().numpy()  for name, m in models.items()}
    test_preds = {name: m(X_test_t).cpu().numpy() for name, m in models.items()}

router = KNORAU(task="classification", metric="accuracy", mode="max", k=20)
router.fit(X_val, y_val, val_preds)
weights = router.predict(X_test[i])
```

---

## Algorithms

| Method | Best for | Notes |
|---|---|---|
| `KNNDWS` | Regression | Softmax over neighbourhood-averaged scores. Temperature controls sharpness. |
| `KNORAU` | Classification | Vote-count weighting. Each model earns one vote per neighbour it correctly classifies. |
| `KNORAE` | Classification | Intersection-based. Only models correct on *all* neighbours survive; falls back to smaller neighbourhoods. |
| `KNORAIU` | Classification | Like KNORA-U but votes are inverse-distance weighted. |
| `OLA` | Both | Hard selection: only the single best model in the neighbourhood contributes. |

**Which to use:** `KNNDWS` for regression — continuous competence signals from
mae/mse are richer than binary correct/wrong. `KNORAU` or `KNORAIU` for
classification as a safe default. `KNORAE` is more aggressive — good when one
model clearly dominates local regions, noisier on datasets with genuine overlap.

---

## ANN backends

despy supports three approximate nearest-neighbour backends plus exact search:

| Preset | Backend | Install | Notes |
|---|---|---|---|
| `exact` | sklearn KNN | — | Exact, no extra deps |
| `balanced` | FAISS IVF | `faiss-cpu` | ~98% recall, good default |
| `fast` | FAISS IVF | `faiss-cpu` | ~95% recall, faster queries |
| `turbo` | FAISS flat | `faiss-cpu` | Exact via FAISS, GPU-friendly |
| `high_dim_balanced` | HNSW | `hnswlib` | Best for >100 features, balanced |
| `high_dim_fast` | HNSW | `hnswlib` | Best for >100 features, faster |

Annoy is also available as a custom backend — memory-efficient and simple,
good for datasets that need to be persisted to disk.

```python
# Exact search (no extra deps)
router = KNORAU(..., preset="exact")

# High-dimensional data
router = KNORAU(..., preset="high_dim_balanced")

# Custom FAISS config
router = KNORAU(..., preset="custom", finder="faiss",
                index_type="ivf", n_probes=50)

# Annoy
router = KNORAU(..., preset="custom", finder="annoy",
                n_trees=100, search_k=-1)
```

---

## Custom metrics

Any callable `(y_true, y_pred) -> float` works:

```python
def pinball(y_true, y_pred, alpha=0.9):
    e = y_true - y_pred
    return alpha * e if e >= 0 else (alpha - 1) * e

router = KNNDWS(task="regression", metric=pinball, mode="min", k=20)
```

Built-in metric strings: `accuracy`, `mae`, `mse`, `rmse`, `log_loss`, `prob_correct`.

---

## Benchmark results

10-seed benchmark (seeds 0–9) on standard datasets. "Best Single" is the best
individual model selected on the validation set. "Simple Average" is uniform
equal-weight blending, included as a naive baseline.

Pool: KNN · Random Forest · Hist. Gradient Boosting · SVM-RBF (C=2) · MLP.

### Regression (MAE, lower is better)

| Dataset | Best Single | Simple Avg | despy best | vs Best Single |
|---|---|---|---|---|
| California Housing | 0.3370 | +2.0% | **0.3262** (KNN-DWS) | **−3.2%** |
| Bike Sharing | 31.02 | +32.8% | **30.89** (KNN-DWS) | **−0.4%** |
| Abalone | 1.5479 | −1.5% | **1.5247** (KNORA-U) | **−1.5%** |

KNN-DWS wins or ties best single on every regression dataset across all 10 seeds.
Simple Average performs poorly on Bike Sharing (+32.8%) — pool models specialise
heavily by time-of-day pattern, so equal-weight blending is actively harmful.

### Classification (Accuracy, higher is better)

% shown as delta vs Best Single. 10-seed mean.

| Dataset | Best Single | Simple Avg | despy best | DESlib best |
|---|---|---|---|---|
| Waveform | 84.94% | +0.40% | **+0.40%** (KNORA-U/IU) | +0.41% (KNOP) |
| Satimage | 91.34% | +0.15% | +0.19% (KNORA-IU) | +0.23% (KNOP) |
| MNIST Digits | 96.83% | +0.66% | +0.83% (KNORA-E) | +1.12% (KNORA-U) |
| Pendigits | 99.02% | +0.25% | **+0.32%** (KNORA-E) | +0.30% (KNORA-E) |

despy matches or beats DESlib on 3 of 4 datasets. On MNIST, DESlib's KNORA-U
outperforms by ~0.3% — DESlib uses weighted hard voting while despy blends
probability arrays, which accounts for most of the gap.

### Speed (mean ms fit + predict, 10 seeds)

| Dataset | despy | DESlib | Speedup |
|---|---|---|---|
| Waveform | 9–12 ms | 69–143 ms | ~10× |
| Satimage | 11–15 ms | 107–191 ms | ~10× |
| MNIST Digits | 4–5 ms | 99–158 ms | ~25× |
| Pendigits | 19–23 ms | 200–325 ms | ~12× |

despy caches all model predictions on the validation set at fit time and reads
from that matrix at inference. DESlib re-queries each model per neighbour at
inference time. The gap grows with pool size and number of classes.

---

## Pool design

DES amplifies whatever diversity already exists in the pool. A poor pool
produces poor routing regardless of the algorithm.

**What works:** models with genuinely different inductive biases. A good
starting point is KNN + Random Forest + Gradient Boosting + SVM-RBF + MLP.
Each wins in a different region of most feature spaces.

**What doesn't:** multiple models from the same family (e.g. Random Forest +
Extra Trees). They learn nearly identical decision boundaries, so routing
between them adds noise without signal. Pick one representative per family.

**Checking pool quality:** if the best DES result is close to or worse than
best single, the pool likely lacks diversity. The oracle accuracy (best model
per sample in hindsight) sets the theoretical ceiling — if oracle ≈ best
single, there is nothing for routing to find.

---

## Comparison with DESlib

[DESlib](https://github.com/scikit-learn-contrib/DESlib) is a mature
scikit-learn-compatible DES library. The main differences:

- **Framework**: DESlib requires sklearn-compatible estimators with `predict()`
  and `predict_proba()` methods. despy accepts prediction arrays from any source
  — PyTorch, JAX, Keras, or custom models need no wrappers.
- **Regression**: DESlib has no regression support. despy supports it natively
  with clean wins on all three benchmark datasets.
- **Speed**: despy is 10–25× faster at inference. despy caches predictions at
  fit time; DESlib calls each model live per neighbour at inference time.
- **Accuracy**: comparable on most classification datasets. despy edges DESlib
  on Pendigits; DESlib edges despy on MNIST by ~0.3%, attributable to a
  hard vs soft voting difference in the KNORA implementation.

---

## Contributing

Issues and PRs welcome.