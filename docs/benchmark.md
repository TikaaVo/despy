# Benchmark

All results are from a 10-seed benchmark (seeds 0–9) on standard datasets. Each seed
produces a different random train/validation/test split, and results are averaged across
all 10 seeds to reduce variance from any single split.

---

## Methodology

**Split:** Each dataset is split into train, validation, and test sets. Models are trained
on the train set, the DES router is fitted on the validation set, and it is evaluated on the test set.

**Best Single** is the best individual model from the pool, selected by validation
set performance. It represents the baseline without any ensembling.

**Simple Average** is a uniform blend of all five models with no fitting or
tuning. It represents the simplest possible ensemble baseline.

All despy algorithms use `preset="balanced"` (FAISS IVF) and `k=20`.

---

## Pool

The same five-model pool is used across all datasets:

| Model |
|---|
| K-Nearest Neighbors |
| Random Forest |
| Hist. Gradient Boosting |
| SVM-RBF (C=2) |
| MLP |

These five were chosen for having different inductive biases and architectures, which is the kind of scenario that
DES would be used in.

---

## Datasets

### California Housing
**Source:** sklearn built-in. **Size:** 20,640 samples, 8 features.

Predict median house value from census block features.

### Bike Sharing
**Source:** [OpenML 42712](https://www.openml.org/d/42712). **Size:** 17,379 samples, 12 features.

Predict hourly bike rental counts from weather and time features.

### Abalone
**Source:** [OpenML 183](https://www.openml.org/d/183). **Size:** 4,177 samples, 8 features.

Predict abalone age from physical measurements.

### Waveform
**Source:** [OpenML 60](https://www.openml.org/d/60). **Size:** 5,000 samples, 40 features.

Three-class classification of artificially constructed waveforms with deliberate class
overlap.

### Satimage
**Source:** [OpenML 182](https://www.openml.org/d/182). **Size:** 6,435 samples, 36 features.

Six-class land-cover classification from satellite pixel data.

### MNIST Digits
**Source:** sklearn built-in. **Size:** 1,797 samples, 64 features (8×8 pixel images).

Ten-class handwritten digit classification.

### Pendigits
**Source:** [OpenML 32](https://www.openml.org/d/32). **Size:** 10,992 samples, 16 features.

Ten-class classification of handwritten digits from pen stroke trajectories.

---

## Regression results

MAE, lower is better. % shown as delta vs Best Single. 10-seed mean ± std.

| Dataset | Best Single | Simple Avg | KNN-DWS | OLA | KNORA-U | KNORA-E | KNORA-IU |
|---|---|---|---|---|---|---|---|
| California Housing | 0.3370 ± 0.005 | +2.0% | **−3.2%** | −2.4% | +0.5% | +5.1% | +0.4% |
| Bike Sharing | 31.02 ± 0.55 | +32.8% | **−0.4%** | +1.1% | +12.8% | +19.6% | +11.9% |
| Abalone | 1.5479 ± 0.026 | −1.5% | −0.2% | +0.4% | **−1.5%** | +1.7% | −1.5% |

---

KNORA variants are designed for classification, which explains the poor performance
on regression datasets; An exception occurs in Abalone, because ring amounts are whole numbers, usually 0-29.

## Classification results

Accuracy, higher is better. % shown as delta vs Best Single. 10-seed mean ± std.
Classification datasets include a comparison against [DESlib](https://github.com/scikit-learn-contrib/DESlib),
a mature sklearn-compatible DES library.

### Waveform

| Method | Mean | Std | vs Best Single |
|---|---|---|---|
| Best Single | 84.94% | 0.69% | — |
| Simple Average | 85.28% | 0.72% | +0.40% |
| despy KNN-DWS | 84.66% | 0.88% | −0.33% |
| despy OLA | 84.28% | 1.08% | −0.78% |
| despy KNORA-U | 85.25% | 0.76% | +0.36% |
| despy KNORA-E | 83.51% | 1.04% | −1.68% |
| despy KNORA-IU | 85.25% | 0.76% | +0.36% |
| DESlib KNORA-U | 85.12% | 0.89% | +0.21% |
| DESlib KNORA-E | 83.78% | 1.01% | −1.37% |
| DESlib OLA | 83.27% | 1.05% | −1.97% |
| DESlib META-DES | 84.66% | 0.95% | −0.33% |
| DESlib KNOP | **85.29%** | 0.82% | **+0.41%** |
| DESlib DESP | 84.96% | 0.88% | +0.02% |
| DESlib DESKNN | 83.75% | 0.70% | −1.40% |

DESlib achieves a best mean score of 85.29%; despy achieves a best mean score of 85.25%.

### Satimage

| Method | Mean | Std | vs Best Single |
|---|---|---|---|
| Best Single | 91.34% | 0.68% | — |
| Simple Average | 91.48% | 0.60% | +0.15% |
| despy KNN-DWS | 91.28% | 0.66% | −0.06% |
| despy OLA | 90.99% | 0.63% | −0.38% |
| despy KNORA-U | 91.47% | 0.57% | +0.14% |
| despy KNORA-E | 91.31% | 0.64% | −0.03% |
| despy KNORA-IU | 91.46% | 0.56% | +0.14% |
| DESlib KNORA-U | 91.46% | 0.63% | +0.14% |
| DESlib KNORA-E | 91.32% | 0.64% | −0.02% |
| DESlib OLA | 91.03% | 0.71% | −0.34% |
| DESlib META-DES | 91.45% | 0.40% | +0.13% |
| DESlib KNOP | **91.55%** | 0.43% | **+0.23%** |
| DESlib DESP | 91.46% | 0.45% | +0.14% |
| DESlib DESKNN | 91.23% | 0.50% | −0.12% |

DESlib achieves a best mean score of 91.55%; despy achieves a best mean score of 91.47%.

### MNIST Digits

| Method | Mean | Std | vs Best Single |
|---|---|---|---|
| Best Single | 96.83% | 1.00% | — |
| Simple Average | 97.47% | 0.47% | +0.66% |
| despy KNN-DWS | 97.47% | 0.83% | +0.66% |
| despy OLA | 96.97% | 0.81% | +0.14% |
| despy KNORA-U | 97.53% | 0.40% | +0.72% |
| despy KNORA-E | 97.64% | 0.43% | +0.83% |
| despy KNORA-IU | 97.53% | 0.40% | +0.72% |
| DESlib KNORA-U | **97.92%** | 0.26% | **+1.12%** |
| DESlib KNORA-E | 97.75% | 0.26% | +0.95% |
| DESlib OLA | 96.78% | 0.64% | −0.06% |
| DESlib META-DES | 97.67% | 0.43% | +0.86% |
| DESlib KNOP | 97.78% | 0.39% | +0.98% |
| DESlib DESP | 97.81% | 0.36% | +1.00% |
| DESlib DESKNN | 96.78% | 0.57% | −0.06% |

DESlib achieves a best mean score of 97.92%; despy achieves a best mean score of 91.64%.

### Pendigits

| Method | Mean | Std | vs Best Single |
|---|---|---|---|
| Best Single | 99.02% | 0.30% | — |
| Simple Average | 99.27% | 0.11% | +0.25% |
| despy KNN-DWS | 99.28% | 0.11% | +0.26% |
| despy OLA | 99.21% | 0.20% | +0.19% |
| despy KNORA-U | 99.27% | 0.12% | +0.25% |
| despy KNORA-E | **99.34%** | 0.10% | **+0.32%** |
| despy KNORA-IU | 99.27% | 0.12% | +0.25% |
| DESlib KNORA-U | 99.29% | 0.14% | +0.27% |
| DESlib KNORA-E | 99.32% | 0.13% | +0.30% |
| DESlib OLA | 99.11% | 0.24% | +0.09% |
| DESlib META-DES | 99.32% | 0.08% | +0.30% |
| DESlib KNOP | 99.30% | 0.13% | +0.28% |
| DESlib DESP | 99.30% | 0.11% | +0.28% |
| DESlib DESKNN | 99.14% | 0.15% | +0.12% |

DESlib achieves a best mean score of 99.32%; despy achieves a best mean score of 99.34.

---

## Timing

Mean fit + predict time in milliseconds, averaged across 10 seeds. Fit is measured once
per dataset per seed; predict is measured over the full test set.

despy caches all model predictions on the validation set at fit time and reads from that
matrix at inference, so no model is called at predict time. This is the primary reason for
the speed advantage over DESlib, which calls each model live per neighbour at inference.

### despy

| Dataset | KNN-DWS | OLA | KNORA-U | KNORA-E | KNORA-IU |
|---|---|---|---|---|---|
| California Housing | 22.6 ms | 21.7 ms | 27.6 ms | 33.0 ms | 26.9 ms |
| Bike Sharing | 18.1 ms | 17.4 ms | 20.2 ms | 27.0 ms | 21.6 ms |
| Abalone | 4.7 ms | 4.7 ms | 5.1 ms | 6.8 ms | 5.2 ms |
| Waveform | 9.7 ms | 9.2 ms | 9.4 ms | 12.1 ms | 9.9 ms |
| Satimage | 11.7 ms | 11.4 ms | 11.4 ms | 14.5 ms | 12.1 ms |
| MNIST Digits | 4.4 ms | 4.0 ms | 3.9 ms | 3.9 ms | 3.8 ms |
| Pendigits | 19.4 ms | 19.1 ms | 19.5 ms | 22.7 ms | 21.1 ms |

### DESlib (classification datasets only)

| Dataset | KNORA-U | KNORA-E | OLA | META-DES | KNOP | DESP | DESKNN |
|---|---|---|---|---|---|---|---|
| Waveform | 71.0 ms | 71.0 ms | 69.5 ms | 143.0 ms | 120.0 ms | 71.5 ms | 78.4 ms |
| Satimage | 107.6 ms | 106.5 ms | 106.2 ms | 191.0 ms | 162.1 ms | 111.2 ms | 111.7 ms |
| MNIST Digits | 102.2 ms | 98.9 ms | 101.8 ms | 158.3 ms | 155.3 ms | 99.2 ms | 103.0 ms |
| Pendigits | 201.4 ms | 212.6 ms | 201.8 ms | 324.7 ms | 306.7 ms | 200.1 ms | 201.3 ms |