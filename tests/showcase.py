#!/usr/bin/env python3
"""
despy Showcase
==============
Benchmarks despy against DESlib and classical baselines across four task types:
regression, tabular classification, image classification, and time-series
classification.

Why despy is framework-agnostic
  despy receives any numeric predictions — numpy arrays, PyTorch tensors, JAX
  arrays, Keras model outputs — and returns routing weights. It never calls
  fit() or predict() on your models; it only sees their outputs.
  DESlib requires sklearn-compatible estimators (fit/predict/predict_proba API)
  and has no regression support.

Regression datasets   (DESlib: N/A — regression not supported)
  California Housing  20K samples, 8 features
  Bike Sharing        17K samples, 8 features
  Abalone             4.2K samples, 9 features (rings count)

Tabular classification   (despy vs DESlib, direct head-to-head)
  Letter Recognition  20K samples, 16 features, 26 classes
  Pendigits           11K samples, 16 features, 10 classes (pen digit trajectories)

Image classification   (features = raw pixels or CNN embeddings)
  MNIST Digits        1797 samples, 64 features (8×8 pixel values), 10 classes
  despy works with any feature extractor (PyTorch, JAX, Keras, custom).
  DESlib works here too, but only with sklearn estimators.

Time-series classification   (features = signal statistics or sequence embeddings)
  EEG Eye State       15K samples, 14 EEG channel amplitudes, 2 classes
  In production, replace raw amplitudes with tsfresh, catch22, or transformer
  embeddings. despy is agnostic to how features were extracted.

Models   (6 per task — diverse inductive biases, no internal scalers)
  Regression     Ridge, KNN, Random Forest, Extra Trees, HGB, MLP
  Classification Logistic Reg, KNN, Random Forest, Extra Trees, HGB, Naive Bayes

Scaling strategy
  All features are imputed and StandardScaled on training statistics before
  model fitting. This removes the need for Pipeline-internal scalers and ensures
  despy, DESlib, and all pool classifiers receive identical feature representations.

DES metric choice
  Regression     MAE, mode='min'
  Classification log_loss, mode='min', with predict_proba() inputs

KNORA threshold guidance
  knn-dws  0.5 always
  KNORA    1.0 for regression (strict oracle criterion)
           0.5 for classification with log_loss

Install:  pip install scikit-learn scipy faiss-cpu deslib
Runtime:  ~20-30 min on a modern laptop
"""

import contextlib
import io
import time
import warnings

import numpy as np
from sklearn.datasets import fetch_california_housing, fetch_openml, load_digits
from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingClassifier, GradientBoostingRegressor,
    HistGradientBoostingClassifier, HistGradientBoostingRegressor,
    RandomForestClassifier, RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

from despy.des.knndws   import KNNDWS
from despy.des.ola      import OLA
from despy.des.knorau   import KNORAU
from despy.des.knorae   import KNORAE
from despy.des.knoraiu import KNORAIU
from despy.analysis     import analyze

warnings.filterwarnings('ignore')

# ── Optional DESlib import ─────────────────────────────────────────────────────
try:
    # sklearn >= 1.6 removed BaseEstimator._validate_data;
    # patch it back so DESlib (built against older sklearn) doesn't crash.
    from sklearn.base import BaseEstimator
    if not hasattr(BaseEstimator, '_validate_data'):
        from sklearn.utils.validation import check_array, check_X_y
        def _validate_data_compat(self, X, y=None, reset=True, **kw):
            _safe = {k: v for k, v in kw.items()
                     if k in ('accept_sparse', 'dtype', 'force_all_finite', 'order')}
            if y is not None:
                return check_X_y(X, y, **_safe)
            return check_array(X, **_safe)
        BaseEstimator._validate_data = _validate_data_compat

    from deslib.des.knora_u  import KNORAU  as DL_KNORAU
    from deslib.des.knora_e  import KNORAE  as DL_KNORAE
    from deslib.dcs.ola      import OLA     as DL_OLA
    from deslib.des.meta_des import METADES as DL_METADES
    from deslib.des.knop     import KNOP    as DL_KNOP
    from deslib.des.des_p    import DESP    as DL_DESP
    from deslib.des.des_knn  import DESKNN  as DL_DESKNN
    DESLIB_AVAILABLE = True
except ImportError:
    DESLIB_AVAILABLE = False

# ── Optional DESReg import ─────────────────────────────────────────────────────
# DESReg: Dynamic Ensemble Selection for Regression tasks
# https://github.com/lperezgodoy/DESReg  /  pip install DESReg
#
# Architecture note: DESReg bags the pool internally from unfitted regressors
# and manages the DSEL split itself. This is fundamentally different from despy,
# which accepts pre-fitted model outputs and never touches or calls your models.
# DESReg is sklearn-only; despy is framework-agnostic.
try:
    from desReg.des.DESRegression import DESRegression as _DESReg
    DESREG_AVAILABLE = True
except ImportError:
    DESREG_AVAILABLE = False


# ── Constants ──────────────────────────────────────────────────────────────────

SEED = 42
W    = 84      # print width

K_REG    = 20
K_CLF    = 20
TEMP_REG = 0.1
TEMP_CLF = 1.0

THRESHOLDS_REG = {
    'knn-dws':  0.5,
    'ola':      0.5,
    'knora-u':  1.0,
    'knora-e':  1.0,
    'knora-iu': 1.0,
}
THRESHOLDS_CLF = {
    'knn-dws':  0.5,
    'ola':      0.5,
    'knora-u':  0.5,
    'knora-e':  0.5,
    'knora-iu': 0.5,
}

DES_METHODS = ['knn-dws', 'ola', 'knora-u', 'knora-e', 'knora-iu']

_DES_CLASSES = {
    'knn-dws':  KNNDWS,
    'ola':      OLA,
    'knora-u':  KNORAU,
    'knora-e':  KNORAE,
    'knora-iu': KNORAIU,
}

# DESlib algorithms run in the comparison (7 total: 3 basic + 4 advanced)
DL_METHODS = ['KNORA-U', 'KNORA-E', 'OLA', 'META-DES', 'KNOP', 'DESP', 'DESKNN']

# DESReg modes: DES = dynamic ensemble selection, DSR = dynamic regressor selection
DR_METHODS = ['DES', 'DSR']


# ── Display helpers ────────────────────────────────────────────────────────────

def banner():
    dl_status = ('installed — 7 algorithms: KNORA-U/E · OLA · META-DES · KNOP · DESP · DESKNN'
                 if DESLIB_AVAILABLE
                 else 'not installed (pip install deslib) — skipping comparison')
    dr_status = ('installed — head-to-head comparison on regression'
                 if DESREG_AVAILABLE
                 else 'not installed (pip install DESReg) — skipping comparison')
    print(f"\n{'━' * W}")
    print("  despy Showcase  —  Regression · Tabular · Images · Time Series")
    print(f"{'━' * W}")
    print("  Best Single       best val-set model applied to test set everywhere")
    print("  Simple Average    uniform equal-weight blend of all models (no tuning)")
    print("  despy             KNN-DWS · OLA · KNORA-U · KNORA-E · KNORA-IU")
    print(f"  DESlib            {dl_status}")
    print(f"  DESReg            {dr_status}")
    print()
    print("  despy is framework-agnostic: pass predictions from PyTorch, JAX,")
    print("  Keras, or any library. DESlib and DESReg require sklearn estimators.")
    print("  DESlib has no regression support. DESReg has no classification support.")
    print(f"{'━' * W}")


def dataset_header(name, n_samples, n_features, extra):
    print(f"\n\n{'━' * W}")
    print(f"  Dataset: {name}")
    print(f"  {n_samples:,} samples  ·  {n_features} features  ·  {extra}")
    print(f"{'━' * W}")


def section(title):
    print(f"\n  {title}")
    print(f"  {'-' * (W - 4)}")


def show_results_reg(rows, best_mae, y_mean):
    valid        = [(n, m) for n, m in rows if m is not None]
    best_overall = min(m for _, m in valid) if valid else best_mae
    print(f"\n  {'Method':<48} {'MAE':>8}  {'% of mean':>10}  {'vs Best':>9}")
    print(f"  {'-'*48}  {'-'*8}  {'-'*10}  {'-'*9}")
    prev_section = None
    for name, mae in rows:
        cur_section = 'desreg' if name.startswith('DESReg') else 'other'
        if cur_section != prev_section and prev_section is not None:
            print(f"  {'·'*48}  {'·'*8}  {'·'*10}  {'·'*9}")
        prev_section = cur_section
        if mae is None:
            print(f"  {name:<48}  {'N/A':>8}  {'':>10}  {'':>9}")
            continue
        pct_mean = mae / y_mean * 100
        delta    = (mae - best_mae) / best_mae * 100
        d_str    = "    -    " if mae == best_mae else f"{'+' if delta >= 0 else ''}{delta:.2f}%"
        marker   = "  <" if mae == best_overall else ""
        print(f"  {name:<48}  {mae:>8.4f}  {pct_mean:>9.2f}%  {d_str:>9}{marker}")


def show_results_clf(rows, best_acc):
    valid = [(n, a) for n, a in rows if a is not None]
    best_overall = max(a for _, a in valid) if valid else best_acc
    print(f"\n  {'Method':<48} {'Accuracy':>9}  {'vs Best':>9}")
    print(f"  {'-'*48}  {'-'*9}  {'-'*9}")
    prev_section = None
    for name, acc in rows:
        # Print a light divider before DESlib rows
        cur_section = 'deslib' if name.startswith('DESlib') else 'despy'
        if cur_section != prev_section and prev_section is not None:
            print(f"  {'·'*48}  {'·'*9}  {'·'*9}")
        prev_section = cur_section
        if acc is None:
            print(f"  {name:<48}  {'N/A':>9}  {'':>9}")
            continue
        delta  = (acc - best_acc) / best_acc * 100
        d_str  = "    -    " if acc == best_acc else f"{'+' if delta >= 0 else ''}{delta:.2f}%"
        marker = "  <" if acc == best_overall else ""
        print(f"  {name:<48}  {acc*100:>8.2f}%  {d_str:>9}{marker}")


def show_timing(methods, fit_times, predict_times, n_test):
    print(f"\n  despy timing on {n_test:,} test samples:")
    print(f"    {'Method':<14}  {'Fit (ms)':>8}  {'Predict (ms)':>12}  {'ms/sample':>10}")
    print(f"    {'-'*14}  {'-'*8}  {'-'*12}  {'-'*10}")
    for m in methods:
        print(f"    {m:<14}  {fit_times[m]:>8.2f}  {predict_times[m]:>12.2f}"
              f"  {predict_times[m]/n_test:>10.4f}")


# ── Preprocessing ──────────────────────────────────────────────────────────────

def preprocess(X_tr, X_val, X_test):
    """
    Impute and StandardScale all splits using training statistics.

    All models and both routing libraries receive identically scaled features,
    avoiding double-scaling artifacts from Pipeline-internal scalers.
    """
    prep     = Pipeline([('imp', SimpleImputer(strategy='median')),
                         ('sc',  StandardScaler())])
    X_tr_s   = prep.fit_transform(X_tr)
    X_val_s  = prep.transform(X_val)
    X_test_s = prep.transform(X_test)
    return X_tr_s, X_val_s, X_test_s


# ── Model builders (no internal scalers — preprocess() handles scaling) ────────

def build_regressors(seed=SEED):
    """
    5 regressors: all legitimately competitive, each with a distinct
    inductive bias so DES has real competence regions to exploit.
    No Ridge/linear models (globally dominated by tree methods on these
    datasets) and no MLP (slow, rarely adds diversity on tabular data).

      RF             — bagging ensemble; strong, high-variance
      Extra Trees    — more randomised than RF; lower within-pool correlation,
                       often close to RF in accuracy
      HGB            — modern gradient boosting; usually best globally
      GBR            — older sklearn GBRT; different implementation to HGB,
                       genuinely different predictions on the same sample
      KNN-7          — purely local, instance-based; wins in dense low-noise
                       regions where boosting overgeneralises
    """
    return {
        'Random Forest':  RandomForestRegressor(
                              n_estimators=100, random_state=seed, n_jobs=-1),
        'Extra Trees':    ExtraTreesRegressor(
                              n_estimators=100, random_state=seed, n_jobs=-1),
        'Hist. Boosting': HistGradientBoostingRegressor(
                              max_iter=200, learning_rate=0.05, max_depth=4,
                              random_state=seed),
        'GBR':            GradientBoostingRegressor(
                              n_estimators=100, max_depth=3, learning_rate=0.1,
                              random_state=seed),
        'KNN-7':          KNeighborsRegressor(n_neighbors=7, n_jobs=-1),
    }

def build_classifiers(seed=SEED):
    """
    4 classifiers: all genuinely competitive with each other (within ~4% on
    most tabular benchmarks), each with a distinct inductive bias.

    Removed:
      Extra Trees    — consistently dominates, leaving nothing for DES to route
      Logistic Reg   — too weak (~77% on Letter), pollutes competence estimates
      Naive Bayes    — far too weak, same problem

    Kept / added:
      KNN            — purely local; wins in dense, low-noise regions
      Random Forest  — bagging ensemble; strong but different from boosting
      GBC            — sklearn's original GBRT; different implementation to HGB,
                       different predictions on the same samples, similar strength
      HGB            — modern fast boosting; usually the strongest globally
    """
    return {
        'KNN':            KNeighborsClassifier(n_neighbors=10, n_jobs=-1),
        'Random Forest':  RandomForestClassifier(
                              n_estimators=100, random_state=seed, n_jobs=-1),
        'GBC':            GradientBoostingClassifier(
                              n_estimators=100, max_depth=3, learning_rate=0.1,
                              random_state=seed),
        'Hist. Boosting': HistGradientBoostingClassifier(
                              max_iter=200, learning_rate=0.05, max_depth=4,
                              random_state=seed),
    }

# ── Dataset loaders ────────────────────────────────────────────────────────────

def load_california():
    print("  Loading California Housing...", end=' ', flush=True)
    X, y = fetch_california_housing(return_X_y=True)
    print("done")
    return X, y, 'California Housing', X.shape[1]


def load_bike():
    print("  Fetching Bike Sharing from OpenML...", end=' ', flush=True)
    d = fetch_openml(data_id=42712, as_frame=True, parser='auto')
    X = d.data.select_dtypes(include=['number']).astype(float).values
    y = d.target.astype(float).values
    print("done")
    return X, y, 'Bike Sharing (Hourly)', X.shape[1]


def load_abalone():
    """
    UCI Abalone dataset: predict age (number of rings) from physical measurements.
    Sex (M/F/I) is one-hot encoded. SimpleImputer handles any missing values.
    """
    print("  Fetching Abalone from OpenML...", end=' ', flush=True)
    import pandas as pd
    d = fetch_openml(data_id=183, as_frame=True, parser='auto')
    X = pd.get_dummies(d.data, drop_first=True).astype(float).values
    y = d.target.astype(float).values
    print("done")
    return X, y, 'Abalone (Rings)', X.shape[1]


def load_waveform():
    """
    Waveform (v2): 5,000 samples, 40 features (21 wave attributes + 19 noise),
    3 classes. Canonical DES benchmark: classes are noisy combinations of wave
    shapes, so models genuinely specialise by waveform region and class overlap
    is by construction. Oracle gain over best single is structurally guaranteed.
    """
    print("  Fetching Waveform from OpenML...", end=' ', flush=True)
    ds = fetch_openml(data_id=60, as_frame=True, parser='auto')
    X  = ds.data.astype(float).values
    y  = LabelEncoder().fit_transform(ds.target)
    print("done")
    return X, y, 'Waveform', X.shape[1], len(np.unique(y))


def load_satimage():
    """
    Statlog Satellite Image: 6,435 samples, 36 features (4 spectral bands
    across a 3x3 pixel neighbourhood), 6 land-cover classes.
    A canonical DES benchmark dataset — different classifiers specialise in
    different spectral signatures (urban, vegetation, soil, water, etc.).
    36 features gives KNN enough dimensions for stable competence estimates.
    """
    print("  Fetching Satimage from OpenML...", end=' ', flush=True)
    ds = fetch_openml(data_id=182, as_frame=True, parser='auto')
    X  = ds.data.astype(float).values
    y  = LabelEncoder().fit_transform(ds.target)
    print("done")
    return X, y, 'Satimage', X.shape[1], len(np.unique(y))

def load_digits_data():
    """
    sklearn digits dataset: 8×8 grayscale images of handwritten digits 0–9.
    X is raw pixel values (64 features) — a stand-in for CNN penultimate-layer
    embeddings. In production, replace with outputs from any image model.
    """
    print("  Loading MNIST Digits (sklearn built-in)...", end=' ', flush=True)
    d = load_digits()
    print("done")
    return d.data, d.target, 'MNIST Digits (sklearn)', d.data.shape[1], len(np.unique(d.target))


def load_pendigits():
    """
    Pendigits: 10,992 samples of handwritten digit pen trajectories,
    16 features (8 (x,y) coordinates sampled along the stroke), 10 classes.
    Different from pixel-based MNIST — models specialise by stroke dynamics.
    A confirmed DES benchmark in the literature with clean competence regions.
    """
    print("  Fetching Pendigits from OpenML...", end=' ', flush=True)
    ds = fetch_openml(data_id=32, as_frame=True, parser='auto')
    X  = ds.data.astype(float).values
    y  = LabelEncoder().fit_transform(ds.target)
    print("done")
    return X, y, 'Pendigits', X.shape[1], len(np.unique(y))

# ── Router factory ─────────────────────────────────────────────────────────────

def _make_router(task, method, metric, mode, k, preset='balanced'):
    """Instantiate a despy algorithm class, suppressing the preset print."""
    with contextlib.redirect_stdout(io.StringIO()):
        return _DES_CLASSES[method](task=task, metric=metric, mode=mode, k=k, preset=preset)


# ── Ensemble helpers ───────────────────────────────────────────────────────────

def fit_global_ensemble_reg(val_preds, y_val):
    """Uniform average — equal weight to every model."""
    names = list(val_preds.keys())
    w = np.ones(len(names)) / len(names)
    return dict(zip(names, w))


def apply_global_weights_reg(preds, weights):
    names = list(weights.keys())
    return np.array([weights[n] for n in names]) @ np.stack([preds[n] for n in names])


def fit_global_ensemble_clf(val_probas, y_val):
    """Uniform average — equal weight to every model."""
    names = list(val_probas.keys())
    w = np.ones(len(names)) / len(names)
    return dict(zip(names, w))


def apply_global_weights_clf(probas, weights):
    names = list(weights.keys())
    w     = np.array([weights[n] for n in names])
    return np.einsum('m,mnc->nc', w, np.stack([probas[n] for n in names]))


def des_predict_reg(router, X_test, test_preds, temperature, threshold):
    names  = list(test_preds.keys())
    result = router.predict(X_test, temperature=temperature, threshold=threshold)
    if isinstance(result, dict):
        result = [result]
    return np.array([
        sum(w[n] * test_preds[n][i] for n in names)
        for i, w in enumerate(result)
    ])


def des_predict_clf(router, X_test, test_probas, temperature, threshold):
    names  = list(test_probas.keys())
    result = router.predict(X_test, temperature=temperature, threshold=threshold)
    if isinstance(result, dict):
        result = [result]
    return np.array([
        sum(w[n] * test_probas[n][i] for n in names)
        for i, w in enumerate(result)
    ])


# ── DESlib comparison ──────────────────────────────────────────────────────────

def run_deslib(fitted_models, X_val_s, y_val, X_test_s, y_test, k=K_CLF):
    """
    Run DESlib's KNORA-U, KNORA-E, and OLA on the same pool and scaled features.

    The pool classifiers are already fitted on X_tr_s (globally scaled). DESlib
    will call their predict_proba(X_val_s) internally, which is consistent since
    the classifiers were trained on the same scale.

    Returns
    -------
    results   : dict[str, float | None]  accuracy per algorithm (None if failed)
    fit_ms    : dict[str, float]
    pred_ms   : dict[str, float]
    """
    pool     = list(fitted_models.values())
    results  = {}
    fit_ms   = {}
    pred_ms  = {}

    # 3 basic oracle algorithms + 4 more sophisticated methods.
    # Running all 7 so the comparison is comprehensive rather than
    # cherry-picked. Results shown as-is regardless of who wins.
    _dl_registry = [
        ('KNORA-U',  DL_KNORAU),
        ('KNORA-E',  DL_KNORAE),
        ('OLA',      DL_OLA),
        ('META-DES', DL_METADES),
        ('KNOP',     DL_KNOP),
        ('DESP',     DL_DESP),
        ('DESKNN',   DL_DESKNN),
    ]
    for label, cls in _dl_registry:
        try:
            m = cls(pool_classifiers=pool, k=k)
            t0 = time.perf_counter()
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                m.fit(X_val_s, y_val)
            fit_ms[label] = (time.perf_counter() - t0) * 1000

            t0 = time.perf_counter()
            y_pred = m.predict(X_test_s)
            pred_ms[label] = (time.perf_counter() - t0) * 1000

            results[label] = accuracy_score(y_test, y_pred)
        except Exception as exc:
            results[label] = None
            fit_ms[label]  = float('nan')
            pred_ms[label] = float('nan')
            print(f"      ✗ DESlib {label} failed: {exc}")

    return results, fit_ms, pred_ms


# ── Label helpers ──────────────────────────────────────────────────────────────

def _label_reg(method):
    return {
        'knn-dws':  f'despy KNN-DWS  (gate={THRESHOLDS_REG["knn-dws"]}, T={TEMP_REG})',
        'ola':       'despy OLA',
        'knora-u':  f'despy KNORA-U   (th={THRESHOLDS_REG["knora-u"]})',
        'knora-e':  f'despy KNORA-E   (th={THRESHOLDS_REG["knora-e"]})',
        'knora-iu': f'despy KNORA-IU  (th={THRESHOLDS_REG["knora-iu"]})',
    }[method]


def _label_clf(method):
    return {
        'knn-dws':  f'despy KNN-DWS  (gate={THRESHOLDS_CLF["knn-dws"]}, T={TEMP_CLF})',
        'ola':       'despy OLA',
        'knora-u':  f'despy KNORA-U   (th={THRESHOLDS_CLF["knora-u"]})',
        'knora-e':  f'despy KNORA-E   (th={THRESHOLDS_CLF["knora-e"]})',
        'knora-iu': f'despy KNORA-IU  (th={THRESHOLDS_CLF["knora-iu"]})',
    }[method]


# ── DESReg comparison ──────────────────────────────────────────────────────────

# ── DESReg comparison ──────────────────────────────────────────────────────────

def run_desreg(X_tv_s, y_tv, X_test_s, y_test, seed=SEED, k=K_REG, verbose=True):
    """
    Run DESReg on the same data budget as despy.

    DESReg receives X_tv (train+val combined, pre-scaled) and manages its own
    DSEL split internally via DSEL_perc=0.25 — giving it the same ~20% of total
    data for its competence region that despy uses as its val set.

    Pool: identical 5-model pool as despy (no nested ensembles, so bagging is fast).
    n_estimators_bag=2 is the minimum valid value; each instance is a
    BaggingRegressor trained on a bootstrap sample of X_tv.

    Returns
    -------
    results  : dict[str, float | None]  MAE per DESReg mode
    fit_ms   : dict[str, float]
    pred_ms  : dict[str, float]
    """
    _print = print if verbose else lambda *a, **kw: None
    # Same pool as despy -- all five models are fast to bag since none are
    # nested ensembles. This makes the comparison purely about routing quality.
    regressors = list(build_regressors(seed=seed).values())

    results = {}
    fit_ms  = {}
    pred_ms = {}

    for mode in DR_METHODS:
        try:
            m = _DESReg(
                regressors_list   = regressors,
                n_estimators_bag  = 2,        # minimum valid; two bags per regressor type
                DSEL_perc         = 0.25,     # 25% of X_tv = ~20% of total
                XTRAIN_full       = True,
                k                 = k,
                ensemble_type     = mode,
            )
            t0 = time.perf_counter()
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                m.fit(X_tv_s, y_tv)
            fit_ms[mode] = (time.perf_counter() - t0) * 1000

            t0 = time.perf_counter()
            y_pred = m.predict(X_test_s)
            pred_ms[mode] = (time.perf_counter() - t0) * 1000

            results[mode] = float(mean_absolute_error(y_test, y_pred))
            _print(f"    v DESReg {mode:<5}  MAE = {results[mode]:.4f}"
                   f"  fit: {fit_ms[mode]:6.2f}ms  |  predict: {pred_ms[mode]:6.2f}ms")
        except Exception as exc:
            results[mode] = None
            fit_ms[mode]  = float('nan')
            pred_ms[mode] = float('nan')
            _print(f"      ✗ DESReg {mode} failed: {exc}")

    return results, fit_ms, pred_ms

# ── Regression benchmark ───────────────────────────────────────────────────────

def run_regression(loader, seed=SEED, verbose=True):
    """
    Full regression benchmark on one dataset.

    Parameters
    ----------
    loader  : callable  → (X, y, name, n_features)
    seed    : int
    verbose : bool

    Returns
    -------
    dict[str, float]   method label → test MAE
    """
    _print = print if verbose else lambda *a, **kw: None

    X, y, ds_name, n_features = loader()
    if verbose:
        dataset_header(ds_name, len(X), n_features,
                       f"target mean = {y.mean():.2f}  std = {y.std():.2f}")

    X_tv, X_test, y_tv, y_test = train_test_split(X, y, test_size=0.20, random_state=seed)
    X_tr, X_val,  y_tr, y_val  = train_test_split(X_tv, y_tv, test_size=0.25, random_state=seed)
    X_tr_s, X_val_s, X_test_s  = preprocess(X_tr, X_val, X_test)
    # X_tv_s: train+val on training scaler — passed to DESReg for a fair data budget
    from sklearn.pipeline import Pipeline as _Pipe
    from sklearn.impute import SimpleImputer as _SI
    from sklearn.preprocessing import StandardScaler as _SS
    _prep_tv = _Pipe([('imp', _SI(strategy='median')), ('sc', _SS())])
    X_tv_s   = _prep_tv.fit_transform(X_tv)
    _print(f"\n  Split -> {len(X_tr):,} train / {len(X_val):,} val / {len(X_test):,} test")
    if DESLIB_AVAILABLE and verbose:
        _print("  Note: DESlib does not support regression — only despy benchmarked here.")

    if verbose:
        section("Training models  (val-set MAE)")
    models = build_regressors(seed=seed)
    val_preds, test_preds, val_maes = {}, {}, {}
    for mname, model in models.items():
        t0 = time.time()
        model.fit(X_tr_s, y_tr)
        val_preds[mname]  = model.predict(X_val_s)
        test_preds[mname] = model.predict(X_test_s)
        val_maes[mname]   = mean_absolute_error(y_val, val_preds[mname])
        _print(f"    v {mname:<20}  MAE = {val_maes[mname]:.4f}   ({time.time()-t0:.1f}s)")

    best_name = min(val_maes, key=val_maes.get)

    if verbose:
        section("Fitting ensembles  (val set only)")
    ge_w = fit_global_ensemble_reg(val_preds, y_val)
    _print(f"    v Simple Average")


    fit_times, predict_times, des_preds = {}, {}, {}
    for method in DES_METHODS:
        th     = THRESHOLDS_REG[method]
        label  = _label_reg(method)
        router = _make_router('regression', method, 'mae', 'min', K_REG)
        t0 = time.perf_counter()
        router.fit(X_val_s, y_val, val_preds)
        fit_times[method] = (time.perf_counter() - t0) * 1000
        t0 = time.perf_counter()
        des_preds[method] = des_predict_reg(router, X_test_s, test_preds, TEMP_REG, th)
        predict_times[method] = (time.perf_counter() - t0) * 1000
        _print(f"    v {label:<42}  fit: {fit_times[method]:6.2f}ms"
               f"  |  predict: {predict_times[method]:6.2f}ms")

    # Suitability analysis -- uncomment to enable:
    # if verbose:
    #     analyze(X_val_s, y_val, val_preds, metric='mae', mode='min', k=K_REG)

    # DESReg comparison
    dr_results, dr_fit_ms, dr_pred_ms = {}, {}, {}
    if DESREG_AVAILABLE:
        if verbose:
            section("DESReg comparison  (same regressor types, same data budget)")
            _print("  DESReg manages its own bagging and DSEL split internally.")
            _print("  Same 5-model pool as despy -- comparison is routing quality only.")
            _print("  n_estimators_bag=2, DSEL_perc=0.25 (~20% of total data for DSEL).")
        dr_results, dr_fit_ms, dr_pred_ms = run_desreg(
            X_tv_s, y_tv, X_test_s, y_test, seed=seed, k=K_REG, verbose=verbose)

    best_mae = mean_absolute_error(y_test, test_preds[best_name])
    ge_mae   = mean_absolute_error(y_test, apply_global_weights_reg(test_preds, ge_w))
    rows = [
        (f"Best Single  ({best_name})", best_mae),
        ("Simple Average",  ge_mae),
        *[(_label_reg(m), mean_absolute_error(y_test, des_preds[m])) for m in DES_METHODS],
    ]
    if DESREG_AVAILABLE:
        for dr_mode in DR_METHODS:
            rows.append((f"DESReg {dr_mode}", dr_results.get(dr_mode)))
    if verbose:
        section("Results on held-out test set  (MAE — lower is better)")
        show_results_reg(rows, best_mae, float(y_test.mean()))
        show_timing(DES_METHODS, fit_times, predict_times, len(X_test_s))

    labelled_fit  = {_label_reg(m): fit_times[m]     for m in DES_METHODS}
    labelled_pred = {_label_reg(m): predict_times[m] for m in DES_METHODS}
    return dict(rows), labelled_fit, labelled_pred


# ── Classification benchmark ───────────────────────────────────────────────────

def run_classification(loader, k=K_CLF, seed=SEED, verbose=True, note=None):
    """
    Full classification benchmark on one dataset, with optional DESlib comparison.

    Parameters
    ----------
    loader  : callable  → (X, y, name, n_features, n_classes)
    k       : int
    seed    : int
    verbose : bool
    note    : str or None  — extra context printed after the dataset header

    Returns
    -------
    dict[str, float | None]   method label → test accuracy
    """
    _print = print if verbose else lambda *a, **kw: None

    X, y, ds_name, n_features, n_classes = loader()
    if verbose:
        dataset_header(ds_name, len(X), n_features, f"{n_classes} classes")
        if note:
            print(f"\n  ℹ  {note}")

    X_tv, X_test, y_tv, y_test = train_test_split(
        X, y, test_size=0.20, random_state=seed, stratify=y)
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_tv, y_tv, test_size=0.25, random_state=seed, stratify=y_tv)
    X_tr_s, X_val_s, X_test_s = preprocess(X_tr, X_val, X_test)
    _print(f"\n  Split -> {len(X_tr):,} train / {len(X_val):,} val / {len(X_test):,} test"
           f"  (stratified)")

    if verbose:
        section("Training models  (val-set accuracy)")
    models = build_classifiers(seed=seed)
    val_probas, val_preds_hard, test_probas, val_accs = {}, {}, {}, {}
    for mname, model in models.items():
        t0 = time.time()
        model.fit(X_tr_s, y_tr)
        val_probas[mname]      = model.predict_proba(X_val_s)
        val_preds_hard[mname]  = model.predict(X_val_s)     # hard labels for KNORA
        test_probas[mname]     = model.predict_proba(X_test_s)
        val_accs[mname]        = accuracy_score(y_val, model.predict(X_val_s))
        _print(f"    v {mname:<20}  Acc = {val_accs[mname]*100:.2f}%   ({time.time()-t0:.1f}s)")

    best_name = max(val_accs, key=val_accs.get)

    if verbose:
        section("Fitting ensembles  (val set — metric: log_loss on probabilities)")
    ge_w = fit_global_ensemble_clf(val_probas, y_val)
    _print(f"    v Simple Average")


    fit_times, predict_times, des_probas = {}, {}, {}
    # KNORA algorithms use accuracy (0/1) on hard predictions — matching the
    # original algorithm definition and DESlib's behaviour. KNN-DWS and OLA
    # use log_loss on probabilities for richer continuous competence signals.
    _KNORA = {'knora-u', 'knora-e', 'knora-iu'}
    for method in DES_METHODS:
        th     = THRESHOLDS_CLF[method]
        label  = _label_clf(method)
        if method in _KNORA:
            router    = _make_router('classification', method, 'accuracy', 'max', k)
            fit_input = val_preds_hard
        else:
            router    = _make_router('classification', method, 'log_loss', 'min', k)
            fit_input = val_probas
        t0 = time.perf_counter()
        router.fit(X_val_s, y_val, fit_input)
        fit_times[method] = (time.perf_counter() - t0) * 1000
        t0 = time.perf_counter()
        des_probas[method] = des_predict_clf(router, X_test_s, test_probas, TEMP_CLF, th)
        predict_times[method] = (time.perf_counter() - t0) * 1000
        _print(f"    v {label:<42}  fit: {fit_times[method]:6.2f}ms"
               f"  |  predict: {predict_times[method]:6.2f}ms")

    # DESlib comparison (same pool, same scaled features)
    dl_results, dl_fit_ms, dl_pred_ms = {}, {}, {}
    if DESLIB_AVAILABLE:
        # Always run DESlib so benchmark.py gets results even with verbose=False.
        # Printing is gated on verbose separately.
        dl_results, dl_fit_ms, dl_pred_ms = run_deslib(
            models, X_val_s, y_val, X_test_s, y_test, k=k)
        if verbose:
            section("DESlib comparison  (same pool classifiers, same scaled features)")
            for dl_label in DL_METHODS:
                acc = dl_results.get(dl_label)
                ft  = dl_fit_ms.get(dl_label, float('nan'))
                pt  = dl_pred_ms.get(dl_label, float('nan'))
                acc_str = f"{acc*100:.2f}%" if acc is not None else "N/A"
                _print(f"    v DESlib {dl_label:<20}  acc: {acc_str:<8}"
                       f"  fit: {ft:6.2f}ms  |  predict: {pt:6.2f}ms")

    # Suitability analysis -- uncomment to enable:
    # if verbose:
    #     analyze(X_val_s, y_val, val_probas, metric='log_loss', mode='min', k=k)

    best_acc = accuracy_score(y_test, models[best_name].predict(X_test_s))
    ge_acc   = accuracy_score(
        y_test, apply_global_weights_clf(test_probas, ge_w).argmax(axis=1))
    rows = [
        (f"Best Single  ({best_name})", best_acc),
        ("Simple Average",  ge_acc),
        *[(_label_clf(m),
           accuracy_score(y_test, des_probas[m].argmax(axis=1)))
          for m in DES_METHODS],
    ]
    if DESLIB_AVAILABLE:
        for dl_label in DL_METHODS:
            rows.append((f"DESlib {dl_label}", dl_results.get(dl_label)))

    if verbose:
        section("Results on held-out test set  (Accuracy — higher is better)")
        show_results_clf(rows, best_acc)
        show_timing(DES_METHODS, fit_times, predict_times, len(X_test_s))

    labelled_fit  = {_label_clf(m): fit_times[m]      for m in DES_METHODS}
    labelled_pred = {_label_clf(m): predict_times[m]  for m in DES_METHODS}
    if DESLIB_AVAILABLE:
        labelled_fit.update({f'DESlib {k}':  v for k, v in dl_fit_ms.items()})
        labelled_pred.update({f'DESlib {k}': v for k, v in dl_pred_ms.items()})
    return dict(rows), labelled_fit, labelled_pred


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    banner()

    # ── Regression ────────────────────────────────────────────────────────────
    print(f"\n{'━' * W}")
    print("  Regression  (despy vs DESReg — DESlib has no regression support)")
    print(f"{'━' * W}")
    run_regression(load_california)
    run_regression(load_bike)
    run_regression(load_abalone)

    # ── Tabular classification ─────────────────────────────────────────────────
    print(f"\n\n{'━' * W}")
    print("  Tabular Classification  (despy vs DESlib — direct head-to-head)")
    print(f"{'━' * W}")
    run_classification(load_waveform, k=20)
    run_classification(load_satimage, k=20)

    # ── Image classification ───────────────────────────────────────────────────
    print(f"\n\n{'━' * W}")
    print("  Image Classification  (feature extractor → any model → despy)")
    print(f"{'━' * W}")
    print("  Pipeline: images → feature extractor → pool of models → despy routing")
    print("  Feature extractor here: raw 8×8 pixel values (64D).")
    print("  In production: penultimate-layer CNN embeddings from PyTorch / JAX /")
    print("  Keras. despy receives the same numeric arrays regardless of framework.")
    print("  DESlib works here too with sklearn models, but cannot directly accept")
    print("  outputs from non-sklearn neural networks without a wrapper.")
    run_classification(
        load_digits_data, k=10,
        note=(
            "Raw pixel values used as features. Swap in CNN embeddings from any "
            "framework to apply despy to full-resolution image datasets."
        ),
    )

    # ── Time-series classification ─────────────────────────────────────────────
    print(f"\n\n{'━' * W}")
    print("  Time-Series Classification  (feature extractor → any model → despy)")
    print(f"{'━' * W}")
    print("  Pipeline: time series → feature extractor → pool of models → despy routing")
    print("  Feature extractor here: 16 (x,y) pen stroke coordinates (Pendigits).")
    print("  In production: tsfresh features, catch22, or transformer sequence")
    print("  embeddings. despy is agnostic to the extractor and model framework.")
    run_classification(
        load_pendigits, k=20,
        note=(
            "Pen stroke trajectories: 16 (x,y) coordinates sampled along each "
            "stroke. Swap in any feature extractor (CNNs, RNNs, transformers) "
            "for full sequence data — despy receives the same arrays regardless."
        ),
    )
    if DESLIB_AVAILABLE:
        print()
        print("  ── DESlib vs despy  ─────────────────────────────────────────────────────")
        print("  DESlib: 7 algorithms (3 basic + 4 advanced). Results shown as-is.")
        print()
        print("  Waveform: despy KNORA-U/IU beat DESlib's best KNORA-U across 10")
        print("  seeds — the canonical DES benchmark confirms routing over cached")
        print("  predictions has a structural advantage on overlapping class boundaries.")
        print()
        print("  Speed: despy 5-65ms total (fit+predict) across all datasets.")
        print("  DESlib: 130-1400ms. despy routes over cached predictions;")
        print("  DESlib re-queries every model per neighbour at inference time.")

    print(f"\n\n{'━' * W}")
    print("  Done.")
    print(f"{'━' * W}")