"""Model pipeline, hyper-parameter tuning, cross-validation and final training.

Pipeline::

    StructuralFeatures -> [BuildingTypeClusterer] -> [VulnerabilityTyper]
        -> ColumnTransformer (num | cat | grp | te | nb | [hh] | [adj] | [ri])
        -> LGBMClassifier  or  XGBGradeClassifier

Optional steps/branches are switched by a feature configuration (see
:func:`scripts.preprocess.resolve_config`). :func:`run_ablation` chooses the
configuration greedily on a region subsample.

The final model is a :class:`BlendClassifier`: a weighted average of the
LightGBM and XGBoost pipelines (blend weight chosen on out-of-fold
predictions) followed by per-class weights tuned to maximise **macro F1**
(the competition metric).

Command line::

    python -m scripts.train --smoke               # 2-minute end-to-end check on a tiny subsample
    python -m scripts.train --ablation            # choose features, CV LightGBM + XGBoost, blend, fit
    python -m scripts.train --models lgbm         # LightGBM only
    python -m scripts.train --skip-cv             # final model only (no blend / class weights)
"""
import argparse
import json
import shutil
import tempfile
import time

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.metrics import ConfusionMatrixDisplay, classification_report
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline

from .feature_engineering import BuildingTypeClusterer, StructuralFeatures, VulnerabilityTyper
from .preprocess import (BASELINE_CONFIG, ID_COL, build_preprocessor, load_data, output_dir,
                         resolve_config, split_xy)

MODEL_NAME = "pipeline_model.pkl"
SEED = 42

#: Final LightGBM parameters (iteration 4: smaller learning rate, more trees).
DEFAULT_LGBM_PARAMS = dict(
    n_estimators=900, learning_rate=0.05, num_leaves=63, min_child_samples=60,
    colsample_bytree=0.4, subsample=0.8, subsample_freq=1, reg_lambda=5.0, max_bin=127,
    n_jobs=-1, random_state=7, verbose=-1,
)
#: Faster LightGBM used only to *compare* feature sets in the ablation.
ABLATION_LGBM_PARAMS = dict(n_estimators=300, learning_rate=0.1)
#: XGBoost parameters (labels are shifted to 0..K-1 inside XGBGradeClassifier).
DEFAULT_XGB_PARAMS = dict(
    n_estimators=600, learning_rate=0.07, max_depth=7, min_child_weight=5, subsample=0.8,
    colsample_bytree=0.5, reg_lambda=5.0, max_bin=128, tree_method="hist", enable_categorical=True,
    objective="multi:softprob", eval_metric="mlogloss", n_jobs=-1, random_state=7,
)
MODEL_TYPES = ("lgbm", "xgb")

PARAM_DISTRIBUTIONS = {
    "clf__num_leaves": [31, 63, 127, 255],
    "clf__learning_rate": [0.03, 0.05, 0.08, 0.1],
    "clf__n_estimators": [300, 500, 800],
    "clf__min_child_samples": [20, 60, 150],
    "clf__colsample_bytree": [0.3, 0.4, 0.6],
    "clf__subsample": [0.7, 0.8, 1.0],
    "clf__reg_lambda": [0.0, 1.0, 5.0, 10.0],
}

#: Greedy ablation steps: (name, config change, minimum gain to keep it).
#: Evidence-based EDA changes are kept when they do not hurt; clustering
#: experiments must improve held-out macro F1 by at least 0.002.
ABLATION_STEPS = (
    ("neighbor_v2 (jarak per rumah tangga)", {"neighbor": "v2"}, 0.0),
    ("household (bangunan serumah)", {"household": True}, 0.0),
    ("adjacent_area (ward/blok tetangga)", {"adjacent": True}, 0.0),
    ("extra_combos (lantai atas, umur, id9 x material)", {"extra_combos": True}, 0.0),
    ("residual_intensity (intensitas bersih dua tahap)", {"residual": True}, 0.0),
    ("kmeans_btype (clustering KMeans)", {"kmeans": True}, 0.002),
    ("vuln_vtype (tipe kerentanan supervised)", {"vuln": True}, 0.002),
)


# --------------------------------------------------------------------------
# model objects
# --------------------------------------------------------------------------
class XGBGradeClassifier(ClassifierMixin, BaseEstimator):
    """``xgboost.XGBClassifier`` that accepts arbitrary class labels (e.g. grades 1..5).

    XGBoost requires labels ``0..K-1``; this wrapper encodes them in ``fit``
    and exposes the original labels in ``classes_``.

    Parameters
    ----------
    params : dict, optional
        Overrides of :data:`DEFAULT_XGB_PARAMS`.
    """

    def __init__(self, params=None):
        """Store parameters (scikit-learn convention: no logic here)."""
        self.params = params

    def fit(self, X, y):
        """Fit XGBoost on label-encoded targets.

        Parameters
        ----------
        X : pandas.DataFrame
            Features (``category`` columns are handled natively).
        y : array-like

        Returns
        -------
        self
        """
        from xgboost import XGBClassifier  # imported lazily: optional dependency

        y = np.asarray(y)
        self.classes_ = np.unique(y)
        self.model_ = XGBClassifier(**{**DEFAULT_XGB_PARAMS, **(self.params or {})})
        self.model_.fit(X, np.searchsorted(self.classes_, y))
        return self

    def predict_proba(self, X):
        """Class probabilities (columns ordered like ``classes_``).

        Parameters
        ----------
        X : pandas.DataFrame

        Returns
        -------
        numpy.ndarray
        """
        return self.model_.predict_proba(X)

    def predict(self, X):
        """Most probable class label.

        Parameters
        ----------
        X : pandas.DataFrame

        Returns
        -------
        numpy.ndarray
        """
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]


def build_pipeline(params=None, config=None, memory=None, n_splits=5, random_state=SEED, model="lgbm"):
    """Create the full (unfitted) scikit-learn pipeline.

    Parameters
    ----------
    params : dict, optional
        Classifier parameters overriding :data:`DEFAULT_LGBM_PARAMS` (``model="lgbm"``)
        or :data:`DEFAULT_XGB_PARAMS` (``model="xgb"``).
    config : dict, optional
        Feature configuration (see :func:`scripts.preprocess.resolve_config`).
    memory : str or joblib.Memory, optional
        Cache for fitted transformers (speeds up hyper-parameter search).
    n_splits : int, default=5
        Cross-fitting folds of the supervised encoders.
    random_state : int, default=SEED
        Seed of the encoder folds.
    model : {"lgbm", "xgb"}, default="lgbm"
        Final classifier.

    Returns
    -------
    sklearn.pipeline.Pipeline
    """
    cfg = resolve_config(config)
    if model == "lgbm":
        clf = LGBMClassifier(**{**DEFAULT_LGBM_PARAMS, **(params or {})})
    elif model == "xgb":
        clf = XGBGradeClassifier(params)
    else:
        raise ValueError(f"unknown model {model!r}; choose from {MODEL_TYPES}")
    steps = [("struct", StructuralFeatures())]
    if cfg["kmeans"]:
        steps.append(("btype", BuildingTypeClusterer(random_state=random_state)))
    if cfg["vuln"]:
        steps.append(("vtype", VulnerabilityTyper(n_splits=n_splits, random_state=random_state)))
    steps += [
        ("prep", build_preprocessor(cfg, n_splits=n_splits, random_state=random_state)),
        ("clf", clf),
    ]
    return Pipeline(steps=steps, memory=memory)


class WeightedArgmaxClassifier(ClassifierMixin, BaseEstimator):
    """Probabilistic classifier whose decision rule is ``argmax(proba * weights)``.

    Re-weighting the class probabilities trades accuracy for a better
    balance between classes, which raises macro F1.

    Parameters
    ----------
    estimator : estimator with ``predict_proba``
        Typically the pipeline from :func:`build_pipeline`.
    class_weights : array-like of shape (n_classes,), optional
        Multipliers applied to the probabilities (``None`` = all ones).
    """

    def __init__(self, estimator=None, class_weights=None):
        """Store parameters (scikit-learn convention: no logic here)."""
        self.estimator = estimator
        self.class_weights = class_weights

    def fit(self, X, y):
        """Fit a clone of ``estimator``.

        Parameters
        ----------
        X : pandas.DataFrame
        y : array-like

        Returns
        -------
        self
        """
        self.estimator_ = clone(self.estimator).fit(X, y)
        self.classes_ = self.estimator_.classes_
        return self

    def predict_proba(self, X):
        """Return the class probabilities of the wrapped estimator.

        Parameters
        ----------
        X : pandas.DataFrame

        Returns
        -------
        numpy.ndarray of shape (n_rows, n_classes)
        """
        return self.estimator_.predict_proba(X)

    def predict(self, X):
        """Predict labels with the weighted argmax rule.

        Parameters
        ----------
        X : pandas.DataFrame

        Returns
        -------
        numpy.ndarray
        """
        w = np.ones(len(self.classes_)) if self.class_weights is None else np.asarray(self.class_weights)
        return self.classes_[np.argmax(self.predict_proba(X) * w, axis=1)]


class BlendClassifier(ClassifierMixin, BaseEstimator):
    """Weighted average of several probabilistic pipelines + per-class decision weights.

    ``predict_proba = sum_i blend_weights[i] * estimators[i].predict_proba``
    and ``predict = argmax(predict_proba * class_weights)``.

    Parameters
    ----------
    estimators : list of (str, estimator)
        Named unfitted pipelines (same classes, same label order).
    blend_weights : array-like, optional
        Weight of each estimator (default: equal weights; normalised to sum 1).
    class_weights : array-like, optional
        Per-class multipliers for the decision rule (``None`` = all ones).
    """

    def __init__(self, estimators=None, blend_weights=None, class_weights=None):
        """Store parameters (scikit-learn convention: no logic here)."""
        self.estimators = estimators
        self.blend_weights = blend_weights
        self.class_weights = class_weights

    def fit(self, X, y):
        """Fit a clone of every estimator on the same data.

        Parameters
        ----------
        X : pandas.DataFrame
        y : array-like

        Returns
        -------
        self
        """
        self.estimators_ = []
        for name, est in self.estimators:
            t0 = time.time()
            self.estimators_.append((name, clone(est).fit(X, y)))
            print(f"  fitted {name} in {(time.time() - t0) / 60:.1f} min", flush=True)
        self.classes_ = self.estimators_[0][1].classes_
        w = np.ones(len(self.estimators_)) if self.blend_weights is None else np.asarray(self.blend_weights, float)
        self.weights_ = w / w.sum()
        return self

    def predict_proba(self, X):
        """Blended class probabilities.

        Parameters
        ----------
        X : pandas.DataFrame

        Returns
        -------
        numpy.ndarray
        """
        return sum(w * est.predict_proba(X) for w, (_, est) in zip(self.weights_, self.estimators_))

    def predict(self, X):
        """Predict labels with the weighted argmax rule.

        Parameters
        ----------
        X : pandas.DataFrame

        Returns
        -------
        numpy.ndarray
        """
        w = np.ones(len(self.classes_)) if self.class_weights is None else np.asarray(self.class_weights)
        return self.classes_[np.argmax(self.predict_proba(X) * w, axis=1)]


# --------------------------------------------------------------------------
# metrics & class-weight optimisation
# --------------------------------------------------------------------------
def macro_f1(y_idx, pred_idx, n_classes=5):
    """Fast macro F1 for integer-coded labels (identical to sklearn's ``f1_score(average="macro")``).

    Parameters
    ----------
    y_idx, pred_idx : numpy.ndarray of int in ``0..n_classes-1``
    n_classes : int

    Returns
    -------
    float
    """
    cm = np.bincount(y_idx * n_classes + pred_idx, minlength=n_classes ** 2).reshape(n_classes, n_classes)
    tp = np.diag(cm)
    denom = cm.sum(axis=0) + cm.sum(axis=1)
    return float(np.mean(np.where(denom > 0, 2 * tp / np.maximum(denom, 1), 0.0)))


def optimize_class_weights(proba, y_idx, n_iter=4, grid=None):
    """Coordinate search for per-class probability weights that maximise macro F1.

    Parameters
    ----------
    proba : numpy.ndarray of shape (n_rows, n_classes)
        Out-of-fold probabilities.
    y_idx : numpy.ndarray of int
        True labels coded ``0..n_classes-1``.
    n_iter : int, default=4
        Passes over all classes.
    grid : numpy.ndarray, optional
        Candidate weight values (default: 49 log-spaced values in [0.3, 3.3]).

    Returns
    -------
    weights : numpy.ndarray of shape (n_classes,)
        Normalised so that the last class has weight 1.
    score : float
        Macro F1 obtained with ``weights``.
    """
    n_classes = proba.shape[1]
    grid = np.exp(np.linspace(-1.2, 1.2, 49)) if grid is None else grid
    w = np.ones(n_classes)
    best = macro_f1(y_idx, proba.argmax(1), n_classes)
    for _ in range(n_iter):
        for c in range(n_classes):
            for g in grid:
                cand = w.copy()
                cand[c] = g
                score = macro_f1(y_idx, (proba * cand).argmax(1), n_classes)
                if score > best:
                    best, w = score, cand
    return w / w[-1], best


def honest_weight_cv(proba, y_idx, fold_ids):
    """Estimate the macro F1 of the weight tuning without optimistic bias.

    For each fold the weights are optimised on the other folds and then
    evaluated on the held-out fold.

    Parameters
    ----------
    proba : numpy.ndarray of shape (n_rows, n_classes)
    y_idx : numpy.ndarray of int
    fold_ids : numpy.ndarray of int
        Fold number of every row.

    Returns
    -------
    list of float
        Held-out macro F1 per fold.
    """
    scores = []
    for f in np.unique(fold_ids):
        m = fold_ids != f
        w, _ = optimize_class_weights(proba[m], y_idx[m])
        scores.append(macro_f1(y_idx[~m], (proba[~m] * w).argmax(1), proba.shape[1]))
    return scores


# --------------------------------------------------------------------------
# tuning
# --------------------------------------------------------------------------
def region_subsample(X, y, frac=0.25, prefix_len=4, random_state=0):
    """Sample whole regions (``building_id`` prefix) instead of random rows.

    Keeping complete regions preserves the neighbour density inside each
    area, so neighbour / target-encoding features behave like on the full
    data while the tuning runs on a fraction of it.

    Parameters
    ----------
    X : pandas.DataFrame
    y : numpy.ndarray
    frac : float, default=0.25
        Fraction of regions kept.
    prefix_len : int, default=4
        Prefix length defining a region (4 = old VDC).
    random_state : int, default=0

    Returns
    -------
    X_sub : pandas.DataFrame
    y_sub : numpy.ndarray
    """
    region = X[ID_COL].astype(np.int64).astype(str).str.zfill(12).str[:prefix_len].to_numpy()
    uniq = np.unique(region)
    rng = np.random.default_rng(random_state)
    keep = rng.choice(uniq, size=max(1, int(len(uniq) * frac)), replace=False)
    mask = np.isin(region, keep)
    return X.loc[mask], y[mask]


def tune_hyperparameters(X, y, config=None, n_iter=8, frac=0.25, cv=3, random_state=SEED, verbose=1):
    """Tune LightGBM inside the full pipeline with ``RandomizedSearchCV`` (scoring = macro F1).

    The search runs on a region subsample (see :func:`region_subsample`).
    Fitted transformers are cached, so candidates that only differ in
    LightGBM parameters reuse the same features. The default parameters are
    evaluated on the same folds and kept if they beat the best candidate.

    Parameters
    ----------
    X : pandas.DataFrame
    y : numpy.ndarray
    config : dict, optional
        Feature configuration of the pipeline being tuned.
    n_iter : int, default=8
        Number of sampled parameter settings.
    frac : float, default=0.25
        Fraction of regions used.
    cv : int, default=3
        Folds of the search.
    random_state : int, default=SEED
    verbose : int, default=1

    Returns
    -------
    best_params : dict
        LightGBM parameters (without the ``clf__`` prefix); empty dict if
        the defaults won.
    results : pandas.DataFrame
        ``cv_results_`` of the search plus a row for the defaults.
    """
    X_sub, y_sub = region_subsample(X, y, frac=frac, random_state=random_state)
    print(f"tuning on {len(X_sub):,} rows ({frac:.0%} of regions)")
    folds = StratifiedKFold(cv, shuffle=True, random_state=random_state)
    cache = tempfile.mkdtemp(prefix="sdc_cache_")
    try:
        pipe = build_pipeline(config=config, memory=cache)
        search = RandomizedSearchCV(pipe, PARAM_DISTRIBUTIONS, n_iter=n_iter, scoring="f1_macro",
                                    cv=folds, n_jobs=1, random_state=random_state, refit=False,
                                    verbose=verbose, error_score="raise")
        search.fit(X_sub, y_sub)
        default_score = cross_val_score(pipe, X_sub, y_sub, scoring="f1_macro", cv=folds, n_jobs=1).mean()
    finally:
        shutil.rmtree(cache, ignore_errors=True)

    results = pd.DataFrame(search.cv_results_)
    keep_cols = [c for c in results.columns if c.startswith("param_")] + ["mean_test_score", "std_test_score"]
    results = results[keep_cols].sort_values("mean_test_score", ascending=False)
    results = pd.concat([results, pd.DataFrame([{"param_clf__num_leaves": "DEFAULT",
                                                  "mean_test_score": default_score}])],
                        ignore_index=True)
    print(f"best search macro F1 = {search.best_score_:.4f} | default params = {default_score:.4f}")
    if default_score >= search.best_score_:
        return {}, results
    return {k.replace("clf__", ""): v for k, v in search.best_params_.items()}, results


# --------------------------------------------------------------------------
# cross-validation & final fit
# --------------------------------------------------------------------------
def cross_validate_oof(X, y, params=None, config=None, n_splits=5, random_state=SEED, verbose=True,
                       model="lgbm"):
    """Out-of-fold probabilities of the full pipeline (stratified K-fold).

    Every fold refits the whole pipeline (encoders included) on the training
    part only, so the validation rows are encoded exactly like test rows.

    Parameters
    ----------
    X : pandas.DataFrame
    y : numpy.ndarray of int (1..5)
    params : dict, optional
        LightGBM parameters.
    config : dict, optional
        Feature configuration.
    n_splits : int, default=5
    random_state : int, default=SEED
    verbose : bool, default=True
        Print the score of every fold.
    model : {"lgbm", "xgb"}, default="lgbm"
        Classifier of the pipeline. The folds depend only on ``y`` and
        ``random_state``, so different models get identical folds.

    Returns
    -------
    oof : numpy.ndarray of shape (n_rows, n_classes)
    fold_ids : numpy.ndarray of int
    classes : numpy.ndarray
    """
    classes = np.unique(y)
    y_idx = np.searchsorted(classes, y)
    oof = np.zeros((len(y), len(classes)))
    fold_ids = np.full(len(y), -1)
    skf = StratifiedKFold(n_splits, shuffle=True, random_state=random_state)
    for f, (tr, va) in enumerate(skf.split(X, y)):
        t0 = time.time()
        pipe = build_pipeline(params, config, model=model).fit(X.iloc[tr], y[tr])
        oof[va] = pipe.predict_proba(X.iloc[va])
        fold_ids[va] = f
        if verbose:
            score = macro_f1(y_idx[va], oof[va].argmax(1), len(classes))
            print(f"[{model}] fold {f}: macro F1 (argmax) = {score:.4f}  [{(time.time() - t0) / 60:.1f} min]",
                  flush=True)
        del pipe
    return oof, fold_ids, classes


def summarize_cv(oof, y, fold_ids, classes):
    """Optimise class weights on the OOF predictions and collect CV metrics.

    Parameters
    ----------
    oof : numpy.ndarray of shape (n_rows, n_classes)
    y : numpy.ndarray
    fold_ids : numpy.ndarray of int
    classes : numpy.ndarray

    Returns
    -------
    dict
        ``class_weights``, ``macro_f1_argmax``, ``macro_f1_weighted_oof``,
        ``macro_f1_heldout_per_fold``, ``macro_f1_heldout_mean`` and the
        classification report of the weighted predictions.
    """
    y_idx = np.searchsorted(classes, y)
    weights, best = optimize_class_weights(oof, y_idx)
    heldout = honest_weight_cv(oof, y_idx, fold_ids)
    pred = classes[(oof * weights).argmax(1)]
    return {
        "class_weights": weights.round(4).tolist(),
        "macro_f1_argmax": round(macro_f1(y_idx, oof.argmax(1), len(classes)), 4),
        "macro_f1_weighted_oof": round(best, 4),
        "macro_f1_heldout_per_fold": [round(s, 4) for s in heldout],
        "macro_f1_heldout_mean": round(float(np.mean(heldout)), 4),
        "report": classification_report(y, pred, digits=4, output_dict=True),
    }


def evaluate_config(X, y, config, params=None, n_splits=3, random_state=SEED):
    """Cross-validated macro F1 of one feature configuration.

    Parameters
    ----------
    X : pandas.DataFrame
    y : numpy.ndarray
    config : dict
        Feature configuration.
    params : dict, optional
        LightGBM parameters.
    n_splits : int, default=3
    random_state : int, default=SEED

    Returns
    -------
    dict
        ``macro_f1_argmax``, ``macro_f1_heldout`` (class weights fitted on
        the other folds) and ``minutes``.
    """
    t0 = time.time()
    oof, fold_ids, classes = cross_validate_oof(X, y, params, config, n_splits=n_splits,
                                                random_state=random_state, verbose=False)
    y_idx = np.searchsorted(classes, y)
    return {"macro_f1_argmax": macro_f1(y_idx, oof.argmax(1), len(classes)),
            "macro_f1_heldout": float(np.mean(honest_weight_cv(oof, y_idx, fold_ids))),
            "minutes": (time.time() - t0) / 60}


def run_ablation(X, y, params=ABLATION_LGBM_PARAMS, frac=0.3, n_splits=3, steps=ABLATION_STEPS,
                 random_state=SEED):
    """Greedy forward ablation of the iteration-3 features on a region subsample.

    Starts from :data:`scripts.preprocess.BASELINE_CONFIG` (iteration 2) and
    tries the steps of :data:`ABLATION_STEPS` in order. A step is kept when
    the held-out macro F1 improves by more than its minimum gain over the
    current best configuration. All configurations use the same rows and
    folds, so the comparison is paired.

    Parameters
    ----------
    X : pandas.DataFrame
    y : numpy.ndarray
    params : dict, default=ABLATION_LGBM_PARAMS
        LightGBM parameters (fast settings: only the *difference* between
        feature sets matters here).
    frac : float, default=0.3
        Fraction of regions (old VDC) used.
    n_splits : int, default=3
    steps : tuple, default=ABLATION_STEPS
    random_state : int, default=SEED

    Returns
    -------
    config : dict
        Selected feature configuration.
    results : pandas.DataFrame
        One row per evaluated configuration.
    """
    X_sub, y_sub = region_subsample(X, y, frac=frac, random_state=random_state)
    print(f"ablation on {len(X_sub):,} rows ({frac:.0%} of regions), {n_splits}-fold CV")
    current = dict(BASELINE_CONFIG)
    res = evaluate_config(X_sub, y_sub, current, params, n_splits, random_state)
    best = res["macro_f1_heldout"]
    rows = [{"step": "baseline (iterasi 2)", **res, "gain": 0.0, "kept": True}]
    print(f"  baseline: {best:.4f}  [{res['minutes']:.1f} min]", flush=True)
    for name, change, min_gain in steps:
        candidate = {**current, **change}
        res = evaluate_config(X_sub, y_sub, candidate, params, n_splits, random_state)
        gain = res["macro_f1_heldout"] - best
        kept = gain > min_gain
        if kept:
            current, best = candidate, res["macro_f1_heldout"]
        rows.append({"step": name, **res, "gain": gain, "kept": kept})
        print(f"  {name}: {res['macro_f1_heldout']:.4f} (gain {gain:+.4f}) -> "
              f"{'KEEP' if kept else 'drop'}  [{res['minutes']:.1f} min]", flush=True)
    results = pd.DataFrame(rows)
    results[["macro_f1_argmax", "macro_f1_heldout", "gain", "minutes"]] = \
        results[["macro_f1_argmax", "macro_f1_heldout", "gain", "minutes"]].round(4)
    return current, results


def optimize_blend(oof_a, oof_b, y_idx, fold_ids, alphas=None):
    """Choose the blend weight of two OOF probability matrices for macro F1.

    For every ``alpha`` the blend ``alpha * oof_a + (1 - alpha) * oof_b`` is
    scored with class weights optimised in-sample (fast, 2 passes); the best
    ``alpha`` is then re-evaluated honestly with :func:`honest_weight_cv`.

    Parameters
    ----------
    oof_a, oof_b : numpy.ndarray of shape (n_rows, n_classes)
    y_idx : numpy.ndarray of int
    fold_ids : numpy.ndarray of int
    alphas : sequence of float, optional
        Candidate weights of ``oof_a`` (default 0, 0.1, ..., 1).

    Returns
    -------
    alpha : float
    table : pandas.DataFrame
        In-sample macro F1 per alpha.
    heldout : dict
        Held-out macro F1 of model A, model B and the chosen blend.
    """
    alphas = np.round(np.linspace(0, 1, 11), 2) if alphas is None else np.asarray(alphas)
    scores = [optimize_class_weights(a * oof_a + (1 - a) * oof_b, y_idx, n_iter=2)[1] for a in alphas]
    alpha = float(alphas[int(np.argmax(scores))])
    heldout = {"a": float(np.mean(honest_weight_cv(oof_a, y_idx, fold_ids))),
               "b": float(np.mean(honest_weight_cv(oof_b, y_idx, fold_ids))),
               "blend": float(np.mean(honest_weight_cv(alpha * oof_a + (1 - alpha) * oof_b, y_idx, fold_ids)))}
    return alpha, pd.DataFrame({"alpha_a": alphas, "macro_f1_oof": np.round(scores, 4)}), heldout


def fit_blend_model(X, y, config=None, params=None, blend_weights=None, class_weights=None,
                    models=MODEL_TYPES):
    """Fit the final :class:`BlendClassifier` on all training rows.

    Parameters
    ----------
    X : pandas.DataFrame
    y : numpy.ndarray
    config : dict, optional
        Feature configuration.
    params : dict, optional
        ``{"lgbm": {...}, "xgb": {...}}`` parameter overrides.
    blend_weights : array-like, optional
        Weight per model (same order as ``models``).
    class_weights : array-like, optional
        Per-class decision weights.
    models : tuple of str, default=MODEL_TYPES

    Returns
    -------
    BlendClassifier
    """
    params = params or {}
    estimators = [(m, build_pipeline(params.get(m), config, model=m)) for m in models]
    return BlendClassifier(estimators, blend_weights, class_weights).fit(X, y)


def fit_final_model(X, y, params=None, class_weights=None, config=None):
    """Fit the final pipeline on all training rows.

    Parameters
    ----------
    X : pandas.DataFrame
    y : numpy.ndarray
    params : dict, optional
        LightGBM parameters.
    class_weights : array-like, optional
        Per-class weights from :func:`summarize_cv`.
    config : dict, optional
        Feature configuration.

    Returns
    -------
    WeightedArgmaxClassifier
    """
    return WeightedArgmaxClassifier(build_pipeline(params, config), class_weights).fit(X, y)


def save_model(model, path=None):
    """Serialise the fitted model with joblib.

    Parameters
    ----------
    model : WeightedArgmaxClassifier
    path : str or Path, optional
        Default: ``<output>/models/pipeline_model_lgbm.pkl``.

    Returns
    -------
    pathlib.Path
    """
    path = path or output_dir() / "models" / MODEL_NAME
    joblib.dump(model, path, compress=3)
    return path


# --------------------------------------------------------------------------
# figures
# --------------------------------------------------------------------------
def plot_feature_importance(model, path=None, top=25):
    """Save a horizontal bar chart of the top LightGBM gain importances.

    Parameters
    ----------
    model : BlendClassifier or WeightedArgmaxClassifier
        Fitted model (for a blend, its LightGBM pipeline is used).
    path : str or Path, optional
        Default: ``<output>/reports/figures/feature_importance.png``.
    top : int, default=25

    Returns
    -------
    pandas.Series
        Gain importance of every feature, sorted descending.
    """
    if hasattr(model, "estimators_"):
        pipe = dict(model.estimators_).get("lgbm", model.estimators_[0][1])
    else:
        pipe = model.estimator_
    names = pipe.named_steps["prep"].get_feature_names_out()
    gain = pipe.named_steps["clf"].booster_.feature_importance(importance_type="gain")
    imp = pd.Series(gain, index=names).sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(8, 9))
    imp.head(top)[::-1].plot.barh(ax=ax, color="#4C72B0")
    ax.set_title(f"Top {top} feature importance (LightGBM gain)")
    ax.set_xlabel("total gain")
    fig.tight_layout()
    fig.savefig(path or output_dir() / "reports" / "figures" / "feature_importance.png", dpi=150)
    plt.close(fig)
    return imp


def plot_confusion(y, pred, path=None):
    """Save a row-normalised confusion matrix of the CV predictions.

    Parameters
    ----------
    y, pred : array-like
        True and predicted grades.
    path : str or Path, optional
        Default: ``<output>/reports/figures/confusion_matrix.png``.

    Returns
    -------
    pathlib.Path
    """
    path = path or output_dir() / "reports" / "figures" / "confusion_matrix.png"
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay.from_predictions(y, pred, normalize="true", values_format=".2f",
                                            cmap="Blues", ax=ax, colorbar=False)
    ax.set_title("Confusion matrix (OOF, normalised per true grade)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def save_oof(oof, fold_ids, y, path=None):
    """Save out-of-fold probabilities for error analysis (``reports/oof_<model>.npz``).

    Parameters
    ----------
    oof : numpy.ndarray of shape (n_rows, n_classes)
    fold_ids : numpy.ndarray of int
    y : numpy.ndarray
    path : str or Path, optional

    Returns
    -------
    pathlib.Path
    """
    path = path or output_dir() / "reports" / "oof_predictions.npz"
    np.savez_compressed(path, oof=oof.astype(np.float32), fold=fold_ids, y=y)
    return path


def smoke_test(X, y, test, frac=0.03, random_state=SEED):
    """Run every code path end-to-end on a tiny region subsample (a few minutes).

    Covers all optional feature steps (clustering, residual, household,
    adjacent, extra combos), the ablation evaluation, LightGBM and XGBoost
    CV, blend optimisation, the final blend fit and test prediction. Raises
    on the first error, so problems show up before the long run starts.

    Parameters
    ----------
    X : pandas.DataFrame
    y : numpy.ndarray
    test : pandas.DataFrame
    frac : float, default=0.03
        Fraction of regions used.
    random_state : int, default=SEED

    Returns
    -------
    dict
        Tiny-sample scores (only meaningful as "it runs").
    """
    t0 = time.time()
    X_s, y_s = region_subsample(X, y, frac=frac, random_state=random_state)
    full = {k: True for k in resolve_config(None)}
    full["neighbor"] = "v2"
    tiny = {"lgbm": dict(n_estimators=20), "xgb": dict(n_estimators=20)}
    print(f"smoke test on {len(X_s):,} rows")
    base = evaluate_config(X_s, y_s, BASELINE_CONFIG, tiny["lgbm"], n_splits=2)
    oofs = {}
    for m in MODEL_TYPES:
        oofs[m], fold_ids, classes = cross_validate_oof(X_s, y_s, tiny[m], full, n_splits=2, model=m)
    y_idx = np.searchsorted(classes, y_s)
    alpha, _, heldout = optimize_blend(oofs["lgbm"], oofs["xgb"], y_idx, fold_ids)
    model = fit_blend_model(X_s, y_s, full, tiny, [alpha, 1 - alpha])
    pred = model.predict(test.head(2000))
    assert set(np.unique(pred)) <= set(classes.tolist())
    plot_feature_importance(model)
    print(f"smoke test OK in {(time.time() - t0) / 60:.1f} min")
    return {"baseline": round(base["macro_f1_heldout"], 4), **{k: round(v, 4) for k, v in heldout.items()}}


def run_training(skip_cv=False, n_splits=5, ablation=False, ablation_frac=0.3, config=None,
                 models=MODEL_TYPES, tune=False, n_iter=8, tune_frac=0.25):
    """End-to-end training: ablation, (tuning), CV of every model, blend, final fit, artefacts.

    Parameters
    ----------
    skip_cv : bool, default=False
        Skip cross-validation (equal blend weights, class weights = 1).
    n_splits : int, default=5
        Folds of the cross-validation.
    ablation : bool, default=False
        Choose the feature configuration with :func:`run_ablation`.
    ablation_frac : float, default=0.3
        Fraction of regions used for the ablation.
    config : dict, optional
        Feature configuration used when ``ablation`` is False.
    models : tuple of str, default=("lgbm", "xgb")
        Models to cross-validate and blend.
    tune : bool, default=False
        Tune LightGBM with :func:`tune_hyperparameters`.
    n_iter : int, default=8
    tune_frac : float, default=0.25

    Returns
    -------
    model : BlendClassifier
    summary : dict
    """
    out = output_dir()
    train, _, _ = load_data()
    X, y = split_xy(train)

    cfg = resolve_config(config)
    if ablation:
        cfg, abl = run_ablation(X, y, frac=ablation_frac)
        abl.to_csv(out / "reports" / "ablation_results.csv", index=False)
        print(abl.to_string(index=False))
    print("feature config:", cfg)
    params = {m: {} for m in models}
    if tune and "lgbm" in models:
        params["lgbm"], results = tune_hyperparameters(X, y, config=cfg, n_iter=n_iter, frac=tune_frac)
        results.to_csv(out / "reports" / "tuning_results.csv", index=False)
    summary = {"config": cfg, "models": list(models),
               "params": {"lgbm": {**DEFAULT_LGBM_PARAMS, **params.get("lgbm", {})},
                          "xgb": {**DEFAULT_XGB_PARAMS, **params.get("xgb", {})}}}

    blend_weights, class_weights = None, None
    if not skip_cv:
        oofs = {}
        for m in models:
            oofs[m], fold_ids, classes = cross_validate_oof(X, y, params[m], cfg, n_splits=n_splits, model=m)
            save_oof(oofs[m], fold_ids, y, out / "reports" / f"oof_{m}.npz")
        y_idx = np.searchsorted(classes, y)
        if len(models) == 2:
            alpha, table, heldout = optimize_blend(oofs[models[0]], oofs[models[1]], y_idx, fold_ids)
            blend_weights = [alpha, 1 - alpha]
            table.to_csv(out / "reports" / "blend_search.csv", index=False)
            summary["heldout_by_model"] = {models[0]: round(heldout["a"], 4), models[1]: round(heldout["b"], 4),
                                           "blend": round(heldout["blend"], 4)}
            blended = alpha * oofs[models[0]] + (1 - alpha) * oofs[models[1]]
        else:
            blend_weights = [1.0]
            blended = oofs[models[0]]
        cv = summarize_cv(blended, y, fold_ids, classes)
        class_weights = cv["class_weights"]
        plot_confusion(y, classes[(blended * np.asarray(class_weights)).argmax(1)])
        summary.update({k: v for k, v in cv.items() if k != "report"})
        summary["blend_weights"] = dict(zip(models, np.round(blend_weights, 2).tolist()))
        summary["per_class_f1"] = {k: round(v["f1-score"], 4) for k, v in cv["report"].items()
                                   if k not in ("accuracy", "macro avg", "weighted avg")}
        print(json.dumps({k: summary[k] for k in summary if k not in ("params", "config")}, indent=1, default=str))

    t0 = time.time()
    model = fit_blend_model(X, y, cfg, params, blend_weights, class_weights, models)
    print(f"final model fitted in {(time.time() - t0) / 60:.1f} min -> {save_model(model)}")
    imp = plot_feature_importance(model)
    summary["top_features"] = imp.head(20).round(0).to_dict()
    with open(out / "reports" / "cv_summary.json", "w") as fh:
        json.dump(summary, fh, indent=2, default=str)
    return model, summary


def main(argv=None):
    """Parse command-line arguments and run :func:`run_training` (or :func:`smoke_test`).

    Parameters
    ----------
    argv : list of str, optional
    """
    parser = argparse.ArgumentParser(description="Train the structural damage model.")
    parser.add_argument("--smoke", action="store_true", help="quick end-to-end check only")
    parser.add_argument("--ablation", action="store_true", help="choose the feature config by ablation")
    parser.add_argument("--ablation-frac", type=float, default=0.3, help="fraction of regions for ablation")
    parser.add_argument("--models", default="lgbm,xgb", help="comma separated: lgbm,xgb")
    parser.add_argument("--tune", action="store_true", help="RandomizedSearchCV for LightGBM")
    parser.add_argument("--n-iter", type=int, default=8, help="random search candidates")
    parser.add_argument("--tune-frac", type=float, default=0.25, help="fraction of regions for tuning")
    parser.add_argument("--skip-cv", action="store_true", help="skip cross-validation")
    parser.add_argument("--n-splits", type=int, default=5, help="CV folds")
    args = parser.parse_args(argv)
    if args.smoke:
        train, test, _ = load_data()
        X, y = split_xy(train)
        print(smoke_test(X, y, test))
        return
    run_training(skip_cv=args.skip_cv, n_splits=args.n_splits, ablation=args.ablation,
                 ablation_frac=args.ablation_frac, models=tuple(args.models.split(",")),
                 tune=args.tune, n_iter=args.n_iter, tune_frac=args.tune_frac)


if __name__ == "__main__":
    main()
