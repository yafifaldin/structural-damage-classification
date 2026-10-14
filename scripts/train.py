"""Model pipeline, hyper-parameter tuning, cross-validation and final training.

Pipeline::

    StructuralFeatures -> [BuildingTypeClusterer] -> [VulnerabilityTyper]
        -> ColumnTransformer (num | cat | grp | te | nb | [hh] | [adj]) -> LGBMClassifier

Optional steps/branches are switched by a feature configuration (see
:func:`scripts.preprocess.resolve_config`). :func:`run_ablation` chooses the
configuration greedily on a region subsample.

The fitted pipeline is wrapped in :class:`WeightedArgmaxClassifier`, which
multiplies the predicted class probabilities by per-class weights tuned to
maximise **macro F1** (the competition metric).

Command line::

    python -m scripts.train                       # CV + final model, default config & params
    python -m scripts.train --ablation            # choose the feature config first
    python -m scripts.train --ablation --tune     # + RandomizedSearchCV on the pipeline
    python -m scripts.train --skip-cv             # final model only (weights = 1)
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

MODEL_NAME = "pipeline_model_lgbm.pkl"
SEED = 42

DEFAULT_LGBM_PARAMS = dict(
    n_estimators=500, learning_rate=0.08, num_leaves=63, min_child_samples=60,
    colsample_bytree=0.4, subsample=0.8, subsample_freq=1, reg_lambda=5.0, max_bin=127,
    n_jobs=-1, random_state=7, verbose=-1,
)

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
    ("kmeans_btype (clustering KMeans)", {"kmeans": True}, 0.002),
    ("vuln_vtype (tipe kerentanan supervised)", {"vuln": True}, 0.002),
)


# --------------------------------------------------------------------------
# model objects
# --------------------------------------------------------------------------
def build_pipeline(params=None, config=None, memory=None, n_splits=5, random_state=SEED):
    """Create the full (unfitted) scikit-learn pipeline.

    Parameters
    ----------
    params : dict, optional
        LightGBM parameters overriding :data:`DEFAULT_LGBM_PARAMS`.
    config : dict, optional
        Feature configuration (see :func:`scripts.preprocess.resolve_config`).
    memory : str or joblib.Memory, optional
        Cache for fitted transformers (speeds up hyper-parameter search).
    n_splits : int, default=5
        Cross-fitting folds of the supervised encoders.
    random_state : int, default=SEED
        Seed of the encoder folds.

    Returns
    -------
    sklearn.pipeline.Pipeline
    """
    cfg = resolve_config(config)
    lgbm_params = {**DEFAULT_LGBM_PARAMS, **(params or {})}
    steps = [("struct", StructuralFeatures())]
    if cfg["kmeans"]:
        steps.append(("btype", BuildingTypeClusterer(random_state=random_state)))
    if cfg["vuln"]:
        steps.append(("vtype", VulnerabilityTyper(n_splits=n_splits, random_state=random_state)))
    steps += [
        ("prep", build_preprocessor(cfg, n_splits=n_splits, random_state=random_state)),
        ("clf", LGBMClassifier(**lgbm_params)),
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
def cross_validate_oof(X, y, params=None, config=None, n_splits=5, random_state=SEED, verbose=True):
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
        pipe = build_pipeline(params, config).fit(X.iloc[tr], y[tr])
        oof[va] = pipe.predict_proba(X.iloc[va])
        fold_ids[va] = f
        if verbose:
            score = macro_f1(y_idx[va], oof[va].argmax(1), len(classes))
            print(f"fold {f}: macro F1 (argmax) = {score:.4f}  [{time.time() - t0:.0f}s]", flush=True)
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


def run_ablation(X, y, params=None, frac=0.3, n_splits=3, steps=ABLATION_STEPS, random_state=SEED):
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
    params : dict, optional
        LightGBM parameters.
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
    model : WeightedArgmaxClassifier
        Fitted model.
    path : str or Path, optional
        Default: ``<output>/reports/figures/feature_importance.png``.
    top : int, default=25

    Returns
    -------
    pandas.Series
        Gain importance of every feature, sorted descending.
    """
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
    """Save out-of-fold probabilities for error analysis (``reports/oof_predictions.npz``).

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


def run_training(tune=False, n_iter=8, tune_frac=0.25, skip_cv=False, n_splits=5,
                 ablation=False, ablation_frac=0.3, config=None):
    """End-to-end training: optional ablation and tuning, CV, class weights, final fit, artefacts.

    Parameters
    ----------
    tune : bool, default=False
        Run :func:`tune_hyperparameters` first.
    n_iter : int, default=8
        Candidates of the random search.
    tune_frac : float, default=0.25
        Fraction of regions used for tuning.
    skip_cv : bool, default=False
        Skip cross-validation (class weights are then all ones).
    n_splits : int, default=5
        Folds of the cross-validation.
    ablation : bool, default=False
        Choose the feature configuration with :func:`run_ablation`.
    ablation_frac : float, default=0.3
        Fraction of regions used for the ablation.
    config : dict, optional
        Feature configuration used when ``ablation`` is False.

    Returns
    -------
    model : WeightedArgmaxClassifier
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
    summary = {"config": cfg, "params": dict(DEFAULT_LGBM_PARAMS)}

    params = {}
    if tune:
        params, results = tune_hyperparameters(X, y, config=cfg, n_iter=n_iter, frac=tune_frac)
        results.to_csv(out / "reports" / "tuning_results.csv", index=False)
    summary["params"].update(params)

    weights = None
    if not skip_cv:
        oof, fold_ids, classes = cross_validate_oof(X, y, params, cfg, n_splits=n_splits)
        save_oof(oof, fold_ids, y)
        cv = summarize_cv(oof, y, fold_ids, classes)
        weights = cv["class_weights"]
        plot_confusion(y, classes[(oof * np.asarray(weights)).argmax(1)])
        summary.update({k: v for k, v in cv.items() if k != "report"})
        summary["per_class_f1"] = {k: round(v["f1-score"], 4) for k, v in cv["report"].items()
                                   if k not in ("accuracy", "macro avg", "weighted avg")}
        print(json.dumps({k: summary[k] for k in ("macro_f1_argmax", "macro_f1_heldout_mean",
                                                  "class_weights")}, indent=1))

    t0 = time.time()
    model = fit_final_model(X, y, params, weights, cfg)
    print(f"final model fitted in {time.time() - t0:.0f}s -> {save_model(model)}")
    plot_feature_importance(model)
    with open(out / "reports" / "cv_summary.json", "w") as fh:
        json.dump(summary, fh, indent=2, default=str)
    return model, summary


def main(argv=None):
    """Parse command-line arguments and run :func:`run_training`.

    Parameters
    ----------
    argv : list of str, optional
    """
    parser = argparse.ArgumentParser(description="Train the structural damage model.")
    parser.add_argument("--ablation", action="store_true", help="choose the feature config by ablation")
    parser.add_argument("--ablation-frac", type=float, default=0.3, help="fraction of regions for ablation")
    parser.add_argument("--tune", action="store_true", help="run RandomizedSearchCV")
    parser.add_argument("--n-iter", type=int, default=8, help="random search candidates")
    parser.add_argument("--tune-frac", type=float, default=0.25, help="fraction of regions for tuning")
    parser.add_argument("--skip-cv", action="store_true", help="skip cross-validation")
    parser.add_argument("--n-splits", type=int, default=5, help="CV folds")
    args = parser.parse_args(argv)
    run_training(tune=args.tune, n_iter=args.n_iter, tune_frac=args.tune_frac, skip_cv=args.skip_cv,
                 n_splits=args.n_splits, ablation=args.ablation, ablation_frac=args.ablation_frac)


if __name__ == "__main__":
    main()
