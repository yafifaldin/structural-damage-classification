"""Paths, data loading, column definitions and the ``ColumnTransformer``.

Works both locally (data in ``<repo>/data``) and on Kaggle (data in
``/kaggle/input/...``). When the repository folder is read-only (e.g. it was
uploaded to Kaggle as a Dataset) all outputs go to ``/kaggle/working``.
"""
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer

from .feature_engineering import (
    DEFAULT_CHAINS,
    DEFAULT_COMBOS,
    EXTRA_COMBOS,
    HELPER_COLS,
    KMEANS_COMBOS,
    LOCATION_FREE_CAT_COLS,
    LOCATION_FREE_NUM_COLS,
    RESIDUAL_LEVELS,
    STRUCTURAL_FEATURES,
    VULN_COMBOS,
    AdjacentAreaEncoder,
    GroupRelativeFeatures,
    HierarchicalTargetEncoder,
    HouseholdEncoder,
    IdNeighborEncoder,
    ResidualIntensityEncoder,
)

COMPETITION = "structural-damage-classification"
ROOT = Path(__file__).resolve().parents[1]
TARGET = "damage_grade"
ID_COL = "building_id"

NUM_COLS = ["count_floors_pre_eq", "age_building", "plinth_area_sq_ft", "height_ft_pre_eq"]
LOC_COLS = ["district_id", "vdcmun_id", "ward_id"]
CAT_COLS = ["land_surface_condition", "foundation_type", "roof_type", "ground_floor_type",
            "other_floor_type", "position", "plan_configuration", "superstructure"]

# Column groups fed to each branch of the ColumnTransformer.
PASSTHROUGH_COLS = NUM_COLS + LOC_COLS + [c for c in STRUCTURAL_FEATURES if c not in HELPER_COLS]
GROUP_INPUT_COLS = ["id6", "id9", "ward_id", "age_building", "height_ft_pre_eq",
                    "count_floors_pre_eq", "plinth_area_sq_ft", "vuln", "superstructure"]
NEIGHBOR_INPUT_COLS = ["bid", "id6"]

# --------------------------------------------------------------------------
# feature configuration (iteration 3 ablation)
# --------------------------------------------------------------------------
#: Feature set of iteration 2 (starting point of the ablation).
BASELINE_CONFIG = dict(neighbor="v1", household=False, adjacent=False, extra_combos=False,
                       residual=False, kmeans=False, vuln=False)
#: Default when no ablation is run: EDA-based changes + residual intensity on, clustering off.
DEFAULT_CONFIG = dict(neighbor="v2", household=True, adjacent=True, extra_combos=True,
                      residual=True, kmeans=False, vuln=False)
RESIDUAL_INPUT_COLS = list(dict.fromkeys(
    list(LOCATION_FREE_CAT_COLS) + list(LOCATION_FREE_NUM_COLS) + [c for c, _ in RESIDUAL_LEVELS] + ["bid", "id6"]))
NEIGHBOR_PARAMS = {
    "v1": dict(ks=(3, 10, 30), window=30, household_div=1, exclude_household=False),
    "v2": dict(ks=(1, 2, 5, 10, 30), window=40, household_div=10, exclude_household=True),
}


def resolve_config(config=None):
    """Complete a (partial) feature configuration with :data:`DEFAULT_CONFIG`.

    Parameters
    ----------
    config : dict, optional
        Keys: ``neighbor`` ("v1"/"v2"), ``household``, ``adjacent``,
        ``extra_combos``, ``residual``, ``kmeans``, ``vuln`` (bool).

    Returns
    -------
    dict
    """
    cfg = {**DEFAULT_CONFIG, **(config or {})}
    unknown = set(cfg) - set(DEFAULT_CONFIG)
    if unknown:
        raise ValueError(f"unknown config keys: {sorted(unknown)}")
    return cfg


def combos_for(config):
    """Location x attribute combinations used by the target encoder for a config.

    Parameters
    ----------
    config : dict
        Resolved configuration.

    Returns
    -------
    tuple
    """
    combos = list(DEFAULT_COMBOS)
    if config["extra_combos"]:
        combos += list(EXTRA_COMBOS)
    if config["kmeans"]:
        combos += list(KMEANS_COMBOS)
    if config["vuln"]:
        combos += list(VULN_COMBOS)
    return tuple(combos)


def te_input_cols(combos):
    """Unique columns needed by the target encoder (chains + combos), in order.

    Parameters
    ----------
    combos : tuple

    Returns
    -------
    list of str
    """
    cols = [col for chain in DEFAULT_CHAINS for col, _ in chain]
    cols += [c for combo_cols, _, _ in combos for c in combo_cols]
    return list(dict.fromkeys(cols))


# --------------------------------------------------------------------------
# paths & IO
# --------------------------------------------------------------------------
def find_data_dir(data_dir=None):
    """Locate the folder containing ``train.csv``, ``test.csv`` and ``sample_submission.csv``.

    Search order: ``data_dir`` argument, ``$SDC_DATA_DIR``, ``<repo>/data``,
    ``/kaggle/input/structural-damage-classification`` and finally any
    folder under ``/kaggle/input`` that holds both ``train.csv`` and
    ``sample_submission.csv``.

    Parameters
    ----------
    data_dir : str or Path, optional
        Explicit data folder.

    Returns
    -------
    pathlib.Path

    Raises
    ------
    FileNotFoundError
        If no candidate folder contains ``train.csv``.
    """
    candidates = []
    if data_dir:
        candidates.append(Path(data_dir))
    if os.environ.get("SDC_DATA_DIR"):
        candidates.append(Path(os.environ["SDC_DATA_DIR"]))
    kaggle_input = Path("/kaggle/input")
    candidates += [ROOT / "data", kaggle_input / COMPETITION,
                   kaggle_input / "competitions" / COMPETITION]
    for cand in candidates:
        if (cand / "train.csv").exists():
            return cand
    if kaggle_input.exists():
        for hit in sorted(kaggle_input.glob("**/train.csv")):
            if (hit.parent / "sample_submission.csv").exists():
                return hit.parent
    raise FileNotFoundError(
        "train.csv not found. Put the competition files in <repo>/data/ "
        "or add the competition data to the Kaggle notebook.")


def output_dir():
    """Return the root folder for outputs and create its sub-folders.

    On Kaggle always ``/kaggle/working`` (outside the cloned repository, so
    re-cloning the repo never deletes results); locally the repository root.
    Creates ``models/``, ``submissions/`` and ``reports/figures/``.

    Returns
    -------
    pathlib.Path
    """
    kaggle = Path("/kaggle/working")
    base = kaggle if kaggle.exists() else ROOT
    for sub in ("models", "submissions", "reports/figures"):
        (base / sub).mkdir(parents=True, exist_ok=True)
    return base


def load_data(data_dir=None):
    """Read the competition CSV files.

    Parameters
    ----------
    data_dir : str or Path, optional
        Explicit data folder, see :func:`find_data_dir`.

    Returns
    -------
    train, test, sample : pandas.DataFrame
    """
    d = find_data_dir(data_dir)
    train = pd.read_csv(d / "train.csv")
    test = pd.read_csv(d / "test.csv")
    sample = pd.read_csv(d / "sample_submission.csv")
    return train, test, sample


def parse_grade(series):
    """Convert labels such as ``"Grade 3"`` (or ``3``) to integers 1..5.

    Parameters
    ----------
    series : pandas.Series

    Returns
    -------
    numpy.ndarray of int
    """
    return series.astype(str).str.extract(r"(\d+)")[0].astype(int).to_numpy()


def split_xy(train):
    """Split the training table into features ``X`` and integer target ``y``.

    Parameters
    ----------
    train : pandas.DataFrame
        Training table including the ``damage_grade`` column.

    Returns
    -------
    X : pandas.DataFrame
    y : numpy.ndarray of int (values 1..5)
    """
    return train.drop(columns=[TARGET]), parse_grade(train[TARGET])


# --------------------------------------------------------------------------
# preprocessing
# --------------------------------------------------------------------------
class CategoricalCoder(TransformerMixin, BaseEstimator):
    """Convert string columns to pandas ``category`` with categories fixed at fit time.

    LightGBM detects ``category`` columns automatically and splits on them
    natively. Fixing the category list in ``fit`` guarantees identical codes
    for train, validation and test; unseen values become NaN (missing).
    """

    def fit(self, X, y=None):
        """Learn the sorted category list of every column.

        Parameters
        ----------
        X : pandas.DataFrame
        y : ignored

        Returns
        -------
        self
        """
        self.categories_ = {c: sorted(X[c].astype(str).unique().tolist()) for c in X.columns}
        self.feature_names_out_ = np.asarray(list(X.columns), dtype=object)
        return self

    def transform(self, X):
        """Return the columns as ``category`` dtype with the learned categories.

        Parameters
        ----------
        X : pandas.DataFrame

        Returns
        -------
        pandas.DataFrame
        """
        return pd.DataFrame(
            {c: pd.Categorical(X[c].astype(str), categories=self.categories_[c])
             for c in self.feature_names_out_},
            index=X.index)

    def get_feature_names_out(self, input_features=None):
        """Return the output column names.

        Parameters
        ----------
        input_features : ignored

        Returns
        -------
        numpy.ndarray of str
        """
        return self.feature_names_out_


def build_preprocessor(config=None, n_splits=5, random_state=42):
    """Build the ``ColumnTransformer`` that turns raw + structural columns into model features.

    Branches (``config`` switches the optional ones)
    ------------------------------------------------
    num : passthrough
        Raw numeric, location and structural-proxy columns (+ ``btype``,
        ``vscore``, ``vtype`` when the clustering steps are enabled).
    cat : :class:`CategoricalCoder`
        Construction categories as pandas ``category``.
    grp : :class:`~scripts.feature_engineering.GroupRelativeFeatures`
        Building compared with its area (unsupervised).
    te : :class:`~scripts.feature_engineering.HierarchicalTargetEncoder`
        Out-of-fold location / location x attribute target encoding.
    nb : :class:`~scripts.feature_engineering.IdNeighborEncoder`
        Out-of-fold labels of the nearest survey numbers (``v1`` or ``v2``).
    hh : :class:`~scripts.feature_engineering.HouseholdEncoder` (optional)
        Out-of-fold grades of the other buildings of the household.
    adj : :class:`~scripts.feature_engineering.AdjacentAreaEncoder` (optional)
        Damage of the neighbouring ward / block codes.
    ri : :class:`~scripts.feature_engineering.ResidualIntensityEncoder` (optional)
        Out-of-fold shaking intensity net of the construction mix (two-stage).

    Parameters
    ----------
    config : dict, optional
        Feature configuration, see :func:`resolve_config`.
    n_splits : int, default=5
        Cross-fitting folds of the supervised encoders.
    random_state : int, default=42
        Seed of the cross-fitting folds.

    Returns
    -------
    sklearn.compose.ColumnTransformer
        Configured to output a pandas DataFrame.
    """
    cfg = resolve_config(config)
    combos = combos_for(cfg)
    passthrough = list(PASSTHROUGH_COLS)
    if cfg["kmeans"]:
        passthrough.append("btype")
    if cfg["vuln"]:
        passthrough += ["vscore", "vtype"]
    transformers = [
        ("num", "passthrough", passthrough),
        ("cat", CategoricalCoder(), CAT_COLS),
        ("grp", GroupRelativeFeatures(), GROUP_INPUT_COLS),
        ("te", HierarchicalTargetEncoder(combos=combos, n_splits=n_splits, random_state=random_state),
         te_input_cols(combos)),
        ("nb", IdNeighborEncoder(n_splits=n_splits, random_state=random_state,
                                 **NEIGHBOR_PARAMS[cfg["neighbor"]]),
         NEIGHBOR_INPUT_COLS),
    ]
    if cfg["household"]:
        transformers.append(("hh", HouseholdEncoder(n_splits=n_splits, random_state=random_state), ["hh"]))
    if cfg["adjacent"]:
        transformers.append(("adj", AdjacentAreaEncoder(), ["id6", "id9"]))
    if cfg["residual"]:
        transformers.append(("ri", ResidualIntensityEncoder(n_splits=n_splits, random_state=random_state),
                             RESIDUAL_INPUT_COLS))
    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        verbose_feature_names_out=False,
    )
    preprocessor.set_output(transform="pandas")
    return preprocessor
