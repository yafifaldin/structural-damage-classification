"""Custom scikit-learn transformers for the structural damage pipeline.

Every transformer follows the scikit-learn API (``fit`` / ``transform`` /
``get_feature_names_out``) so it can be placed inside a ``Pipeline`` or a
``ColumnTransformer``.

The two *supervised* encoders (:class:`HierarchicalTargetEncoder` and
:class:`IdNeighborEncoder`) use **cross-fitting**, the same idea as
``sklearn.preprocessing.TargetEncoder``:

* ``fit_transform(X, y)`` splits the data into ``n_splits`` folds and encodes
  each fold using statistics computed from the *other* folds only
  (out-of-fold), so a row never sees its own label.
* ``transform(X)`` (new data, e.g. validation or test) averages the encodings
  produced by the ``n_splits`` fold-specific tables. Every table was built
  from ``(n_splits - 1) / n_splits`` of the training data, so the encoded
  values of new rows follow the same distribution as the out-of-fold values
  the model was trained on.

:class:`HouseholdEncoder` and :class:`VulnerabilityTyper` use the same
cross-fitting scheme. :class:`AdjacentAreaEncoder` needs no cross-fitting
because it only reads labels of *other* areas, never the row's own area.

This module only depends on numpy, pandas and scikit-learn.
"""
import warnings

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import MiniBatchKMeans
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, OrdinalEncoder, StandardScaler

# Prefix lengths of ``building_id`` that encode the (pre-2017) administrative
# hierarchy: 2 = district, 4 = old VDC, 6 = old ward, longer = finer blocks.
ID_PREFIX_LENGTHS = (2, 4, 6, 8, 9, 10)

# EMS-98-inspired vulnerability rank of the main superstructure material
# (0 = most vulnerable, 5 = least vulnerable).
VULNERABILITY_RANK = {
    "has_superstructure_adobe_mud": 0,
    "has_superstructure_mud_mortar_stone": 0,
    "has_superstructure_stone_flag": 1,
    "has_superstructure_mud_mortar_brick": 1,
    "has_superstructure_bamboo": 2,
    "has_superstructure_timber": 2,
    "has_superstructure_other": 2,
    "has_superstructure_cement_mortar_stone": 3,
    "has_superstructure_cement_mortar_brick": 3,
    "has_superstructure_rc_non_engineered": 4,
    "has_superstructure_rc_engineered": 5,
}

STRUCTURAL_FEATURES = (
    ["bid"] + [f"id{k}" for k in ID_PREFIX_LENGTHS]
    + ["id_tail", "id_last", "h_per_floor", "period_T", "slender",
       "total_area", "log_area", "age_x_floors", "vuln", "age_bin", "hh", "is_secondary"]
)
# Helper keys created by StructuralFeatures that are only used as grouping keys
# (not passed to the model directly, they duplicate information of other columns).
HELPER_COLS = ("age_bin", "hh")
AGE_BIN_EDGES = (5, 10, 15, 20, 30, 50, 100)

# Construction columns used by the building-type models.
CONSTRUCTION_CAT_COLS = ("superstructure", "foundation_type", "roof_type", "ground_floor_type",
                         "other_floor_type")
CONSTRUCTION_NUM_COLS = ("age_building", "count_floors_pre_eq", "plinth_area_sq_ft", "height_ft_pre_eq")

# Location hierarchies for the target encoder: (column, smoothing strength m).
DEFAULT_CHAINS = (
    (("id2", 50), ("id4", 30), ("id6", 20), ("id8", 15), ("id9", 10), ("id10", 5)),
    (("district_id", 50), ("vdcmun_id", 30), ("ward_id", 20)),
)
# Location x construction combinations: (columns, parent level used as prior, m).
DEFAULT_COMBOS = (
    (("id6", "superstructure"), "id6", 20),
    (("id4", "superstructure"), "id4", 20),
    (("vdcmun_id", "superstructure"), "vdcmun_id", 20),
    (("id6", "foundation_type"), "id6", 20),
    (("id6", "roof_type"), "id6", 20),
    (("id6", "count_floors_pre_eq"), "id6", 20),
    (("id4", "superstructure", "count_floors_pre_eq"), "id4", 20),
    (("id6", "ground_floor_type"), "id6", 20),
)
# Iteration 3 (from EDA section F): construction features still informative after location.
EXTRA_COMBOS = (
    (("id6", "other_floor_type"), "id6", 20),
    (("id6", "age_bin"), "id6", 20),
    (("id9", "superstructure"), "id9", 10),
)
# Location x building type (clustering experiments).
KMEANS_COMBOS = (
    (("id4", "btype"), "id4", 30),
    (("id6", "btype"), "id6", 20),
    (("id9", "btype"), "id9", 10),
)
VULN_COMBOS = (
    (("id4", "vtype"), "id4", 30),
    (("id6", "vtype"), "id6", 20),
    (("id9", "vtype"), "id9", 10),
)


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def make_key(X, cols):
    """Build a pandas (Multi)Index from one or more columns of ``X``.

    Parameters
    ----------
    X : pandas.DataFrame
        Input data.
    cols : sequence of str
        Column names forming the grouping key.

    Returns
    -------
    pandas.Index or pandas.MultiIndex
        One entry per row of ``X``; used both to aggregate (fit) and to look
        up (transform), which guarantees identical key dtypes in both steps.
    """
    cols = list(cols)
    if len(cols) == 1:
        return pd.Index(X[cols[0]].to_numpy())
    return pd.MultiIndex.from_arrays([X[c].to_numpy() for c in cols])


def group_sum(values, key):
    """Sum the rows of ``values`` per unique entry of ``key``.

    Parameters
    ----------
    values : numpy.ndarray of shape (n_rows, n_cols)
        Values to aggregate.
    key : pandas.Index or pandas.MultiIndex
        Grouping key with ``n_rows`` entries (see :func:`make_key`).

    Returns
    -------
    pandas.DataFrame
        Indexed by the unique keys, one column per column of ``values``.
    """
    level = 0 if key.nlevels == 1 else list(range(key.nlevels))
    return pd.DataFrame(values, index=key).groupby(level=level).sum()


def encode_labels(y):
    """Map arbitrary class labels to integer indices ``0..K-1``.

    Parameters
    ----------
    y : array-like of shape (n_rows,)
        Class labels.

    Returns
    -------
    classes : numpy.ndarray
        Sorted unique labels.
    y_idx : numpy.ndarray of int
        Index of each label inside ``classes``.
    """
    if y is None:
        raise ValueError("This encoder is supervised: y must be provided.")
    y = np.asarray(y)
    classes = np.unique(y)
    return classes, np.searchsorted(classes, y)


# --------------------------------------------------------------------------
# 1. stateless structural / physics-proxy features
# --------------------------------------------------------------------------
class StructuralFeatures(TransformerMixin, BaseEstimator):
    """Add ID-based location codes and structural-dynamics proxy features.

    New columns
    -----------
    bid, id2..id10, id_tail, id_last
        ``building_id`` as integer and its prefixes (old administrative units).
    h_per_floor
        Storey height (ft).
    period_T
        Approximate fundamental period ``0.075 * H**0.75`` (H in metres).
    slender
        Slenderness ``height / sqrt(plinth_area)``.
    total_area, log_area, age_x_floors
        Size and ageing proxies.
    vuln
        Vulnerability rank of the superstructure material (EMS-98 inspired).
    age_bin
        Age group (edges 5, 10, 15, 20, 30, 50, 100 years), used as grouping key.
    hh
        Household number (``building_id // 10``); the last digit numbers the
        buildings inside a household (EDA section E).
    is_secondary
        1 when the building is not the first one of its household.

    Parameters
    ----------
    id_col : str, default="building_id"
        Name of the building identifier column.
    material_col : str, default="superstructure"
        Name of the superstructure material column.
    """

    def __init__(self, id_col="building_id", material_col="superstructure"):
        """Store parameters (scikit-learn convention: no logic here)."""
        self.id_col = id_col
        self.material_col = material_col

    def fit(self, X, y=None):
        """Record input/output column names; the transformer is stateless.

        Parameters
        ----------
        X : pandas.DataFrame
            Raw building table.
        y : ignored

        Returns
        -------
        self
        """
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        extra = [c for c in STRUCTURAL_FEATURES if c not in X.columns]
        self.feature_names_out_ = np.asarray(list(X.columns) + extra, dtype=object)
        return self

    def transform(self, X):
        """Return ``X`` with the structural features appended.

        Parameters
        ----------
        X : pandas.DataFrame
            Raw building table with the same columns seen in ``fit``.

        Returns
        -------
        pandas.DataFrame
        """
        X = X.copy()
        ids = X[self.id_col].astype(np.int64)
        sid = ids.astype(str).str.zfill(12)
        X["bid"] = ids
        for k in ID_PREFIX_LENGTHS:
            X[f"id{k}"] = sid.str[:k].astype(np.int64)
        X["id_tail"] = sid.str[6:].astype(np.int64)
        X["id_last"] = sid.str[-1].astype(np.int64)

        floors = X["count_floors_pre_eq"].clip(lower=1)
        height_m = X["height_ft_pre_eq"].clip(lower=1) * 0.3048
        X["h_per_floor"] = X["height_ft_pre_eq"] / floors
        X["period_T"] = 0.075 * height_m ** 0.75
        X["slender"] = X["height_ft_pre_eq"] / np.sqrt(X["plinth_area_sq_ft"].clip(lower=1))
        X["total_area"] = X["plinth_area_sq_ft"] * floors
        X["log_area"] = np.log1p(X["plinth_area_sq_ft"])
        X["age_x_floors"] = X["age_building"] * floors
        X["vuln"] = X[self.material_col].map(VULNERABILITY_RANK).fillna(2).astype(float)
        X["age_bin"] = np.searchsorted(np.asarray(AGE_BIN_EDGES), X["age_building"].to_numpy(), side="right")
        # last digit = building number inside a household; the first 11 digits identify the household
        X["hh"] = ids // 10
        X["is_secondary"] = (X["id_last"] > 1).astype(np.int64)
        return X[list(self.feature_names_out_)]

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


# --------------------------------------------------------------------------
# 2. unsupervised group statistics
# --------------------------------------------------------------------------
class GroupRelativeFeatures(TransformerMixin, BaseEstimator):
    """Compare each building with the other buildings of its area (no labels).

    For every ``group_col`` it adds the group size and, for every
    ``value_col``, the difference between the building and the group mean.
    For ``mean_groups`` the group mean itself is also returned. For every
    ``(a, b)`` in ``share_pairs`` it adds the share of buildings in group
    ``a`` that have the same value of ``b`` (e.g. how common this material is
    in the ward).

    Parameters
    ----------
    group_cols : tuple of str
        Area columns to group by.
    value_cols : tuple of str
        Numeric columns to compare against the group mean.
    mean_groups : tuple of str
        Subset of ``group_cols`` for which the raw group mean is returned.
    share_pairs : tuple of (str, str)
        ``(group column, attribute column)`` pairs for share features.
    """

    def __init__(self, group_cols=("id6", "id9", "ward_id"),
                 value_cols=("age_building", "height_ft_pre_eq", "count_floors_pre_eq",
                             "plinth_area_sq_ft", "vuln"),
                 mean_groups=("id6",), share_pairs=(("id6", "superstructure"),)):
        """Store parameters (scikit-learn convention: no logic here)."""
        self.group_cols = group_cols
        self.value_cols = value_cols
        self.mean_groups = mean_groups
        self.share_pairs = share_pairs

    def fit(self, X, y=None):
        """Compute per-group means, counts and attribute shares.

        Parameters
        ----------
        X : pandas.DataFrame
            Must contain ``group_cols``, ``value_cols`` and the share columns.
        y : ignored

        Returns
        -------
        self
        """
        values = X[list(self.value_cols)].to_numpy(dtype=float)
        self.stats_ = {}
        for g in self.group_cols:
            key = make_key(X, [g])
            sums = group_sum(np.column_stack([values, np.ones(len(X))]), key)
            cnt = sums.iloc[:, -1]
            means = sums.iloc[:, :-1].div(cnt, axis=0)
            means.columns = list(self.value_cols)
            means["__cnt"] = cnt
            self.stats_[g] = means
        self.shares_ = {}
        for a, b in self.share_pairs:
            pair = group_sum(np.ones((len(X), 1)), make_key(X, [a, b])).iloc[:, 0]
            total = group_sum(np.ones((len(X), 1)), make_key(X, [a])).iloc[:, 0]
            self.shares_[(a, b)] = pair / total.reindex(pair.index.get_level_values(0)).to_numpy()

        names = []
        for g in self.group_cols:
            names.append(f"cnt_{g}")
            for v in self.value_cols:
                names.append(f"{v}_rel_{g}")
                if g in self.mean_groups:
                    names.append(f"{v}_mean_{g}")
        names += [f"share_{a}_{b}" for a, b in self.share_pairs]
        self.feature_names_out_ = np.asarray(names, dtype=object)
        return self

    def transform(self, X):
        """Map the stored group statistics onto ``X``.

        Unseen groups get count 0 and NaN means (LightGBM handles NaN).

        Parameters
        ----------
        X : pandas.DataFrame

        Returns
        -------
        pandas.DataFrame
        """
        out = {}
        for g in self.group_cols:
            st = self.stats_[g].reindex(make_key(X, [g]))
            out[f"cnt_{g}"] = st["__cnt"].fillna(0).to_numpy()
            for v in self.value_cols:
                mean = st[v].to_numpy(dtype=float)
                out[f"{v}_rel_{g}"] = X[v].to_numpy(dtype=float) - mean
                if g in self.mean_groups:
                    out[f"{v}_mean_{g}"] = mean
        for a, b in self.share_pairs:
            share = self.shares_[(a, b)].reindex(make_key(X, [a, b]))
            out[f"share_{a}_{b}"] = share.fillna(0).to_numpy(dtype=float)
        return pd.DataFrame(out, index=X.index)[list(self.feature_names_out_)]

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


# --------------------------------------------------------------------------
# 3. hierarchical out-of-fold target encoding
# --------------------------------------------------------------------------
class HierarchicalTargetEncoder(TransformerMixin, BaseEstimator):
    """Smoothed multiclass target encoding along a location hierarchy.

    For each level of a chain (e.g. district -> VDC -> ward -> block) the
    class distribution of the level is shrunk towards the encoding of its
    parent level::

        enc(level) = (class_counts + m * enc(parent)) / (n + m)

    The first level is shrunk towards the global class distribution.
    Combination keys (location x construction attribute) are shrunk towards
    the encoding of their parent location level.

    Output per chain level: ``p1..pK`` (class probabilities), ``mean``
    (expected grade) and ``n`` (number of training rows in the group).
    Output per combination: ``mean``, selected class probabilities and ``n``.

    Uses cross-fitting, see the module docstring.

    Parameters
    ----------
    chains : tuple
        Tuple of chains; each chain is a tuple of ``(column, m)``.
    combos : tuple
        Tuple of ``(columns, parent_column, m)``.
    combo_classes : tuple of int
        Class indices whose probability is returned for combinations.
    n_splits : int, default=5
        Number of cross-fitting folds.
    random_state : int, default=42
        Seed of the fold split.
    """

    def __init__(self, chains=DEFAULT_CHAINS, combos=DEFAULT_COMBOS, combo_classes=(0, 4),
                 n_splits=5, random_state=42):
        """Store parameters (scikit-learn convention: no logic here)."""
        self.chains = chains
        self.combos = combos
        self.combo_classes = combo_classes
        self.n_splits = n_splits
        self.random_state = random_state

    # ----- internals -----
    def _key_specs(self):
        """List every grouping key (tuple of columns) that needs a table."""
        specs = [(col,) for chain in self.chains for col, _ in chain]
        specs += [tuple(cols) for cols, _, _ in self.combos]
        return specs

    def _build_names(self):
        """Build output feature names in the same order as :meth:`_encode`."""
        k = self.n_classes_
        names = []
        for chain in self.chains:
            for col, _ in chain:
                names += [f"te_{col}_p{j + 1}" for j in range(k)]
                names += [f"te_{col}_mean", f"te_{col}_n"]
        for cols, _, _ in self.combos:
            nm = "c_" + "_".join(cols)
            names += [f"te_{nm}_mean"] + [f"te_{nm}_p{j + 1}" for j in self.combo_classes]
            names += [f"te_{nm}_n"]
        return names

    def _fit_tables(self, X, Y):
        """Aggregate one-hot labels ``Y`` per grouping key.

        Parameters
        ----------
        X : pandas.DataFrame
            Rows whose labels are known.
        Y : numpy.ndarray of shape (n_rows, K)
            One-hot labels.

        Returns
        -------
        dict
            ``{key_spec: DataFrame of class sums + count, "__global__": prior}``
        """
        vals = np.column_stack([Y, np.ones(len(Y))])
        tables = {"__global__": Y.mean(axis=0)}
        for spec in self._key_specs():
            tables[spec] = group_sum(vals, make_key(X, spec))
        return tables

    def _lookup(self, X, spec, tables, prior, m):
        """Smoothed class probabilities and counts of ``X`` for one key."""
        k = self.n_classes_
        tab = tables[spec].reindex(make_key(X, spec)).to_numpy(dtype=float)
        tab = np.nan_to_num(tab, nan=0.0)
        sums, cnt = tab[:, :k], tab[:, k]
        return (sums + m * prior) / (cnt[:, None] + m), cnt

    def _encode(self, X, tables):
        """Encode ``X`` with one set of tables; returns a float32 array."""
        n, k = len(X), self.n_classes_
        grade = np.arange(1, k + 1, dtype=float)
        glob = np.broadcast_to(tables["__global__"], (n, k))
        blocks, enc_by_level = [], {}
        for chain in self.chains:
            prior = glob
            for col, m in chain:
                enc, cnt = self._lookup(X, (col,), tables, prior, m)
                enc_by_level[col] = enc
                blocks += [enc, (enc @ grade)[:, None], cnt[:, None]]
                prior = enc
        for cols, parent, m in self.combos:
            enc, cnt = self._lookup(X, tuple(cols), tables, enc_by_level[parent], m)
            blocks += [(enc @ grade)[:, None], enc[:, list(self.combo_classes)], cnt[:, None]]
        return np.hstack(blocks).astype(np.float32)

    def _cross_fit(self, X, y):
        """Build one table set per fold; return the fold indices."""
        self.classes_, y_idx = encode_labels(y)
        self.n_classes_ = len(self.classes_)
        Y = np.eye(self.n_classes_)[y_idx]
        skf = StratifiedKFold(self.n_splits, shuffle=True, random_state=self.random_state)
        folds = list(skf.split(np.zeros(len(y_idx)), y_idx))
        self.tables_ = [self._fit_tables(X.iloc[tr], Y[tr]) for tr, _ in folds]
        self.feature_names_out_ = np.asarray(self._build_names(), dtype=object)
        return folds

    # ----- public API -----
    def fit(self, X, y):
        """Fit the fold-specific encoding tables.

        Parameters
        ----------
        X : pandas.DataFrame
            Must contain every column used by ``chains`` and ``combos``.
        y : array-like of shape (n_rows,)
            Damage grade labels.

        Returns
        -------
        self
        """
        self._cross_fit(X, y)
        return self

    def fit_transform(self, X, y=None, **fit_params):
        """Fit and return the **out-of-fold** encoding of the training rows.

        Parameters
        ----------
        X : pandas.DataFrame
        y : array-like of shape (n_rows,)
        **fit_params : ignored

        Returns
        -------
        pandas.DataFrame
        """
        folds = self._cross_fit(X, y)
        out = np.empty((len(X), len(self.feature_names_out_)), dtype=np.float32)
        for (_, va), tables in zip(folds, self.tables_):
            out[va] = self._encode(X.iloc[va], tables)
        return pd.DataFrame(out, index=X.index, columns=self.feature_names_out_)

    def transform(self, X):
        """Encode new rows as the average over the fold-specific tables.

        Parameters
        ----------
        X : pandas.DataFrame
            Rows **not** used in ``fit`` (validation / test).

        Returns
        -------
        pandas.DataFrame
        """
        out = np.zeros((len(X), len(self.feature_names_out_)), dtype=np.float64)
        for tables in self.tables_:
            out += self._encode(X, tables)
        out /= len(self.tables_)
        return pd.DataFrame(out.astype(np.float32), index=X.index, columns=self.feature_names_out_)

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


# --------------------------------------------------------------------------
# 4. ID-neighbour label features
# --------------------------------------------------------------------------
class IdNeighborEncoder(TransformerMixin, BaseEstimator):
    """Label statistics of the buildings with the closest survey number.

    Survey IDs are assigned sequentially inside an (old) ward, so buildings
    with close IDs tend to be physically close and to have suffered similar
    ground shaking.

    Iteration 3 (EDA section E): the last digit of ``building_id`` numbers
    the buildings inside a household and households are numbered in steps of
    10. Buildings of the *same* household are *less* similar than random
    buildings of the ward, while adjacent households are the most similar.
    Therefore distances are measured on the household number
    (``id // household_div``) and, with ``exclude_household=True``, buildings
    of the query's own household are skipped (they get their own features in
    :class:`HouseholdEncoder`).

    For every ``k`` in ``ks`` the encoder returns: number of neighbours
    found, their mean grade, their class shares and the log mean distance.
    It also returns a distance-weighted mean grade of the ``weighted_k``
    nearest neighbours and the log distance to the nearest one.

    ``household_div=1, exclude_household=False, ks=(3, 10, 30)`` reproduces
    the iteration-2 behaviour.

    Uses cross-fitting, see the module docstring.

    Parameters
    ----------
    id_col : str, default="bid"
        Integer building ID column.
    group_col : str, default="id6"
        Neighbours must share this area code.
    ks : tuple of int, default=(1, 2, 5, 10, 30)
        Neighbourhood sizes.
    window : int, default=40
        Candidates searched on each side of the sorted position
        (``max(ks)`` and ``weighted_k`` must be <= ``2 * window``).
    weighted_k : int, default=10
        Neighbourhood size for the distance-weighted mean.
    household_div : int, default=10
        Distances are computed on ``id // household_div``.
    exclude_household : bool, default=True
        Skip neighbours with the same ``id // household_div`` as the query.
    n_splits : int, default=5
        Number of cross-fitting folds.
    random_state : int, default=42
        Seed of the fold split.
    chunk_size : int, default=100_000
        Rows processed at once (limits memory use).
    """

    def __init__(self, id_col="bid", group_col="id6", ks=(1, 2, 5, 10, 30), window=40, weighted_k=10,
                 household_div=10, exclude_household=True, n_splits=5, random_state=42,
                 chunk_size=100_000):
        """Store parameters (scikit-learn convention: no logic here)."""
        self.id_col = id_col
        self.group_col = group_col
        self.ks = ks
        self.window = window
        self.weighted_k = weighted_k
        self.household_div = household_div
        self.exclude_household = exclude_household
        self.n_splits = n_splits
        self.random_state = random_state
        self.chunk_size = chunk_size

    def _build_names(self):
        """Build output feature names in the same order as :meth:`_encode_chunk`."""
        names = []
        for k in self.ks:
            names += [f"nb{k}_n", f"nb{k}_mean"]
            names += [f"nb{k}_p{j + 1}" for j in range(self.n_classes_)]
            names += [f"nb{k}_logdist"]
        return names + ["nbw_mean", "nb1_dist"]

    def _arrays(self, X):
        """Extract integer id and group arrays from ``X``."""
        return (X[self.id_col].to_numpy(dtype=np.int64),
                X[self.group_col].to_numpy(dtype=np.int64))

    @staticmethod
    def _make_store(ids, groups, y_idx):
        """Sort labelled rows by id so neighbours can be found by binary search."""
        order = np.argsort(ids, kind="stable")
        return ids[order], groups[order], y_idx[order]

    def _encode_chunk(self, ids, groups, store):
        """Compute neighbour features for one chunk of query rows."""
        kb, kg, ky = store
        n_known, n_cls, w, div = len(kb), self.n_classes_, self.window, self.household_div
        pos = np.searchsorted(kb, ids)
        cand = pos[:, None] + np.arange(-w, w)[None, :]
        valid = (cand >= 0) & (cand < n_known)
        cc = np.clip(cand, 0, n_known - 1)
        q_serial = ids // div
        c_serial = kb[cc] // div
        dist = np.abs(c_serial - q_serial[:, None]).astype(np.float64)
        bad = ~valid | (kg[cc] != groups[:, None])
        if self.exclude_household:
            bad |= c_serial == q_serial[:, None]
        dist[bad] = np.inf
        order = np.argsort(dist, axis=1, kind="stable")
        cand_y = ky[cc]
        cols = []
        for k in self.ks:
            sel = order[:, :k]
            d = np.take_along_axis(dist, sel, axis=1)
            yy = np.take_along_axis(cand_y, sel, axis=1)
            ok = np.isfinite(d)
            cnt = ok.sum(axis=1)
            den = np.maximum(cnt, 1)
            has = cnt > 0
            cols.append(cnt.astype(float))
            cols.append(np.where(has, np.where(ok, yy + 1, 0).sum(axis=1) / den, np.nan))
            for j in range(n_cls):
                cols.append(np.where(has, ((yy == j) & ok).sum(axis=1) / den, np.nan))
            cols.append(np.log1p(np.nanmean(np.where(ok, d, np.nan), axis=1)))
        sel = order[:, :self.weighted_k]
        d = np.take_along_axis(dist, sel, axis=1)
        yy = np.take_along_axis(cand_y, sel, axis=1) + 1.0
        wt = np.where(np.isfinite(d), 1.0 / (1.0 + d), 0.0)
        wsum = wt.sum(axis=1)
        cols.append(np.where(wsum > 0, (wt * yy).sum(axis=1) / np.maximum(wsum, 1e-12), np.nan))
        cols.append(np.log1p(np.where(np.isfinite(d[:, 0]), d[:, 0], np.nan)))
        return np.column_stack(cols)

    def _encode(self, ids, groups, store):
        """Compute neighbour features for all query rows (chunked)."""
        parts = []
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            for s in range(0, len(ids), self.chunk_size):
                e = s + self.chunk_size
                parts.append(self._encode_chunk(ids[s:e], groups[s:e], store))
        if not parts:
            return np.empty((0, len(self.feature_names_out_)))
        return np.vstack(parts)

    def _cross_fit(self, X, y):
        """Build one sorted neighbour store per fold; return the fold indices."""
        if max(self.ks) > 2 * self.window or self.weighted_k > 2 * self.window:
            raise ValueError("max(ks) and weighted_k must be <= 2 * window")
        self.classes_, y_idx = encode_labels(y)
        self.n_classes_ = len(self.classes_)
        ids, groups = self._arrays(X)
        skf = StratifiedKFold(self.n_splits, shuffle=True, random_state=self.random_state)
        folds = list(skf.split(np.zeros(len(y_idx)), y_idx))
        self.stores_ = [self._make_store(ids[tr], groups[tr], y_idx[tr]) for tr, _ in folds]
        self.feature_names_out_ = np.asarray(self._build_names(), dtype=object)
        return folds, ids, groups

    def fit(self, X, y):
        """Fit the fold-specific neighbour stores.

        Parameters
        ----------
        X : pandas.DataFrame
            Must contain ``id_col`` and ``group_col``.
        y : array-like of shape (n_rows,)
            Damage grade labels.

        Returns
        -------
        self
        """
        self._cross_fit(X, y)
        return self

    def fit_transform(self, X, y=None, **fit_params):
        """Fit and return the **out-of-fold** neighbour features.

        Parameters
        ----------
        X : pandas.DataFrame
        y : array-like of shape (n_rows,)
        **fit_params : ignored

        Returns
        -------
        pandas.DataFrame
        """
        folds, ids, groups = self._cross_fit(X, y)
        out = np.empty((len(X), len(self.feature_names_out_)), dtype=np.float32)
        for (_, va), store in zip(folds, self.stores_):
            out[va] = self._encode(ids[va], groups[va], store)
        return pd.DataFrame(out, index=X.index, columns=self.feature_names_out_)

    def transform(self, X):
        """Neighbour features of new rows, averaged over the fold stores.

        NaN values (no neighbour found in a store) are ignored in the average.

        Parameters
        ----------
        X : pandas.DataFrame
            Rows **not** used in ``fit`` (validation / test).

        Returns
        -------
        pandas.DataFrame
        """
        ids, groups = self._arrays(X)
        out = nan_average([self._encode(ids, groups, store) for store in self.stores_])
        return pd.DataFrame(out.astype(np.float32), index=X.index, columns=self.feature_names_out_)

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


def nan_average(arrays):
    """Element-wise mean of equally shaped arrays, ignoring NaN.

    Parameters
    ----------
    arrays : list of numpy.ndarray

    Returns
    -------
    numpy.ndarray
        NaN where every input is NaN.
    """
    total = np.zeros_like(arrays[0], dtype=np.float64)
    count = np.zeros_like(total)
    for a in arrays:
        fin = np.isfinite(a)
        total += np.where(fin, a, 0.0)
        count += fin
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(count > 0, total / np.maximum(count, 1), np.nan)


# --------------------------------------------------------------------------
# 5. household (same first 11 digits) label features
# --------------------------------------------------------------------------
class HouseholdEncoder(TransformerMixin, BaseEstimator):
    """Grades of the *other* labelled buildings of the same household.

    EDA section E: about 8.6 % of the buildings share their household
    (``building_id // 10``) with another building. Buildings of one household
    are not simply similar (secondary buildings are on average 0.22 grade
    lighter than their neighbours), so the raw statistics are given to the
    model and the trees decide how to use them.

    Output: ``sib_n`` (number of labelled siblings), ``sib_mean``,
    ``sib_min`` and ``sib_max`` (NaN when there is no sibling).

    Uses cross-fitting, see the module docstring.

    Parameters
    ----------
    hh_col : str, default="hh"
        Household key column.
    n_splits : int, default=5
        Number of cross-fitting folds.
    random_state : int, default=42
        Seed of the fold split.
    """

    def __init__(self, hh_col="hh", n_splits=5, random_state=42):
        """Store parameters (scikit-learn convention: no logic here)."""
        self.hh_col = hh_col
        self.n_splits = n_splits
        self.random_state = random_state

    def _fit_table(self, keys, grades):
        """Aggregate count / mean / min / max grade per household."""
        return pd.DataFrame({"g": grades}, index=pd.Index(keys)).groupby(level=0)["g"].agg(
            ["size", "mean", "min", "max"])

    def _encode(self, keys, table):
        """Look up the household statistics of ``keys``."""
        tab = table.reindex(pd.Index(keys))
        n = tab["size"].fillna(0).to_numpy(dtype=float)
        return np.column_stack([n, tab["mean"].to_numpy(dtype=float),
                                tab["min"].to_numpy(dtype=float), tab["max"].to_numpy(dtype=float)])

    def _cross_fit(self, X, y):
        """Build one household table per fold; return the fold indices."""
        self.classes_, y_idx = encode_labels(y)
        grades = (y_idx + 1).astype(float)
        keys = X[self.hh_col].to_numpy(dtype=np.int64)
        skf = StratifiedKFold(self.n_splits, shuffle=True, random_state=self.random_state)
        folds = list(skf.split(np.zeros(len(y_idx)), y_idx))
        self.tables_ = [self._fit_table(keys[tr], grades[tr]) for tr, _ in folds]
        self.feature_names_out_ = np.asarray(["sib_n", "sib_mean", "sib_min", "sib_max"], dtype=object)
        return folds, keys

    def fit(self, X, y):
        """Fit the fold-specific household tables.

        Parameters
        ----------
        X : pandas.DataFrame
            Must contain ``hh_col``.
        y : array-like of shape (n_rows,)

        Returns
        -------
        self
        """
        self._cross_fit(X, y)
        return self

    def fit_transform(self, X, y=None, **fit_params):
        """Fit and return the **out-of-fold** household features.

        Parameters
        ----------
        X : pandas.DataFrame
        y : array-like of shape (n_rows,)
        **fit_params : ignored

        Returns
        -------
        pandas.DataFrame
        """
        folds, keys = self._cross_fit(X, y)
        out = np.empty((len(X), 4), dtype=np.float32)
        for (_, va), table in zip(folds, self.tables_):
            out[va] = self._encode(keys[va], table)
        return pd.DataFrame(out, index=X.index, columns=self.feature_names_out_)

    def transform(self, X):
        """Household features of new rows, averaged over the fold tables.

        Parameters
        ----------
        X : pandas.DataFrame

        Returns
        -------
        pandas.DataFrame
        """
        keys = X[self.hh_col].to_numpy(dtype=np.int64)
        out = nan_average([self._encode(keys, t) for t in self.tables_])
        return pd.DataFrame(out.astype(np.float32), index=X.index, columns=self.feature_names_out_)

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


# --------------------------------------------------------------------------
# 6. neighbouring area codes
# --------------------------------------------------------------------------
class AdjacentAreaEncoder(TransformerMixin, BaseEstimator):
    """Damage distribution of the previous and next area code inside the same parent area.

    EDA section E: consecutive ward codes (``id6``) inside one VDC have more
    similar mean damage than random wards of that VDC, i.e. the codes are
    roughly ordered geographically. For every level the encoder pools the
    labels of the neighbouring codes (the closest smaller and larger code
    sharing the same parent ``code // parent_div``) and returns the smoothed
    class shares, the expected grade and the number of pooled rows.

    No cross-fitting is needed: a row's own area is never part of the pool,
    so its own label is never used.

    Parameters
    ----------
    levels : tuple of (str, int), default=(("id6", 100), ("id9", 1000))
        ``(column, parent_div)``; the parent of a code is ``code // parent_div``.
    m : float, default=20
        Smoothing strength towards the global class distribution.
    """

    def __init__(self, levels=(("id6", 100), ("id9", 1000)), m=20):
        """Store parameters (scikit-learn convention: no logic here)."""
        self.levels = levels
        self.m = m

    def fit(self, X, y):
        """Store the per-code class counts of every level.

        Parameters
        ----------
        X : pandas.DataFrame
            Must contain the level columns.
        y : array-like of shape (n_rows,)

        Returns
        -------
        self
        """
        self.classes_, y_idx = encode_labels(y)
        k = len(self.classes_)
        Y = np.eye(k)[y_idx]
        self.global_ = Y.mean(axis=0)
        self.codes_, self.sums_, self.counts_ = {}, {}, {}
        for col, _ in self.levels:
            tab = group_sum(np.column_stack([Y, np.ones(len(Y))]), make_key(X, [col])).sort_index()
            self.codes_[col] = tab.index.to_numpy(dtype=np.int64)
            self.sums_[col] = tab.iloc[:, :k].to_numpy(dtype=float)
            self.counts_[col] = tab.iloc[:, k].to_numpy(dtype=float)
        names = []
        for col, _ in self.levels:
            names += [f"adj_{col}_mean"] + [f"adj_{col}_p{j + 1}" for j in range(k)] + [f"adj_{col}_n"]
        self.feature_names_out_ = np.asarray(names, dtype=object)
        return self

    def transform(self, X):
        """Encode the neighbouring codes of every row.

        Parameters
        ----------
        X : pandas.DataFrame

        Returns
        -------
        pandas.DataFrame
        """
        k = len(self.classes_)
        grade = np.arange(1, k + 1, dtype=float)
        blocks = []
        for col, div in self.levels:
            codes, sums, counts = self.codes_[col], self.sums_[col], self.counts_[col]
            q = X[col].to_numpy(dtype=np.int64)
            left = np.searchsorted(codes, q, side="left") - 1
            right = np.searchsorted(codes, q, side="right")
            ok_l = (left >= 0) & (codes[np.clip(left, 0, len(codes) - 1)] // div == q // div)
            ok_r = (right < len(codes)) & (codes[np.clip(right, 0, len(codes) - 1)] // div == q // div)
            li, ri = np.clip(left, 0, len(codes) - 1), np.clip(right, 0, len(codes) - 1)
            s = sums[li] * ok_l[:, None] + sums[ri] * ok_r[:, None]
            n = counts[li] * ok_l + counts[ri] * ok_r
            enc = (s + self.m * self.global_) / (n[:, None] + self.m)
            blocks += [(enc @ grade)[:, None], enc, n[:, None]]
        out = np.hstack(blocks).astype(np.float32)
        return pd.DataFrame(out, index=X.index, columns=self.feature_names_out_)

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


# --------------------------------------------------------------------------
# 7. building types by clustering (experiments)
# --------------------------------------------------------------------------
class BuildingTypeClusterer(TransformerMixin, BaseEstimator):
    """Unsupervised building archetypes with K-Means (adds column ``btype``).

    Construction categories are one-hot encoded, numeric columns are
    ``log1p``-transformed and standardised, then ``MiniBatchKMeans`` groups
    the buildings into ``n_clusters`` types. The type is used as a key for
    location x building-type target encoding (``KMEANS_COMBOS``), which
    compresses six construction columns into one key with large enough
    groups.

    Pipeline step placed *before* the ``ColumnTransformer``; returns the
    input frame with ``btype`` appended.

    Parameters
    ----------
    n_clusters : int, default=16
    cat_cols : tuple of str
        Construction categories.
    num_cols : tuple of str
        Construction numeric columns.
    random_state : int, default=42
    """

    def __init__(self, n_clusters=16, cat_cols=CONSTRUCTION_CAT_COLS, num_cols=CONSTRUCTION_NUM_COLS,
                 random_state=42):
        """Store parameters (scikit-learn convention: no logic here)."""
        self.n_clusters = n_clusters
        self.cat_cols = cat_cols
        self.num_cols = num_cols
        self.random_state = random_state

    def fit(self, X, y=None):
        """Fit the encoding + K-Means pipeline.

        Parameters
        ----------
        X : pandas.DataFrame
        y : ignored

        Returns
        -------
        self
        """
        prep = ColumnTransformer([
            ("cat", OneHotEncoder(handle_unknown="ignore"), list(self.cat_cols)),
            ("num", Pipeline([("log", FunctionTransformer(np.log1p)), ("scale", StandardScaler())]),
             list(self.num_cols)),
        ])
        self.model_ = Pipeline([
            ("prep", prep),
            ("kmeans", MiniBatchKMeans(n_clusters=self.n_clusters, random_state=self.random_state,
                                       n_init=3, batch_size=4096)),
        ]).fit(X[list(self.cat_cols) + list(self.num_cols)])
        self.feature_names_out_ = np.asarray(list(X.columns) + ["btype"], dtype=object)
        return self

    def transform(self, X):
        """Append the cluster label ``btype``.

        Parameters
        ----------
        X : pandas.DataFrame

        Returns
        -------
        pandas.DataFrame
        """
        X = X.copy()
        X["btype"] = self.model_.predict(X[list(self.cat_cols) + list(self.num_cols)]).astype(np.int64)
        return X

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


class VulnerabilityTyper(TransformerMixin, BaseEstimator):
    """Supervised building types from a construction-only vulnerability score.

    A small gradient-boosting regressor learns, from construction features
    only, how much worse/better a building does than the other buildings of
    its area (target = grade - leave-one-out mean grade of ``location_col``).
    The prediction ``vscore`` is a location-free vulnerability score; its
    quantile bin ``vtype`` (``n_bins`` types) is used as a key for location x
    building-type target encoding (``VULN_COMBOS``).

    Uses cross-fitting: in ``fit_transform`` every row is scored by a model
    trained on the other folds (residuals computed inside the training folds
    only); ``transform`` averages the fold models.

    Pipeline step placed *before* the ``ColumnTransformer``; returns the
    input frame with ``vscore`` and ``vtype`` appended.

    Parameters
    ----------
    cat_cols : tuple of str
        Categorical construction columns (ordinal-encoded, native categorical splits).
    num_cols : tuple of str
        Numeric construction columns.
    location_col : str, default="id6"
        Area used for the leave-one-out mean.
    n_bins : int, default=12
    max_iter : int, default=150
        Boosting iterations of each fold model.
    n_splits : int, default=5
    random_state : int, default=42
    """

    def __init__(self, cat_cols=CONSTRUCTION_CAT_COLS + ("position", "plan_configuration",
                                                         "land_surface_condition"),
                 num_cols=CONSTRUCTION_NUM_COLS + ("h_per_floor", "slender", "id_last"),
                 location_col="id6", n_bins=12, max_iter=150, n_splits=5, random_state=42):
        """Store parameters (scikit-learn convention: no logic here)."""
        self.cat_cols = cat_cols
        self.num_cols = num_cols
        self.location_col = location_col
        self.n_bins = n_bins
        self.max_iter = max_iter
        self.n_splits = n_splits
        self.random_state = random_state

    def _new_model(self):
        """Create an unfitted encoder + regressor pipeline."""
        n_cat = len(self.cat_cols)
        prep = ColumnTransformer([
            ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), list(self.cat_cols)),
            ("num", "passthrough", list(self.num_cols)),
        ])
        reg = HistGradientBoostingRegressor(max_iter=self.max_iter, learning_rate=0.1, max_leaf_nodes=31,
                                            min_samples_leaf=200, categorical_features=list(range(n_cat)),
                                            random_state=self.random_state)
        return Pipeline([("prep", prep), ("reg", reg)])

    @staticmethod
    def _loo_residual(keys, grades):
        """Grade minus the leave-one-out mean grade of the row's area."""
        s = pd.Series(grades).groupby(keys).transform("sum").to_numpy()
        n = pd.Series(grades).groupby(keys).transform("size").to_numpy()
        loo = np.where(n > 1, (s - grades) / np.maximum(n - 1, 1), grades.mean())
        return grades - loo

    def _cols(self):
        """All input columns of the regressor."""
        return list(self.cat_cols) + list(self.num_cols)

    def _cross_fit(self, X, y):
        """Train one model per fold; return the out-of-fold scores."""
        self.classes_, y_idx = encode_labels(y)
        grades = (y_idx + 1).astype(float)
        keys = X[self.location_col].to_numpy()
        Xc = X[self._cols()]
        skf = StratifiedKFold(self.n_splits, shuffle=True, random_state=self.random_state)
        scores = np.empty(len(X))
        self.models_ = []
        for tr, va in skf.split(np.zeros(len(y_idx)), y_idx):
            resid = self._loo_residual(keys[tr], grades[tr])
            model = self._new_model().fit(Xc.iloc[tr], resid)
            scores[va] = model.predict(Xc.iloc[va])
            self.models_.append(model)
        self.edges_ = np.unique(np.quantile(scores, np.linspace(0, 1, self.n_bins + 1)[1:-1]))
        self.feature_names_out_ = np.asarray(list(X.columns) + ["vscore", "vtype"], dtype=object)
        return scores

    def _append(self, X, scores):
        """Return ``X`` with ``vscore`` and ``vtype`` columns."""
        X = X.copy()
        X["vscore"] = scores.astype(np.float32)
        X["vtype"] = np.searchsorted(self.edges_, scores, side="right").astype(np.int64)
        return X

    def fit(self, X, y):
        """Fit the fold models and the bin edges.

        Parameters
        ----------
        X : pandas.DataFrame
        y : array-like of shape (n_rows,)

        Returns
        -------
        self
        """
        self._cross_fit(X, y)
        return self

    def fit_transform(self, X, y=None, **fit_params):
        """Fit and append the **out-of-fold** vulnerability score and type.

        Parameters
        ----------
        X : pandas.DataFrame
        y : array-like of shape (n_rows,)
        **fit_params : ignored

        Returns
        -------
        pandas.DataFrame
        """
        return self._append(X, self._cross_fit(X, y))

    def transform(self, X):
        """Append the vulnerability score (mean of fold models) and type of new rows.

        Parameters
        ----------
        X : pandas.DataFrame

        Returns
        -------
        pandas.DataFrame
        """
        Xc = X[self._cols()]
        scores = np.mean([m.predict(Xc) for m in self.models_], axis=0)
        return self._append(X, scores)

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
