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

This module only depends on numpy, pandas and scikit-learn.
"""
import warnings

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import StratifiedKFold

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
       "total_area", "log_area", "age_x_floors", "vuln"]
)

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
    """Label statistics of the buildings with the closest ``building_id``.

    Survey IDs are assigned sequentially inside an (old) ward, so buildings
    with close IDs tend to be physically close and to have suffered similar
    ground shaking. For each row this encoder finds the ``k`` labelled
    buildings in the same ``group_col`` with the smallest
    ``|id - id_neighbour|`` and returns, for every ``k`` in ``ks``: the
    number of neighbours found, their mean grade, their class shares and the
    log mean ID distance. It also returns a distance-weighted mean grade of
    the ``weighted_k`` nearest neighbours and the log distance to the
    nearest one.

    Uses cross-fitting, see the module docstring.

    Parameters
    ----------
    id_col : str, default="bid"
        Integer building ID column.
    group_col : str, default="id6"
        Neighbours must share this area code.
    ks : tuple of int, default=(3, 10, 30)
        Neighbourhood sizes.
    window : int, default=30
        Candidates searched on each side of the sorted position
        (``max(ks)`` must be <= ``2 * window``).
    weighted_k : int, default=10
        Neighbourhood size for the distance-weighted mean.
    n_splits : int, default=5
        Number of cross-fitting folds.
    random_state : int, default=42
        Seed of the fold split.
    chunk_size : int, default=100_000
        Rows processed at once (limits memory use).
    """

    def __init__(self, id_col="bid", group_col="id6", ks=(3, 10, 30), window=30, weighted_k=10,
                 n_splits=5, random_state=42, chunk_size=100_000):
        """Store parameters (scikit-learn convention: no logic here)."""
        self.id_col = id_col
        self.group_col = group_col
        self.ks = ks
        self.window = window
        self.weighted_k = weighted_k
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
        n_known, n_cls, w = len(kb), self.n_classes_, self.window
        pos = np.searchsorted(kb, ids)
        cand = pos[:, None] + np.arange(-w, w)[None, :]
        valid = (cand >= 0) & (cand < n_known)
        cc = np.clip(cand, 0, n_known - 1)
        dist = np.abs(kb[cc] - ids[:, None]).astype(np.float64)
        dist[~valid | (kg[cc] != groups[:, None])] = np.inf
        order = np.argsort(dist, axis=1)
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
        total = np.zeros((len(X), len(self.feature_names_out_)))
        count = np.zeros_like(total)
        for store in self.stores_:
            enc = self._encode(ids, groups, store)
            fin = np.isfinite(enc)
            total += np.where(fin, enc, 0.0)
            count += fin
        with np.errstate(invalid="ignore", divide="ignore"):
            out = np.where(count > 0, total / np.maximum(count, 1), np.nan)
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
