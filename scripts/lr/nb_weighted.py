"""
NB-weighted Logistic Regression (Wang & Manning, ACL 2012).

For each class c (one-vs-rest), a Naive Bayes log-count ratio vector r_c is
computed from the training labels, then a binary LR is trained on X * r_c.
Prediction takes the class whose OvR classifier assigns the highest
positive-class probability.

    r_i = log( (p_i / Σ_j p_j) / (q_i / Σ_j q_j) )

where p_i = α + Σ_{d∈pos} x_{d,i}  and  q_i = α + Σ_{d∈neg} x_{d,i}.

GPU mode (use_gpu=True):
    Requires cuML (RAPIDS). Internally converts scipy sparse → cupy sparse
    and uses cuml.linear_model.LogisticRegression. Gives a large speedup for
    grid/random search since each LR fit is GPU-accelerated.
"""

import numpy as np
from scipy.sparse import issparse
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.utils.validation import check_is_fitted


def _to_cupy_sparse(X):
    import cupyx.scipy.sparse as csp
    return csp.csr_matrix(X)


def _to_cupy(arr):
    import cupy as cp
    return cp.asarray(arr)


def _to_numpy(arr):
    """Convert cupy array → numpy if needed."""
    try:
        import cupy as cp
        if isinstance(arr, cp.ndarray):
            return arr.get()
    except ImportError:
        pass
    return np.asarray(arr)


class NBWeightedLR(BaseEstimator, ClassifierMixin):
    """
    NB-weighted Logistic Regression for binary or multiclass sentiment.

    Parameters
    ----------
    C : float
        Inverse regularisation strength passed to LogisticRegression.
    alpha : float
        Laplace smoothing parameter for NB count ratios (default 1.0).
    l1_ratio : float
        0.0 → L2, 1.0 → L1 (saga solver; ignored in GPU mode which uses 'qn').
    class_weight : None or 'balanced'
        Passed through to LogisticRegression.
    use_gpu : bool
        If True, use cuML GPU-accelerated LR. Requires RAPIDS cuML.
    """

    def __init__(self, C=1.0, alpha=1.0, l1_ratio=0.0, class_weight=None, use_gpu=False):
        self.C = C
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.class_weight = class_weight
        self.use_gpu = use_gpu

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _nb_ratio(self, X, pos_mask):
        """Compute log-count ratio for a binary pos/neg split (always CPU numpy)."""
        p = np.asarray(X[pos_mask].sum(axis=0)).flatten() + self.alpha
        q = np.asarray(X[~pos_mask].sum(axis=0)).flatten() + self.alpha
        return np.log((p / p.sum()) / (q / q.sum()))

    def _scale(self, X, r):
        """Element-wise multiply each feature column by r."""
        if self.use_gpu:
            import cupy as cp
            X_gpu = _to_cupy_sparse(X) if issparse(X) else _to_cupy(X)
            return X_gpu.multiply(_to_cupy(r))
        return X.multiply(r) if issparse(X) else X * r

    def _make_binary_lr(self):
        if self.use_gpu:
            from cuml.linear_model import LogisticRegression as cuLR
            # cuML LR uses 'qn' solver which supports L1/L2/elasticnet
            penalty = "l1" if self.l1_ratio == 1.0 else "l2"
            return cuLR(
                C=self.C,
                penalty=penalty,
                max_iter=1000,
                class_weight=self.class_weight,
            )
        return LogisticRegression(
            C=self.C,
            l1_ratio=self.l1_ratio,
            solver="saga",
            class_weight=self.class_weight,
            max_iter=1000,
            random_state=42,
        )

    # ------------------------------------------------------------------
    # sklearn API
    # ------------------------------------------------------------------

    def fit(self, X, y):
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        self.ratios_ = {}
        self.clfs_ = {}

        for c in self.classes_:
            pos_mask = y == c
            # NB ratio always computed on CPU (sparse sum is fast there)
            r = self._nb_ratio(X, pos_mask)
            self.ratios_[c] = r

            X_scaled = self._scale(X, r)
            labels = pos_mask.astype(int)
            if self.use_gpu:
                labels = _to_cupy(labels)

            clf = self._make_binary_lr()
            clf.fit(X_scaled, labels)
            self.clfs_[c] = clf

        return self

    def predict_proba(self, X):
        check_is_fitted(self)
        cols = []
        for c in self.classes_:
            X_scaled = self._scale(X, self.ratios_[c])
            proba = self.clfs_[c].predict_proba(X_scaled)
            # cuML returns cupy; sklearn returns numpy — normalise to numpy
            proba = _to_numpy(proba)
            cols.append(proba[:, 1])
        scores = np.column_stack(cols)
        scores = scores / scores.sum(axis=1, keepdims=True)
        return scores

    def predict(self, X):
        check_is_fitted(self)
        scores = np.column_stack([
            _to_numpy(self.clfs_[c].predict_proba(self._scale(X, self.ratios_[c])))[:, 1]
            for c in self.classes_
        ])
        return self.classes_[np.argmax(scores, axis=1)]
