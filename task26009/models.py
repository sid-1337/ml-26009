"""Transparent implementations of bootstrap aggregation and discrete SAMME.

Only the base decision tree comes from scikit-learn. Resampling, voting,
out-of-bag predictions, boosting coefficients, and weight updates are implemented
here. All ties select the first label in ``classes_`` (NumPy's sorted order).
"""

from __future__ import annotations

from collections.abc import Iterator
from time import perf_counter

import numpy as np
from sklearn.tree import DecisionTreeClassifier


def _training_data(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    X, y = np.asarray(X), np.asarray(y)
    if X.ndim != 2 or y.ndim != 1 or X.shape[0] != y.size or y.size == 0:
        raise ValueError("X must be a nonempty 2D array and y a matching 1D array")
    return X, y


def _positive_count(n_estimators: int) -> None:
    if not isinstance(n_estimators, (int, np.integer)) or n_estimators < 1:
        raise ValueError("n_estimators must be a positive integer")


def _prediction_data(model: object, X: np.ndarray) -> np.ndarray:
    if not hasattr(model, "classes_") or not model.estimators_:
        raise ValueError("Fit the model before requesting predictions")
    X = np.asarray(X)
    if X.ndim != 2 or X.shape[1] != model.n_features_in_:
        raise ValueError("Prediction data must have the same feature count as training data")
    return X


class Bagging:
    """Decision-tree bagging using full-size bootstrap samples and hard voting.

    Each bootstrap draws n training rows with replacement. A bootstrap may omit
    an entire class; votes are aligned to the original full class set using the
    predicted label, rather than a tree's local probability-column order.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int | None = None,
        min_samples_leaf: int = 1,
        random_state: int = 0,
    ) -> None:
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state

    def fit(self, X: np.ndarray, y: np.ndarray) -> Bagging:
        started = perf_counter()
        _positive_count(self.n_estimators)
        X, y = _training_data(X, y)
        self.classes_ = np.unique(y)
        self.n_features_in_ = X.shape[1]
        self.n_training_samples_ = len(y)
        self.estimators_: list[DecisionTreeClassifier] = []
        self.bootstrap_indices_: list[np.ndarray] = []
        self.fit_times_: list[float] = []
        rng = np.random.default_rng(self.random_state)
        for _ in range(self.n_estimators):
            indices = rng.integers(0, len(y), size=len(y))
            tree = DecisionTreeClassifier(
                max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf,
                random_state=int(rng.integers(0, np.iinfo(np.int32).max)),
            )
            tree.fit(X[indices], y[indices])
            self.estimators_.append(tree)
            self.bootstrap_indices_.append(indices)
            self.fit_times_.append(perf_counter() - started)
        return self

    def vote_counts(self, X: np.ndarray) -> np.ndarray:
        """Return integer hard-vote counts, in the full ``classes_`` order."""
        X = _prediction_data(self, X)
        counts = np.zeros((len(X), len(self.classes_)), dtype=np.int64)
        rows = np.arange(len(X))
        for tree in self.estimators_:
            columns = np.searchsorted(self.classes_, tree.predict(X))
            counts[rows, columns] += 1
        return counts

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return normalized hard votes, rather than averaged leaf probabilities."""
        return self.vote_counts(X) / len(self.estimators_)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]

    def staged_predict_proba(self, X: np.ndarray) -> Iterator[np.ndarray]:
        """Yield a fresh probability array for each ensemble prefix."""
        X = _prediction_data(self, X)
        counts = np.zeros((len(X), len(self.classes_)), dtype=np.int64)
        rows = np.arange(len(X))
        for number, tree in enumerate(self.estimators_, start=1):
            columns = np.searchsorted(self.classes_, tree.predict(X))
            counts[rows, columns] += 1
            yield counts / number

    def oob_predict_proba(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Predict original training rows using only trees that excluded each row.

        Pass the original training matrix, in its original order. Rows with no
        eligible tree are NaN and have ``covered=False``; exclude them when
        computing out-of-bag accuracy. ``oob_counts_`` records votes per row.
        """
        X = _prediction_data(self, X)
        if len(X) != self.n_training_samples_:
            raise ValueError("OOB prediction requires the original training rows in order")
        counts = np.zeros((len(X), len(self.classes_)), dtype=np.int64)
        totals = np.zeros(len(X), dtype=np.int64)
        for tree, indices in zip(self.estimators_, self.bootstrap_indices_):
            eligible = np.ones(len(X), dtype=bool)
            eligible[indices] = False
            rows = np.flatnonzero(eligible)
            if rows.size:
                columns = np.searchsorted(self.classes_, tree.predict(X[rows]))
                counts[rows, columns] += 1
                totals[rows] += 1
        covered = totals > 0
        probabilities = np.full(counts.shape, np.nan, dtype=float)
        probabilities[covered] = counts[covered] / totals[covered, None]
        self.oob_counts_ = totals
        return probabilities, covered


class SAMME:
    """Multiclass discrete AdaBoost with decision-tree weak learners.

    At each accepted round, alpha = log((1-error)/error) + log(K-1).
    Incorrectly classified rows receive an exp(alpha) multiplier, after which
    weights are normalized. A random-or-worse learner is rejected. If it is the
    first learner, fit raises ValueError; otherwise the accepted ensemble is
    retained. A perfect learner gets a finite coefficient using machine epsilon
    and terminates training. The learning rate is fixed at one.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int | None = 1,
        random_state: int = 0,
    ) -> None:
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        initial_sample_weight: np.ndarray | None = None,
    ) -> SAMME:
        started = perf_counter()
        _positive_count(self.n_estimators)
        X, y = _training_data(X, y)
        self.classes_ = np.unique(y)
        if len(self.classes_) < 2:
            raise ValueError("SAMME requires at least two classes")
        self.n_features_in_ = X.shape[1]
        if initial_sample_weight is None:
            weights = np.full(len(y), 1.0 / len(y))
        else:
            weights = np.asarray(initial_sample_weight, dtype=float).copy()
            if (
                weights.shape != (len(y),)
                or not np.all(np.isfinite(weights))
                or np.any(weights < 0)
                or weights.sum() <= 0
            ):
                raise ValueError("Initial weights must be finite, nonnegative, and have positive sum")
            weights /= weights.sum()
        self.estimators_: list[DecisionTreeClassifier] = []
        coefficients: list[float] = []
        self.history_: list[dict[str, float | int]] = []
        self.sample_weight_history_ = [weights.copy()]
        self.training_predictions_: list[np.ndarray] = []
        self.fit_times_: list[float] = []
        self.termination_reason_ = "max_estimators"
        rng = np.random.default_rng(self.random_state)
        random_error = 1.0 - 1.0 / len(self.classes_)
        for round_number in range(1, self.n_estimators + 1):
            tree = DecisionTreeClassifier(
                max_depth=self.max_depth,
                random_state=int(rng.integers(0, np.iinfo(np.int32).max)),
            )
            tree.fit(X, y, sample_weight=weights)
            predictions = tree.predict(X)
            incorrect = predictions != y
            error = float(np.sum(weights[incorrect]))
            # Numerical tolerance prevents accepting a chance-level learner
            # because of summation roundoff, yielding a meaningless tiny alpha.
            if error >= random_error - 1e-12:
                self.termination_reason_ = "weak_learner_not_better_than_random"
                self.estimator_weights_ = np.asarray(coefficients, dtype=float)
                if not self.estimators_:
                    raise ValueError("The first weak learner is not better than random multiclass prediction")
                break
            clipped_error = max(error, np.finfo(float).eps)
            alpha = float(
                np.log1p(-clipped_error)
                - np.log(clipped_error)
                + np.log(len(self.classes_) - 1)
            )
            weights = weights * np.exp(alpha * incorrect)
            weights /= weights.sum()
            self.estimators_.append(tree)
            coefficients.append(alpha)
            self.training_predictions_.append(predictions.copy())
            self.sample_weight_history_.append(weights.copy())
            self.history_.append(
                {
                    "round": round_number,
                    "weighted_error": error,
                    "alpha": alpha,
                    "max_weight": float(weights.max()),
                    "top10_weight_sum": float(np.sort(weights)[-10:].sum()),
                }
            )
            self.fit_times_.append(perf_counter() - started)
            if error == 0.0:
                self.termination_reason_ = "perfect_learner"
                break
        self.estimator_weights_ = np.asarray(coefficients, dtype=float)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return normalized positive-alpha hard votes in full class order."""
        X = _prediction_data(self, X)
        scores = np.zeros((len(X), len(self.classes_)), dtype=float)
        rows = np.arange(len(X))
        for tree, alpha in zip(self.estimators_, self.estimator_weights_):
            columns = np.searchsorted(self.classes_, tree.predict(X))
            scores[rows, columns] += alpha
        return scores / self.estimator_weights_.sum()

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]

    def staged_predict_proba(self, X: np.ndarray) -> Iterator[np.ndarray]:
        X = _prediction_data(self, X)
        scores = np.zeros((len(X), len(self.classes_)), dtype=float)
        rows = np.arange(len(X))
        total_weight = 0.0
        for tree, alpha in zip(self.estimators_, self.estimator_weights_):
            columns = np.searchsorted(self.classes_, tree.predict(X))
            scores[rows, columns] += alpha
            total_weight += alpha
            yield scores / total_weight
