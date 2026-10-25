"""Algorithm checks independent of the assignment's held-out face test set."""

import unittest
from unittest.mock import patch

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal

from task99.models import Bagging, SAMME


class TestBagging(unittest.TestCase):
    def setUp(self):
        self.X = np.arange(12, dtype=float).reshape(-1, 1)
        self.y = np.repeat([10, 20, 90], 4)

    def test_oob_predictions_exclude_every_in_bag_tree(self):
        model = Bagging(n_estimators=11, max_depth=2, random_state=5).fit(self.X, self.y)
        probabilities, covered = model.oob_predict_proba(self.X)
        expected_counts = np.zeros((len(self.X), 3), dtype=int)
        expected_totals = np.zeros(len(self.X), dtype=int)
        # Membership, not bootstrap multiplicity, decides OOB eligibility.
        for row in range(len(self.X)):
            for tree, bootstrap in zip(model.estimators_, model.bootstrap_indices_):
                if row not in set(bootstrap):
                    label = tree.predict(self.X[row:row + 1])[0]
                    expected_counts[row, np.flatnonzero(model.classes_ == label)[0]] += 1
                    expected_totals[row] += 1
        assert_array_equal(covered, expected_totals > 0)
        assert_array_equal(model.oob_counts_, expected_totals)
        assert_allclose(probabilities[covered], expected_counts[covered] / expected_totals[covered, None])
        self.assertTrue(np.isnan(probabilities[~covered]).all())

    def test_hard_votes_keep_original_classes_when_bootstrap_omits_one(self):
        X = np.array([[0.0], [1.0], [2.0]])
        y = np.array([11, 35, 81])
        model = Bagging(n_estimators=5, random_state=2).fit(X, y)
        self.assertTrue(any(len(tree.classes_) < 3 for tree in model.estimators_))
        expected = np.zeros((3, 3), dtype=int)
        for tree in model.estimators_:
            for row, prediction in enumerate(tree.predict(X)):
                expected[row, list(model.classes_).index(prediction)] += 1
        assert_array_equal(model.vote_counts(X), expected)
        assert_allclose(model.predict_proba(X), expected / 5)
        assert_allclose(model.predict_proba(X).sum(axis=1), 1)

    def test_seed_and_prefix_reproducibility(self):
        small = Bagging(n_estimators=3, random_state=17).fit(self.X, self.y)
        large = Bagging(n_estimators=6, random_state=17).fit(self.X, self.y)
        for actual, expected in zip(small.bootstrap_indices_, large.bootstrap_indices_):
            assert_array_equal(actual, expected)
        stages = list(large.staged_predict_proba(self.X))
        assert_allclose(stages[2], small.predict_proba(self.X))
        assert_allclose(stages[-1], large.predict_proba(self.X))
        # Generator outputs must not be views of the same accumulating array.
        self.assertFalse(np.shares_memory(stages[0], stages[-1]))

    def test_no_oob_prediction_is_reported_for_in_bag_only_rows(self):
        model = Bagging(n_estimators=1, random_state=0).fit(self.X, self.y)
        probabilities, covered = model.oob_predict_proba(self.X)
        self.assertTrue((~covered).any())
        assert_array_equal(np.flatnonzero(~covered), np.unique(model.bootstrap_indices_[0]))
        self.assertTrue(np.isnan(probabilities[~covered]).all())


class _FixedTree:
    """Weak learner with controlled predictions to isolate boundary behavior."""

    def __init__(self, predictions):
        self.predictions = np.asarray(predictions)

    def fit(self, X, y, sample_weight=None):
        return self

    def predict(self, X):
        return self.predictions[:len(X)]


class TestSAMME(unittest.TestCase):
    def setUp(self):
        self.X = np.arange(6, dtype=float).reshape(-1, 1)
        self.y = np.array([0, 0, 1, 1, 2, 2])

    def test_three_class_one_round_has_hand_calculated_weights(self):
        model = SAMME(n_estimators=1, max_depth=1, random_state=3).fit(self.X, self.y)
        self.assertAlmostEqual(model.history_[0]["weighted_error"], 1 / 3)
        self.assertAlmostEqual(model.estimator_weights_[0], np.log(4))
        incorrect = model.training_predictions_[0] != self.y
        self.assertEqual(int(incorrect.sum()), 2)
        assert_allclose(model.sample_weight_history_[0], np.full(6, 1 / 6))
        assert_allclose(model.sample_weight_history_[1][incorrect], np.full(2, 1 / 3))
        assert_allclose(model.sample_weight_history_[1][~incorrect], np.full(4, 1 / 12))
        self.assertEqual(len(model.history_), 1)
        self.assertAlmostEqual(model.history_[0]["max_weight"], 1 / 3)
        self.assertAlmostEqual(model.history_[0]["top10_weight_sum"], 1)

    def test_initial_weights_change_weak_learner_and_are_normalized(self):
        weighted = SAMME(n_estimators=1, max_depth=1, random_state=3).fit(
            self.X, self.y, initial_sample_weight=np.array([1, 1, 1, 1, 8, 8]),
        )
        # The last class carries 80% of the initial mass and must be recognized.
        assert_array_equal(weighted.training_predictions_[0][-2:], [2, 2])
        self.assertAlmostEqual(weighted.history_[0]["weighted_error"], 0.1)
        self.assertAlmostEqual(weighted.estimator_weights_[0], np.log(18))
        incorrect = weighted.training_predictions_[0] != self.y
        assert_allclose(weighted.sample_weight_history_[1][incorrect], [1 / 3, 1 / 3])
        self.assertAlmostEqual(weighted.sample_weight_history_[1].sum(), 1)

    def test_chance_level_first_learner_is_rejected_even_with_roundoff(self):
        with patch("task99.models.DecisionTreeClassifier", return_value=_FixedTree([0] * 6)):
            with self.assertRaisesRegex(ValueError, "first weak learner"):
                SAMME(n_estimators=5).fit(self.X, self.y)

    def test_later_random_or_worse_learner_preserves_accepted_ensemble(self):
        # First error is 1/3. Its two mistakes then carry total weight 2/3.
        first = _FixedTree([0, 0, 1, 1, 1, 1])
        second = _FixedTree([1, 1, 0, 0, 0, 0])
        with patch("task99.models.DecisionTreeClassifier", side_effect=[first, second]):
            model = SAMME(n_estimators=5).fit(self.X, self.y)
        self.assertEqual(len(model.estimators_), 1)
        self.assertEqual(model.termination_reason_, "weak_learner_not_better_than_random")
        self.assertEqual(len(model.sample_weight_history_), 2)
        assert_array_equal(model.predict(self.X), first.predictions)

    def test_perfect_learner_terminates_with_finite_confidence(self):
        model = SAMME(n_estimators=20, max_depth=None, random_state=1).fit(self.X, self.y)
        self.assertEqual(model.termination_reason_, "perfect_learner")
        self.assertEqual(len(model.estimators_), 1)
        self.assertTrue(np.isfinite(model.estimator_weights_).all())
        assert_array_equal(model.predict(self.X), self.y)
        assert_allclose(model.predict_proba(self.X), np.eye(3)[self.y])
        assert_allclose(model.sample_weight_history_[0], model.sample_weight_history_[1])

    def test_votes_are_alpha_weighted_and_aligned_for_nonconsecutive_labels(self):
        X = np.arange(4, dtype=float).reshape(-1, 1)
        y = np.array([10, 20, 90, 90])
        first = _FixedTree([10, 20, 20, 90])
        second = _FixedTree([20, 20, 90, 90])
        with patch("task99.models.DecisionTreeClassifier", side_effect=[first, second]):
            model = SAMME(n_estimators=2).fit(X, y)
        # Error_1=1/4 => alpha_1=log(6). Weight_2(row 0)=1/9,
        # so error_2=1/9 => alpha_2=log(16).
        a, b = np.log(6), np.log(16)
        assert_allclose(model.estimator_weights_, [a, b])
        assert_allclose(model.predict_proba(X)[0], [a / (a + b), b / (a + b), 0])
        assert_allclose(model.predict_proba(X)[2], [0, a / (a + b), b / (a + b)])
        stages = list(model.staged_predict_proba(X))
        assert_allclose(stages[-1], model.predict_proba(X))
        self.assertFalse(np.shares_memory(stages[0], stages[-1]))

    def test_deterministic_seed(self):
        first = SAMME(n_estimators=8, random_state=19).fit(self.X, self.y)
        second = SAMME(n_estimators=8, random_state=19).fit(self.X, self.y)
        assert_allclose(first.estimator_weights_, second.estimator_weights_)
        assert_allclose(first.sample_weight_history_, second.sample_weight_history_)
        assert_allclose(first.predict_proba(self.X), second.predict_proba(self.X))

    def test_invalid_initial_weights_are_rejected(self):
        for invalid in (np.zeros(6), -np.ones(6), np.ones(5), np.full(6, np.nan)):
            with self.subTest(weights=invalid):
                with self.assertRaises(ValueError):
                    SAMME(n_estimators=1).fit(self.X, self.y, initial_sample_weight=invalid)


if __name__ == "__main__":
    unittest.main()
