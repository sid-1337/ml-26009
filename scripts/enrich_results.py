"""Add measured, per-face ensemble evidence to the completed masking study.

Run after ``python -m task99.experiments`` finishes. This loads the frozen
models; it never refits, tunes, or changes case selection. Pairwise disagreement
is measured over unordered pairs of trees on the selected single face. Global
feature importance is descriptive training-set impurity importance, not a causal
explanation of that face's response or a measure of calibrated confidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import joblib
import numpy as np

# Allow the documented ``python scripts/enrich_results.py`` invocation without
# requiring a separately installed package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from task99.data import mask_faces
from task99.metrics import disagreement


def enrich_case(summary: dict, models: dict, X: np.ndarray, test: np.ndarray) -> dict:
    """Return per-face diagnostics; all numbers come from the frozen models."""
    case = summary["cases"]["q2"]
    index, position = int(case["index"]), int(case["position"])
    if int(test[position]) != index:
        raise ValueError("The chosen case does not match the saved test split")
    clean = X[index:index + 1]
    masked = mask_faces(clean)
    bag, boost = models["bagging"], models["boosting"]
    if not np.array_equal(bag.classes_, boost.classes_):
        raise ValueError("Frozen ensembles have inconsistent class orders")
    classes = bag.classes_
    bag_predictions = {
        name: np.asarray([tree.predict(image)[0] for tree in bag.estimators_])
        for name, image in [("clean", clean), ("masked", masked)]
    }
    boost_predictions = {
        name: np.asarray([tree.predict(image)[0] for tree in boost.estimators_])
        for name, image in [("clean", clean), ("masked", masked)]
    }
    alpha = boost.estimator_weights_
    diagnostics = {
        "class_order": classes.tolist(),
        "bagging_n_trees": len(bag.estimators_),
        "boosting_n_trees": len(boost.estimators_),
        "boosting_total_alpha": float(alpha.sum()),
        "bagging_tree_change_fraction": float(
            np.mean(bag_predictions["clean"] != bag_predictions["masked"])
        ),
        "boosting_alpha_weighted_tree_change_fraction": float(
            np.sum(alpha[boost_predictions["clean"] != boost_predictions["masked"]])
            / np.sum(alpha)
        ),
    }
    for condition, image in [("clean", clean), ("masked", masked)]:
        counts = bag.vote_counts(image)[0]
        weighted_votes = np.bincount(
            np.searchsorted(classes, boost_predictions[condition]),
            weights=alpha,
            minlength=len(classes),
        )
        diagnostics[f"bagging_{condition}_vote_counts"] = counts.tolist()
        diagnostics[f"bagging_{condition}_pairwise_disagreement"] = disagreement(
            bag_predictions[condition][:, None]
        )
        diagnostics[f"boosting_{condition}_weighted_votes"] = weighted_votes.tolist()
        # These checks guard against attaching evidence from a different model
        # to an already-selected example. No test-set outcome changes selection.
        bag_label = int(classes[np.argmax(counts)])
        boost_label = int(classes[np.argmax(weighted_votes)])
        if (
            bag_label != case[f"bagging_{condition}_prediction"]
            or boost_label != case[f"boosting_{condition}_prediction"]
        ):
            raise ValueError("Frozen model predictions differ from the saved case study")
        np.testing.assert_allclose(
            weighted_votes / alpha.sum(), boost.predict_proba(image)[0], atol=1e-12
        )
    central = np.zeros((64, 64), dtype=bool)
    central[24:40, 24:40] = True
    central = central.ravel()
    bag_importance = [float(t.feature_importances_[central].sum()) for t in bag.estimators_]
    boost_importance = [float(t.feature_importances_[central].sum()) for t in boost.estimators_]
    diagnostics["bagging_central_region_impurity_importance"] = float(np.mean(bag_importance))
    diagnostics["boosting_central_region_impurity_importance"] = float(
        np.average(boost_importance, weights=alpha)
    )
    return diagnostics


def write_json(path: Path, content: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(content, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    root = args.root.resolve()
    summary_path = root / "results/summary.json"
    summary = json.loads(summary_path.read_text())
    if "total_experiment_wall_seconds" not in summary["metadata"]:
        raise RuntimeError("Wait for all Q3 refits to finish before enriching summary.json")
    data_path = root / "data/olivetti.npz"
    digest = hashlib.sha256(data_path.read_bytes()).hexdigest()
    if digest != summary["metadata"]["dataset_sha256"]:
        raise ValueError("The dataset differs from the completed experiment's dataset")
    with np.load(data_path) as dataset:
        X = dataset["X"]
    with np.load(root / "data/split.npz") as saved_split:
        test = saved_split["test"]
    # Loading both saved predictions and models also verifies the case remains
    # part of the same completed experiment before writing additional evidence.
    with np.load(root / "results/predictions.npz") as predictions:
        pos = int(summary["cases"]["q2"]["position"])
        for name in ["bagging", "boosting"]:
            for condition in ["clean", "masked"]:
                expected = summary["cases"]["q2"][f"{name}_{condition}_prediction"]
                actual = int(np.argmax(predictions[f"{name}_{condition}"][pos]))
                if actual != expected:
                    raise ValueError("Saved predictions disagree with the case study")
    models = joblib.load(root / "models/selected.joblib")
    summary["cases"]["q2"].update(enrich_case(summary, models, X, test))
    summary["metadata"]["q2_case_evidence"] = {
        "pairwise_disagreement": "Fraction of unordered bagging-tree pairs disagreeing on this face.",
        "tree_change_fraction": "Fraction changing predicted class when the central block is masked.",
        "boosting_change_fraction": "Sum of alpha for changed weak learners divided by total alpha.",
        "weighted_votes": "Raw sums of alpha by predicted class; divide by boosting_total_alpha for confidence.",
        "central_region_importance": "Global training-set impurity importance summed over rows/columns 24:40; bagging mean or alpha-weighted boosting mean. Descriptive, not causal.",
    }
    write_json(root / "results/cases.json", summary["cases"])
    write_json(summary_path, summary)
    print("Enriched frozen Q2 case with per-face votes, disagreement, mask response, and global feature importance.")


if __name__ == "__main__":
    main()
