"""Class-vector metrics used throughout the assignment."""
import numpy as np


def accuracy(probabilities, y):
    return float(np.mean(np.argmax(probabilities, axis=1) == y))


def mean_margin(probabilities):
    top = np.partition(probabilities, -2, axis=1)[:, -2:]
    return float(np.mean(np.max(top, axis=1) - np.min(top, axis=1)))


def disagreement(predictions):
    """Mean fraction of differing labels over unordered classifier pairs."""
    predictions = np.asarray(predictions)
    B, n = predictions.shape
    if B < 2:
        return None
    # Count agreements by class per image, avoiding a quadratic pairwise tensor.
    agreement_pairs = np.zeros(n)
    for label in np.unique(predictions):
        count = np.sum(predictions == label, axis=0)
        agreement_pairs += count * (count - 1) / 2
    return float(1 - np.mean(agreement_pairs) / (B * (B - 1) / 2))


def leaf_stats(tree):
    counts = tree.tree_.n_node_samples[tree.tree_.children_left == -1]
    return {'n_leaves': int(len(counts)), 'mean_leaf_samples': float(np.mean(counts)),
            'singleton_fraction': float(np.mean(counts == 1))}


def bias_variance(probabilities, y):
    """Assignment's finite-resample Euclidean class-vector decomposition."""
    p = np.asarray(probabilities)
    target = np.eye(p.shape[-1])[y]
    mean = p.mean(axis=0)
    bias = float(np.mean(np.sum((mean - target)**2, axis=1)))
    variance = float(np.mean(np.sum((p - mean)**2, axis=-1)))
    brier = float(np.mean(np.sum((p - target)**2, axis=-1)))
    accuracies = np.mean(np.argmax(p, axis=-1) == y[None, :], axis=1)
    if not np.isclose(bias + variance, brier, atol=1e-10):
        raise AssertionError('Squared-error bias/variance decomposition failed.')
    return {'bias_squared': bias, 'variance': variance, 'brier_mse': brier,
            'mean_accuracy': float(accuracies.mean()), 'std_accuracy': float(accuracies.std(ddof=1))}
