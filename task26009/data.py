"""Fixed per-person splits, central occlusion, and within-person resampling."""
from pathlib import Path
import json
import numpy as np


def roll_seed(roll_number):
    digits = ''.join(c for c in str(roll_number) if c.isdecimal())
    if not digits:
        raise ValueError('The complete roll number must contain digits.')
    return int(digits)


def make_split(y, seed):
    rng = np.random.default_rng(seed)
    parts = {'train': [], 'validation': [], 'test': []}
    for identity in np.unique(y):
        indices = np.flatnonzero(y == identity)
        if len(indices) != 10:
            raise ValueError('Exactly ten images per identity are required.')
        indices = rng.permutation(indices)
        for name, selected in [('train', indices[:6]), ('validation', indices[6:8]), ('test', indices[8:])]:
            parts[name].extend(selected.tolist())
    return {name: np.asarray(indices, dtype=int) for name, indices in parts.items()}


def mask_faces(X):
    masked = np.asarray(X).reshape(-1, 64, 64).copy()
    masked[:, 24:40, 24:40] = 0
    return masked.reshape(-1, 4096)


def make_resamples(y, train, seed, repetitions=30):
    rng = np.random.default_rng(np.random.SeedSequence([seed, 3003]))
    pools = [train[y[train] == k] for k in np.unique(y)]
    return np.asarray([np.concatenate([rng.choice(pool, size=6, replace=True) for pool in pools])
                       for _ in range(repetitions)])


def prepare(root, config):
    root = Path(root)
    path = root / 'data/olivetti.npz'
    if not path.exists():
        from sklearn.datasets import fetch_olivetti_faces
        faces = fetch_olivetti_faces(data_home=str(root / 'data/cache'), shuffle=False)
        np.savez_compressed(path, X=faces.data, y=faces.target, images=faces.images)
        (root / 'data/DATASET_DESCRIPTION.txt').write_text(faces.DESCR)
    with np.load(path) as data:
        X, y = data['X'], data['y']
    seed = roll_seed(config['roll_number']) if config.get('roll_number') else int(config['seed'])
    split = make_split(y, seed)
    split_path = root / 'data/split.npz'
    split_meta = root / 'data/split_metadata.json'
    if split_path.exists() and split_meta.exists():
        saved_seed = json.loads(split_meta.read_text())['seed']
        if saved_seed != seed:
            raise ValueError('Saved split has another seed. Use a separate --root or remove old generated outputs intentionally.')
        with np.load(split_path) as existing:
            if any(not np.array_equal(existing[k], v) for k, v in split.items()):
                raise ValueError('Saved split does not match this seed and dataset.')
    np.savez_compressed(split_path, **split)
    resamples = make_resamples(y, split['train'], seed, config['resamples'])
    np.savez_compressed(root / 'data/resamples.npz', indices=resamples)
    split_meta.write_text(json.dumps({'seed': seed, 'seed_is_provisional': not bool(config.get('roll_number')),
                                     'roll_number': config.get('roll_number'),
                                     'split_counts': {k: len(v) for k, v in split.items()}}, indent=2) + '\n')
    return X, y, split, resamples, seed
