#!/usr/bin/env python3
"""Refit the three original selected models and compare saved class vectors."""
from pathlib import Path
import argparse
import json
import sys
from time import perf_counter
import numpy as np
from sklearn.tree import DecisionTreeClassifier
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from task99.models import Bagging,SAMME
from task99.data import make_split,make_resamples,mask_faces
from task99.experiments import child_seed


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1])
    root=p.parse_args().root
    s=json.loads((root/'results/summary.json').read_text())
    seed=s['metadata']['seed'];selected=s['selected']
    with np.load(root/'data/olivetti.npz') as z:
        X,y=z['X'],z['y']
    with np.load(root/'data/split.npz') as z:
        split={key:z[key] for key in z.files}
    with np.load(root/'data/resamples.npz') as z:
        resamples=z['indices']
    expected_split=make_split(y,seed)
    assert all(np.array_equal(expected_split[key],split[key]) for key in split)
    assert np.array_equal(make_resamples(y,split['train'],seed,30),resamples)
    models={
        'tree':DecisionTreeClassifier(max_depth=selected['tree']['depth'],min_samples_leaf=selected['tree']['min_samples_leaf'],random_state=child_seed(seed,11)),
        'bagging':Bagging(selected['bagging']['B'],max_depth=selected['tree']['depth'],min_samples_leaf=selected['tree']['min_samples_leaf'],random_state=child_seed(seed,22)),
        'boosting':SAMME(selected['boosting']['rounds'],max_depth=selected['boosting']['depth'],random_state=child_seed(seed,33))}
    started=perf_counter();checks=[]
    with np.load(root/'results/predictions.npz') as saved:
        for name,model in models.items():
            model.fit(X[split['train']],y[split['train']])
            for condition,features in [('clean',X[split['test']]),('masked',mask_faces(X[split['test']]))]:
                measured=model.predict_proba(features)
                expected=saved[name+'_'+condition]
                exact=bool(np.array_equal(measured,expected))
                maximum_error=float(np.max(np.abs(measured-expected)))
                assert np.allclose(measured,expected,atol=1e-12,rtol=0),(name,condition,maximum_error)
                checks.append({'model':name,'condition':condition,'bitwise_equal':exact,'maximum_absolute_error':maximum_error})
            print(name,'reproduced',flush=True)
    result={'status':'PASS','environment_executable':sys.executable,'environment_prefix':sys.prefix,'scope':'Saved split, 30 resamples, and fresh original selected-model clean/masked class vectors; this check does not rerun all 30 ensemble refits.',
            'checks':checks,'wall_seconds':perf_counter()-started}
    (root/'logs/reproducibility_check.json').write_text(json.dumps(result,indent=2)+'\n')
    print('PASS: split, resamples, and all selected-model predictions reproduced.')


if __name__=='__main__':
    main()
