#!/usr/bin/env python3
"""Check completed artifacts against experimental invariants, without refitting."""
from pathlib import Path
import argparse
import json
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from task99.metrics import accuracy,bias_variance


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1])
    root=p.parse_args().root
    s=json.loads((root/'results/summary.json').read_text())
    assert 'total_experiment_wall_seconds' in s['metadata'], 'Incomplete experiment run'
    assert len(s['tree_search'])==15
    assert {r['B'] for r in s['bagging_search']}=={1,5,10,25,50,100,200}
    assert len(s['boosting_search'])==16
    assert len(s['matched'])==6
    assert len(s['refit_accuracy_records'])==30
    with np.load(root/'data/olivetti.npz') as z:
        y=z['y']
    with np.load(root/'data/split.npz') as z:
        split={k:z[k] for k in z.files}
    with np.load(root/'data/resamples.npz') as z:
        samples=z['indices']
    assert len(set(split['train']) & set(split['test']))==0
    assert len(set(split['validation']) & set(split['test']))==0
    assert samples.shape==(30,240)
    assert set(samples.ravel())<=set(split['train'])
    with np.load(root/'results/predictions.npz') as z:
        weights=z['boost_sample_weights']
        weak_predictions=z['boost_train_predictions']
        assert np.allclose(weights.sum(axis=1),1)
        assert len(weights)==len(weak_predictions)+1==len(s['boosting_history'])+1
        for t,history in enumerate(s['boosting_history']):
            mistakes=weak_predictions[t]!=y[split['train']]
            error=weights[t,mistakes].sum()
            assert np.isclose(error,history['weighted_error'])
            if error>0:
                alpha=np.log((1-error)/error)+np.log(39)
                assert np.isclose(alpha,history['alpha'])
                # A separate consequence of SAMME's formula, not an update replay.
                assert np.isclose(weights[t+1,mistakes].sum(),39/40)
        for name in ['tree','bagging','boosting']:
            record=next(r for r in s['biasvariance'] if r['model']==name)
            values=z['q3_'+name]
            assert values.shape==(30,80,40)
            assert np.isfinite(values).all() and (values>=0).all()
            assert np.allclose(values.sum(axis=-1),1)
            measured=bias_variance(values,y[split['test']])
            for key in ['bias_squared','variance','mean_accuracy','std_accuracy','brier_mse']:
                assert np.isclose(measured[key],record[key]),(name,key)
            robust=next(r for r in s['robustness'] if r['model']==name)
            for condition in ['clean','masked']:
                assert np.isclose(accuracy(z[name+'_'+condition],y[split['test']]),robust[condition+'_accuracy'])
        for record in s['depth_study']:
            depth='unlimited' if record['depth'] is None else str(record['depth'])
            values=z[f"q3_depth_{depth}_{record['model']}"]
            assert values.shape==(30,80,40)
            assert np.isfinite(values).all() and (values>=0).all()
            assert np.allclose(values.sum(axis=-1),1)
            measured=bias_variance(values,y[split['test']])
            for key in ['bias_squared','variance','mean_accuracy','std_accuracy','brier_mse']:
                assert np.isclose(measured[key],record[key]),(depth,record['model'],key)
        for case in s['cases']['q1']:
            assert sum(case['vote_counts'])==s['selected']['bagging']['B']
            assert case['truth']==int(y[case['index']])
            if case['category']=='correction':
                assert case['bagging_prediction']==case['truth']!=case['tree_prediction']
            else:
                assert case['bagging_prediction']!=case['truth'] and case['tree_prediction']!=case['truth']
    chosen=min(s['tree_search'],key=lambda r:(-r['validation_accuracy'],r['depth'] if r['depth'] is not None else float('inf'),-r['min_samples_leaf']))
    assert {k:chosen[k] for k in ['depth','min_samples_leaf']}==s['selected']['tree']
    assert min(s['bagging_search'],key=lambda r:(-r['validation_accuracy'],r['B']))['B']==s['selected']['bagging']['B']
    chosen=min(s['boosting_search'],key=lambda r:(-r['validation_accuracy'],r['depth'],r['rounds']))
    assert chosen['depth']==s['selected']['boosting']['depth'] and chosen['rounds']==s['selected']['boosting']['rounds']
    print('PASS: complete grids, split isolation, 30 resamples/refits, normalized class vectors, recomputed metrics, empirical cases, and validation-only selections.')


if __name__=='__main__':
    main()
