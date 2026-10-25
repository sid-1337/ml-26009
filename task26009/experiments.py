"""Actual, reproducible experiments for every marked assignment question."""
from __future__ import annotations
import argparse
import copy
import hashlib
import importlib.metadata
import json
import platform
import shutil
import sys
from pathlib import Path
from time import perf_counter

import joblib
import numpy as np
from joblib import Parallel, delayed, parallel_config
from sklearn.tree import DecisionTreeClassifier

from .data import prepare, mask_faces, roll_seed
from .metrics import accuracy, mean_margin, disagreement, leaf_stats, bias_variance
from .models import Bagging, SAMME


def write_json(path, value):
    def convert(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.generic):
            return obj.item()
        raise TypeError(type(obj).__name__)
    Path(path).write_text(json.dumps(value, default=convert, indent=2, allow_nan=False) + '\n')


def child_seed(seed, stream):
    return int(np.random.SeedSequence([seed, stream]).generate_state(1)[0])


def prefix(model, count):
    result = copy.copy(model)
    count = min(count, len(model.estimators_))
    result.estimators_ = model.estimators_[:count]
    if isinstance(model, Bagging):
        result.bootstrap_indices_ = model.bootstrap_indices_[:count]
    else:
        result.estimator_weights_ = model.estimator_weights_[:count]
        result.history_ = model.history_[:count]
        result.sample_weight_history_ = model.sample_weight_history_[:count + 1]
        result.training_predictions_ = model.training_predictions_[:count]
    result.fit_times_ = model.fit_times_[:count]
    return result


def split_scores(model, X, y, split):
    return {name + '_accuracy': accuracy(model.predict_proba(X[split[name]]), y[split[name]])
            for name in ['train', 'validation', 'test']}


def ensemble_leaf_stats(model):
    records = [leaf_stats(t) for t in model.estimators_]
    return {name: float(np.mean([r[name] for r in records])) for name in records[0]}


def resample_worker(rep, indices, X, y, split, selected, depths, seed, root, fingerprint):
    """One paired repetition; cached only for this exact run specification."""
    folder = Path(root) / 'results/refits'
    stem = folder / f'refit_{rep:02d}'
    if stem.with_suffix('.json').exists() and stem.with_suffix('.npz').exists():
        meta = json.loads(stem.with_suffix('.json').read_text())
        if meta.get('fingerprint') == fingerprint:
            with np.load(stem.with_suffix('.npz')) as arrays:
                return rep, meta, {k: arrays[k] for k in arrays.files}
    Xr, yr = X[indices], y[indices]
    Xt, Xv = X[split['test']], X[split['validation']]
    depth_records, arrays = [], {}
    selected_depth = selected['tree']['depth']
    selected_B = selected['bagging']['B']
    tree_setting = selected['tree']
    for depth in depths:
        tree = DecisionTreeClassifier(max_depth=depth, min_samples_leaf=tree_setting['min_samples_leaf'],
                                      random_state=child_seed(seed, 11)).fit(Xr, yr)
        tree_p = tree.predict_proba(Xt)
        maximum = max(100, selected_B) if depth == selected_depth else 100
        bag = Bagging(maximum, max_depth=depth, min_samples_leaf=tree_setting['min_samples_leaf'],
                      random_state=child_seed(seed, 22)).fit(Xr, yr)
        bag100 = prefix(bag, 100)
        label = 'unlimited' if depth is None else str(depth)
        arrays[f'depth_{label}_tree'] = tree_p
        arrays[f'depth_{label}_bagging'] = bag100.predict_proba(Xt)
        tree_rec = {'model':'tree', 'depth':depth, **leaf_stats(tree), 'disagreement':None}
        bag_rec = {'model':'bagging', 'depth':depth, **ensemble_leaf_stats(bag100),
                   'disagreement':disagreement(np.asarray([t.predict(Xv) for t in bag100.estimators_]))}
        depth_records.extend([tree_rec, bag_rec])
        if depth == selected_depth:
            arrays['tree'] = tree_p
            selected_bag = prefix(bag, selected_B)
            arrays['bagging'] = selected_bag.predict_proba(Xt)
            selected_bag_stats = {**ensemble_leaf_stats(selected_bag),
                                  'disagreement':disagreement(np.asarray([t.predict(Xv) for t in selected_bag.estimators_]))}
            selected_tree_stats = leaf_stats(tree)
    boost = SAMME(selected['boosting']['rounds'], max_depth=selected['boosting']['depth'],
                  random_state=child_seed(seed, 33)).fit(Xr, yr)
    arrays['boosting'] = boost.predict_proba(Xt)
    selected_stats = {'tree':{**selected_tree_stats,'disagreement':None},
                      'bagging':selected_bag_stats,
                      'boosting':{**ensemble_leaf_stats(boost),'disagreement':None}}
    meta = {'repetition':rep, 'fingerprint':fingerprint, 'depth_records':depth_records,
            'selected_stats':selected_stats, 'boost_actual_rounds':len(boost.estimators_),
            'boost_termination':boost.termination_reason_,
            'accuracies':{name:accuracy(arrays[name],y[split['test']]) for name in selected_stats}}
    np.savez_compressed(stem.with_suffix('.npz'), **arrays)
    write_json(stem.with_suffix('.json'),meta)
    return rep, meta, arrays


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1])
    parser.add_argument('--config',type=Path)
    parser.add_argument('--roll-number', help='Overrides provisional seed using all digits of your complete roll number')
    parser.add_argument('--jobs',type=int)
    args=parser.parse_args()
    root=args.root.resolve()
    for name in ['data','results/refits','logs','models','figures','report']:
        (root/name).mkdir(parents=True,exist_ok=True)
    source_root=Path(__file__).resolve().parents[1]
    config=json.loads((args.config or source_root/'configs/default.json').read_text())
    if root != source_root:
        for relative in ['data/olivetti.npz','report/cvpr.sty','report/ieeenat_fullname.bst']:
            source=source_root/relative
            target=root/relative
            if source.exists() and not target.exists():
                shutil.copy2(source,target)
    (root/'configs').mkdir(exist_ok=True)
    if args.roll_number:
        config.update(roll_number=args.roll_number,seed=roll_seed(args.roll_number),seed_is_provisional=False)
    if args.jobs is not None:
        config['n_jobs']=args.jobs
    log=(root/'logs/experiments.log').open('w',buffering=1)
    def note(message):
        print(message,flush=True)
        print(message,file=log,flush=True)
    started=perf_counter()
    X,y,split,resamples,seed=prepare(root,config)
    write_json(root/'configs/resolved.json',config)
    metadata={'seed':seed,'seed_is_provisional':not bool(config.get('roll_number')),
              'student_name':config.get('student_name'),'roll_number':config.get('roll_number'),
              'repository_url':config.get('repository_url'),'authorship':'AI-generated implementation and report',
              'final_commit_hash':config.get('final_commit_hash'),
              'environment':{p:importlib.metadata.version(p) for p in ['numpy','scipy','scikit-learn','matplotlib','joblib','reportlab','pypdf']},
              'python':sys.version.split()[0],'platform':platform.platform(),'machine':platform.machine(),
              'n_jobs':config['n_jobs'],'split_counts':{k:len(v) for k,v in split.items()},
              'dataset_sha256':hashlib.sha256((root/'data/olivetti.npz').read_bytes()).hexdigest(),
              'refit_randomness':'Model RNG streams held fixed across repetitions; resampled training images vary.',
              'diagnostic_split':'clean validation','confidence_conventions':{'tree':'leaf class probability','bagging':'hard vote fraction','boosting':'normalized alpha-weighted hard vote fraction'}}
    summary={'metadata':metadata}
    note(f'Data prepared: seed={seed}, train=240, validation=80, test=80; provisional={metadata["seed_is_provisional"]}')

    trees,rows=[],[]
    for depth in config['tree_depths']:
        for leaf in config['minimum_leaf_sizes']:
            t0=perf_counter()
            model=DecisionTreeClassifier(max_depth=depth,min_samples_leaf=leaf,random_state=child_seed(seed,11)).fit(X[split['train']],y[split['train']])
            row={'depth':depth,'min_samples_leaf':leaf,'fit_seconds':perf_counter()-t0,
                 'train_accuracy':accuracy(model.predict_proba(X[split['train']]),y[split['train']]),
                 'validation_accuracy':accuracy(model.predict_proba(X[split['validation']]),y[split['validation']]),**leaf_stats(model)}
            trees.append(model); rows.append(row)
    winner=min(range(len(rows)),key=lambda i:(-rows[i]['validation_accuracy'],rows[i]['depth'] if rows[i]['depth'] is not None else float('inf'),-rows[i]['min_samples_leaf']))
    tree=trees[winner]
    selected={'tree':{k:rows[winner][k] for k in ['depth','min_samples_leaf']}}
    for row,model in zip(rows,trees):
        row['test_accuracy']=accuracy(model.predict_proba(X[split['test']]),y[split['test']])
    summary['tree_search']=rows
    summary['selected']=selected
    write_json(root/'results/tree_search.json',rows)
    note(f'Baseline selected by validation: {selected["tree"]}')

    bag=Bagging(max(config['bagging_sizes']),max_depth=selected['tree']['depth'],min_samples_leaf=selected['tree']['min_samples_leaf'],random_state=child_seed(seed,22)).fit(X[split['train']],y[split['train']])
    bag_rows=[]
    for B in config['bagging_sizes']:
        part=prefix(bag,B)
        oob,covered=part.oob_predict_proba(X[split['train']])
        pval=part.predict_proba(X[split['validation']])
        bag_rows.append({'B':B,**split_scores(part,X,y,split),'unique_mean':float(np.mean([len(np.unique(i)) for i in part.bootstrap_indices_])),
                         'unique_theory':float(240*(1-(239/240)**240)),'oob_accuracy':accuracy(oob[covered],y[split['train']][covered]) if covered.any() else None,
                         'oob_coverage':float(covered.mean()),'oob_covered_examples':int(covered.sum()),
                         'disagreement':disagreement(np.asarray([t.predict(X[split['validation']]) for t in part.estimators_])),
                         'margin':mean_margin(pval),'fit_seconds':part.fit_times_[-1]})
    chosen=min(bag_rows,key=lambda r:(-r['validation_accuracy'],r['B']))
    selected['bagging']={'B':chosen['B']}
    final_bag=prefix(bag,chosen['B'])
    summary['bagging_search']=bag_rows
    write_json(root/'results/bagging_search.json',bag_rows)
    note(f'Bagging selected: B={chosen["B"]}, validation={chosen["validation_accuracy"]:.4f}')

    toy_X=np.arange(9,dtype=float).reshape(-1,1)
    toy_y=np.array([0,0,1,1,2,2,0,1,2])
    toy=SAMME(1,max_depth=1,random_state=child_seed(seed,44)).fit(toy_X,toy_y)
    summary['toy']={'X':toy_X,'y':toy_y,'weights_before':toy.sample_weight_history_[0],
                    'predictions':toy.training_predictions_[0],'weighted_error':toy.history_[0]['weighted_error'],
                    'alpha':toy.estimator_weights_[0],'weights_after':toy.sample_weight_history_[1]}
    write_json(root/'results/toy_trace.json',summary['toy'])
    boost_models,boost_rows={},[]
    for depth in config['boosting_depths']:
        try:
            booster=SAMME(max(config['boosting_rounds']),max_depth=depth,random_state=child_seed(seed,33)).fit(X[split['train']],y[split['train']])
        except ValueError as exc:
            note(f'Invalid boosting candidate depth={depth}: {exc}')
            continue
        boost_models[depth]=booster
        for rounds in config['boosting_rounds']:
            part=prefix(booster,rounds)
            boost_rows.append({'depth':depth,'rounds':rounds,'actual_rounds':len(part.estimators_),**split_scores(part,X,y,split),
                               'fit_seconds':part.fit_times_[-1], 'termination':booster.termination_reason_ if len(part.estimators_)<rounds else 'requested_rounds'})
        note(f'Boosting depth={depth}, accepted={len(booster.estimators_)}')
    chosen_boost=min(boost_rows,key=lambda r:(-r['validation_accuracy'],r['depth'],r['rounds']))
    selected['boosting']={k:chosen_boost[k] for k in ['depth','rounds','actual_rounds']}
    final_boost=prefix(boost_models[chosen_boost['depth']],chosen_boost['rounds'])
    summary['boosting_search']=boost_rows
    summary['boosting_history']=final_boost.history_
    write_json(root/'results/boosting_search.json',boost_rows)
    write_json(root/'results/boosting_history.json',final_boost.history_)
    note(f'Boosting selected: {selected["boosting"]}')

    matched=[]
    for B in [25,50,100]:
        for name,model in [('bagging',Bagging(B,max_depth=selected['tree']['depth'],min_samples_leaf=selected['tree']['min_samples_leaf'],random_state=child_seed(seed,22))),
                           ('boosting',SAMME(B,max_depth=selected['boosting']['depth'],random_state=child_seed(seed,33)))]:
            t0=perf_counter()
            model.fit(X[split['train']],y[split['train']])
            elapsed=perf_counter()-t0
            matched.append({'model':name,'n_learners':B,'actual_learners':len(model.estimators_),**split_scores(model,X,y,split),'fit_seconds':elapsed})
    summary['matched']=matched
    write_json(root/'results/matched_comparison.json',matched)

    models={'tree':tree,'bagging':final_bag,'boosting':final_boost}
    predictions={}
    robustness=[]
    test=split['test']; truth=y[test]
    for name,model in models.items():
        clean=model.predict_proba(X[test]); masked=model.predict_proba(mask_faces(X[test]))
        predictions[name+'_clean']=clean; predictions[name+'_masked']=masked
        robustness.append({'model':name,'clean_accuracy':accuracy(clean,truth),'masked_accuracy':accuracy(masked,truth),
                           'clean_confidence':float(clean.max(axis=1).mean()),'masked_confidence':float(masked.max(axis=1).mean()),
                           'accuracy_drop':accuracy(clean,truth)-accuracy(masked,truth)})
    predictions['boost_train_predictions']=np.asarray(final_boost.training_predictions_)
    predictions['boost_sample_weights']=np.asarray(final_boost.sample_weight_history_)
    cases={'q1':[]}
    used=set()
    availability={}
    for category in ['correction','both_wrong']:
        available=[]
        for condition in ['clean','masked']:
            tp=predictions['tree_'+condition].argmax(axis=1); bp=predictions['bagging_'+condition].argmax(axis=1)
            eligible=(tp!=truth)&((bp==truth) if category=='correction' else (bp!=truth))
            for pos in np.flatnonzero(eligible):
                available.append((condition,int(pos)))
        availability[category]=len(set(int(test[p]) for _,p in available))
        for condition,pos in available:
            if int(test[pos]) in used:
                continue
            used.add(int(test[pos]))
            cases['q1'].append({'split':condition,'position':pos,'index':int(test[pos]),'category':category,'truth':int(truth[pos]),
                                'tree_prediction':int(predictions['tree_'+condition][pos].argmax()),'bagging_prediction':int(predictions['bagging_'+condition][pos].argmax()),
                                'tree_confidence':float(predictions['tree_'+condition][pos].max()),
                                'vote_counts':np.rint(predictions['bagging_'+condition][pos]*selected['bagging']['B']).astype(int)})
            if sum(c['category']==category for c in cases['q1'])==2:
                break
    cases['q1_availability']=availability
    weights=predictions['boost_sample_weights']; weak=predictions['boost_train_predictions']
    high=int(np.argmax(np.max(weights[1:],axis=0)))
    cases['high_weight']={'train_position':high,'index':int(split['train'][high]),'truth':int(y[split['train'][high]]),
                         'peak_weight':float(weights[:,high].max()),
                         'history':[{'round':i+1,'weight_before':float(weights[i,high]),'weight_after':float(weights[i+1,high]),
                                     'prediction':int(weak[i,high]),'misclassified':bool(weak[i,high]!=y[split['train'][high]])} for i in range(len(weak))]}
    bc=predictions['bagging_clean'].argmax(axis=1); bm=predictions['bagging_masked'].argmax(axis=1)
    ac=predictions['boosting_clean'].argmax(axis=1); am=predictions['boosting_masked'].argmax(axis=1)
    differential=((bc!=bm)!=(ac!=am))
    candidates=np.flatnonzero(differential)
    rule='One ensemble changes class under masking while the other does not.'
    if not len(candidates):
        candidates=np.arange(len(test)); rule='Fallback: largest difference in confidence response; no asymmetric class-change case exists.'
    diff=np.abs((predictions['bagging_clean'].max(axis=1)-predictions['bagging_masked'].max(axis=1))-(predictions['boosting_clean'].max(axis=1)-predictions['boosting_masked'].max(axis=1)))
    pos=int(candidates[np.argmax(diff[candidates])])
    cases['q2']={'position':pos,'index':int(test[pos]),'truth':int(truth[pos]),'selection_rule':rule,
                 **{name+'_'+condition+'_prediction':int(predictions[name+'_'+condition][pos].argmax()) for name in ['bagging','boosting'] for condition in ['clean','masked']},
                 **{name+'_'+condition+'_confidence':float(predictions[name+'_'+condition][pos].max()) for name in ['bagging','boosting'] for condition in ['clean','masked']}}
    summary['cases']=cases;summary['robustness']=robustness
    write_json(root/'results/cases.json',cases)
    write_json(root/'results/robustness.json',robustness)
    joblib.dump(models,root/'models/selected.joblib',compress=3)
    np.savez_compressed(root/'results/predictions.npz',**predictions)
    write_json(root/'results/summary.json',summary)
    note('Original-model comparisons and all case studies saved. Beginning paired resampling study.')

    chosen_depth=selected['tree']['depth']
    depths=([2,8,None] if chosen_depth is None else [1 if chosen_depth==2 else 2,chosen_depth,None])
    depths=list(dict.fromkeys(depths))
    source_hashes={name:hashlib.sha256((source_root/'task99'/name).read_bytes()).hexdigest()
                   for name in ['models.py','data.py','metrics.py','experiments.py']}
    fingerprint=hashlib.sha256(json.dumps({'config':config,'selected':selected,'schema':2,
                                         'dataset':metadata['dataset_sha256'],
                                         'resamples':hashlib.sha256(resamples.tobytes()).hexdigest(),
                                         'sources':source_hashes},sort_keys=True).encode()).hexdigest()
    repeats=[]
    with parallel_config(backend='loky',inner_max_num_threads=1):
        generator=Parallel(n_jobs=config['n_jobs'],return_as='generator_unordered')(
            delayed(resample_worker)(r,indices,X,y,split,selected,depths,seed,str(root),fingerprint) for r,indices in enumerate(resamples))
        for record in generator:
            repeats.append(record)
            note(f'Resampling complete: repetition {record[0]+1}/{len(resamples)} ({len(repeats)} finished)')
    repeats.sort(key=lambda x:x[0])
    bv=[]
    for name in models:
        array=np.asarray([r[2][name] for r in repeats]); predictions['q3_'+name]=array
        statistics=bias_variance(array,truth)
        extra={key:float(np.mean([r[1]['selected_stats'][name][key] for r in repeats])) for key in ['n_leaves','mean_leaf_samples','singleton_fraction']}
        dis=[r[1]['selected_stats'][name]['disagreement'] for r in repeats]
        bv.append({'model':name,**statistics,**extra,'disagreement':float(np.mean(dis)) if dis[0] is not None else None})
    depth_rows=[]
    for depth in depths:
        label='unlimited' if depth is None else str(depth)
        for name in ['tree','bagging']:
            key=f'depth_{label}_{name}'
            array=np.asarray([r[2][key] for r in repeats]);predictions['q3_'+key]=array
            details=[next(x for x in r[1]['depth_records'] if x['depth']==depth and x['model']==name) for r in repeats]
            extra={key:float(np.mean([r[key] for r in details])) for key in ['n_leaves','mean_leaf_samples','singleton_fraction']}
            depth_rows.append({'depth':depth,'model':name,**bias_variance(array,truth),**extra,
                               'disagreement':float(np.mean([r['disagreement'] for r in details])) if name=='bagging' else None})
    summary['biasvariance']=bv;summary['depth_study']=depth_rows
    summary['refit_accuracy_records']=[{'repetition':r[0],**r[1]['accuracies'],'boost_actual_rounds':r[1]['boost_actual_rounds']} for r in repeats]
    summary['final_table']=[{'model':r['model'],'clean_accuracy':r['clean_accuracy'],'masked_accuracy':r['masked_accuracy'],
                             **{k:next(x for x in bv if x['model']==r['model'])[k] for k in ['bias_squared','variance']}} for r in robustness]
    metadata['total_experiment_wall_seconds']=perf_counter()-started
    metadata['run_fingerprint']=fingerprint
    np.savez_compressed(root/'results/predictions.npz',**predictions)
    for name in ['biasvariance','depth_study','final_table','refit_accuracy_records']:
        write_json(root/f'results/{name}.json',summary[name])
    write_json(root/'results/summary.json',summary)
    note(f'ALL EXPERIMENTS COMPLETE in {metadata["total_experiment_wall_seconds"]:.1f} seconds')
    log.close()


if __name__=='__main__':
    main()
