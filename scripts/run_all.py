#!/usr/bin/env python3
"""Run tests, all experiments, evidence enrichment, figures, and report."""
from pathlib import Path
import argparse
import shutil
import subprocess
import sys


def remove_intermediates(root):
    """Keep consolidated results after all plots and report sources are built."""
    result_names = ['tree_search', 'bagging_search', 'boosting_search',
                    'boosting_history', 'matched_comparison', 'matched',
                    'robustness', 'biasvariance', 'depth_study', 'depth_effects',
                    'final_table', 'refit_accuracy_records', 'toy_trace',
                    'high_weight_face_history', 'cases', 'figures_manifest']
    for name in result_names:
        for suffix in ['json', 'csv']:
            (root/'results'/f'{name}.{suffix}').unlink(missing_ok=True)
    cache = root/'results/refits'
    for pattern in ['refit_*.json', 'refit_*.npz']:
        for path in cache.glob(pattern):
            path.unlink()
    if cache.exists() and not any(cache.iterdir()):
        cache.rmdir()
    (root/'data/DATASET_DESCRIPTION.txt').unlink(missing_ok=True)
    (root/'report/report.log').unlink(missing_ok=True)


def main():
    source=Path(__file__).resolve().parents[1]
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=source)
    parser.add_argument('--roll-number')
    parser.add_argument('--jobs',type=int,default=4)
    parser.add_argument('--latex',help='Path to Tectonic; detected on PATH if omitted')
    parser.add_argument('--skip-experiments',action='store_true',help='Regenerate analysis/figures/report from existing final results')
    args=parser.parse_args()
    root=args.root.resolve()
    subprocess.run([sys.executable,'-m','pytest','-q'],cwd=source,check=True)
    if not args.skip_experiments:
        command=[sys.executable,str(source/'scripts/run_experiments.py'),'--root',str(root),'--jobs',str(args.jobs)]
        if args.roll_number:
            command+=['--roll-number',args.roll_number]
        subprocess.run(command,check=True)
    for script in ['enrich_results.py','validate_results.py','make_figures.py','make_report.py']:
        subprocess.run([sys.executable,str(source/'scripts'/script),'--root',str(root)],check=True)
    latex=args.latex or shutil.which('tectonic')
    if latex:
        subprocess.run([str(Path(latex).resolve()),'--keep-logs','report.tex'],cwd=root/'report',check=True)
    else:
        print('CVPR TeX generated. To compile the official PDF: cd report && tectonic report.tex')
    remove_intermediates(root)


if __name__=='__main__':
    main()
