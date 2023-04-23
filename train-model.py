#!/usr/bin/env python3

import numpy as np
import polars as pl
from sklearn.model_selection import cross_val_score
import xgboost as xgb
from skopt import BayesSearchCV

import pickle
import json
import sys
import argparse
import os
import multiprocessing
from typing import Dict, Tuple, List, Optional
from pathlib import Path

MIN_COMPOUNDS = 10

def cli() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Train an XGBoost model for a kinase')
    p.add_argument('--prot_idx', '-i', type=int, required=True)
    p.add_argument('--json', '-j', required=True, help='JSON config file')
    p.add_argument('--data_dir', '-d', required=True, help='Directory containing datafiles specified in config')
    p.add_argument('--output', '-o', required=True, help='Output directory for model and info')
    p.add_argument('--force', '-f', action='store_true', help='overwrite model if output already present')
    p.add_argument('--rounds', '-r', type=int, help='Overide number of Bayesian Opt Rounds')

    return p.parse_args()

def index_kinases(tsv_path: str, idx: int) -> Optional[str]:
    ''' read Uniprot TSV file and get the Uniprot ID at line `idx` '''
    with open(tsv_path, 'rt') as f:
        for i, line in enumerate(f):
            if i == idx: return line.split('\t')[0]
    
    return None

def name_outputs(base: str, idx: int) -> Tuple[Path, Path]:
    base = Path(base)
    models = base / 'models'
    metrics = base / 'metrics'

    models.mkdir(parents=True, exist_ok=True)
    metrics.mkdir(parents=True, exist_ok=True)
    return (
        models / f'{idx:>05}.ubj',
        metrics / f'{idx:>05}.csv',
        metrics / f'cvres-{idx:>05}.csv'
    )

def print_metrics(metrics: Dict):
    print("### Metrics ###")
    width = max(map(len, metrics.keys())) + 2
    for k,v in metrics.items():
        print(f'{k:<{width}}{v}')

def get_process_cores() -> int:
    try:
        return len(os.sched_getaffinity(0))
    except:
        return multiprocessing.cpu_count()

def main(args):
    CORES = get_process_cores()
    METRICS = {}
    with open(args.json, 'r') as f:
        info = json.load(f)
    ROUNDS = args.rounds if args.rounds is not None else info['optRounds']
    IDX = args.prot_idx

    # Format file paths
    DATA_DIR = Path(args.data_dir)
    kinaseTSV = DATA_DIR / info['kinases']['file']
    dataParq = DATA_DIR / info['dataFile']
    cmpdFPs = DATA_DIR / info['fingerprints']['file']

    # Get the target kinase for this model
    targetUniprot = index_kinases(kinaseTSV, IDX)
    if targetUniprot is None:
        print(f'Index {IDX} out of range for <{kinaseTSV}>', file=sys.stderr)
        return -1
    METRICS['index'] = IDX
    METRICS['uniprot'] = targetUniprot

    # Check if model is already created
    outModel, outMetrics, outRes = name_outputs(args.output, IDX)
    if outModel.exists() and not args.force:
        print(f'Model for {targetUniprot} already exists: {outModel}')
        return 0


    # get the ChemBL IDs and Activities for this kinase
    CCID = 'compound_chembl_id'
    ACTIVITY = 'Median Activity [-logP]'
    df = (
        pl.scan_parquet(dataParq)
        .filter(pl.col('uniprot') == targetUniprot)
        .filter(pl.col('standard_type').is_in(info['activityTypes']))
        .select(CCID, ACTIVITY)
        .groupby(CCID)
        .median()
        .collect()
    )
    METRICS['num_compounds'] = df.height
    
    if df.height < MIN_COMPOUNDS:
        print(f'Too few compounds to train {targetUniprot} (#{IDX}):\n{df.height} (min {MIN_COMPOUNDS})')
        return -1

    print(f'Running for {ROUNDS} rounds for Kinase {targetUniprot} (#{IDX}) with {CORES} cores for {df.height} CMPDs')
    # Convert ChemBl IDs to precomputed fingerprints for features
    with open(cmpdFPs, 'rb') as f:
        idToFP = pickle.load(f)
    
    # Use a basic nested CV approach for baysian opt and final scoring
    X = np.array([idToFP[cid] for cid in df[CCID]])
    Y = df[ACTIVITY].to_numpy()
    # https://github.com/scikit-optimize/scikit-optimize/issues/1138#issuecomment-1467698866
    np.int = int
    xgbr = xgb.XGBRegressor(tree_method='hist', random_state=42)
    opt = BayesSearchCV(
        xgbr,
        {
            'n_estimators': (50, 600),
            'learning_rate': (0.0, 0.3),
            'gamma': (1e-2, 1e1, 'log-uniform'),
            'min_child_weight': (1, 1e3, 'log-uniform'),
            'max_depth': (1, 8),
            'max_leaves': (0, 20),
            'subsample': (0.6, 1.0, 'uniform'),
        },
        scoring='r2',
        n_iter=ROUNDS,
        cv=5,
        n_jobs=CORES,
        n_points=3,
        optimizer_kwargs={'n_initial_points':35},
        refit=False,
    )
    print('Total Iterations', opt.total_iterations)
    opt.fit(X, Y)
    for k, v in opt.best_params_.items(): METRICS[k] = v
    METRICS['bayes_best_score'] = opt.best_score_

    # Retrain final model to save it...?
    final = xgb.XGBRegressor(n_jobs=CORES, tree_method='hist', random_state=42, **opt.best_params_)
    final.fit(X, Y)
    METRICS['rmse'] = -cross_val_score(final, X, Y, scoring='neg_root_mean_squared_error').mean()
    METRICS['r2'] = cross_val_score(final, X, Y, scoring='r2').mean()
    METRICS['corrcoef'] = np.corrcoef(final.predict(X), Y).min()

    # output model based on name 
    METRICS['output_model'] = str(outModel.relative_to(args.output))
    final.save_model(outModel)
    pl.DataFrame(METRICS).write_csv(outMetrics)
    print_metrics(METRICS)
    converted = {k: v if type(v) is not np.ma.MaskedArray else v.filled().astype(float) for (k, v) in opt.cv_results_.items()}
    cvres = (
        pl.DataFrame(converted)
        .with_columns([pl.Series('uniprot', [targetUniprot]), pl.Series('num_compounds', [df.height])])
        .drop('params')
        .sort(by='rank_test_score')
    )
    cvres.write_csv(outRes)
    print(cvres.select('mean_fit_time', 'mean_test_score').head(5))

if __name__ == "__main__":
    args = cli()
    main(args)
