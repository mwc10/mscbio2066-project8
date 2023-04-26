from model import Model

import polars as pl
import numpy as np
from scipy.stats import spearmanr

import argparse, multiprocessing as mp
from itertools import cycle
from pathlib import Path

def cli():
    p = argparse.ArgumentParser()
    p.add_argument('--models', '-m', required=True, type=Path, help='Directory containing model tar.gz bundles')
    p.add_argument('--scores', '-s', required=True, type=Path, help='Parquet file with scored sequences')
    p.add_argument('--evals', '-e', required=True, type=Path, help='Directory contain evaluation text files (space separated)')
    p.add_argument('--output', '-o', default=None, type=Path, help='Output for CSV, stdout otherwise')

    return p.parse_args()

def record_model_info(m: Model, corrcoef: float):
    boosters, inferred = m.describe()
    c = m.config
    fp = c['fingerprints']
    mt = c['metrics']

    return {
        'bit_size': fp['bitSize'],
        'use_features': fp['useFeatures'],
        'use_chirality': fp['useChirality'],
        'radius': fp['radius'],
        'activities': ', '.join(c['activityTypes']),
        'metric': mt['metric'] if mt else None,
        'metric_op': mt['op'] if mt else None,
        'metric_val': mt['val'] if mt else None,
        'min_cmpds': mt['min_cmpds'] if mt else 10,
        'k': m.topK,
        'boosters': boosters,
        'inferred': inferred,
        'corrcoef': corrcoef,
    }

def run_prediction(args):
    (mtar, r2min, cmin, k), scores, evals = args
    m = Model(mtar, cores=1, scores=scores, metric_val=r2min, min_cmpds=cmin, verbose=False, similar_k=k)
    predicted = m.predict_batch(evals['SMILES'], evals['UniProt'])
    coef = spearmanr(evals['pKd'], predicted).statistic
    print(mtar.stem, k, r2min, cmin, coef, sep='\t')

    return record_model_info(m, coef)

# Constants
R2_MIN = [None, -2.0, -1.0, -0.5, -0.25, 0.0, 0.25, 0.5]
CMPD_MIN = [10, 15, 30, 60, 120, 240]
TOPK = [1, 3, 5, 10]

# main
def main(args):
    dfScores = pl.read_parquet(args.scores)
    scores = {uid: nn for uid, _, nn in zip(*tuple(dfScores.to_dict().values()))}
    evals = pl.concat(pl.scan_csv(p, separator=' ') for p in args.evals.glob('*.txt')).collect()

    # for now
    # mtar = next(args.models.glob('*.tar.gz'))
    # print(mtar)
    vrs = ((mtar, r2min, cmin, k) for k in TOPK for cmin in CMPD_MIN for r2min in R2_MIN for mtar in args.models.glob('*.tar.gz'))
    itr = zip(vrs, cycle([scores]), cycle([evals]))
    with mp.Pool() as p:
        print('in pool')
        output = list(p.imap_unordered(run_prediction, itr, chunksize=10))
    
    print('out of pool')
    df = pl.from_dicts(output)
    if args.output:
        df.write_csv(args.output)
    else:
        print(df.write_csv())

    return df

if __name__ == '__main__':
    args = cli()
    df = main(args)
