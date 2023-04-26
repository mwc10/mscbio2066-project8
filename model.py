#!/usr/bin/env python3

# import polars as pl
import pandas as pd
import numpy as np
import numpy.typing as npt
import xgboost as xgb
from rdkit.Chem import AllChem
from Bio import Align
import requests
from scipy.stats import spearmanr

import json
import heapq
import tarfile
import multiprocessing
from pathlib import Path
from typing import IO, Dict, List, Optional, Sequence, Tuple, Union

class Model:
    def __init__(self,
        tar: Union[str, Path, IO[bytes]],
        cores: int=1,
        similar_k: int=3,
        scores: Optional[Dict[str, List[str]]]=None,
        metric: str='r2',
        metric_op: str='min',
        metric_val: Optional[float]=None,
        min_cmpds: Optional[int]=None,
        verbose=True,
    ) -> None:
        def load_model(b: bytearray):
            r = xgb.XGBRegressor(random_state=42)
            r.load_model(b)
            r.set_params(n_jobs=cores)
            return r
        
        # Check tar to see if path, otherwise fileobject
        if type(tar) is str or isinstance(tar, Path):
            openParams = {'name': tar, 'mode':'r'}
        else:
            openParams = {'fileobj': tar, 'mode':'r'}

        # load the kinase specific XGB trees and sequences from the tar archive
        # there is an 'info.csv' that map uniprot ids to trained models and the full sequence
        # a 'config.json' with basic info about the model and the fingerprinting parameters
        # finally, there is a folder with XGBoost .ubj files for each uniprot id
        with tarfile.open(**openParams) as tf:
            ubj_bytes = lambda p: bytearray(tf.extractfile(p).read())
            configure = lambda tup: (tup[0], {'models': [load_model(ubj_bytes(tup[1]))], 'seq': tup[2], 'direct': True})
            df = pd.read_csv(tf.extractfile('info.csv'))

            # filter trees by training metrics or by number of compounds
            if min_cmpds is not None or metric_val is not None:
                match (metric_op, metric_val):
                    case (_, None): qm = None
                    case ('min', _): qm = f'{metric} >= {metric_val}'
                    case ('max', _): qm = f'{metric} <= {metric_val}'
                
                qc = f'num_compounds >= {min_cmpds}' if min_cmpds is not None else None
                q = ' and '.join(filter(lambda x: x is not None, [qm, qc]))
                df = df.query(q)
            
            itr = zip(df['uniprot'], df['output_model'], df['sequence'])

            self.protModels = dict(map(configure, itr))
            self.config = json.load(tf.extractfile('config.json'))
        
        # store basic configuration parameters
        self.verbose = verbose
        self.cores = cores
        self.topK = similar_k

        # use the pre-configured blastp aligning scoring
        # note that this has to match for pre-calculated scores
        self.aligner = Align.PairwiseAligner(scoring='blastp')
        
        # use pre-processed sequence similarity scores to find most similar models for each uniprot id
        if scores is not None:
            self.use_precalced(scores)

        # save metric and compount filtering info to the model's config dictionary
        if min_cmpds is not None or metric_val is not None:
            metrics = {
                'min_cmpds': min_cmpds, 
                'metric': metric,
                'op': metric_op,
                'val': metric_val,
                }
            self.config['metrics'] = metrics
        else:
            self.config['metrics'] = None


    def predict(self, smiles: Union[str, npt.NDArray], uniprot: str) -> float:
        if type(smiles) is str:
            fp = self.fingerprint(smiles).reshape(1, -1)
        else:
            fp = smiles

        if uniprot not in self.protModels:
            if self.verbose: print('novel kinase target; fetching info from uniprot')
            seqs = self.fetch_seqs([uniprot])
            for uid, seq in seqs.items():
                models = self.get_similar(seq)
                self.protModels[uid] = {'models': models, 'seq': seq, 'direct': False}

        return np.mean([m.predict(fp) for m in self.protModels[uniprot]['models']])

    def predict_batch(self, smiles: Sequence[str], uniprot: Sequence[str]) -> List[float]:
        unique_smiles = set(smiles)
        novel_uniprot = set(uniprot).difference(set(self.protModels.keys()))
        if self.verbose:
            print('total unique smiles', len(unique_smiles), '/', len(smiles))
            print('total novel uniprot', len(novel_uniprot), '/', len(set(uniprot)), '/', len(uniprot))

        if len(novel_uniprot) > 0:
            novel_seqs = self.fetch_seqs(novel_uniprot)
            if self.verbose: print('got novel sequences from uniprot API')
            # Ideally this would all be done with multiprocessing...
            # but it doesn't seem to work on the grader...
            # with multiprocessing.Pool(self.cores) as p:
            #     models = p.map(self.get_similar, novel_seqs.values())
            #     for (uid, seq), m in zip(novel_seqs.items(), models):
            #         self.protModels[uid] = {'models': m, 'seq': seq}

            for uid, seq in novel_seqs.items():
                models = self.get_similar(seq)
                self.protModels[uid] = {'models': models, 'seq': seq, 'direct': False}
            if self.verbose: print('found similar models for unique sequences')

        fps = {smi: self.fingerprint(smi).reshape(1, -1) for smi in unique_smiles}
        fps = [fps[smi] for smi in smiles]
        if self.verbose: print('done fingerprinting')
        return [self.predict(fp, uid) for fp, uid in zip(fps, uniprot)]
    
    def fingerprint(self, smiles: str) -> npt.NDArray[np.uint8]:
        mol = AllChem.MolFromSmiles(smiles)
        c = self.config['fingerprints']
        fp = AllChem.GetMorganFingerprintAsBitVect(
            mol,
            radius=c['radius'],
            nBits=c['bitSize'],
            useFeatures=c['useFeatures'],
            useChirality=c['useChirality']
        )

        return np.array(fp, dtype=np.uint8)

    def fetch_seqs(self, uids: Sequence[str]) -> Dict[str, str]:
        url = 'https://rest.uniprot.org/uniprotkb/stream'
        query = " OR ".join(f'(accession:{uid})' for uid in uids)
        params = {
            'query': query,
            'fields': 'accession,sequence',
            'format': 'tsv',
        }
        res = requests.get(url, params=params)
        it = map(lambda l: l.strip().split('\t'), res.text.splitlines()[1:])
        return {uid: seq for uid, seq in it}

    def get_similar(self, targetSeq: str) -> List[xgb.XGBRegressor]:
        # use only models that were trained, for now...
        only_singles = lambda x: len(x['models']) == 1
        score_seq = lambda x: (self.aligner.score(targetSeq, x['seq']), x['models'][0])
        itr = map(score_seq, filter(only_singles, self.protModels.values()))

        return [x[1] for x in heapq.nlargest(self.topK, itr, key=lambda x: x[0])]

    def use_precalced(self, scores):
        ''' For testing, use pre-calculated similarities of all test seqs 
            against all trained seqs. 
            The selection of most similar models has to be done dynamically,
            as the current forests in the model changes based on metric filter criteria
        '''
        for uid, nns in scores.items():
            # there is an XGBoost tree for this kinase
            if uid in self.protModels: continue;
            # otherwise, pick the topK models that are closest to this one
            models = []
            for nnid in nns:
                if nnid in self.protModels and len(self.protModels[nnid]['models']) == 1:
                    models.append(self.protModels[nnid]['models'][0])
                if len(models) >= self.topK: break;

            self.protModels[uid] = {'models': models, 'seq': None, 'direct': False}
    
    def describe(self) -> Tuple[int, int]:
        ''' return the number of kinase boosters and inferred kinases '''
        boosters, inferred = 0, 0
        for info in self.protModels.values():
            if info['direct']:
                boosters += 1
            else:
                inferred += 1
        
        return boosters, inferred

    def __str__(self):
        return f'''\
        '''


if __name__ == '__main__':
    import argparse
    import fsspec
    import os

    def cli():
        p = argparse.ArgumentParser('Run model in `tarfile` on `datafile`')
        p.add_argument('input', help='input columnar data file with SMILES and UniProt IDs')
        p.add_argument('output', nargs='?', default=None, help="save predictions to space-separated text file if present")
        p.add_argument('--tarfile', '-t', default='gs://mwc10-mscbio2066/model_hu-fp2048r3-KDKI.tar.gz', help='tar containing model info')
        p.add_argument('--eval', '-e', action='store_true')
        p.add_argument('--cmin', '-c', type=int, default=16, help='min amount of compounds used to train a kinase GBF')
        p.add_argument('--rmin', '-r', type=float, default=None)
        p.add_argument('-k', type=int, default=3, help='Infer from K models if kinase not directly trained')
        p.add_argument('--scores', '-s', default=None, help='precalculated kinase similiarities')
        return p.parse_args()
    
    def get_num_cores():
        try:
            return len(os.sched_getaffinity(0))
        except:
            return multiprocessing.cpu_count()
    
    args = cli()
    cores = get_num_cores()

    df = pd.read_csv(args.input, sep=' ').dropna(axis='columns', how='all')

    # use pre-calced similarity scores to speed up local testing
    if args.scores:
        # too lazy to rewrite this to use pandas..
        import polars as pl
        dfScores = pl.read_parquet(args.scores)
        scores = {uid: nn for uid, _, nn in zip(*tuple(dfScores.to_dict().values()))}
        print('parsed pre-calculated scores')
    else:
        scores = None
    
    print(f'creating model for {args.tarfile}')
    with fsspec.open(args.tarfile) as tar:
        m = Model(tar, cores, scores=scores, min_cmpds=args.cmin, metric_val=args.rmin, similar_k=args.k)
    
    predictions = m.predict_batch(df['SMILES'], df['UniProt'])
    df['Predicted'] = pd.Series(np.array(predictions), dtype=float)
    # test = df.with_columns([pl.Series('Predict pKd', predictions, dtype=pl.Float64)])
    if args.output is not None:
        df.to_csv(args.output, sep=' ', float_format='%.4f', index=False)
        print('saved predictions to ', args.output)

    if 'pKd' in df and args.eval:
        print(df)
        res = spearmanr(df['Predicted'], df['pKd'])
        corr, p = res.statistics, res.pvalue
        print(f'Correlation with labels in input:\t{corr:.4f} +/- {p:.4f}')
    
    print(m.describe())
    print(m.config)
