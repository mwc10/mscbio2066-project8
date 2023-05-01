#!/usr/bin/env python3

# import polars as pl
import pandas as pd
import numpy as np
import numpy.typing as npt
import xgboost as xgb
from rdkit.Chem import AllChem
from Bio import Align
import requests
from requests.adapters import HTTPAdapter, Retry
from scipy.stats import spearmanr

import re
import json
import heapq
import tarfile
import multiprocessing
import urllib.parse
from pathlib import Path
from typing import IO, Dict, List, Optional, Sequence, Set, Tuple, Union
from itertools import cycle

class ModelEnsemble:
    def __init__(self,
        params: Sequence[Tuple[str, int, int, float]],
        cores=1,
        scores: Optional[Dict[str, List[str]]]=None,
        verbose=True,
    ) -> None:
        ''' Create an ensemble XGBoost tree kinase forset models '''
        def create_model(m):
            file, k, cmin, rmin = m
            with fsspec.open(file) as tar:
                return Model(tar, cores, scores=scores, min_cmpds=cmin, metric_val=rmin, similar_k=k, verbose=verbose)
        
        self.aligner=Align.PairwiseAligner(scoring='blastp')
        self.cores=cores
        self.verbose=verbose
        self.ensemble=list(map(create_model, params))

    def predict(self, 
        smiles: Sequence[str], 
        uniprot: Sequence[str]
    ) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        ''' returns ensemble prediction and matrix of each model's prediction '''

        # preprocess any novel/untrained uniprot ids
        uids = set(uniprot)
        self.prefetch_seqs(uids)

        # run predictions for each model in the ensemble, and average
        predictions = np.array([m.predict_batch(smiles, uniprot) for m in self.ensemble], dtype=float)
        avg_predict = np.mean(predictions, axis=0)

        return avg_predict, predictions

    def prefetch_seqs(self, uids: Set[str]):
        '''
        Each model in the ensemble may have a different set of trained kinases.
        But, the method for adding an inferred kinase requires the same thing, so
        this function batches the sequence acquistion and scoring for all novel kinases
        across all models in the ensemble.
        '''

        novel_uids = set.union(*[m.missing_boosters(uids) for m in self.ensemble])
        if len(novel_uids) == 0:
            return None
        
        if self.verbose:
            print('found', len(novel_uids), 'novel UniProt IDs')
        novel_seqs = UniProtQuery(novel_uids).get(verbose=self.verbose)
        if self.verbose:
            print('fetched', len(novel_seqs), 'novel sequences from uniprot')
        
        queries = {}
        for m in self.ensemble:
            for uid, info in m.protModels.items():
                if info['direct'] and uid not in queries:
                    queries[uid] = info['seq']
        if self.verbose:
            print('collected sequences from all trained kinase XGB models')
        

        with multiprocessing.Pool(self.cores) as p:
            itr = p.imap_unordered(
                _find_uid_nn, 
                zip(novel_seqs.items(), cycle([queries]), cycle([self.aligner]), cycle([self.verbose]))
            )
            novel_scores = dict(itr)
        
        if self.verbose: print(len(novel_scores), 'scored novel sequences')

        for m in self.ensemble:
            m.use_precalced(novel_scores)

def _find_uid_nn(args) -> Tuple[str, List[str]]:
    ''' 
    Args: target: (ID, Seq), 
        queries: Map<ID, Seq>, 
        aligner: PairwiseAligner, 
        verbose: bool
    Returns: ranked list of the nearest neighbors for `target` in `queries` using `aligner` 
    Top-level function for multiprocessing
    '''
    (uid, target), queries, aligner, verbose = args
    scores = [(quid, aligner.score(target, query)) for quid, query in queries.items()]
    scores.sort(reverse=True, key=lambda x: x[1])
    scores = [x[0] for x in scores]

    if verbose: print('scored', uid, '\t', scores[:5])

    return uid, scores
    
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
            def ubj_bytes(p):
                ''' get the decompressed UBJ bytes from the tar.gz file '''
                return bytearray(tf.extractfile(p).read())
            def configure(tup):
                uid, model_path, seq = tup
                value = {
                    'models': [load_model(ubj_bytes(model_path))],
                    'seq': seq,
                    'direct': True
                }
                return (uid, value)
            
            # read model information csv from tar.gz
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
            
            # load models into XGBoost with supporting information from info.csv
            # and save model config.json into class
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

        # save metric and compound filtering info to the model's config dictionary
        metrics = {
            'min_cmpds': min_cmpds, 
            'metric': metric,
            'op': metric_op,
            'val': metric_val,
            }
        self.config['metrics'] = metrics

    def predict(self, smiles: Union[str, npt.NDArray], uniprot: str) -> float:
        if type(smiles) is str:
            fp = self.fingerprint(smiles).reshape(1, -1)
        else:
            fp = smiles

        if uniprot not in self.protModels:
            if self.verbose: print(f'novel kinase target <{uniprot}>; fetching info from uniprot')
            # a map of one UniProt to AA sequence
            seqs = UniProtQuery([uniprot]).get()
            uid, seq = next(iter(seqs.items()))
            self.add_booster(uid, seq)

        return np.mean([m.predict(fp) for m in self.protModels[uniprot]['models']])

    def predict_batch(self, 
        smiles: Sequence[str], 
        uniprot: Sequence[str], 
    ) -> List[float]:
        ''' Predict a collection of [SMILES], [UniProt ID] at one time. 
            Reduces API calls to UniProt if there are IDs without predictors
        '''

        unique_smiles = set(smiles)
        novel_uniprot = set(uniprot).difference(set(self.protModels.keys()))
        if self.verbose:
            print('unique smiles', len(unique_smiles), '/', len(smiles))
            print('novel uniprot', len(novel_uniprot), '/', len(set(uniprot)), '/', len(uniprot))

        if len(novel_uniprot) > 0:
            novel_seqs = UniProtQuery(novel_uniprot).get(verbose=self.verbose)
            if self.verbose: 
                print('got novel sequences from uniprot API')
            
            # Ideally this would all be done with multiprocessing...
            # but it doesn't seem to work on the grader...
            for uid, seq in novel_seqs.items():
                self.add_booster(uid, seq)

            if self.verbose: 
                print('found similar models for unique sequences')

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

    def missing_boosters(self, uniprots: Sequence[str]) -> Set[str]:
        ''' Check `uniprots` to find any UniProt IDs that do not have a booster '''
        return {uid for uid in uniprots if uid not in self.protModels}
    
    def add_booster(self, uid: str, seq: str):
        ''' Create an inferred booster for `uid` based on similarity of `seq` to XGBoost models '''
        models = self.get_similar(seq)
        self.protModels[uid] = {'models': models, 'seq': seq, 'direct': False}

    def get_similar(self, targetSeq: str) -> List[xgb.XGBRegressor]:
        # use only models that were trained, for now...
        only_trained = lambda x: x['direct']
        score_seq = lambda x: (self.aligner.score(targetSeq, x['seq']), x['models'][0])
        itr = map(score_seq, filter(only_trained, self.protModels.values()))

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
            # otherwise, pick the topK directly trained models that are closest to this one
            models = []
            for nnid in nns:
                if nnid in self.protModels and self.protModels[nnid]['direct']:
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
        c = self.config
        fp = c['fingerprints']
        C = 'C' if fp['useChirality'] else ''
        F = 'F' if fp['useFeatures'] else ''
        CON = '+' if C or F else ''
        b, i = self.describe()
        acts = ', '.join(c['activityTypes'])

        return f"Model <{fp['bitSize']}b ECFP{fp['radius']}{CON}{C}{F} over {acts}>: {b} trees ({i} inferred)"


class UniProtQuery():
    '''
        Use the paginated search for UniProt, 
        as the stream API errors unpredictibly yet frequently.

        Adapted from: https://www.uniprot.org/help/api_queries
    '''
    def __init__(self, uids: Sequence[str]):
        self.url = 'https://rest.uniprot.org/uniprotkb/search'
        self.params = {
            'query': " OR ".join(f'(accession:{uid})' for uid in uids),
            'fields': 'accession,sequence',
            'format': 'tsv',
            'size': 500,
        }
        self.reNextLink = re.compile(r'<(.+)>; rel="next"')
        self.retries = Retry(total=5, backoff_factor=0.25, status_forcelist=[500, 502, 503, 504])
    
    def get_next_link(self, headers) -> Optional[str]:
        if 'Link' in headers:
            match = self.reNextLink(headers['Link'])
            if match:
                return match.group(1)
        return None

    def get_batch(self, session: requests.Session):
        batch_url = self.url + '?' + urllib.parse.urlencode(self.params)
        while batch_url:
            res = session.get(batch_url)
            res.raise_for_status()
            total = res.headers['x-total-results']
            yield res, total
            batch_url = self.get_next_link(res.headers)

    def get(self, verbose=False) -> Dict[str, str]:
        ''' Returns: Map<UniProt ID, Seq> '''
        with requests.Session() as s:
            s.mount('https://', HTTPAdapter(max_retries=self.retries))
            output = []
            for batch, total in self.get_batch(s):
                output.extend(map(lambda l: l.strip().split('\t'), batch.text.splitlines()[1:]))
                if verbose: print(len(output), '/', total)

            return {uid: seq for uid, seq in output}


################# CLI Interface for Predictions #####################
if __name__ == '__main__':
    import argparse
    import fsspec
    import os

    def cli():
        p = argparse.ArgumentParser('Run model in `tarfile` on `datafile`')
        p.add_argument('input', help='input columnar data file with SMILES and UniProt IDs')
        p.add_argument('output', nargs='?', default=None, help="save predictions to space-separated text file if present")
        p.add_argument('--tarfile', '-t', action='append', help='tar containing model info (Path,[K],[MinCmpds],[MinR2])' )
        p.add_argument('--eval', '-e', action='store_true')
        p.add_argument('--scores', '-s', default=None, help='precalculated kinase similarities')
        return p.parse_args()
    
    def try_none(x, idx, wrap=int):
        try:
            return wrap(x[idx])
        except:
            return None

    def extract_model_params(s):
        ''' Parse CLI string "ModelPath,Option<K>,Option<MinCmpd>,Option<MinR2>" '''
        parts = s.split(',')
        n = parts[0]
        k = try_none(parts, 1)
        c = try_none(parts, 2)
        r = try_none(parts, 3, float)

        return n, k, c, r  

    def get_num_cores():
        try:
            return len(os.sched_getaffinity(0))
        except:
            return multiprocessing.cpu_count()

    def create_model(scores, m):
        file, k, cmin, rmin = m
        with fsspec.open(file) as tar:
            return Model(tar, cores, scores=scores, min_cmpds=cmin, metric_val=rmin, similar_k=k)

    DEFAULT_MODELS = [
        # model, k, min_compounds, min_r2
        ('gs://mwc10-mscbio2066/model_hu-fp2048r1-KDKIEC50IC50.tar.gz', 3, 30, -0.5), 
        ('gs://mwc10-mscbio2066/model_hu-fp2048r1-KDKIEC50IC50-CF.tar.gz', 1, 30, -0.5), 
        ('gs://mwc10-mscbio2066/model_hu-fp2048r2-KDKIEC50IC50-CF.tar.gz', 3, 30, -0.5),
        ('gs://mwc10-mscbio2066/model_hu-fp2048r2-KDKI.tar.gz', 5, 15, -0.5),
        ('gs://mwc10-mscbio2066/model_hu-fp2048r2-KDKI-CF.tar.gz', 5, 15, -0.5), 
    ]
    
    args = cli()
    cores = get_num_cores()
    modelSettings = list(map(extract_model_params, args.tarfile)) if args.tarfile else DEFAULT_MODELS

    print(modelSettings)

    df = pd.read_csv(args.input, sep=' ').dropna(axis='columns', how='all')

    # use pre-calced similarity scores to speed up local testing
    if args.scores:
        # too lazy to rewrite this to use pandas..
        import polars as pl
        dfScores = pl.read_parquet(args.scores)
        scores = {uid: nn for uid, _, nn in zip(*tuple(dfScores.to_dict().values()))}
        print('parsed pre-calculated kinase similarity scores')
    else:
        scores = None

    print('creating ensemble model')
    ensemble = ModelEnsemble(modelSettings, cores, scores)
    avg_predict, predictions = ensemble.predict(df['SMILES'], df['UniProt'])

    df['Predicted'] = pd.Series(avg_predict)
    if args.output is not None:
        df.to_csv(args.output, sep=' ', float_format='%.4f', index=False)
        print('saved predictions to ', args.output)

    if 'pKd' in df and args.eval:
        for m, pred in zip(ensemble.ensemble, predictions):
            res = spearmanr(pred, df['pKd'])
            print(m)
            print('\t',res) 

        res = spearmanr(avg_predict, df['pKd'])
        corr, p = res.statistic, res.pvalue
        print(f'Correlation with labels in input:\t{corr:.4f} +/- {p:.4f}')
    
