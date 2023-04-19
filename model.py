#!/usr/bin/env python3

from pathlib import Path
# import polars as pl
import pandas as pd
import numpy as np
import numpy.typing as npt
import xgboost as xgb
from rdkit.Chem import AllChem
from Bio import Align
import requests

import json
from typing import IO, Dict, List, Sequence, Union
import heapq
import tarfile
import multiprocessing

class Model:
    def __init__(self,
        tar: Union[str, Path, IO[bytes]],
        cores: int=1,
        similar_k: int=3
    ) -> None:
        def load_model(b: bytearray):
            r = xgb.XGBRegressor(random_state=42)
            r.load_model(b)
            r.set_params(n_jobs=cores)
            return r
        
        # Check tar to see if path, otherwise fileobject
        if type(tar) is str or type(tar) is Path:
            openParams = {'name': tar, 'mode':'r'}
        else:
            openParams = {'fileobj': tar, 'mode':'r'}

        # load the kinase specific XGB trees and sequences from the tar archive
        # there is an 'info.csv' that map uniprot ids to trained models and the full sequence
        # a 'config.json' with basic info about the model and the fingerprinting parameters
        # finally, there is a folder with .ubj files for each uniprot id
        with tarfile.open(**openParams) as tf:
            ubj_bytes = lambda p: bytearray(tf.extractfile(p).read())
            configure = lambda tup: (tup[0], {'models': [load_model(ubj_bytes(tup[1]))], 'seq': tup[2]})
            df = pd.read_csv(tf.extractfile('info.csv'))
            itr = zip(df['uniprot'], df['output_model'], df['sequence'])

            self.protModels = dict(map(configure, itr))
            self.config = json.load(tf.extractfile('config.json'))
        
        self.cores = cores
        self.topK = similar_k
        self.aligner = Align.PairwiseAligner(scoring='blastp')

    def predict(self, smiles: Union[str, npt.NDArray], uniprot: str) -> float:
        if type(smiles) is str:
            fp = self.fingerprint(smiles).reshape(1, -1)
        else:
            fp = smiles

        if uniprot not in self.protModels:
            seqs = self.fetch_seqs([uniprot])
            for uid, seq in seqs.items():
                models = self.get_similar(seq)
                self.protModels[uid] = {'models': models, 'seq': seq}

        return np.mean([m.predict(fp) for m in self.protModels[uniprot]['models']])

    def predict_batch(self, smiles: Sequence[str], uniprot: Sequence[str]) -> List[float]:
        unique_smiles = set(smiles)
        novel_uniprot = set(uniprot).difference(set(self.protModels.keys()))
        print('total unique smiles', len(unique_smiles), '/', len(smiles))
        print('total novel uniprot', len(novel_uniprot), '/', len(uniprot))

        # Ideally this would all be done with multiprocessing...
        novel_seqs = self.fetch_seqs(novel_uniprot)
        print('got novel sequences from uniprot API')
        # with multiprocessing.Pool(self.cores) as p:
        #     models = p.map(self.get_similar, novel_seqs.values())
        #     for (uid, seq), m in zip(novel_seqs.items(), models):
        #         self.protModels[uid] = {'models': m, 'seq': seq}

        for uid, seq in novel_seqs.items():
            models = self.get_similar(seq)
            self.protModels[uid] = {'models': models, 'seq': seq}
        print('found similar models for unique sequences')


        fingerprints = {smi: self.fingerprint(smi).reshape(-1, 1) for smi in unique_smiles}
        print('done fingerprinting')
        return [self.predict(smi, uid) for smi, uid in zip(smiles, uniprot)]
        print(fingerprints)
        # for smiles, uid in zip(smiles, uniprot):
        #     np.mean([m.predict(fingerprints[smiles]) for m in self.protModels[uid]])
        # print(seqs)
    
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

if __name__ == '__main__':
    import argparse
    import fsspec
    import os

    def cli():
        p = argparse.ArgumentParser('Run model in `tarfile` on `datafile`')
        p.add_argument('input', help='input columnar data file with SMILES and UniProt IDs')
        p.add_argument('output', nargs='?', default=None, help="save DF to file, or print to stdout if not present")
        p.add_argument('--tarfile', '-t', default='gs://mwc10-mscbio2066/model_hu-b2048-r2-kikd.tar.gz', help='tar containing model info')
        p.add_argument('--eval', '-e', action='store_true')
        return p.parse_args()
    
    def get_num_cores():
        try:
            return len(os.sched_getaffinity(0))
        except:
            return multiprocessing.cpu_count()
    
    args = cli()
    cores = get_num_cores()
    df = pd.read_csv(args.input, sep=' ').dropna(axis='columns', how='all')
    print(f'creating model for {args.tarfile}')
    with fsspec.open(args.tarfile) as tar:
        m = Model(tar, cores)
    predictions = m.predict_batch(df['SMILES'], df['UniProt'])
    df['Predicted'] = pd.Series(np.array(predictions), dtype=float)
    # test = df.with_columns([pl.Series('Predict pKd', predictions, dtype=pl.Float64)])
    if args.output is not None:
        df.to_csv(args.output, sep=' ', float_format='%.4f', index=False)
        print('saved predictions to ', args.output)

    if 'pKd' in df and args.eval:
        print(df)
        corr = np.corrcoef(df['Predicted'], df['pKd']).min()
        print(f'Correlation with labels in input:\t{corr:.4f}')
