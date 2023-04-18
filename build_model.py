#!/usr/bin/env python3

from pathlib import Path
import polars as pl
import numpy as np
import numpy.typing as npt
import xgboost as xgb
from rdkit.Chem import AllChem
from Bio import Align
import requests

import json
from typing import IO, Dict, List, Sequence, Tuple, Union
import heapq
from itertools import chain
import tarfile

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
        # TODO: Check tar to see if local path or bytes?
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
            df = pl.read_csv(tf.extractfile('info.csv').read())
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
        print('total unique smiles', len(unique_smiles), 'out of', len(smiles))
        print('total novel uniprot', len(novel_uniprot))

        # Ideally this would all be done with multiprocessing...
        novel_seqs = self.fetch_seqs(novel_uniprot)
        print('got novel sequences from uniprot API')
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

    def predict_array(self, smiles: Sequence[str], uniprot: Sequence[str]) -> List[float]:
        # smiles to featurize
        novel_smiles = set(smiles)
        # get set of uniprot strings not already known
        novel_uniprots = set(uniprot).difference(chain(self.knownModels.keys(), self.similarModels.keys()))
        novel_uniprots = self.fetch_seqs(novel_uniprots)
        # with a pool, fingerprint all novel smiles and request seqs for novel uniprots
        smilesToFP = {s: self.fingerprint(s).reshape(1, -1) for s in novel_smiles}
        for uid, seq in novel_uniprots.items():
            self.similarModels[uid] = self.get_similar(seq)
            

        # then, get iterator of (fp, models) to run predictions
        for smiles, uniprot in zip(smiles, uniprot):
            fp = smilesToFP[smiles]
            if uniprot in self.knownModels:
                return self.knownModels[uniprot].predict(fp)
            elif uniprot in self.similarModels:
                return np.mean([m.predict(fp) for m in self.similarModels[uniprot]])
            else:
                raise Exception('no model for', uniprot)

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

