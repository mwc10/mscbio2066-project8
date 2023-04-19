#!/usr/bin/env python3
from typing import Dict, Union
import numpy as np
import numpy.typing as npt
import polars as pl
from rdkit.Chem import AllChem

from multiprocessing import Pool, cpu_count
import os
import argparse
import pickle
import json
import itertools

def cli() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument('--data', '-d', required=True)
    p.add_argument('--config', '-c', required=True, help='JSON model file with fingerprints key')

    return p.parse_args()

def ecfp_from_inchi(inchi:str, radius: int, bits: int, features: bool, chirality: bool) -> npt.NDArray[np.uint8]:
    mol = AllChem.MolFromInchi(inchi)
    return np.array(
        AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=bits, useFeatures=features, useChirality=chirality),
        dtype=np.uint8
    )

def wrap_mapping(cid: str, inchi: str, conf: Dict[str, Union[str, int, bool]]):
    fp = ecfp_from_inchi(inchi, conf['radius'], conf['bitSize'], conf['useFeatures'], conf['useChirality'])
    return (cid, fp)


def main(args: argparse.Namespace):
    try:
        procs = len(os.sched_getaffinity(0))
    except:
        procs = cpu_count()

    config = json.load(open(args.config))['fingerprints']

    df = (pl.scan_parquet(args.data)
        .select('compound_chembl_id', 'canonical_smiles', 'standard_inchi')
        .unique()
        .collect()
    )

    ids = df['compound_chembl_id']
    keys = df['standard_inchi']

    with Pool(procs) as pool:
        hashmap = {i: key for i, key in pool.starmap(wrap_mapping, zip(ids, keys, itertools.cycle([config])))}

    with open(config['file'], 'wb') as f:
        pickle.dump(hashmap, f)

if __name__ == '__main__':
    args = cli()
    main(args)
