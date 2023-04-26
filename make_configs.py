#!/usr/bin/env python3

import json

RADII = [1, 2, 3, 5, 7, 9, 11]
BITS = [1024, 2048]
ACTIVITIES = [['KD', 'KI'], ['KD', 'KI', 'EC50'], ['KD', 'KI', 'EC50', 'IC50']]

def make_fp(radius, bits):
    return {
        'file': f'ecfp-r{radius}-b{bits}-cf.pkl',
        'bitSize': bits,
        'radius': radius,
        'useFeatures': True,
        'useChirality': True,
    }

def make_config(radius, bits, acts):
    return {
        'kinases': {
            'file': 'map-uniprot-seq.tsv',
            'species': ['Human'],
        },
        'fingerprints': make_fp(radius, bits),
        'dataFile': 'full-median.parquet',
        'activityTypes': acts,
        'optRounds': 100,
    }

def make_fname(radius, bits, acts):
    return f'hu-fp{bits}r{radius}-{"".join(acts)}-CF.json'

def pair(*args):
    return make_fname(*args), make_config(*args)

sets = [pair(r,b,a) for r in RADII for b in BITS for a in ACTIVITIES]

for f, c in sets:
    json.dump(c, open(f'configs/{f}', 'w'), indent=4)
