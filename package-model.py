#!/usr/bin/env python3
import polars as pl

from pathlib import Path
import json
import argparse

def cli() -> argparse.Namespace:
    p = argparse.ArgumentParser(usage='Bundle Config and Metric CSV into folder')
    p.add_argument('--model_base', '-m', required=True, help='Dir with model UBJ')
    p.add_argument('--config', '-c', required=True, help='Training Config')
    p.add_argument('--data', '-d', required=True, help='Training Data Directory')
    
    return p.parse_args()


def main(args: argparse.Namespace):
    base = Path(args.model_base)
    data = Path(args.data)
    config = json.load(open(args.config, 'rt'))

    # filter squences to what's needed
    sequences = (
        pl.scan_csv(
            data / config['kinases']['file'], 
            separator='\t', 
            has_header=False, 
            new_columns=['uniprot', 'sequence']
        ).unique(subset='uniprot')
    )
    info = (
        pl.scan_csv(str(base / 'metrics' / '*.csv'))
        .join(sequences, on='uniprot', how='left')
    ).collect()


    del config['fingerprints']['file']
    del config['kinases']['file']

    print(info)
    print(config)
    output = base/ 'info.csv'
    info.write_csv(output)
    with open(base/'config.json', 'wt') as f:
        json.dump(config, f) 

if __name__ == '__main__':
    args = cli()
    main(args)
