#!/usr/bin/env python3
import polars as pl

from pathlib import Path
import json
import argparse
import tarfile

def cli() -> argparse.Namespace:
    p = argparse.ArgumentParser(usage='Bundle Config and Metric CSV into folder')
    p.add_argument('--model_base', '-m', required=True, help='Dir with model UBJ')
    p.add_argument('--config', '-c', required=True, help='Training Config')
    p.add_argument('--data', '-d', required=True, help='Training Data Directory')
    p.add_argument('--tar', '-t', action='store_true', help='Store all model files into a .tar')
    
    return p.parse_args()


def main(args: argparse.Namespace):
    base = Path(args.model_base)
    data = Path(args.data)
    cf = Path(args.config)
    config = json.load(open(cf, 'rt'))

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
    config['name'] = cf.stem

    print(info)
    print(config)
    output = base/ 'info.csv'
    outConf = base / 'config.json'
    info.write_csv(output)
    with open(outConf, 'wt') as f:
        json.dump(config, f) 

    if args.tar:
        tarout = Path('model_'+cf.stem).with_suffix('.tar.gz')
        with tarfile.open(tarout, 'w:gz') as f:
            f.add(output, arcname=output.name)
            f.add(outConf, arcname='config.json')
            f.add(base/'models', arcname='models')
        print(tarout)


if __name__ == '__main__':
    args = cli()
    main(args)
