import os
import sys
path2this = os.path.dirname(os.path.abspath(__file__)).split('/')
for i, folder in enumerate(path2this):
    if folder.lower()=='elisl':
        project_path = '/'.join(path2this[:i+1])
sys.path.insert(0,project_path)
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
import pandas as pd
import numpy as np
from src import config
import src.embedding.dependency_embedding as dep_e
import src.data_functions as dfnc
import logging
#GCATSL_root = str(config.ROOT_DIR / 'src' / 'comparison' / 'GCATSL/')

PROJECT_LOC = config.ROOT_DIR
import argparse

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

parser = argparse.ArgumentParser(description='Run Single Cancer Experiment')
parser.add_argument('--sample_name', '-sn', metavar='the-name-of-samples', dest='sample', type=str, help='Choose sample name', default='unknown_repair_cancer_SKCM')
parser.add_argument('--sample_loc', '-sl', metavar='the-loc-of-samples', dest='sample_loc', type=str, help='Choose sample file', default='labels/unknown_repair_cancer_SKCM_pairs.csv')
parser.add_argument('--out_loc', '-ol', metavar='the-loc-of-out', dest='out_loc', type=str, help='Choose out loc', default='feature_sets/unknown_repair_cancer_SKCM_crispr_dependency_mut.csv')
parser.add_argument('--dependency_source', '-ds', metavar='the-dependency-source', dest='dependency_source', type=str, help='Choose dependency source', default='crispr')
parser.add_argument('--alteration_source', '-as', metavar='the-alteration-source', dest='alteration_source', type=str, help='Choose alteration source', default='mutation')
parser.add_argument('--cutoff', '-co', metavar='the-cutoff', dest='cutoff', type=float, help='Choose the cutoff', default=1.96)
parser.add_argument('--ready_data', '-rd', metavar='the-extra-ready-data', dest='ready_data_loc', type=str, help='Choose ready data', default='feature_sets/unknown_repair_cancer_SKCM_crispr_dependency_mut.csv')

args = parser.parse_args()
print(f'Running args:{args}')
task = f'CCLEFeatureGeneration'
log_name = config.ROOT_DIR / 'logs' / f'{task}.txt'
config.ensure_dir(log_name)
logging.basicConfig(level=logging.DEBUG, filename=log_name, filemode="a+",
                    format="%(asctime)-15s %(levelname)-8s %(message)s")
logging.info(f'Single cancer experiment started for with arguments\n{args}')
mapping_not_found = []
ABBV = {'dimensions': 'dim', 'walk_length': 'wl', 'num_walks': 'nw', 'p': 'p', 'q': 'q', 'workers': 'ws',
            'window': 'wd', 'min_count': 'mc', 'batch_words': 'bw', 'cutoff': 'co'}


def create_dep_features_per_omic(df, dep_df, mut_df, cancer2ccle, alt_name='mut', cutoff=None, ready_data=None):
    if 'expr' in alt_name:
        df = dep_e.calculate_emb_expr(df, dep_df, mut_df, cancer2ccle, cutoff, ready_data)
    elif 'cnv' in alt_name:
        df = dep_e.calculate_emb_cnv(df, dep_df, mut_df, cancer2ccle, ready_data)
    else:
        df = dep_e.calculate_emb_mut(df, dep_df, mut_df, cancer2ccle, ready_data)
    return df


def insert_features():
    sample_name = args.sample
    print(f'Insertion of sequence features started for {sample_name} using {args.dependency_source} and {args.alteration_source}')
    data_loc = config.DATA_DIR / args.out_loc
    gz_loc = config.DATA_DIR / f'{args.out_loc}.gz'

    ready_data_dict = {}
    if 'None' not in args.ready_data_loc:
        ready_data_loc = config.DATA_DIR / args.ready_data_loc
        ready_data = pd.read_csv(ready_data_loc)
        for ind, row in ready_data.set_index(['gene1', 'gene2']).iterrows():
            ready_data_dict[ind] = row.values[2:]
        print()
    sample_loc = config.DATA_DIR / args.sample_loc
    samples = pd.read_csv(sample_loc)

    if 'unknown' in str(sample_loc):
        out_loc_splitted = args.out_loc.split('.')[0].split('_')
        cancer=[i for i in out_loc_splitted if i.isupper()][0]
        samples['cancer']=cancer

    dep_df = dep_e.get_dataset(args.dependency_source)
    cancer2ccle = dep_e.get_cancer2ccle()
    alt_df = dep_e.get_dataset(args.alteration_source)

    resulting_df = create_dep_features_per_omic(samples, dep_df, alt_df, cancer2ccle, args.alteration_source, args.cutoff, ready_data_dict)


    resulting_df.to_csv(data_loc, index=None, sep=',')
    try:
        resulting_df.to_csv(gz_loc, index=None, sep=',', compression="gzip")
    except:
        print('Compressed file cannot be saved.')

    not_mappeds = list(set(mapping_not_found))
    unmap_loc = config.DATA_DIR / f'{args.out_loc}_unmapped.pkl'
    dfnc.save_pickle(unmap_loc, not_mappeds)


def main():
    insert_features()


if __name__ == '__main__':
    main()
