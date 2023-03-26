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
import src.embedding.tissue_embedding as tis_e
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

parser = argparse.ArgumentParser(description='Generate Tissue Features')
parser.add_argument('--sample_name', '-sn', metavar='the-name-of-samples', dest='sample', type=str, help='Choose sample name', default='unknown_repair_cancer_CESC')
parser.add_argument('--sample_loc', '-sl', metavar='the-loc-of-samples', dest='sample_loc', type=str, help='Choose sample file', default='labels/unknown_repair_cancer_CESC_pairs.csv')
parser.add_argument('--out_loc', '-ol', metavar='the-loc-of-out', dest='out_loc', type=str, help='Choose out loc', default='feature_sets/unknown_repair_cancer_CESC_tissue.csv')
parser.add_argument('--cutoff', '-co', metavar='the-cutoff', dest='cutoff', type=float, help='Choose the cutoff', default=1.96)
parser.add_argument('--ready_data', '-rd', metavar='the-extra-ready-data', dest='ready_data_loc', type=str, help='Choose ready data', default='feature_sets/unknown_repair_cancer_CESC_tissue.csv')

args = parser.parse_args()
print(f'Running args:{args}')
task = f'TissueFeatureGeneration_{args.sample}'
log_name = config.ROOT_DIR / 'logs' / f'{task}.txt'
config.ensure_dir(log_name)
logging.basicConfig(level=logging.DEBUG, filename=log_name, filemode="a+",
                    format="%(asctime)-15s %(levelname)-8s %(message)s")
mapping_not_found = []
ABBV = {'dimensions': 'dim', 'walk_length': 'wl', 'num_walks': 'nw', 'p': 'p', 'q': 'q', 'workers': 'ws',
            'window': 'wd', 'min_count': 'mc', 'batch_words': 'bw', 'cutoff': 'co'}


def create_tissue_features(df, feature_type=None, cutoff=None, ready_data=None):
    for cancer in df['cancer'].unique():
        logging.info(f'Creating tissue features for {cancer}-{feature_type}-{cutoff}')
        if feature_type is None or 'surv' in feature_type:
            mut_df = tis_e.get_dataset('mutation', cancer)
            std_expr_df = tis_e.get_dataset('std_expression', cancer)
            cnv_df = tis_e.get_dataset('cnv', cancer)
            sample_df = tis_e.get_dataset('patients', cancer)
            df = tis_e.calculate_surv_any_para(df, cancer, mut_df, std_expr_df, cnv_df, sample_df, cutoff, ready_data)
        if feature_type is None or 'tcoexp' in feature_type:
            cnv_df = tis_e.get_dataset('cnv', cancer)
            tumor_expr_df, normal_expr_df = tis_e.get_dataset('expression', cancer)
            df = tis_e.calculate_tumor_coexp_cnv(df, cancer, tumor_expr_df, normal_expr_df, cnv_df, ready_data)
        if feature_type is None or 'hcoexp' in feature_type:
            gtex_expr_df = tis_e.get_dataset('healthy_expression', cancer)
            df = tis_e.calculate_healthy_coexp(df, cancer, gtex_expr_df, ready_data)
        if feature_type is None or 'diff_expr' in feature_type:
            mut_df = tis_e.get_dataset('mutation', cancer)
            std_expr_df = tis_e.get_dataset('std_expression', cancer)
            sample_df = tis_e.get_dataset('patients', cancer)
            df = tis_e.calculate_expr_by_mut(df, cancer, mut_df, std_expr_df, sample_df, ready_data)

    return df


def insert_features():
    sample = args.sample
    print(f'Insertion of sequence features started for {sample}')
    logging.info(f'Insertion of tissue features started for {sample}')
    data_loc = config.DATA_DIR / args.out_loc
    gz_loc = config.DATA_DIR / f'{args.out_loc}.gz'

    ready_data_dict = {}
    if 'None' not in args.ready_data_loc:
        ready_data_loc = config.DATA_DIR / args.ready_data_loc
        ready_data = pd.read_csv(ready_data_loc)
        for ind, row in ready_data.set_index(['gene1', 'gene2']).iterrows():
            ready_data_dict[ind] = row.to_dict()
        print()

    sample_loc = config.DATA_DIR / args.sample_loc
    samples = pd.read_csv(sample_loc)
    if 'unknown' in str(sample_loc):
        out_loc_splitted = args.out_loc.split('.')[-2].split('_')
        cancer=[i for i in out_loc_splitted if i.isupper()][0]
        samples['cancer']=cancer
    dfs = []
    feature_type_suffixes = {}
    for feature_type in ['tissue_diff_expr', 'tissue_hcoexp', 'tissue_tcoexp', 'tissue_surv']:
        df_tmp = create_tissue_features(samples, feature_type=feature_type, cutoff=args.cutoff, ready_data = ready_data_dict)
        dfs.append(df_tmp)

    resulting_df = dfs[0]
    for i in range(1, len(dfs)):
        resulting_df = pd.merge(resulting_df, dfs[i], on=['gene1', 'gene2', 'cancer', 'class'])

    resulting_df.to_csv(data_loc, index=None, sep=',')
    logging.info(f'Tissue features saved to {data_loc}')
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
