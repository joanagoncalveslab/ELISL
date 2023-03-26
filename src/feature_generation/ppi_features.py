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
import src.embedding.ppi_embedding as ppi_e
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

parser = argparse.ArgumentParser(description='Generate PPI Features')
parser.add_argument('--sample_name', '-sn', metavar='the-name-of-samples', dest='sample', type=str, help='Choose sample name', default='unknown_repair_cancer_BRCA')
parser.add_argument('--sample_loc', '-sl', metavar='the-loc-of-samples', dest='sample_loc', type=str, help='Choose sample file', default='labels/unknown_repair_cancer_BRCA_pairs.csv')
parser.add_argument('--out_loc', '-ol', metavar='the-loc-of-out', dest='out_loc', type=str, help='Choose out loc', default='feature_sets/unknown_repair_cancer_BRCA_ppi_ec.csv')
parser.add_argument('--node2vec_emb_loc', '-el', metavar='the-seqvec-emb-loc', dest='emb_loc', type=str, help='Choose node2vec embeddings', default='embeddings/node2vec/embs_ppi_ec=0.0_dim=64_p=1_q=1_wl=30_nw=200_ws=4_wd=10_mc=1_bw=4')
#parser.add_argument('--mapping_loc', '-ml', metavar='the-mapping-loc', dest='mapping_loc', type=str, help='Choose mappings', default='sequences/uniprot_reviewed_9606_mappings.csv')
parser.add_argument('--extra_mapping_loc', '-eml', metavar='the-extra-mapping-loc', dest='extra_mapping_loc', type=str, help='Choose extra mappings', default='PPI/STRING/uniprot_extra_mapping.tab')
parser.add_argument('--ready_data', '-rd', metavar='the-extra-ready-data', dest='ready_data_loc', type=str, help='Choose ready data', default='feature_sets/unknown_repair_cancer_BRCA_ppi_ec.csv')

args = parser.parse_args()
print(f'Running args:{args}')
task = f'PPIFeatureGeneration'
log_name = config.ROOT_DIR / 'logs' / f'{task}.txt'
config.ensure_dir(log_name)
logging.basicConfig(level=logging.DEBUG, filename=log_name, filemode="a+",
                    format="%(asctime)-15s %(levelname)-8s %(message)s")
mapping_not_found = []
ABBV = {'dimensions': 'dim', 'walk_length': 'wl', 'num_walks': 'nw', 'p': 'p', 'q': 'q', 'workers': 'ws',
            'window': 'wd', 'min_count': 'mc', 'batch_words': 'bw', 'cutoff': 'co'}


def get_mappings():
    mappings = []
    mappings.append(dfnc.get_PPI_genename_maps()[0])
    mappings.append(dfnc.get_PPI_uniprot_maps(aim='name')[0])
    mappings.append(dfnc.get_extra_mapping()[0])
    extra_map1, extra_map2 = get_uniprot_extra_mapping()
    extra_map1_dict = {}
    for key, val in extra_map1.items():
        if val in mappings[0].keys():
            extra_map1_dict[key] = mappings[0][val]
        elif val in mappings[1].keys():
            extra_map1_dict[key] = mappings[1][val]
        elif val in mappings[2].keys():
            extra_map1_dict[key] = mappings[2][val]
    mappings.append(extra_map1_dict)
    extra_map2_dict = {}
    for key, val in extra_map2.items():
        if val in mappings[0].keys():
            extra_map2_dict[key] = mappings[0][val]
        elif val in mappings[1].keys():
            extra_map2_dict[key] = mappings[1][val]
        elif val in mappings[2].keys():
            extra_map2_dict[key] = mappings[2][val]
    mappings.append(extra_map2_dict)

    return mappings


def get_uniprot_extra_mapping():
    loc = config.DATA_DIR / args.extra_mapping_loc
    if not os.path.exists(loc):
        return {}, {}
    data = pd.read_csv(loc, sep='\t')
    data.columns = ['query', 'uniprot_id', 'uniprot_name', 'status', 'prot_names', 'gene_names', 'organism', 'primary_gene_names','synonym_gene_names']
    reviewed = data[data['status'] == 'reviewed']

    reviewed['uniprot_name'] = reviewed['uniprot_name'].str.split('_', n=1, expand=True)[0]

    query2uniprot = reviewed.set_index('query')['uniprot_name'].to_dict()
    query2genename = reviewed.set_index('query')['primary_gene_names'].to_dict()
    return query2uniprot, query2genename


def find_mapping(key, mappings=[], target_dict=[]):
    for mapping in mappings:
        if key in mapping.keys() and mapping[key] in target_dict.keys():
            return target_dict[mapping[key]]
    print(f'{key} mapping not found.')
    mapping_not_found.append(key)
    return [-1] * list(target_dict.values())[0].shape[0]


def find_genes_mapping(genes, mappings, emb_dict):
    name2vec = {}
    for gene in genes:
        name2vec[gene] = find_mapping(gene, mappings, emb_dict)

    return name2vec


def create_ppi_features(df, emb_dict={}, mappings=[{}], emb_name='', ready_data=None):
    emb_length = list(emb_dict.values())[0].shape[0]
    cols = []
    for i in range(emb_length):
        col_name = emb_name + str(i)
        cols.append(col_name)

    genes = np.union1d(df['gene1'].values, df['gene2'].values)
    name2vector = find_genes_mapping(genes, mappings, emb_dict)

    all_rows = np.zeros(shape=(len(df),emb_length))
    all_rows[:]=-1
    for ind, a_row in enumerate(df[['gene1','gene2']].values):
        g1_emb = name2vector[a_row[0]]
        g2_emb = name2vector[a_row[1]]
        if np.average(g1_emb) == -1.0 or np.average(g2_emb) == -1.0:
            pass
        else:
            all_rows[ind] = abs(np.subtract(g1_emb, g2_emb))
    val_df = pd.DataFrame(all_rows, columns=cols, index=df.index.values)
    df = pd.concat([df, val_df], axis=1)
    return df


def insert_features():
    sample = args.sample
    print(f'Insertion of ppi features started for {sample} using {args.emb_loc}')
    data_loc = config.DATA_DIR / args.out_loc
    gz_loc = config.DATA_DIR / f'{args.out_loc}.gz'

    sample_loc = config.DATA_DIR / args.sample_loc
    samples = pd.read_csv(sample_loc)

    ppi_w2v_vector = ppi_e.load_embs_from_raw_loc(config.DATA_DIR / args.emb_loc)
    ppi_embs = ppi_e.get_embs_dict(ppi_w2v_vector)
    resulting_df = create_ppi_features(samples, ppi_embs, get_mappings(), emb_name='ppi_ec')

    resulting_df.to_csv(data_loc, index=None, sep=',')
    try:
        resulting_df.to_csv(gz_loc, index=None, sep=',', compression="gzip")
    except:
        print('Compressed file cannot be saved.')

    not_mappeds = list(set(mapping_not_found))
    unmap_loc = config.DATA_DIR / f'{args.out_loc}_unmapped.pkl'
    dfnc.save_pickle(unmap_loc, not_mappeds)


def main():
    logging.info(f'PPI feature generation is started for with arguments\n{args}')
    insert_features()


if __name__ == '__main__':
    main()
