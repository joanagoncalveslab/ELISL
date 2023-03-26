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
import src.embedding.sequence_embedding as seq_e
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

parser = argparse.ArgumentParser(description='Generate Sequence Features')
parser.add_argument('--sample_name', '-sn', metavar='the-name-of-samples', dest='sample', type=str, help='Choose sample name', default='unknown_repair_cancer_BRCA')
parser.add_argument('--sample_loc', '-sl', metavar='the-loc-of-samples', dest='sample_loc', type=str, help='Choose sample file', default='labels/unknown_repair_cancer_BRCA_pairs.csv')
parser.add_argument('--out_loc', '-ol', metavar='the-loc-of-out', dest='out_loc', type=str, help='Choose out loc', default='feature_sets/unknown_repair_cancer_BRCA_seq_1024.csv')
parser.add_argument('--seqvec_emb_loc', '-el', metavar='the-seqvec-emb-loc', dest='emb_loc', type=str, help='Choose seqvec embeddings', default='embeddings/seqvec/uniprot_1024_embeddings.npz')
parser.add_argument('--mapping_loc', '-ml', metavar='the-mapping-loc', dest='mapping_loc', type=str, help='Choose mappings', default='sequences/uniprot_reviewed_9606_mappings.csv')
parser.add_argument('--extra_mapping_loc', '-eml', metavar='the-extra-mapping-loc', dest='extra_mapping_loc', type=str, help='Choose extra mappings', default='sequences/uniprot_extra_mapping.tab')
parser.add_argument('--ready_data', '-rd', metavar='the-extra-ready-data', dest='ready_data_loc', type=str, help='Choose ready data', default='feature_sets/unknown_repair_cancer_BRCA_seq_1024.csv')

args = parser.parse_args()
print(f'Running args:{args}')
task = f'SequenceFeatureGeneration'
log_name = config.ROOT_DIR / 'logs' / f'{task}.txt'
config.ensure_dir(log_name)
logging.basicConfig(level=logging.DEBUG, filename=log_name, filemode="a+",
                    format="%(asctime)-15s %(levelname)-8s %(message)s")
mapping_not_found = []
ABBV = {'dimensions': 'dim', 'walk_length': 'wl', 'num_walks': 'nw', 'p': 'p', 'q': 'q', 'workers': 'ws',
            'window': 'wd', 'min_count': 'mc', 'batch_words': 'bw', 'cutoff': 'co'}


def get_mappings():
    mappings = []
    uniprot_fasta_mapping_loc = config.DATA_DIR / args.mapping_loc
    fasta_df = pd.read_csv(uniprot_fasta_mapping_loc)
    name2uid = fasta_df.set_index('gene_name')['uniprot_id'].to_dict()
    mappings.append(name2uid)
    uname2uid = fasta_df.set_index('uniprot_name')['uniprot_id'].to_dict()
    mappings.append(uname2uid)
    extra_map1, extra_map2 = get_uniprot_extra_mapping()
    extra_map1_dict = {}
    for key, val in extra_map1.items():
        if val in mappings[0].keys():
            extra_map1_dict[key] = mappings[0][val]
        elif val in mappings[1].keys():
            extra_map1_dict[key] = mappings[1][val]
    mappings.append(extra_map1_dict)
    extra_map2_dict = {}
    for key, val in extra_map2.items():
        if val in mappings[0].keys():
            extra_map2_dict[key] = mappings[0][val]
        elif val in mappings[1].keys():
            extra_map2_dict[key] = mappings[1][val]
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


def create_seq_features(df, emb_dict={}, mappings=[{}], emb_name='', ready_data_dict={}):
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
        if ind%10000==0:
            print(f'Iteration {ind} is done.')
        if (a_row[0], a_row[1]) in ready_data_dict.keys():
            all_rows[ind] = ready_data_dict[(a_row[0], a_row[1])]
        else:
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
    print(f'Insertion of sequence features started for {sample} using {args.emb_loc}')
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

    seq_embs = seq_e.load_embeddings(args.emb_loc)
    resulting_df = create_seq_features(samples, seq_embs, get_mappings(), emb_name='seq', ready_data_dict=ready_data_dict)

    resulting_df.to_csv(data_loc, index=None, sep=',')
    try:
        resulting_df.to_csv(gz_loc, index=None, sep=',', compression="gzip")
    except:
        print('Compressed file cannot be saved.')

    not_mappeds = list(set(mapping_not_found))
    unmap_loc = config.DATA_DIR / f'feature_sets/unmapped_{args.out_loc}.pkl'
    dfnc.save_pickle(unmap_loc, not_mappeds)


def main():
    logging.info(f'Sequence feature generation is started for with arguments\n{args}')
    insert_features()


if __name__ == '__main__':
    main()
