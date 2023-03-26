import pandas as pd
import sys, os
import numpy as np
import csv
import collections as coll
from src import config


def load_mut(file_loc, remove_silent=True):
    file_loc = config.DATA_DIR / file_loc
    df = pd.read_csv(file_loc, sep='\t', header=2)
    df = df[['Hugo_Symbol', 'Entrez_Gene_Id', 'Variant_Classification', 'Variant_Type', 'Tumor_Sample_Barcode', 'Broad_ID']]
    if remove_silent:
        df = df[df['Variant_Classification'] != 'Silent']
    return df


def load_expr_cnv(file_loc, mapping):
    file_loc = config.DATA_DIR / file_loc
    df = pd.read_csv(file_loc, sep='\t')
    df = df.set_index('Hugo_Symbol')
    #df_upd = df.rename(columns=mapping)
    #df_upd = df_upd.sort_index()
    #df_upd = df_upd.sort_index(axis=1)
    df = df.sort_index()
    df = df.sort_index(axis=1)
    return df


def load_cell_info(file_loc, extra_maps):
    file_loc = config.DATA_DIR / file_loc
    df = pd.read_csv(file_loc, sep='\t', header=4)
    df = df[['SAMPLE_ID', 'DEPMAPID', 'CANCER_TYPE']]
    sample2depmap = df.set_index('DEPMAPID')['SAMPLE_ID'].to_dict()
    for depmap_id, sample_id in extra_maps.items():
        if depmap_id not in sample2depmap.keys():
            sample2depmap[depmap_id] = sample_id
    return df, sample2depmap


def load_ccle_map(file_loc):
    file_loc = config.DATA_DIR / file_loc
    df = pd.read_csv(file_loc, sep=',')
    sample2depmap = df.set_index('Broad_ID')['CCLE_Name'].to_dict()
    return sample2depmap


def get_ccle_maps(df, map_dict=coll.OrderedDict({'id': 'name', 'name': 'id', 'id': 'strip_name'})):
    mappings = []
    col_maps = {'id': df.DepMap_ID, 'name': df.CCLE_Name, 'strip_name': df.stripped_cell_line_name}
    for fr, to in map_dict.items():
        mappings.append(dict(zip(col_maps[fr], col_maps[to])))
    return tuple(mappings)

    #ccle_id2name = dict(zip(df.DepMap_ID, df.CCLE_Name))
    #ccle_name2id = dict(zip(df.CCLE_Name, df.DepMap_ID))

def load_crispr_dep(file_loc, mapping, is_entrez=False, ):
    '''

    :param file_loc: location of file in data directory.
    :return:
        df(cell_line_id X gene_name(entrez_id))
    '''
    file_loc = config.DATA_DIR / file_loc
    df = pd.read_csv(file_loc)
    df = df.set_index('Broad_ID')
    df = df.sort_index()
    df = df.sort_index(axis=1)
    if is_entrez:
        df.columns = [col.split(' ', maxsplit=1)[1][1:-1] for col in df.columns]
    else:
        df.columns = [col.split(' ', maxsplit=1)[0] for col in df.columns]

    df = df.T
    df_upd = df.rename(columns=mapping)
    df_upd = df_upd.sort_index()
    df_upd = df_upd.sort_index(axis=1)
    return df_upd


def load_d2_dep(file_loc, lst_col='Unnamed: 0'):
    '''

    :param file_loc: location of file in data directory.
    :return:
        df(cell_line_name X gene_name1&gene_name2 (entrez_id1&entrez_id2))
    '''
    file_loc = config.DATA_DIR / file_loc
    df_base = pd.read_csv(file_loc)
    df_base[lst_col] = df_base[lst_col].str.split(' ',n=1, expand=True)[0]
    df = df_base.assign(**{lst_col: df_base[lst_col].str.split('&')})

    new_df = pd.DataFrame({
        col: np.repeat(df[col].values, df[lst_col].str.len())
        for col in df.columns.difference([lst_col])
    }).assign(**{lst_col: np.concatenate(df[lst_col].values)})[df.columns.tolist()]

    new_df = new_df.set_index(lst_col)

    new_df = new_df.sort_index()
    new_df = new_df.sort_index(axis=1)
    return new_df


def get_locs():
    loc_dict = coll.OrderedDict()
    loc_dict['mutation'] = 'ccle_broad_2019/data_mutations_extended.txt'
    loc_dict['expression'] = 'ccle_broad_2019/data_RNA_Seq_mRNA_median_all_sample_Zscores.txt'
    loc_dict['crispr_dependency'] = 'ccle_broad_2019/gene_dependency.csv'
    loc_dict['d2_dependency'] = 'ccle_broad_2019/D2_combined_gene_dep_scores.csv'
    loc_dict['cnv'] = 'ccle_broad_2019/data_CNA.txt'
    loc_dict['sample_info'] = 'ccle_broad_2019/data_clinical_sample.txt'
    loc_dict['sample_info_20q4'] = 'ccle_broad_2019/sample_info_20q4.csv'
    loc_dict['patient_info'] = 'ccle_broad_2019/data_clinical_patient.txt'
    loc_dict['segment_cn'] = 'ccle_broad_2019/data_cna_hg19.seg'
    loc_dict['ccle_map'] = 'ccle_broad_2019/DepMap-2018q3-celllines.csv'

    return loc_dict


def main():
    loc_dict = get_locs()
    ccle_map = load_ccle_map(loc_dict['ccle_map'])
    #mut_df = load_mut(loc_dict['mutation'])
    sample_df, mapping = load_cell_info(loc_dict['sample_info'], extra_maps=ccle_map)
    #expr_df = load_expr_cnv(loc_dict['expression'], mapping)
    #cnv_df = load_expr_cnv(loc_dict['cnv'], mapping)
    d2_combined_dep = load_d2_dep(loc_dict['d2_dependency'])
    crispr_dep = load_crispr_dep(loc_dict['crispr_dependency'], mapping=ccle_map)
    print()

    #segment_cn_df = load_cnv_segment(loc_dict['segment_cn'])
    #df = segment_cn_df.drop(columns=['Source'])
    #df['Chromosome'] = df['Chromosome'].apply(lambda x: x.replace('chr', '') if 'chr' in x else x)
    #df1 = df.drop_duplicates(subset=['DepMap_ID', 'Chromosome'], keep='first')
    #df.to_csv('/Users/yitepeli/PycharmProjects/SL_experiments/data/cell_line_data/CCLE_segment_cn_gistic.csv',
    #          index=None, sep='\t')

    #id2name, name2id = get_ccle_maps(sample_df, coll.OrderedDict({'id': 'name', 'name': 'id'}))
    #mut_df = load_mut(loc_dict['mutation'])
    #expr_df = load_sample_gene_data(loc_dict['expression'])
    #std_expr(expr_df)
    #cnv_df = load_sample_gene_data(loc_dict['cnv'])
    #rnai_dep_df = load_rnai_dep(loc_dict['rnai_dependency'], name2id)
    #crispr_dep_df = load_crispr_dep(loc_dict['crispr_dependency'])
    return 0


if __name__ == '__main__':
    main()
