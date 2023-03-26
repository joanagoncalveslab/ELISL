import numpy as np
'''
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
'''
import collections as coll
from src.datasets import ccle_broad as cb
from src import config
import pandas as pd


def get_locs():
    loc_dict = coll.OrderedDict()
    loc_dict['mutation'] = 'cell_line_data/CCLE_mutations.csv'
    loc_dict['expression'] = 'cell_line_data/CCLE_expression.csv'
    loc_dict['crispr_dependency'] = 'cell_line_data/Achilles_gene_effect.csv'
    loc_dict['rnai_dependency'] = 'cell_line_data/D2_combined_gene_dep_scores.csv'
    loc_dict['cnv'] = 'cell_line_data/CCLE_gene_cn.csv'
    loc_dict['sample_info'] = 'cell_line_data/sample_info.csv'

    return loc_dict

def get_maps(df, map_dict):
    return cb.get_ccle_maps(df, map_dict)

def get_uniprot_extra_mapping(folder):
    loc = config.DATA_DIR / folder / 'uniprot_extra_mapping.tab'
    data = pd.read_csv(loc, sep='\t')
    data.columns = ['query', 'uniprot_id', 'uniprot_name', 'status', 'prot_names', 'gene_names', 'organism', 'primary_gene_names', 'synonym_gene_names']
    reviewed = data[data['status'] == 'reviewed']

    reviewed['uniprot_name'] = reviewed['uniprot_name'].str.split('_', n=1, expand=True)[0]

    query2uniprot = reviewed.set_index('query')['uniprot_name'].to_dict()
    query2genename = reviewed.set_index('query')['primary_gene_names'].to_dict()
    return query2uniprot, query2genename

def get_dataset(name):
    loc_dict = cb.get_locs()
    ccle_map = cb.load_ccle_map(loc_dict['ccle_map'])
    sample_df, mapping = cb.load_cell_info(loc_dict['sample_info'], extra_maps=ccle_map)

    if name == 'samples_20q4':
        file_loc = cb.config.DATA_DIR / loc_dict['sample_info_20q4']
        sample_df = cb.pd.read_csv(file_loc)
        return sample_df
    elif name == 'samples':
        return sample_df
    elif name == 'mappings':
        uniprot_extra_mapping_loc = 'ccle_broad_2019/uniprot_extra_mapping.tab'
        uniprot_extra1, uniprot_extra2 = get_uniprot_extra_mapping(uniprot_extra_mapping_loc)
    elif name == 'mutation':
        return cb.load_mut(loc_dict['mutation'])
    elif name == 'expression':
        tmp = cb.load_expr_cnv(loc_dict['expression'], mapping).T.dropna(axis=1, how='all')
        uniq_cols = np.unique(tmp.columns.values, return_index=True, return_counts=True)
        dupl_indices = np.argwhere(uniq_cols[2] > 1)[:,0]
        col_ids = uniq_cols[1]
        col_names = uniq_cols[0]
        for dupl_pos in dupl_indices:
            max_var_id = np.argmax(tmp[col_names[dupl_pos]].var().values)
            chosen_id = col_ids[dupl_pos]+max_var_id
            col_ids[dupl_pos]=chosen_id
        return tmp.iloc[:,col_ids]
    elif name == 'cnv':
        tmp = cb.load_expr_cnv(loc_dict['cnv'], mapping).T.dropna(axis=1, how='all')
        uniq_cols = np.unique(tmp.columns.values, return_index=True, return_counts=True)
        dupl_indices = np.argwhere(uniq_cols[2] > 1)[:,0]
        col_ids = uniq_cols[1]
        col_names = uniq_cols[0]
        for dupl_pos in dupl_indices:
            max_var_id = np.argmax(tmp[col_names[dupl_pos]].var().values)
            chosen_id = col_ids[dupl_pos]+max_var_id
            col_ids[dupl_pos]=chosen_id
        return tmp.iloc[:,col_ids]
    elif 'crispr' in name:
        return cb.load_crispr_dep(loc_dict['crispr_dependency'], mapping=mapping).T
    elif 'd2' in name or 'rnai' in name:
        return cb.load_d2_dep(loc_dict['d2_dependency']).T


def get_cancer2ccle():
    sample_df = get_dataset('samples')
    sample_df['tissue'] = sample_df['SAMPLE_ID'].str.split('_', n=1, expand=True)[1]
    sample_20q4_df = get_dataset('samples_20q4')
    cancer_dict = {'BRCA': 'BREAST', 'CESC': 'CERVIX', 'COAD': 'LARGE_INTESTINE', 'KIRC': 'KIDNEY',
                   'LAML':'HAEMATOPOIETIC_AND_LYMPHOID_TISSUE', 'LUAD': 'LUNG',
                   'OV': 'OVARY', 'SKCM': 'SKIN'}

    cancer_20q4_dict = {'BRCA': 'Breast Cancer', 'CESC': 'Cervical Cancer', 'COAD': 'Colon/Colorectal Cancer',
                   'KIRC': 'Kidney Cancer', 'LAML': 'Leukemia', 'LUAD': 'Lung Cancer',
                   'OV': 'Ovarian Cancer', 'SKCM': 'Skin Cancer'}

    cancer_CCLE = {}
    for cancer, disease in cancer_dict.items():
        non_match_20q4 = []
        cancer_CCLE[cancer] = sample_20q4_df[sample_20q4_df['primary_disease'] == cancer_20q4_dict[cancer]]['CCLE_Name'].values
        extra_match = sample_df[sample_df['tissue'] == cancer_dict[cancer]]['SAMPLE_ID'].values
        cancer_CCLE[cancer] = np.union1d(cancer_CCLE[cancer], extra_match)

    return cancer_CCLE


def calculate_emb_mut(df, dep_df, mut_df, cancer2ccle, ready_data):
    cancer_dict = {'BRCA': 'BREAST', 'CESC': ['OESOPHAGUS','STOMACH'], 'COAD': 'LARGE_INTESTINE', 'KIRC': 'KIDNEY',
                   'LAML':'HAEMATOPOIETIC_AND_LYMPHOID_TISSUE', 'LUAD': 'LUNG',
                   'OV': 'OVARY', 'SKCM': 'SKIN'}

    cancer_dict = {'BRCA': 'Breast Cancer', 'CESC': 'Cervical Cancer', 'COAD': 'Colon/Colorectal Cancer',
                   'KIRC': 'Kidney Cancer', 'LAML': 'Leukemia', 'LUAD': 'Lung Cancer',
                   'OV': 'Ovarian Cancer', 'SKCM': 'Skin Cancer'}
    cancer_CCLE = cancer2ccle
    extra_map1, extra_map2 = get_uniprot_extra_mapping('ccle_broad_2019')
    grouped_mut = mut_df.groupby('Hugo_Symbol')['Tumor_Sample_Barcode']\
        .agg(list= lambda x: list(np.unique(x))).reset_index()
    gene_mut_ccle = grouped_mut.set_index('Hugo_Symbol')['list'].to_dict()

    emb_dict = {}
    all_rows = np.zeros(shape=(len(df),4))
    for idx, a_row in enumerate(df[['gene1', 'gene2', 'cancer']].values):
        if idx%5000==0:
            print(f'Iteration {idx} is done.')
        embedding = []
        gene1 = a_row[0]
        gene2 = a_row[1]
        cancer = a_row[2]
        emb_name = gene1+'|'+gene2+'|'+cancer
        if ready_data is not None and type(ready_data)==dict and (gene1, gene2) in ready_data.keys():
            all_rows[idx] = ready_data[(gene1, gene2)]
            continue
        elif (ready_data is not None) and type(ready_data)!=dict:
            chosen_row = ready_data[(ready_data[['gene1', 'gene2', 'cancer']] == a_row[0:3]).all(1)]
            if len(chosen_row)>0:
                emb_dict[emb_name] = chosen_row.loc[:, ['dep0', 'dep1', 'dep2', 'dep3']].values[0]
                continue

        '''
        gene2_mutated_CLLEs = np.array([])
        if gene2 in mut_df['Hugo_Symbol'].values:
            gene2_mutated_CLLEs = np.unique(mut_df[mut_df['Hugo_Symbol'] == gene2]['Tumor_Sample_Barcode'].values)
        elif gene2 in extra_map1.keys() and extra_map1[gene2] in mut_df['Hugo_Symbol'].values:
            gene2_mutated_CLLEs = np.unique(mut_df[mut_df['Hugo_Symbol'] == extra_map1[gene2]]['Tumor_Sample_Barcode'].values)
        elif gene2 in extra_map2.keys() and extra_map2[gene2] in mut_df['Hugo_Symbol'].values:
            gene2_mutated_CLLEs = np.unique(mut_df[mut_df['Hugo_Symbol'] == extra_map2[gene2]]['Tumor_Sample_Barcode'].values)
        gene2_mutated_CLLEs = np.intersect1d(gene2_mutated_CLLEs, cancer_CCLE[cancer])

        gene1_mutated_CLLEs = np.array([])
        if gene1 in mut_df['Hugo_Symbol'].values:
            gene1_mutated_CLLEs = np.unique(mut_df[mut_df['Hugo_Symbol'] == gene1]['Tumor_Sample_Barcode'].values)
        elif gene1 in extra_map1.keys() and extra_map1[gene1] in mut_df['Hugo_Symbol'].values:
            gene1_mutated_CLLEs = np.unique(mut_df[mut_df['Hugo_Symbol'] == extra_map1[gene1]]['Tumor_Sample_Barcode'].values)
        elif gene1 in extra_map2.keys() and extra_map2[gene1] in mut_df['Hugo_Symbol'].values:
            gene1_mutated_CLLEs = np.unique(mut_df[mut_df['Hugo_Symbol'] == extra_map2[gene1]]['Tumor_Sample_Barcode'].values)
        gene1_mutated_CLLEs = np.intersect1d(gene1_mutated_CLLEs, cancer_CCLE[cancer])

        #gene1_mutated_CLLEs = np.unique(mut_df[mut_df['Hugo_Symbol'] == gene1]['Tumor_Sample_Barcode'].values)
        #gene1_mutated_CLLEs = np.intersect1d(gene1_mutated_CLLEs, cancer_CCLE[cancer])
        '''
        gene2_mutated_CLLEs = np.array([])
        if gene2 in gene_mut_ccle.keys():
            gene2_mutated_CLLEs = np.array(gene_mut_ccle[gene2])
        elif gene2 in extra_map1.keys() and extra_map1[gene2] in gene_mut_ccle.keys():
            gene2_mutated_CLLEs = np.array(gene_mut_ccle[extra_map1[gene2]])
        elif gene2 in extra_map2.keys() and extra_map2[gene2] in gene_mut_ccle.keys():
            gene2_mutated_CLLEs = np.array(gene_mut_ccle[extra_map2[gene2]])
        gene2_mutated_CLLEs = np.intersect1d(gene2_mutated_CLLEs, cancer_CCLE[cancer])

        gene1_mutated_CLLEs = np.array([])
        if gene1 in gene_mut_ccle.keys():
            gene1_mutated_CLLEs = np.array(gene_mut_ccle[gene1])
        elif gene1 in extra_map1.keys() and extra_map1[gene1] in gene_mut_ccle.keys():
            gene1_mutated_CLLEs = np.array(gene_mut_ccle[extra_map1[gene1]])
        elif gene1 in extra_map2.keys() and extra_map2[gene1] in gene_mut_ccle.keys():
            gene1_mutated_CLLEs = np.array(gene_mut_ccle[extra_map2[gene1]])
        gene1_mutated_CLLEs = np.intersect1d(gene1_mutated_CLLEs, cancer_CCLE[cancer])


        avg_dep_gene1_w_gene2_mut = np.nan
        avg_dep_gene1_wo_gene2_mut = np.nan
        avg_dep_gene2_w_gene1_mut = np.nan
        avg_dep_gene2_wo_gene1_mut = np.nan
        if gene1 in dep_df.columns:
            avg_dep_gene1_w_gene2_mut = dep_df[dep_df.index.isin(gene2_mutated_CLLEs)][gene1].mean(skipna=True)
            avg_dep_gene1_wo_gene2_mut = dep_df[(~dep_df.index.isin(gene2_mutated_CLLEs))&(dep_df.index.isin(cancer_CCLE[cancer]))][gene1].mean(skipna=True)
        elif gene1 in extra_map1.keys() and extra_map1[gene1] in dep_df.columns:
            avg_dep_gene1_w_gene2_mut = dep_df[dep_df.index.isin(gene2_mutated_CLLEs)][extra_map1[gene1]].mean(skipna=True)
            avg_dep_gene1_wo_gene2_mut = dep_df[(~dep_df.index.isin(gene2_mutated_CLLEs))&(dep_df.index.isin(cancer_CCLE[cancer]))][extra_map1[gene1]].mean(skipna=True)
        elif gene1 in extra_map2.keys() and extra_map2[gene1] in dep_df.columns:
            avg_dep_gene1_w_gene2_mut = dep_df[dep_df.index.isin(gene2_mutated_CLLEs)][extra_map2[gene1]].mean(skipna=True)
            avg_dep_gene1_wo_gene2_mut = dep_df[(~dep_df.index.isin(gene2_mutated_CLLEs))&(dep_df.index.isin(cancer_CCLE[cancer]))][extra_map2[gene1]].mean(skipna=True)

        if gene2 in dep_df.columns:
            avg_dep_gene2_w_gene1_mut = dep_df[dep_df.index.isin(gene1_mutated_CLLEs)][gene2].mean(skipna=True)
            avg_dep_gene2_wo_gene1_mut = dep_df[(~dep_df.index.isin(gene1_mutated_CLLEs))&(dep_df.index.isin(cancer_CCLE[cancer]))][gene2].mean(skipna=True)
        elif gene2 in extra_map1.keys() and extra_map1[gene2] in dep_df.columns:
            avg_dep_gene2_w_gene1_mut = dep_df[dep_df.index.isin(gene1_mutated_CLLEs)][extra_map1[gene2]].mean(skipna=True)
            avg_dep_gene2_wo_gene1_mut = dep_df[(~dep_df.index.isin(gene1_mutated_CLLEs))&(dep_df.index.isin(cancer_CCLE[cancer]))][extra_map1[gene2]].mean(skipna=True)
        elif gene2 in extra_map2.keys() and extra_map2[gene2] in dep_df.columns:
            avg_dep_gene2_w_gene1_mut = dep_df[dep_df.index.isin(gene1_mutated_CLLEs)][extra_map2[gene2]].mean(skipna=True)
            avg_dep_gene2_wo_gene1_mut = dep_df[(~dep_df.index.isin(gene1_mutated_CLLEs))&(dep_df.index.isin(cancer_CCLE[cancer]))][extra_map2[gene2]].mean(skipna=True)

        #emb_dict[emb_name] = np.array([avg_dep_gene1_w_gene2_mut, avg_dep_gene1_wo_gene2_mut,
        #   avg_dep_gene2_w_gene1_mut, avg_dep_gene2_wo_gene1_mut])
        all_rows[idx]=np.array([avg_dep_gene1_w_gene2_mut, avg_dep_gene1_wo_gene2_mut,
                                avg_dep_gene2_w_gene1_mut, avg_dep_gene2_wo_gene1_mut])
    val_df = pd.DataFrame(all_rows, columns=['dep0', 'dep1', 'dep2', 'dep3'], index=df.index.values)
    df = pd.concat([df, val_df], axis=1)
    return df


def calculate_emb_cnv(df, dep_df, cnv_df, cancer2ccle, cutoff=2, ready_data=None):
    cancer_dict = {'BRCA': 'BREAST', 'CESC': ['OESOPHAGUS','STOMACH'], 'COAD': 'LARGE_INTESTINE', 'KIRC': 'KIDNEY',
                   'LAML':'HAEMATOPOIETIC_AND_LYMPHOID_TISSUE', 'LUAD': 'LUNG',
                   'OV': 'OVARY', 'SKCM': 'SKIN'}

    cancer_dict = {'BRCA': 'Breast Cancer', 'CESC': 'Cervical Cancer', 'COAD': 'Colon/Colorectal Cancer',
                   'KIRC': 'Kidney Cancer', 'LAML': 'Leukemia', 'LUAD': 'Lung Cancer',
                   'OV': 'Ovarian Cancer', 'SKCM': 'Skin Cancer'}
    cancer_CCLE = cancer2ccle
    extra_map1, extra_map2 = get_uniprot_extra_mapping('ccle_broad_2019')
    emb_dict = {}
    all_rows = np.zeros(shape=(len(df),4))
    for idx, a_row in enumerate(df[['gene1', 'gene2', 'cancer']].values):
        if idx%5000==0:
            print(f'Iteration {idx} is done.')
        embedding = []
        gene1 = a_row[0]
        gene2 = a_row[1]
        cancer = a_row[2]
        emb_name = gene1+'|'+gene2+'|'+cancer
        if (ready_data is not None):
            chosen_row = ready_data[(ready_data[['gene1', 'gene2', 'cancer']] == a_row[0:3]).all(1)]
        if ready_data is not None and len(chosen_row)>0:
            emb_dict[emb_name] = chosen_row.loc[:, ['dep0', 'dep1', 'dep2', 'dep3']].values[0]
            continue

        #https: // www.hindawi.com / journals / bmri / 2015 / 901303 /  # materials-and-methods
        gene2_altcnv_CLLEs = np.array([])
        if gene2 in cnv_df.columns:
            gene2_altcnv_CLLEs = np.unique(cnv_df[(cnv_df[gene2] >= cutoff) | (cnv_df[gene2] <= -cutoff)].index.values)
        elif gene2 in extra_map1.keys() and extra_map1[gene2] in cnv_df.columns:
            gene2_altcnv_CLLEs = np.unique(cnv_df[(cnv_df[extra_map1[gene2]] >= cutoff) | (cnv_df[extra_map1[gene2]] <= -cutoff)].index.values)
        elif gene2 in extra_map2.keys() and extra_map2[gene2] in cnv_df.columns:
            gene2_altcnv_CLLEs = np.unique(cnv_df[(cnv_df[extra_map2[gene2]] >= cutoff) | (cnv_df[extra_map2[gene2]] <= -cutoff)].index.values)
        gene2_altcnv_CLLEs = np.intersect1d(gene2_altcnv_CLLEs, cancer_CCLE[cancer])

        gene1_altcnv_CLLEs = np.array([])
        if gene1 in cnv_df.columns:
            gene1_altcnv_CLLEs = np.unique(cnv_df[(cnv_df[gene1] >= cutoff) | (cnv_df[gene1] <= -cutoff)].index.values)
        elif gene1 in extra_map1.keys() and extra_map1[gene1] in cnv_df.columns:
            gene1_altcnv_CLLEs = np.unique(cnv_df[(cnv_df[extra_map1[gene1]] >= cutoff) | (cnv_df[extra_map1[gene1]] <= -cutoff)].index.values)
        elif gene1 in extra_map2.keys() and extra_map2[gene1] in cnv_df.columns:
            gene1_altcnv_CLLEs = np.unique(cnv_df[(cnv_df[extra_map2[gene1]] >= cutoff) | (cnv_df[extra_map2[gene1]] <= -cutoff)].index.values)
        gene1_altcnv_CLLEs = np.intersect1d(gene1_altcnv_CLLEs, cancer_CCLE[cancer])

        avg_dep_gene1_w_gene2_altcnv = np.nan
        avg_dep_gene1_wo_gene2_altcnv = np.nan
        avg_dep_gene2_w_gene1_altcnv = np.nan
        avg_dep_gene2_wo_gene1_altcnv = np.nan

        if gene1 in dep_df.columns:
            avg_dep_gene1_w_gene2_altcnv = dep_df[dep_df.index.isin(gene2_altcnv_CLLEs)][gene1].mean(skipna=True)
            avg_dep_gene1_wo_gene2_altcnv = dep_df[(~dep_df.index.isin(gene2_altcnv_CLLEs))&(dep_df.index.isin(cancer_CCLE[cancer]))][gene1].mean(skipna=True)
        elif gene1 in extra_map1.keys() and extra_map1[gene1] in dep_df.columns:
            avg_dep_gene1_w_gene2_altcnv = dep_df[dep_df.index.isin(gene2_altcnv_CLLEs)][extra_map1[gene1]].mean(skipna=True)
            avg_dep_gene1_wo_gene2_altcnv = dep_df[(~dep_df.index.isin(gene2_altcnv_CLLEs))&(dep_df.index.isin(cancer_CCLE[cancer]))][extra_map1[gene1]].mean(skipna=True)
        elif gene1 in extra_map2.keys() and extra_map2[gene1] in dep_df.columns:
            avg_dep_gene1_w_gene2_altcnv = dep_df[dep_df.index.isin(gene2_altcnv_CLLEs)][extra_map2[gene1]].mean(skipna=True)
            avg_dep_gene1_wo_gene2_altcnv = dep_df[(~dep_df.index.isin(gene2_altcnv_CLLEs))&(dep_df.index.isin(cancer_CCLE[cancer]))][extra_map2[gene1]].mean(skipna=True)


        if gene2 in dep_df.columns:
            avg_dep_gene2_w_gene1_altcnv = dep_df[dep_df.index.isin(gene1_altcnv_CLLEs)][gene2].mean(skipna=True)
            avg_dep_gene2_wo_gene1_altcnv = dep_df[(~dep_df.index.isin(gene1_altcnv_CLLEs))&(dep_df.index.isin(cancer_CCLE[cancer]))][gene2].mean(skipna=True)
        elif gene2 in extra_map1.keys() and extra_map1[gene2] in dep_df.columns:
            avg_dep_gene2_w_gene1_altcnv = dep_df[dep_df.index.isin(gene1_altcnv_CLLEs)][extra_map1[gene2]].mean(skipna=True)
            avg_dep_gene2_wo_gene1_altcnv = dep_df[(~dep_df.index.isin(gene1_altcnv_CLLEs))&(dep_df.index.isin(cancer_CCLE[cancer]))][extra_map1[gene2]].mean(skipna=True)
        elif gene2 in extra_map2.keys() and extra_map2[gene2] in dep_df.columns:
            avg_dep_gene2_w_gene1_altcnv = dep_df[dep_df.index.isin(gene1_altcnv_CLLEs)][extra_map2[gene2]].mean(skipna=True)
            avg_dep_gene2_wo_gene1_altcnv = dep_df[(~dep_df.index.isin(gene1_altcnv_CLLEs))&(dep_df.index.isin(cancer_CCLE[cancer]))][extra_map2[gene2]].mean(skipna=True)

        #emb_dict[emb_name] = np.array([avg_dep_gene1_w_gene2_altcnv, avg_dep_gene1_wo_gene2_altcnv,
        #                               avg_dep_gene2_w_gene1_altcnv, avg_dep_gene2_wo_gene1_altcnv])
        all_rows[idx]=np.array([avg_dep_gene1_w_gene2_altcnv, avg_dep_gene1_wo_gene2_altcnv,
                                       avg_dep_gene2_w_gene1_altcnv, avg_dep_gene2_wo_gene1_altcnv])
    val_df = pd.DataFrame(all_rows, columns=['dep0', 'dep1', 'dep2', 'dep3'], index=df.index.values)
    df = pd.concat([df, val_df], axis=1)

    return df


def calculate_emb_expr(df, dep_df, expr_df, cancer2ccle, cutoff=1.96, ready_data=None):
    cancer_dict = {'BRCA': 'BREAST', 'CESC': ['OESOPHAGUS','STOMACH'], 'COAD': 'LARGE_INTESTINE', 'KIRC': 'KIDNEY',
                   'LAML':'HAEMATOPOIETIC_AND_LYMPHOID_TISSUE', 'LUAD': 'LUNG',
                   'OV': 'OVARY', 'SKCM': 'SKIN'}

    cancer_dict = {'BRCA': 'Breast Cancer', 'CESC': 'Cervical Cancer', 'COAD': 'Colon/Colorectal Cancer',
                   'KIRC': 'Kidney Cancer', 'LAML': 'Leukemia', 'LUAD': 'Lung Cancer',
                   'OV': 'Ovarian Cancer', 'SKCM': 'Skin Cancer'}
    cancer_CCLE = cancer2ccle
    extra_map1, extra_map2 = get_uniprot_extra_mapping('ccle_broad_2019')
    emb_dict = {}
    all_rows = np.zeros(shape=(len(df),4))
    for idx, a_row in enumerate(df[['gene1', 'gene2', 'cancer']].values):
        if idx%5000==0:
            print(f'Iteration {idx} is done.')
        embedding = []
        gene1 = a_row[0]
        gene2 = a_row[1]
        cancer = a_row[2]
        emb_name = gene1+'|'+gene2+'|'+cancer
        if ready_data is not None and type(ready_data)==dict and (gene1, gene2) in ready_data.keys():
            all_rows[idx] = ready_data[(gene1, gene2)]
        #elif (ready_data is not None):
        #    chosen_row = ready_data[(ready_data[['gene1', 'gene2', 'cancer']] == a_row[0:3]).all(1)]
        #    if len(chosen_row)>0:
        #        emb_dict[emb_name] = chosen_row.loc[:, ['dep0', 'dep1', 'dep2', 'dep3']].values[0]
            continue

        #https: // www.hindawi.com / journals / bmri / 2015 / 901303 /  # materials-and-methods
        gene2_altexpr_CLLEs = np.array([])
        if gene2 in expr_df.columns:
            gene2_altexpr_CLLEs = np.unique(expr_df[(expr_df[gene2] >= cutoff) | (expr_df[gene2] <= -cutoff)].index.values)
        elif gene2 in extra_map1.keys() and extra_map1[gene2] in expr_df.columns:
            gene2_altexpr_CLLEs = np.unique(expr_df[(expr_df[extra_map1[gene2]] >= cutoff) | (expr_df[extra_map1[gene2]] <= -cutoff)].index.values)
        elif gene2 in extra_map2.keys() and extra_map2[gene2] in expr_df.columns:
            gene2_altexpr_CLLEs = np.unique(expr_df[(expr_df[extra_map2[gene2]] >= cutoff) | (expr_df[extra_map2[gene2]] <= -cutoff)].index.values)
        gene2_altexpr_CLLEs = np.intersect1d(gene2_altexpr_CLLEs, cancer_CCLE[cancer])

        gene1_altexpr_CLLEs = np.array([])
        if gene1 in expr_df.columns:
            gene1_altexpr_CLLEs = np.unique(expr_df[(expr_df[gene1] >= cutoff) | (expr_df[gene1] <= -cutoff)].index.values)
        elif gene1 in extra_map1.keys() and extra_map1[gene1] in expr_df.columns:
            gene1_altexpr_CLLEs = np.unique(expr_df[(expr_df[extra_map1[gene1]] >= cutoff) | (expr_df[extra_map1[gene1]] <= -cutoff)].index.values)
        elif gene1 in extra_map2.keys() and extra_map2[gene1] in expr_df.columns:
            gene1_altexpr_CLLEs = np.unique(expr_df[(expr_df[extra_map2[gene1]] >= cutoff) | (expr_df[extra_map2[gene1]] <= -cutoff)].index.values)
        gene1_altexpr_CLLEs = np.intersect1d(gene1_altexpr_CLLEs, cancer_CCLE[cancer])

        avg_dep_gene1_w_gene2_altexpr = np.nan
        avg_dep_gene1_wo_gene2_altexpr = np.nan
        avg_dep_gene2_w_gene1_altexpr = np.nan
        avg_dep_gene2_wo_gene1_altexpr = np.nan

        if gene1 in dep_df.columns:
            avg_dep_gene1_w_gene2_altexpr = dep_df[dep_df.index.isin(gene2_altexpr_CLLEs)][gene1].mean(skipna=True)
            avg_dep_gene1_wo_gene2_altexpr = dep_df[(~dep_df.index.isin(gene2_altexpr_CLLEs))&(dep_df.index.isin(cancer_CCLE[cancer]))][gene1].mean(skipna=True)
        elif gene1 in extra_map1.keys() and extra_map1[gene1] in dep_df.columns:
            avg_dep_gene1_w_gene2_altexpr = dep_df[dep_df.index.isin(gene2_altexpr_CLLEs)][extra_map1[gene1]].mean(skipna=True)
            avg_dep_gene1_wo_gene2_altexpr = dep_df[(~dep_df.index.isin(gene2_altexpr_CLLEs))&(dep_df.index.isin(cancer_CCLE[cancer]))][extra_map1[gene1]].mean(skipna=True)
        elif gene1 in extra_map2.keys() and extra_map2[gene1] in dep_df.columns:
            avg_dep_gene1_w_gene2_altexpr = dep_df[dep_df.index.isin(gene2_altexpr_CLLEs)][extra_map2[gene1]].mean(skipna=True)
            avg_dep_gene1_wo_gene2_altexpr = dep_df[(~dep_df.index.isin(gene2_altexpr_CLLEs))&(dep_df.index.isin(cancer_CCLE[cancer]))][extra_map2[gene1]].mean(skipna=True)


        if gene2 in dep_df.columns:
            avg_dep_gene2_w_gene1_altexpr = dep_df[dep_df.index.isin(gene1_altexpr_CLLEs)][gene2].mean(skipna=True)
            avg_dep_gene2_wo_gene1_altexpr = dep_df[(~dep_df.index.isin(gene1_altexpr_CLLEs))&(dep_df.index.isin(cancer_CCLE[cancer]))][gene2].mean(skipna=True)
        elif gene2 in extra_map1.keys() and extra_map1[gene2] in dep_df.columns:
            avg_dep_gene2_w_gene1_altexpr = dep_df[dep_df.index.isin(gene1_altexpr_CLLEs)][extra_map1[gene2]].mean(skipna=True)
            avg_dep_gene2_wo_gene1_altexpr = dep_df[(~dep_df.index.isin(gene1_altexpr_CLLEs))&(dep_df.index.isin(cancer_CCLE[cancer]))][extra_map1[gene2]].mean(skipna=True)
        elif gene2 in extra_map2.keys() and extra_map2[gene2] in dep_df.columns:
            avg_dep_gene2_w_gene1_altexpr = dep_df[dep_df.index.isin(gene1_altexpr_CLLEs)][extra_map2[gene2]].mean(skipna=True)
            avg_dep_gene2_wo_gene1_altexpr = dep_df[(~dep_df.index.isin(gene1_altexpr_CLLEs))&(dep_df.index.isin(cancer_CCLE[cancer]))][extra_map2[gene2]].mean(skipna=True)

        #emb_dict[emb_name] = np.array([avg_dep_gene1_w_gene2_altexpr, avg_dep_gene1_wo_gene2_altexpr,
        #                               avg_dep_gene2_w_gene1_altexpr, avg_dep_gene2_wo_gene1_altexpr])
        all_rows[idx]=np.array([avg_dep_gene1_w_gene2_altexpr, avg_dep_gene1_wo_gene2_altexpr,
                                       avg_dep_gene2_w_gene1_altexpr, avg_dep_gene2_wo_gene1_altexpr])
    val_df = pd.DataFrame(all_rows, columns=['dep0', 'dep1', 'dep2', 'dep3'], index=df.index.values)
    df = pd.concat([df, val_df], axis=1)

    return df


def calculate_emb_any(pairs, dep_df, mut_df, expr_df, cnv_df, cancer2ccle, cutoff=1.96):
    cancer_dict = {'BRCA': 'BREAST', 'CESC': ['OESOPHAGUS','STOMACH'], 'COAD': 'LARGE_INTESTINE', 'KIRC': 'KIDNEY',
                   'LAML':'HAEMATOPOIETIC_AND_LYMPHOID_TISSUE', 'LUAD': 'LUNG',
                   'OV': 'OVARY', 'SKCM': 'SKIN'}

    cancer_dict = {'BRCA': 'Breast Cancer', 'CESC': 'Cervical Cancer', 'COAD': 'Colon/Colorectal Cancer',
                   'KIRC': 'Kidney Cancer', 'LAML': 'Leukemia', 'LUAD': 'Lung Cancer',
                   'OV': 'Ovarian Cancer', 'SKCM': 'Skin Cancer'}

    cancer_CCLE = cancer2ccle
    extra_map1, extra_map2 = get_uniprot_extra_mapping('ccle_broad_2019')
    grouped_mut = mut_df.groupby('Hugo_Symbol')['Tumor_Sample_Barcode']\
        .agg(list= lambda x: list(np.unique(x))).reset_index()
    gene_mut_ccle = grouped_mut.set_index('Hugo_Symbol')['list'].to_dict()

    emb_dict = {}
    for ind, pair in enumerate(pairs):
        if ind%10000 ==0:
            print(f'Iteration for row {ind} is started for in dep any.')
        embedding = []
        gene1 = pair[0]
        gene2 = pair[1]
        cancer = pair[2]
        emb_name = gene1+'|'+gene2+'|'+cancer

        gene2_mutated_CLLEs = np.array([])
        if gene2 in gene_mut_ccle.keys():
            gene2_mutated_CLLEs = np.array(gene_mut_ccle[gene2])
        elif gene2 in extra_map1.keys() and extra_map1[gene2] in gene_mut_ccle.keys():
            gene2_mutated_CLLEs = np.array(gene_mut_ccle[extra_map1[gene2]])
        elif gene2 in extra_map2.keys() and extra_map2[gene2] in gene_mut_ccle.keys():
            gene2_mutated_CLLEs = np.array(gene_mut_ccle[extra_map2[gene2]])

        gene2_altexpr_CLLEs = np.array([])
        if gene2 in expr_df.columns:
            gene2_altexpr_CLLEs = np.unique(expr_df[(expr_df[gene2] >= cutoff) | (expr_df[gene2] <= -cutoff)].index.values)
        elif gene2 in extra_map1.keys() and extra_map1[gene2] in expr_df.columns:
            gene2_altexpr_CLLEs = np.unique(expr_df[(expr_df[extra_map1[gene2]] >= cutoff) | (expr_df[extra_map1[gene2]] <= -cutoff)].index.values)
        elif gene2 in extra_map2.keys() and extra_map2[gene2] in expr_df.columns:
            gene2_altexpr_CLLEs = np.unique(expr_df[(expr_df[extra_map2[gene2]] >= cutoff) | (expr_df[extra_map2[gene2]] <= -cutoff)].index.values)

        gene2_altcnv_CLLEs = np.array([])
        if gene2 in cnv_df.columns:
            gene2_altcnv_CLLEs = np.unique(cnv_df[(cnv_df[gene2] >= 2) | (cnv_df[gene2] <= -2)].index.values)
        elif gene2 in extra_map1.keys() and extra_map1[gene2] in cnv_df.columns:
            gene2_altcnv_CLLEs = np.unique(cnv_df[(cnv_df[extra_map1[gene2]] >= 2) | (cnv_df[extra_map1[gene2]] <= -2)].index.values)
        elif gene2 in extra_map2.keys() and extra_map2[gene2] in cnv_df.columns:
            gene2_altcnv_CLLEs = np.unique(cnv_df[(cnv_df[extra_map2[gene2]] >= 2) | (cnv_df[extra_map2[gene2]] <= -2)].index.values)
        gene2_alt_CLLEs = np.union1d(gene2_mutated_CLLEs, gene2_altexpr_CLLEs)
        gene2_alt_CLLEs = np.union1d(gene2_alt_CLLEs, gene2_altcnv_CLLEs)
        gene2_alt_CLLEs = np.intersect1d(gene2_alt_CLLEs, cancer_CCLE[cancer])


        gene1_mutated_CLLEs = np.array([])
        if gene1 in gene_mut_ccle.keys():
            gene1_mutated_CLLEs = np.array(gene_mut_ccle[gene1])
        elif gene1 in extra_map1.keys() and extra_map1[gene1] in gene_mut_ccle.keys():
            gene1_mutated_CLLEs = np.array(gene_mut_ccle[extra_map1[gene1]])
        elif gene1 in extra_map2.keys() and extra_map2[gene1] in gene_mut_ccle.keys():
            gene1_mutated_CLLEs = np.array(gene_mut_ccle[extra_map2[gene1]])

        gene1_altexpr_CLLEs = np.array([])
        if gene1 in expr_df.columns:
            gene1_altexpr_CLLEs = np.unique(expr_df[(expr_df[gene1] >= cutoff) | (expr_df[gene1] <= -cutoff)].index.values)
        elif gene1 in extra_map1.keys() and extra_map1[gene1] in expr_df.columns:
            gene1_altexpr_CLLEs = np.unique(expr_df[(expr_df[extra_map1[gene1]] >= cutoff) | (expr_df[extra_map1[gene1]] <= -cutoff)].index.values)
        elif gene1 in extra_map2.keys() and extra_map2[gene1] in expr_df.columns:
            gene1_altexpr_CLLEs = np.unique(expr_df[(expr_df[extra_map2[gene1]] >= cutoff) | (expr_df[extra_map2[gene1]] <= -cutoff)].index.values)

        gene1_altcnv_CLLEs = np.array([])
        if gene1 in cnv_df.columns:
            gene1_altcnv_CLLEs = np.unique(cnv_df[(cnv_df[gene1] >= cutoff) | (cnv_df[gene1] <= -cutoff)].index.values)
        elif gene1 in extra_map1.keys() and extra_map1[gene1] in cnv_df.columns:
            gene1_altcnv_CLLEs = np.unique(cnv_df[(cnv_df[extra_map1[gene1]] >= cutoff) | (cnv_df[extra_map1[gene1]] <= -cutoff)].index.values)
        elif gene1 in extra_map2.keys() and extra_map2[gene1] in cnv_df.columns:
            gene1_altcnv_CLLEs = np.unique(cnv_df[(cnv_df[extra_map2[gene1]] >= cutoff) | (cnv_df[extra_map2[gene1]] <= -cutoff)].index.values)
        gene1_alt_CLLEs = np.union1d(gene1_mutated_CLLEs, gene1_altexpr_CLLEs)
        gene1_alt_CLLEs = np.union1d(gene1_alt_CLLEs, gene1_altcnv_CLLEs)
        gene1_alt_CLLEs = np.intersect1d(gene1_alt_CLLEs, cancer_CCLE[cancer])

        avg_dep_gene1_w_gene2_alt = np.nan
        avg_dep_gene1_wo_gene2_alt = np.nan
        avg_dep_gene2_w_gene1_alt = np.nan
        avg_dep_gene2_wo_gene1_alt = np.nan

        if gene1 in dep_df.columns:
            avg_dep_gene1_w_gene2_alt = dep_df[dep_df.index.isin(gene2_alt_CLLEs)][gene1].mean(skipna=True)
            avg_dep_gene1_wo_gene2_alt = dep_df[(~dep_df.index.isin(gene2_alt_CLLEs))&(dep_df.index.isin(cancer_CCLE[cancer]))][gene1].mean(skipna=True)

        if gene2 in dep_df.columns:
            avg_dep_gene2_w_gene1_alt = dep_df[dep_df.index.isin(gene1_alt_CLLEs)][gene2].mean(skipna=True)
            avg_dep_gene2_wo_gene1_alt = dep_df[(~dep_df.index.isin(gene1_alt_CLLEs))&(dep_df.index.isin(cancer_CCLE[cancer]))][gene2].mean(skipna=True)

        emb_dict[emb_name] = np.array([avg_dep_gene1_w_gene2_alt, avg_dep_gene1_wo_gene2_alt,
                                       avg_dep_gene2_w_gene1_alt, avg_dep_gene2_wo_gene1_alt])

    return emb_dict




def calculate_emb2_mut(pairs, dep_df, mut_df, cancer2ccle):
    cancer_dict = {'BRCA': 'BREAST', 'CESC': ['OESOPHAGUS','STOMACH'], 'COAD': 'LARGE_INTESTINE', 'KIRC': 'KIDNEY',
                   'LAML':'HAEMATOPOIETIC_AND_LYMPHOID_TISSUE', 'LUAD': 'LUNG',
                   'OV': 'OVARY', 'SKCM': 'SKIN'}

    cancer_dict = {'BRCA': 'Breast Cancer', 'CESC': 'Cervical Cancer', 'COAD': 'Colon/Colorectal Cancer',
                   'KIRC': 'Kidney Cancer', 'LAML': 'Leukemia', 'LUAD': 'Lung Cancer',
                   'OV': 'Ovarian Cancer', 'SKCM': 'Skin Cancer'}
    cancer_CCLE = cancer2ccle
    extra_map1, extra_map2 = get_uniprot_extra_mapping('ccle_broad_2019')
    emb_dict = {}
    for pair in pairs:
        embedding = []
        gene1 = pair[0]
        gene2 = pair[1]
        cancer = pair[2]
        emb_name = gene1+'|'+gene2+'|'+cancer
        gene2_mutated_CLLEs = np.unique(mut_df[mut_df['Hugo_Symbol'] == gene2]['DepMap_ID'].values)
        gene2_mutated_CLLEs = np.intersect1d(gene2_mutated_CLLEs, cancer_CCLE[cancer])
        gene1_mutated_CLLEs = np.unique(mut_df[mut_df['Hugo_Symbol'] == gene1]['DepMap_ID'].values)
        gene1_mutated_CLLEs = np.intersect1d(gene1_mutated_CLLEs, cancer_CCLE[cancer])
        avg_dep_gene1_w_gene2_mut = np.nan
        avg_dep_gene1_wo_gene2_mut = np.nan
        avg_dep_gene2_w_gene1_mut = np.nan
        avg_dep_gene2_wo_gene1_mut = np.nan
        if gene1 in dep_df.columns:
            avg_dep_gene1_w_gene2_mut = dep_df[dep_df.index.isin(gene2_mutated_CLLEs)][gene1].mean(skipna=True)
            avg_dep_gene1_wo_gene2_mut = dep_df[(~dep_df.index.isin(gene2_mutated_CLLEs))&(dep_df.index.isin(cancer_CCLE[cancer]))][gene1].mean(skipna=True)
        if gene2 in dep_df.columns:
            avg_dep_gene2_w_gene1_mut = dep_df[dep_df.index.isin(gene1_mutated_CLLEs)][gene2].mean(skipna=True)
            avg_dep_gene2_wo_gene1_mut = dep_df[(~dep_df.index.isin(gene1_mutated_CLLEs))&(dep_df.index.isin(cancer_CCLE[cancer]))][gene2].mean(skipna=True)

        emb_dict[emb_name] = np.array([avg_dep_gene1_w_gene2_mut, avg_dep_gene1_wo_gene2_mut,
                                       avg_dep_gene2_w_gene1_mut, avg_dep_gene2_wo_gene1_mut])

    return emb_dict



def main():
    loc_dict = get_locs()
    sample_df = cb.load_cell_info(loc_dict['sample_info'])
    id2name, name2id = cb.get_ccle_maps(sample_df, coll.OrderedDict({'id': 'name', 'name': 'id'}))
    mut_df = cb.load_mut(loc_dict['mutation'])
    expr_df = cb.load_sample_gene_data(loc_dict['expression'])
    cnv_df = cb.load_sample_gene_data(loc_dict['cnv'])
    rnai_dep_df = cb.load_rnai_dep(loc_dict['rnai_dependency'], name2id)
    crispr_dep_df = cb.load_crispr_dep(loc_dict['crispr_dependency'])
    return 0


if __name__ == '__main__':
    main()
