import sys
import numpy as np
import collections as coll
from src.datasets import tissue
from src import config
import pandas as pd
import os
from lifelines import CoxPHFitter
from scipy.stats import pearsonr, spearmanr
from scipy import stats
from joblib import Parallel, delayed
import multiprocessing
import time
import warnings
import logging
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
#warnings.filterwarnings("ignore", category=DtypeWarning)

def get_locs():
    loc_dict = coll.OrderedDict()
    for cancer in ['BRCA', 'CESC', 'COAD', 'KIRC', 'LAML', 'LUAD', 'OV', 'SKCM']:
        loc_dict[cancer]={}
        loc_dict[cancer]['mutation'] = 'tissue_data/'+cancer.lower()+'_tcga/data_mutations_extended.txt'
        loc_dict[cancer]['std_expression'] = 'tissue_data/'+cancer.lower()+'_tcga/data_RNA_Seq_v2_mRNA_median_all_sample_Zscores.txt'
        loc_dict[cancer]['cnv'] = 'tissue_data/'+cancer.lower()+'_tcga/data_CNA.txt'
        loc_dict[cancer]['sample_info'] = 'tissue_data/'+cancer.lower()+'_tcga/data_bcr_clinical_data_sample.txt'
        loc_dict[cancer]['patient_info'] = 'tissue_data/'+cancer.lower()+'_tcga/data_bcr_clinical_data_patient.txt'
        loc_dict[cancer]['expression'] = 'tissue_data/'+cancer.lower()+'_tcga/'+cancer+'.rnaseqv2_norm.txt'
        loc_dict[cancer]['gtex_expression'] = 'tissue_data/'+cancer.lower()+'_gtex/'+cancer.lower()+'_RNASeQ_tpm.csv.gz'
    loc_dict['gtex_attribute'] = 'tissue_data/gtex/GTEx_Analysis_v8_Annotations_SampleAttributesDS.txt'
    loc_dict['gtex_expression'] = 'tissue_data/gtex/GTEx_Analysis_2017-06-05_v8_RNASeQCv1.1.9_gene_tpm.gct.gz'
    return loc_dict

def get_uniprot_extra_mapping(folder):
    loc = config.DATA_DIR / folder / 'uniprot_extra_mapping.tab'
    if os.path.isfile(loc):
        data = pd.read_csv(loc, sep='\t')
    else:
        return None
    data.columns = ['query', 'uniprot_id', 'uniprot_name', 'status', 'prot_names', 'gene_names', 'organism', 'primary_gene_names', 'synonym_gene_names']
    reviewed = data[data['status'] == 'reviewed']

    reviewed['uniprot_name'] = reviewed['uniprot_name'].str.split('_', n=1, expand=True)[0]

    query2uniprot = reviewed.set_index('query')['uniprot_name'].to_dict()
    query2genename = reviewed.set_index('query')['primary_gene_names'].to_dict()
    return query2uniprot, query2genename

def get_dataset(name, cancer=None):
    loc_dict = get_locs()

    if name == 'patients':
        file_loc = tissue.config.DATA_DIR / loc_dict[cancer]['patient_info']
        sample_df = tissue.load_patient_info(file_loc)
        return sample_df
    elif name == 'samples':
        file_loc = tissue.config.DATA_DIR / loc_dict[cancer]['sample_info']
        sample_df = tissue.load_patient_info(file_loc)
    elif name == 'mappings':
        uniprot_extra_mapping_loc = 'tissue_data/'+cancer+'_tcga/uniprot_extra_mapping.tab'
        uniprot_extra1, uniprot_extra2 = get_uniprot_extra_mapping(uniprot_extra_mapping_loc)
    elif name == 'mutation':
        return tissue.load_mut(loc_dict[cancer]['mutation'])
    elif name == 'expression':
        expr_df = tissue.load_all_expr(loc_dict[cancer]['expression']).dropna(axis=0, how='all')
        tumor_pats = [int(row[3][0:2]) < 10 for row in expr_df.columns.str.split('-').tolist()]
        expr_df = expr_df.T
        expr_df.columns = expr_df.columns.values.astype(str)
        if 'nan' in expr_df.columns.values:
            expr_df = expr_df.drop(columns=['nan'])
        uniq_cols = np.unique(expr_df.columns.values, return_index=True, return_counts=True)
        dupl_indices = np.argwhere(uniq_cols[2] > 1)[:, 0]
        col_ids = uniq_cols[1]
        col_names = uniq_cols[0]
        for dupl_pos in dupl_indices:
            max_var_id = np.argmax(expr_df[col_names[dupl_pos]].var().values)
            chosen_id = col_ids[dupl_pos] + max_var_id
            col_ids[dupl_pos] = chosen_id

        expr_df = expr_df.iloc[:, col_ids].T
        t_expr_df = expr_df[expr_df.columns[tumor_pats]].T
        n_expr_df = expr_df[expr_df.columns[~np.array(tumor_pats)]].T
        return t_expr_df, n_expr_df
    elif name == 'healthy_expression':
        if cancer==None:
            expr_df = tissue.load_gtex_expr(loc_dict['gtex_expression'])
            cancer_pat_dict = tissue.load_gtex_sample_info(loc_dict['gtex_attribute'])
            cancer2expr = {}
            for cancer, pats in cancer_pat_dict.items():
                col_list = np.union1d(['Name', 'Description'], pats)
                col_list = np.intersect1d(expr_df.columns.values, col_list)
                cancer2expr[cancer] = expr_df[col_list]
            return cancer2expr
        else:
            expr_df = tissue.load_cancer_gtex_expr(loc_dict[cancer]['gtex_expression']).T.dropna(axis=1, how='all')
            expr_df.columns = expr_df.columns.values.astype(str)
            if 'nan' in expr_df.columns.values:
                expr_df = expr_df.drop(columns=['nan'])
            uniq_cols = np.unique(expr_df.columns.values, return_index=True, return_counts=True)
            dupl_indices = np.argwhere(uniq_cols[2] > 1)[:, 0]
            col_ids = uniq_cols[1]
            col_names = uniq_cols[0]
            for dupl_pos in dupl_indices:
                max_var_id = np.argmax(expr_df[col_names[dupl_pos]].var().values)
                chosen_id = col_ids[dupl_pos] + max_var_id
                col_ids[dupl_pos] = chosen_id

            expr_df = expr_df.iloc[:, col_ids]
            return expr_df
    elif name == 'std_expression':
        tmp = tissue.load_expr_cnv(loc_dict[cancer]['std_expression']).T.dropna(axis=1, how='all')
        tmp.columns = tmp.columns.values.astype(str)
        if 'nan' in tmp.columns.values:
            tmp = tmp.drop(columns=['nan'])
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
        tmp = tissue.load_expr_cnv(loc_dict[cancer]['cnv']).T.dropna(axis=1, how='all')
        tmp.columns = tmp.columns.values.astype(str)
        if 'nan' in tmp.columns.values:
            tmp = tmp.drop(columns=['nan'])
        uniq_cols = np.unique(tmp.columns.values, return_index=True, return_counts=True)
        dupl_indices = np.argwhere(uniq_cols[2] > 1)[:,0]
        col_ids = uniq_cols[1]
        col_names = uniq_cols[0]
        for dupl_pos in dupl_indices:
            max_var_id = np.argmax(tmp[col_names[dupl_pos]].var().values)
            chosen_id = col_ids[dupl_pos]+max_var_id
            col_ids[dupl_pos]=chosen_id
        return tmp.iloc[:,col_ids]


def calculate_surv_any(df, cancer_aim, mut_df, std_expr_df, cnv_df, sample_df, cutoff, ready_data):
    uniprot_extra_mapping_loc = 'tissue_data/'+cancer_aim.lower()+'_tcga'
    ext_map = get_uniprot_extra_mapping(uniprot_extra_mapping_loc)
    if ext_map is None:
        print(f'No extra mapping')
    else:
        extra_map1, extra_map2 = ext_map
    grouped_mut = mut_df.groupby('Hugo_Symbol')['Tumor_Sample_Barcode']\
        .agg(list= lambda x: list(np.unique(x))).reset_index()
    gene_mut_pats = grouped_mut.set_index('Hugo_Symbol')['list'].to_dict()
    sample_df['AGE'] = sample_df['AGE'].astype(str)
    sample_df['OS_STATUS'] = sample_df['OS_STATUS'].astype(str)
    sample_df['OS_MONTHS'] = sample_df['OS_MONTHS'].astype(str)

    sample_df = sample_df[(sample_df['AGE'] != '[Not Available]') & (sample_df['OS_STATUS'] != '[Not Available]') &
                          (sample_df['OS_MONTHS'] != '[Not Available]')]
    sample_df['AGE'] = sample_df['AGE'].astype(int)
    sample_df['OS_STATUS'] = sample_df['OS_STATUS'].astype(int)
    sample_df['OS_MONTHS'] = sample_df['OS_MONTHS'].astype(float)
    age_strata = pd.qcut(x=sample_df['AGE'], q=[0, .25, .5, .75, 1.])
    sample_df = sample_df.drop(columns=['AGE', 'ETHNICITY'])
    sample_df['AGE_STRATA'] = age_strata
    #sample_df['PATIENT_ID'] = sample_df['PATIENT_ID'].str[:-3]


    cols = ['surv']
    if cols[0] in df.columns:
        all_rows=df[cols].values
        df = df.drop(columns=cols)
    else:
        all_rows = np.zeros(shape=(len(df),1))

    start = time.time()
    print("Survival non-paralel")
    for ind, a_row in enumerate(df[['gene1', 'gene2', 'cancer']].values):
        if ind%10000 ==0:
            print(f'Iteration for row {ind} is started for {cancer_aim} in survival embeddings.')
            logging.info(f'Iteration for row {ind} is started for {cancer_aim} in survival embeddings.')
        gene1 = a_row[0]
        gene2 = a_row[1]
        cancer = a_row[2]
        if cancer != cancer_aim:
            continue
        if ready_data is not None and type(ready_data)==dict and (gene1, gene2) in ready_data.keys():
            all_rows[ind,0] = ready_data[(gene1, gene2)]['surv']
            continue
        elif (ready_data is not None) and type(ready_data)!=dict:
            chosen_row = ready_data[(ready_data[['gene1', 'gene2', 'cancer']] == a_row[0:3].values).all(1)]
            if ready_data is not None and len(chosen_row)>0:
                df.loc[ind, 'surv'] = chosen_row.loc[:, 'surv'].values[0]
                continue
        #--------Mutated Patients--------#
        gene2_mutated_pats = np.array([])
        if gene2 in gene_mut_pats.keys():
            gene2_mutated_pats = np.array(gene_mut_pats[gene2])
        elif ext_map is not None and gene2 in extra_map1.keys() and extra_map1[gene2] in gene_mut_pats.keys():
            gene2_mutated_pats = np.array(gene_mut_pats[extra_map1[gene2]])
        elif ext_map is not None and gene2 in extra_map2.keys() and extra_map2[gene2] in gene_mut_pats.keys():
            gene2_mutated_pats = np.array(gene_mut_pats[extra_map2[gene2]])
        gene1_mutated_pats = np.array([])
        if gene1 in gene_mut_pats.keys():
            gene1_mutated_pats = np.array(gene_mut_pats[gene1])
        elif ext_map is not None and gene1 in extra_map1.keys() and extra_map1[gene1] in gene_mut_pats.keys():
            gene1_mutated_pats = np.array(gene_mut_pats[extra_map1[gene1]])
        elif ext_map is not None and gene1 in extra_map2.keys() and extra_map2[gene1] in gene_mut_pats.keys():
            gene1_mutated_pats = np.array(gene_mut_pats[extra_map2[gene1]])
        #gene1_2_mutated_pats = np.intersect1d(gene1_mutated_pats, gene2_mutated_pats)
        #--------Mutated Patients--------#


        #--------Altered CNV Patients--------#
        gene2_altcnv_pats = np.array([])
        if gene2 in cnv_df.columns:
            gene2_altcnv_pats = np.unique(cnv_df[(cnv_df[gene2] >= 2) | (cnv_df[gene2] <= -2)].index.values)
        elif ext_map is not None and gene2 in extra_map1.keys() and extra_map1[gene2] in cnv_df.columns:
            gene2_altcnv_pats = np.unique(cnv_df[(cnv_df[extra_map1[gene2]] >= 2) | (cnv_df[extra_map1[gene2]] <= -2)].index.values)
        elif ext_map is not None and gene2 in extra_map2.keys() and extra_map2[gene2] in cnv_df.columns:
            gene2_altcnv_pats = np.unique(cnv_df[(cnv_df[extra_map2[gene2]] >= 2) | (cnv_df[extra_map2[gene2]] <= -2)].index.values)
        gene1_altcnv_pats = np.array([])
        if gene1 in cnv_df.columns:
            gene1_altcnv_pats = np.unique(cnv_df[(cnv_df[gene1] >= 2) | (cnv_df[gene1] <= -2)].index.values)
        elif ext_map is not None and gene1 in extra_map1.keys() and extra_map1[gene1] in cnv_df.columns:
            gene1_altcnv_pats = np.unique(cnv_df[(cnv_df[extra_map1[gene1]] >= 2) | (cnv_df[extra_map1[gene1]] <= -2)].index.values)
        elif ext_map is not None and gene1 in extra_map2.keys() and extra_map2[gene1] in cnv_df.columns:
            gene1_altcnv_pats = np.unique(cnv_df[(cnv_df[extra_map2[gene1]] >= 2) | (cnv_df[extra_map2[gene1]] <= -2)].index.values)
        #gene1_2_altcnv_pats = np.intersect1d(gene1_altcnv_pats, gene2_altcnv_pats)
        #--------Altered CNV Patients--------#


        #--------Altered Expr Patients--------#
        gene2_altexpr_pats = np.array([])
        if gene2 in std_expr_df.columns:
            gene2_altexpr_pats = np.unique(std_expr_df[(std_expr_df[gene2] >= cutoff) | (std_expr_df[gene2] <= -cutoff)].index.values)
        elif ext_map is not None and gene2 in extra_map1.keys() and extra_map1[gene2] in std_expr_df.columns:
            gene2_altexpr_pats = np.unique(std_expr_df[(std_expr_df[extra_map1[gene2]] >= cutoff) | (std_expr_df[extra_map1[gene2]] <= -cutoff)].index.values)
        elif ext_map is not None and gene2 in extra_map2.keys() and extra_map2[gene2] in std_expr_df.columns:
            gene2_altexpr_pats = np.unique(std_expr_df[(std_expr_df[extra_map2[gene2]] >= cutoff) | (std_expr_df[extra_map2[gene2]] <= -cutoff)].index.values)

        gene1_altexpr_pats = np.array([])
        if gene1 in std_expr_df.columns:
            gene1_altexpr_pats = np.unique(std_expr_df[(std_expr_df[gene1] >= cutoff) | (std_expr_df[gene1] <= -cutoff)].index.values)
        elif ext_map is not None and gene1 in extra_map1.keys() and extra_map1[gene1] in std_expr_df.columns:
            gene1_altexpr_pats = np.unique(std_expr_df[(std_expr_df[extra_map1[gene1]] >= cutoff) | (std_expr_df[extra_map1[gene1]] <= -cutoff)].index.values)
        elif ext_map is not None and gene1 in extra_map2.keys() and extra_map2[gene1] in std_expr_df.columns:
            gene1_altexpr_pats = np.unique(std_expr_df[(std_expr_df[extra_map2[gene1]] >= cutoff) | (std_expr_df[extra_map2[gene1]] <= -cutoff)].index.values)
        #gene1_2_altexpr_pats = np.intersect1d(gene1_altexpr_pats, gene2_altexpr_pats)
        #--------Altered Expr Patients--------#
        gene1_alt_pats = np.union1d(gene1_mutated_pats, gene1_altexpr_pats)
        gene1_alt_pats = np.union1d(gene1_alt_pats, gene1_altcnv_pats)
        gene2_alt_pats = np.union1d(gene2_mutated_pats, gene2_altexpr_pats)
        gene2_alt_pats = np.union1d(gene2_alt_pats, gene2_altcnv_pats)

        gene_1_2_alt_pats = np.intersect1d(gene1_alt_pats, gene2_alt_pats)
        gene_1_2_alt_pats = np.array([x[:-3] for x in gene_1_2_alt_pats])


        if len(np.intersect1d(sample_df['PATIENT_ID'].values, gene_1_2_alt_pats)) in [0,len(sample_df['PATIENT_ID'].values)]:
            surv_p_val = 1
        else:
            res_df = sample_df.copy()
            res_df['mutated'] = 0
            res_df.loc[res_df['PATIENT_ID'].isin(gene_1_2_alt_pats), 'mutated'] = 1
            res_df = res_df.set_index('PATIENT_ID')
            # Create the Cox model  'SEX', 'RACE', 'ETHNICITY', 'AGE', 'OS_STATUS', 'OS_MONTHS'
            cph_model = CoxPHFitter(strata=['SEX', 'RACE', 'AGE_STRATA'])
            # Train the model on the data set
            try:
                cph_model.fit(df=res_df, duration_col='OS_MONTHS', event_col='OS_STATUS')
                surv_p_val = cph_model.summary.loc['mutated','p']
            except:
                print(sys.exc_info()[0])
                surv_p_val=1
            # Print the model summary
            #cph_model.print_summary()
            #surv_p_val = cph_model.summary.loc['mutated','p']

        #df.loc[ind, 'surv'] = surv_p_val
        all_rows[ind,0]=surv_p_val

    end = time.time()
    print(end - start)
    val_df = pd.DataFrame(all_rows, columns=cols, index=df.index.values)
    df = pd.concat([df, val_df], axis=1)

    return df


def calculate_surv_any_para(df, cancer_aim, mut_df, std_expr_df, cnv_df, sample_df, cutoff, ready_data):
    uniprot_extra_mapping_loc = 'tissue_data/'+cancer_aim.lower()+'_tcga'
    ext_map = get_uniprot_extra_mapping(uniprot_extra_mapping_loc)
    if ext_map is None:
        print(f'No extra mapping')
    else:
        extra_map1, extra_map2 = ext_map
    grouped_mut = mut_df.groupby('Hugo_Symbol')['Tumor_Sample_Barcode']\
        .agg(list=lambda x: list(np.unique(x))).reset_index()
    gene_mut_pats = grouped_mut.set_index('Hugo_Symbol')['list'].to_dict()
    sample_df['AGE'] = sample_df['AGE'].astype(str)
    sample_df['OS_STATUS'] = sample_df['OS_STATUS'].astype(str)
    sample_df['OS_MONTHS'] = sample_df['OS_MONTHS'].astype(str)

    sample_df = sample_df[(sample_df['AGE'] != '[Not Available]') & (sample_df['OS_STATUS'] != '[Not Available]') &
                          (sample_df['OS_MONTHS'] != '[Not Available]')]
    sample_df['AGE'] = sample_df['AGE'].astype(int)
    sample_df['OS_STATUS'] = sample_df['OS_STATUS'].astype(int)
    sample_df['OS_MONTHS'] = sample_df['OS_MONTHS'].astype(float)
    age_strata = pd.qcut(x=sample_df['AGE'], q=[0, .25, .5, .75, 1.])
    sample_df = sample_df.drop(columns=['AGE', 'ETHNICITY'])
    sample_df['AGE_STRATA'] = age_strata
    #sample_df['PATIENT_ID'] = sample_df['PATIENT_ID'].str[:-3]


    cols = ['surv']
    if cols[0] in df.columns:
        all_rows=df[cols].values
        df = df.drop(columns=cols)
    else:
        all_rows = np.ones(shape=(len(df),1))

    def parallel_assign(cancer_aim, gene_mut_pats, std_expr_df, cnv_df, sample_df, cutoff, ind, a_row):
        warnings.filterwarnings("ignore", category=UserWarning)
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        warnings.filterwarnings("ignore", category=FutureWarning)
        if ind%10000 ==0:
            print(f'Iteration for row {ind} is started for {cancer_aim} in survival embeddings.')
            logging.info(f'Iteration for row {ind} is started for {cancer_aim} in survival embeddings.')
        gene1 = a_row[0]
        gene2 = a_row[1]
        cancer = a_row[2]

        if cancer != cancer_aim:
            return np.nan
        #--------Mutated Patients--------#
        gene2_mutated_pats = np.array([])
        if gene2 in gene_mut_pats.keys():
            gene2_mutated_pats = np.array(gene_mut_pats[gene2])
        elif ext_map is not None and gene2 in extra_map1.keys() and extra_map1[gene2] in gene_mut_pats.keys():
            gene2_mutated_pats = np.array(gene_mut_pats[extra_map1[gene2]])
        elif ext_map is not None and gene2 in extra_map2.keys() and extra_map2[gene2] in gene_mut_pats.keys():
            gene2_mutated_pats = np.array(gene_mut_pats[extra_map2[gene2]])
        gene1_mutated_pats = np.array([])
        if gene1 in gene_mut_pats.keys():
            gene1_mutated_pats = np.array(gene_mut_pats[gene1])
        elif ext_map is not None and gene1 in extra_map1.keys() and extra_map1[gene1] in gene_mut_pats.keys():
            gene1_mutated_pats = np.array(gene_mut_pats[extra_map1[gene1]])
        elif ext_map is not None and gene1 in extra_map2.keys() and extra_map2[gene1] in gene_mut_pats.keys():
            gene1_mutated_pats = np.array(gene_mut_pats[extra_map2[gene1]])
        #gene1_2_mutated_pats = np.intersect1d(gene1_mutated_pats, gene2_mutated_pats)
        #--------Mutated Patients--------#


        #--------Altered CNV Patients--------#
        gene2_altcnv_pats = np.array([])
        if gene2 in cnv_df.columns:
            gene2_altcnv_pats = np.unique(cnv_df[(cnv_df[gene2] >= 2) | (cnv_df[gene2] <= -2)].index.values)
        elif ext_map is not None and gene2 in extra_map1.keys() and extra_map1[gene2] in cnv_df.columns:
            gene2_altcnv_pats = np.unique(cnv_df[(cnv_df[extra_map1[gene2]] >= 2) | (cnv_df[extra_map1[gene2]] <= -2)].index.values)
        elif ext_map is not None and gene2 in extra_map2.keys() and extra_map2[gene2] in cnv_df.columns:
            gene2_altcnv_pats = np.unique(cnv_df[(cnv_df[extra_map2[gene2]] >= 2) | (cnv_df[extra_map2[gene2]] <= -2)].index.values)
        gene1_altcnv_pats = np.array([])
        if gene1 in cnv_df.columns:
            gene1_altcnv_pats = np.unique(cnv_df[(cnv_df[gene1] >= 2) | (cnv_df[gene1] <= -2)].index.values)
        elif ext_map is not None and gene1 in extra_map1.keys() and extra_map1[gene1] in cnv_df.columns:
            gene1_altcnv_pats = np.unique(cnv_df[(cnv_df[extra_map1[gene1]] >= 2) | (cnv_df[extra_map1[gene1]] <= -2)].index.values)
        elif ext_map is not None and gene1 in extra_map2.keys() and extra_map2[gene1] in cnv_df.columns:
            gene1_altcnv_pats = np.unique(cnv_df[(cnv_df[extra_map2[gene1]] >= 2) | (cnv_df[extra_map2[gene1]] <= -2)].index.values)
        #gene1_2_altcnv_pats = np.intersect1d(gene1_altcnv_pats, gene2_altcnv_pats)
        #--------Altered CNV Patients--------#


        #--------Altered Expr Patients--------#
        gene2_altexpr_pats = np.array([])
        if gene2 in std_expr_df.columns:
            gene2_altexpr_pats = np.unique(std_expr_df[(std_expr_df[gene2] >= cutoff) | (std_expr_df[gene2] <= -cutoff)].index.values)
        elif ext_map is not None and gene2 in extra_map1.keys() and extra_map1[gene2] in std_expr_df.columns:
            gene2_altexpr_pats = np.unique(std_expr_df[(std_expr_df[extra_map1[gene2]] >= cutoff) | (std_expr_df[extra_map1[gene2]] <= -cutoff)].index.values)
        elif ext_map is not None and gene2 in extra_map2.keys() and extra_map2[gene2] in std_expr_df.columns:
            gene2_altexpr_pats = np.unique(std_expr_df[(std_expr_df[extra_map2[gene2]] >= cutoff) | (std_expr_df[extra_map2[gene2]] <= -cutoff)].index.values)

        gene1_altexpr_pats = np.array([])
        if gene1 in std_expr_df.columns:
            gene1_altexpr_pats = np.unique(std_expr_df[(std_expr_df[gene1] >= cutoff) | (std_expr_df[gene1] <= -cutoff)].index.values)
        elif ext_map is not None and gene1 in extra_map1.keys() and extra_map1[gene1] in std_expr_df.columns:
            gene1_altexpr_pats = np.unique(std_expr_df[(std_expr_df[extra_map1[gene1]] >= cutoff) | (std_expr_df[extra_map1[gene1]] <= -cutoff)].index.values)
        elif ext_map is not None and gene1 in extra_map2.keys() and extra_map2[gene1] in std_expr_df.columns:
            gene1_altexpr_pats = np.unique(std_expr_df[(std_expr_df[extra_map2[gene1]] >= cutoff) | (std_expr_df[extra_map2[gene1]] <= -cutoff)].index.values)
        #gene1_2_altexpr_pats = np.intersect1d(gene1_altexpr_pats, gene2_altexpr_pats)
        #--------Altered Expr Patients--------#
        gene1_alt_pats = np.union1d(gene1_mutated_pats, gene1_altexpr_pats)
        gene1_alt_pats = np.union1d(gene1_alt_pats, gene1_altcnv_pats)
        gene2_alt_pats = np.union1d(gene2_mutated_pats, gene2_altexpr_pats)
        gene2_alt_pats = np.union1d(gene2_alt_pats, gene2_altcnv_pats)

        gene_1_2_alt_pats = np.intersect1d(gene1_alt_pats, gene2_alt_pats)
        gene_1_2_alt_pats = np.array([x[:-3] for x in gene_1_2_alt_pats])


        if len(np.intersect1d(sample_df['PATIENT_ID'].values, gene_1_2_alt_pats)) in [0,len(sample_df['PATIENT_ID'].values)]:
            surv_p_val = 1
        else:
            res_df = sample_df.copy()
            res_df['mutated'] = 0
            res_df.loc[res_df['PATIENT_ID'].isin(gene_1_2_alt_pats), 'mutated'] = 1
            res_df = res_df.set_index('PATIENT_ID')
            # Create the Cox model  'SEX', 'RACE', 'ETHNICITY', 'AGE', 'OS_STATUS', 'OS_MONTHS'
            cph_model = CoxPHFitter(strata=['SEX', 'RACE', 'AGE_STRATA'])
            # Train the model on the data set
            try:
                cph_model.fit(df=res_df, duration_col='OS_MONTHS', event_col='OS_STATUS')
                surv_p_val = cph_model.summary.loc['mutated','p']
            except:
                surv_p_val=1
            # Print the model summary
            #cph_model.print_summary()
            #surv_p_val = cph_model.summary.loc['mutated','p']

        #df.loc[ind, 'surv'] = surv_p_val
        #all_rows[ind,0]=surv_p_val
        return surv_p_val


    #for ind, a_row in enumerate(df[['gene1', 'gene2', 'cancer']].values):
    #    parallel_assign(ind, a_row)

    start = time.time()
    print("Survival paralel")
    res = Parallel(backend='loky', n_jobs=16, verbose=1)(delayed(parallel_assign)(cancer_aim, gene_mut_pats, std_expr_df, cnv_df, sample_df, cutoff, ind, a_row) for ind, a_row in enumerate(df[['gene1', 'gene2', 'cancer']].values))
    #print(res)
    end = time.time()
    print(end - start)
    res=np.array(res)
    nan_inds = np.argwhere(np.isnan(res))
    res[nan_inds] = all_rows[nan_inds,0]
    #val_df = pd.DataFrame(all_rows, columns=cols, index=df.index.values)
    #df['surv'] = pd.concat([df, val_df], axis=1)
    df['surv'] = res

    return df


def calculate_tumor_coexp_cnv(df, cancer_aim, tumor_expr_df, norm_expr_df, cnv_df, ready_data):
    '''
    For continuous use Pearson, for ordinal use Spearman
    :param df: Trainin gene pairs
    :param cancer_aim: For which cancer
    :param tumor_expr_df: expression matrix
    :param norm_expr_df: expression matrix
    :param cnv_df: gistic cnv matrix
    :return: dataframe filled with 2 part.
    '''
    uniprot_extra_mapping_loc = 'tissue_data/'+cancer_aim.lower()+'_tcga'
    ext_map = get_uniprot_extra_mapping(uniprot_extra_mapping_loc)
    if ext_map is None:
        print(f'No extra mapping')
    else:
        extra_map1, extra_map2 = ext_map

    cols = ['tumor_coexp', 'tumor_coexp_p', 'normal_coexp', 'normal_coexp_p', 'tumor_cocnv', 'tumor_cocnv_p']
    if cols[0] in df.columns:
        all_rows=df[cols].values
        df = df.drop(columns=cols)
    else:
        all_rows = np.zeros(shape=(len(df),6))
    for idx, a_row in enumerate(df[['gene1', 'gene2', 'cancer']].values):
        if idx%10000 ==0:
            print(f'Iteration for row {idx} is started for {cancer_aim} in tcga_coexp.')
            logging.info(f'Iteration for row {idx} is started for {cancer_aim} in tcga_coexp.')
        gene1 = a_row[0]
        gene2 = a_row[1]
        cancer = a_row[2]
        if cancer != cancer_aim:
            continue

        if ready_data is not None and type(ready_data)==dict and (gene1, gene2) in ready_data.keys():
            all_rows[idx] = [ready_data[(gene1, gene2)][aim_col] for aim_col in cols]
            continue
        elif (ready_data is not None) and type(ready_data)!=dict:
            chosen_row = ready_data[(ready_data[['gene1', 'gene2', 'cancer']] == a_row[0:3].values).all(1)]
            if ready_data is not None and len(chosen_row)>0:
                df.loc[idx, ['tumor_coexp', 'tumor_coexp_p', 'normal_coexp', 'normal_coexp_p', 'tumor_cocnv', 'tumor_cocnv_p']] = \
                    chosen_row.loc[:, ['tumor_coexp', 'tumor_coexp_p', 'normal_coexp', 'normal_coexp_p', 'tumor_cocnv', 'tumor_cocnv_p']].values[0]
                continue

        gene1_expr = None
        if gene1 in tumor_expr_df.columns:
            gene1_expr = tumor_expr_df[gene1]
        elif ext_map is not None and gene1 in extra_map1.keys() and extra_map1[gene1] in tumor_expr_df.columns:
            gene1_expr = tumor_expr_df[extra_map1[gene1]]
        elif ext_map is not None and gene1 in extra_map2.keys() and extra_map2[gene1] in tumor_expr_df.columns:
            gene1_expr = tumor_expr_df[extra_map2[gene1]]

        gene2_expr = None
        if gene2 in tumor_expr_df.columns:
            gene2_expr = tumor_expr_df[gene2]
        elif ext_map is not None and gene2 in extra_map1.keys() and extra_map1[gene2] in tumor_expr_df.columns:
            gene2_expr = tumor_expr_df[extra_map1[gene2]]
        elif ext_map is not None and gene2 in extra_map2.keys() and extra_map2[gene2] in tumor_expr_df.columns:
            gene2_expr = tumor_expr_df[extra_map2[gene2]]

        if gene1_expr is not None and gene2_expr is not None:
            try:
                coexp = pearsonr(gene1_expr, gene2_expr)
                all_rows[idx,0:2]=[coexp[0], coexp[1]]
                #df.loc[ind, 'tumor_coexp'] = coexp[0]
                #df.loc[ind, 'tumor_coexp_p'] = coexp[1]
            except:
                all_rows[idx,0:2]=[0, 1]
                print(f'Generating pearson correlation for {gene1}-{gene2} tumor expression not successful!')


        gene1_nexpr=None
        if gene1 in norm_expr_df.columns:
            gene1_nexpr = norm_expr_df[gene1]
        elif ext_map is not None and gene1 in extra_map1.keys() and extra_map1[gene1] in norm_expr_df.columns:
            gene1_nexpr = norm_expr_df[extra_map1[gene1]]
        elif ext_map is not None and gene1 in extra_map2.keys() and extra_map2[gene1] in norm_expr_df.columns:
            gene1_nexpr = norm_expr_df[extra_map2[gene1]]

        gene2_nexpr=None
        if gene2 in norm_expr_df.columns:
            gene2_nexpr = norm_expr_df[gene2]
        elif ext_map is not None and gene2 in extra_map1.keys() and extra_map1[gene2] in norm_expr_df.columns:
            gene2_nexpr = norm_expr_df[extra_map1[gene2]]
        elif ext_map is not None and gene2 in extra_map2.keys() and extra_map2[gene2] in norm_expr_df.columns:
            gene2_nexpr = norm_expr_df[extra_map2[gene2]]

        if gene1_nexpr is not None and gene2_nexpr is not None:
            try:
                conexp = pearsonr(gene1_nexpr, gene2_nexpr)
                all_rows[idx,2:4]=[conexp[0], conexp[1]]
                #df.loc[ind, 'normal_coexp'] = conexp[0]
                #df.loc[ind, 'normal_coexp_p'] = conexp[1]
            except:
                all_rows[idx,2:4]=[0,1]
                print(f'Generating pearson correlation for {gene1}-{gene2} normal expression not successful!')

        gene1_cnv = None
        if gene1 in cnv_df.columns:
            gene1_cnv = cnv_df[gene1]
        elif ext_map is not None and gene1 in extra_map1.keys() and extra_map1[gene1] in cnv_df.columns:
            gene1_cnv = cnv_df[extra_map1[gene1]]
        elif ext_map is not None and gene1 in extra_map2.keys() and extra_map2[gene1] in cnv_df.columns:
            gene1_cnv = cnv_df[extra_map2[gene1]]

        gene2_cnv = None
        if gene2 in cnv_df.columns:
            gene2_cnv = cnv_df[gene2]
        elif ext_map is not None and gene2 in extra_map1.keys() and extra_map1[gene2] in cnv_df.columns:
            gene2_cnv = cnv_df[extra_map1[gene2]]
        elif ext_map is not None and gene2 in extra_map2.keys() and extra_map2[gene2] in cnv_df.columns:
            gene2_cnv = cnv_df[extra_map2[gene2]]

        if gene1_cnv is not None and gene2_cnv is not None:
            try:
                cocnv = spearmanr(gene1_cnv, gene2_cnv)
                all_rows[idx,4:6]=[cocnv.correlation, cocnv.pvalue]
                #df.loc[ind, 'tumor_cocnv'] = cocnv.correlation
                #df.loc[ind, 'tumor_cocnv_p'] = cocnv.pvalue
            except:
                all_rows[idx,4:6]=[0,1]
                print(f'Generating pearson correlation for {gene1}-{gene2} tumor cnv not successful!')

    val_df = pd.DataFrame(all_rows, columns=cols, index=df.index.values)
    df = pd.concat([df, val_df], axis=1)
    return df


def calculate_healthy_coexp(df, cancer_aim, expr_df, ready_data):
    '''
    For continuous use Pearson, for ordinal use Spearman
    :param df: Trainin gene pairs
    :param cancer_aim: For which cancer
    :param tumor_expr_df: expression matrix
    :param cnv_df: gistic cnv matrix
    :return: dataframe filled with 2 part.
    '''
    uniprot_extra_mapping_loc = 'tissue_data/' + cancer_aim.lower() + '_gtex'
    ext_map = get_uniprot_extra_mapping(uniprot_extra_mapping_loc)
    if ext_map is None:
        print(f'No extra mapping')
    else:
        extra_map1, extra_map2 = ext_map

    cols = ['gtex_coexp', 'gtex_coexp_p']
    if cols[0] in df.columns:
        all_rows=df[cols].values
        df = df.drop(columns=cols)
    else:
        all_rows = np.zeros(shape=(len(df),2))
    for ind, a_row in enumerate(df[['gene1', 'gene2', 'cancer']].values):
        if ind%10000 ==0:
            print(f'Iteration for row {ind} is started for {cancer_aim} in healthy embeddings.')
            logging.info(f'Iteration for row {ind} is started for {cancer_aim} in healthy embeddings.')
        gene1 = a_row[0]
        gene2 = a_row[1]
        cancer = a_row[2]
        if cancer != cancer_aim:
            continue
        if ready_data is not None and type(ready_data)==dict and (gene1, gene2) in ready_data.keys():
            all_rows[ind] = [ready_data[(gene1, gene2)][aim_col] for aim_col in cols]
            continue
        elif (ready_data is not None) and type(ready_data)!=dict:
            chosen_row = ready_data[(ready_data[['gene1', 'gene2', 'cancer']] == a_row[0:3].values).all(1)]
            if ready_data is not None and len(chosen_row)>0:
                df.loc[ind, ['gtex_coexp', 'gtex_coexp_p']] = chosen_row.loc[:, ['gtex_coexp', 'gtex_coexp_p']].values[0]
                continue

        gene1_expr = None
        if gene1 in expr_df.columns:
            gene1_expr = expr_df[gene1]
        elif ext_map is not None and gene1 in extra_map1.keys() and extra_map1[gene1] in expr_df.columns:
            gene1_expr = expr_df[extra_map1[gene1]]
        elif ext_map is not None and gene1 in extra_map2.keys() and extra_map2[gene1] in expr_df.columns:
            gene1_expr = expr_df[extra_map2[gene1]]

        gene2_expr = None
        if gene2 in expr_df.columns:
            gene2_expr = expr_df[gene2]
        elif ext_map is not None and gene2 in extra_map1.keys() and extra_map1[gene2] in expr_df.columns:
            gene2_expr = expr_df[extra_map1[gene2]]
        elif ext_map is not None and gene2 in extra_map2.keys() and extra_map2[gene2] in expr_df.columns:
            gene2_expr = expr_df[extra_map2[gene2]]


        if gene1_expr is not None and gene2_expr is not None:
            try:
                coexp = pearsonr(gene1_expr, gene2_expr)
                all_rows[ind]=[coexp[0], coexp[1]]
                #df.loc[ind, 'gtex_coexp'] = coexp[0]
                #df.loc[ind, 'gtex_coexp_p'] = coexp[1]
            except:
                all_rows[ind]=[0,1]
                print(f'Generating pearson correlation for {gene1}-{gene2} healthy expression not successful!')
    val_df = pd.DataFrame(all_rows, columns=cols, index=df.index.values)
    df = pd.concat([df, val_df], axis=1)

    return df


def calculate_expr_by_mut(df, cancer_aim, mut_df, std_expr_df, sample_df, ready_data):
    '''
    For continuous use Pearson, for ordinal use Spearman
    :param df: Trainin gene pairs
    :param cancer_aim: For which cancer
    :param tumor_expr_df: expression matrix
    :param cnv_df: gistic cnv matrix
    :return: dataframe filled with 2 part.
    '''
    uniprot_extra_mapping_loc = 'tissue_data/' + cancer_aim.lower() + '_tcga'
    ext_map = get_uniprot_extra_mapping(uniprot_extra_mapping_loc)
    if ext_map is None:
        print(f'No extra mapping')
    else:
        extra_map1, extra_map2 = ext_map

    grouped_mut = mut_df.groupby('Hugo_Symbol')['Tumor_Sample_Barcode']\
        .agg(list= lambda x: list(np.unique(x))).reset_index()
    gene_mut_patients = grouped_mut.set_index('Hugo_Symbol')['list'].to_dict()
    cols = ['gene1_expr_m1', 'gene1_expr_m0', 'gene2_expr_m1', 'gene2_expr_m0']
    if cols[0] in df.columns:
        all_rows=df[cols].values
        df = df.drop(columns=cols)
    else:
        all_rows = np.zeros(shape=(len(df),4))
    for ind, a_row in enumerate(df[['gene1', 'gene2', 'cancer']].values):
        if ind%10000 ==0:
            print(f'Iteration for row {ind} is started for {cancer_aim} for expression with mutation.')
            logging.info(f'Iteration for row {ind} is started for {cancer_aim} for expression with mutation.')
        gene1 = a_row[0]
        gene2 = a_row[1]
        cancer = a_row[2]
        if cancer != cancer_aim:
            continue
        if ready_data is not None and type(ready_data)==dict and (gene1, gene2) in ready_data.keys():
            all_rows[ind] = [ready_data[(gene1, gene2)][aim_col] for aim_col in cols]
            continue
        elif (ready_data is not None) and type(ready_data)!=dict:
            chosen_row = ready_data[(ready_data[['gene1', 'gene2', 'cancer']] == a_row[0:3].values).all(1)]
            if ready_data is not None and len(chosen_row)>0:
                df.loc[ind, ['gene1_expr_m1', 'gene1_expr_m0', 'gene2_expr_m1', 'gene2_expr_m0']] = \
                    chosen_row.loc[:, ['gene1_expr_m1', 'gene1_expr_m0', 'gene2_expr_m1', 'gene2_expr_m0']].values[0]
                continue

        gene2_mutated_CLLEs = np.array([])
        if gene2 in gene_mut_patients.keys():
            gene2_mutated_CLLEs = np.array(gene_mut_patients[gene2])
        elif ext_map is not None and gene2 in extra_map1.keys() and extra_map1[gene2] in gene_mut_patients.keys():
            gene2_mutated_CLLEs = np.array(gene_mut_patients[extra_map1[gene2]])
        elif ext_map is not None and gene2 in extra_map2.keys() and extra_map2[gene2] in gene_mut_patients.keys():
            gene2_mutated_CLLEs = np.array(gene_mut_patients[extra_map2[gene2]])

        gene1_mutated_CLLEs = np.array([])
        if gene1 in gene_mut_patients.keys():
            gene1_mutated_CLLEs = np.array(gene_mut_patients[gene1])
        elif ext_map is not None and gene1 in extra_map1.keys() and extra_map1[gene1] in gene_mut_patients.keys():
            gene1_mutated_CLLEs = np.array(gene_mut_patients[extra_map1[gene1]])
        elif ext_map is not None and gene1 in extra_map2.keys() and extra_map2[gene1] in gene_mut_patients.keys():
            gene1_mutated_CLLEs = np.array(gene_mut_patients[extra_map2[gene1]])


        avg_expr_gene1_w_gene2_mut = np.nan
        avg_expr_gene1_wo_gene2_mut = np.nan
        avg_expr_gene2_w_gene1_mut = np.nan
        avg_expr_gene2_wo_gene1_mut = np.nan
        if gene1 in std_expr_df.columns:
            avg_expr_gene1_w_gene2_mut = std_expr_df[std_expr_df.index.isin(gene2_mutated_CLLEs)][gene1].mean(skipna=True)
            avg_expr_gene1_wo_gene2_mut = std_expr_df[(~std_expr_df.index.isin(gene2_mutated_CLLEs))][gene1].mean(skipna=True)
        elif ext_map is not None and gene1 in extra_map1.keys() and extra_map1[gene1] in std_expr_df.columns:
            avg_expr_gene1_w_gene2_mut = std_expr_df[std_expr_df.index.isin(gene2_mutated_CLLEs)][extra_map1[gene1]].mean(skipna=True)
            avg_expr_gene1_wo_gene2_mut = std_expr_df[(~std_expr_df.index.isin(gene2_mutated_CLLEs))][extra_map1[gene1]].mean(skipna=True)
        elif ext_map is not None and gene1 in extra_map2.keys() and extra_map2[gene1] in std_expr_df.columns:
            avg_expr_gene1_w_gene2_mut = std_expr_df[std_expr_df.index.isin(gene2_mutated_CLLEs)][extra_map2[gene1]].mean(skipna=True)
            avg_expr_gene1_wo_gene2_mut = std_expr_df[(~std_expr_df.index.isin(gene2_mutated_CLLEs))][extra_map2[gene1]].mean(skipna=True)

        if gene2 in std_expr_df.columns:
            avg_expr_gene2_w_gene1_mut = std_expr_df[std_expr_df.index.isin(gene1_mutated_CLLEs)][gene2].mean(skipna=True)
            avg_expr_gene2_wo_gene1_mut = std_expr_df[(~std_expr_df.index.isin(gene1_mutated_CLLEs))][gene2].mean(skipna=True)
        elif ext_map is not None and gene2 in extra_map1.keys() and extra_map1[gene2] in std_expr_df.columns:
            avg_expr_gene2_w_gene1_mut = std_expr_df[std_expr_df.index.isin(gene1_mutated_CLLEs)][extra_map1[gene2]].mean(skipna=True)
            avg_expr_gene2_wo_gene1_mut = std_expr_df[(~std_expr_df.index.isin(gene1_mutated_CLLEs))][extra_map1[gene2]].mean(skipna=True)
        elif ext_map is not None and gene2 in extra_map2.keys() and extra_map2[gene2] in std_expr_df.columns:
            avg_expr_gene2_w_gene1_mut = std_expr_df[std_expr_df.index.isin(gene1_mutated_CLLEs)][extra_map2[gene2]].mean(skipna=True)
            avg_expr_gene2_wo_gene1_mut = std_expr_df[(~std_expr_df.index.isin(gene1_mutated_CLLEs))][extra_map2[gene2]].mean(skipna=True)

        all_rows[ind] = [avg_expr_gene1_w_gene2_mut, avg_expr_gene1_wo_gene2_mut,
                         avg_expr_gene2_w_gene1_mut, avg_expr_gene2_wo_gene1_mut]
        #df.loc[ind, 'gene1_expr_m1'] = avg_expr_gene1_w_gene2_mut
        #df.loc[ind, 'gene1_expr_m0'] = avg_expr_gene1_wo_gene2_mut
        #df.loc[ind, 'gene2_expr_m1'] = avg_expr_gene2_w_gene1_mut
        #df.loc[ind, 'gene2_expr_m0'] = avg_expr_gene2_wo_gene1_mut
    val_df = pd.DataFrame(all_rows, columns=cols, index=df.index.values)
    df = pd.concat([df, val_df], axis=1)

    return df


def calculate_discoversl(df, cancer_aim, mut_df, cnv_df, ready_data=None):
    uniprot_extra_mapping_loc = 'tissue_data/'+cancer_aim.lower()+'_tcga'
    ext_map = get_uniprot_extra_mapping(uniprot_extra_mapping_loc)
    if ext_map is None:
        print(f'No extra mapping')
    else:
        extra_map1, extra_map2 = ext_map
    grouped_mut = mut_df.groupby('Hugo_Symbol')['Tumor_Sample_Barcode']\
        .agg(list= lambda x: list(np.unique(x))).reset_index()
    gene_mut_pats = grouped_mut.set_index('Hugo_Symbol')['list'].to_dict()
    '''
    cancer_df = df[df['cancer']==cancer_aim]
    all_genes = np.union1d(cancer_df['gene1'].values, cancer_df['gene2'].values)
    cnv_mut = cnv_df != 0
    cnv_mut = cnv_mut.T
    mut_df = mut_df[['Hugo_Symbol', 'Tumor_Sample_Barcode']].drop_duplicates()
    mut_df = mut_df[mut_df['Hugo_Symbol'].isin(cnv_mut.index.values)]
    mut_df = mut_df[mut_df['Tumor_Sample_Barcode'].isin(cnv_mut.columns.values)]
    #for ind, row in mut_df.iterrows():
    #    cnv_mut.loc[row['Hugo_Symbol'], row['Tumor_Sample_Barcode']] = True
    try:
        cnv_mut = pd.read_csv('temp_cnv_mut.csv', index_col=0)
    except:
        for col in cnv_mut.columns.values:
            mutated_genes = mut_df[mut_df['Tumor_Sample_Barcode'] == col]['Hugo_Symbol'].unique()
            if len(mutated_genes)>0:
                cnv_mut.loc[mutated_genes, col] = True


    from rpy2.robjects.packages import importr
    discover = importr('discover')
    import rpy2.robjects.numpy2ri
    rpy2.robjects.numpy2ri.activate()
    import rpy2.robjects as ro
    nr, nc = cnv_mut.shape
    Br = ro.r.matrix(cnv_mut.values, nrow=nr, ncol=nc)
    event = discover.discover_matrix(Br)
    event[cnv_mut.index.isin(np.intersect1d(all_genes, cnv_mut.index.values))]
    '''


    for ind, row in df.iterrows():
        if ind%3000 ==0:
            print(f'Iteration for row {ind} is started for {cancer_aim} in survival embeddings.')
        gene1 = row['gene1']
        gene2 = row['gene2']
        cancer = row['cancer']
        if cancer != cancer_aim:
            continue

        if (ready_data is not None):
            chosen_row = ready_data[(ready_data[['gene1', 'gene2', 'cancer']].values == [gene1, gene2, cancer]).all(1)]
            if ready_data is not None and len(chosen_row)>0:
                df.loc[ind, ['dsl_mutex_mut', 'dsl_mutex_amp', 'dsl_mutex_del', 'dsl_mutex_all', 'dsl_mutex_alt']] = \
                    chosen_row.loc[:, ['dsl_mutex_mut', 'dsl_mutex_amp', 'dsl_mutex_del', 'dsl_mutex_all', 'dsl_mutex_alt']].values[0]
                continue

        #--------Mutated Patients--------#
        gene2_mutated_pats = np.array([])
        if gene2 in gene_mut_pats.keys():
            gene2_mutated_pats = np.array(gene_mut_pats[gene2])
        elif ext_map is not None and gene2 in extra_map1.keys() and extra_map1[gene2] in gene_mut_pats.keys():
            gene2_mutated_pats = np.array(gene_mut_pats[extra_map1[gene2]])
        elif ext_map is not None and gene2 in extra_map2.keys() and extra_map2[gene2] in gene_mut_pats.keys():
            gene2_mutated_pats = np.array(gene_mut_pats[extra_map2[gene2]])
        gene1_mutated_pats = np.array([])

        if gene1 in gene_mut_pats.keys():
            gene1_mutated_pats = np.array(gene_mut_pats[gene1])
        elif ext_map is not None and gene1 in extra_map1.keys() and extra_map1[gene1] in gene_mut_pats.keys():
            gene1_mutated_pats = np.array(gene_mut_pats[extra_map1[gene1]])
        elif ext_map is not None and gene1 in extra_map2.keys() and extra_map2[gene1] in gene_mut_pats.keys():
            gene1_mutated_pats = np.array(gene_mut_pats[extra_map2[gene1]])
        gene1_2_mutated_pats = np.intersect1d(gene1_mutated_pats, gene2_mutated_pats)
        #--------Mutated Patients--------#


        #--------Altered CNV Patients--------#
        gene2_amp_pats = np.array([])
        gene2_del_pats = np.array([])
        if gene2 in cnv_df.columns:
            gene2_amp_pats = np.unique(cnv_df[cnv_df[gene2] >= 2].index.values)
            gene2_del_pats = np.unique(cnv_df[cnv_df[gene2] <= -2].index.values)
        elif ext_map is not None and gene2 in extra_map1.keys() and extra_map1[gene2] in cnv_df.columns:
            gene2_amp_pats = np.unique(cnv_df[cnv_df[extra_map1[gene2]] >= 2].index.values)
            gene2_del_pats = np.unique(cnv_df[cnv_df[extra_map1[gene2]] <= -2].index.values)
        elif ext_map is not None and gene2 in extra_map2.keys() and extra_map2[gene2] in cnv_df.columns:
            gene2_amp_pats = np.unique(cnv_df[cnv_df[extra_map2[gene2]] >= 2].index.values)
            gene2_del_pats = np.unique(cnv_df[cnv_df[extra_map2[gene2]] <= -2].index.values)
        gene1_amp_pats = np.array([])
        gene1_del_pats = np.array([])
        if gene1 in cnv_df.columns:
            gene1_amp_pats = np.unique(cnv_df[cnv_df[gene1] >= 2].index.values)
            gene1_del_pats = np.unique(cnv_df[cnv_df[gene1] <= -2].index.values)
        elif ext_map is not None and gene1 in extra_map1.keys() and extra_map1[gene1] in cnv_df.columns:
            gene1_amp_pats = np.unique(cnv_df[cnv_df[extra_map1[gene1]] >= 2].index.values)
            gene1_del_pats = np.unique(cnv_df[cnv_df[extra_map1[gene1]] <= -2].index.values)
        elif ext_map is not None and gene1 in extra_map2.keys() and extra_map2[gene1] in cnv_df.columns:
            gene1_amp_pats = np.unique(cnv_df[cnv_df[extra_map2[gene1]] >= 2].index.values)
            gene1_del_pats = np.unique(cnv_df[cnv_df[extra_map2[gene1]] <= -2].index.values)
        gene1_2_amp_pats = np.intersect1d(gene1_amp_pats, gene2_amp_pats)
        gene1_2_del_pats = np.intersect1d(gene1_del_pats, gene2_del_pats)
        #--------Altered CNV Patients--------#


        K = len(gene1_mutated_pats)
        n = len(gene2_mutated_pats)
        k = len(gene1_2_mutated_pats)
        N = len(mut_df['Tumor_Sample_Barcode'].unique())
        mutex_mut = stats.hypergeom.cdf(k-1, N, K, n)
        df.loc[ind, 'dsl_mutex_mut'] = mutex_mut

        K = len(gene1_amp_pats)
        n = len(gene2_amp_pats)
        k = len(gene1_2_amp_pats)
        N = len(cnv_df)
        mutex_amp = stats.hypergeom.cdf(k-1, N, K, n)
        df.loc[ind, 'dsl_mutex_amp'] = mutex_amp

        K = len(gene1_del_pats)
        n = len(gene2_del_pats)
        k = len(gene1_2_del_pats)
        N = len(cnv_df)
        mutex_del = stats.hypergeom.cdf(k-1, N, K, n)
        df.loc[ind, 'dsl_mutex_del'] = mutex_del

        try:
            comb= np.array([mutex_mut, mutex_amp, mutex_del])
            mutex_comb = stats.combine_pvalues(comb[comb != 0], method='fisher')
            if not np.isnan(mutex_comb)[1]:
                df.loc[ind, 'dsl_mutex_all'] = mutex_comb[1]
        except:
            df.loc[ind, 'dsl_mutex_all'] = 1


        gene1_alt_pats = np.array(list(set().union(gene1_mutated_pats, gene1_amp_pats, gene1_del_pats)))
        gene2_alt_pats = np.array(list(set().union(gene2_mutated_pats, gene2_amp_pats, gene2_del_pats)))
        gene1_2_alt_pats = np.intersect1d(gene1_alt_pats, gene2_alt_pats)

        K = len(gene1_alt_pats)
        n = len(gene2_alt_pats)
        k = len(gene1_2_alt_pats)
        mut_df['Tumor_Sample_Barcode'].unique()
        N = len(np.union1d(cnv_df.index.values, mut_df['Tumor_Sample_Barcode'].unique()))
        mutex_alt = stats.hypergeom.cdf(k-1, N, K, n)
        df.loc[ind, 'dsl_mutex_alt'] = mutex_alt


    return df


def calculate_discover(df, cancer_aim, mut_df, cnv_df):
    return df

def main():
    loc_dict = get_locs()
    a = get_dataset('healthy_expression')
    for cancer, data in a.items():
        loc = config.DATA_DIR / 'tissue_data' / (cancer.lower()+'_gtex') / (cancer.lower()+'_RNASeQ_tpm.csv.gz')
        data.to_csv(loc, sep='\t', compression='gzip')
    return 0


if __name__ == '__main__':
    main()
