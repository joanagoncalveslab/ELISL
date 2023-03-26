import pandas as pd
import sys, os
import numpy as np
import csv
import collections as coll
from src import config


def load_mut(file_loc, remove_silent=True):
    file_loc = config.DATA_DIR / file_loc
    df = pd.read_csv(file_loc, sep='\t')
    df = df[['Hugo_Symbol', 'Entrez_Gene_Id', 'Variant_Classification', 'Variant_Type', 'Tumor_Sample_Barcode', 'Matched_Norm_Sample_Barcode']]
    if remove_silent:
        df = df[df['Variant_Classification'] != 'Silent']
    return df


def load_expr_cnv(file_loc):
    file_loc = config.DATA_DIR / file_loc
    df = pd.read_csv(file_loc, sep='\t')
    df['Hugo_Symbol'] = df['Hugo_Symbol'].str.split('|', n=1, expand=True)[0]
    df = df.set_index('Hugo_Symbol')
    df = df.drop(columns='Entrez_Gene_Id')
    #df_upd = df.rename(columns=mapping)
    #df_upd = df_upd.sort_index()
    #df_upd = df_upd.sort_index(axis=1)
    df = df.sort_index()
    df = df.sort_index(axis=1)
    return df


def load_all_expr(file_loc):
    file_loc = config.DATA_DIR / file_loc
    df = pd.read_csv(file_loc, sep='\t')
    df['Hybridization REF'] = df['Hybridization REF'].str.split('|', n=1, expand=True)[0]
    df = df.drop(index=0)
    df = df.set_index('Hybridization REF')
    df = df.astype(float)
    #df_upd = df.rename(columns=mapping)
    #df_upd = df_upd.sort_index()
    #df_upd = df_upd.sort_index(axis=1)
    df = df.sort_index()
    df = df.sort_index(axis=1)
    return df


def load_patient_info(file_loc):
    file_loc = config.DATA_DIR / file_loc
    df = pd.read_csv(file_loc, sep='\t', header=4)
    df = df[['PATIENT_ID', 'SEX', 'RACE', 'ETHNICITY', 'AGE', 'OS_STATUS', 'OS_MONTHS']]
    df['OS_STATUS'] = df['OS_STATUS'].str.split(':', n=1, expand=True)[0]
    return df


def load_gtex_expr(file_loc):
    file_loc = config.DATA_DIR / file_loc
    expr = pd.read_csv(file_loc, sep='\t', header=2)
    return expr


def load_cancer_gtex_expr(file_loc):
    file_loc = config.DATA_DIR / file_loc
    df = pd.read_csv(file_loc, sep='\t')
    if 'Name' in df.columns:
        df = df.drop(columns=['Name'])
    if 'Unnamed: 0' in df.columns:
        df = df.drop(columns=['Unnamed: 0'])
    df = df.set_index('Description')
    df = df.astype(float)
    df = df.sort_index()
    df = df.sort_index(axis=1)

    return df


def load_gtex_sample_info(file_loc):
    file_loc = config.DATA_DIR / file_loc
    gtex_samples = pd.read_csv(file_loc, sep='\t')
    gtex_short = gtex_samples[['SAMPID', 'SMTSD']]
    cancer2tissue = {'BRCA': 'Breast - Mammary Tissue', 'CESC': ['Cervix - Ectocervix', 'Cervix - Endocervix'],
                     'COAD': 'Colon - Transverse', 'KIRC': 'Kidney - Cortex', 'LAML': 'Whole Blood', 'LUAD': 'Lung',
                     'OV': 'Ovary', 'SKCM': ['Skin - Not Sun Exposed (Suprapubic)', 'Skin - Sun Exposed (Lower leg)']}
    cancer_pat_dict = coll.OrderedDict()
    for cancer, tissue in cancer2tissue.items():
        if type(tissue) == str:
            cancer_pat_dict[cancer] = gtex_short[gtex_short['SMTSD'] == tissue]['SAMPID'].values
        else:
            cancer_pat_dict[cancer] = gtex_short[(gtex_short['SMTSD'] == tissue[0]) | (gtex_short['SMTSD'] == tissue[1])]['SAMPID'].values

    return cancer_pat_dict


def get_locs():
    loc_dict = coll.OrderedDict()
    for cancer in ['BRCA', 'CESC', 'COAD', 'KIRC', 'LAML', 'LUAD', 'OV', 'SKCM']:
        loc_dict[cancer]={}
        loc_dict[cancer]['mutation'] = 'tissue_data/'+cancer.lower()+'_tcga/data_mutations_extended.txt'
        loc_dict[cancer]['std_expression'] = 'tissue_data/'+cancer.lower()+'_tcga/data_RNA_Seq_v2_mRNA_median_all_sample_Zscores.txt'
        loc_dict[cancer]['expression'] = 'tissue_data/'+cancer.lower()+'_tcga/'+cancer+'.rnaseqv2_norm.txt'
        loc_dict[cancer]['cnv'] = 'tissue_data/'+cancer.lower()+'_tcga/data_CNA.txt'
        loc_dict[cancer]['sample_info'] = 'tissue_data/'+cancer.lower()+'_tcga/data_bcr_clinical_data_sample.txt'
        loc_dict[cancer]['patient_info'] = 'tissue_data/'+cancer.lower()+'_tcga/data_bcr_clinical_data_patient.txt'
        loc_dict[cancer]['gtex_expression'] = 'tissue_data/'+cancer.lower()+'_gtex/'+cancer.lower()+'_RNASeQ_tpm.csv.gz'
    loc_dict['gtex_attribute'] = 'tissue_data/gtex/GTEx_Analysis_v8_Annotations_SampleAttributesDS.txt'
    loc_dict['gtex_expression'] = 'tissue_data/gtex/GTEx_Analysis_2017-06-05_v8_RNASeQCv1.1.9_gene_tpm.gct.gz'

    return loc_dict


def main():
    loc_dict = get_locs()
    expr_df = load_gtex_expr(loc_dict['gtex_expression'])
    cancer_pat_dict = load_gtex_sample_info(loc_dict['gtex_attribute'])
    cancer2expr = {}
    for cancer, pats in cancer_pat_dict.items():
        col_list = np.union1d(['Name', 'Description'], pats)
        col_list = np.intersect1d(expr_df.columns.values, col_list)
        cancer2expr[cancer] = expr_df[col_list]

    for cancer, data in cancer2expr.items():
        loc = config.DATA_DIR / 'tissue_data' / (cancer.lower()+'_gtex') / (cancer.lower()+'_RNASeQ_tpm.csv.gz')
        data.to_csv(loc, sep='\t', compression='gzip')
    '''
    for cancer in ['BRCA', 'OV', 'LUAD', 'COAD']:
        #mut_df = load_mut(loc_dict[cancer]['mutation'])
        #sample_df = load_sample_info(loc_dict[cancer]['sample_info'])
        patient_df = load_patient_info(loc_dict[cancer]['patient_info'])
        #expr_df = load_expr_cnv(loc_dict[cancer]['expression'])
        #cnv_df = load_expr_cnv(loc_dict[cancer]['cnv'])
    #samp = load_gtex_sample_info(loc_dict['gtex_attribute'])
    '''


    return 0


if __name__ == '__main__':
    main()
