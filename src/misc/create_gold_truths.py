import os
import src.data_functions as dfnc
from src import config
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from Bio import SeqIO
import src.embedding.ppi_embedding as ppi_e
import src.datasets.ccle_broad as cb
import src.datasets.tissue as tcga


def create_pairs_for_each_source():
    data_dict = {}
    exp2sl = dfnc.get_exp2sl_data()
    lu15 = dfnc.get_lu15_data(dfnc.get_loc_dict()['lu15_dataset_loc'])
    isle = dfnc.get_ISLE_training_set()
    dsl = dfnc.get_DiscoverSL_training_set()
    syn_sl_df, syn_non_sl_df, syn_sr_df, syn_df = dfnc.get_SynLethDB_data()
    # colm = dfnc.load_rdata_file('labels/combined.RData', isRDS=True)
    # colm_dsl = dfnc.load_rdata_file('labels/discoversl.RData', isRDS=True)
    # colm_isle = dfnc.load_rdata_file('labels/isle.RData', isRDS=True)
    data_dict['exp2sl'] = exp2sl
    data_dict['lu15'] = lu15
    data_dict['isle'] = isle
    data_dict['dsl'] = dsl
    # data_dict['syn_df'] = syn_df
    for name, data in data_dict.items():
        print(f'Analysis for {name} started.')
        if 'pmid' in data.columns:
            data = data.drop(columns=['pmid'])
        data = data.astype({"class": int})
        clean_data = dfnc.remove_duplicates_inner(data)
        clean_data = clean_data.sort_values(by=['cancer', 'gene1', 'gene2'])
        print(f'Analysis for {name} ended.\n')
        file_loc = config.DATA_DIR / ('labels/'+name+'_pairs.csv')
        #clean_data.to_csv(file_loc, sep=',', index=None)


def get_golden_truth_by_set():
    data_dict = {}
    exp2sl = dfnc.get_exp2sl_data()
    exp2sl = exp2sl.astype({"class": int})
    lu15 = dfnc.get_lu15_data(dfnc.get_loc_dict()['lu15_dataset_loc'])
    lu15 = lu15.astype({"class": int})
    isle = dfnc.get_ISLE_training_set()
    isle = isle.astype({"class": int})
    dsl = dfnc.get_DiscoverSL_training_set()
    dsl = dsl.astype({"class": int})
    #syn_sl_df, syn_non_sl_df, syn_sr_df, syn_df = dfnc.get_SynLethDB_data()
    #colm = dfnc.load_rdata_file('labels/combined.RData', isRDS=True)
    #colm_dsl = dfnc.load_rdata_file('labels/discoversl.RData', isRDS=True)
    #colm_isle = dfnc.load_rdata_file('labels/isle.RData', isRDS=True)
    data_dict['exp2sl'] = exp2sl
    data_dict['lu15'] = lu15
    data_dict['isle'] = isle
    data_dict['dsl'] = dsl
    # data_dict['syn_df'] = syn_df
    for name, data in data_dict.items():
        print(f'Analysis for {name} started.')
        ddata = dfnc.remove_duplicates_inner(data)
        print(f'Analysis for {name} ended.\n')
        save_loc = config.DATA_DIR / 'labels' / (name+'_pairs.csv')
        ddata = ddata[['gene1', 'gene2', 'class', 'cancer']]
        ddata.to_csv(save_loc, index=None)


def get_golden_truth_by_cancer(file_loc='labels/labels_per_cancer_deneme.pickle'):
    file_loc = config.DATA_DIR / file_loc
    if os.path.isfile(file_loc):
        cancer_dict = dfnc.load_pickle(file_loc)
    else:
        data_dict = {}
        exp2sl = dfnc.get_exp2sl_data()
        lu15 = dfnc.get_lu15_data(dfnc.get_loc_dict()['lu15_dataset_loc'])
        isle = dfnc.get_ISLE_training_set()
        dsl = dfnc.get_DiscoverSL_training_set()
        #syn_sl_df, syn_non_sl_df, syn_sr_df, syn_df = dfnc.get_SynLethDB_data()
        #colm = dfnc.load_rdata_file('labels/combined.RData', isRDS=True)
        #colm_dsl = dfnc.load_rdata_file('labels/discoversl.RData', isRDS=True)
        #colm_isle = dfnc.load_rdata_file('labels/isle.RData', isRDS=True)
        data_dict['exp2sl'] = exp2sl
        data_dict['lu15'] = lu15
        data_dict['isle'] = isle
        data_dict['dsl'] = dsl
        # data_dict['syn_df'] = syn_df
        for name, data in data_dict.items():
            print(f'Analysis for {name} started.')
            data_dict[name] = dfnc.remove_duplicates_inner(data)
            print(f'Analysis for {name} ended.\n')
        cancer_dict = {}
        for cancer in ['BRCA', 'CESC', 'COAD', 'KIRC', 'LAML', 'LUAD', 'OV', 'SKCM']:
            cancer_df = pd.DataFrame(columns=['gene1', 'gene2', 'cancer', 'class', 'from'])
            for name, data in data_dict.items():
                cancer_select = data[data['cancer'] == cancer]
                cancer_select['from'] = name
                cancer_df = pd.concat([cancer_df, cancer_select])
            cancer_df = cancer_df.astype({"class": int})
            cancer_dict[cancer] = cancer_df
        cancer_dict = dfnc.remove_duplicates_outer(cancer_dict)

        dfnc.save_pickle(file_loc, cancer_dict)

    return cancer_dict

def split_dataset_train_test(cancer_dict, test_size=0.2):
    train_test_dict = {}
    training_data = pd.DataFrame(columns=['gene1','gene2','class', 'cancer'])
    test_data = pd.DataFrame(columns=['gene1','gene2','class', 'cancer'])
    for cancer, cancer_data in cancer_dict.items():
        cancer_data['cancer'] = cancer
        train_df, test_df = train_test_split(cancer_data, stratify=cancer_data['class'], test_size=test_size, random_state=2628)
        training_data = pd.concat([training_data, train_df])
        test_data = pd.concat([test_data, test_df])
    tr_save_loc = config.DATA_DIR / 'labels/train_pairs.csv'
    te_save_loc = config.DATA_DIR / 'labels/test_pairs.csv'
    training_data.to_csv(tr_save_loc, sep=',', index=None)
    test_data.to_csv(te_save_loc, sep=',', index=None)


def check_seq_mappings_and_trim(cancer_dict):
    all_genes = np.array([])
    for cancer, cancer_data in cancer_dict.items():
        cancer_genes = np.union1d(cancer_data['gene1'].values, cancer_data['gene2'].values)
        all_genes = np.union1d(all_genes, cancer_genes)

    mapped_genes = []
    second_mapped_genes = []
    unmapped_genes = []
    uniprot_fasta_last_loc = config.DATA_DIR / 'sequences/uniprot_reviewed_9606.fasta'
    uniprot_fasta_last_mapping_loc = config.DATA_DIR / 'sequences/uniprot_reviewed_9606_mappings.csv'
    uniprot_extra_mapping_loc = 'sequences/uniprot_extra_mapping.tab'
    uniprot_extra1, uniprot_extra2 = get_uniprot_extra_mapping(uniprot_extra_mapping_loc)

    if os.path.isfile(uniprot_fasta_last_mapping_loc):
        fasta_df = pd.read_csv(uniprot_fasta_last_mapping_loc)
    else:
        fasta_df = create_fasta_mappings(uniprot_fasta_last_loc)
        fasta_df.to_csv(uniprot_fasta_last_mapping_loc, index=None)
    for gene in all_genes:
        if gene in fasta_df['gene_name'].values or gene in fasta_df['uniprot_name'].values:
            mapped_genes.append(gene)
        elif gene in uniprot_extra1.keys() and \
                ((uniprot_extra1[gene] in fasta_df['gene_name'].values) or
                 (uniprot_extra1[gene] in fasta_df['uniprot_name'].values)):
            second_mapped_genes.append(gene)
        elif gene in uniprot_extra2.keys() and \
                ((uniprot_extra2[gene] in fasta_df['gene_name'].values) or
                 (uniprot_extra2[gene] in fasta_df['uniprot_name'].values)):
            second_mapped_genes.append(gene)
        else:
            unmapped_genes.append(gene)

    unmapped_df = pd.DataFrame(data={'gene_name': unmapped_genes})
    unmapped_loc = config.DATA_DIR / 'sequences/unmapped_v2.csv'
    unmapped_df.to_csv(unmapped_loc, index=None, header=None)
    missing_set = list(set(unmapped_genes))
    print(f'There are {len(unmapped_genes)} genes that are not mapped.')


def check_ppi_mappings_and_trim(cancer_dict):
    all_genes = np.array([])
    for cancer, cancer_data in cancer_dict.items():
        cancer_genes = np.union1d(cancer_data['gene1'].values, cancer_data['gene2'].values)
        all_genes = np.union1d(all_genes, cancer_genes)

    mapped_genes = []
    second_mapped_genes = []
    unmapped_genes = []
    ppi_mappings = []
    uniprot_extra_mapping_loc = 'PPI/STRING/uniprot_extra_mapping.tab'
    ppi_w2v_vector = ppi_e.load_embs(source='ec', opts_extra={})
    ppi_embs = ppi_e.get_embs_dict(ppi_w2v_vector)
    ppi_mappings.append(dfnc.get_PPI_genename_maps()[0])
    ppi_mappings.append(dfnc.get_PPI_uniprot_maps(aim='name')[0])
    ppi_mappings.append(dfnc.get_extra_mapping()[0])
    uniprot_extra1, uniprot_extra2 = get_uniprot_extra_mapping(uniprot_extra_mapping_loc)

    for gene in all_genes:
        if gene in ppi_mappings[0].keys() or gene in ppi_mappings[1].keys() or gene in ppi_mappings[2].keys():
            mapped_genes.append(gene)
        elif gene in uniprot_extra1.keys() and \
                ((uniprot_extra1[gene] in ppi_mappings[0].keys()) or
                 (uniprot_extra1[gene] in ppi_mappings[1].keys()) or
                 (uniprot_extra1[gene] in ppi_mappings[2].keys())):
            second_mapped_genes.append(gene)
        elif gene in uniprot_extra2.keys() and \
                ((uniprot_extra2[gene] in ppi_mappings[0].keys()) or
                 (uniprot_extra2[gene] in ppi_mappings[1].keys()) or
                 (uniprot_extra2[gene] in ppi_mappings[2].keys())):
            second_mapped_genes.append(gene)
        else:
            unmapped_genes.append(gene)

    unmapped_df = pd.DataFrame(data={'gene_name': unmapped_genes})
    unmapped_loc = config.DATA_DIR / 'PPI/STRING/unmapped_v2.csv'
    unmapped_df.to_csv(unmapped_loc, index=None, header=None)
    missing_set = list(set(unmapped_genes))
    print(f'There are {len(unmapped_genes)} genes that are not mapped.')



def check_dep_mappings_and_trim(cancer_dict):
    all_genes = np.array([])
    for cancer, cancer_data in cancer_dict.items():
        cancer_genes = np.union1d(cancer_data['gene1'].values, cancer_data['gene2'].values)
        all_genes = np.union1d(all_genes, cancer_genes)

    loc_dict = cb.get_locs()
    ccle_map = cb.load_ccle_map(loc_dict['ccle_map'])
    mut_df = cb.load_mut(loc_dict['mutation'])
    sample_df, mapping = cb.load_cell_info(loc_dict['sample_info'], extra_maps=ccle_map)
    expr_df = cb.load_expr_cnv(loc_dict['expression'], mapping)
    cnv_df = cb.load_expr_cnv(loc_dict['cnv'], mapping)
    d2_combined_dep = cb.load_d2_dep(loc_dict['d2_dependency'])
    crispr_dep = cb.load_crispr_dep(loc_dict['crispr_dependency'], mapping=mapping)


    mapped_genes = []
    second_mapped_genes = []
    unmapped_genes = []
    ppi_mappings = []

    uniprot_extra_mapping_loc = 'ccle_broad_2019/uniprot_extra_mapping.tab'
    uniprot_extra1, uniprot_extra2 = get_uniprot_extra_mapping(uniprot_extra_mapping_loc)

    for gene in all_genes:
        if (gene in mut_df['Hugo_Symbol'].values) or \
                (gene in expr_df.index.values) or (gene in cnv_df.index.values) or \
                (gene in d2_combined_dep.index.values) or (gene in crispr_dep.index.values):
            mapped_genes.append(gene)
        elif gene in uniprot_extra1.keys() and \
                ((uniprot_extra1[gene] in mut_df['Hugo_Symbol'].values) or
                 (uniprot_extra1[gene] in expr_df.index.values) or
                 (uniprot_extra1[gene] in cnv_df.index.values) or
                 (uniprot_extra1[gene] in d2_combined_dep.index.values) or
                 (uniprot_extra1[gene] in crispr_dep.index.values)):
            second_mapped_genes.append(gene)
        elif gene in uniprot_extra2.keys() and \
                ((uniprot_extra2[gene] in mut_df['Hugo_Symbol'].values) or
                 (uniprot_extra2[gene] in expr_df.index.values) or
                 (uniprot_extra2[gene] in cnv_df.index.values) or
                 (uniprot_extra2[gene] in d2_combined_dep.index.values) or
                 (uniprot_extra2[gene] in crispr_dep.index.values)):
            second_mapped_genes.append(gene)
        else:
            unmapped_genes.append(gene)

    unmapped_df = pd.DataFrame(data={'gene_name': unmapped_genes})
    unmapped_loc = config.DATA_DIR / 'ccle_broad_2019/unmapped_v2.csv'
    unmapped_df.to_csv(unmapped_loc, index=None, header=None)
    missing_set = list(set(unmapped_genes))
    print(f'There are {len(unmapped_genes)} genes that are not mapped.')


def check_tissue_mappings_and_trim(cancer, cancer_dict):
    all_genes = np.array([])
    cancer_data = cancer_dict[cancer]
    cancer_genes = np.union1d(cancer_data['gene1'].values, cancer_data['gene2'].values)
    all_genes = np.union1d(all_genes, cancer_genes)

    loc_dict = tcga.get_locs()
    mut_df = tcga.load_mut(loc_dict[cancer]['mutation'])
    #patient_df = tcga.load_patient_info(loc_dict[cancer]['patient_info'])
    expr_df = tcga.load_expr_cnv(loc_dict[cancer]['std_expression'])
    cnv_df = tcga.load_expr_cnv(loc_dict[cancer]['cnv'])


    mapped_genes = []
    second_mapped_genes = []
    unmapped_genes = []
    ppi_mappings = []

    uniprot_extra_mapping_loc = 'tissue_data/'+cancer.lower()+'_tcga/uniprot_extra_mapping.tab'
    ext_map = get_uniprot_extra_mapping(uniprot_extra_mapping_loc)
    if ext_map is None:
        print(f'No extra mapping')
    else:
        uniprot_extra1, uniprot_extra2 = ext_map

    for gene in all_genes:
        if (gene in mut_df['Hugo_Symbol'].values) or \
                (gene in expr_df.index.values) or (gene in cnv_df.index.values):
            mapped_genes.append(gene)
        elif ext_map is not None and gene in uniprot_extra1.keys() and \
                ((uniprot_extra1[gene] in mut_df['Hugo_Symbol'].values) or
                 (uniprot_extra1[gene] in expr_df.index.values) or
                 (uniprot_extra1[gene] in cnv_df.index.values)):
            second_mapped_genes.append(gene)
        elif ext_map is not None and gene in uniprot_extra2.keys() and \
                ((uniprot_extra2[gene] in mut_df['Hugo_Symbol'].values) or
                 (uniprot_extra2[gene] in expr_df.index.values) or
                 (uniprot_extra2[gene] in cnv_df.index.values)):
            second_mapped_genes.append(gene)
        else:
            unmapped_genes.append(gene)

    unmapped_df = pd.DataFrame(data={'gene_name': unmapped_genes})
    unmapped_loc = config.DATA_DIR / ('tissue_data/'+cancer.lower()+'_tcga/unmapped_v1.csv')
    unmapped_df.to_csv(unmapped_loc, index=None, header=None)
    missing_set = list(set(unmapped_genes))
    print(f'There are {len(unmapped_genes)} genes that are not mapped.')


def check_gtex_mappings_and_trim(cancer, cancer_dict):
    all_genes = np.array([])
    cancer_data = cancer_dict[cancer]
    cancer_genes = np.union1d(cancer_data['gene1'].values, cancer_data['gene2'].values)
    all_genes = np.union1d(all_genes, cancer_genes)

    loc_dict = tcga.get_locs()
    expr_df = tcga.load_cancer_gtex_expr(loc_dict[cancer]['gtex_expression'])

    mapped_genes = []
    second_mapped_genes = []
    unmapped_genes = []

    uniprot_extra_mapping_loc = 'tissue_data/'+cancer.lower()+'_gtex/uniprot_extra_mapping.tab'
    ext_map = get_uniprot_extra_mapping(uniprot_extra_mapping_loc)
    if ext_map is None:
        print(f'No extra mapping')
    else:
        uniprot_extra1, uniprot_extra2 = ext_map

    for gene in all_genes:
        if gene in expr_df.index.values:
            mapped_genes.append(gene)
        elif ext_map is not None and gene in uniprot_extra1.keys() and \
                (uniprot_extra1[gene] in expr_df.index.values):
            second_mapped_genes.append(gene)
        elif ext_map is not None and gene in uniprot_extra2.keys() and \
                (uniprot_extra2[gene] in expr_df.index.values):
            second_mapped_genes.append(gene)
        else:
            unmapped_genes.append(gene)

    unmapped_df = pd.DataFrame(data={'gene_name': unmapped_genes})
    unmapped_loc = config.DATA_DIR / ('tissue_data/'+cancer.lower()+'_gtex/unmapped_v2.csv')
    unmapped_df.to_csv(unmapped_loc, index=None, header=None)
    missing_set = list(set(unmapped_genes))
    print(f'There are {len(unmapped_genes)} genes that are not mapped.')


def create_fasta_mappings(fasta_loc):
    fasta_loc = config.DATA_DIR / fasta_loc
    fasta_data = SeqIO.parse(open(fasta_loc), 'fasta')
    df = pd.DataFrame(columns=['uniprot_id', 'uniprot_name', 'gene_name'])
    for i, fasta in enumerate(fasta_data):
        id, sequence, desc = fasta.id, str(fasta.seq), str(fasta.description)
        if 'GN=' in desc:
            gn_ind = desc.find('GN=')
            spc_ind = desc.find(' ', gn_ind+3)
            gene_name = desc[gn_ind+3: spc_ind]
        else:
            gene_name = ''

        *tmp, uniprot_id, uniprot_name = id.split('|')
        uniprot_name = uniprot_name.split('_')[0]
        df.loc[i]=[uniprot_id, uniprot_name, gene_name]
    return df

def is_mapped_in(key, mappings=[], target_list=[]):
    ppi_fasta_loc = config.DATA_DIR / 'sequences/ppi.fasta'
    ppi_fasta = SeqIO.parse(open(ppi_fasta_loc), 'fasta')
    uniprot_fasta_last = SeqIO.parse(open(uniprot_fasta_last_loc), 'fasta')
    for mapping in mappings:
        if key in mapping.keys() and mapping[key] in target_list:
            return None
    #print(f'{key} mapping not found.')
    return key


def get_uniprot_extra_mapping(loc):
    loc = config.DATA_DIR / loc
    if os.path.isfile(loc):
        data = pd.read_csv(loc, sep='\t')
    else:
        return None
    data.columns = ['query', 'uniprot_id', 'uniprot_name', 'status', 'prot_names', 'gene_names', 'organism', 'primary_gene_names','synonym_gene_names']
    reviewed = data[data['status'] == 'reviewed']

    reviewed['uniprot_name'] = reviewed['uniprot_name'].str.split('_', n=1, expand=True)[0]

    query2uniprot = data.set_index('query')['uniprot_name'].to_dict()
    query2genename = data.set_index('query')['primary_gene_names'].to_dict()
    return query2uniprot, query2genename

def find_intersection_unmapped(folders):
    locs = []
    unmapped_dict={}
    genes = np.array([])
    for folder in folders:
        tmp_loc = config.DATA_DIR / folder / 'unmapped_v2.csv'
        locs.append(tmp_loc)
        unmapped_dict[folder]=pd.read_csv(tmp_loc, header=None, names=['genes'])
        genes = np.intersect1d(genes, unmapped_dict[folder]['genes'].values)

    print(genes)


def analyze_train_test(folder):
    train_loc = config.DATA_DIR /folder / 'train_pairs.csv'
    test_loc = config.DATA_DIR /folder / 'test_pairs.csv'
    train_df = pd.read_csv(train_loc)
    test_df = pd.read_csv(test_loc)
    cancers = train_df['cancer'].unique()
    for cancer in cancers:
        cancer_train = train_df[train_df['cancer']==cancer]
        cancer_test = test_df[test_df['cancer']==cancer]
        cancer_train_total = cancer_train.shape[0]
        cancer_test_total = cancer_test.shape[0]
        cancer_train_sl = cancer_train[cancer_train['class']==1].shape[0]
        cancer_train_nonsl = cancer_train[cancer_train['class']==0].shape[0]
        cancer_test_sl = cancer_test[cancer_test['class']==1].shape[0]
        cancer_test_nonsl = cancer_test[cancer_test['class']==0].shape[0]
        print(f'Train->\t{cancer}:\tSL:{cancer_train_sl}\tnonSL:{cancer_train_nonsl}\tTotal:{cancer_train_total}')
        print(f'Test->\t{cancer}:\tSL:{cancer_test_sl}\tnonSL:{cancer_test_nonsl}\tTotal:{cancer_test_total}')


def main():
    #cancer_dict = get_golden_truth_by_cancer()
    #check_seq_mappings_and_trim(cancer_dict)
    #check_dep_mappings_and_trim(cancer_dict)
    #check_ppi_mappings_and_trim(cancer_dict)
    #check_tissue_mappings_and_trim('SKCM', cancer_dict)
    #check_gtex_mappings_and_trim('SKCM', cancer_dict)
    #split_dataset_train_test(cancer_dict, 0.2)
    #create_pairs_for_each_source()
    a= create_pairs_for_each_source()
    print()
    #folders = ['sequences', 'PPI/STRING', 'ccle_broad_2019']
    #find_intersection_unmapped(folders)
    #analyze_train_test(folder='labels')


if __name__ == '__main__':
    main()
