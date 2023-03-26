import rpy2
from rpy2 import robjects
from rpy2.robjects import pandas2ri
import collections as coll
import pandas as pd
import numpy as np
import os
import gzip
from Bio import SeqIO
from src import config
import csv
# import pyreadr
import pickle
from itertools import islice

PROJECT_LOC = config.ROOT_DIR


# read .RData file as a pandas dataframe
def load_rdata_file(file_loc, isRDS=False, is_abs_loc=False):
    pandas2ri.activate()
    data_loc = file_loc
    if not is_abs_loc:
        data_loc = config.DATA_DIR / data_loc
    if isRDS:
        r_data = robjects.r['readRDS'](str(data_loc))
    else:
        r_data = robjects.r['get'](robjects.r['load'](data_loc))
    try:
        df = pandas2ri.ri2py(r_data)
    except:
        df=r_data
    return df


# write pandas dataframe to an .RData file
def save_rdata_file(df, file_loc, isRDS=False, is_abs_loc=False):
    pandas2ri.activate()
    data_loc = file_loc
    if not is_abs_loc:
        data_loc = config.DATA_DIR / data_loc
    try:
        r_data = pandas2ri.py2ri(df)
    except:
        r_data = pandas2ri.py2rpy_pandasdataframe(df)
    robjects.r.assign("my_df", r_data)
    if isRDS:
        robjects.r("saveRDS(my_df, file='{}')".format(data_loc))
    else:
        robjects.r("save(my_df, file='{}')".format(data_loc))


def rdata2csv(file_loc, isRDS=False):
    new_loc = (config.DATA_DIR / file_loc).with_suffix('.csv')
    df = load_rdata_file(file_loc, isRDS=isRDS)
    df = df.rename(columns={'cancer_type':'cancer', 'SL': 'class'}, inplace=False)
    df.to_csv(new_loc, index=None, sep=',')

def save_pickle(loc, data):
    with open(loc, "wb") as output_file:
        pickle.dump(data, output_file, protocol=pickle.HIGHEST_PROTOCOL)


def load_pickle(loc):
    with open(loc, "rb") as output_file:
        data = pickle.load(output_file)
    return data


def chunks(data, SIZE=10000):
    it = iter(data)
    for i in range(0, len(data), SIZE):
        yield {k:data[k] for k in islice(it, SIZE)}


def load_csv(loc, sep=',', out_type='dict', vals=2):
    if out_type=='dict':
        out={}
    with open(loc) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=sep)
        for row in csv_reader:
            out[row[0]]=row[2:]
    return out

def date2_genename(x):
    convertion = {'03': 'MARCH', '09': 'SEPT', '12': 'DEC'}
    month = x.split(' ')[0].split('-')[-2]
    day = x.split(' ')[0].split('-')[-1]
    if int(day) < 10:
        day = day[-1]
    corrected = convertion[month] + day
    return corrected


def prepare_cancer_dataset(df, cancer='BRCA', reverse=False, is_cancer=False, reduce_min=False):
    df_cancer = df.copy()
    if cancer is None:
        df_cancer = df_cancer.reset_index()
    elif reverse:
        df_cancer = df_cancer[~(df_cancer['cancer'] == cancer)].reset_index()
    else:
        df_cancer = df_cancer[df_cancer['cancer'] == cancer].reset_index()

    df_cancer = df_cancer.sort_values(by=['gene1', 'gene2'])
    df_cancer.insert(loc=0, column='pair_name', value=df_cancer[['gene1', 'gene2']].agg('|'.join, axis=1))
    if is_cancer:
        df_cancer = df_cancer.drop(columns=['gene1', 'gene2', 'index'])
    else:
        df_cancer = df_cancer.drop(columns=['gene1', 'gene2', 'cancer', 'index'])
    return df_cancer


def process_ready_data(data_loc):
    # r_data = load_rdata_file(data_loc, isRDS=True)
    pandas2ri.activate()
    data_loc = config.DATA_DIR / data_loc
    r_data = robjects.r['readRDS'](str(data_loc))
    folds_dict = coll.OrderedDict()
    for fold in r_data.names:
        fold_data = r_data.rx2(fold)
        fold_dict = coll.OrderedDict()
        for data_name in fold_data.names:
            df = pandas2ri.ri2py(fold_data.rx2(data_name))
            df['SL'] = pd.Series(np.where(df.SL.values == 'Y', 1, 0), df.index)
            fold_dict[data_name] = df
        folds_dict[fold] = fold_dict

    return folds_dict


def find_diff(ds1, ds2, columns='sl_pair', remove_from='first'):
    dupl = np.intersect1d(ds1[columns].values, ds2[columns].values)

    if remove_from == 'first':
        df1 = ds1[~ds1[columns].isin(dupl)]
        return df1, ds2, dupl
    elif remove_from in ['second', 'last']:
        df2 = ds2[~ds2[columns].isin(dupl)]
        return ds1, df2, dupl
    else:
        return ds1, ds2, dupl


def get_PPI_data(in_file='PPI/STRING/9606.protein.links.full.v11.0.txt', cutoff_list=[0.0], source='ec'):
    data_loc = config.DATA_DIR / in_file
    df = pd.read_csv(data_loc, sep=" ")
    # exp_cur_df = df[(df['experiments'] > 0) | (df['database'] > 0) | (df['coexpression'] > 0)]
    # Only the one from experiments
    # df = df[df['experiments'] != 0]
    cutoff_dict = {}
    for cutoff in cutoff_list:
        if 'e' in source and 'c' in source:
            cutoff_df = df[(df['experiments'] > cutoff * 1000) | (df['database'] > cutoff * 1000)]
        elif 'e' in source:
            cutoff_df = df[df['experiments'] > cutoff * 1000]
        elif 'c' in source:
            cutoff_df = df[df['database'] > cutoff * 1000]

        cutoff_dict[str(cutoff)] = cutoff_df
        unique_genes = np.unique(np.concatenate([cutoff_df['protein1'].values, cutoff_df['protein2'].values]))
        mean = np.average(cutoff_df['experiments'].values)
        max = np.max(cutoff_df['experiments'].values)
        min = np.min(cutoff_df['experiments'].values)
        # print(f'There are {len(unique_genes)} genes and {cutoff_df.shape[0]} interactions by cutoff {cutoff} '
        #      f'score-->mean: {mean}, max: {max}, min: {min}')

    return cutoff_dict, df


def get_PPI_fasta_seq(in_file='PPI/STRING/9606.protein.sequences.v11.0.fa.gz'):
    data_loc = config.DATA_DIR / in_file
    fasta_list = list(SeqIO.parse(gzip.open(data_loc, "rt"), "fasta"))
    fasta_seq_dict = {}
    for gene in fasta_list:
        fasta_seq_dict[gene.id] = str(gene.seq)
    ordered_dict = coll.OrderedDict(sorted(fasta_seq_dict.items()))

    return ordered_dict


def get_PPI_entrez_maps(in_file='PPI/STRING/human.entrez_2_string.2018.tsv'):
    data_loc = config.DATA_DIR / in_file
    df = pd.read_csv(data_loc, sep="\t", header=None, names=['taxid', 'entrez', 'STRING'], skiprows=1)
    entrez2string = df.set_index('entrez')['STRING'].to_dict()
    string2entrez = df.set_index('STRING')['entrez'].to_dict()

    return entrez2string, string2entrez


def get_PPI_GO_maps(in_file='PPI/STRING/human.GO_2_string.2018.tsv'):
    return 0


def get_PPI_genename_maps(in_file='PPI/STRING/human.name_2_string.tsv'):
    data_loc = config.DATA_DIR / in_file
    df = pd.read_csv(data_loc, sep="\t", header=None, names=['taxid', 'name', 'STRING'], skiprows=1)
    name2string = df.set_index('name')['STRING'].to_dict()
    string2name = df.set_index('STRING')['name'].to_dict()

    return name2string, string2name


def get_PPI_refseq_maps(in_file='PPI/STRING/human.refseq_2_string.2018.tsv'):
    return 0


def get_PPI_uniprot_maps(in_file='PPI/STRING/human.uniprot_2_string.2018.tsv', aim='name'):
    data_loc = config.DATA_DIR / in_file
    df = pd.read_csv(data_loc, sep="\t", header=None, names=['taxid', 'combination', 'STRING', 's1', 's2'])
    df[['uniprot', 'name']] = df['combination'].str.split('|', n=1, expand=True)
    df['name'] = df['name'].str.split('_', n=1, expand=True)[0]
    df = df.drop(columns=['taxid', 'combination', 's1', 's2'])
    if aim == 'name':
        name2string = df.set_index('name')['STRING'].to_dict()
        string2name = df.set_index('STRING')['name'].to_dict()
    elif aim == 'uniprot':
        name2string = df.set_index('uniprot')['STRING'].to_dict()
        string2name = df.set_index('STRING')['uniprot'].to_dict()

    return name2string, string2name


def ppiseq2fasta(ppiseq_dict, fasta_out):
    out_loc = config.DATA_DIR / fasta_out

    with open(out_loc, 'w') as out_file:
        for id, seq in ppiseq_dict.items():
            name = '>' + str(id)
            out_file.write(name)
            out_file.write("\n")
            out_file.write(str(seq))
            out_file.write("\n")


def get_inbio_ppi(file_loc='PPI/InBio_Map_core_2016_09_12/core.psimitab'):
    #https://inbio-discover.intomics.com/map.html?utm_source=intern&utm_medium=webpage#downloads
    #https://www.biorxiv.org/content/biorxiv/suppl/2016/07/26/064535.DC2/064535-1.pdf
    file_loc = config.DATA_DIR / file_loc
    df = pd.read_csv(file_loc, header=None, sep='\t')
    return df

def get_tissue_mutation(loc):
    return 0


def get_ISLE_training_set(dataset_name='labels/ISLE.xlsx', include_mut = True):
    data_loc = config.DATA_DIR / dataset_name
    df = pd.read_excel(data_loc, skiprows=2)
    df = df.astype({"gene1": str, "gene2": str})
    df['gene1'] = df['gene1'].apply(lambda x: date2_genename(x) if '00:00:00' in x else x)
    df['gene2'] = df['gene2'].apply(lambda x: date2_genename(x) if '00:00:00' in x else x)
    if not include_mut:
        df = df[(df['gene1 perturbation']!='mut')|(df['gene2 perturbation']!='mut')]

    df = df[['gene1', 'gene2', 'SL', 'cancer type tested', 'PMID']]
    df = df.dropna()
    df.columns = ['gene1', 'gene2', 'class', 'cancer', 'pmid']

    # Process so that for all the rows, gene1 name < gene2 name
    gene_switch_ids = (df['gene1'] > df['gene2'])
    df.loc[gene_switch_ids, ['gene1', 'gene2']] = df.loc[gene_switch_ids, ['gene2', 'gene1']].values

    df = df.sort_values(by=['cancer', 'gene1', 'gene2'])
    print()
    # df[(df['SL'] == 1) & (df['cancer type tested'] == 'COAD')].shape
    # df[df['cancer type tested'] == 'COAD'].groupby(['gene1', 'gene2'], as_index=False)['SL'].agg([np.max, np.min])
    return df


def get_DiscoverSL_training_set(dataset_name='labels/DiscoverSL.txt'):
    data_loc = config.DATA_DIR / dataset_name
    df = pd.read_csv(data_loc, sep="\t")
    df = df.dropna()
    df['Class'] = df['Class'].apply(lambda x: 1 if x == 'positive' else 0)
    df = df[['Gene1', 'Gene2', 'Class', 'Cancer']]
    df.columns = ['gene1', 'gene2', 'class', 'cancer']
    gene_switch_ids = (df['gene1'] > df['gene2'])
    df.loc[gene_switch_ids, ['gene1', 'gene2']] = df.loc[gene_switch_ids, ['gene2', 'gene1']].values
    df = df.sort_values(by=['cancer', 'gene1', 'gene2'])
    return df


def get_lu15_data(dataset_name='labels/lu15.xlsx'):
    data_loc = config.DATA_DIR / dataset_name
    df = pd.read_excel(data_loc, header=0)
    df = df.astype({"geneA_Symbol": str, "geneB_Symbol": str})
    df['geneA_Symbol'] = df['geneA_Symbol'].apply(lambda x: date2_genename(x) if '00:00:00' in x else x)
    df['geneB_Symbol'] = df['geneB_Symbol'].apply(lambda x: date2_genename(x) if '00:00:00' in x else x)
    df['class'] = df['class'].apply(lambda x: 1 if x == 'Neg' else 0)
    df['cancer'] = 'COAD'
    df['pmid'] = '24104479,23563794'

    df = df[['geneA_Symbol', 'geneB_Symbol', 'class', 'cancer', 'pmid']]
    df = df.dropna()
    df.columns = ['gene1', 'gene2', 'class', 'cancer', 'pmid']

    # Process so that for all the rows, gene1 name < gene2 name
    gene_switch_ids = (df['gene1'] > df['gene2'])
    df.loc[gene_switch_ids, ['gene1', 'gene2']] = df.loc[gene_switch_ids, ['gene2', 'gene1']].values

    df = df.sort_values(by=['cancer', 'gene1', 'gene2'])
    print()
    # df[(df['SL'] == 1) & (df['cancer type tested'] == 'COAD')].shape
    # df[df['cancer type tested'] == 'COAD'].groupby(['gene1', 'gene2'], as_index=False)['SL'].agg([np.max, np.min])
    return df


def get_exp2sl_data(folder='database/exp2sl', return_type='dataframe'):
    folder = config.DATA_DIR / folder
    CCLE2cancer = {'A375': 'SKCM', 'A549': 'LUAD', 'HEK293T': 'LUAD', 'HT29': 'COAD'}
    CCLE2pmid = {'A375': '29251726', 'A549': '29452643,28319113,29251726', 'HEK293T': '28319113', 'HT29': '29251726'}
    dataset = {}
    all_df = pd.DataFrame(columns=['gene1', 'gene2', 'class', 'cancer', 'pmid'])
    for file in os.listdir(folder):
        if file.startswith("gemini") and file.endswith(".tsv"):
            data_name = file.split('_')[1]
            new_df = pd.read_csv(os.path.join(folder, file), delimiter='\t', header=None,
                                 names=['gene1', 'entrez1', 'gene2', 'entrez2', 'class'])
            new_df['cell_line'] = data_name
            new_df['cancer'] = CCLE2cancer[data_name]
            new_df['pmid'] = CCLE2pmid[data_name]
            new_df = new_df[['gene1', 'gene2', 'class', 'cancer', 'pmid']]
            new_df = new_df.dropna()
            switch_ids = (new_df['gene1'] > new_df['gene2'])
            new_df.loc[switch_ids, ['gene1', 'gene2']] = new_df.loc[switch_ids, ['gene2', 'gene1']].values
            new_df = new_df.sort_values(by=['gene1', 'gene2'])
            dataset[data_name] = new_df
            all_df = pd.concat([all_df,new_df])
    if return_type=='dataframe':
        # Process so that for all the rows, gene1 name < gene2 name
        gene_switch_ids = (all_df['gene1'] > all_df['gene2'])
        all_df.loc[gene_switch_ids, ['gene1', 'gene2']] = all_df.loc[gene_switch_ids, ['gene2', 'gene1']].values
        all_df = all_df.sort_values(by=['cancer', 'gene1', 'gene2'])
        return all_df
    elif return_type=='dict':
        return dataset
    else:
        return all_df, dataset


def get_SynLethDB_data(folder='database/SynLethDB', organism='Human'):
    important_sources = ['GenomeRNAi', 'Decipher', 'CRISPR/CRISPRi', 'CRISPR screen', 'Decipher;Text Mining',
                         'Synlethality', 'RNAi Screen', 'BioGrid', 'chemical library screen',
                         'Synlethality;Decipher', 'RNAi Screen (shRNA)?+?Gefitinib', 'Decipher;Daisy',
                         'GenomeRNAi;Text Mining', 'RNAi Screen (shRNA)', 'GenomeRNAi;Decipher', 'BioGRID',
                         'RNAi Screen (siRNA)?+?Cisplatin', 'Compound library screen',
                         'Synlethality;Text Mining', 'Chemical library Compound Screen',
                         'Synlethality;GenomeRNAi', 'RNAi Screen (siRNA)?+?Low-Dose Cisplatin',
                         'Text Mining;Synlethality']

    folder_loc = config.DATA_DIR / folder
    sl_data_loc = folder_loc / (organism + '_SL.csv')
    non_sl_data_loc = folder_loc / (organism + '_nonSL.csv')
    sr_data_loc = folder_loc / (organism + '_SR.csv')

    sl_df = pd.read_csv(sl_data_loc, sep=",")
    non_sl_df = pd.read_csv(non_sl_data_loc, sep=",")
    sr_df = pd.read_csv(sr_data_loc, sep=",")

    sl_df['class'] = [1] * len(sl_df)
    non_sl_df['class'] = [0] * len(non_sl_df)
    sr_df['class'] = [-1] * len(sr_df)

    sl_df = sl_df[['gene_a.name', 'gene_b.name', 'class', 'SL.source', 'SL.cellline', 'SL.pubmed_id']]
    sl_df.columns = ['gene1', 'gene2', 'class', 'source', 'cell_line', 'pmid']
    sl_df = sl_df[sl_df['source'].isin(important_sources)]
    sl_gene_switch_ids = (sl_df['gene1'] > sl_df['gene2'])
    sl_df.loc[sl_gene_switch_ids, ['gene1', 'gene2']] = sl_df.loc[sl_gene_switch_ids, ['gene2', 'gene1']].values
    sl_df = sl_df.sort_values(by=['gene1', 'gene2'])

    non_sl_df = non_sl_df[['gene_a.name', 'gene_b.name', 'class', 'NonSL.source', 'NonSL.cell_line', 'NonSL.pubmed_id']]
    non_sl_df.columns = ['gene1', 'gene2', 'class', 'source', 'cell_line', 'pmid']
    non_sl_df = non_sl_df[non_sl_df['source'].isin(important_sources)]
    non_sl_gene_switch_ids = (non_sl_df['gene1'] > non_sl_df['gene2'])
    non_sl_df.loc[non_sl_gene_switch_ids, ['gene1', 'gene2']] = non_sl_df.loc[
        non_sl_gene_switch_ids, ['gene2', 'gene1']].values
    non_sl_df = non_sl_df.sort_values(by=['gene1', 'gene2'])

    sr_df['SR.cell_line'] = np.nan
    sr_df = sr_df[['gene_a.name', 'gene_b.name', 'class', 'SR.source', 'SR.cell_line', 'SR.pubmed_id']]
    sr_df.columns = ['gene1', 'gene2', 'class', 'source', 'cell_line', 'pmid']
    sr_df = sr_df[sr_df['source'].isin(important_sources)]
    sr_gene_switch_ids = (sr_df['gene1'] > sr_df['gene2'])
    sr_df.loc[sr_gene_switch_ids, ['gene1', 'gene2']] = sr_df.loc[sr_gene_switch_ids, ['gene2', 'gene1']].values
    sr_df = sr_df.sort_values(by=['gene1', 'gene2'])

    # sl_nosl_df = pd.concat([sl_df, non_sl_df])
    all_df = pd.concat([sl_df, non_sl_df, sr_df])
    all_df = all_df.sort_values(by=['gene1', 'gene2'])
    '''
    genepair_df = all_df.groupby(['gene1', 'gene2'])['class'].agg(['max', 'min', 'mean','count']).reset_index()
    agreement_df = genepair_df[(genepair_df['max'] == genepair_df['min'])]
    #disagreement_df = genepair_df[genepair_df['max'] != genepair_df['min']]
    disagreement_df = genepair_df[(genepair_df['max'] == 1) & (genepair_df['min'] != 1)]
    for idx, row in disagreement_df.iterrows():
        print(all_df[(all_df['gene1'] == row['gene1']) & (all_df['gene2'] == row['gene2'])])

    all_df['source'].unique().shape ## 33 different source
    all_df['pmid'].unique().shape ## 531 different pmid
    sources = sl_df['source'].unique()
    for source in sources:
        size=sl_df[sl_df['source'] == source].shape[0]
        print(f'{source} has {size} gene pairs.')
    #df = df[['gene_a.name', 'gene_b.name', 'class', 'SL.source']]
    #df.columns = ['gene1', 'gene2', 'class', 'source']
    '''

    return sl_df, non_sl_df, sr_df, all_df


def get_colm_data(in_file='labels/combined.RData'):
    pandas2ri.activate()
    data_loc = config.DATA_DIR / in_file
    # rdata = pyreadr.read_r(data_loc) # also works for Rds
    import rpy2.robjects as robjects
    from rpy2.robjects import pandas2ri
    rdata = robjects.r['readRDS'](str(data_loc))
    df = pandas2ri.ri2py_dataframe(rdata)

    return df


def list_diff(li1, li2):
    return list(list(set(li1) - set(li2)) + list(set(li2) - set(li1)))


def get_extra_mapping(loc='PPI/STRING/hand_mapping.tsv'):
    loc = config.DATA_DIR / loc
    df = pd.read_csv(loc, names=['name','STRING'], header=None, sep='\t')
    name2string = df.set_index('name')['STRING'].to_dict()
    string2name = df.set_index('STRING')['name'].to_dict()

    return name2string, string2name


def gene_name_to_uniprot(converted_list_loc='PPI/STRING/uniprot-yourlist_M20210113A94466D2655679D1FD8953E075198DA82E2F63D.tab'):
    loc = config.DATA_DIR / converted_list_loc
    data = pd.read_csv(loc, sep='\t')
    data.columns = ['query', 'uniprot_id', 'uniprot_name', 'status', 'prot_names', 'gene_names', 'organism', 'length']
    reviewed = data[data['status'] == 'reviewed']
    unreviewed = data[data['status'] == 'unreviewed']
    reviewed.groupby('query')['gene_names'].count()
    grouped_reviewed = reviewed.groupby('query')['gene_names'].count()
    multiple_response = grouped_reviewed[grouped_reviewed > 1]

    aa = data.groupby(['query'])['Status'].agg({'is_rev': lambda x: 'reviewed' in x}).reset_index()


def unmapped_to_uniprot_query(loc='unmapped_train_seq_1024_only.data', out_loc = 'unmapped_train_seq_1024_query.txt'):
    unmapped_data = load_pickle(loc)
    df = pd.DataFrame(data={'unmapped':unmapped_data})
    df.to_csv(out_loc, header=None, index=None)


def get_intersection_colm_ppi(colm_data, ppi_data, ppi_all_data, string2name, cut_off):
    colm_all_genes = np.unique(np.concatenate([colm_data['gene1'].values, colm_data['gene2'].values]))
    ppi_df = ppi_data[str(cut_off)]
    n2s_extra_map, s2n_extra_map = get_PPI_uniprot_maps()
    all_maps = {**s2n_extra_map, **string2name}
    ppi_all_genes_entrez = np.unique(np.concatenate([ppi_df['protein1'].values, ppi_df['protein2'].values]))
    ppi_all_gene_names = list(map(all_maps.get, ppi_all_genes_entrez))
    common_genes = list(set(colm_all_genes) & set(ppi_all_gene_names))
    print()


def strip_dataframe(df, start_col=0, end_col=None, start_row=0, end_row=None):
    if end_col is None: end_col = df.shape[1]+1
    #print(f'Stripping columns from {start_col} to {end_col}')
    if end_row is None: end_row = df.shape[0]+1
    #print(f'Stripping rows from {start_row} to {end_row}')
    stripped_df = df.iloc[start_row:end_row, start_col:end_col]
    return stripped_df


def remove_duplicates_inner(dataset):
    cancer_types = np.unique(dataset['cancer'])
    for cancer in cancer_types:
        unique_stats = dataset[dataset['cancer'] == cancer].groupby(['gene1', 'gene2'])['class'].agg(['count', 'max', 'min'])
        duplicate_pair_size = unique_stats[unique_stats['count']>1].shape[0]
        disagree_pair_size = unique_stats[(unique_stats['count']>1)&(unique_stats['max']!=unique_stats['min'])].shape[0]
        agree_pair_size =unique_stats[(unique_stats['count']>1)&(unique_stats['max']==unique_stats['min'])].shape[0]
        jaccard_score = 0
        if duplicate_pair_size != 0:
            jaccard_score = 1-(agree_pair_size/duplicate_pair_size)
        print(f"For {cancer}: There are {duplicate_pair_size} duplicates where {agree_pair_size} agrees and {disagree_pair_size} disagrees. Jaccard Score: {jaccard_score}")

    print(f"Removing disagreements and duplicates...")
    unique_stats = dataset.groupby(['gene1', 'gene2','cancer'])['class'].agg(
        ['max', 'min'])
    unique_data = unique_stats[unique_stats['max'] == unique_stats['min']].reset_index()
    unique_data['class'] = unique_data['max']
    unique_data = unique_data.drop(columns=['max', 'min'])

    for cancer in cancer_types:
        neg_size = unique_data[(unique_data['cancer']==cancer)&(unique_data['class']==0)].shape[0]
        pos_size = unique_data[(unique_data['cancer']==cancer)&(unique_data['class']==1)].shape[0]
        print(f'{cancer} has {pos_size} positive samples and {neg_size} negative samples.')

    return unique_data


def remove_duplicates_outer(cancer_dict):
    for cancer, cancer_data in cancer_dict.items():
        grouped_pairs = cancer_data.groupby(['gene1', 'gene2'])['from','class'].agg(
        {'from': {'list': lambda x: list(set(list(x))), 'unique': lambda y: len(set(list(y)))},
         'class': ['max', 'min', 'mean', 'count']}).reset_index()

        agree_pair_size = grouped_pairs[(grouped_pairs['max'] == grouped_pairs['min'])&(grouped_pairs['count'] > 1)].shape[0]
        disagree_pair_size = grouped_pairs[(grouped_pairs['max'] != grouped_pairs['min'])&(grouped_pairs['count'] > 1)].shape[0]
        duplicate_pair_size = grouped_pairs[grouped_pairs['count'] > 1].shape[0]
        total_size = grouped_pairs.shape[0]
        print(
            f"For {cancer}: There are {duplicate_pair_size} duplicates where {agree_pair_size} agrees and {disagree_pair_size} disagrees. Total samples: {total_size}")

        agree_data = grouped_pairs[grouped_pairs['max'] == grouped_pairs['min']]
        agree_data['class'] = agree_data['max']
        agree_data = agree_data.drop(columns=['max', 'min', 'mean', 'count', 'list', 'unique'])

        neg_size = agree_data[agree_data['class'] == 0].shape[0]
        pos_size = agree_data[agree_data['class'] == 1].shape[0]
        print(f'{cancer} has {pos_size} positive samples and {neg_size} negative samples.')
        cancer_dict[cancer] = agree_data
    return cancer_dict

def get_loc_dict():
    loc_dict = {}
    loc_dict['fasta_file_loc'] = 'sequences/UP000005640_9606.fasta'
    loc_dict['prot_dict_loc'] = 'sequences/sorted_fasta_sequences_dict.json'
    loc_dict['isle_dataset_loc'] = 'labels/ISLE.xlsx'
    loc_dict['lu15_dataset_loc'] = 'labels/lu15.xlsx'
    loc_dict['discoverSL_dataset_loc'] = 'labels/DiscoverSL.txt'
    loc_dict['PPI_dataset_loc'] = 'PPI/STRING/9606.protein.links.full.v11.0.txt'
    return loc_dict


def delete_model(folder='results/elrrf/models'):
    folder = config.ROOT_DIR / folder
    for file in os.listdir(folder):
        if file.endswith(".pickle") and 'True_True' in file and 'train_test' in file:
            file = folder / file
            tmp = load_pickle(file)
            for key1, item1 in tmp.items():
                if type(key1) == int:
                    for key2, val2 in item1.items():
                        tmp[key1][key2].pop('model')
            save_pickle(file, tmp)

def main():
    #loc_dict = get_loc_dict()
    #ppi_data, ppi_all_data = get_PPI_data(loc_dict['PPI_dataset_loc'],
    #                                      cutoff_list=[0.0, 0.1])
    #get_inbio_ppi()
    #print()
    #unmapped_to_uniprot_query()
    #gene_name_to_uniprot()
    #delete_model()
    #syn_sl_df, syn_non_sl_df, syn_sr_df, syn_df = get_SynLethDB_data()
    '''
    print()
    data_dict = {}
    exp2sl = get_exp2sl_data()
    lu15 = get_lu15_data(get_loc_dict()['lu15_dataset_loc'])
    isle = get_ISLE_training_set()
    dsl = get_DiscoverSL_training_set()
    syn_sl_df, syn_non_sl_df, syn_sr_df, syn_df = get_SynLethDB_data()
    colm = load_rdata_file('labels/combined.RData', isRDS=True)
    colm_dsl = load_rdata_file('labels/discoversl.RData', isRDS=True)
    colm_isle = load_rdata_file('labels/isle.RData', isRDS=True)
    data_dict['exp2sl'] = exp2sl
    data_dict['lu15'] = lu15
    data_dict['isle'] = isle
    data_dict['dsl'] = dsl
    #data_dict['syn_df'] = syn_df
    for name, data in data_dict.items():
        print(f'Analysis for {name} started.')
        data_dict[name] = remove_duplicates_inner(data)
        print(f'Analysis for {name} ended.\n')
    cancer_dict = {}
    for cancer in ['BRCA', 'CESC', 'COAD', 'KIRC', 'LAML', 'LUAD', 'OV', 'SKCM']:
        cancer_df = pd.DataFrame(columns=['gene1', 'gene2', 'cancer', 'class', 'from'])
        for name, data in data_dict.items():
            cancer_select = data[data['cancer']==cancer]
            cancer_select['from'] = name
            cancer_df = pd.concat([cancer_df, cancer_select])
        cancer_df = cancer_df.astype({"class": int})
        cancer_dict[cancer] = cancer_df
    remove_duplicates_outer(cancer_dict)
    data_dict['colm'] = colm
    data_dict['colm_dsl'] = colm_dsl
    data_dict['colm_isle'] = colm_isle
    '''
    '''
    remove_duplicates_inner()


    dsl.groupby(['cancer', 'class']).count()
    isle.groupby(['cancer', 'class']).count()
    lu15.groupby(['cancer', 'class']).count()
    colm.groupby(['cancer_type', 'SL']).count()

    grouped_isle = isle.groupby(['gene1', 'gene2', 'cancer'])['class'].agg(['max', 'min', 'mean', 'count'])
    agree_grouped_isle = grouped_isle[grouped_isle['max'] == grouped_isle['min']]
    agree_isle = agree_grouped_isle.reset_index()
    for cancer in np.unique(isle['cancer']):
        agree_isle[(agree_isle['cancer'] == 'COAD') & (agree_isle['max'] == 0)]
        agree_isle[(agree_isle['cancer'] == 'COAD') & (agree_isle['max'] == 1)]
    '''
    #analysis = rdata_dataset_analysis()
    print()

    # ppi_data, ppi_all_data = get_PPI_data(loc_dict['PPI_dataset_loc'], cutoff_list=[0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.75,0.8,0.85,0.9,0.95])
    # ppi_seq_dict = get_PPI_fasta_seq()
    # ppiseq2fasta(ppi_seq_dict, 'sequences/ppi.fasta')
    # colm_data = get_colm_data()
    # name2string, string2name = get_PPI_genename_maps()
    # get_intersection_colm_ppi(colm_data,ppi_data, ppi_all_data,string2name, 0.0)


if __name__ == '__main__':
    main()
