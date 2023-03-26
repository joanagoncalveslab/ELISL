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
import src.embedding.sequence_embedding as seq_e
import src.data_functions as dfnc
#import src.datasets.cell_lines as cl
import src.embedding.dependency_embedding as dep_e
import src.embedding.tissue_embedding as tis_e

PROJECT_LOC = config.ROOT_DIR
mapping_not_found = []
ABBV = {'dimensions': 'dim', 'walk_length': 'wl', 'num_walks': 'nw', 'p': 'p', 'q': 'q', 'workers': 'ws',
            'window': 'wd', 'min_count': 'mc', 'batch_words': 'bw', 'cutoff': 'co'}


def get_mappings(target='ppi'):
    if target=='ppi':
        mappings = []
        mappings.append(dfnc.get_PPI_genename_maps()[0])
        mappings.append(dfnc.get_PPI_uniprot_maps(aim='name')[0])
        mappings.append(dfnc.get_extra_mapping()[0])
        extra_map1, extra_map2 = get_uniprot_extra_mapping('PPI/STRING')
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
    if target=='seq':
        mappings = []
        uniprot_fasta_mapping_loc = config.DATA_DIR / 'sequences/uniprot_reviewed_9606_mappings.csv'
        fasta_df = pd.read_csv(uniprot_fasta_mapping_loc)
        name2uid = fasta_df.set_index('gene_name')['uniprot_id'].to_dict()
        mappings.append(name2uid)
        uname2uid = fasta_df.set_index('uniprot_name')['uniprot_id'].to_dict()
        mappings.append(uname2uid)
        extra_map1, extra_map2 = get_uniprot_extra_mapping('sequences')
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


def get_uniprot_extra_mapping(folder):
    loc = config.DATA_DIR / folder / 'uniprot_extra_mapping.tab'
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


def create_seq_features(df, emb_dict={}, mappings=[{}], emb_name='', ready_data=None):
    emb_length = list(emb_dict.values())[0].shape[0]
    cols = []
    for i in range(emb_length):
        col_name = emb_name + str(i)
        cols.append(col_name)
        #df[col_name] = -1

    genes = np.union1d(df['gene1'].values, df['gene2'].values)
    name2vector = find_genes_mapping(genes, mappings, emb_dict)
    all_rows = np.zeros(shape=(len(df),emb_length))
    all_rows[:]=-1
    for ind, a_row in enumerate(df[['gene1','gene2']].values):
        if ind%10000==0:
            print(f'Iteration {ind} is done.')
        if (ready_data is not None):
            chosen_row = ready_data[(ready_data[['gene1', 'gene2', 'cancer']] == a_row[['gene1', 'gene2', 'cancer']].values).all(1)]
        if ready_data is not None and len(chosen_row)>0:
            try:
                df.loc[ind, 'seq0':] = chosen_row.loc[:, 'seq0':].values[0]
            except:
                pass
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


def create_ppi_features(df, emb_dict={}, mappings=[{}], emb_name='', ready_data=None):
    emb_length = list(emb_dict.values())[0].shape[0]
    cols = []
    for i in range(emb_length):
        col_name = emb_name + str(i)
        cols.append(col_name)
        #df[col_name] = 0


    genes = np.union1d(df['gene1'].values, df['gene2'].values)
    name2vector = find_genes_mapping(genes, mappings, emb_dict)

    all_rows = np.zeros(shape=(len(df),emb_length))
    all_rows[:]=-1
    for ind, a_row in enumerate(df[['gene1','gene2']].values):
        if (ready_data is not None):
            chosen_row = ready_data[(ready_data[['gene1', 'gene2', 'cancer']] == a_row[['gene1', 'gene2', 'cancer']].values).all(1)]
        if ready_data is not None and len(chosen_row)>0:
            df.loc[ind, 'ppi_ec0':] = chosen_row.loc[:, 'ppi_ec0':].values[0]
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


def create_onehot_features(df):
    all_genes = np.union1d(df['gene1'].values,df['gene2'].values)
    gene_dict = {}
    for i, gene in enumerate(all_genes):
        gene_dict[gene] = 'o'+str(i)
    emb_length = len(all_genes)
    tmp = pd.DataFrame(0, index=df.index, columns=['o' + str(i) for i in range(emb_length)])
    df = pd.concat([df, tmp], axis=1)
    for ind, row in df[['gene1','gene2']].iterrows():
        df.loc[ind, gene_dict[row['gene1']]] = 1
        df.loc[ind, gene_dict[row['gene2']]] = 1

    return df


def create_random_features(df, col_size=28):
    emb_length = col_size
    row_size = df.shape[0]
    for i in range(emb_length):
        col_name = 'rand' + str(i)
        df[col_name] = np.random.normal(0, 1, row_size)

    return df


def create_dep_features_per_omic(df, dep_df, mut_df, cancer2ccle, alt_name='mut', cutoff=None, ready_data=None):
    #for i in range(4):
    #    col_name = 'dep' + str(i)
    #    df[col_name] = 0
    if 'expr' in alt_name:
        df = dep_e.calculate_emb_expr(df, dep_df, mut_df, cancer2ccle, cutoff, ready_data)
    elif 'cnv' in alt_name:
        df = dep_e.calculate_emb_cnv(df, dep_df, mut_df, cancer2ccle, ready_data)
    else:
        df = dep_e.calculate_emb_mut(df, dep_df, mut_df, cancer2ccle, ready_data)
    #for ind, a_row in enumerate(df[['gene1','gene2', 'cancer']].values):
    #    pair_name = a_row[0]+'|'+a_row[1]+'|'+a_row[2]
    #    df.loc[ind, 'dep0':] = emb_dict[pair_name]
    return df


def create_dep_features_any_omic(df, dep_df, mut_df, expr_df, cnv_df, cancer2ccle, cutoff=None):
    for i in range(4):
        col_name = 'dep' + str(i)
        df[col_name] = 0

    emb_dict = dep_e.calculate_emb_any(df[['gene1', 'gene2','cancer']].values, dep_df, mut_df, expr_df, cnv_df, cancer2ccle, cutoff)
    for ind, row in df.iterrows():
        pair_name = row['gene1']+'|'+row['gene2']+'|'+row['cancer']
        df.loc[ind, 'dep0':] = emb_dict[pair_name]
    return df


def create_tissue_features(df, feature_type=None, cancer_types=[], cutoff=None, already_calc=False, ready_data=None):
    if False:
        if feature_type is None or 'surv' in feature_type:
            df['surv'] = 1
        if feature_type is None or 'tcoexp' in feature_type:
            df['tumor_coexp'] = 0
            df['tumor_coexp_p'] = 1
            df['normal_coexp'] = 0
            df['normal_coexp_p'] = 1
            df['tumor_cocnv'] = 0
            df['tumor_cocnv_p'] = 1
        if feature_type is None or 'hcoexp' in feature_type:
            df['gtex_coexp'] = 0
            df['gtex_coexp_p'] = 1
        if feature_type is None or 'diff_expr' in feature_type:
            df['gene1_expr_m1'] = 0
            df['gene1_expr_m0'] = 0
            df['gene2_expr_m1'] = 0
            df['gene2_expr_m0'] = 0
    for cancer in df['cancer'].unique():
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


def insert_features(feature='ppi_ec', sample='train', loc_dict={}, extra_opts={}, to_ready=False):
    print(f'Feature insertion started | {sample} | {feature}')
    if extra_opts:
        for key, item in extra_opts.items():
            if key in ABBV.keys():
                feature = feature+"_"+ABBV[key]+"="+str(item)
    data_loc = 'feature_sets/' + sample + '_' + feature + '.csv'
    gz_loc = data_loc+'.gz'
    data_loc = config.DATA_DIR / data_loc
    gz_loc = config.DATA_DIR / gz_loc
    if to_ready:
        try:
            data = pd.read_csv(gz_loc)
        except:
            data = pd.read_csv(data_loc)
    else:
        sample_loc = config.DATA_DIR / loc_dict[sample]
        data = pd.read_csv(sample_loc)


    if 'ppi' in feature:
        source = feature.split('_')[1]
        ppi_w2v_vector = ppi_e.load_embs(source=source, opts_extra=extra_opts)
        ppi_embs = ppi_e.get_embs_dict(ppi_w2v_vector)
        resulting_df = create_ppi_features(data, ppi_embs, get_mappings('ppi'), emb_name=feature)
    elif 'seq' in feature:
        dim=feature.split('_')[1]
        seq_emb_loc = loc_dict['seq_emb_'+dim+'_loc']
        seq_embs = seq_e.load_embeddings(seq_emb_loc)
        resulting_df = create_seq_features(data, seq_embs, get_mappings('seq'), emb_name='seq')
    elif 'onehot' in feature:
        resulting_df = create_onehot_features(data)
    elif 'random' in feature:
        resulting_df = create_random_features(data, col_size=extra_opts['dimensions'])
    elif 'depend' in feature and 'any' in feature:
        dep_df = dep_e.get_dataset(feature)
        cancer2ccle = dep_e.get_cancer2ccle()
        sample_df = dep_e.get_dataset('samples')
        exp_df = dep_e.get_dataset('expression')
        cnv_df = dep_e.get_dataset('cnv')
        mut_df = dep_e.get_dataset('mutation')
        resulting_df = create_dep_features_any_omic(data, dep_df, mut_df, exp_df, cnv_df, cancer2ccle,
                                              extra_opts['cutoff'])
    elif 'depend' in feature:
        dep_df = dep_e.get_dataset(feature)
        cancer2ccle = dep_e.get_cancer2ccle()
        if 'expr' in feature:
            mut_df = dep_e.get_dataset('expression')
        elif 'cnv' in feature:
            mut_df = dep_e.get_dataset('cnv')
        else:
            mut_df = dep_e.get_dataset('mutation')
        cutoff = extra_opts['cutoff'] if 'cutoff' in extra_opts.keys() else None
        resulting_df = create_dep_features_per_omic(data, dep_df, mut_df, cancer2ccle, feature, cutoff)
    elif 'tissue' in feature:
        feature_type=feature.split('_', 1)[1]
        cutoff = extra_opts['cutoff'] if 'cutoff' in extra_opts.keys() else None
        resulting_df = create_tissue_features(data, feature_type=feature_type,
                                              cancer_types=data['cancer'].unique(),
                                              cutoff=cutoff)

    resulting_df.to_csv(data_loc, index=None, sep=',')
    try:
        resulting_df.to_csv(gz_loc, index=None, sep=',', compression="gzip")
    except:
        print('Compressed file cannot be saved.')
    not_mappeds = list(set(mapping_not_found))
    unmap_loc = 'feature_sets/unmapped_'+sample+'_'+feature+'.data'
    unmap_loc = config.DATA_DIR / unmap_loc
    dfnc.save_pickle(unmap_loc, not_mappeds)


def insert_features2sets(aim='lu15', feature='seq_1024', loc_dict={}, extra_opts={}, to_ready=False):
    print(f'Feature insertion started | {aim} | {feature}')
    if extra_opts:
        for key, item in extra_opts.items():
            if key in ABBV.keys():
                feature = feature+"_"+ABBV[key]+"="+str(item)
    data_loc = 'feature_sets/' + aim + '_' + feature + '.csv'
    gz_loc = data_loc+'.gz'
    data_loc = config.DATA_DIR / data_loc
    gz_loc = config.DATA_DIR / gz_loc
    if to_ready:
        try:
            data = pd.read_csv(gz_loc)
        except:
            data = pd.read_csv(data_loc)
    else:
        sample_loc = config.DATA_DIR / loc_dict[aim]
        data = pd.read_csv(sample_loc)

    tr_loc = config.DATA_DIR / ('feature_sets/train_' + feature + '.csv')
    te_loc = config.DATA_DIR / ('feature_sets/test_' + feature + '.csv')
    tr_data = pd.read_csv(tr_loc)
    te_data = pd.read_csv(te_loc)
    all_data = pd.concat([tr_data, te_data])

    if 'ppi' in feature:
        source = feature.split('_')[1]
        ppi_w2v_vector = ppi_e.load_embs(source=source, opts_extra=extra_opts)
        ppi_embs = ppi_e.get_embs_dict(ppi_w2v_vector)
        resulting_df = create_ppi_features(data, ppi_embs, get_mappings('ppi'), emb_name=feature, ready_data=all_data)
    elif 'seq' in feature:
        dim=feature.split('_')[1]
        seq_emb_loc = loc_dict['seq_emb_'+dim+'_loc']
        seq_embs = seq_e.load_embeddings(seq_emb_loc)
        resulting_df = create_seq_features(data, seq_embs, get_mappings('seq'), emb_name='seq', ready_data=all_data)
    elif 'onehot' in feature:
        resulting_df = create_onehot_features(data)
    elif 'random' in feature:
        resulting_df = create_random_features(data, col_size=extra_opts['dimensions'])
    elif 'depend' in feature and 'any' in feature:
        dep_df = dep_e.get_dataset(feature)
        cancer2ccle = dep_e.get_cancer2ccle()
        sample_df = dep_e.get_dataset('samples')
        exp_df = dep_e.get_dataset('expression')
        cnv_df = dep_e.get_dataset('cnv')
        mut_df = dep_e.get_dataset('mutation')
        resulting_df = create_dep_features_any_omic(data, dep_df, mut_df, exp_df, cnv_df, cancer2ccle,
                                              extra_opts['cutoff'])
    elif 'depend' in feature:
        dep_df = dep_e.get_dataset(feature)
        cancer2ccle = dep_e.get_cancer2ccle()
        if 'expr' in feature:
            mut_df = dep_e.get_dataset('expression')
        elif 'cnv' in feature:
            mut_df = dep_e.get_dataset('cnv')
        else:
            mut_df = dep_e.get_dataset('mutation')
        cutoff = extra_opts['cutoff'] if 'cutoff' in extra_opts.keys() else None
        resulting_df = create_dep_features_per_omic(data, dep_df, mut_df, cancer2ccle, feature, cutoff, all_data)
    elif 'tissue' in feature:
        feature_type=feature.split('_', 1)[1]
        cutoff = extra_opts['cutoff'] if 'cutoff' in extra_opts.keys() else None
        resulting_df = create_tissue_features(data, feature_type=feature_type,
                                              cancer_types=data['cancer'].unique(),
                                              cutoff=cutoff, ready_data=all_data)

    resulting_df.to_csv(data_loc, index=None, sep=',')
    try:
        resulting_df.to_csv(gz_loc, index=None, sep=',', compression="gzip")
    except:
        print('Compressed file cannot be saved.')
    not_mappeds = list(set(mapping_not_found))
    unmap_loc = 'feature_sets/unmapped_'+aim+'_'+feature+'.data'
    unmap_loc = config.DATA_DIR / unmap_loc
    dfnc.save_pickle(unmap_loc, not_mappeds)


def correct_set_features(aim='lu15', feature='seq_1024', loc_dict={}):
    print(f'Feature insertion started | {aim} | {feature}')
    data_loc = 'feature_sets/' + aim + '_' + feature + '.csv'
    gz_loc = data_loc+'.gz'
    data_loc = config.DATA_DIR / data_loc
    gz_loc = config.DATA_DIR / gz_loc
    try:
        data = pd.read_csv(gz_loc)
    except:
        data = pd.read_csv(data_loc)
    sample_loc = config.DATA_DIR / loc_dict[aim]
    samples = pd.read_csv(sample_loc)
    res = pd.merge(samples, data.drop_duplicates(subset=["gene1", "gene2", "class", "cancer"]), how="left", on=["gene1", "gene2", "class", "cancer"])

    res.to_csv(data_loc, index=None, sep=',')
    try:
        res.to_csv(gz_loc, index=None, sep=',', compression="gzip")
    except:
        print('Compressed file cannot be saved.')



def insert_extra_to_tissue(feature='tissue_diff_expr', sample='train', loc_dict={}, extra_opts={}):
    print(f'Feature insertion started | {sample} | {feature}')
    if extra_opts:
        for key, item in extra_opts.items():
            if key in ABBV.keys():
                feature = feature+"_"+ABBV[key]+"="+str(item)
    data_loc = 'feature_sets/' + sample + '_' + feature + '.csv'
    gz_loc = data_loc+'.gz'
    data_loc = config.DATA_DIR / data_loc
    gz_loc = config.DATA_DIR / gz_loc
    try:
        data = pd.read_csv(gz_loc)
    except:
        data = pd.read_csv(data_loc)

    if 'tissue' in feature:
        feature_type=feature.split('_', 1)[1]
        cutoff = extra_opts['cutoff'] if 'cutoff' in extra_opts.keys() else None
        resulting_df = create_tissue_features(data, feature_type=feature_type,cancer_types=['CESC','KIRC','LAML','SKCM'],
                                              cutoff=cutoff, already_calc=True)

    resulting_df.to_csv(data_loc, index=None, sep=',')
    try:
        resulting_df.to_csv(gz_loc, index=None, sep=',', compression="gzip")
    except:
        print('Compressed file cannot be saved.')
    not_mappeds = list(set(mapping_not_found))
    unmap_loc = 'feature_sets/unmapped_'+sample+'_'+feature+'.data'
    unmap_loc = config.DATA_DIR / unmap_loc
    dfnc.save_pickle(unmap_loc, not_mappeds)

def find_missing_data(colm_data, ppi_data):
    ppi_mappings = []
    ppi_mappings.append(dfnc.get_PPI_genename_maps()[0])
    ppi_mappings.append(dfnc.get_PPI_uniprot_maps(aim='name')[0])
    ppi_mappings.append(dfnc.get_extra_mapping()[0])

    missing_ones = []
    colm_all_genes = np.unique(np.concatenate([colm_data['gene1'].values, colm_data['gene2'].values]))
    ppi_df = ppi_data
    ppi_all_genes_entrez = np.unique(np.concatenate([ppi_df['protein1'].values, ppi_df['protein2'].values]))
    for gene in colm_all_genes:
        not_mapped = is_mapped_in(gene, ppi_mappings, target_list=ppi_all_genes_entrez)
        if not_mapped != None:
            missing_ones.append(not_mapped)

    missing_set = list(set(missing_ones))
    print(f'There are {len(missing_set)} genes that are not mapped.')


def convert_csv2gzip(file_loc):
    out_loc = file_loc+'.gz'
    file_loc = config.DATA_DIR / file_loc
    out_loc = config.DATA_DIR / out_loc
    df = pd.read_csv(file_loc)
    df.to_csv(out_loc, index=None, sep=',', compression="gzip")


def convert_nans(feature='tissue', sample='train'):
    data_loc = 'feature_sets/' + sample + '_' + feature + '.csv'
    gz_loc = data_loc+'.gz'
    data_loc = config.DATA_DIR / data_loc
    gz_loc = config.DATA_DIR / gz_loc
    try:
        data = pd.read_csv(gz_loc)
    except:
        data = pd.read_csv(data_loc)

    data['surv'] = data.surv.fillna(1)
    data['tumor_coexp'] = data.tumor_coexp.fillna(0)
    data['tumor_coexp_p'] = data.tumor_coexp_p.fillna(1)
    data['normal_coexp'] = data.normal_coexp.fillna(0)
    data['normal_coexp_p'] = data.normal_coexp_p.fillna(1)
    data['tumor_cocnv'] = data.tumor_cocnv.fillna(0)
    data['tumor_cocnv_p'] = data.tumor_cocnv_p.fillna(1)
    data['gtex_coexp'] = data.gtex_coexp.fillna(0)
    data['gtex_coexp_p'] = data.gtex_coexp_p.fillna(1)
    data.to_csv(data_loc, index=None, sep=',')
    try:
        data.to_csv(gz_loc, index=None, sep=',', compression="gzip")
    except:
        print('Compressed file cannot be saved.')

def combine_2_dataset(ds_locs, suffixes, keys=['gene1', 'gene2', 'class', 'cancer'], out_loc=''):
    dfs =[]
    for i in range(len(ds_locs)):
        in_loc = config.DATA_DIR / 'feature_sets' / ds_locs[i]
        df = pd.read_csv(in_loc)
        df.columns = [col+'_'+suffixes[i] if 'dep' in col else col
                      for col in df.columns ]
        dfs.append(df)
    merged = dfs[0]
    for i in range(1,len(dfs)):
        merged = pd.merge(merged, dfs[i], on=keys)
    out_loc = config.DATA_DIR / 'feature_sets' / out_loc
    merged.to_csv(out_loc, index=False, compression="gzip")


def find_all_genes():
    emb_loc = config.DATA_DIR / 'embeddings' / 'seqvec' / 'uniprot_1024_embeddings.npz'
    emb_dict = seq_e.load_embeddings(emb_loc)
    uniprot_fasta_mapping_loc = config.DATA_DIR / 'sequences/uniprot_reviewed_9606_mappings.csv'
    fasta_df = pd.read_csv(uniprot_fasta_mapping_loc)
    name2uid = fasta_df.set_index('gene_name')['uniprot_id'].to_dict()
    uid2name = fasta_df.set_index('uniprot_id')['gene_name'].to_dict()

    genes = list(emb_dict.keys())
    gene_names = list(map(uid2name.get, genes))
    gene_names = np.sort(gene_names)
    gene_names = list(filter(lambda v: v == v, gene_names))
    repeated_gene1s = np.repeat(gene_names, len(gene_names) - 1)
    gene1s = []
    gene2s = []

    print()

def add_class_to_unknowns(sample='unknown_cancer_BRCA', is_label=True,
                          features=['seq_1024', 'ppi_ec', 'crispr_dependency_mut', 'crispr_dependency_expr_co=1.96', 'tissue']):
    for feature in features:
        csv_loc = config.DATA_DIR / 'feature_sets' / (sample+'_'+feature+'.csv')
        csv_gz_loc = config.DATA_DIR / 'feature_sets' / (sample+'_'+feature+'.csv.gz')
        if 'seq' in feature:
            df = pd.read_csv(csv_loc)
        else:
            df = pd.read_csv(csv_gz_loc)
        df['class']=0
        df.insert(2, 'class', df.pop('class'))
        df.to_csv(csv_loc, index=False)
        df.to_csv(csv_gz_loc, index=False, compression="gzip")

    if is_label:
        csv_label_loc = config.DATA_DIR / 'labels' / (sample + '_pairs.csv')
        labels = pd.read_csv(csv_label_loc)
        labels['class']=0
        labels.insert(2, 'class', labels.pop('class'))
        labels.to_csv(csv_label_loc, index=False)

def main():
    loc_dict = {}
    loc_dict['fasta_file_loc'] = 'sequences/UP000005640_9606.fasta'
    loc_dict['prot_dict_loc'] = 'sequences/sorted_fasta_sequences_dict.json'
    loc_dict['isle_dataset_loc'] = 'labels/ISLE_train_data.xlsx'
    loc_dict['discoverSL_dataset_loc'] = 'labels/DiscoverSL_trainingSet.txt'
    loc_dict['PPI_dataset_loc'] = 'PPI/STRING/9606.protein.links.full.v11.0.txt'
    for i in range(4,11,1):
        dim=str(int(np.power(2,i)))
        loc_dict['seq_emb_'+dim+'_loc'] = 'embeddings/seqvec/uniprot_'+dim+'_embeddings.npz'
    loc_dict['combined'] = 'labels/combined.RData'
    loc_dict['isle'] = 'labels/isle_pairs.csv'
    loc_dict['dsl'] = 'labels/dsl_pairs.csv'
    loc_dict['exp2sl'] = 'labels/exp2sl_pairs.csv'
    loc_dict['lu15'] = 'labels/lu15_pairs.csv'
    loc_dict['train'] = 'labels/train_pairs.csv'
    loc_dict['test'] = 'labels/test_pairs.csv'
    loc_dict['unknown_cancer_BRCA'] = 'labels/unknown_cancer_BRCA_pairs.csv'
    loc_dict['unknown_repair_cancer_BRCA'] = 'labels/unknown_repair_cancer_BRCA_pairs.csv'
    loc_dict['unknown_families_BRCA'] = 'labels/unknown_families_BRCA_pairs.csv'
    loc_dict['unknown_families_extra_BRCA'] = 'labels/unknown_families_extra_BRCA_pairs.csv'
    loc_dict['negative_families_LUAD'] = 'labels/negative_families_LUAD_pairs.csv'
    loc_dict['negative_families_extra_LUAD'] = 'labels/negative_families_extra_LUAD_pairs.csv'


    extra_opts = {}
    #convert_csv2gzip('feature_sets/train_onehot.csv')
    sample='negative_families_extra_LUAD'
    insert_features(feature='seq_1024', sample=sample, loc_dict=loc_dict, extra_opts=extra_opts)
    insert_features(feature='ppi_ec', sample=sample, loc_dict=loc_dict, extra_opts=extra_opts)
    insert_features(feature='crispr_dependency_mut', sample=sample, loc_dict=loc_dict, extra_opts=extra_opts)
    insert_features(feature='tissue_diff_expr', sample=sample, loc_dict=loc_dict, extra_opts=extra_opts)
    extra_opts = {'cutoff': 1.96}
    insert_features(feature='crispr_dependency_expr', sample=sample, loc_dict=loc_dict, extra_opts=extra_opts)
    insert_features(feature='tissue_hcoexp', sample=sample, loc_dict=loc_dict, extra_opts=extra_opts)
    insert_features(feature='tissue_tcoexp', sample=sample, loc_dict=loc_dict, extra_opts=extra_opts)
    insert_features(feature='tissue_surv', sample=sample, loc_dict=loc_dict, extra_opts=extra_opts)

    #add_class_to_unknowns(sample='unknown_repair_cancer_BRCA', features=['tissue'], is_label=False)
    #for dim in ['128', '64', '32']:#['1024','512','256']:
    #    insert_features(feature='seq_'+dim, sample='train', loc_dict=loc_dict, extra_opts=extra_opts)
    #for xx in ['crispr','d2']:
    #    insert_features(feature=xx+'_dependency_cnv', sample='test', loc_dict=loc_dict, extra_opts=extra_opts)
    print('tissue')
    #insert_features(feature='tissue_surv', sample='test', loc_dict=loc_dict, extra_opts=extra_opts)
    #insert_extra_to_tissue(feature='tissue_surv', sample='train', loc_dict=loc_dict, extra_opts=extra_opts)
    #insert_features(feature='onehot', sample='train', loc_dict=loc_dict, extra_opts=extra_opts)
    #convert_nans(sample='train', feature='tissue')

    #insert_features2sets('exp2sl', 'tissue_diff_expr', loc_dict=loc_dict, extra_opts=extra_opts )
    #for dataset in ['lu15', 'exp2sl', 'isle', 'dsl']:
    #    #for feature in ['seq_1024', 'ppi_ec', 'tissue', 'crispr_dependency_mut', 'crispr_dependency_expr_co=1.96']:
    #        #correct_set_features(dataset, feature, loc_dict=loc_dict)
    #correct_set_features('dsl', 'crispr_dependency_mut', loc_dict=loc_dict)
    #ppi_data, ppi_all_data = dfnc.get_PPI_data(loc_dict['PPI_dataset_loc'], cutoff_list=[0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.75,0.8,0.85,0.9,0.95], sources='ec')

    #colm_data = dfnc.load_rdata_file('labels/combined.RData', isRDS=True)
    #for co in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]:
    #    find_missing_data(colm_data, ppi_data[str(co)])

    #find_missing_data(colm_data, ppi_all_data)
    extra_opts = {'dimensions': 28, 'p': 0.25, 'q': 0.25}
    extra_opts = {'dimensions': 28}
    extra_opts = {'cutoffs':[0.5,1.5]}
    extra_opts = {'cutoff':1.96}
    #for i in range(4, 10, 1):
        #dim = str(int(np.power(2, i)))
        #insert_to_colm(choice='seq_'+dim, colm_choice='discoversl', loc_dict=loc_dict, extra_opts=extra_opts)
    #insert_to_colm(choice='rnai_dependency2_any', colm_choice='combined', loc_dict=loc_dict, extra_opts=extra_opts)
    #combine_2_dataset(['feature_sets/combined_all_dependency2_only.RData',
    #                  'feature_sets/combined_all_dependency2_expr_co=1.96_only.RData',
    #                  'feature_sets/combined_all_dependency2_cnv_co=1.96_only.RData'],
    #                  ['mut','expr','cnv'],
    #                  out_loc='combined_all_dependency2_comb_only.RData')
    label='negative_families_extra_LUAD'
    if False:
        for source in ['crispr', 'd2']:
            combine_2_dataset([label+'_'+source+'_dependency_mut.csv.gz',
                               label+'_'+source+'_dependency_cnv.csv.gz',
                               label+'_'+source+'_dependency_expr_co=1.96.csv.gz'],
                              ['mut', 'cnv', 'expr'],
                              out_loc=label+'_'+source+'_dependency_comb.csv.gz')
    if False:
        for alt in ['mut', 'cnv', 'expr_co=1.96', 'comb']:
            combine_2_dataset([label+'_crispr_dependency_'+alt+'.csv.gz',
                               label+'_d2_dependency_'+alt+'.csv.gz'],
                              ['crispr', 'd2'],
                              out_loc=label+'_comb_dependency_'+alt+'.csv.gz')
    if False:
        for source in ['crispr', 'd2', 'comb']:
            combine_2_dataset([label+'_'+source+'_dependency_mut.csv.gz',
                               label+'_'+source+'_dependency_expr_co=1.96.csv.gz'],
                              ['mut', 'cnv', 'expr'],
                              out_loc=label+'_'+source+'_dependency_muex.csv.gz')
    if True:
        combine_2_dataset([label+'_tissue_tcoexp_co=1.96.csv.gz',
                           label+'_tissue_hcoexp_co=1.96.csv.gz'],
                          ['t', 'h'],keys=['gene1', 'gene2', 'cancer', 'class'],
                          out_loc=label+'_tissue_coexp_co=1.96.csv.gz')
    if True:
        combine_2_dataset([label+'_tissue_surv_co=1.96.csv.gz',
                           label+'_tissue_coexp_co=1.96.csv.gz',
                           label+'_tissue_diff_expr.csv.gz'],
                          ['surv', 'co', 'diff'],keys=['gene1', 'gene2', 'cancer', 'class'],
                          out_loc=label+'_tissue.csv.gz')
    #unmappeds = dfnc.load_pickle('unmapped.data')
    # name2string, string2name = get_PPI_genename_maps()
    # get_intersection_colm_ppi(colm_data,ppi_data, ppi_all_data,string2name, 0.0)
    #train_test_indices_loc = 'labels/Exp1.2.train_test_indices.Rdata'
    #train_test_dict = get_train_test_indices_colm(train_test_indices_loc)
    #new_dict = convert_train_test_to_names(colm_data, train_test_dict)
    #train_test_names_loc = config.DATA_DIR / 'labels/Exp1.2.train_test_names.pickle'
    #dfnc.save_pickle(train_test_names_loc, new_dict)

    print()


if __name__ == '__main__':
    main()
