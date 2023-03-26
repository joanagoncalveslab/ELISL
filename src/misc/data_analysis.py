from pandas import DataFrame

import src.data_functions as dfnc
from src import config
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.manifold import TSNE
import src.datasets.tissue as tcga
import collections as coll
from scipy.spatial import distance
import os

cancer_list = ['BRCA', 'CESC', 'COAD', 'KIRC', 'LAML', 'LUAD', 'OV', 'SKCM']
cancer_list = ['BRCA', 'CESC', 'COAD', 'KIRC', 'LAML', 'LUAD', 'OV', 'SKCM']

def rdata_dataset_analysis(dataset='combined', cancers=None):
    if cancers is None:
        cancers = ['BRCA', 'COAD', 'LUAD', 'OV']
    colm_data = dfnc.load_rdata_file('labels/' + dataset + '.RData', isRDS=True)
    strip = colm_data.iloc[:, 0:4]
    analysis = {}
    for cancer1 in cancers:
        analysis[cancer1] = {}
        cancer1_data = strip[strip['cancer_type'] == cancer1]
        analysis[cancer1]['data'] = cancer1_data
        for cancer2 in np.setdiff1d(cancers, cancer1):
            cancer2_data = strip[strip['cancer_type'] == cancer2]
            analysis[cancer1][cancer2] = {}
            common_data = pd.concat([cancer1_data, cancer2_data])
            analysis[cancer1][cancer2]['data'] = common_data
            grouped = common_data.groupby(['gene1', 'gene2']).agg({
                'cancer_type': {'unique_size': lambda x: len(set(list(x)))},
                'SL': ['min', 'max', 'mean', 'count']})
            analysis[cancer1][cancer2]['grouped'] = grouped
            count1 = grouped[(grouped['cancer_type']['unique_size'] != 1) & (grouped['SL']['mean'] == 1)].shape[0]
            count = grouped[grouped['cancer_type']['unique_size'] != 1].shape[0]
            count0 = grouped[(grouped['cancer_type']['unique_size'] != 1) & (grouped['SL']['mean'] == 0)].shape[0]
            count01 = count-count1-count0
            analysis[cancer1][cancer2]['counts'] = [count1, count01, count0]
    return analysis


def label_dataset_analysis(dataset_list=None):
    if dataset_list is None:
        dataset_list = []
    loc_dict = dfnc.get_loc_dict()
    datasets = {}
    for dataset in dataset_list:
        datasets[dataset.lower()] = {}
        if 'isle' in dataset.lower():
            isle = dfnc.get_ISLE_training_set(loc_dict['isle_dataset_loc'])
            isle = isle.astype({"class": int})
            datasets[dataset.lower()]['data'] = isle
        elif 'discoversl' in dataset.lower():
            dsl = dfnc.get_DiscoverSL_training_set(loc_dict['discoverSL_dataset_loc'])
            dsl = dsl.astype({"class": int})
            datasets[dataset.lower()]['data'] = dsl
        elif 'synleth' in dataset.lower():
            synleth = dfnc.get_SynLethDB_data(folder='database', organism='Human')
            synleth = synleth.astype({"class": int})
            datasets[dataset.lower()]['data'] = synleth
        elif 'exp2sl' in dataset.lower():
            exp2sl = dfnc.get_exp2sl_data(folder='database/exp2sl')
            exp2sl = exp2sl.astype({"class": int})
            datasets[dataset.lower()]['data'] = exp2sl
        elif 'lu15' in dataset.lower():
            lu15 = dfnc.get_lu15_data(loc_dict['lu15_dataset_loc'])
            lu15 = lu15.astype({"class": int})
            datasets[dataset.lower()]['data'] = lu15

    for name, ds_dict in datasets.items():
        data = ds_dict['data']
        ds_dict['cancer_types'] = np.unique(data['cancer'].values)
        grouped_pairs = data.groupby(['gene1', 'gene2'])['cancer', 'class'].agg(
            {'cancer': {'list': lambda x: list(set(list(x))), 'unique': lambda y: len(set(list(y)))},
             'class': ['max', 'min', 'mean', 'count']})
        ds_dict['grouped_genes'] = grouped_pairs
        ds_dict['disagree'] = grouped_pairs[grouped_pairs['max'] != grouped_pairs['min']]
        ds_dict['multiple_cancers'] = grouped_pairs[grouped_pairs['unique'] > 1]
        ds_dict['cancer_stats'] = {}
        for cancer1 in ds_dict['cancer_types']:
            ds_dict['cancer_stats'][cancer1]={'count': sum(np.array([cancer1 in l for l in grouped_pairs['list']]))}
            for cancer2 in ds_dict['cancer_types']:
                two_cancer_pairs = grouped_pairs[np.array([cancer1 in l for l in grouped_pairs['list']]) &
                                                    np.array([cancer2 in l for l in grouped_pairs['list']])]
                ds_dict['cancer_stats'][cancer1][cancer2] = two_cancer_pairs

    return datasets


def detailed_label_dataset_analysis():
    label_datasets = ['exp2sl', 'isle', 'discoversl', 'lu15']  # ,'synlethdb'
    analysis = label_dataset_analysis(label_datasets)
    for data_name, data_dict in analysis.items():
        print(f'Cleaning {data_name}')
        df_cancer = data_dict['data'].copy()
        df_cancer = dfnc.remove_duplicates_inner(df_cancer)
        df_cancer.insert(loc=0, column='pair_name', value=df_cancer[['gene1', 'gene2']].agg('|'.join, axis=1))
        df_cancer = df_cancer.reset_index()
        df_cancer = df_cancer.drop(columns=['gene1', 'gene2', 'index'])
        analysis[data_name]['clean_data'] = df_cancer
    lu15 = analysis['lu15']['clean_data']
    lu15['from'] = 'lu15'
    isle = analysis['isle']['clean_data']
    isle['from'] = 'isle'
    dsl = analysis['discoversl']['clean_data']
    dsl['from'] = 'dsl'
    exp2sl = analysis['exp2sl']['clean_data']
    exp2sl['from'] = 'exp2sl'
    colm = dfnc.load_rdata_file('labels/combined.RData', isRDS=True)
    colm = dfnc.strip_dataframe(colm, end_col=4)

    gene_switch_ids = (colm['gene1'] > colm['gene2'])
    colm.loc[gene_switch_ids, ['gene1', 'gene2']] = colm.loc[gene_switch_ids, ['gene2', 'gene1']].values
    colm = colm.sort_values(by=['cancer_type', 'gene1', 'gene2'])
    df_cancer = colm.copy()
    df_cancer.insert(loc=0, column='pair_name', value=df_cancer[['gene1', 'gene2']].agg('|'.join, axis=1))
    df_cancer = df_cancer.reset_index()
    colm = df_cancer.drop(columns=['gene1', 'gene2', 'index'])
    colm['from'] = 'colm'
    colm.columns = ['pair_name', 'cancer', 'class', 'from']

    all_data = pd.concat([colm, lu15, isle, dsl, exp2sl])

    grouped_all = all_data.groupby(['pair_name', 'cancer'])['class', 'from'].agg(
        {'from': {'list': lambda x: list(set(list(x))), 'unique': lambda y: len(set(list(y)))},
         'class': ['max', 'min', 'mean', 'count']}).reset_index()

    print(analysis)
    colm_coad = colm[colm['cancer_type'] == 'COAD']
    lu15_coad = lu15[lu15['cancer'] == 'COAD']
    exp2sl_coad = exp2sl[exp2sl['cancer'] == 'COAD']
    np.setdiff1d(lu15_coad[lu15_coad['class'] == 1]['pair_name'].values,
                 colm_coad[colm_coad['SL'] == 1]['pair_name'].values)

    colm_coad['from'] = 'colm'
    lu15_coad['from'] = 'lu15'
    exp2sl_coad['from'] = 'exp2sl'
    all_coad = pd.concat([colm_coad, lu15_coad, exp2sl_coad])
    all_coad.columns = ['pair_name', 'cancer', 'class', 'from']
    all_coad.groupby('pair_name')['class', 'from'].agg(
        {'from': {'list': lambda x: list(set(list(x))), 'unique': lambda y: len(set(list(y)))},
         'class': ['max', 'min', 'mean', 'count']})

    print(analysis)


def visualize_cancer_specific_labels(data, cancer, save_loc=None):
    plt.clf()
    gene_switch_ids = (data['gene1'] > data['gene2'])
    data.loc[gene_switch_ids, ['gene1', 'gene2']] = data.loc[gene_switch_ids, ['gene2', 'gene1']].values
    data = data.sort_values(by=['cancer_type', 'gene1', 'gene2'])
    data.columns = ['gene1', 'gene2', 'cancer', 'class']
    cancer_data = data[data['cancer'] == cancer]
    unique_genes = np.sort(np.unique(np.concatenate([cancer_data['gene1'].unique(), cancer_data['gene1'].unique()])))
    color_dict = {0: 'blue', 1: 'red'}
    colors = [color_dict[i] for i in cancer_data['class'].values]
    cancer_plt = plt.scatter(cancer_data['gene1'], cancer_data['gene2'], s=1, c=colors)
    plt.xticks(rotation=90, fontsize=5)
    plt.yticks(fontsize=5)

    red_patch = mpatches.Patch(color='red', label='SL')
    blue_patch = mpatches.Patch(color='blue', label='nonSL')
    plt.legend(handles=[red_patch, blue_patch])
    if save_loc is not None:
        save_loc = config.DATA_DIR / save_loc
        #plt.savefig(save_loc)
    plt.show()

    #plt.xticks(np.arange(len(unique_genes)), unique_genes)
    #plt.yticks(np.arange(len(unique_genes)), unique_genes)


def visualize_cancer_specific_labels2(data, cancer, data_name, save_loc=None):
    plt.clf()
    gene_switch_ids = (data['gene1'] > data['gene2'])
    data.loc[gene_switch_ids, ['gene1', 'gene2']] = data.loc[gene_switch_ids, ['gene2', 'gene1']].values
    if data_name in ['combined', 'discoversl', 'isle']:
        data = data.sort_values(by=['cancer_type', 'gene1', 'gene2'])
    if data_name in ['train_pairs', 'test_pairs']:
        data = data.sort_values(by=['cancer', 'gene1', 'gene2'])
        data = data[['gene1', 'gene2', 'cancer', 'class']]
    data.columns = ['gene1', 'gene2', 'cancer', 'class']
    cancer_data = data[data['cancer'] == cancer]
    unique_genes = np.sort(np.unique(np.concatenate([cancer_data['gene1'].unique(), cancer_data['gene2'].unique()])))
    df_unique_genes: DataFrame = pd.DataFrame(0,index=unique_genes, columns=['count', 'freq(%)', 'pos_count', 'neg_count'])
    for unique_gene in unique_genes:
        gene_data = cancer_data[(cancer_data['gene1'] == unique_gene) | (cancer_data['gene2'] == unique_gene)]
        occ = gene_data.shape[0]
        pos_occ = sum(gene_data['class']==1)
        neg_occ = occ-pos_occ
        df_unique_genes.loc[unique_gene]['count'] = occ
        df_unique_genes.loc[unique_gene]['freq(%)'] = 100*float(occ)/cancer_data.shape[0]
        df_unique_genes.loc[unique_gene]['pos_count'] = pos_occ
        df_unique_genes.loc[unique_gene]['neg_count'] = neg_occ
    df_unique_genes = df_unique_genes.sort_values(by=['freq(%)'], ascending=False)
    print(f'{data_name}\t{cancer} ==> {cancer_data.shape[0]} pairs.\n{df_unique_genes.head()}\n')

    cancer_data_conv = cancer_data.copy()
    cancer_data_conv.loc[:,['gene1', 'gene2']] = cancer_data_conv.loc[:,['gene2', 'gene1']].values
    cancer_data = pd.concat([cancer_data, cancer_data_conv])
    ids = list(range(len(unique_genes)))
    gene2id = dict(zip(unique_genes, ids))
    color_dict = {0: 'blue', 1: 'red'}
    colors = [color_dict[i] for i in cancer_data['class'].values]
    cancer_plt = plt.scatter(cancer_data['gene1'].map(gene2id).values, cancer_data['gene2'].map(gene2id).values, s=1, c=colors)
    unique_x = unique_genes.copy()
    unique_y = unique_genes.copy()
    for idx_gene, unique_gene in enumerate(unique_genes):
        if unique_gene not in cancer_data['gene1'].values:
            unique_x[idx_gene] = ''
        if unique_gene not in cancer_data['gene2'].values:
            unique_y[idx_gene] = ''
    plt.xticks(ids, unique_x, rotation=90, fontsize=5)
    plt.yticks(ids, unique_y, fontsize=5)

    red_patch = mpatches.Patch(color='red', label='SL')
    blue_patch = mpatches.Patch(color='blue', label='nonSL')
    plt.legend(handles=[red_patch, blue_patch])
    if save_loc is not None:
        save_loc = config.DATA_DIR / save_loc
        plt.savefig(save_loc, dpi=300, bbox_inches='tight')
    #plt.show()

    #plt.xticks(np.arange(len(unique_genes)), unique_genes)
    #plt.yticks(np.arange(len(unique_genes)), unique_genes)


def visualize_data_tsne(df, cancer, data_name, perp, save_loc=None):
    plt.clf()
    df_cancer = df.copy()
    df_cancer.insert(loc=0, column='pair_name', value=df[['gene1', 'gene2']].agg('|'.join, axis=1))
    if cancer == None:
        df_cancer = df_cancer.reset_index()
    else:
        df_cancer = df_cancer[df_cancer['cancer_type'] == cancer].reset_index()
    df_cancer = df_cancer.drop(columns=['gene1', 'gene2', 'cancer_type', 'index'])
    pairs = df_cancer['pair_name'].values
    labels = df_cancer['SL'].values
    features = df_cancer.drop(columns=['pair_name', 'SL']).values

    sl_ind = labels == 1
    nonsl_ind = labels == 0
    tsne_features = TSNE(n_components=2, perplexity=perp).fit_transform(features)
    plt.scatter(tsne_features[sl_ind,0], tsne_features[sl_ind,1], c='red', label='SL')
    plt.scatter(tsne_features[nonsl_ind,0], tsne_features[nonsl_ind,1], c='blue', label='nonSL')

    plt.legend()
    plt.title(f'{data_name} {cancer}')
    plt.xticks([])
    plt.yticks([])
    if save_loc is not None:
        save_loc = config.DATA_DIR / save_loc
        plt.savefig(save_loc, dpi=300, bbox_inches='tight')
    #plt.show()


def visualize_cancer_specific_labels2_exp():
    cancers = ['BRCA', 'CESC', 'COAD', 'KIRC', 'LAML', 'LUAD', 'OV', 'SKCM']
    data_names = ['combined','isle','discoversl']
    data_names=['train_pairs']
    #cancers = ['OV']
    #data_names = ['combined']
    for data_name in data_names:
        for cancer in cancers:
            #print(f'data: {data_name}, cancer: {cancer} started.')
            #data = dfnc.load_rdata_file('labels/'+data_name+'.RData', isRDS=True)
            csv_loc = config.DATA_DIR / ('labels/'+data_name+'.csv')
            data = pd.read_csv(csv_loc)
            #data = dfnc.strip_dataframe(data, end_col=4)
            visualize_cancer_specific_labels2(data, cancer, data_name, 'labels/visualize/'+data_name+'_'+cancer+'.png')
            #visualize_data_tsne()
            #print(f'data: {data_name}, cancer: {cancer} ended.')


def visualize_feature_set(loc='feature_sets/train_tissue_hcoexp_co=1.96.csv.gz'):
    loc = config.DATA_DIR / loc
    data = pd.read_csv(loc)
    cancer_inds = data['cancer']
    data = data.fillna({'gtex_coexp':0, 'gtex_coexp_p':1})
    data = data.drop(columns=['gene1', 'gene2','class', 'cancer']).values

    tsne_features = TSNE(n_components=2, perplexity=5).fit_transform(data)
    for cancer in ['BRCA', 'CESC', 'COAD', 'KIRC', 'LAML', 'LUAD', 'OV', 'SKCM']:
        chosen_ind = cancer_inds == cancer
        plt.scatter(tsne_features[chosen_ind,0], tsne_features[chosen_ind,1], label=cancer, s=1)

    plt.legend()
    #plt.savefig('gtex_pats.png', bbox_inches="tight")
    plt.show()


def tsne_related():
    cancers = ['BRCA', 'COAD', 'LUAD', 'OV']
    # cancers = ['BRCA','LUAD']
    data_names = ['combined']  # ,
    data_names = ['combined_crispr_dependency2_only', 'combined_rnai_dependency2_only',
                  'combined_ppi_ec_only']
    data_names = ['discoversl_seq_only', 'discoversl_crispr_dependency2_only',
                  'discoversl_rnai_dependency2_only', 'discoversl_ppi_ec_only']
    data_names = ['isle_seq_only', 'isle_crispr_dependency2_only', 'isle_rnai_dependency2_only',
                  'isle_ppi_ec_only']
    data_names = ['isle', 'discoversl']  # ,
    data_names = ['combined_rnai_dependency2_only', ]

    # cancers = ['OV']
    # data_names = ['combined']
    for data_name in data_names:
        for cancer in cancers:
            if data_name in ['combined', 'discoversl', 'isle']:
                data = dfnc.load_rdata_file('labels/' + data_name + '.RData', isRDS=True)
                data = data.fillna(0)
                visualize_data_tsne(data, cancer, data_name, 30,
                                    'labels/visualize/tsne_' + data_name + '_' + cancer + '.png')
            else:
                data = dfnc.load_rdata_file('feature_sets/' + data_name + '.RData', isRDS=True)
                data = data.fillna(0)
                visualize_data_tsne(data, cancer, data_name, 30,
                                    'feature_sets/visualize/tsne_' + data_name + '_' + cancer + '.png')


def analyze_cluster_distances(source='gtex', dist='euclidean', method='avg'):
    out_res = config.DATA_DIR / 'Analysis' / (source + '_' + str(dist) + '_' + method + '_dist.pkl')
    if os.path.isfile(out_res):
        return dfnc.load_pickle(out_res)
    distances=[dist]
    if type(dist) != str:
        distances = dist
    loc_dict = tcga.get_locs()
    # cancer_dict = cgt.get_golden_truth_by_cancer()
    expr_dict = coll.OrderedDict()
    #dist_m_dict = coll.OrderedDict()
    dist_dict = coll.OrderedDict()
    cancer_ind = np.array([])
    last_ind = 0
    for dist in distances:
        tmp_res = config.DATA_DIR / 'Analysis' / (source + '_' + str(dist) + '_' + method + '_dist.pkl')
        if os.path.isfile(tmp_res):
            dist_dict[dist] = dfnc.load_pickle(tmp_res)
        else:
            dist_dict[dist] = coll.OrderedDict()
        for cancer in cancer_list:
            if cancer not in dist_dict[dist].keys():
                dist_dict[dist].update({cancer:coll.OrderedDict()})
    for cancer in cancer_list:
        #dist_dict[dist][cancer]==coll.OrderedDict()
        #dist_m_dict[dist][cancer]==coll.OrderedDict()
        if source == 'gtex':
            expr_dict[cancer] = tcga.load_cancer_gtex_expr(loc_dict[cancer]['gtex_expression']).T
        else:
            expr_dict[cancer] = tcga.load_all_expr(loc_dict[cancer]['expression']).T
        for c2_k, c2_d in expr_dict.items():
            if cancer!=c2_k:
                for dist in distances:
                    if c2_k not in dist_dict[dist][cancer].keys():
                        dist_m = distance.cdist(expr_dict[cancer], c2_d, dist)
                        dist_dict[dist][cancer][c2_k] = dist_m.mean()
                    #dist_m_dict[dist][cancer][c2_k] = dist_m

    for dist in distances:
        out_res = config.DATA_DIR / 'Analysis' / (source+'_'+dist+'_'+method+'_dist.pkl')
        dfnc.save_pickle(out_res,dist_dict[dist])
    if len(distances)==1:
        return dist_dict

def plot_heatmap_dist(gtex_dist, tissue_dist, out):
    dist_df = pd.DataFrame(0, columns=cancer_list, index=cancer_list)
    for g_c1, g_d1 in gtex_dist.items():
        for g_c2, dist in g_d1.items():
            dist_df.loc[g_c1,g_c2]=dist
    for g_c1, g_d1 in tissue_dist.items():
        for g_c2, dist in g_d1.items():
            dist_df.loc[g_c2,g_c1]=dist

    print()
    plt.clf()
    fig, ax = plt.subplots()
    tmp = plt.pcolor(dist_df)
    fig.colorbar(tmp, ax=ax)
    plt.yticks(np.arange(0.5, len(dist_df.index), 1), dist_df.index)
    plt.xticks(np.arange(0.5, len(dist_df.columns), 1), dist_df.columns)
    plt.ylabel('GTEX')
    plt.xlabel('TCGA')
    out_pdf = config.DATA_DIR / 'Analysis' / (out+'.pdf')
    out_png = config.DATA_DIR / 'Analysis' / (out+'.png')
    plt.savefig(out_pdf, bbox_inches="tight")
    plt.savefig(out_png, bbox_inches="tight")
    plt.show()

def main() -> object:
    detailed_label_dataset_analysis()
    #visualize_cancer_specific_labels2_exp()
    #visualize_feature_set()
    dist = 'euclidean'
    dists=['euclidean', 'seuclidean', 'sqeuclidean', 'cosine', 'correlation']
    #analyze_cluster_distances(source='gtex', dist=dists)
    #analyze_cluster_distances(source='tissue', dist=dists)
    #for dist in dists:
    #    gtex_dist_dict = analyze_cluster_distances(source='gtex', dist=dist)
    #    tissue_dist_dict = analyze_cluster_distances(source='tissue', dist=dist)
    #    plot_heatmap_dist(gtex_dist_dict, tissue_dist_dict,'dist_'+dist)


if __name__ == '__main__':
    main()