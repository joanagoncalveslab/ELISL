import os
import sys
path2this = os.path.dirname(os.path.abspath(__file__)).split('/')
for i, folder in enumerate(path2this):
    if folder.lower()=='elisl':
        project_path = '/'.join(path2this[:i+1])
sys.path.insert(0,project_path)
from src import config
import pandas as pd
import numpy as np
from src import result_analyzer as ra
import matplotlib.pyplot as plt
import src.datasets.tissue as tcga
import src.create_gold_truths as cgt
import matplotlib.patches as mpatches
from sklearn.manifold import TSNE
import collections as coll
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering
from scipy import stats
import seaborn as sns
import warnings
from collections import OrderedDict
warnings.filterwarnings("ignore")

cancer_list = ['BRCA', 'CESC', 'COAD', 'KIRC', 'LAML', 'LUAD', 'OV', 'SKCM']
#cancer_list = ['BRCA', 'COAD', 'LUAD', 'OV']

cancer_baselines = {
    'trus': {
        'theory': {
            'AUROC': {'BRCA': 0.5, 'CESC':0.5, 'COAD': 0.5, 'KIRC':0.5, 'LAML': 0.5, 'LUAD': .5, 'SKCM':0.5, 'OV': 0.5},
            'AUPRC': {'BRCA': 0.58, 'CESC':0., 'COAD': 0.02, 'KIRC':0., 'LAML': 0.06, 'LUAD': .10, 'SKCM':0., 'OV': 0.32}},
        'random': {
            'AUROC': {'BRCA': 0.54, 'CESC':0., 'COAD': 0.49, 'KIRC':0., 'LAML': 0.60, 'LUAD': .5, 'SKCM':0., 'OV': 0.55},
            'AUPRC': {'BRCA': 0.61, 'CESC':0., 'COAD': 0.02, 'KIRC':0., 'LAML': 0.15, 'LUAD': .10, 'SKCM':0., 'OV': 0.41}},
        'onehot': {
            'AUROC': {'BRCA': 0.92, 'CESC':0., 'COAD': 0.59, 'KIRC':0., 'LAML': 0.60, 'LUAD': .78, 'SKCM':0., 'OV': 0.61},
            'AUPRC': {'BRCA': 0.94, 'CESC':0., 'COAD': 0.03, 'KIRC':0., 'LAML': 0.14, 'LUAD': .36, 'SKCM':0., 'OV': 0.37}}},
    'trteus': {
        'theory': {
            'AUROC': {'BRCA': 0.5, 'CESC':0.5, 'COAD': 0.5, 'KIRC':0.5, 'LAML': 0.5, 'LUAD': .5, 'SKCM':0.5, 'OV': 0.5},
            'AUPRC': {'BRCA': 0.5, 'CESC':0.5, 'COAD': 0.5, 'KIRC':0.5, 'LAML': 0.5, 'LUAD': .5, 'SKCM':0.5, 'OV': 0.5}},
        'random': {
            'AUROC': {'BRCA': 0.54, 'CESC':0.53, 'COAD': 0.50, 'KIRC':0.47, 'LAML': 0.50, 'LUAD': 0.60, 'SKCM':0.55, 'OV': 0.60},
            'AUPRC': {'BRCA': 0.53, 'CESC':0.54, 'COAD': 0.50, 'KIRC':0.55, 'LAML': 0.51, 'LUAD': 0.58, 'SKCM':0.55, 'OV': 0.61}},
        'onehot': {
            'AUROC': {'BRCA': 0.91, 'CESC':0., 'COAD': 0.61, 'KIRC':0., 'LAML': 0., 'LUAD': .77, 'SKCM':0., 'OV': 0.62},
            'AUPRC': {'BRCA': 0.92, 'CESC':0., 'COAD': 0.56, 'KIRC':0., 'LAML': 0., 'LUAD': .75, 'SKCM':0., 'OV': 0.58}}}
}
colm_res = {'BRCA':{'AUROC_m':0.86, 'AUROC_std':0.01,'AUPRC_m':0.89, 'AUPRC_std':0.01},
            'COAD':{'AUROC_m':0.63, 'AUROC_std':0.02,'AUPRC_m':0.63, 'AUPRC_std':0.02},
            'LUAD':{'AUROC_m':0.87, 'AUROC_std':0.02,'AUPRC_m':0.87, 'AUPRC_std':0.02},
            'OV':{'AUROC_m':0.59, 'AUROC_std':0.03,'AUPRC_m':0.58, 'AUPRC_std':0.04}}

pca_gCMF_res = {'BRCA':{'AUROC_m':0.92, 'AUROC_std':0.01,'AUPRC_m':0.92, 'AUPRC_std':0.01},
            'COAD':{'AUROC_m':0.54, 'AUROC_std':0.03,'AUPRC_m':0.56, 'AUPRC_std':0.03},
            'LUAD':{'AUROC_m':0.87, 'AUROC_std':0.03,'AUPRC_m':0.81, 'AUPRC_std':0.06},
            'OV':{'AUROC_m':0.94, 'AUROC_std':0.02,'AUPRC_m':0.92, 'AUPRC_std':0.04}}

best_single_res = {'BRCA':{'AUROC_m':0.87, 'AUROC_std':0.02,'AUPRC_m':0.87, 'AUPRC_std':0.02},
            'COAD':{'AUROC_m':0.64, 'AUROC_std':0.01,'AUPRC_m':0.63, 'AUPRC_std':0.01},
            'LUAD':{'AUROC_m':0.83, 'AUROC_std':0.03,'AUPRC_m':0.84, 'AUPRC_std':0.02},
            'OV':{'AUROC_m':0.80, 'AUROC_std':0.03,'AUPRC_m':0.81, 'AUPRC_std':0.03}}


onehot_res = {'trus':{'BRCA':{'AUROC_m':0.92, 'AUROC_std':0.01,'AUPRC_m':0.94, 'AUPRC_std':0.01},
                        'CESC':{'AUROC_m':0., 'AUROC_std':0.0,'AUPRC_m':0., 'AUPRC_std':0.0},
                        'KIRC':{'AUROC_m':0., 'AUROC_std':0.0,'AUPRC_m':0., 'AUPRC_std':0.0},
                        'COAD':{'AUROC_m':0.59, 'AUROC_std':0.02,'AUPRC_m':0.03, 'AUPRC_std':0.00},
                        'LAML':{'AUROC_m':0., 'AUROC_std':0.0,'AUPRC_m':0., 'AUPRC_std':0.0},
                        'LUAD':{'AUROC_m':0.78, 'AUROC_std':0.03,'AUPRC_m':0.36, 'AUPRC_std':0.03},
                        'OV':{'AUROC_m':0.61, 'AUROC_std':0.02,'AUPRC_m':0.37, 'AUPRC_std':0.01},
                        'SKCM':{'AUROC_m':0., 'AUROC_std':0.0,'AUPRC_m':0., 'AUPRC_std':0.0}},
              'trteus': {'BRCA': {'AUROC_m': 0.91, 'AUROC_std': 0.01, 'AUPRC_m': 0.92, 'AUPRC_std': 0.01},
                        'CESC':{'AUROC_m':0.5, 'AUROC_std':0.0,'AUPRC_m':0.5, 'AUPRC_std':0.0},
                        'KIRC':{'AUROC_m':0.5, 'AUROC_std':0.0,'AUPRC_m':0.5, 'AUPRC_std':0.0},
                         'COAD': {'AUROC_m': 0.61, 'AUROC_std': 0.01, 'AUPRC_m': 0.56, 'AUPRC_std': 0.01},
                        'LAML':{'AUROC_m':0.60, 'AUROC_std':0.02,'AUPRC_m':0.60, 'AUPRC_std':0.02},
                         'LUAD': {'AUROC_m': 0.77, 'AUROC_std': 0.03, 'AUPRC_m': 0.75, 'AUPRC_std': 0.03},
                         'OV': {'AUROC_m': 0.62, 'AUROC_std': 0.02, 'AUPRC_m': 0.58, 'AUPRC_std': 0.02},
                        'SKCM':{'AUROC_m':0.5, 'AUROC_std':0.0,'AUPRC_m':0.5, 'AUPRC_std':0.0}}
              }


def plot_dendrogram(model, **kwargs):
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack([model.children_, model.distances_,
                                      counts]).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)
    return linkage_matrix


def plot_res(loc= 'results/ELRRF/single_cancer_test.csv', score= 'AUROC', selections=None, res_id=14,
             us='', out_name='temp.png'):
    if selections is None:
        selections = {'balance_strat': ['undersample_train_test'],
                      'comb_type': ['type2'], 'process': [True],
                      'grid_search': ['True']}
    out_loc = config.ROOT_DIR / ('/'.join(loc.split('/')[:-1])) / 'images'/ out_name
    config.ensure_dir(out_loc)
    plt.rcdefaults()
    fig, ax = plt.subplots()
    for cancer in cancer_list:
        selections.update({'cancer':[cancer]})
        res = ra.get_result_loc(loc=loc, cancer=cancer, res_names=ra.get_res_name(13), selections=selections,
                             out_cols=[score+'_m', score+'_std'], bestMC=False, chosen_th=None)

        x_axis = ['1024', '512', '256', '128', '64', '32']
        means = []
        std = []

        tmp_plt = plt.errorbar(x_axis, res[score+'_m'].astype('float').values, yerr=res[score+'_std'].astype('float').values, label=cancer, capsize=3)
        #tmp_plt[-1][0].set_linestyle('--')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left',)
    if 'AUROC' in score or ('AUPRC' in score and us=='trteus'):
        plt.ylim((0.45,1))
    ax.set_xlabel('Sequence Dimension')
    ax.set_ylabel(score)
    #plt.title(score+'\tSeq+PPI+Crispr_Dep_Mut+Crispr_Dep_Exp+Tissue')
    plt.savefig(out_loc, bbox_inches="tight")
    plt.show()


def plot_single_for_cancer(loc= 'results/elrrf/single_cancer_validation.csv', score= 'AUROC', selections=None, res_id=14,
                           us='', out_name='temp.png'):
    if selections is None:
        selections = {'balance_strat': ['undersample_train'],
                      'comb_type': ['type2'], 'process': [True],
                      'grid_search': ['False']}
    out_loc = config.ROOT_DIR / ('/'.join(loc.split('/')[:-1])) / 'images'/ out_name
    config.ensure_dir(out_loc)
    plt.rcdefaults()
    fig, ax = plt.subplots()
    cancer_list=['BRCA', 'COAD', 'LUAD', 'OV']

    width = 0.15  # the width of the bars
    x_axis = []
    for i, cancer in enumerate(cancer_list):
        selections.update({'cancer':[cancer]})
        res = ra.get_result_loc(loc=loc, cancer=cancer, res_names=ra.get_res_name(res_id), selections=selections,
                             out_cols=[score+'_m', score+'_std'], bestMC=True, chosen_th=None)

        labels = ra.get_res_name(res_id)
        if i ==0:
            x_axis.insert(i, np.arange(len(labels)))  # the label locations
        else:
            x_axis.insert(i, [x + width for x in x_axis[i-1]])
        means = []
        std = []
        tmp_plt = ax.bar(x_axis[i], res[score+'_m'].astype('float').values,width, yerr=res[score+'_std'].astype('float').values, label=cancer, capsize=3)
        #tmp_plt[-1][0].set_linestyle('--')




    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left',)
    if 'AUROC' in score or ('AUPRC' in score and us=='trteus'):
        plt.ylim((0.45,1))
    plt.xlabel('group', fontweight='bold')
    #plt.xticks([r + width for r in range(len(labels))], ['A', 'B', 'C', 'D', 'E'])
    ax.set_xlabel('Sequence Dimension')
    ax.set_ylabel(score)
    #plt.title(score+'\tSeq+PPI+Crispr_Dep_Mut+Crispr_Dep_Exp+Tissue')
    #plt.savefig(out_loc, bbox_inches="tight")
    plt.show()


def plot_for_cancer2(loc= 'results/elrrf/single_cancer_validation.csv', score= 'AUROC', selections=None, res_id=2,
                            baselines = {}, us='', out_name='temp.png'):
    if selections is None:
        selections = {'balance_strat': ['undersample_train'],
                      'comb_type': ['type2'], 'process': [True],
                      'grid_search': ['False']}
    out_loc = config.ROOT_DIR / ('/'.join(loc.split('/')[:-1])) / 'images'/ out_name
    config.ensure_dir(out_loc)
    plt.rcdefaults()
    fig, ax = plt.subplots(figsize=(12,8))

    cancer_list=['BRCA', 'COAD', 'LAML', 'LUAD', 'OV']

    width = 0.12  # the width of the bars
    x_axis = []
    cancer_dict=coll.OrderedDict()

    for cancer in cancer_list:
        selections.update({'cancer': [cancer]})
        res = ra.get_result_loc(loc=loc, cancer=cancer, res_names=ra.get_res_name(res_id), selections=selections,
                                out_cols=[score + '_m', score + '_std'], bestMC=True, chosen_th=None)
        cancer_dict[cancer] = res
    label_dict= {'seq_1024': 'Sequence', 'ppi_ec':'PPI', 'crispr_dependency_mut':'CRISPR_mut', 'd2_dependency_mut':'RNAi_mut',
                 'crispr_dependency_expr': 'CRISPR_expr', 'd2_dependency_expr': 'RNAi_expr', 'tissue': 'Tissue',
                 'seq_1024|ppi_ec|crispr_dependency_mut|crispr_dependency_expr|tissue':'Our Model'
                 }
    labels = list(cancer_dict.keys())
    x_axiss = np.arange(len(labels))
    for i, res_name in enumerate(ra.get_res_name(res_id)):
        x_axis.insert(i, [x + i*width for x in x_axiss])
        means = [float(res.loc[res_name, score+'_m']) for cancer, res in cancer_dict.items()]
        stds = [float(res.loc[res_name, score+'_std']) for cancer, res in cancer_dict.items()]
        tmp_plt = ax.bar(x_axis[i], means,width, yerr=stds, label=label_dict[res_name], capsize=3)
        #tmp_plt[-1][0].set_linestyle('--')

    if 'AUROC' in score or ('AUPRC' in score and us=='trteus'):
        plt.ylim((0.45,1))
    for i, cancer in enumerate(labels):
        if i==0:
            ax.hlines(baselines[us]['theory'][score][cancer], xmin=-width/2+i, xmax=i+width*len(ra.get_res_name(res_id))-width/2,
                    linestyles='dashed', color='black', label='Theoretical Random')
            ax.hlines(baselines[us]['random'][score][cancer], xmin=-width/2+i, xmax=i+width*len(ra.get_res_name(res_id))-width/2,
                    linestyles='dotted', color='black', label='Computational Random')
        else:
            ax.hlines(baselines[us]['theory'][score][cancer], xmin=-width/2+i, xmax=i+width*len(ra.get_res_name(res_id))-width/2,
                    linestyles='dashed', color='black')
            ax.hlines(baselines[us]['random'][score][cancer], xmin=-width/2+i, xmax=i+width*len(ra.get_res_name(res_id))-width/2,
                    linestyles='dotted', color='black')

    plt.legend(loc='upper right',)
    plt.xlim((-0.1,5))
    plt.xlabel('Cancer Type', fontweight='bold')
    plt.ylabel(score, fontweight='bold')
    plt.xticks([r + len(labels)*(width)/2+width for r in range(len(labels))], cancer_list)
    #ax.set_xlabel('Sequence Dimension')
    ax.set_ylabel(score)
    #plt.title(score+'\tSeq+PPI+Crispr_Dep_Mut+Crispr_Dep_Exp+Tissue')
    #plt.savefig(out_loc, bbox_inches="tight")
    plt.show()

def plot_single_for_cancer2(loc= 'results/elrrf/single_cancer_validation.csv', score= 'AUROC', selections=None, res_id=14,
                            baselines = {}, us='', out_name='temp.png'):
    if selections is None:
        selections = {'balance_strat': ['undersample_train'],
                      'comb_type': ['type2'], 'process': [True],
                      'grid_search': ['False']}
    out_loc = config.ROOT_DIR / ('/'.join(loc.split('/')[:-1])) / 'images'/ out_name
    config.ensure_dir(out_loc)
    plt.rcdefaults()
    fig, ax = plt.subplots(figsize=(12,8))

    cancer_list=['BRCA', 'CESC', 'COAD', 'KIRC', 'LAML', 'LUAD', 'OV', 'SKCM']

    width = 0.45  # the width of the bars
    x_axis = []
    cancer_dict=coll.OrderedDict()

    for cancer in cancer_list:
        selections.update({'cancer': [cancer]})
        res = ra.get_result_loc(loc=loc, cancer=cancer, res_names=ra.get_res_name(res_id), selections=selections,
                                out_cols=[score + '_m', score + '_std'], bestMC=True, chosen_th=None)
        cancer_dict[cancer] = res

    labels = list(cancer_dict.keys())
    x_axiss = np.arange(len(labels))
    means = [float(res.loc[ra.get_res_name(res_id)[0], score+'_m']) for cancer, res in cancer_dict.items()]
    stds = [float(res.loc[ra.get_res_name(res_id)[0], score+'_std']) for cancer, res in cancer_dict.items()]
    tmp_plt = ax.bar(labels, means,width, yerr=stds, label='Our Model', capsize=3)
    #tmp_plt[-1][0].set_linestyle('--')
    if 'AUROC' in score or ('AUPRC' in score and us=='trteus'):
        plt.ylim((0.45,1))
    for i, cancer in enumerate(labels):
        if i==0:
            ax.hlines(baselines[us]['theory'][score][cancer], xmin=-width/2+i, xmax=i+width*len(ra.get_res_name(res_id))-width/2,
                    linestyles='dashed', color='black', label='Theoretical Random')
            ax.hlines(baselines[us]['random'][score][cancer], xmin=-width/2+i, xmax=i+width*len(ra.get_res_name(res_id))-width/2,
                    linestyles='dotted', color='black', label='Computational Random')
        else:
            ax.hlines(baselines[us]['theory'][score][cancer], xmin=-width/2+i, xmax=i+width*len(ra.get_res_name(res_id))-width/2,
                    linestyles='dashed', color='black')
            ax.hlines(baselines[us]['random'][score][cancer], xmin=-width/2+i, xmax=i+width*len(ra.get_res_name(res_id))-width/2,
                    linestyles='dotted', color='black')

    plt.legend(loc='upper right',)
    #plt.xlim((-0.1,5))
    plt.xlabel('Cancer Type', fontweight='bold')
    plt.ylabel(score, fontweight='bold')
    #plt.xticks([r + len(labels)*(width)/2+width for r in range(len(labels))], cancer_list)
    #ax.set_xlabel('Sequence Dimension')
    ax.set_ylabel(score)
    #plt.title(score+'\tSeq+PPI+Crispr_Dep_Mut+Crispr_Dep_Exp+Tissue')
    plt.savefig(out_loc, bbox_inches="tight")
    plt.show()


def plot_single_with_colm(loc= 'results/elrrf/single_cancer_validation.csv', score= 'AUROC', selections=None, res_id=14,
                            baselines = {}, us='', out_name='temp.png'):
    if selections is None:
        selections = {'balance_strat': ['undersample_train'],
                      'comb_type': ['type2'], 'process': [True],
                      'grid_search': ['False']}
    out_loc = config.ROOT_DIR / ('/'.join(loc.split('/')[:-1])) / 'images'/ out_name
    config.ensure_dir(out_loc)
    plt.rcdefaults()
    fig, ax = plt.subplots(figsize=(12,8))

    cancer_list=['BRCA', 'COAD', 'LUAD', 'OV']

    width = 0.15  # the width of the bars
    x_axis = []
    cancer_dict=coll.OrderedDict()

    for cancer in cancer_list:
        selections.update({'cancer': [cancer]})
        res = ra.get_result_loc(loc=loc, cancer=cancer, res_names=ra.get_res_name(res_id), selections=selections,
                                out_cols=[score + '_m', score + '_std'], bestMC=True, chosen_th=None)
        cancer_dict[cancer] = res

    labels = list(cancer_dict.keys())
    x_axiss = np.arange(len(labels))
    x_axis = [x + 0 * width for x in x_axiss]
    means = [float(res.loc[ra.get_res_name(res_id)[0], score+'_m']) for cancer, res in cancer_dict.items()]
    stds = [float(res.loc[ra.get_res_name(res_id)[0], score+'_std']) for cancer, res in cancer_dict.items()]
    tmp_plt = ax.bar(x_axis, means,width, yerr=stds, label='ELRRF', capsize=3)
    '''
    c_x_axis = [x + 1 * width for x in x_axiss]
    c_means = [best_single_res[cancer][score+'_m'] for cancer in cancer_dict.keys()]
    c_stds = [best_single_res[cancer][score+'_std'] for cancer in cancer_dict.keys()]
    best_single_plt = ax.bar(c_x_axis, c_means,width, yerr=c_stds, label="Single best", capsize=3)

    c_x_axis = [x + 2 * width for x in x_axiss]
    c_means = [onehot_res[us][cancer][score+'_m'] for cancer in cancer_dict.keys()]
    c_stds = [onehot_res[us][cancer][score+'_std'] for cancer in cancer_dict.keys()]
    onehot_plt = ax.bar(c_x_axis, c_means,width, yerr=c_stds, label="Onehot vectors", capsize=3)

    c_x_axis = [x + 3 * width for x in x_axiss]
    c_means = [colm_res[cancer][score+'_m'] for cancer in cancer_dict.keys()]
    c_stds = [colm_res[cancer][score+'_std'] for cancer in cancer_dict.keys()]
    colm_plt = ax.bar(c_x_axis, c_means,width, yerr=c_stds, label="Colm's method", capsize=3)
    '''
    #c_x_axis = [x + 4 * width for x in x_axiss]
    #c_means = [pca_gCMF_res[cancer][score+'_m'] for cancer in cancer_dict.keys()]
    #c_stds = [pca_gCMF_res[cancer][score+'_std'] for cancer in cancer_dict.keys()]
    #pca_gCMF_plt = ax.bar(c_x_axis, c_means,width, yerr=c_stds, label="pca_gCMF", capsize=3)

    #tmp_plt[-1][0].set_linestyle('--')
    if 'AUROC' in score or ('AUPRC' in score and us=='trteus'):
        plt.ylim((0.45,1))
    no_of_exp = 1
    for i, cancer in enumerate(labels):
        if i==0:
            ax.hlines(baselines[us]['theory'][score][cancer], xmin=-width/2+i, xmax=i+width*no_of_exp-width/2,
                    linestyles='dashed', color='black', label='Theoretical Random')
            ax.hlines(baselines[us]['random'][score][cancer], xmin=-width/2+i, xmax=i+width*no_of_exp-width/2,
                    linestyles='dotted', color='black', label='Computational Random')
        else:
            ax.hlines(baselines[us]['theory'][score][cancer], xmin=-width/2+i, xmax=i+width*no_of_exp-width/2,
                    linestyles='dashed', color='black')
            ax.hlines(baselines[us]['random'][score][cancer], xmin=-width/2+i, xmax=i+width*no_of_exp-width/2,
                    linestyles='dotted', color='black')

    plt.legend(loc='upper right',)
    #plt.xlim((-0.1,5))
    plt.xlabel('Cancer Type', fontweight='bold')
    plt.ylabel(score, fontweight='bold')
    plt.xticks([r + width*(3/2) for r in range(len(labels))], cancer_list)
    #ax.set_xlabel('Sequence Dimension')
    ax.set_ylabel(score)
    #plt.title(score+'\tSeq+PPI+Crispr_Dep_Mut+Crispr_Dep_Exp+Tissue')
    #plt.savefig(out_loc, bbox_inches="tight")
    plt.show()


def plot_single_with_onehot(loc= 'results/elrrf/single_cancer_validation.csv', score= 'AUROC', selections=None, res_id=14,
                            baselines = {}, us='', out_name='temp.png'):
    if selections is None:
        selections = {'balance_strat': ['undersample_train'],
                      'comb_type': ['type2'], 'process': [True],
                      'grid_search': ['False']}
    out_loc = config.ROOT_DIR / ('/'.join(loc.split('/')[:-1])) / 'images'/ out_name
    config.ensure_dir(out_loc)
    plt.rcdefaults()
    fig, ax = plt.subplots(figsize=(12,8))

    cancer_list=['BRCA', 'CESC', 'KIRC', 'COAD', 'LAML', 'LUAD', 'OV', 'SKCM']

    width = 0.35  # the width of the bars
    x_axis = []
    cancer_dict=coll.OrderedDict()

    for cancer in cancer_list:
        selections.update({'cancer': [cancer]})
        res = ra.get_result_loc(loc=loc, cancer=cancer, res_names=ra.get_res_name(res_id), selections=selections,
                                out_cols=[score + '_m', score + '_std'], bestMC=True, chosen_th=None)
        cancer_dict[cancer] = res

    labels = list(cancer_dict.keys())
    x_axiss = np.arange(len(labels))
    x_axis = [x + 0 * width for x in x_axiss]
    means = [float(res.loc[ra.get_res_name(res_id)[0], score+'_m']) for cancer, res in cancer_dict.items()]
    stds = [float(res.loc[ra.get_res_name(res_id)[0], score+'_std']) for cancer, res in cancer_dict.items()]
    tmp_plt = ax.bar(x_axis, means,width, yerr=stds, label='ELRRF', capsize=3)

    c_x_axis = [x + 1 * width for x in x_axiss]
    c_means = [onehot_res[us][cancer][score+'_m'] for cancer in cancer_dict.keys()]
    c_stds = [onehot_res[us][cancer][score+'_std'] for cancer in cancer_dict.keys()]
    colm_plt = ax.bar(c_x_axis, c_means,width, yerr=c_stds, label="Onehot vectors", capsize=3)

    #tmp_plt[-1][0].set_linestyle('--')
    if 'AUROC' in score or ('AUPRC' in score and us=='trteus'):
        plt.ylim((0.45,1))
    for i, cancer in enumerate(labels):
        if i==0:
            ax.hlines(baselines[us]['theory'][score][cancer], xmin=-width/2+i, xmax=i+width*len(ra.get_res_name(res_id))-width/2,
                    linestyles='dashed', color='black', label='Theoretical Random')
            ax.hlines(baselines[us]['random'][score][cancer], xmin=-width/2+i, xmax=i+width*len(ra.get_res_name(res_id))-width/2,
                    linestyles='dotted', color='black', label='Computational Random')
        else:
            ax.hlines(baselines[us]['theory'][score][cancer], xmin=-width/2+i, xmax=i+width*len(ra.get_res_name(res_id))-width/2,
                    linestyles='dashed', color='black')
            ax.hlines(baselines[us]['random'][score][cancer], xmin=-width/2+i, xmax=i+width*len(ra.get_res_name(res_id))-width/2,
                    linestyles='dotted', color='black')

    plt.legend(loc='upper right',)
    #plt.xlim((-0.1,5))
    plt.xlabel('Cancer Type', fontweight='bold')
    plt.ylabel(score, fontweight='bold')
    plt.xticks([r + width/2 for r in range(len(labels))], cancer_list)
    #ax.set_xlabel('Sequence Dimension')
    ax.set_ylabel(score)
    #plt.title(score+'\tSeq+PPI+Crispr_Dep_Mut+Crispr_Dep_Exp+Tissue')
    plt.savefig(out_loc, bbox_inches="tight")
    plt.show()


def tsne_gtex_pats(method='tsne'):
    loc_dict = tcga.get_locs()
    cancer_dict = cgt.get_golden_truth_by_cancer()
    gtex_dict = coll.OrderedDict()
    cancer_ind = np.array([])
    last_ind = 0
    for cancer in cancer_dict.keys():
        gtex_dict[cancer] = tcga.load_cancer_gtex_expr(loc_dict[cancer]['gtex_expression']).T
        cancer_ind = np.concatenate([cancer_ind, np.repeat([cancer], gtex_dict[cancer].shape[0])])

    features = np.concatenate(list(gtex_dict.values()))
    tsne_features = TSNE(n_components=2, perplexity=30).fit_transform(features)
    for cancer in gtex_dict.keys():
        chosen_ind = cancer_ind == cancer
        plt.scatter(tsne_features[chosen_ind,0], tsne_features[chosen_ind,1], label=cancer)

    plt.legend()
    plt.savefig('gtex_pats.png', bbox_inches="tight")
    plt.show()


def cluster_pats(source='gtex', method='hierarchical'):
    loc_dict = tcga.get_locs()
    #cancer_dict = cgt.get_golden_truth_by_cancer()
    expr_dict = coll.OrderedDict()
    cancer_ind = np.array([])
    last_ind = 0
    for cancer in cancer_list:
        if source=='gtex':
            expr_dict[cancer] = tcga.load_cancer_gtex_expr(loc_dict[cancer]['gtex_expression']).T
        else:
            expr_dict[cancer] = tcga.load_all_expr(loc_dict[cancer]['expression']).T
        cancer_ind = np.concatenate([cancer_ind, np.repeat([cancer], expr_dict[cancer].shape[0])])

    features = np.concatenate(list(expr_dict.values()))
    n_samples=len(features)
    plt.figure(figsize=(15, 7))
    '''
    linked = linkage(features, 'single')
    dendrogram(linked,
               orientation='top',
               distance_sort='descending',
               show_leaf_counts=True)
    plt.show()
    '''
    model = AgglomerativeClustering(n_clusters=2, compute_distances=True)
    model = model.fit(features)
    plt.title('Hierarchical Clustering Dendrogram')
    # plot the top three levels of the dendrogram
    linkage_matrix = plot_dendrogram(model, truncate_mode='level', p=4)
    link_df = pd.DataFrame(linkage_matrix, columns=['child1', 'child2', 'dist', 'total_ch'])
    for cancer in cancer_list:
        link_df[cancer]=0
    for i, merge in link_df.iterrows():
        for child_idx in merge[:2]:
            if child_idx < n_samples:
                link_df.loc[i, cancer_ind[int(child_idx)]] = link_df.loc[i, cancer_ind[int(child_idx)]]+1  # leaf node
            else:
                link_df.loc[i, cancer_list] = link_df.loc[i, cancer_list] + link_df.loc[child_idx - n_samples, cancer_list]

    link_df[['child1', 'child2']] = link_df[['child1', 'child2']].astype(int)
    link_df[cancer_list] = link_df[cancer_list].astype(int)
    for i in range(len(link_df) - 1, -1, -1):
        c1 = int(link_df.loc[i, 'child1'] - n_samples)
        c2 = int(link_df.loc[i, 'child2'] - n_samples)
        sum = ''
        for cancer in cancer_list:
            sum = sum + cancer + '(' + str(link_df.loc[i, cancer]) + '/'+str(len(expr_dict[cancer]))+'), '
        print(f'Cluster{i} from C{c1}+C{c2} with {sum}')

    plt.xlabel("Number of points in node (or index of point if no parenthesis).")
    plt.savefig(source+'_pats_dendogram.pdf')
    plt.show()

def plot_box_with_points(cancer='BRCA', aim='AUROC', out='single'):
    if out=='single':
        all_res = ra.analyze_single_model_res(cancer)
    elif out=='dho':
        all_res = ra.analyze_dho_model_res(cancer)
    vals, names, xs = [], [], []
    for i, (model, res) in enumerate(all_res.items()):
        vals.append(res[aim])
        names.append(model)
        xs.append(np.random.normal(i + 1, 0.04, len(res[aim])) )
    plt.boxplot(vals, labels=names)
    palette = ['#ff0000','#ff7f00','#ffff00','#7fff00','#00ffff','#0000ff','#7f00ff','#ff00ff','#007fff']
    for x, val, c in zip(xs, vals, palette):
        plt.scatter(x, val, alpha=0.4, color=c)
    plt.xticks(rotation=60)
    out_loc = config.ROOT_DIR / ('results/images/'+out+'_'+cancer+'_'+aim)
    config.ensure_dir(out_loc)
    plt.ylabel(cancer+' '+aim)
    plt.ylim(0,1)
    plt.savefig(out_loc, bbox_inches="tight")
    plt.show()


def plot_facet_box_with_points(cancer_list=['BRCA'], aim='AUROC', out='single', one_colm=False):

    palettes = ['#ff0000','#ff7f00','#ffff00','#7fff00','#00ffff','#0000ff','#7f00ff','#ff00ff','#007fff']

    out_res_loc = config.ROOT_DIR / ('results/tmp_res/' + out + '_' + str(one_colm) + '_10folds.csv')
    config.ensure_dir(out_res_loc)
    out_single_res_loc = config.ROOT_DIR / ('results/tmp_res/' + 'single2' + '_' + str(one_colm) + '_10folds.csv')
    config.ensure_dir(out_single_res_loc)
    if os.path.exists(out_res_loc):
        res_df = pd.read_csv(out_res_loc, index_col=0)
    else:
        res_rows = []
        res_single_rows = []
        for cancer in cancer_list:
            if out=='single':
                all_res = ra.analyze_single_model_res(cancer)
            if out=='crossds':
                cancer_only, train_ds, test_ds = cancer.split('_')
                all_res = ra.analyze_cd_model_res(cancer_only, train_ds, test_ds)
            elif out=='dho' or out=='dho2':
                if cancer in ['COAD', 'SKCM', 'KIRC', 'LAML']:
                    continue
                all_res = ra.analyze_dho_model_res(cancer)
                if not os.path.exists(out_single_res_loc):
                    all_single_res = ra.analyze_single_model_res(cancer)
                    if one_colm:
                        del all_single_res['Seale_L0L2'], all_single_res['Seale_RRF']
                    for method, scores in all_single_res.items():
                        for i in range(10):
                            try:
                                one_row = [cancer, method, i, scores['AUROC'][i], scores['AUPRC'][i], scores['MCC'][i]]
                            except:
                                print()
                            res_single_rows.append(one_row)

            if one_colm:
                del all_res['Seale_L0L2'], all_res['Seale_RRF']
            for method, scores in all_res.items():
                for i in range(10):
                    try:
                        one_row = [cancer, method, i, scores['AUROC'][i], scores['AUPRC'][i], scores['MCC'][i]]
                    except:
                        print()
                    res_rows.append(one_row)
        res_df = pd.DataFrame(res_rows, columns=['cancer', 'method', 'fold_id', 'AUROC', 'AUPRC', 'MCC'])
        res_df.to_csv(out_res_loc)

    if out == 'dho' or out=='dho2':
        if os.path.exists(out_single_res_loc):
            res_single_df = pd.read_csv(out_single_res_loc, index_col=0)
        else:
            res_single_df = pd.DataFrame(res_single_rows, columns=['cancer', 'method', 'fold_id', 'AUROC', 'AUPRC', 'MCC'])
            res_single_df.to_csv(out_single_res_loc)
        res_single_df['model']='single'
        res_df['model']='dho'
        res_single_df['metmod']=res_single_df['method']
        res_df['metmod']=res_single_df['method']+'_dho'
        res_df = pd.concat([res_df, res_single_df])
        if one_colm:
            res_df['metmod'] = pd.Categorical(res_df['metmod'], ["GCATSL", "GCATSL_dho", "GRSMF", "GRSMF_dho",
                                                                 "pca-gCMF", "pca-gCMF_dho", "Seale_EN", "Seale_EN_dho",
                                                                 "Seale_MUVR", "Seale_MUVR_dho", "ELRRF", "ELRRF_dho",
                                                                 "ELGBDT", "ELGBDT_dho"
                                                                 ])
        else:
            res_df['metmod'] = pd.Categorical(res_df['metmod'], ["GCATSL", "GCATSL_dho", "GRSMF", "GRSMF_dho",
                                                             "pca-gCMF", "pca-gCMF_dho", "Seale_EN", "Seale_EN_dho",
                                                             "Seale_L0L2", "Seale_L0L2_dho", "Seale_RRF", "Seale_RRF_dho",
                                                             "Seale_MUVR", "Seale_MUVR_dho", "ELRRF", "ELRRF_dho",
                                                             "ELGBDT", "ELGBDT_dho"
                                                             ])

    if out=='single' or out=='crossds':
        if one_colm:
            res_df['method'] = pd.Categorical(res_df['method'], ["GCATSL", "GRSMF", "pca-gCMF", "Seale_EN", "Seale_MUVR", "ELRRF",
                                                                 "ELGBDT"
                                                                 ])
    plt.clf()
    if out=='dho2':
        res_df.sort_values('metmod')
    def boxandswamp(x, y, **kwargs):
        ax = plt.gca()
        from scipy.stats import wilcoxon
        from scipy.stats import ranksums
        data = kwargs.pop("data")
        mean_df = data.groupby('method').agg({'AUPRC':'median'})
        best_el = mean_df.loc[['ELRRF', 'ELGBDT'],:].idxmax()[0]
        best_others = mean_df.drop(index=['ELRRF', 'ELGBDT']).idxmax()[0]
        w, p = wilcoxon(data[data['method']==best_el]['AUPRC'].values, data[data['method']==best_others]['AUPRC'].values)
        print(f'{data["cancer"].unique()[0]}: Wilcoxon signed test p-val between {best_el}, {best_others}={p}')
        colors = ['#882E72', '#1965B0', '#4EB265', '#F7F056', '#EE8026', '#DC050C', '#42150A']
        box_ax = sns.boxplot(x, y, data=data, palette=colors)#, color='skyblue')
        for patch in box_ax.artists:
            r, g, b, a = patch.get_facecolor()
            patch.set_facecolor((r, g, b, .1))
        sns.swarmplot(x, y, data=data, palette=colors)#, color='blue')
        x_axis_ticks = ax.get_xaxis().get_majorticklocs()
        method_names = mean_df.index.values
        best1_loc, = np.where(method_names == best_others)
        best2_loc, = np.where(method_names == best_el)
        best1_loc = x_axis_ticks[best1_loc]
        best2_loc = x_axis_ticks[best2_loc]
        label_txt = 'P-val: '+str(round(p,5))
        plt.text((best2_loc+best1_loc)/2, 0.43, label_txt, horizontalalignment='center', size='medium', color='black',
                 weight='semibold')
        plt.hlines(y = 0.42, xmin=best1_loc, xmax=best2_loc, color = 'red')
        lowest_best_el = data[data['method']==best_el]['AUPRC'].min()
        lowest_best_others = data[data['method']==best_others]['AUPRC'].min()
        plt.vlines(x = best1_loc, ymin=0.42, ymax=lowest_best_others-0.03, color = 'red')
        plt.vlines(x = best2_loc, ymin=0.42, ymax=lowest_best_el-0.03, color = 'red')

    def boxandswamp_dho2(x, y, **kwargs):
        ax = plt.gca()
        from scipy.stats import wilcoxon
        data = kwargs.pop("data")
        mean_df = data.groupby(x).agg({'AUPRC':'median'})
        best_el = mean_df.loc[['ELRRF', 'ELGBDT'],:].idxmax()[0]+'_dho'
        single_ress = [idx_name for idx_name in mean_df.index if 'dho' in idx_name]
        single_ress.extend(['ELRRF', 'ELGBDT'])
        best_others = mean_df.drop(index=single_ress).idxmax()[0]+'_dho'
        w, p = wilcoxon(data[data[x]==best_el]['AUPRC'].values, data[data[x]==best_others]['AUPRC'].values)
        data_dho = data[data['model']=='dho']
        palette = ['ff0000', 'ff7f00', 'ffff00', '7fff00', '00ffff', '0000ff', '7f00ff', 'ff00ff', '007fff']
        palette2 = ['#ff0000', '#ff0000', '#ff7f00', '#ff7f00', '#ffff00', '#ffff00', '#7fff00', '#7fff00',
                    '#00ffff', '#00ffff', '#0000ff', '#0000ff', '#7f00ff', '#7f00ff', '#ff00ff', '#ff00ff',
                    '#007fff', '#007fff']
        palette3 = ['#A9A9A9', '#ff0000', '#A9A9A9', '#ff7f00', '#A9A9A9', '#ffff00', '#A9A9A9', '#7fff00',
                    '#A9A9A9', '#00ffff', '#A9A9A9', '#0000ff', '#A9A9A9', '#7f00ff', '#A9A9A9', '#ff00ff',
                    '#A9A9A9', '#007fff']
        palette4 = ['#944F88', '#882E72', '#437DBF', '#1965B0', '#90C987', '#4EB265', '#fffcba', '#F7F056',
                    '#F4A736', '#EE8026', '#ff595f', '#DC050C', '#633c33', '#42150A']
        box_ax = sns.boxplot(x, y, data=data, palette=palette4)#, color='skyblue')
        for idd, patch in enumerate(box_ax.artists):
            #if idd%2==0:
            #    r,g,b = 169,169,169
            #else:
            #r,g,b = tuple(int(palette4[int(idd/2)][1:][i:i + 2], 16) for i in (0, 2, 4))
            r, g, b, a = patch.get_facecolor()
            patch.set_facecolor((r, g, b, .1))
        box_ax2 = sns.swarmplot(x, y, data=data, palette=palette4)#, color='blue')
        x_axis_ticks = ax.get_xaxis().get_majorticklocs()
        method_names = mean_df.index.values
        best1_loc, = np.where(method_names == best_others)
        best2_loc, = np.where(method_names == best_el)
        best1_loc = x_axis_ticks[best1_loc]
        best2_loc = x_axis_ticks[best2_loc]
        label_txt = 'P-val: ' + str(round(p, 5))
        plt.text((best2_loc + best1_loc) / 2, 0.40, label_txt, horizontalalignment='center', size='medium',
                 color='black',
                 weight='semibold')
        plt.hlines(y=0.39, xmin=best1_loc, xmax=best2_loc, color='red')
        lowest_best_el = data[data[x] == best_el]['AUPRC'].min()
        lowest_best_others = data[data[x] == best_others]['AUPRC'].min()
        plt.vlines(x=best1_loc, ymin=0.39, ymax=lowest_best_others - 0.03, color='red')
        plt.vlines(x=best2_loc, ymin=0.39, ymax=lowest_best_el - 0.03, color='red')
        #for idd, patch in enumerate(box_ax2.artists):
        #    r,g,b = tuple(int(palette[int(idd/2)][i:i + 2], 16) for i in (0, 2, 4))
            #r, g, b, a = patch.get_facecolor()
        #    patch.set_facecolor((r/255.0, g/255.0, b/255.0, .1))
        data_single = data[data['model'] == 'single']
        #sbox_ax = sns.boxplot(x, y, data=data_single, color='gray')
        #for patch in sbox_ax.artists:

        #for i in range(len(data['metmod'].unique())):
        #    patch = box_ax.artists[i]
        #    r, g, b, a = patch.get_facecolor()
        #    patch.set_facecolor((r, g, b, .1))
            #meanv = data_single[data_single['method']==data['method'].unique()[i]][aim].mean()
            #plt.axhline(y = meanv, xmin=0.01+i*0.11, xmax=+0.11+i*0.11, ls = '--', c = (r, g, b, .6))

        #for i in range(i, 2*len(data['method'].unique())):
        #    patch = box_ax.artists[i]
        #    r, g, b, a = patch.get_facecolor()
        #    patch.set_facecolor((r, g, b, .05))

    def boxandswamp_dho(x, y, z, **kwargs):
        ax = plt.gca()
        data = kwargs.pop("data")
        data_dho = data[data['model']=='dho']
        box_ax = sns.boxplot(x, y, data=data_dho, hue=z)#, color='skyblue')
        #for patch in box_ax.artists:
        #    r, g, b, a = patch.get_facecolor()
        #    patch.set_facecolor((r, g, b, .1))
        sns.swarmplot(x, y, data=data_dho)#, color='blue')
        data_single = data[data['model'] == 'single']
        #sbox_ax = sns.boxplot(x, y, data=data_single, color='gray')
        #for patch in sbox_ax.artists:
        for i in range(len(data['method'].unique())):
            patch = box_ax.artists[i]
            r, g, b, a = patch.get_facecolor()
            patch.set_facecolor((r, g, b, .1))
            meanv = data_single[data_single['method']==data['method'].unique()[i]][aim].mean()
            plt.axhline(y = meanv, xmin=0.01+i*0.11, xmax=+0.11+i*0.11, ls = '--', c = (r, g, b, .6))
        #for i in range(i, 2*len(data['method'].unique())):
        #    patch = box_ax.artists[i]
        #    r, g, b, a = patch.get_facecolor()
        #    patch.set_facecolor((r, g, b, .05))

    if out=='single':
        g = sns.FacetGrid(res_df, col='cancer', col_wrap=4, height=5, aspect=1)
    elif out=='crossds':
        g = sns.FacetGrid(res_df, col='cancer', col_wrap=3, height=5, aspect=1)
    elif out=='dho' or out=='dho2':
        g = sns.FacetGrid(res_df, col='cancer',  col_wrap=2, height=5, aspect=1.4, palette=palettes)
    #g.add_legend()
    if out=='single' or out=='crossds':
        g_map = g.map_dataframe(boxandswamp, "method", aim)
        [plt.setp(ax.get_xticklabels(), rotation=75) for ax in g_map.axes.flat]
    elif out=='dho':
        g_map = g.map_dataframe(boxandswamp_dho, "method", aim)
        [plt.setp(ax.get_xticklabels(), rotation=75) for ax in g_map.axes.flat]
    elif out=='dho2':
        g_map = g.map_dataframe(boxandswamp_dho2, "metmod", aim)
        [plt.setp(ax.get_xticklabels(), rotation=90) for ax in g_map.axes.flat]
    plt.ylabel('')
    g.map(plt.axhline, y=0.5, ls='--', c='black')
    g.set_axis_labels(x_var='Methods', y_var='AUPRC')
    legend_elements=[]
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    leg_labels = list(res_df['method'].unique())
    leg_labels.append('Random')
    if out=='dho2' or out=='single' or out=='crossds':
        met_names = ["GCATSL", "GRSMF", "pca-gCMF", "Seale_EN", "Seale_L0L2", "Seale_RRF", "Seale_MUVR", "ELRRF","ELGBDT"]
        colors=['#ff0000', '#ff7f00', '#ffff00', '#7fff00', '#00ffff', '#0000ff', '#7f00ff', '#ff00ff', '#007fff']
        if one_colm:
            met_names = ["GCATSL", "GRSMF", "pca-gCMF", "Seale_EN", "Seale_MUVR", "ELRRF","ELGBDT"]
            colors = [ '#882E72', '#1965B0', '#4EB265', '#F7F056', '#EE8026', '#DC050C', '#42150A']
        for collor_id in range(len(met_names)):
            legend_elements.append(Line2D([0], [0], marker='o', color='w', label=met_names[collor_id],
                                          markerfacecolor=colors[collor_id], markersize=8))
        legend_elements.append(Line2D([0], [0], linestyle='--', color='black', label='Random'))
    else:
        for leg_id, leg_name in enumerate(leg_labels):
            #x_axis_id, p in enumerate(g.axes_dict['BRCA'].artists):
            if leg_name=='Random':
                legend_elements.append(Line2D([0], [0], linestyle='--', color='black', label=leg_name))
            else:
                if out=='crossds':
                    p = g.axes_dict['BRCA_isle_dsl'].artists[leg_id]
                else:
                    p = g.axes_dict['BRCA'].artists[leg_id]

                r, gr, b, a = p.get_facecolor()
                legend_elements.append(Line2D([0], [0], marker='o', color='w', label=leg_name,
                    markerfacecolor=(r, gr, b, 1), markersize=8))

    #g.add_legend()

    # Create the figure
    #fig, ax = plt.subplots()
    #ax = plt.gca()
    #ax.legend(handles=legend_elements)
    if out=='single':
        plt.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1.1, 0.5, 1,1), borderaxespad=0.)
    elif out=='crossds':
        plt.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1.1, 0.5, 1,1), borderaxespad=0.)
    elif out=='dho2':
        plt.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1.15, 0.5, 1,1), borderaxespad=0.)

    out_loc = config.ROOT_DIR / ('results/images/'+out+'_'+aim+'_colorful_10_leg_7june')
    config.ensure_dir(out_loc)
    plt.savefig(out_loc,  bbox_inches="tight")
    plt.savefig(out_loc, format='pdf',  bbox_inches="tight")
    plt.tight_layout()
    plt.show()



def trauc_vs_contr(cancers=['BRCA'], aim='ELRRF'):
    ELRRF_BRCA = {'0':{
        'Contr':{'seq_1024': 0.1752, 'ppi_ec': 0.1682, 'crispr_dependency_mut': 0.1674, 'crispr_dependency_expr': 0.1691, 'tissue': 0.1416, 'seq_1024&ppi_ec&crispr_dependency_mut&crispr_dependency_expr&tissue': 0.1785},
        'TR_auc':{'seq_1024': 0.1778, 'ppi_ec': 0.1684, 'crispr_dependency_mut': 0.163, 'crispr_dependency_expr': 0.1615, 'tissue': 0.1512, 'seq_1024&ppi_ec&crispr_dependency_mut&crispr_dependency_expr&tissue': 0.178}},
        '1':{
            'Contr':{'seq_1024': 0.1723, 'ppi_ec': 0.1727, 'crispr_dependency_mut': 0.1692, 'crispr_dependency_expr': 0.176, 'tissue': 0.1315, 'seq_1024&ppi_ec&crispr_dependency_mut&crispr_dependency_expr&tissue': 0.1782},
            'TR_auc':{'seq_1024': 0.1766, 'ppi_ec': 0.1718, 'crispr_dependency_mut': 0.163, 'crispr_dependency_expr': 0.1612, 'tissue': 0.1494, 'seq_1024&ppi_ec&crispr_dependency_mut&crispr_dependency_expr&tissue': 0.178}},
        '2':{
            'Contr':{'seq_1024': 0.1715, 'ppi_ec': 0.1673, 'crispr_dependency_mut': 0.1716, 'crispr_dependency_expr': 0.1704, 'tissue': 0.1427, 'seq_1024&ppi_ec&crispr_dependency_mut&crispr_dependency_expr&tissue': 0.1766},
            'TR_auc':{'seq_1024': 0.1766, 'ppi_ec': 0.1715, 'crispr_dependency_mut': 0.1645, 'crispr_dependency_expr': 0.1613, 'tissue': 0.1483, 'seq_1024&ppi_ec&crispr_dependency_mut&crispr_dependency_expr&tissue': 0.1778}},
        '3':{
            'Contr':{'seq_1024': 0.1766, 'ppi_ec': 0.1687, 'crispr_dependency_mut': 0.1761, 'crispr_dependency_expr': 0.1747, 'tissue': 0.1249, 'seq_1024&ppi_ec&crispr_dependency_mut&crispr_dependency_expr&tissue': 0.179},
            'TR_auc':{'seq_1024': 0.1762, 'ppi_ec': 0.1708, 'crispr_dependency_mut': 0.1635, 'crispr_dependency_expr': 0.1607, 'tissue': 0.1517, 'seq_1024&ppi_ec&crispr_dependency_mut&crispr_dependency_expr&tissue': 0.1772}},
        '4':{
            'Contr':{'seq_1024': 0.1719, 'ppi_ec': 0.1692, 'crispr_dependency_mut': 0.1714, 'crispr_dependency_expr': 0.1743, 'tissue': 0.1364, 'seq_1024&ppi_ec&crispr_dependency_mut&crispr_dependency_expr&tissue': 0.1767},
            'TR_auc':{'seq_1024': 0.1759, 'ppi_ec': 0.1692, 'crispr_dependency_mut': 0.1642, 'crispr_dependency_expr': 0.1621, 'tissue': 0.1518, 'seq_1024&ppi_ec&crispr_dependency_mut&crispr_dependency_expr&tissue': 0.1768}}}



    cancer='BRCA'
    contr_vals, trauc_vals, x_names = [], [], []
    df_vals, data_ids = [], []
    for fold in ['0', '1', '2', '3', '4']:
        for data_id, data in enumerate(['seq_1024', 'ppi_ec', 'crispr_dependency_mut', 'crispr_dependency_expr', 'tissue', 'seq_1024&ppi_ec&crispr_dependency_mut&crispr_dependency_expr&tissue']):
            df_row1 = [str(data_id+1), data, fold, ELRRF_BRCA[fold]['Contr'][data], 'CV_AUPRC']
            df_row2 = [str(data_id+1), data, fold, ELRRF_BRCA[fold]['TR_auc'][data], 'TR_AUPRC']
            df_vals.append(df_row1)
            df_vals.append(df_row2)
            contr_vals.append(ELRRF_BRCA[fold]['Contr'][data])
            trauc_vals.append(ELRRF_BRCA[fold]['TR_auc'][data])
            x_names.append('Fold'+fold+': '+str(data_id+1))
            data_ids.append(str(data_id+1))

    df = pd.DataFrame(df_vals, columns=['Data_Id', 'Data', 'Fold', 'Contr', 'Type'])
    g = sns.FacetGrid(df, col='Fold', hue='Type', col_wrap=3,gridspec_kws={"wspace":0.2})
    g = g.map_dataframe(sns.scatterplot, "Data_Id", "Contr")
    g.set_axis_labels("Data Ids", "Contribution")
    #g.set(xticks=[1,2,3,4,5,6])
    #g.add_legend(loc='upper right')
    plt.legend(loc='upper right', bbox_to_anchor=(1.8, 1))
    for ax in g.axes.flat:
        for data_id in data_ids:
            ax.axvline(data_id, c=".2", ls="--", zorder=0)
    out_loc = config.ROOT_DIR / ('results/images/contr_'+cancer+'_'+aim)

    plt.savefig(out_loc, bbox_inches="tight")
    plt.tight_layout()
    plt.show()
    return 0
    r, p_val = stats.pearsonr(contr_vals, trauc_vals)
    plt.scatter(range(len(contr_vals)), contr_vals, alpha=1, color='blue', label='CV-AUPRC')
    plt.scatter(range(len(trauc_vals)), trauc_vals, alpha=1, color='red', label='TRAIN-AUPRC')
    p = '%.3E' % p_val
    title = f'Pearson Correlation: Corr:{np.round(r,2)}, p_val:{p}'
    plt.title(title, fontsize=12)
    plt.legend(loc='lower left')
    plt.grid(axis='x')
    plt.xticks(range(len(x_names)), x_names, rotation=90, fontsize=8)
    out_loc = config.ROOT_DIR / ('results/images/contr_'+cancer+'_'+aim)
    config.ensure_dir(out_loc)
    plt.ylabel(cancer+' '+aim)
    plt.savefig(out_loc, bbox_inches="tight")
    plt.show()


def plt_dataset_importance(aim='ELGBDT'):

    res_loc = config.ROOT_DIR / 'results' / aim / 'single_cancer_set_imp_test.csv'
    res = pd.read_csv(res_loc)
    contr_vals, trauc_vals, x_names = [], [], []
    df_vals, data_ids = [], []

    def errplot(x, y, yerr, **kwargs):
        ax = plt.gca()
        data = kwargs.pop("data")
        data.plot(x=x, y=y, yerr=yerr, kind="bar", ax=ax, **kwargs)


    for row_id, row in res.iterrows():
        for data_id, data in enumerate(
                ['seq_1024', 'ppi_ec', 'crispr_dependency_mut', 'crispr_dependency_expr', 'tissue',
                 'seq_1024&ppi_ec&crispr_dependency_mut&crispr_dependency_expr&tissue']):
            df_new_row = [row['cancer']]
            df_new_row.append(str(data_id + 1))
            df_new_row.append(data)
            df_new_row.append(row[data+'_m'])
            df_new_row.append(row[data+'_std'])
            df_vals.append(df_new_row)

    df = pd.DataFrame(df_vals, columns=['cancer', 'data_id', 'data_name', 'mean', 'std'])

    g = sns.FacetGrid(df, col="cancer", col_wrap=4, col_order=['BRCA','CESC', 'COAD', 'KIRC', 'LAML', 'LUAD', 'OV', 'SKCM'])
    g.map_dataframe(errplot, "data_id", "mean", "std")

    g.set_axis_labels("Data Ids", "Importance")
    # g.set(xticks=[1,2,3,4,5,6])
    # g.add_legend(loc='upper right')
    #plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    #for ax in g.axes.flat:
    #    for data_id in data_ids:
    #        ax.axvline(data_id, c=".2", ls="--", zorder=0)
    out_loc = config.ROOT_DIR / ('results/images/set_imp_type1_'+aim+'.png')

    plt.savefig(out_loc, bbox_inches="tight")
    plt.tight_layout()
    plt.show()


def plt_dataset_detail_importance(aim='ELRRF', typ_name = 'type2'):
    contr_vals, trauc_vals, x_names = [], [], []
    df_vals, data_ids = [], []

    def errplot(x, y, yerr, **kwargs):
        ax = plt.gca()
        data = kwargs.pop("data")
        data.plot(x=x, y=y, yerr=yerr, kind="bar", ax=ax, **kwargs)

    def myplot(x, y, **kwargs):
        ax = plt.gca()
        data = kwargs.pop("data")
        sns.pointplot(x=x, y=y, data=data, ci='sd', errwidth= 2, markers='d', join=False)

    def myplot2(m, std, d_name, **kwargs):
        data = kwargs.pop("data")
        #sns.stripplot(data= data, x=m, y=d_name, order=list(d_name_coord.keys()))
        #ax = plt.gca()
        #for id, row in data.iterrows():
        #    ax.axhline(y=d_name_coord[row['data_name']], xmin=row[m]-row[std], xmax=row[m]+row[std])


    data_name_dict = {'seq_1024':'Sequence', 'ppi_ec':'PPI', 'crispr_dependency_mut':'Crispr Dependency with Mutation', 'crispr_dependency_expr':'Crispr Dependency with Expression', 'tissue':'Tissue'}
    data_id_dict = {'seq_1024':0, 'ppi_ec':1, 'crispr_dependency_mut':2, 'crispr_dependency_expr':3, 'tissue':4}
    d_name_coord = {'Sequence':4, 'PPI':3, 'Crispr Dependency with Mutation':2, 'Crispr Dependency with Expression':1, 'Tissue':0}
    all_res = None
    for cancer in ['BRCA', 'CESC', 'COAD', 'LAML', 'LUAD', 'OV']:#
        res_loc = config.ROOT_DIR / 'results' / aim / ('single_cancer_' + cancer + '_set_imp_test2_ratio.csv')
        res = pd.read_csv(res_loc)
        if all_res is None:
            all_res = res.copy()
        else:
            all_res = pd.concat([all_res, res.copy()])
        for fold_id in [0, 1, 2, 3, 4, 5,6,7,8,9]:
            for data_id, data_name in enumerate(['seq_1024', 'ppi_ec', 'crispr_dependency_mut', 'crispr_dependency_expr', 'tissue']):
                found_vals = res[(res['fold']==fold_id)&(res['perm_ds']==data_name)]['val'].values
                df_new_row = [cancer, fold_id, data_id, data_name, np.mean(found_vals), np.std(found_vals) ]
                df_vals.append(df_new_row)

    all_res['data_id'] = all_res['perm_ds'].map(data_id_dict)
    all_res['data_name'] = all_res['perm_ds'].map(data_name_dict)
    all_res_grouped = all_res.groupby(['cancer', 'data_name'])['val'].agg(['mean', 'std']).reset_index()
    df = pd.DataFrame(df_vals, columns=['cancer', 'fold_id', 'data_id', 'data_name', 'mean', 'std'])
    df['long_data_name'] = df['data_name'].map(data_name_dict)
    phi = np.linspace(0, 2 * np.pi, 60)
    rgb_cycle = np.vstack((  # Three sinusoids
        .5 * (1. + np.cos(phi)),  # scaled to [0,1]
        .5 * (1. + np.cos(phi + 2 * np.pi / 3)),  # 120° phase shifted.
        .5 * (1. + np.cos(phi - 2 * np.pi / 3)))).T  # Shape = (60,3)
    if typ_name == 'type2':
        grid = sns.FacetGrid(all_res, col="fold", row='cancer',
                      row_order=['BRCA', 'CESC', 'COAD', 'LAML', 'LUAD', 'OV'],
                      col_order=[0,1,2,3,4])
    elif typ_name == 'type3':
        grid = sns.FacetGrid(all_res, col='cancer', col_wrap=3, aspect=1.5, height=2.2,
                      col_order=['BRCA', 'CESC', 'COAD', 'LAML', 'LUAD', 'OV'])


    #g = sns.FacetGrid(df, col='Fold', hue='Type', col_wrap=3, gridspec_kws={"wspace": 0.2})
    #g = grid.map_dataframe(myplot, "val", "perm_ds", alpha=.7)
    g = grid.map_dataframe(sns.pointplot, "val", "data_name", ci='sd', errwidth= 2, markers='d', join=False, palette='colorblind')
    #g.map_dataframe(errplot, "data_id", "mean", "std")

    #g.set_axis_labels("Data Ids", "Importance")
    # g.set(xticks=[1,2,3,4,5,6])
    # g.add_legend(loc='upper right')
    #plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    #for ax in g.axes.flat:
    #    for data_id in data_ids:
    #        ax.axvline(data_id, c=".2", ls="--", zorder=0)
    grid.set_axis_labels("Importance", "Datasets")
    out_loc = config.ROOT_DIR / ('results/images/set_detail_imp_pointplot_'+typ_name+'_'+aim+'_10.png')

    plt.savefig(out_loc, bbox_inches="tight")
    plt.savefig(out_loc, format='pdf', bbox_inches="tight")
    plt.tight_layout()
    plt.show()


def plt_feature_importance(aim='ELRRF'):
    res_loc = config.ROOT_DIR / 'results' / aim / 'single_cancer_feat_imp_test2.csv'
    res = pd.read_csv(res_loc)
    res = res.dropna(axis=1, how="any")
    #res = res.set_index(['cancer', 'datasets'])
    base_cols = [name[:-4] for name in res.columns.values if '_std' in name]
    #m_cols = [name[:-2] for name in base_cols]

    new_rows = []
    for cancer in res['cancer'].unique():
        for dataset in res['datasets'].unique():
            target_row = res[(res['cancer']==cancer)&(res['datasets']==dataset)]
            for feature in base_cols:
                if 'seq_1024' in feature:
                    corrected_feature_name = 'seq_1024'+'|'+feature[8:]
                elif 'crispr_dependency_expr' in feature:
                    corrected_feature_name = 'crispr_dependency_expr'+'|'+feature[22:]
                elif 'crispr_dependency_mut' in feature:
                    corrected_feature_name = 'crispr_dependency_mut'+'|'+feature[21:]
                elif 'ppi_ec' in feature:
                    corrected_feature_name = 'ppi_ec'+'|'+feature[6:]
                elif 'tissue' in feature:
                    corrected_feature_name = 'tissue'+'|'+feature[6:]
                else:
                    corrected_feature_name = feature
                new_row = [cancer, dataset, corrected_feature_name, target_row[feature+'_m'].values[0], target_row[feature+'_std'].values[0]]
                new_rows.append(new_row)

    def errplot(x, y, yerr, **kwargs):
        ax = plt.gca()
        data = kwargs.pop("data")
        data.plot(x=x, y=y, yerr=yerr, kind="bar", ax=ax, **kwargs)

    df = pd.DataFrame(new_rows, columns=['cancer', 'datasets', 'feature', 'mean', 'std'])
    for cancer in res['cancer'].unique():
        plt.clf()
        df_c = df[df['cancer']==cancer]
        df_c_sorted = df_c.sort_values(by=['mean', 'std']).drop(columns=['cancer', 'datasets'])
        df_c_sorted_only_top = df_c_sorted.iloc[-10:,:].set_index('feature')['mean']
        #g = sns.FacetGrid(df_c_sorted_only_top, col="cancer", col_wrap=4)
        #g.map_dataframe(errplot, "feature", "mean", "std")
        #g.set_axis_labels("Data Ids", "Importance")
        plt.style.use('ggplot')
        df_c_sorted_only_top.plot.barh(xerr=df_c_sorted.iloc[-10:,:].set_index('feature')['std'].values)
        plt.title(cancer+' Feature Importance')
        plt.ylabel('Feature')
        plt.xlabel('Importance')
        out_loc = config.ROOT_DIR / ('results/images/feature_imp_type2_' + aim + '_'+cancer+'.png')
        plt.savefig(out_loc, bbox_inches="tight")
        plt.show()
        print()

def plot_topn_genes_multi_cancer(cancer_list=['BRCA', 'LUAD', 'OV'], method='ELRRF', n=3):
    topn_dfs = []
    def topnplot(**kwargs):
        ax = plt.gca()
        data = kwargs.pop("data")
        data = data.reset_index()
        data.plot(x='mean', y='pair_name', kind="bar", ax=ax, **kwargs)

    for cancer in cancer_list:
        loc = 'results/'+method+'/models_test/seq_1024|ppi_ec|crispr_dependency_mut|crispr_dependency_expr|tissue_'+cancer+'_True_True_type2_10_undersample_train_test.pickle'
        top_pairs = ra.find_top_n_genes(loc, n)
        topn_pairs = top_pairs[:n].sort_values(by='mean', ascending=False)
        topn_pairs_rep = top_pairs.sort_values(by='mean', ascending=False).reset_index()
        print(top_pairs.head(10))
        first_true_pairs = topn_pairs_rep['labels'].eq(0).idxmax()
        total_true_pairs = sum(topn_pairs_rep['labels'][:n].eq(1))
        print(f'First {first_true_pairs} predicted SL pairs are true SL pairs for {cancer} in model {method}.')
        print(f'{total_true_pairs} of top {n} predicted SL pairs are actually true SL pairs for {cancer} in model {method}.')
        topn_pairs['cancer'] = cancer
        topn_dfs.append(topn_pairs)
    all_dfs = pd.concat(topn_dfs).reset_index()
    all_dfs['mean'] = all_dfs['mean'].astype(float)
    g = sns.FacetGrid(all_dfs, col='cancer', col_wrap=1, height=1, aspect=5, sharey=False)
    # g.add_legend()
    g.map_dataframe(sns.barplot,'mean', 'pair_name')
    #g_map = g.map_dataframe(topnplot)

    g.set_axis_labels(x_var='SL Score', y_var='Pairs')
    out_loc = config.ROOT_DIR / ('results/images/top'+str(n)+'_' + '_'.join(cancer_list) + '_'+method)
    #plt.savefig(out_loc, bbox_inches="tight")
    #plt.savefig(out_loc, format='pdf', bbox_inches="tight")

    #plt.show()


def plot_topn_genes_unknown_cancer(cancer='BRCA', method='ELRRF', features=['cancer', 'repair_cancer'], n=30, typ='type1', is_bottom=False):
    loc = 'results/'+method+'/models_test/seq_1024|ppi_ec|crispr_dependency_mut|crispr_dependency_expr|tissue_'+cancer+'_True_True_type2_10_undersample_train_test.pickle'
    top_pairs = ra.find_top_n_genes(loc, n)
    topn_pairs = top_pairs[:n].sort_values(by='mean', ascending=False)
    top_pairs['source']='Known'
    topn_pairs_rep = top_pairs.sort_values(by='mean', ascending=False).reset_index()
    first_true_pairs = topn_pairs_rep['labels'].eq(0).idxmax()
    total_true_pairs = sum(topn_pairs_rep['labels'][:n].eq(1))
    print(top_pairs.head(10))
    print(f'First {first_true_pairs} predicted SL pairs are true SL pairs for {cancer} in model {method}.')
    print(f'{total_true_pairs} of top {n} predicted SL pairs are actually true SL pairs for {cancer} in model {method}.')
    unknown_list = []
    for feature in features:
        loc_unknown = 'results/'+method+'/models_cross_ds/seq_1024|ppi_ec|crispr_dependency_mut|crispr_dependency_expr|tissue_'+cancer+'_train_unknown_'+feature+'_BRCA_True_True_type2_10_undersample_train.pickle'
        top_unknown_pairs = ra.find_top_n_genes(loc_unknown, n)
        topn_unknown_pairs = top_unknown_pairs[:n].sort_values(by='mean', ascending=False)
        top_unknown_pairs['source']='Unknown'
        print('\nUnknowns')
        print(top_unknown_pairs.head(10))
        unknown_list.append(top_unknown_pairs)
    unknown_list.append(top_pairs)
    all_dfs = pd.concat(unknown_list).reset_index()

    all_dfs['mean'] = all_dfs['mean'].astype(float)
    if typ=='type1':
        g = sns.FacetGrid(all_dfs, col='source', col_wrap=2, aspect=2, sharey=False)
        g.map_dataframe(sns.barplot,'mean', 'pair_name')
        g.set_axis_labels(x_var='SL Score', y_var='Pairs')
    if typ=='type2':
        if is_bottom:
            all_dfs = all_dfs.sort_values(by='mean', ascending=True)[:n]
        else:
            all_dfs = all_dfs.sort_values(by='mean', ascending=False)[:n]

        color_list = ['#1f77b4'  if s=='Known' else '#ff7f0e' for s in all_dfs['source'].values]
        g = sns.barplot(x='mean', y='pair_name', data=all_dfs, palette=color_list)
        g.set(xlabel="SL Score", ylabel = "Pairs")
        from matplotlib.lines import Line2D
        custom_lines = [Line2D([0], [0], color='#1f77b4', lw=6),
                        Line2D([0], [0], color='#ff7f0e', lw=6)]
        if is_bottom:
            g.legend(custom_lines, ['Negative', 'Unknown'])
        else:
            g.legend(custom_lines, ['Positive', 'Unknown'])
    # g.add_legend()
    #g.map_dataframe(sns.barplot,'mean', 'pair_name')
    #g_map = g.map_dataframe(topnplot)
    import matplotlib.ticker as ticker
    if is_bottom:
        g.set(xlim=(0.1, 0.6), xticks=[0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.50, 0.55, 0.60])
        out_loc = config.ROOT_DIR / ('results/images/bottom'+str(n)+'_unknown_' + cancer+'_'+ typ + '_'+method)
    else:
        g.set(xlim=(0.5, 0.9), xticks=[0.5, 0.55, 0.6, 0.65, 0.70, 0.75, 0.80])
        out_loc = config.ROOT_DIR / ('results/images/top'+str(n)+'_unknown_' + cancer+'_'+ typ + '_'+method)
    plt.savefig(out_loc, bbox_inches="tight")
    plt.savefig(out_loc, format='pdf', bbox_inches="tight")

    plt.show()


def plot_distr_genes_unknown_cancer(cancer='BRCA', method='ELRRF', features=['cancer', 'repair_cancer']):
    loc = 'results/'+method+'/models_test/seq_1024|ppi_ec|crispr_dependency_mut|crispr_dependency_expr|tissue_'+cancer+'_True_True_type2_10_undersample_train_test.pickle'
    top_pairs = ra.find_top_n_genes(loc, 0)
    top_pairs['source']='Known'
    top_pairs['status']='Positive'
    top_pairs.loc[top_pairs['labels']==0,'status'] = 'Negative'
    unknown_list = []
    for feature in features:
        loc_unknown = 'results/'+method+'/models_cross_ds/seq_1024|ppi_ec|crispr_dependency_mut|crispr_dependency_expr|tissue_'+cancer+'_train_unknown_'+feature+'_BRCA_True_True_type2_10_undersample_train.pickle'
        top_unknown_pairs = ra.find_top_n_genes(loc_unknown, 0)
        top_unknown_pairs['source']='Unknown'
        top_unknown_pairs['status']='Unknown'
        unknown_list.append(top_unknown_pairs)
    unknown_list.append(top_pairs)
    all_dfs = pd.concat(unknown_list).reset_index()

    all_dfs['mean'] = all_dfs['mean'].astype(float)
    all_dfs_sorted = all_dfs.sort_values(by='mean', ascending=False)
    import matplotlib
    # change font
    matplotlib.rcParams['font.sans-serif'] = "Arial"
    matplotlib.rcParams['font.family'] = "Arial"

    colors = ['#1f77b4', '#ff7f0e', '#DC050C']
    '''
    box_ax = sns.boxplot(x="status", y="mean", data=all_dfs, order=["Positive", "Unknown", "Negative"], palette=colors[:3])
    for patch in box_ax.artists:
       r, g, b, a = patch.get_facecolor()
       patch.set_facecolor((r, g, b, .1))
    '''
    ax2 = sns.violinplot(x="status", y="mean", data=all_dfs, order=["Positive", "Unknown", "Negative"],
                        palette=colors[:3], alpha=0.6)
    for violin in ax2.collections[::2]:
        violin.set_alpha(0.6)
    status_dict = OrderedDict()
    for i, statu in enumerate(["Positive", "Unknown", "Negative"]):
        vals = all_dfs[all_dfs['status'] == statu]['mean'].values
        median = np.median(vals)
        perc5 = np.percentile(vals,5)
        perc95 = np.percentile(vals,95)
        status_dict[statu]=(perc5, perc95)
        xloc = ax2.get_xmajorticklabels()[i]._x
        ax2.hlines(perc5, xmin=xloc-0.3, xmax=xloc+0.3, linestyles='dashed', color=colors[i])
        ax2.hlines(perc95, xmin=xloc-0.3, xmax=xloc+0.3, linestyles='dashed', color=colors[i])
        print(f'{statu} median: {median}')
    status_dict["All"] = (np.percentile(all_dfs['mean'].values,5), np.percentile(all_dfs['mean'].values,95))
    print(status_dict)


    #for patch in box_ax.artists:
    #    r, g, b, a = patch.get_facecolor()
    #    patch.set_facecolor((r, g, b, .1))

    # g.add_legend()
    #g.map_dataframe(sns.barplot,'mean', 'pair_name')
    #g_map = g.map_dataframe(topnplot)
    #import matplotlib.ticker as ticker
    plt.ylabel('SL Score', family='Arial', fontdict={'weight':'bold'})
    plt.xlabel('Pairs', family='Arial', fontdict={'weight':'bold'})
    out_loc = config.ROOT_DIR / ('results/images/distr_unknown_' + cancer+''+method)
    plt.savefig(out_loc, bbox_inches="tight")
    plt.savefig(out_loc, format='pdf', bbox_inches="tight")

    plt.show()


def search_specific_relations(all_genes={}, cancer= 'BRCA', method='ELRRF', features=['cancer', 'repair_cancer'], fams=['BRCA', 'WNT']):
    loc = 'results/'+method+'/models_test/seq_1024|ppi_ec|crispr_dependency_mut|crispr_dependency_expr|tissue_'+cancer+'_True_True_type2_10_undersample_train_test.pickle'
    train_labels_loc = config.DATA_DIR / 'labels' / 'train_pairs.csv'
    train_labels = pd.read_csv(train_labels_loc)
    train_labels=train_labels[train_labels['cancer']==cancer]
    test_labels_loc = config.DATA_DIR / 'labels' / 'test_pairs.csv'
    test_labels = pd.read_csv(test_labels_loc)
    test_labels=test_labels[test_labels['cancer']==cancer]
    all_known_dict = pd.concat([train_labels, test_labels]).set_index(['gene1', 'gene2'])['class'].to_dict()
    top_pairs = ra.find_top_n_genes(loc)
    top_pairs['source']='Known'
    unknown_list = []
    for feature in features:
        loc_unknown = 'results/'+method+'/models_cross_ds/seq_1024|ppi_ec|crispr_dependency_mut|crispr_dependency_expr|tissue_'+cancer+'_train_'+feature+'_'+cancer+'_True_True_type2_10_undersample_train.pickle'
        top_unknown_pairs = ra.find_top_n_genes(loc_unknown)
        if 'extra' in feature:
            top_unknown_pairs['source']='Known'
            for idd, row_ex in top_unknown_pairs.iterrows():
                g1, g2 = idd.split('|')
                top_unknown_pairs.loc[idd, 'labels'] = all_known_dict[g1,g2]
        else:
            top_unknown_pairs['source'] = 'Unknown'
        unknown_list.append(top_unknown_pairs)
    unknown_list.append(top_pairs)
    all_dfs = pd.concat(unknown_list).reset_index()

    all_dfs['mean'] = all_dfs['mean'].astype(float)
    all_dfs = all_dfs.sort_values(by='mean', ascending=False)
    pairs= [ ]
    for id1, id2_list in all_genes.items():
        for id2 in id2_list:
            pair_row = min(id1, id2)+'|'+ max(id1, id2)
            pairs.append(pair_row)


    #color_list = ['#1f77b4'  if s[1]['source']=='Known' and s[1]['label']==1 elif s[1]['source']=='Known' and s[1]['label']==0 else '#ff7f0e' for s in all_dfs.iterrows()]
    color_list = []
    only_pairs = all_dfs.set_index('pair_name').loc[pairs].reset_index()
    for idx, row in only_pairs.iterrows():
        if np.isnan(row['labels']):
            gene1, gene2 = row['pair_name'].split('|')
            clss = train_labels[(train_labels['gene1'] == gene1) & (train_labels['gene2'] == gene2)]['class'].values
            if len(clss)<1:
                clss = test_labels[(test_labels['gene1'] == gene1) & (test_labels['gene2'] == gene2)]['class'].values
            only_pairs.loc[idx, ['mean', 'labels','source']] = [clss, clss, 'Known']
            row[['mean', 'labels','source']] =[clss, clss, 'Known']
        if row['source']=='Known' and row['labels']==1:
            color_list.append('#1f77b4')
        elif row['source']=='Known' and row['labels']==0:
            color_list.append('#DC050C')
        else:
            color_list.append('#ff7f0e')

    import matplotlib
    # change font
    matplotlib.rcParams['font.sans-serif'] = "Arial"
    matplotlib.rcParams['font.family'] = "Arial"
    g = sns.barplot(x='mean', y='pair_name', data=only_pairs, palette=color_list)
    g.vlines(0.75681, ymin=g.dataLim.ymin, ymax=g.dataLim.ymax, linestyles='dashed', color='#1f77b4')
    g.vlines(0.52965, ymin=g.dataLim.ymin, ymax=g.dataLim.ymax, linestyles='dashed', color='#DC050C')
    g.vlines(0.58663, ymin=g.dataLim.ymin, ymax=g.dataLim.ymax, linestyles='dashed', color='#ff7f0e')
    #g.set(xlabel="SL Score", ylabel = "Pairs")

    from matplotlib.lines import Line2D
    custom_lines = [Line2D([0], [0], color='#1f77b4', lw=6),
                    Line2D([0], [0], color='#DC050C', lw=6),
                    Line2D([0], [0], color='#ff7f0e', lw=6)]
    g.legend(custom_lines, ['Positive', 'Negative', 'Unknown'])
    # g.add_legend()
    #g.map_dataframe(sns.barplot,'mean', 'pair_name')
    #g_map = g.map_dataframe(topnplot)
    import matplotlib.ticker as ticker
    g.set(xlim=(0.0, 1), xticks=np.arange(0,1.01,0.1))

    plt.ylabel('Pairs', family='Arial', fontdict={'weight':'bold'})
    plt.xlabel('SL Score', family='Arial', fontdict={'weight':'bold'})
    out_loc = config.ROOT_DIR / ('results/images/'+cancer+'_' +'|'.join(features)+'_'+'|'.join(fams)+'_'+method)
    plt.savefig(out_loc, bbox_inches="tight")
    plt.savefig(str(out_loc)+'.pdf', format='pdf', bbox_inches="tight")

    plt.show()



def plot_topn_genes(cancer='BRCA', method='ELRRF', n=3):
    loc = 'results/'+method+'/models_test/seq_1024|ppi_ec|crispr_dependency_mut|crispr_dependency_expr|tissue_'+cancer+'_True_True_type2_10_undersample_train_test.pickle'
    top_pairs = ra.find_top_n_genes(loc, n)
    topn_pairs = top_pairs[:n].sort_values(by='mean')
    topn_pairs_rep = top_pairs.sort_values(by='mean', ascending=False).reset_index()
    first_true_pairs = topn_pairs_rep['labels'].eq(0).idxmax()
    total_true_pairs = sum(topn_pairs_rep['labels'][:n].eq(1))
    print(f'First {first_true_pairs} predicted SL pairs are true SL pairs for {cancer} in model {method}.')
    print(f'{total_true_pairs} of top {n} predicted SL pairs are actually true SL pairs for {cancer} in model {method}.')
    topn_pairs['mean'].plot.barh()
    plt.xlabel('SL Score')
    plt.ylabel('Pairs')
    out_loc = config.ROOT_DIR / ('results/images/top'+str(n)+'_' + cancer + '_'+method+'.png')
    #plt.savefig(out_loc, bbox_inches="tight")
    #plt.show()


    print()
    # g.set(xticks=[1,2,3,4,5,6])
    # g.add_legend(loc='upper right')
    #plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    #for ax in g.axes.flat:
    #    for data_id in data_ids:
    #        ax.axvline(data_id, c=".2", ls="--", zorder=0)

def plt_crosscancer_hmap(model='ELRRF', score='AUPRC'):
    cc_loc = 'results/'+model+'/cross_cancer_train_train_test_test.csv'
    cc_table = ra.report_cross_cancer_res(res_id=14, loc=cc_loc, selections={'balance_strat': ['undersample_train_test'],
                                                            'comb_type': ['type2'], 'threshold': [0.5],
                                                            'process': [True], 'n_split': [10],
                                                            'grid_search': ['True']}, score=score)
    single_loc = 'results/ELRRF/single_cancer_test.csv'
    single_table = ra.report_cancer_res(res_id=14, loc=single_loc, selections={'balance_strat': ['undersample_train_test'],
                                                     'comb_type': ['type2'], 'threshold': [0.5],'process': [True], 'n_split': [10],
                                                     'grid_search': ['True']})  # ,chosen_th=chosen_th_train)
    for idx, row in single_table.iterrows():
        cc_table.loc[row['cancer'], row['cancer']] = row[score].split(' ')[0]
    cc_table = cc_table.fillna(0).astype(float)
    ax = sns.heatmap(cc_table, cmap="Greens", annot=True, annot_kws={"size": 7})
    ax.set(xlabel='Test', ylabel='Train')
    out_loc = config.ROOT_DIR / ('results/images/crosscancer' + model + '_' + score + '_10split.png')
    plt.savefig(out_loc, bbox_inches="tight")
    plt.savefig(str(out_loc)[:-4]+'.pdf', format='pdf', bbox_inches="tight")
    plt.show()
    print()

def main():
    selections = {'balance_strat': ['undersample_train_test'],
                  'comb_type': ['type2'], 'process': [True],
                  'grid_search': ['True']}
    score='AUPRC'
    us = 'trteus'
    #for cancer in []:#cancer_list:
    #    try:
    #        plot_box_with_points(cancer=cancer, aim=score, out='single')
    #    except:
    #        pass
    cancer_list = ['BRCA_isle_dsl', 'LUAD_dsl_isle', 'LUAD_dsl_exp2sl',
                   'BRCA_dsl_isle', 'LUAD_isle_dsl', 'LUAD_exp2sl_dsl',]
    cancer_list = ['BRCA', 'CESC', 'COAD', 'KIRC', 'LAML', 'LUAD', 'OV', 'SKCM']
    #plot_facet_box_with_points(cancer_list=cancer_list, aim=score, out='dho2', one_colm=True)
    #trauc_vs_contr()
    #plt_dataset_detail_importance(aim='ELRRF', typ_name='type3')
    #plt_crosscancer_hmap(model='ELRRF', score='AUPRC')
    #for cancer in ['LUAD']:
    #    plot_topn_genes(cancer=cancer, method='ELRRF', n=50)
    cancer_list=['BRCA', 'LUAD', 'OV']
    #plot_topn_genes_multi_cancer(cancer_list=['LUAD'], method='ELRRF', n=10)
    #plot_topn_genes_unknown_cancer(cancer='BRCA', method='ELRRF', features=['cancer', 'repair_cancer'], n=10, typ='type2', is_bottom=True)
    #plot_distr_genes_unknown_cancer(cancer='BRCA', method='ELRRF', features=['cancer', 'repair_cancer'])

    all_genes_bw = {'BRCA1': ['WNT3A', 'WNT7A', 'WNT6', 'WNT16', 'IHH', 'SHH', 'DHH', 'PTCH1',
                           'FGF1', 'FGF2', 'FGF3', 'FGF4', 'FGF5', 'FGF6', 'FGF7', 'FGF8', 'FGF9', 'FGF10', 'FGF11',
                           'FGF12', 'FGF13',
                           'FGF14', 'FGF16', 'FGF17', 'FGF18', 'FGF19', 'FGF20', 'FGF21', 'FGF22', 'FGF23'],
                 'BRCA2': ['WNT3A', 'WNT7A', 'WNT6', 'WNT16', 'IHH', 'SHH', 'DHH', 'PTCH1',
                           'FGF1', 'FGF2', 'FGF3', 'FGF4', 'FGF5', 'FGF6', 'FGF7', 'FGF8', 'FGF9', 'FGF10', 'FGF11',
                           'FGF12', 'FGF13',
                           'FGF14', 'FGF16', 'FGF17', 'FGF18', 'FGF19', 'FGF20', 'FGF21', 'FGF22', 'FGF23']}

    all_genes_bw = {'BRCA1': ['WNT3A', 'WNT7A', 'WNT6', 'WNT16'],
                    'BRCA2': ['WNT3A', 'WNT7A', 'WNT6', 'WNT16']}
    all_genes_bf = {'BRCA1': ['FGF1', 'FGF2', 'FGF3', 'FGF4', 'FGF5', 'FGF6', 'FGF7', 'FGF8', 'FGF9', 'FGF10', 'FGF11',
                           'FGF12', 'FGF13', 'FGF14', 'FGF16', 'FGF17', 'FGF18', 'FGF19', 'FGF20', 'FGF21', 'FGF22', 'FGF23'],
                    'BRCA2': ['FGF1', 'FGF2', 'FGF3', 'FGF4', 'FGF5', 'FGF6', 'FGF7', 'FGF8', 'FGF9', 'FGF10', 'FGF11',
                           'FGF12', 'FGF13', 'FGF14', 'FGF16', 'FGF17', 'FGF18', 'FGF19', 'FGF20', 'FGF21', 'FGF22', 'FGF23']}
    all_genes_bh = {'BRCA1': ['HHIP', 'IHH', 'SHH', 'DHH', 'PTCH1',],
                    'BRCA2': ['HHIP', 'IHH', 'SHH', 'DHH', 'PTCH1',]}

    all_genes_l = {'KRAS': ['IL6', 'IL11', 'IL27', 'IL31', 'CNTF', 'LIF', 'OSM',
                          'THRA', 'THRB', 'RARA', 'RARB', 'RARG', 'PPARA', 'PPARD', 'PPARG', 'NR1D1', 'NR1D2', 'RORA',
                          'RORB', 'RORC', 'NR1H4', 'NR1H5P', 'NR1H3', 'NR1H2', 'VDR', 'NR1I2', 'NR1I3',
                          'POLL', 'POLB', 'POLM', 'TENT4A', 'DNTT',
                          'MRPL1', 'MRPL2', 'MRPL3', 'MRPL4', 'MRPL9', 'MRPL10', 'MRPL11', 'MRPL12', 'MRPL13', 'MRPL14',
                          'MRPL15', 'MRPL16', 'MRPL17', 'MRPL18', 'MRPL19', 'MRPL20', 'MRPL21', 'MRPL22', 'MRPL23',
                          'MRPL24', 'MRPL27', 'MRPL28', 'MRPL30', 'MRPL32', 'MRPL33', 'MRPL34', 'MRPL35', 'MRPL36',
                          'MRPL37', 'MRPL38', 'MRPL39', 'MRPL40', 'MRPL41', 'MRPL42', 'MRPL43', 'MRPL44', 'MRPL45',
                          'MRPL46', 'MRPL47', 'MRPL48', 'MRPL49', 'MRPL50', 'MRPL51', 'MRPL52', 'MRPL53', 'MRPL54',
                          'MRPL55', 'MRPL57', 'MRPL58'
                          ]}


    all_genes_lil = {'KRAS': ['IL6', 'IL11', 'IL27', 'IL31', 'CNTF', 'LIF', 'OSM']}
    all_genes_lnr1 = {'KRAS': ['THRA', 'THRB', 'RARA', 'RARB', 'RARG', 'PPARA', 'PPARD', 'PPARG', 'NR1D1', 'NR1D2', 'RORA',
                          'RORB', 'RORC', 'NR1H4', 'NR1H5P', 'NR1H3', 'NR1H2', 'VDR', 'NR1I2', 'NR1I3',]}
    all_genes_lmrpl = {'KRAS': ['MRPL1', 'MRPL2', 'MRPL3', 'MRPL4', 'MRPL9', 'MRPL10', 'MRPL11', 'MRPL12', 'MRPL13', 'MRPL14',
                          'MRPL15', 'MRPL16', 'MRPL17', 'MRPL18', 'MRPL19', 'MRPL20', 'MRPL21', 'MRPL22', 'MRPL23',
                          'MRPL24', 'MRPL27', 'MRPL28', 'MRPL30', 'MRPL32', 'MRPL33', 'MRPL34', 'MRPL35', 'MRPL36',
                          'MRPL37', 'MRPL38', 'MRPL39', 'MRPL40', 'MRPL41', 'MRPL42', 'MRPL43', 'MRPL44', 'MRPL45',
                          'MRPL46', 'MRPL47', 'MRPL48', 'MRPL49', 'MRPL50', 'MRPL51', 'MRPL52', 'MRPL53', 'MRPL54',
                          'MRPL55', 'MRPL57', 'MRPL58']}
    all_genes_lpolx = {'KRAS': ['POLL', 'POLB', 'POLM', 'TENT4A', 'DNTT']}
    cancer = 'BRCA'
    if cancer=='BRCA':
        search_specific_relations(all_genes_bh, cancer='BRCA', method='ELRRF',
                              features=['unknown_cancer','unknown_repair_cancer','unknown_families','unknown_families_extra'],
                              fams= ['BRCA','HH'])
    if cancer=='LUAD':
        search_specific_relations(all_genes_lnr1, cancer='LUAD', method='ELRRF', features=['negative_families', 'negative_families_extra'], fams= ['KRAS','NR1'])
    #plt_dataset_importance(aim='elrrf')
    #plot_res(loc='results/ELGBDT/single_cancer_test.csv', score=score, selections = selections,
    #         us=us, out_name ='test_dim_c_t_trteus_'+score+'.png')
    #plot_single_for_cancer2(loc='results/elrrf/single_cancer_validation.csv', score=score, selections = selections,
    #                        baselines=cancer_baselines, us=us, out_name ='val_ELRRF_'+us+'_'+score+'.png')
    #plot_for_cancer2(loc='results/elrrf/single_cancer_validation.csv', score=score, selections = selections,
    #                        baselines=cancer_baselines, us=us, out_name ='val_single_'+us+'_'+score+'.png', res_id=14)
    #plot_single_with_onehot(loc='results/elrrf/single_cancer_validation.csv', score=score, selections = selections,
    #                        baselines=cancer_baselines, us=us, out_name ='val_single_'+us+'_'+score+'_onehot.png',
    #                        res_id=14)
    #plot_single_with_colm(loc='results/elrrf/single_cancer_validation.csv', score=score, selections = selections,
    #                        baselines=cancer_baselines, us=us, out_name ='val_single_'+us+'_'+score+'_all_comp.png')
    #tsne_gtex_pats()
    #cluster_pats('tissue')


if __name__ == '__main__':
    main()