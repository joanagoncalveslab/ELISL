"""
Visualizing RNA sequencing data using tmap.

Data Source:
https://gdc.cancer.gov/about-data/publications/pancanatlas
"""
from typing import List
import os
import sys

path2this = os.path.dirname(os.path.abspath(__file__)).split('/')
project_path = ''
for i, folder in enumerate(path2this):
    if folder.lower() == 'elisl':
        project_path = '/'.join(path2this[:i + 1])
sys.path.insert(0, project_path)
from src.lib.sutils import *
import numpy as np
import pandas as pd
from timeit import default_timer as timer
from src import config
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.manifold import TSNE
import umap
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import os
import json
import lightgbm as lgb
from src.lib import data_locations as dl
from src import data_functions as dfnc

parser = argparse.ArgumentParser(description='RFE arguments')
parser.add_argument('--visualize', '-v', metavar='the-visual-tool', dest='visual_tool', type=str,
                    help='Choose visual tool', default='umap')
parser.add_argument('--nneighbors', '-nn', metavar='the-n-neighbors', dest='nneighbors', type=int,
                    help='Choose the max iteration', default=100)
parser.add_argument('--mindist', '-md', metavar='the-min-dist', dest='mindist', type=float,
                    help='Choose the min-dist', default=0.001)
parser.add_argument('--perplexity', '-p', metavar='the-perplexity', dest='perplexity', type=float,
                    help='Choose the perplexity of t-sne', default=30)
args = parser.parse_args()
print(f'Running args:{args}')

sftp_loc = {'SKCM': 'AI5Q41L~.PIC',
            'KIRC': 'N3AV5QK~.PIC',
            'OV': 'WK8TDL7~.PIC',
            'CESC': '8V12SDV~.PIC',
            'LUAD': '14AJWKV~.PIC',
            'BRCA': '51DSCSH~.PIC',
            'LAML': 'JV43PPC~.PIC',
            'COAD': 'H45REOR~.PIC'}


def rfe_ds_cancer(ds_aims=['seq_1024'], cancer_aims=['BRCA'], chosen_gene=None):
    arguments = ''
    if args.visual_tool == 'tsne':
        arguments = f'_perp{args.perplexity}'
    elif args.visual_tool == 'umap':
        arguments = f'_nn={args.nneighbors}_md={args.mindist}'
    loc_dict = dl.get_all_locs()
    for ds_name in ds_aims:
        tr_data = pd.read_csv(loc_dict[f'train_{ds_name}_data_loc'])
        tr_data = tr_data.fillna(0)
        for cancer in cancer_aims:
            processed_data_tr = dfnc.prepare_cancer_dataset(tr_data, cancer=cancer)
            train_ind_loc = config.DATA_DIR / 'feature_sets' / ('train_test_' + cancer + '_' + str(10) + '.json')
            with open(train_ind_loc, 'r') as fp:
                fold_dict = json.load(fp)
            for fold_id, items in fold_dict.items():
                if int(fold_id) in [1, 3, 6, 8, 9]:
                    continue
                print(f'Fold: {fold_id}')
                png_loc = config.RESULT_DIR / 'sample_vis' / \
                          f'RFE_{args.visual_tool}_{cancer}{fold_id}_train_{ds_name}_{chosen_gene}{arguments}.png'
                pdf_loc = config.RESULT_DIR / 'sample_vis' / \
                          f'RFE_{args.visual_tool}_{cancer}{fold_id}_train_{ds_name}_{chosen_gene}{arguments}.pdf'
                ensure_file_dir(png_loc)
                if os.path.exists(png_loc):
                    continue
                plt.clf()

                y = processed_data_tr['class'].values.copy()[items['train']]
                raw_y = processed_data_tr['class'].values.copy()[items['train']]
                labels = processed_data_tr['pair_name'].values.copy()[items['train']]
                if chosen_gene is not None:
                    chosen_gene_pairs = [chosen_gene in label for label in labels]
                    y[chosen_gene_pairs & (y == 1)] = 2
                    y[chosen_gene_pairs & (y == 0)] = 3
                all_x = processed_data_tr.drop(columns=['pair_name', 'class']).values.copy()[items['train']]

                scaler = StandardScaler()
                x = scaler.fit_transform(all_x)

                print("Running RFE ...")
                start = timer()
                clf = lgb.LGBMClassifier(boosting_type='rf', num_leaves=165, max_depth=- 1, learning_rate=0.1,
                                         n_estimators=400,
                                         subsample_for_bin=200000, objective=None, class_weight=None,
                                         min_split_gain=0.0,
                                         min_child_weight=5, min_child_samples=10, subsample=0.632,
                                         subsample_freq=1,
                                         colsample_bytree=0.8, reg_alpha=0.0, reg_lambda=5.0, random_state=124,
                                         n_jobs=- 1, is_unbalance=False)
                clf.fit(x, raw_y)
                leaves = clf.predict(x, pred_leaf=True)
                onehot_x = OneHotEncoder().fit_transform(leaves)
                S = (onehot_x * onehot_x.transpose()).todense()
                # lastly, we normalize and subtract from 1, to get dissimilarities
                D = 1 - S / S.max()
                if args.visual_tool == 'tsne':
                    x_transformed = TSNE(metric='precomputed', perplexity=args.perplexity).fit_transform(D)
                elif args.visual_tool == 'umap':
                    x_transformed = umap.UMAP(metric='precomputed', n_neighbors=args.nneighbors,
                                              min_dist=args.mindist).fit_transform(D)
                algo_time = timer() - start
                print(f"RFE: {algo_time}")

                negatives = y == 0
                positives = y == 1
                unknowns = y == -1
                negatives_chosen = y == 3
                positives_chosen = y == 2

                plt.scatter(x_transformed[negatives, 0], x_transformed[negatives, 1], c="#EE99AA",
                            s=0.25, label='Negative')
                plt.scatter(x_transformed[positives, 0], x_transformed[positives, 1], c="#6699CC",
                            s=0.25, label='Positive')
                if chosen_gene is not None:
                    plt.scatter(x_transformed[negatives_chosen, 0], x_transformed[negatives_chosen, 1], c="#994455",
                                s=0.25, label=f'Negative_{chosen_gene}')
                    plt.scatter(x_transformed[positives_chosen, 0], x_transformed[positives_chosen, 1], c="#004488",
                                s=0.25, label=f'Positive_{chosen_gene}')

                # ax[int(fold_id), ds_id].legend()
                plt.xticks([])
                plt.yticks([])
                plt.legend()
                plt.title(
                    f'RFE({cancer}-{fold_id}{all_x.shape}-{ds_name} in {format(algo_time, ".2f")} sec')

                plt.savefig(pdf_loc, type='pdf', dpi=300,
                            bbox_inches='tight')
                plt.savefig(png_loc, type='png', dpi=300,
                            bbox_inches='tight')
                plt.show()


def rfe_semi_ds_cancer(ds_aims=['seq_1024'], cancer_aims=['BRCA'], chosen_gene=None, unk_folds=[1, 2, 3]):
    arguments = ''
    if args.visual_tool == 'tsne':
        arguments = f'_perp{args.perplexity}'
    elif args.visual_tool == 'umap':
        arguments = f'_nn={args.nneighbors}_md={args.mindist}'
    loc_dict = dl.get_all_locs()
    for ds_name in ds_aims:
        tr_data = pd.read_csv(loc_dict[f'train_{ds_name}_data_loc'])
        tr_data = tr_data.fillna(0)
        for cancer in cancer_aims:
            processed_data_tr = dfnc.prepare_cancer_dataset(tr_data, cancer=cancer)
            train_ind_loc = config.DATA_DIR / 'feature_sets' / ('train_test_' + cancer + '_' + str(10) + '.json')
            with open(train_ind_loc, 'r') as fp:
                fold_dict = json.load(fp)
            for fold_id, items in fold_dict.items():
                if int(fold_id) in [1, 3, 6, 8, 9]:
                    continue
                print(f'Fold: {fold_id}')
                unk_data = pd.read_csv(loc_dict[f'unk_{ds_name}_{cancer}_data_loc'])
                unk_data = unk_data.fillna(0)
                for unk_fold in unk_folds:
                    png_loc = config.RESULT_DIR / 'sample_vis' / \
                              f'RFE_{args.visual_tool}_{cancer}{fold_id}_train_{ds_name}_{chosen_gene}_unk{unk_fold}{arguments}.png'
                    pdf_loc = config.RESULT_DIR / 'sample_vis' / \
                              f'RFE_{args.visual_tool}_{cancer}{fold_id}_train_{ds_name}_{chosen_gene}_unk{unk_fold}{arguments}.pdf'
                    ensure_file_dir(png_loc)
                    if os.path.exists(png_loc):
                        continue
                    plt.clf()

                    y = processed_data_tr['class'].values.copy()[items['train']]
                    raw_y = processed_data_tr['class'].values.copy()[items['train']]
                    labels = processed_data_tr['pair_name'].values.copy()[items['train']]
                    if chosen_gene is not None:
                        chosen_gene_pairs = [chosen_gene in label for label in labels]
                        y[chosen_gene_pairs & (y == 1)] = 2
                        y[chosen_gene_pairs & (y == 0)] = 3
                    all_x = processed_data_tr.drop(columns=['pair_name', 'class']).values.copy()[items['train']]

                    processed_data_unk = dfnc.prepare_cancer_dataset(unk_data, cancer=cancer)
                    if unk_fold != -1:
                        processed_data_unk = processed_data_unk.sample(n=len(y), replace=False,
                                                                       random_state=unk_fold)
                    all_x_unk = processed_data_unk.drop(columns=['pair_name', 'class']).values.copy()
                    y_unk = processed_data_unk['class'].values.copy()
                    raw_y_unk = processed_data_unk['class'].values.copy()

                    scaler = StandardScaler()
                    x = scaler.fit_transform(all_x)
                    x_unk = scaler.transform(all_x_unk)

                    print("Running UMAP ...")
                    start = timer()
                    clf = lgb.LGBMClassifier(boosting_type='rf', num_leaves=165, max_depth=- 1, learning_rate=0.1,
                                             n_estimators=400,
                                             subsample_for_bin=200000, objective=None, class_weight=None,
                                             min_split_gain=0.0,
                                             min_child_weight=5, min_child_samples=10, subsample=0.632,
                                             subsample_freq=1,
                                             colsample_bytree=0.8, reg_alpha=0.0, reg_lambda=5.0, random_state=124,
                                             n_jobs=- 1, is_unbalance=False)
                    clf.fit(x, raw_y)
                    leaves = clf.predict(x, pred_leaf=True)
                    leaves_unk = clf.predict(x_unk, pred_leaf=True)
                    onehotencoder = OneHotEncoder()
                    all_leaves = np.concatenate((leaves, leaves_unk))
                    onehot_x_all = onehotencoder.fit_transform(all_leaves)
                    # onehot_x_all = onehotencoder.transform(all_leaves)
                    # onehot_x = onehotencoder.transform(leaves)
                    # onehot_x_unk = onehotencoder.transform(leaves_unk)
                    S = (onehot_x_all * onehot_x_all.transpose()).todense()
                    # lastly, we normalize and subtract from 1, to get dissimilarities
                    D = 1 - S / S.max()
                    if args.visual_tool == 'tsne':
                        x_all_transformed = TSNE(metric='precomputed', perplexity=args.perplexity).fit_transform(D)
                    elif args.visual_tool == 'umap':
                        x_all_transformed = umap.UMAP(metric='precomputed', n_neighbors=args.nneighbors,
                                                      min_dist=args.mindist).fit_transform(D)
                    x_transformed = x_all_transformed[:len(leaves), :]
                    x_unk_transformed = x_all_transformed[len(leaves):, :]
                    algo_time = timer() - start
                    print(f"RFE: {algo_time}")

                    negatives = y == 0
                    positives = y == 1
                    unknowns = y == -1
                    negatives_chosen = y == 3
                    positives_chosen = y == 2
                    plt.scatter(x_transformed[negatives, 0], x_transformed[negatives, 1], c="#EE99AA",
                                s=0.25, label='Negative')
                    plt.scatter(x_transformed[positives, 0], x_transformed[positives, 1], c="#6699CC",
                                s=0.25, label='Positive')
                    if chosen_gene is not None:
                        plt.scatter(x_transformed[negatives_chosen, 0], x_transformed[negatives_chosen, 1], c="#994455",
                                    s=0.25, label=f'Negative_{chosen_gene}')
                        plt.scatter(x_transformed[positives_chosen, 0], x_transformed[positives_chosen, 1], c="#004488",
                                    s=0.25, label=f'Positive_{chosen_gene}')

                    plt.scatter(x_unk_transformed[:, 0], x_unk_transformed[:, 1], c="#FF8000",
                                s=0.25, label='Unknown')

                    # ax[int(fold_id), ds_id].legend()
                    plt.xticks([])
                    plt.yticks([])
                    plt.legend()
                    plt.title(
                        f'RFE({cancer}-{fold_id}{all_x.shape}-{ds_name}, Unk-{unk_fold} in {format(algo_time, ".2f")} sec')

                    plt.savefig(pdf_loc, type='pdf', dpi=300,
                                bbox_inches='tight')
                    plt.savefig(png_loc, type='png', dpi=300,
                                bbox_inches='tight')
                    #plt.show()


def rfe_all_cancer(ds_aims=['seq_1024'], cancer_aims=['BRCA'], chosen_gene=None):
    s=0.9
    r_state=124
    all_ds = '|'.join(ds_aims)
    arguments = ''
    if args.visual_tool == 'tsne':
        arguments = f'_perp{args.perplexity}_rs{r_state}'
    elif args.visual_tool == 'umap':
        arguments = f'_nn={args.nneighbors}_md={args.mindist}_rs{r_state}'
    loc_dict = dl.get_all_locs()
    for cancer in cancer_aims:
        model_loc = config.RESULT_DIR / 'ELRRF' / 'models_test' / f'{all_ds}_{cancer}_True_True_type2_10_undersample_train_test.pickle'
        #model_loc = f'/Volumes/tudelft/ELISL/results/ELRRF/models_test/{sftp_loc[cancer]}'
        prev_model = dfnc.load_pickle(model_loc)
        train_ind_loc = config.DATA_DIR / 'feature_sets' / f'train_test_{cancer}_{10}.json'
        with open(train_ind_loc, 'r') as fp:
            fold_dict = json.load(fp)
        cancer_data_dict = {}
        final_tr_data = None
        for ds_name in ds_aims:
            tr_data = pd.read_csv(loc_dict[f'train_{ds_name}_data_loc'])
            tr_data = tr_data.fillna(0)
            processed_data_tr = dfnc.prepare_cancer_dataset(tr_data, cancer=cancer)
            processed_data_tr.append(processed_data_tr)
            cancer_data_dict[ds_name] = processed_data_tr
            if final_tr_data is None:
                final_tr_data = processed_data_tr.copy()
            else:
                final_tr_data = pd.merge(final_tr_data, processed_data_tr, how='inner', on=['pair_name', 'class'])
        cancer_data_dict['&'.join(ds_aims)] = final_tr_data
        for fold_id, items in fold_dict.items():
            if int(fold_id) in [1, 3, 6, 8, 9]:
                continue
            print(f'Fold: {fold_id}')
            png_loc = config.RESULT_DIR / 'sample_vis' / \
                      f'RFE_{args.visual_tool}_{cancer}{fold_id}_train_all_ds_{chosen_gene}{arguments}.png'
            pdf_loc = config.RESULT_DIR / 'sample_vis' / \
                      f'RFE_{args.visual_tool}_{cancer}{fold_id}_train_all_ds_{chosen_gene}{arguments}.pdf'
            ensure_file_dir(png_loc)
            if os.path.exists(png_loc):
                continue

            best_params = prev_model['best_params'][fold_id]
            plt.clf()
            final_D = None
            for ds_name, ds_data in cancer_data_dict.items():
                y = ds_data['class'].values.copy()[items['train']]
                raw_y = ds_data['class'].values.copy()[items['train']]
                labels = ds_data['pair_name'].values.copy()[items['train']]
                if chosen_gene is not None:
                    chosen_gene_pairs = [chosen_gene in label for label in labels]
                    y[chosen_gene_pairs & (y == 1)] = 2
                    y[chosen_gene_pairs & (y == 0)] = 3
                all_x = ds_data.drop(columns=['pair_name', 'class']).values.copy()[items['train']]

                scaler = StandardScaler()
                x = scaler.fit_transform(all_x)

                print("Running RFE ...")
                start = timer()
                clf = lgb.LGBMClassifier(boosting_type='rf', num_leaves=165, max_depth=best_params['max_depth'],
                                         learning_rate=0.1,
                                         n_estimators=best_params['n_estimators'],
                                         subsample_for_bin=200000, objective=None, class_weight=None,
                                         min_split_gain=0.0,
                                         min_child_weight=5, min_child_samples=best_params['min_child_samples'],
                                         subsample=best_params['subsample'],
                                         subsample_freq=1,
                                         colsample_bytree=best_params['colsample_bytree'], reg_alpha=0.0,
                                         reg_lambda=best_params['reg_lambda'], random_state=124,
                                         n_jobs=- 1, is_unbalance=False)
                clf.fit(x, raw_y)
                leaves = clf.predict(x, pred_leaf=True)
                onehot_x = OneHotEncoder().fit_transform(leaves)
                S = (onehot_x * onehot_x.transpose()).todense()
                # lastly, we normalize and subtract from 1, to get dissimilarities
                D = 1 - S / S.max()
                contr = prev_model['params']['contr'][fold_id][ds_name] / sum(
                    prev_model['params']['contr'][fold_id].values())
                if final_D is None:
                    final_D = D.copy() * contr
                else:
                    final_D = D.copy() * contr + final_D.copy()
            if args.visual_tool == 'tsne':
                x_transformed = TSNE(metric='precomputed', perplexity=args.perplexity, random_state=r_state).fit_transform(final_D)
            elif args.visual_tool == 'umap':
                x_transformed = umap.UMAP(metric='precomputed', n_neighbors=args.nneighbors,
                                          min_dist=args.mindist, random_state=r_state).fit_transform(final_D)
            algo_time = timer() - start
            print(f"RFE: {algo_time}")

            negatives = y == 0
            positives = y == 1
            unknowns = y == -1
            negatives_chosen = y == 3
            positives_chosen = y == 2

            plt.scatter(x_transformed[negatives, 0], x_transformed[negatives, 1], c="#994455",
                        s=s, label='Negative')
            plt.scatter(x_transformed[positives, 0], x_transformed[positives, 1], c="#004488",
                        s=s, label='Positive')
            if chosen_gene is not None:
                plt.scatter(x_transformed[negatives_chosen, 0], x_transformed[negatives_chosen, 1], c="#994455",
                            s=s, label=f'Negative_{chosen_gene}')
                plt.scatter(x_transformed[positives_chosen, 0], x_transformed[positives_chosen, 1], c="#004488",
                            s=s, label=f'Positive_{chosen_gene}')

            # ax[int(fold_id), ds_id].legend()
            plt.xticks([])
            plt.yticks([])
            plt.legend()
            plt.title(
                f'RFE({args.visual_tool}){cancer}-{fold_id}({all_x.shape[0]})-All in {format(algo_time, ".2f")} sec')

            plt.savefig(pdf_loc, type='pdf', dpi=300,
                        bbox_inches='tight')
            plt.savefig(png_loc, type='png', dpi=300,
                        bbox_inches='tight')
            #plt.show()


def rfe_semi_all_cancer(ds_aims=['seq_1024'], cancer_aims=['BRCA'], chosen_gene=None, unk_folds=[1, 2, 3], transparent=False):
    s = 0.9
    r_state=124
    all_ds = '|'.join(ds_aims)
    arguments = ''
    if args.visual_tool == 'tsne':
        arguments = f'_perp{args.perplexity}_rs{r_state}'
    elif args.visual_tool == 'umap':
        arguments = f'_nn={args.nneighbors}_md={args.mindist}_rs{r_state}'
    if transparent:
        arguments = f'{arguments}_tp'
    loc_dict = dl.get_all_locs()
    for cancer in cancer_aims:
        model_loc = config.RESULT_DIR / 'ELRRF' / 'models_test' / f'{all_ds}_{cancer}_True_True_type2_10_undersample_train_test.pickle'
        #model_loc = f'/Volumes/tudelft/ELISL/results/ELRRF/models_test/{sftp_loc[cancer]}'
        prev_model = dfnc.load_pickle(model_loc)
        train_ind_loc = config.DATA_DIR / 'feature_sets' / f'train_test_{cancer}_{10}.json'
        with open(train_ind_loc, 'r') as fp:
            fold_dict = json.load(fp)
        cancer_data_dict = {'train': {}, 'unk': {}}
        final_tr_data = None
        final_unk_data = None
        for ds_name in ds_aims:
            tr_data = pd.read_csv(loc_dict[f'train_{ds_name}_data_loc'])
            tr_data = tr_data.fillna(0)
            processed_data_tr = dfnc.prepare_cancer_dataset(tr_data, cancer=cancer)
            processed_data_tr.append(processed_data_tr)
            cancer_data_dict['train'][ds_name] = processed_data_tr

            # Unknown Data for this dataset and cancer
            unk_data = pd.read_csv(loc_dict[f'unk_{ds_name}_{cancer}_data_loc'])
            unk_data = unk_data.fillna(0)
            processed_data_unk = dfnc.prepare_cancer_dataset(unk_data, cancer=cancer)
            cancer_data_dict['unk'][ds_name] = processed_data_unk

            if final_tr_data is None:
                final_tr_data = processed_data_tr.copy()
                final_unk_data = processed_data_unk.copy()
            else:
                final_tr_data = pd.merge(final_tr_data, processed_data_tr, how='inner', on=['pair_name', 'class'])
                final_unk_data = pd.merge(final_unk_data, processed_data_unk, how='inner', on=['pair_name', 'class'])

        cancer_data_dict['train']['&'.join(ds_aims)] = final_tr_data
        cancer_data_dict['unk']['&'.join(ds_aims)] = final_unk_data

        for fold_id, items in fold_dict.items():
            if int(fold_id) in [1, 3, 6, 8, 9]:
                continue
            print(f'Fold: {fold_id}')
            for unk_fold in unk_folds:
                png_loc = config.RESULT_DIR / 'sample_vis' / \
                          f'RFE_{args.visual_tool}_{cancer}{fold_id}_train_all_ds_{chosen_gene}_unk{unk_fold}x1{arguments}.png'
                pdf_loc = config.RESULT_DIR / 'sample_vis' / \
                          f'RFE_{args.visual_tool}_{cancer}{fold_id}_train_all_ds_{chosen_gene}_unk{unk_fold}x1{arguments}.pdf'
                ensure_file_dir(png_loc)
                if os.path.exists(png_loc):
                    continue

                best_params = prev_model['best_params'][fold_id]
                plt.clf()
                final_D = None
                final_unk_D = None
                unk_norm_probs = None
                for ds_name, ds_data in cancer_data_dict['train'].items():
                    y = ds_data['class'].values.copy()[items['train']]
                    raw_y = ds_data['class'].values.copy()[items['train']]
                    labels = ds_data['pair_name'].values.copy()[items['train']]
                    if chosen_gene is not None:
                        chosen_gene_pairs = [chosen_gene in label for label in labels]
                        y[chosen_gene_pairs & (y == 1)] = 2
                        y[chosen_gene_pairs & (y == 0)] = 3
                    all_x = ds_data.drop(columns=['pair_name', 'class']).values.copy()[items['train']]
                    processed_data_unk = cancer_data_dict['unk'][ds_name]
                    if unk_fold != -1:
                        processed_data_unk = processed_data_unk.sample(n=len(all_x), replace=False,
                                                                       random_state=unk_fold)
                    all_x_unk = processed_data_unk.drop(columns=['pair_name', 'class']).values.copy()

                    scaler = StandardScaler()
                    x = scaler.fit_transform(all_x)
                    x_unk = scaler.transform(all_x_unk)

                    print("Running RFE ...")
                    start = timer()
                    clf = lgb.LGBMClassifier(boosting_type='rf', num_leaves=165, max_depth=best_params['max_depth'],
                                             learning_rate=0.1,
                                             n_estimators=best_params['n_estimators'],
                                             subsample_for_bin=200000, objective=None, class_weight=None,
                                             min_split_gain=0.0,
                                             min_child_weight=5, min_child_samples=best_params['min_child_samples'],
                                             subsample=best_params['subsample'],
                                             subsample_freq=1,
                                             colsample_bytree=best_params['colsample_bytree'], reg_alpha=0.0,
                                             reg_lambda=best_params['reg_lambda'], random_state=124,
                                             n_jobs=- 1, is_unbalance=False)
                    clf.fit(x, raw_y)
                    leaves = clf.predict(x, pred_leaf=True)
                    leaves_unk = clf.predict(x_unk, pred_leaf=True)
                    one_ind = np.where(clf.classes_ == 1)[0][0]
                    unk_probs = clf.predict_proba(x_unk)[:,one_ind]
                    all_leaves = np.concatenate((leaves, leaves_unk))
                    onehot_x_all = OneHotEncoder().fit_transform(all_leaves)
                    S = (onehot_x_all * onehot_x_all.transpose()).todense()
                    # lastly, we normalize and subtract from 1, to get dissimilarities
                    D = 1 - S / S.max()
                    contr = prev_model['params']['contr'][fold_id][ds_name] / sum(
                        prev_model['params']['contr'][fold_id].values())
                    if final_D is None:
                        final_D = D.copy() * contr
                        unk_norm_probs = unk_probs.copy() * contr
                    else:
                        final_D = D.copy() * contr + final_D.copy()
                        unk_norm_probs = unk_probs.copy() * contr + unk_norm_probs.copy()
                if args.visual_tool == 'tsne':
                    x_all_transformed = TSNE(metric='precomputed', perplexity=args.perplexity, random_state=r_state).fit_transform(final_D)
                elif args.visual_tool == 'umap':
                    x_all_transformed = umap.UMAP(metric='precomputed', n_neighbors=args.nneighbors,
                                                  min_dist=args.mindist, random_state=r_state).fit_transform(final_D)

                x_transformed = x_all_transformed[:len(leaves), :]
                x_unk_transformed = x_all_transformed[len(leaves):, :]

                import matplotlib
                neg_rgba = list(matplotlib.colors.to_rgba('#EE99AA'))
                pos_rgba = list(matplotlib.colors.to_rgba('#6699CC'))
                unk_colors = np.array([neg_rgba] * len(unk_norm_probs))
                unk_colors[unk_norm_probs > 0.5] = pos_rgba
                unk_alpha = np.abs(unk_norm_probs.copy()-0.5)
                unk_colors[:, 3] = unk_alpha
                algo_time = timer() - start
                print(f"RFE: {algo_time}")

                negatives = y == 0
                positives = y == 1
                unknowns = y == -1
                negatives_chosen = y == 3
                positives_chosen = y == 2

                
                plt.scatter(x_transformed[negatives, 0], x_transformed[negatives, 1], c="#994455",  # 994455
                            s=s, label='Negative')
                plt.scatter(x_transformed[positives, 0], x_transformed[positives, 1], c="#004488",  # 004488
                            s=s, label='Positive')
                if chosen_gene is not None:
                    plt.scatter(x_transformed[negatives_chosen, 0], x_transformed[negatives_chosen, 1], c="#994455",
                                s=s, label=f'Negative_{chosen_gene}')
                    plt.scatter(x_transformed[positives_chosen, 0], x_transformed[positives_chosen, 1], c="#004488",
                                s=s, label=f'Positive_{chosen_gene}')
                if transparent:
                    plt.scatter(x_unk_transformed[:, 0], x_unk_transformed[:, 1], c=unk_colors,
                                s=0.6, label='Unknown')#, alpha=unk_alpha) # c=unk_colors,#c=unk_colors,
                else:
                    plt.scatter(x_unk_transformed[:, 0], x_unk_transformed[:, 1], c='#FF8000',
                                s=0.6, label='Unknown')#, alpha=unk_alpha) # c=unk_colors,#c=unk_colors,


                # ax[int(fold_id), ds_id].legend()
                plt.xticks([])
                plt.yticks([])
                plt.legend()
                plt.title(
                    f'RFE({args.visual_tool}){cancer}-{fold_id}({all_x.shape[0]})-All, Unk-{unk_fold} in {format(algo_time, ".2f")} sec')

                plt.savefig(pdf_loc, type='pdf', dpi=300,
                            bbox_inches='tight')
                plt.savefig(png_loc, type='png', dpi=300,
                            bbox_inches='tight')
                #plt.show()


if __name__ == "__main__":
    ds_aims = ['seq_1024', 'ppi_ec', 'crispr_dependency_mut', 'crispr_dependency_expr', 'tissue']
    cancer_aims = ['BRCA', 'CESC', 'KIRC', 'LAML', 'LUAD', 'OV', 'SKCM']
    # umap_ds_cancer(ds_aim=None, cancer_aim=None, chosen_gene = None, label_usage='supervised')
    # umap_ds_cancer(ds_aims=ds_aims, cancer_aims=['BRCA'], chosen_gene = 'BRCA1', label_usage='supervised')
    # umap_ds_cancer(ds_aims=ds_aims, cancer_aims=['BRCA'], chosen_gene = 'BRCA2', label_usage='supervised')
    # umap_ds_cancer(ds_aims=ds_aims, cancer_aims=['BRCA'], chosen_gene = 'PARP1', label_usage='supervised')
    # umap_ds_cancer(ds_aims=ds_aims, cancer_aims=['BRCA'], chosen_gene = 'TP53', label_usage='supervised')
    # umap_ds_cancer(ds_aims=ds_aims, cancer_aims=['LUAD'], chosen_gene = 'KRAS', label_usage='supervised')
    # rfe_ds_cancer(ds_aims=ds_aims, cancer_aims=cancer_aims, chosen_gene=None)
    # rfe_semi_ds_cancer(ds_aims=ds_aims, cancer_aims=cancer_aims, chosen_gene=None, unk_folds=[1,2,3])
    rfe_all_cancer(ds_aims=ds_aims, cancer_aims=cancer_aims, chosen_gene=None)
    rfe_semi_all_cancer(ds_aims=ds_aims, cancer_aims=cancer_aims, chosen_gene=None, unk_folds=[1,2,3], transparent=False)
    rfe_semi_all_cancer(ds_aims=ds_aims, cancer_aims=cancer_aims, chosen_gene=None, unk_folds=[1,2,3], transparent=True)
