import os
import sys
path2this = os.path.dirname(os.path.abspath(__file__)).split('/')
for i, folder in enumerate(path2this):
    if folder.lower()=='elisl':
        project_path = '/'.join(path2this[:i+1])
sys.path.insert(0,project_path)
#sys.path.insert(1, "/Users/yitepeli/PycharmProjects/GCATSL/source")
from src.models.ELRRF import *
from src.comparison.GCATSL.source.GCATSL import *
from src.comparison.GRSMF.GRSMF import *

import collections as coll
import pandas as pd
import numpy as np
import json

from skopt.space import Categorical
from skopt.utils import use_named_args
from skopt import gp_minimize
import src.data_functions as dfnc
from src import config
import time
import logging
GCATSL_root = str(config.ROOT_DIR / 'src' / 'comparison' / 'GCATSL/')
import argparse

parser = argparse.ArgumentParser(description='Run Feature Generation')
parser.add_argument('--task', '-t', metavar='the-task', dest='task', type=str, help='Choose task', default='ELRRF_unknown_test_gs_trteus')
parser.add_argument('--cancer', '-c', metavar='the-cancer', dest='cancer', type=str, help='Choose cancer', default='LUAD')

args = parser.parse_args()
print(f'Running args:{args}')
task=args.task
#task='ELRRF_crossc_test_gs_trteus'#
#task='GCATSL_holdout_d_test_trteus'

#task='ELRRF_single_val_trteus'
#task='single_val_trteus_both'
#task='holdout_d_val_trteus_both'
#task='ELRRF_crossc_test_gs_trteus'
#task='cross_mc_val_trteus'
#task='loco_val_trteus'
#task='loco_val_trus'
#task='pretrain_val_trus'
log_name = config.ROOT_DIR / 'logs' / task
#log_name = task
logging.basicConfig(level=logging.DEBUG, filename=log_name, filemode="a+",
                    format="%(asctime)-15s %(levelname)-8s %(message)s")

PROJECT_LOC = config.ROOT_DIR
RANDOMIZING=False


def fit_predict_cross_validation(model, samples, fold_type, n_split, train_idx=None):
    n_split= min(n_split, len(samples[samples['class']==1]), len(samples[samples['class']==0]))
    if 'stratified' in fold_type and 'shuffle' in fold_type:
        skf = StratifiedKFold(n_splits=n_split, shuffle=True, random_state=model.random_state)
    elif 'stratified' in fold_type:
        skf = StratifiedKFold(n_splits=n_split, shuffle=False, random_state=model.random_state)
    elif 'shuffle' in fold_type:
        skf = KFold(n_splits=n_split, shuffle=True, random_state=model.random_state)
    else:
        skf = KFold(n_splits=n_split, shuffle=False, random_state=model.random_state)
    indices = skf.split(samples, samples['class'].values)

    fold_ind = coll.OrderedDict()
    fold_samples = coll.OrderedDict()

    for i, (tr_indices, te_indices) in enumerate(indices):
        if 'undersample' in model.balance_strategy and 'train' in model.balance_strategy:
            tr_indices = balance_by_index(samples, tr_indices, rand_seed=124+i)
        if 'undersample' in model.balance_strategy and 'test' in model.balance_strategy:
            te_indices = balance_by_index(samples, te_indices, rand_seed=124+i)
        fold_ind[i] = {'train': tr_indices, 'test': te_indices}
        fold_samples[i] = {'train': samples.iloc[tr_indices], 'test': samples.iloc[te_indices]}

    results = coll.OrderedDict()
    for fold, sample_dict in fold_ind.items():
        fold_start_time = time.time()
        if model.pretrained_model is not None:
            results[fold] = model.fit_predict_all_datasets(sample_dict, samples, init_model=model.pretrained_model[fold])
        elif train_idx is not None:
            results[fold] = model.fit_predict_cv(sample_dict, samples, train_idx)
        else:
            results[fold] = model.fit_predict_all_datasets(sample_dict, samples)
        lasted = str(time.time() - fold_start_time)
        results[fold]['time'] = lasted

    return results


def fit_predict_2set_cc(model, samples_train, samples_test, n_split=10, fold_usage={}):
    if fold_usage:
        fold_ind = coll.OrderedDict()
        train_cancer = list(fold_usage.keys())[0]
        test_cancer = list(fold_usage.keys())[1]
        train_ind_loc = config.DATA_DIR / 'feature_sets' / ('train_test_'+train_cancer+'_'+str(n_split)+'.json')
        test_ind_loc = config.DATA_DIR / 'feature_sets' / ('train_test_'+test_cancer+'_'+str(n_split)+'.json')
        with open(train_ind_loc, 'r') as fp_tr:
            prev_tr_fold_ind = json.load(fp_tr)
        with open(test_ind_loc, 'r') as fp_te:
            prev_te_fold_ind = json.load(fp_te)
        for prev_i, prev_fold in prev_tr_fold_ind.items():
            fold_ind[prev_i] = {'train': prev_fold[fold_usage[train_cancer]],
                                        'test':prev_te_fold_ind[prev_i][fold_usage[test_cancer]]}
    else:
        fold_ind = coll.OrderedDict()
        fold_samples = coll.OrderedDict()
        tr_all_indices = np.array(list(range(len(samples_train))))
        te_all_indices = np.array(list(range(len(samples_test))))
        for i in range(n_split):
            tr_indices = tr_all_indices.copy()
            te_indices = te_all_indices.copy()
            if 'undersample' in model.balance_strategy and 'train' in model.balance_strategy:
                tr_indices = balance_by_index(samples_train, tr_all_indices, rand_seed=124 + i)
            if 'undersample' in model.balance_strategy and 'test' in model.balance_strategy:
                te_indices = balance_by_index(samples_test, te_all_indices, rand_seed=124 + i)

            fold_ind[i] = {'train': tr_indices, 'test': te_indices}
            fold_samples[i] = {'train': samples_train.iloc[tr_indices], 'test': samples_test.iloc[te_indices]}
        train_test_ind={}
        for fold, sample_dict in fold_ind.items():
            train_test_ind[fold] = {'train':sample_dict['train'].tolist(), 'test': sample_dict['test'].tolist()}

    if model.grid_searched:
        data_choices = '|'.join(list(model.both_datasets_dict.keys())[:-1])
        model_loc = data_choices + '_' + model.cancer + '_' + str(model.grid_searched) + '_' + str(
            model.use_single) + '_' + 'type2' + '_' + str(n_split) + \
                    '_' + model.balance_strategy + '.pickle'

        model_loc = config.ROOT_DIR / 'results' / type(model).__name__ / 'models_test' / model_loc
        tuned_model = dfnc.load_pickle(model_loc)
        model.contr = tuned_model['params']['contr'].copy()

    results = coll.OrderedDict()
    results['best_params']={}
    for fold, sample_dict in fold_ind.items():
        if model.grid_searched:
            try:
                model.set_params(**tuned_model['best_params'][fold])
                results['best_params'][fold] =tuned_model['best_params'][fold]
            except:
                model.set_params(**tuned_model['best_params'][str(fold)])
                results['best_params'][fold] =tuned_model['best_params'][str(fold)]
        fold_all_time = time.time()
        results[fold] = model.fit_predict_all_datasets_2sets(sample_dict, samples_train, samples_test, fold)
        lasted_all = str(time.time() - fold_all_time)
        results[fold]['time_all'] = lasted_all

    return results


def fit_predict_2set(model, samples_train, samples_test, n_split=5, ds_names=None):
    prev_fold_ind=None
    train_inds=None
    if ds_names is None:
        train_test_ind_loc = config.DATA_DIR / 'feature_sets' / ('train_test_'+model.cancer+'_'+str(n_split)+'.json')
        train_test_ind_prev_loc = config.DATA_DIR / 'feature_sets' / ('train_test_'+model.cancer+'.json')
    elif 'train' in ds_names or 'test' in ds_names:
        train_ind_loc = config.DATA_DIR / 'feature_sets' / ('train_test_'+model.cancer+'_'+str(n_split)+'.json')
        train_test_ind_loc = config.DATA_DIR / 'feature_sets' / \
                             ('train_test_' + model.cancer + '_' + ds_names[0] + '_' + ds_names[1] + '_' + str(
                                 n_split) + '.json')
        train_test_ind_prev_loc = config.DATA_DIR / 'feature_sets' / ('train_test_jhu'+model.cancer+'.json')
        with open(train_ind_loc, 'r') as fpp:
            train_inds = json.load(fpp)
    else:
        train_test_ind_loc = config.DATA_DIR / 'feature_sets' / \
                             ('train_test_' + model.cancer + '_' + ds_names[0] + '_' + ds_names[1] + '_' + str(
                                 n_split) + '.json')
        train_test_ind_prev_loc = config.DATA_DIR / 'feature_sets' / \
                                  ('train_test_' + model.cancer + '_' + ds_names[0] + '_' + ds_names[1] + '.json')

    if os.path.exists(train_test_ind_prev_loc):
        with open(train_test_ind_prev_loc, 'r') as fpp:
            prev_fold_ind = json.load(fpp)

    if os.path.exists(train_test_ind_loc):
        with open(train_test_ind_loc, 'r') as fp:
            fold_ind = json.load(fp)
    else:
        fold_ind = coll.OrderedDict()
        fold_samples = coll.OrderedDict()
        tr_all_indices = np.array(list(range(len(samples_train))))
        te_all_indices = np.array(list(range(len(samples_test))))
        train_test_ind={}
        for i in range(n_split):
            if prev_fold_ind is not None and str(i) in prev_fold_ind.keys():
                train_test_ind[i] = prev_fold_ind[str(i)]
                continue
            tr_indices = tr_all_indices.copy()
            te_indices = te_all_indices.copy()
            if 'undersample' in model.balance_strategy and 'train' in model.balance_strategy:
                tr_indices = balance_by_index(samples_train, tr_all_indices, rand_seed=124 + i)
            if 'undersample' in model.balance_strategy and 'test' in model.balance_strategy:
                te_indices = balance_by_index(samples_test, te_all_indices, rand_seed=124 + i)
            fold_ind[i] = {'train': tr_indices, 'test': te_indices}
            fold_samples[i] = {'train': samples_train.iloc[tr_indices], 'test': samples_test.iloc[te_indices]}
            if train_inds is not None and str(i) in train_inds.keys():
                train_test_ind[i] = {'train': train_inds[str(i)]['train'], 'test': te_indices.tolist()}
                fold_ind[i] = {'train': train_inds[str(i)]['train'], 'test': te_indices}
            else:
                train_test_ind[i] = {'train':tr_indices.tolist(), 'test': te_indices.tolist()}
        #for fold, sample_dict in fold_ind.items():
        #    train_test_ind[fold] = {'train':sample_dict['train'].tolist(), 'test': sample_dict['test'].tolist()}
        with open(train_test_ind_loc, 'w') as fp:
            json.dump(train_test_ind, fp)

    results = coll.OrderedDict()
    results['best_params']={}
    already_calc_model = None

    if model.grid_searched and train_inds is not None:
        data_choices = '|'.join(list(model.both_datasets_dict.keys())[:-1])
        model_loc = data_choices + '_' + model.cancer + '_' + str(model.grid_searched) + '_' + str(
            model.use_single) + '_' + 'type2' + '_' + str(n_split) + \
                    '_undersample_train_test.pickle'

        model_loc = config.ROOT_DIR / 'results' / type(model).__name__ / 'models_test' / model_loc
        tuned_model = dfnc.load_pickle(model_loc)
        model.contr = tuned_model['params']['contr'].copy()

    for fold, sample_dict in fold_ind.items():
        fold_all_time = time.time()
        if model.grid_searched and train_inds is not None:
            try:
                model.set_params(**tuned_model['best_params'][fold])
                results['best_params'][fold] =tuned_model['best_params'][fold]
            except:
                model.set_params(**tuned_model['best_params'][str(fold)])
                results['best_params'][fold] =tuned_model['best_params'][str(fold)]
        elif model.grid_searched:
            gs_start_time = time.time()
            random_grid = model.get_tree_search_grid()
            best_params = {}
            search_space = list()
            for par_name, par_val in random_grid.items():
                search_space.append(Categorical(par_val, name=par_name))

            @use_named_args(search_space)
            def evaluate_model(**params):
                model.set_params(**params)
                samples_gs = samples_train.copy().iloc[sample_dict['train']]
                result_cv = fit_predict_cross_validation(model, samples_gs, fold_type='stratified_shuffled', n_split=10, train_idx=sample_dict['train'])
                AUROC, AUPRC, MC = model.evaluate_folds(result_cv, report=False)
                return 1 - np.mean(AUPRC)

            result_best = gp_minimize(evaluate_model, search_space, n_calls=20, random_state=model.random_state, n_jobs=-1)
            for idx in range(len(search_space)):
                best_params[search_space[idx]._name] = result_best.x[idx]

            model.set_params(**best_params)
            results['best_params'][fold] =best_params

            print(f'Fold{fold} Best params: {best_params}')
            gs_lasted = str(time.time() - gs_start_time)
        fold_start_time = time.time()
        if 'EL' in type(model).__name__ and train_inds is None:
            samples_cont = samples_train.copy().iloc[sample_dict['train']]
            best_val_res = fit_predict_cross_validation(model, samples_cont, fold_type='stratified_shuffled', n_split=10,
                                                    train_idx=sample_dict['train'])
            model.calculate_data_weights(best_val_res, fold_main=fold)
        results[fold] = model.fit_predict_all_datasets_2sets(sample_dict, samples_train, samples_test, fold)

        if model.grid_searched and train_inds is None:
            results[fold]['time_gs'] = gs_lasted
        lasted = str(time.time() - fold_start_time)
        results[fold]['time_final'] = lasted
        lasted_all = str(time.time() - fold_all_time)
        results[fold]['time_all'] = lasted_all

    return results


def calc_dataset_importance(model, samples_train, samples_test, n_split=5, typex=''):
    train_test_ind_loc = config.DATA_DIR / 'feature_sets' / ('train_test_'+model.cancer+'_'+str(n_split)+'.json')
    if os.path.exists(train_test_ind_loc):
        with open(train_test_ind_loc, 'r') as fp:
            fold_ind = json.load(fp)
    else:
        fold_ind = coll.OrderedDict()
        fold_samples = coll.OrderedDict()
        tr_all_indices = np.array(list(range(len(samples_train))))
        te_all_indices = np.array(list(range(len(samples_test))))
        for i in range(n_split):
            tr_indices = tr_all_indices.copy()
            te_indices = te_all_indices.copy()
            if 'undersample' in model.balance_strategy and 'train' in model.balance_strategy:
                tr_indices = balance_by_index(samples_train, tr_all_indices, rand_seed=124 + i)
            if 'undersample' in model.balance_strategy and 'test' in model.balance_strategy:
                te_indices = balance_by_index(samples_test, te_all_indices, rand_seed=124 + i)
            fold_ind[i] = {'train': tr_indices, 'test': te_indices}
            fold_samples[i] = {'train': samples_train.iloc[tr_indices], 'test': samples_test.iloc[te_indices]}
        train_test_ind={}
        for fold, sample_dict in fold_ind.items():
            train_test_ind[fold] = {'train':sample_dict['train'].tolist(), 'test': sample_dict['test'].tolist()}
        with open(train_test_ind_loc, 'w') as fp:
            json.dump(train_test_ind, fp)


    if model.grid_searched:
        data_choices = '|'.join(list(model.both_datasets_dict.keys())[:-1])
        model_loc = data_choices + '_' + model.cancer + '_' + str(model.grid_searched) + '_' + str(
            model.use_single) + '_' + 'type2' + '_' + str(n_split) + \
                    '_' + model.balance_strategy + '.pickle'

        model_loc = config.ROOT_DIR / 'results' / type(model).__name__ / 'models_test' / model_loc
        tuned_model = dfnc.load_pickle(model_loc)
        model.contr = tuned_model['params']['contr'].copy()

    results = coll.OrderedDict()
    results['best_params']={}
    for fold, sample_dict in fold_ind.items():
        if model.grid_searched:
            try:
                model.set_params(**tuned_model['best_params'][fold])
                results['best_params'][fold] =tuned_model['best_params'][fold]
            except:
                model.set_params(**tuned_model['best_params'][str(fold)])
                results['best_params'][fold] =tuned_model['best_params'][str(fold)]

        fold_all_time = time.time()
        if typex=='feature2':
            results[fold] = model.calc_feature_importance(sample_dict, samples_train, samples_test, fold)
        elif typex=='cancer_set2':
            results[fold] = model.calc_detail_dataset_importance(sample_dict, samples_train, samples_test, fold)
        else:
            results[fold] = model.calc_dataset_importance(sample_dict, samples_train, samples_test, fold)

        lasted_all = str(time.time() - fold_all_time)
        results[fold]['time_all'] = lasted_all

    return results


def fit_predict_dho_cross_validation(model, samples, n_split, ho=None):
    train_test_ind_loc = config.DATA_DIR / 'feature_sets' / ('train_test_dho_'+model.cancer+'_'+str(n_split)+'.json')
    if os.path.exists(train_test_ind_loc):
        with open(train_test_ind_loc, 'r') as fp:
            fold_ind = json.load(fp)
    else:
        fold_ind = coll.OrderedDict()
        fold_samples = coll.OrderedDict()
        print(f'Creating datasets')
        for i in range(n_split):
            genes = samples['pair_name'].str.split('|', expand=True)
            genes['SL'] = samples['class']
            all_genes = np.union1d(genes[0].values, genes[1].values)
            gene2ind = dict(zip(all_genes, range(len(all_genes))))
            adj_genes_np = np.empty(shape=(len(all_genes), len(all_genes)))
            adj_genes_np[:] = np.nan
            for ind, row in genes.iterrows():
                adj_genes_np[gene2ind[row[0]], gene2ind[row[1]]] = row['SL']
                adj_genes_np[gene2ind[row[1]], gene2ind[row[0]]] = row['SL']
            adj_genes = pd.DataFrame(adj_genes_np,index=all_genes, columns=all_genes)
            train_size, test_size = 1.0, 1.0
            train_genes, test_genes = [], []
            seed_id = i * len(all_genes)
            while (len(all_genes) > 0):
                if (train_size / test_size <= 4) & (len(all_genes) > 0):
                    np.random.seed(seed_id)
                    chosen_gene = np.random.choice(all_genes, 1)
                    all_genes = np.delete(all_genes, np.where(all_genes == chosen_gene))
                    train_genes.append(chosen_gene[0])
                    train_size = adj_genes.loc[train_genes, train_genes].notna().sum().sum() / 2
                if (train_size / test_size > 4) & (len(all_genes) > 0):
                    np.random.seed(seed_id)
                    chosen_gene = np.random.choice(all_genes, 1)
                    all_genes = np.delete(all_genes, np.where(all_genes == chosen_gene))
                    test_genes.append(chosen_gene[0])
                    test_size = adj_genes.loc[test_genes, test_genes].notna().sum().sum() / 2
                seed_id += 1

            print('double')
            tr_indices = ((genes[0].isin(train_genes)) & (genes[1].isin(train_genes))).values.nonzero()[0]
            te_indices = ((genes[0].isin(test_genes)) & (genes[1].isin(test_genes))).values.nonzero()[0]

            if 'undersample' in model.balance_strategy and 'train' in model.balance_strategy:
                tr_indices = balance_by_index(samples, tr_indices, rand_seed=124 + i)
            if 'undersample' in model.balance_strategy and 'test' in model.balance_strategy:
                te_indices = balance_by_index(samples, te_indices, rand_seed=124 + i)
            fold_ind[i] = {'train': tr_indices, 'test': te_indices}
            fold_samples[i] = {'train': samples.iloc[tr_indices], 'test': samples.iloc[te_indices]}

        train_test_ind = {}
        for fold, sample_dict in fold_ind.items():
            train_test_ind[fold] = {'train': sample_dict['train'].tolist(), 'test': sample_dict['test'].tolist()}
        with open(train_test_ind_loc, 'w') as fp:
            json.dump(train_test_ind, fp)

    results = coll.OrderedDict()
    results['best_params']={}
    for fold, sample_dict in fold_ind.items():
        results[fold]={}
        fold_all_time = time.time()
        if model.grid_searched:
            print(f'Grid search started for fold{fold}')
            gs_start_time = time.time()
            random_grid = model.get_tree_search_grid()
            best_params = {}
            search_space = list()
            for par_name, par_val in random_grid.items():
                search_space.append(Categorical(par_val, name=par_name))

            @use_named_args(search_space)
            def evaluate_model(**params):
                model.set_params(**params)
                samples_gs = samples.copy().iloc[sample_dict['train']]
                result_cv = fit_predict_cross_validation(model, samples_gs, fold_type='stratified_shuffled', n_split=10, train_idx=sample_dict['train'])
                AUROC, AUPRC, MC = model.evaluate_folds(result_cv, report=False)
                return 1 - np.mean(AUPRC)
            try:
                result_best = gp_minimize(evaluate_model, search_space, n_calls=20, random_state=model.random_state, n_jobs=-1)
                for idx in range(len(search_space)):
                    best_params[search_space[idx]._name] = result_best.x[idx]

                model.set_params(**best_params)
                results['best_params'][fold] =best_params


                print(f'Fold{fold} Best params: {best_params}')
                gs_lasted = str(time.time() - gs_start_time)
                results[fold]['time_gs'] = gs_lasted
            except:
                print('Finetune broked')
        fold_start_time = time.time()
        if 'EL' in type(model).__name__:
            samples_cont = samples.copy().iloc[sample_dict['train']]
            best_val_res = fit_predict_cross_validation(model, samples_cont, fold_type='stratified_shuffled', n_split=10,
                                                    train_idx=sample_dict['train'])
            model.calculate_data_weights(best_val_res, fold_main=fold)
        print(f'Fold {fold} started!')
        results[fold] = model.fit_predict_all_datasets(sample_dict, samples, cv=fold)
        print(f'Fold {fold} ended!')
        lasted = str(time.time() - fold_start_time)
        results[fold]['time_final'] = lasted
        lasted_all = str(time.time() - fold_all_time)
        results[fold]['time_all'] = lasted_all

    return results


def compare_contr(model, samples_train, samples_test, n_split=5):
    train_test_ind_loc = config.DATA_DIR / 'feature_sets' / ('train_test_'+model.cancer+'.json')
    if os.path.exists(train_test_ind_loc):
        with open(train_test_ind_loc, 'r') as fp:
            fold_ind = json.load(fp)
    else:
        fold_ind = coll.OrderedDict()
        fold_samples = coll.OrderedDict()
        tr_all_indices = np.array(list(range(len(samples_train))))
        te_all_indices = np.array(list(range(len(samples_test))))
        for i in range(n_split):
            tr_indices = tr_all_indices.copy()
            te_indices = te_all_indices.copy()
            if 'undersample' in model.balance_strategy and 'train' in model.balance_strategy:
                tr_indices = balance_by_index(samples_train, tr_all_indices, rand_seed=124 + i)
            if 'undersample' in model.balance_strategy and 'test' in model.balance_strategy:
                te_indices = balance_by_index(samples_test, te_all_indices, rand_seed=124 + i)
            fold_ind[i] = {'train': tr_indices, 'test': te_indices}
            fold_samples[i] = {'train': samples_train.iloc[tr_indices], 'test': samples_test.iloc[te_indices]}
        train_test_ind={}
        for fold, sample_dict in fold_ind.items():
            train_test_ind[fold] = {'train':sample_dict['train'].tolist(), 'test': sample_dict['test'].tolist()}
        with open(train_test_ind_loc, 'w') as fp:
            json.dump(train_test_ind, fp)

    data_choices = '|'.join(list(model.both_datasets_dict.keys())[:-1])
    model_loc = data_choices + '_' + model.cancer + '_' + str(model.grid_searched) + '_' + str(
        model.use_single) + '_' + 'type2' + '_' + str(n_split) + \
                '_' + model.balance_strategy + '.pickle'

    model_loc = config.ROOT_DIR / 'results' / type(model).__name__ / 'models_test' / model_loc
    tuned_model = dfnc.load_pickle(model_loc)
    #model.contr = tuned_model['params']['contr'].copy()

    results = coll.OrderedDict()
    results['best_params'] = coll.OrderedDict()
    for fold, sample_dict in fold_ind.items():
        try:
            model.set_params(**tuned_model['best_params'][fold])
            results['best_params'][fold] =tuned_model['best_params'][fold]
            contr = tuned_model['params']['contr'][fold]
        except:
            model.set_params(**tuned_model['best_params'][str(fold)])
            results['best_params'][fold] =tuned_model['best_params'][str(fold)]
            contr = tuned_model['params']['contr'][str(fold)]
        results[fold] = model.fit_predict_all_datasets_2sets(sample_dict, samples_train, samples_test, fold)
        tr_auc_dicts = {k: v['tr_auc'] for k, v in results[fold].items()}
        tr_auc_sum = sum(list(tr_auc_dicts.values()))
        tr_auc_freq = {k: np.around(v / tr_auc_sum, 4) for k, v in tr_auc_dicts.items()}

        contr_sum = sum(list(contr.values()))
        contr_freq = {k: np.around(v / contr_sum, 4) for k, v in contr.items()}
        print(f'Fold {fold}:\nContr:{contr_freq}\nTR_auc:{tr_auc_freq}\n')
    print()
    return results


def balance_cancer_by_index(samples, indices, rand_seed=-1):
    indices_from_cancers = []
    for cancer in samples['cancer'].unique():
        tr_samples = samples.iloc[indices]
        pos_indices = indices[(tr_samples['cancer'].values==cancer)&(tr_samples['class'].values == 1)]
        neg_indices = indices[(tr_samples['cancer'].values==cancer)&(tr_samples['class'].values == 0)]
        n = min(len(pos_indices), len(neg_indices))
        np.random.seed(rand_seed)
        if len(neg_indices)==len(pos_indices):
            pass
        elif n == len(neg_indices):
            np.random.seed(rand_seed)
            pos_indices = np.random.choice(pos_indices, n, replace=False)
        elif n == len(pos_indices):
            np.random.seed(rand_seed)
            neg_indices = np.random.choice(neg_indices, n, replace=False)
        else:
            np.random.seed(rand_seed)
            pos_indices = np.random.choice(pos_indices, n, replace=False)
            np.random.seed(rand_seed)
            neg_indices = np.random.choice(neg_indices, n, replace=False)

        combined_indices = np.union1d(pos_indices, neg_indices)
        combined_indices.sort()
        indices_from_cancers.append(combined_indices)
    combined_indices = np.concatenate(indices_from_cancers)
    return combined_indices

def balance_by_index(samples, indices, rand_seed=-1):
    if 'cancer' in samples.columns:
        return balance_cancer_by_index(samples, indices, rand_seed=rand_seed)
    tr_samples = samples.iloc[indices]
    pos_indices = indices[tr_samples['class'].values == 1]
    neg_indices = indices[tr_samples['class'].values == 0]
    n = min(len(pos_indices), len(neg_indices))
    np.random.seed(rand_seed)
    if len(neg_indices)==len(pos_indices):
        pass
    elif n == len(neg_indices):
        np.random.seed(rand_seed)
        pos_indices = np.random.choice(pos_indices, n, replace=False)
    elif n == len(pos_indices):
        np.random.seed(rand_seed)
        neg_indices = np.random.choice(neg_indices, n, replace=False)
    else:
        np.random.seed(rand_seed)
        pos_indices = np.random.choice(pos_indices, n, replace=False)
        np.random.seed(rand_seed)
        neg_indices = np.random.choice(neg_indices, n, replace=False)

    combined_indices = np.union1d(pos_indices, neg_indices)
    combined_indices.sort()

    return combined_indices


def combine_preds(x):
    d = {}
    weights = x['tr_auc'] / x['tr_auc'].sum()
    #weights = np.repeat([1/x.shape[0]], x.shape[0])
    weighted_probs = x['probability'] * weights
    d['probability'] = weighted_probs.sum()
    d['avg'] = x['probability'].mean()
    return pd.Series(d, index=['avg', 'probability'])


def balance_set(set, col='', n=None, random_seed=124):
    n = np.unique(set[col].values, return_counts=True)[1].min() if n is None else n
    neg_indices = set[set[col] == 0].index.values
    pos_indices = set[set[col] == 1].index.values

    np.random.seed(random_seed)
    if n == neg_indices.shape[0]:
        pos_sample_ind = np.random.choice(np.unique(set[col].values, return_counts=True)[1][1], n, replace=False)
    elif n == pos_indices.shape[0]:
        neg_sample_ind = np.random.choice(np.unique(set[col].values, return_counts=True)[1][0], n, replace=False)
    else:
        pos_sample_ind = np.random.choice(np.unique(set[col].values, return_counts=True)[1][1], n, replace=False)
        neg_sample_ind = np.random.choice(np.unique(set[col].values, return_counts=True)[1][0], n, replace=False)

    chosen_neg_ind = neg_indices[neg_sample_ind]
    chosen_pos_ind = pos_indices[pos_sample_ind]
    if 'int' in str(type(set.index)).lower():
        neg_set = set.loc[chosen_neg_ind]
        pos_set = set.loc[chosen_pos_ind]
    else:
        neg_set = set.loc[chosen_neg_ind.astype(str)]
        pos_set = set.loc[chosen_pos_ind.astype(str)]

    return pd.concat([pos_set, neg_set])


def prepare_cancer_dataset(df, cancer='BRCA', reverse=False, is_cancer=False, reduce_min=False):
    df_cancer = df.copy()

    #df_cancer = df_cancer[df_cancer['gene1']!=df_cancer['gene2']]
    if cancer == None:
        df_cancer = df_cancer.reset_index()
    elif reverse:
        df_cancer = df_cancer[~(df_cancer['cancer'] == cancer)].reset_index()
    else:
        df_cancer = df_cancer[df_cancer['cancer'] == cancer].reset_index()

    df_cancer = df_cancer.sort_values(by=['gene1', 'gene2'])
    #df_cancer["gene1.l"] = df_cancer["gene1"].str.lower()
    #df_cancer["gene2.l"] = df_cancer["gene2"].str.lower()
    #df_cancer = df_cancer.sort_values(by=['gene1.l', 'gene2.l'], inplace=False)
    df_cancer.insert(loc=0, column='pair_name', value=df_cancer[['gene1', 'gene2']].agg('|'.join, axis=1))
    if is_cancer:
        df_cancer = df_cancer.drop(columns=['gene1', 'gene2', 'index'])
        #df_cancer = df_cancer.sort_values(by=['cancer', 'pair_name'])
    else:
        df_cancer = df_cancer.drop(columns=['gene1', 'gene2', 'cancer', 'index'])
    return df_cancer


def single_contr_test_experiment(data_choice_list, cancer, grid_search, thold=None):
    model_type='ELRRF'
    comb_type='type2'
    boosting='rf'
    use_single=True
    if 'GBDT' in task:
        model_type='ELGBDT'
        boosting='gbdt'

    data_choices = '|'.join(data_choice_list)
    print(f'Experiment: {data_choice_list} for {cancer} with grid_search={grid_search}')
    #print(f'Experiment {model_type}: for {cancer}, grid_search={grid_search}')
    if 'trus' in task:
        balance_strategy = 'undersample_train'
    if 'trteus' in task:
        balance_strategy = 'undersample_train_test'

    n_split=5
    process=True
    if thold == None:
        thold = 0.5

    loc_dict = get_locs()
    res_loc = config.ROOT_DIR / 'results' / model_type / 'single_cancer_test.csv'
    config.ensure_dir(res_loc)

    use_comb = False
    use_all_comb = True
    model = ELRRF(boosting_type=boosting, use_single=use_single, grid_searched=grid_search, balance_strategy=balance_strategy,
                  use_comb=use_comb, thold=thold, process=process, cancer=cancer)
    model_loc = data_choices + '_' + cancer + '_' + str(grid_search) + '_' + str(
        use_single) + '_' + comb_type + '_' + str(n_split) + \
                '_' + balance_strategy + '.pickle'

    model_loc = config.ROOT_DIR / 'results' / model_type / 'models_test' / model_loc
    config.ensure_dir(model_loc)
    tr_samples_loc = 'labels/train_pairs.csv'
    tr_samples_loc = config.DATA_DIR / tr_samples_loc
    tr_samples = pd.read_csv(tr_samples_loc)
    te_samples_loc = 'labels/test_pairs.csv'
    te_samples_loc = config.DATA_DIR / te_samples_loc
    te_samples = pd.read_csv(te_samples_loc)
    samples_tr = prepare_cancer_dataset(tr_samples, cancer=cancer)
    samples_te = prepare_cancer_dataset(te_samples, cancer=cancer)
    #If model is Early Late
    if comb_type == 'type2' and len(data_choice_list) > 1:
        final_trdata = samples_tr.copy()
        final_tedata = samples_te.copy()
    for data_choice in data_choice_list:
        tr_data = pd.read_csv(loc_dict['train_' + data_choice + '_data_loc'])
        tr_data = tr_data.fillna(0)
        te_data = pd.read_csv(loc_dict['test_' + data_choice + '_data_loc'])
        te_data = te_data.fillna(0)
        processed_data_tr = prepare_cancer_dataset(tr_data, cancer=cancer)
        processed_data_te = prepare_cancer_dataset(te_data, cancer=cancer)
        model.add_dataset(data_choice, processed_data_tr, processed_data_te)
        if comb_type == 'type2' and len(data_choice_list) > 1:
            processed_data_tr = processed_data_tr.drop(columns='class')
            final_trdata = pd.merge(final_trdata, processed_data_tr, how='inner', on=['pair_name'])
            processed_data_te = processed_data_te.drop(columns='class')
            final_tedata = pd.merge(final_tedata, processed_data_te, how='inner', on=['pair_name'])
    if comb_type=='type2' and len(data_choice_list) > 1:
        model.add_dataset('&'.join(data_choice_list), final_trdata,final_tedata)

    results = compare_contr(model, samples_tr, samples_te, n_split=n_split)
    #dfnc.save_pickle(model_loc, model_dict)


def single_cancer_test_experiment(data_choice_list, cancer, grid_search, thold=None):
    model_type='ELRRF'
    comb_type='type2'
    boosting='rf'
    if 'GCATSL' in task:
        model_type='GCATSL'
        data_choice_list=['PPI', 'CC', 'BP']#, 'GO1', 'GO2']
        feature_dim = 128
        n_feature = len(data_choice_list)
    elif 'GRSMF' in task:
        model_type='GRSMF'
        data_choice_list=['BP']#, 'GO1', 'GO2']
        lambda_d = 2 ** (-7)
        beta = 2 ** (-5)
        max_iter = 10
    else:
        use_single=True
        if 'GBDT' in task:
            model_type='ELGBDT'
            boosting='gbdt'

    data_choices = '|'.join(data_choice_list)
    print(f'Experiment: {data_choice_list} for {cancer} with grid_search={grid_search}')
    #print(f'Experiment {model_type}: for {cancer}, grid_search={grid_search}')
    if 'trus' in task:
        balance_strategy = 'undersample_train'
    if 'trteus' in task:
        balance_strategy = 'undersample_train_test'

    n_split=10
    process=True
    if thold == None:
        thold = 0.5

    loc_dict = get_locs()
    res_loc = config.ROOT_DIR / 'results' / model_type / 'single_cancer_test.csv'
    config.ensure_dir(res_loc)
    grid_search_settings_loc = config.ROOT_DIR / 'results' / model_type / 'result.json'
    if os.path.isfile(res_loc):
        result_df = pd.read_csv(res_loc)
        if model_type=='GCATSL':
            chosen_cols = result_df[
                ['datasets', 'cancer', 'grid_search', 'n_split', 'balance_strat', 'feature_dim', 'n_feature']]
            chosen_cols = chosen_cols.astype(str)
            chosen_vals = np.array([data_choices, cancer, grid_search, n_split, balance_strategy, feature_dim, n_feature])
            chosen_vals = chosen_vals.astype(str)
        elif model_type=='GRSMF':
            chosen_cols = result_df[
                ['datasets', 'cancer', 'grid_search', 'n_split', 'balance_strat', 'lambda_d', 'beta', 'max_iter']]
            chosen_cols = chosen_cols.astype(str)
            chosen_vals = np.array([data_choices, cancer, grid_search, n_split, balance_strategy, lambda_d, beta, max_iter])
            chosen_vals = chosen_vals.astype(str)
        else:
            chosen_cols = result_df[
                ['datasets', 'cancer', 'grid_search', 'use_single', 'n_split', 'balance_strat', 'threshold',
                 'comb_type', 'process']]
            chosen_cols = chosen_cols.astype(str)
            chosen_vals = np.array([data_choices, cancer, grid_search, use_single, n_split, balance_strategy, thold,
                                    comb_type, process])
            chosen_vals = chosen_vals.astype(str)

        if (chosen_cols == chosen_vals).all(1).any():
            print(f'{data_choices} is already calculated!')
            return 0
    elif model_type=='GCATSL':
        result_df = pd.DataFrame(
            columns=['datasets', 'cancer', 'AUROC_m', 'AUROC_std', 'AUPRC_m', 'AUPRC_std',
                     'MC_m', 'MC_std', 'grid_search', 'n_split',
                     'balance_strat', 'feature_dim', 'n_feature'])
    elif model_type=='GRSMF':
        result_df = pd.DataFrame(
            columns=['datasets', 'cancer', 'AUROC_m', 'AUROC_std', 'AUPRC_m', 'AUPRC_std',
                     'MC_m', 'MC_std','grid_search', 'n_split', 'balance_strat', 'lambda_d', 'beta', 'max_iter'])
    else:
        result_df = pd.DataFrame(
            columns=['datasets', 'cancer', 'AUROC_m', 'AUROC_std', 'AUPRC_m', 'AUPRC_std',
                     'MC_m', 'MC_std', 'grid_search', 'use_single', 'n_split',
                     'balance_strat', 'threshold', 'comb_type', 'process'])

    if model_type=='GCATSL':
        model = GCATSL(random_state=124, n_jobs=- 1,
                 verbose=False, thold=0.5, process=True, is_ready_data=False,
                 use_single=True, use_comb=True, grid_searched=False, sep_train_test=False, balance=True,
                 balance_strategy=balance_strategy, return_model=False, pretrained_model=None, cancer=cancer,
                   feature_dim=feature_dim, n_feature= n_feature, dataset_name = data_choices)
        model_loc = data_choices + '_' + cancer + '_' + str(grid_search) + '_' + str(n_split) + '_' + balance_strategy + '_' + str(feature_dim) + '.pickle'
    if model_type=='GRSMF':
        model = GRSMF(random_state=124, n_jobs=- 1,
                 verbose=False, thold=0.5, process=True, grid_searched=False, sep_train_test=False, balance=True,
                 balance_strategy=balance_strategy, return_model=False, cancer=cancer,  dataset_name = data_choices)
        model_loc = data_choices + '_' + cancer + '_' + str(grid_search) + '_' + str(n_split) + '_' + balance_strategy+ '.pickle'
    elif model_type=='ELRRF' or model_type=='ELGBDT' :
        use_comb = False
        use_all_comb = True
        model = ELRRF(boosting_type=boosting, use_single=use_single, grid_searched=grid_search, balance_strategy=balance_strategy,
                      use_comb=use_comb, thold=thold, process=process, cancer=cancer)
        model_loc = data_choices + '_' + cancer + '_' + str(grid_search) + '_' + str(
            use_single) + '_' + comb_type + '_' + str(n_split) + \
                    '_' + balance_strategy + '.pickle'

    model_loc = config.ROOT_DIR / 'results' / model_type / 'models_test' / model_loc
    config.ensure_dir(model_loc)
    if os.path.isfile(model_loc):
        results = dfnc.load_pickle(model_loc)
    else:
        tr_samples_loc = 'labels/train_pairs.csv'
        tr_samples_loc = config.DATA_DIR / tr_samples_loc
        tr_samples = pd.read_csv(tr_samples_loc)
        te_samples_loc = 'labels/test_pairs.csv'
        te_samples_loc = config.DATA_DIR / te_samples_loc
        te_samples = pd.read_csv(te_samples_loc)
        samples_tr = prepare_cancer_dataset(tr_samples, cancer=cancer)
        samples_te = prepare_cancer_dataset(te_samples, cancer=cancer)
        #If model is Early Late
        if 'EL' in model_type:
            if comb_type == 'type2' and len(data_choice_list) > 1:
                final_trdata = samples_tr.copy()
                final_tedata = samples_te.copy()
            for data_choice in data_choice_list:
                tr_data = pd.read_csv(loc_dict['train_' + data_choice + '_data_loc'])
                tr_data = tr_data.fillna(0)
                te_data = pd.read_csv(loc_dict['test_' + data_choice + '_data_loc'])
                te_data = te_data.fillna(0)
                processed_data_tr = prepare_cancer_dataset(tr_data, cancer=cancer)
                processed_data_te = prepare_cancer_dataset(te_data, cancer=cancer)
                model.add_dataset(data_choice, processed_data_tr, processed_data_te)
                if comb_type == 'type2' and len(data_choice_list) > 1:
                    processed_data_tr = processed_data_tr.drop(columns='class')
                    final_trdata = pd.merge(final_trdata, processed_data_tr, how='inner', on=['pair_name'])
                    processed_data_te = processed_data_te.drop(columns='class')
                    final_tedata = pd.merge(final_tedata, processed_data_te, how='inner', on=['pair_name'])
            if comb_type=='type2' and len(data_choice_list) > 1:
                model.add_dataset('&'.join(data_choice_list), final_trdata,final_tedata)

        if 'GCATSL' in task:
            param_list = ['args_dict', 'random_state',
                          'n_jobs', 'thold', 'process', 'grid_searched',
                          'sep_train_test', 'balance', 'balance_strategy', 'cancer']
            model.args_dict['input_dir'] = GCATSL_root+'data/SL_'+cancer+'/'
            model.args_dict['output_dir'] = GCATSL_root+'output/SL_'+cancer+'/'
            model.args_dict['log_dir'] = GCATSL_root+'logs/SL_'+cancer+'/'
            model.args_dict['id2name_dir'] = config.DATA_DIR / 'feature_sets' / 'GCATSL' / cancer
            config.ensure_dir(model.args_dict['input_dir']+'/deneme.csv')
            config.ensure_dir(model.args_dict['output_dir']+'/deneme.csv')
            config.ensure_dir(model.args_dict['log_dir']+'/deneme.csv')
        elif 'GRSMF' in task:
            param_list = ['lambda_d', 'beta', 'max_iter', 'seed', 'random_state',
                          'n_jobs', 'thold', 'process', 'grid_searched',
                          'sep_train_test', 'balance', 'balance_strategy', 'cancer']
            model.args_dict['id2name_dir'] = config.DATA_DIR / 'feature_sets' / 'GRSMF' / cancer

        elif 'EL' in task:
            param_list = ['boosting_type', 'num_leaves', 'max_depth', 'learning_rate', 'n_estimators',
                          'subsample_for_bin', 'objective', 'class_weight', 'min_split_gain',
                          'min_child_weight', 'min_child_samples', 'subsample', 'subsample_freq',
                          'colsample_bytree', 'reg_alpha', 'reg_lambda', 'random_state', 'n_jobs',
                          'verbose', 'thold', 'process', 'is_ready_data',
                          'use_single', 'use_comb', 'grid_searched', 'sep_train_test', 'balance',
                          'balance_strategy', 'return_model', 'contr']


        results = fit_predict_2set(model, samples_tr, samples_te, n_split=n_split)
        all_params = model.get_params(deep=False)
        selected_params = {}
        for key in param_list:
            selected_params[key] = all_params.pop(key)
        model_dict = {'params': selected_params}
        model_dict.update(results)
        dfnc.save_pickle(model_loc, model_dict)

    AUROC, AUPRC, MC = model.evaluate_folds(results)
    if model_type == 'GCATSL':
        res_dict = {'datasets': data_choices, 'cancer': cancer,
                    'AUROC_m': np.around(AUROC.mean(), 2), 'AUROC_std': np.around(np.std(AUROC), 2),
                    'AUPRC_m': np.around(AUPRC.mean(), 2), 'AUPRC_std': np.around(np.std(AUPRC), 2),
                    'MC_m': np.around(MC.mean(), 2), 'MC_std': np.around(np.std(MC), 2),
                    'grid_search': grid_search, 'n_split': str(int(n_split)),
                    'balance_strat': balance_strategy, 'feature_dim':feature_dim, 'n_feature': n_feature}
    elif model_type == 'GRSMF':
        res_dict = {'datasets': data_choices, 'cancer': cancer,
                    'AUROC_m': np.around(AUROC.mean(), 2), 'AUROC_std': np.around(np.std(AUROC), 2),
                    'AUPRC_m': np.around(AUPRC.mean(), 2), 'AUPRC_std': np.around(np.std(AUPRC), 2),
                    'MC_m': np.around(MC.mean(), 2), 'MC_std': np.around(np.std(MC), 2),
                    'grid_search': grid_search, 'n_split': str(int(n_split)),
                    'balance_strat': balance_strategy, 'lambda_d':lambda_d, 'beta': beta, 'max_iter': max_iter}
    else:
        res_dict = {'datasets': data_choices, 'cancer': cancer,
                'AUROC_m': np.around(AUROC.mean(), 2), 'AUROC_std': np.around(np.std(AUROC), 2),
                'AUPRC_m': np.around(AUPRC.mean(), 2), 'AUPRC_std': np.around(np.std(AUPRC), 2),
                'MC_m': np.around(MC.mean(), 2), 'MC_std': np.around(np.std(MC), 2),
                'grid_search': grid_search, 'use_single': use_single, 'n_split': n_split,
                    'balance_strat': balance_strategy,
                'threshold': thold, 'comb_type': comb_type, 'process': process}

    if os.path.isfile(res_loc):
        result_df = pd.read_csv(res_loc)
    result_df = result_df.append(res_dict, ignore_index=True)

    result_df.to_csv(res_loc, index=False)

    # fold_predictions = elrrf.fit_predict_all_datasets_fold(train_test_sample_dict=train_test_sample_names[cancer])
    # elrrf.evaluate_folds(fold_predictions)

    print(f'Experiment with {data_choice_list} finished for {cancer}...')
    print()


def single_cancer_dho_test_experiment(data_choice_list, cancer, grid_search, thold=None):
    model_type='ELRRF'
    comb_type='type2'
    boosting='rf'
    if 'GCATSL' in task:
        model_type='GCATSL'
        data_choice_list=['PPI', 'CC', 'BP']#, 'GO1', 'GO2']
        feature_dim = 128
        n_feature = len(data_choice_list)
    elif 'GRSMF' in task:
        model_type='GRSMF'
        data_choice_list=['BP']#, 'GO1', 'GO2']
        lambda_d = 2 ** (-7)
        beta = 2 ** (-5)
        max_iter = 10
    else:
        use_single=True
        if 'GBDT' in task:
            model_type='ELGBDT'
            boosting='gbdt'

    data_choices = '|'.join(data_choice_list)
    print(f'Experiment: {data_choice_list} for {cancer} with grid_search={grid_search}')
    if 'trus' in task:
        balance_strategy = 'undersample_train'
    if 'trteus' in task:
        balance_strategy = 'undersample_train_test'

    n_split=10
    process=True
    if thold == None:
        thold = 0.5

    loc_dict = get_locs()
    res_loc = config.ROOT_DIR / 'results' / model_type / 'single_cancer_dho2_test.csv'
    config.ensure_dir(res_loc)
    if os.path.isfile(res_loc):
        result_df = pd.read_csv(res_loc)
        if model_type=='GCATSL':
            chosen_cols = result_df[
                ['datasets', 'cancer', 'grid_search', 'n_split', 'balance_strat', 'feature_dim', 'n_feature']]
            chosen_cols = chosen_cols.astype(str)
            chosen_vals = np.array([data_choices, cancer, grid_search, n_split, balance_strategy, feature_dim, n_feature])
            chosen_vals = chosen_vals.astype(str)
        elif model_type=='GRSMF':
            chosen_cols = result_df[
                ['datasets', 'cancer', 'grid_search', 'n_split', 'balance_strat', 'lambda_d', 'beta', 'max_iter']]
            chosen_cols = chosen_cols.astype(str)
            chosen_vals = np.array([data_choices, cancer, grid_search, n_split, balance_strategy, lambda_d, beta, max_iter])
            chosen_vals = chosen_vals.astype(str)
        else:
            chosen_cols = result_df[
                ['datasets', 'cancer', 'grid_search', 'use_single', 'n_split', 'balance_strat', 'threshold',
                 'comb_type', 'process']]
            chosen_cols = chosen_cols.astype(str)
            chosen_vals = np.array([data_choices, cancer, grid_search, use_single, n_split, balance_strategy, thold,
                                    comb_type, process])
            chosen_vals = chosen_vals.astype(str)

        if (chosen_cols == chosen_vals).all(1).any():
            print(f'{data_choices} is already calculated!')
            return 0
    elif model_type=='GCATSL':
        result_df = pd.DataFrame(
            columns=['datasets', 'cancer', 'AUROC_m', 'AUROC_std', 'AUPRC_m', 'AUPRC_std',
                     'MC_m', 'MC_std', 'grid_search', 'n_split',
                     'balance_strat', 'feature_dim', 'n_feature'])
    elif model_type=='GRSMF':
        result_df = pd.DataFrame(
            columns=['datasets', 'cancer', 'AUROC_m', 'AUROC_std', 'AUPRC_m', 'AUPRC_std',
                     'MC_m', 'MC_std','grid_search', 'n_split', 'balance_strat', 'lambda_d', 'beta', 'max_iter'])
    else:
        result_df = pd.DataFrame(
            columns=['datasets', 'cancer', 'AUROC_m', 'AUROC_std', 'AUPRC_m', 'AUPRC_std',
                     'MC_m', 'MC_std', 'grid_search', 'use_single', 'n_split',
                     'balance_strat', 'threshold', 'comb_type', 'process'])

    if model_type=='GCATSL':
        model = GCATSL(random_state=124, n_jobs=- 1,
                 verbose=False, thold=0.5, process=True, is_ready_data=False,
                 use_single=True, use_comb=True, grid_searched=False, sep_train_test=False, balance=True,
                 balance_strategy=balance_strategy, return_model=False, pretrained_model=None, cancer=cancer,
                   feature_dim=feature_dim, n_feature= n_feature, dataset_name = data_choices)
        model_loc = data_choices + '_' + cancer + '_' + str(grid_search) + '_' + str(n_split) + '_' + balance_strategy + '_' + str(feature_dim) + '.pickle'
    if model_type=='GRSMF':
        model = GRSMF(random_state=124, n_jobs=- 1,
                 verbose=False, thold=0.5, process=True, grid_searched=False, sep_train_test=False, balance=True,
                 balance_strategy=balance_strategy, return_model=False, cancer=cancer,  dataset_name = data_choices)
        model_loc = data_choices + '_' + cancer + '_' + str(grid_search) + '_' + str(n_split) + '_' + balance_strategy+ '.pickle'
    elif model_type=='ELRRF' or model_type=='ELGBDT' :
        use_comb = False
        use_all_comb = True
        model = ELRRF(boosting_type=boosting, use_single=use_single, grid_searched=grid_search, balance_strategy=balance_strategy,
                      use_comb=use_comb, thold=thold, process=process, cancer=cancer)
        model_loc = data_choices + '_' + cancer + '_' + str(grid_search) + '_' + str(
            use_single) + '_' + comb_type + '_' + str(n_split) + \
                    '_' + balance_strategy + '.pickle'

    model_loc = config.ROOT_DIR / 'results' / model_type / 'models_dho2_test' / model_loc
    config.ensure_dir(model_loc)
    if os.path.isfile(model_loc):
        results = dfnc.load_pickle(model_loc)
    else:
        tr_samples_loc = 'labels/train_pairs.csv'
        tr_samples_loc = config.DATA_DIR / tr_samples_loc
        tr_samples = pd.read_csv(tr_samples_loc)
        te_samples_loc = 'labels/test_pairs.csv'
        te_samples_loc = config.DATA_DIR / te_samples_loc
        te_samples = pd.read_csv(te_samples_loc)
        samples_tr = prepare_cancer_dataset(tr_samples, cancer=cancer)
        samples_te = prepare_cancer_dataset(te_samples, cancer=cancer)
        samples = pd.concat([samples_tr, samples_te])
        #If model is Early Late
        if 'EL' in model_type:
            if comb_type == 'type2' and len(data_choice_list) > 1:
                final_trdata = samples_tr.copy()
                final_tedata = samples_te.copy()
            for data_choice in data_choice_list:
                tr_data = pd.read_csv(loc_dict['train_' + data_choice + '_data_loc'])
                tr_data = tr_data.fillna(0)
                te_data = pd.read_csv(loc_dict['test_' + data_choice + '_data_loc'])
                te_data = te_data.fillna(0)
                processed_data_tr = prepare_cancer_dataset(tr_data, cancer=cancer)
                processed_data_te = prepare_cancer_dataset(te_data, cancer=cancer)
                processed_data = pd.concat([processed_data_tr, processed_data_te])
                model.add_dataset(data_choice, processed_data)
                if comb_type == 'type2' and len(data_choice_list) > 1:
                    processed_data_tr = processed_data_tr.drop(columns='class')
                    final_trdata = pd.merge(final_trdata, processed_data_tr, how='inner', on=['pair_name'])
                    processed_data_te = processed_data_te.drop(columns='class')
                    final_tedata = pd.merge(final_tedata, processed_data_te, how='inner', on=['pair_name'])
            if comb_type=='type2' and len(data_choice_list) > 1:
                model.add_dataset('&'.join(data_choice_list), pd.concat([final_trdata, final_tedata]))

        if 'GCATSL' in task:
            model.args_dict['input_dir'] = GCATSL_root+'data/SL_dho_'+cancer+'/'
            model.args_dict['output_dir'] = GCATSL_root+'output/SL_dho_'+cancer+'/'
            model.args_dict['log_dir'] = GCATSL_root+'logs/SL_dho_'+cancer+'/'
            config.ensure_dir(model.args_dict['input_dir']+'/deneme.csv')
            config.ensure_dir(model.args_dict['output_dir']+'/deneme.csv')
            config.ensure_dir(model.args_dict['log_dir']+'/deneme.csv')
            #model_dict = {'GCATSL_params': model.args_dict}
            param_list = ['args_dict', 'random_state',
                          'n_jobs', 'thold', 'process', 'grid_searched',
                          'sep_train_test', 'balance', 'balance_strategy', 'cancer']
        elif 'GRSMF' in task:
            param_list = ['lambda_d', 'beta', 'max_iter', 'seed', 'random_state',
                          'n_jobs', 'thold', 'process', 'grid_searched',
                          'sep_train_test', 'balance', 'balance_strategy', 'cancer']
        elif 'EL' in task:
            param_list = ['boosting_type', 'num_leaves', 'max_depth', 'learning_rate', 'n_estimators',
                      'subsample_for_bin', 'objective', 'class_weight', 'min_split_gain',
                      'min_child_weight', 'min_child_samples', 'subsample', 'subsample_freq',
                      'colsample_bytree', 'reg_alpha', 'reg_lambda', 'random_state', 'n_jobs',
                      'verbose', 'thold', 'process', 'is_ready_data',
                      'use_single', 'use_comb', 'grid_searched', 'sep_train_test', 'balance',
                      'balance_strategy', 'return_model', 'contr']
        results = fit_predict_dho_cross_validation(model, samples, n_split=n_split)
        all_params = model.get_params(deep=False)
        selected_params = {}
        for key in param_list:
            selected_params[key] = all_params.pop(key)
        model_dict = {'params': selected_params}
        model_dict.update(results)
        dfnc.save_pickle(model_loc, model_dict)

    AUROC, AUPRC, MC = model.evaluate_folds(results)
    if model_type == 'GCATSL':
        res_dict = {'datasets': data_choices, 'cancer': cancer,
                    'AUROC_m': np.around(AUROC.mean(), 2), 'AUROC_std': np.around(np.std(AUROC), 2),
                    'AUPRC_m': np.around(AUPRC.mean(), 2), 'AUPRC_std': np.around(np.std(AUPRC), 2),
                    'MC_m': np.around(MC.mean(), 2), 'MC_std': np.around(np.std(MC), 2),
                    'grid_search': grid_search, 'n_split': str(int(n_split)),
                    'balance_strat': balance_strategy, 'feature_dim':feature_dim, 'n_feature': n_feature}
    elif model_type == 'GRSMF':
        res_dict = {'datasets': data_choices, 'cancer': cancer,
                    'AUROC_m': np.around(AUROC.mean(), 2), 'AUROC_std': np.around(np.std(AUROC), 2),
                    'AUPRC_m': np.around(AUPRC.mean(), 2), 'AUPRC_std': np.around(np.std(AUPRC), 2),
                    'MC_m': np.around(MC.mean(), 2), 'MC_std': np.around(np.std(MC), 2),
                    'grid_search': grid_search, 'n_split': str(int(n_split)),
                    'balance_strat': balance_strategy, 'lambda_d':lambda_d, 'beta': beta, 'max_iter': max_iter}
    else:
        res_dict = {'datasets': data_choices, 'cancer': cancer,
                'AUROC_m': np.around(AUROC.mean(), 2), 'AUROC_std': np.around(np.std(AUROC), 2),
                'AUPRC_m': np.around(AUPRC.mean(), 2), 'AUPRC_std': np.around(np.std(AUPRC), 2),
                'MC_m': np.around(MC.mean(), 2), 'MC_std': np.around(np.std(MC), 2),
                'grid_search': grid_search, 'use_single': use_single, 'n_split': n_split,
                    'balance_strat': balance_strategy,
                'threshold': thold, 'comb_type': comb_type, 'process': process}

    if os.path.isfile(res_loc):
        result_df = pd.read_csv(res_loc)
    result_df = result_df.append(res_dict, ignore_index=True)

    result_df.to_csv(res_loc, index=False)

    # fold_predictions = elrrf.fit_predict_all_datasets_fold(train_test_sample_dict=train_test_sample_names[cancer])
    # elrrf.evaluate_folds(fold_predictions)

    print(f'Experiment with {data_choice_list} finished for {cancer}...')
    print()


def cross_dataset_test_experiment(data_choice_list, cancer, train_ds, test_ds, grid_search, thold=None):
    model_type='ELRRF'
    comb_type='type2'
    boosting='rf'
    if 'GCATSL' in task:
        model_type='GCATSL'
        data_choice_list=['PPI', 'CC', 'BP']#, 'GO1', 'GO2']
        feature_dim = 128
        n_feature = len(data_choice_list)
    elif 'GRSMF' in task:
        model_type='GRSMF'
        data_choice_list=['BP']#, 'GO1', 'GO2']
        lambda_d = 2 ** (-7)
        beta = 2 ** (-5)
        max_iter = 10
    else:
        use_single=True
        if 'GBDT' in task:
            model_type='ELGBDT'
            boosting='gbdt'

    data_choices = '|'.join(data_choice_list)
    print(f'Experiment: {data_choice_list} for {cancer}  from {train_ds} to {test_ds} with grid_search={grid_search}')
    #print(f'Experiment {model_type}: for {cancer}, grid_search={grid_search}')
    if 'trus' in task:
        balance_strategy = 'undersample_train'
    if 'trteus' in task:
        balance_strategy = 'undersample_train_test'

    n_split=10
    process=True
    if thold == None:
        thold = 0.5

    loc_dict = get_locs()
    res_loc = config.ROOT_DIR / 'results' / model_type / 'cross_dataset_test.csv'
    config.ensure_dir(res_loc)
    grid_search_settings_loc = config.ROOT_DIR / 'results' / model_type / 'result.json'
    if os.path.isfile(res_loc):
        result_df = pd.read_csv(res_loc)
        if model_type=='GCATSL':
            chosen_cols = result_df[
                ['datasets', 'cancer', 'train_ds', 'test_ds', 'grid_search', 'n_split', 'balance_strat', 'feature_dim', 'n_feature']]
            chosen_cols = chosen_cols.astype(str)
            chosen_vals = np.array([data_choices, cancer, train_ds, test_ds,  grid_search, n_split, balance_strategy, feature_dim, n_feature])
            chosen_vals = chosen_vals.astype(str)
        elif model_type=='GRSMF':
            chosen_cols = result_df[
                ['datasets', 'cancer', 'train_ds', 'test_ds', 'grid_search', 'n_split', 'balance_strat', 'lambda_d', 'beta', 'max_iter']]
            chosen_cols = chosen_cols.astype(str)
            chosen_vals = np.array([data_choices, cancer, train_ds, test_ds,  grid_search, n_split, balance_strategy, lambda_d, beta, max_iter])
            chosen_vals = chosen_vals.astype(str)
        else:
            chosen_cols = result_df[
                ['datasets', 'cancer', 'train_ds', 'test_ds', 'grid_search', 'use_single', 'n_split', 'balance_strat', 'threshold',
                 'comb_type', 'process']]
            chosen_cols = chosen_cols.astype(str)
            chosen_vals = np.array([data_choices, cancer, train_ds, test_ds, grid_search, use_single, n_split, balance_strategy, thold,
                                    comb_type, process])
            chosen_vals = chosen_vals.astype(str)

        if (chosen_cols == chosen_vals).all(1).any():
            print(f'{data_choices} is already calculated!')
            return 0
    elif model_type=='GCATSL':
        result_df = pd.DataFrame(
            columns=['datasets', 'cancer', 'train_ds', 'test_ds', 'AUROC_m', 'AUROC_std', 'AUPRC_m', 'AUPRC_std',
                     'MC_m', 'MC_std', 'grid_search', 'n_split',
                     'balance_strat', 'feature_dim', 'n_feature'])
    elif model_type=='GRSMF':
        result_df = pd.DataFrame(
            columns=['datasets', 'cancer', 'train_ds', 'test_ds', 'AUROC_m', 'AUROC_std', 'AUPRC_m', 'AUPRC_std',
                     'MC_m', 'MC_std','grid_search', 'n_split', 'balance_strat', 'lambda_d', 'beta', 'max_iter'])
    else:
        result_df = pd.DataFrame(
            columns=['datasets', 'cancer', 'train_ds', 'test_ds', 'AUROC_m', 'AUROC_std', 'AUPRC_m', 'AUPRC_std',
                     'MC_m', 'MC_std', 'grid_search', 'use_single', 'n_split',
                     'balance_strat', 'threshold', 'comb_type', 'process'])

    if model_type=='GCATSL':
        model = GCATSL(random_state=124, n_jobs=- 1,
                 verbose=False, thold=0.5, process=True, is_ready_data=False,
                 use_single=True, use_comb=True, grid_searched=False, sep_train_test=False, balance=True,
                 balance_strategy=balance_strategy, return_model=False, pretrained_model=None, cancer=cancer,
                   feature_dim=feature_dim, n_feature= n_feature, dataset_name = data_choices)
        model_loc = data_choices + '_' + cancer + '_' + train_ds + '_' + test_ds + '_' + str(grid_search) + '_' + str(n_split) + '_' + balance_strategy + '_' + str(feature_dim) + '.pickle'
    if model_type=='GRSMF':
        model = GRSMF(random_state=124, n_jobs=- 1,
                 verbose=False, thold=0.5, process=True, grid_searched=False, sep_train_test=False, balance=True,
                 balance_strategy=balance_strategy, return_model=False, cancer=cancer,  dataset_name = data_choices)
        model_loc = data_choices + '_' + cancer + '_' + train_ds + '_' + test_ds + '_' + str(grid_search) + '_' + str(n_split) + '_' + balance_strategy+ '.pickle'
    elif model_type=='ELRRF' or model_type=='ELGBDT' :
        use_comb = False
        use_all_comb = True
        model = ELRRF(boosting_type=boosting, use_single=use_single, grid_searched=grid_search, balance_strategy=balance_strategy,
                      use_comb=use_comb, thold=thold, process=process, cancer=cancer)
        model_loc = data_choices + '_' + cancer + '_' + train_ds + '_' + test_ds + '_' + str(grid_search) + '_' + str(
            use_single) + '_' + comb_type + '_' + str(n_split) + \
                    '_' + balance_strategy + '.pickle'

    model_loc = config.ROOT_DIR / 'results' / model_type / 'models_cross_ds' / model_loc
    config.ensure_dir(model_loc)
    if os.path.isfile(model_loc):
        results = dfnc.load_pickle(model_loc)
    else:
        tr_samples_loc = 'labels/'+train_ds+'_pairs.csv'
        tr_samples_loc = config.DATA_DIR / tr_samples_loc
        tr_samples = pd.read_csv(tr_samples_loc)
        te_samples_loc = 'labels/'+test_ds+'_pairs.csv'
        te_samples_loc = config.DATA_DIR / te_samples_loc
        te_samples = pd.read_csv(te_samples_loc)
        samples_tr = prepare_cancer_dataset(tr_samples, cancer=cancer)
        samples_te = prepare_cancer_dataset(te_samples, cancer=cancer)
        if 'unknown' not in test_ds and 'negative' not in test_ds:
            samples_tr = samples_tr[~samples_tr['pair_name'].isin(samples_te['pair_name'].values)]
        #If model is Early Late
        if 'EL' in model_type:
            if comb_type == 'type2' and len(data_choice_list) > 1:
                final_trdata = samples_tr.copy()
                final_tedata = samples_te.copy()
            for data_choice in data_choice_list:
                tr_data = pd.read_csv(loc_dict[train_ds+'_' + data_choice + '_data_loc'])
                tr_data = tr_data.fillna(0)
                te_data = pd.read_csv(loc_dict[test_ds+'_' + data_choice + '_data_loc'])
                te_data = te_data.fillna(0)
                processed_data_tr = prepare_cancer_dataset(tr_data, cancer=cancer)
                processed_data_te = prepare_cancer_dataset(te_data, cancer=cancer)
                if 'unknown' not in test_ds and 'negative' not in test_ds:
                    processed_data_tr = processed_data_tr[~processed_data_tr['pair_name'].isin(processed_data_te['pair_name'].values)]
                model.add_dataset(data_choice, processed_data_tr, processed_data_te)
                if comb_type == 'type2' and len(data_choice_list) > 1:
                    processed_data_tr = processed_data_tr.drop(columns='class')
                    final_trdata = pd.merge(final_trdata, processed_data_tr, how='inner', on=['pair_name'])
                    processed_data_te = processed_data_te.drop(columns='class')
                    final_tedata = pd.merge(final_tedata, processed_data_te, how='inner', on=['pair_name'])
            if comb_type=='type2' and len(data_choice_list) > 1:
                model.add_dataset('&'.join(data_choice_list), final_trdata,final_tedata)

        if 'GCATSL' in task:
            model.args_dict['input_dir'] = GCATSL_root+'data/SL_partial_'+cancer+'/'
            model.args_dict['output_dir'] = GCATSL_root+'output/SL_partial_'+cancer+'/'
            model.args_dict['log_dir'] = GCATSL_root+'logs/SL_partial_'+cancer+'/'
            model.args_dict['id2name_dir'] = config.DATA_DIR / 'feature_sets' / 'GCATSL' / cancer
            config.ensure_dir(model.args_dict['input_dir']+'/deneme.csv')
            config.ensure_dir(model.args_dict['output_dir']+'/deneme.csv')
            config.ensure_dir(model.args_dict['log_dir']+'/deneme.csv')
            param_list = ['args_dict', 'random_state',
                          'n_jobs', 'thold', 'process', 'grid_searched',
                          'sep_train_test', 'balance', 'balance_strategy', 'cancer']
            #model_dict = {'GCATSL_params': model.args_dict}
        elif 'GRSMF' in task:
            param_list = ['lambda_d', 'beta', 'max_iter', 'seed', 'random_state',
                          'n_jobs', 'thold', 'process', 'grid_searched',
                          'sep_train_test', 'balance', 'balance_strategy', 'cancer']
            model.args_dict['id2name_dir'] = config.DATA_DIR / 'feature_sets' / 'GRSMF' / cancer
        elif 'EL' in task:
            param_list = ['boosting_type', 'num_leaves', 'max_depth', 'learning_rate', 'n_estimators',
                          'subsample_for_bin', 'objective', 'class_weight', 'min_split_gain',
                          'min_child_weight', 'min_child_samples', 'subsample', 'subsample_freq',
                          'colsample_bytree', 'reg_alpha', 'reg_lambda', 'random_state', 'n_jobs',
                          'verbose', 'thold', 'process', 'is_ready_data',
                          'use_single', 'use_comb', 'grid_searched', 'sep_train_test', 'balance',
                          'balance_strategy', 'return_model', 'contr']
        results = fit_predict_2set(model, samples_tr, samples_te, n_split=n_split, ds_names=[train_ds, test_ds])

        all_params = model.get_params(deep=False)
        selected_params = {}
        for key in param_list:
            selected_params[key] = all_params.pop(key)
        model_dict = {'params': selected_params}
        model_dict.update(results)
        dfnc.save_pickle(model_loc, model_dict)

    AUROC, AUPRC, MC = model.evaluate_folds(results)
    if model_type == 'GCATSL':
        res_dict = {'datasets': data_choices, 'cancer': cancer, 'train_ds': train_ds, 'test_ds':test_ds,
                    'AUROC_m': np.around(AUROC.mean(), 2), 'AUROC_std': np.around(np.std(AUROC), 2),
                    'AUPRC_m': np.around(AUPRC.mean(), 2), 'AUPRC_std': np.around(np.std(AUPRC), 2),
                    'MC_m': np.around(MC.mean(), 2), 'MC_std': np.around(np.std(MC), 2),
                    'grid_search': grid_search, 'n_split': str(int(n_split)),
                    'balance_strat': balance_strategy, 'feature_dim':feature_dim, 'n_feature': n_feature}
    elif model_type == 'GRSMF':
        res_dict = {'datasets': data_choices, 'cancer': cancer, 'train_ds': train_ds, 'test_ds':test_ds,
                    'AUROC_m': np.around(AUROC.mean(), 2), 'AUROC_std': np.around(np.std(AUROC), 2),
                    'AUPRC_m': np.around(AUPRC.mean(), 2), 'AUPRC_std': np.around(np.std(AUPRC), 2),
                    'MC_m': np.around(MC.mean(), 2), 'MC_std': np.around(np.std(MC), 2),
                    'grid_search': grid_search, 'n_split': str(int(n_split)),
                    'balance_strat': balance_strategy, 'lambda_d':lambda_d, 'beta': beta, 'max_iter': max_iter}
    else:
        res_dict = {'datasets': data_choices, 'cancer': cancer, 'train_ds': train_ds, 'test_ds':test_ds,
                'AUROC_m': np.around(AUROC.mean(), 2), 'AUROC_std': np.around(np.std(AUROC), 2),
                'AUPRC_m': np.around(AUPRC.mean(), 2), 'AUPRC_std': np.around(np.std(AUPRC), 2),
                'MC_m': np.around(MC.mean(), 2), 'MC_std': np.around(np.std(MC), 2),
                'grid_search': grid_search, 'use_single': use_single, 'n_split': n_split,
                    'balance_strat': balance_strategy,
                'threshold': thold, 'comb_type': comb_type, 'process': process}

    if os.path.isfile(res_loc):
        result_df = pd.read_csv(res_loc)
    result_df = result_df.append(res_dict, ignore_index=True)

    result_df.to_csv(res_loc, index=False)

    # fold_predictions = elrrf.fit_predict_all_datasets_fold(train_test_sample_dict=train_test_sample_names[cancer])
    # elrrf.evaluate_folds(fold_predictions)

    print(f'Experiment with {data_choice_list} finished for {cancer}...')
    print()


def cross_cancer_test_experiment(data_choice_list, cancer_train, cancer_test, grid_search, thold=None):
    model_type='ELRRF'
    comb_type='type2'
    boosting='rf'
    use_single=True
    if 'GBDT' in task:
        model_type='ELGBDT'
        boosting='gbdt'

    data_choices = '|'.join(data_choice_list)
    print(f'Experiment: {data_choice_list} for {cancer_train} to {cancer_test} with grid_search={grid_search}')
    #print(f'Experiment {model_type}: for {cancer}, grid_search={grid_search}')
    if 'trus' in task:
        balance_strategy = 'undersample_train'
    if 'trteus' in task:
        balance_strategy = 'undersample_train_test'

    n_split=10
    process=True
    if thold == None:
        thold = 0.5
    testing_set = 'train'

    loc_dict = get_locs()
    res_loc = config.ROOT_DIR / 'results' / model_type / ('cross_cancer_train_train_test_'+testing_set+'.csv')
    config.ensure_dir(res_loc)
    grid_search_settings_loc = config.ROOT_DIR / 'results' / model_type / 'result.json'
    if os.path.isfile(res_loc):
        result_df = pd.read_csv(res_loc)
        chosen_cols = result_df[
            ['datasets', 'cancer_train', 'cancer_test', 'grid_search', 'use_single', 'n_split', 'balance_strat', 'threshold',
             'comb_type', 'process']]
        chosen_cols = chosen_cols.astype(str)
        chosen_vals = np.array([data_choices, cancer_train, cancer_test, grid_search, use_single, n_split, balance_strategy, thold,
                                comb_type, process])
        chosen_vals = chosen_vals.astype(str)
        if (chosen_cols == chosen_vals).all(1).any():
            print(f'{data_choices} is already calculated!')
            return 0
    else:
        result_df = pd.DataFrame(
            columns=['datasets', 'cancer_train', 'cancer_test', 'AUROC_m', 'AUROC_std', 'AUPRC_m', 'AUPRC_std',
                     'MC_m', 'MC_std', 'grid_search', 'use_single', 'n_split',
                     'balance_strat', 'threshold', 'comb_type', 'process'])

    use_comb = False
    use_all_comb = True
    model = ELRRF(boosting_type=boosting, use_single=use_single, grid_searched=grid_search, balance_strategy=balance_strategy,
                  use_comb=use_comb, thold=thold, process=process, cancer=cancer_train)
    model_loc = data_choices + '_' + cancer_train + '_' + cancer_test + "_" + str(grid_search) + '_' + str(
        use_single) + '_' + comb_type + '_' + str(n_split) + \
                '_' + balance_strategy + '.pickle'

    model_loc = config.ROOT_DIR / 'results' / model_type / ('models_cc_train_train_test_'+testing_set) / model_loc
    config.ensure_dir(model_loc)
    if os.path.isfile(model_loc):
        results = dfnc.load_pickle(model_loc)
    else:
        tr_samples_loc = 'labels/train_pairs.csv'
        tr_samples_loc = config.DATA_DIR / tr_samples_loc
        te_samples_loc = 'labels/test_pairs.csv'
        te_samples_loc = config.DATA_DIR / te_samples_loc
        tr_samples = pd.read_csv(tr_samples_loc)
        samples_tr = prepare_cancer_dataset(tr_samples, cancer=cancer_train)
        if testing_set=='train':
            tr2_samples = tr_samples.copy()
            samples_te = prepare_cancer_dataset(tr2_samples, cancer=cancer_test)
        elif testing_set=='test':
            te_samples = pd.read_csv(te_samples_loc)
            samples_te = prepare_cancer_dataset(te_samples, cancer=cancer_test)
        elif testing_set=='all':
            te_samples = pd.read_csv(te_samples_loc)
            samples_te1 = prepare_cancer_dataset(te_samples, cancer=cancer_test)
            tr2_samples = tr_samples.copy()
            samples_te2 = prepare_cancer_dataset(tr2_samples, cancer=cancer_test)
            samples_te = pd.concat([samples_te1, samples_te2])
        if comb_type == 'type2' and len(data_choice_list) > 1:
            final_trdata = samples_tr.copy()
            final_tedata = samples_te.copy()
        for data_choice in data_choice_list:
            tr_data = pd.read_csv(loc_dict['train_' + data_choice + '_data_loc'])
            tr_data = tr_data.fillna(0)
            processed_data_tr = prepare_cancer_dataset(tr_data, cancer=cancer_train)

            if testing_set == 'train':
                tr2_data = tr_data.copy()
                processed_data_te = prepare_cancer_dataset(tr2_data, cancer=cancer_test)
            elif testing_set == 'test':
                te_data = pd.read_csv(loc_dict['test_' + data_choice + '_data_loc'])
                te_data = te_data.fillna(0)
                processed_data_te = prepare_cancer_dataset(te_data, cancer=cancer_test)
            elif testing_set == 'all':
                te_data = pd.read_csv(loc_dict['test_' + data_choice + '_data_loc'])
                te_data = te_data.fillna(0)
                processed_data_te1 = prepare_cancer_dataset(te_data, cancer=cancer_test)
                tr2_data = tr_data.copy()
                processed_data_te2 = prepare_cancer_dataset(tr2_data, cancer=cancer_test)
                processed_data_te = pd.concat([processed_data_te1, processed_data_te2])
            model.add_dataset(data_choice, processed_data_tr, processed_data_te)
            if comb_type == 'type2' and len(data_choice_list) > 1:
                processed_data_tr = processed_data_tr.drop(columns='class')
                final_trdata = pd.merge(final_trdata, processed_data_tr, how='inner', on=['pair_name'])
                processed_data_te = processed_data_te.drop(columns='class')
                final_tedata = pd.merge(final_tedata, processed_data_te, how='inner', on=['pair_name'])
        if comb_type=='type2' and len(data_choice_list) > 1:
            model.add_dataset('&'.join(data_choice_list), final_trdata,final_tedata)

        results = fit_predict_2set_cc(model, samples_tr, samples_te, n_split=n_split,
                                      fold_usage={cancer_train:'train', cancer_test:testing_set})
        param_list = ['boosting_type', 'num_leaves', 'max_depth', 'learning_rate', 'n_estimators',
                      'subsample_for_bin', 'objective', 'class_weight', 'min_split_gain',
                      'min_child_weight', 'min_child_samples', 'subsample', 'subsample_freq',
                      'colsample_bytree', 'reg_alpha', 'reg_lambda', 'random_state', 'n_jobs',
                      'verbose', 'thold', 'process', 'is_ready_data',
                      'use_single', 'use_comb', 'grid_searched', 'sep_train_test', 'balance',
                      'balance_strategy', 'return_model', 'contr']
        all_params = model.get_params(deep=False)
        selected_params = {}
        for key in param_list:
            selected_params[key] = all_params.pop(key)
        model_dict = {'params': selected_params}
        model_dict.update(results)
        dfnc.save_pickle(model_loc, model_dict)

    AUROC, AUPRC, MC = model.evaluate_folds(results)
    res_dict = {'datasets': data_choices, 'cancer_train': cancer_train, 'cancer_test': cancer_test,
            'AUROC_m': np.around(AUROC.mean(), 2), 'AUROC_std': np.around(np.std(AUROC), 2),
            'AUPRC_m': np.around(AUPRC.mean(), 2), 'AUPRC_std': np.around(np.std(AUPRC), 2),
            'MC_m': np.around(MC.mean(), 2), 'MC_std': np.around(np.std(MC), 2),
            'grid_search': grid_search, 'use_single': use_single, 'n_split': n_split,
                'balance_strat': balance_strategy,
            'threshold': thold, 'comb_type': comb_type, 'process': process}

    if os.path.isfile(res_loc):
        result_df = pd.read_csv(res_loc)
    result_df = result_df.append(res_dict, ignore_index=True)

    result_df.to_csv(res_loc, index=False)

    # fold_predictions = elrrf.fit_predict_all_datasets_fold(train_test_sample_dict=train_test_sample_names[cancer])
    # elrrf.evaluate_folds(fold_predictions)

    print(f'Experiment with {data_choice_list} finished for {cancer_train} to {cancer_test}...')
    print()


def multi_cross_cancer_test_experiment(data_choice_list, cancer_trains, cancer_test, grid_search, thold=None):
    model_type='ELRRF'
    comb_type='type2'
    boosting='rf'
    use_single=True
    if 'GBDT' in task:
        model_type='ELGBDT'
        boosting='gbdt'

    data_choices = '|'.join(data_choice_list)
    print(f'Experiment: {data_choice_list} for {cancer_trains} to {cancer_test} with grid_search={grid_search}')
    #print(f'Experiment {model_type}: for {cancer}, grid_search={grid_search}')
    if 'trus' in task:
        balance_strategy = 'undersample_train'
    if 'trteus' in task:
        balance_strategy = 'undersample_train_test'
    n_split=10
    contr=True
    process=True
    if thold == None:
        thold = 0.5

    loc_dict = get_locs()
    if contr:
        res_loc = config.ROOT_DIR / 'results' / model_type / 'multi_cross_cancer_train_train_test_test.csv'
    else:
        res_loc = config.ROOT_DIR / 'results' / model_type / 'multi_cross_cancer_train_train_test_test_nocontr.csv'
    config.ensure_dir(res_loc)
    if os.path.isfile(res_loc):
        result_df = pd.read_csv(res_loc)
        chosen_cols = result_df[
            ['datasets', 'cancer_test', 'grid_search', 'use_single', 'n_split', 'balance_strat', 'threshold',
             'comb_type', 'process']]
        chosen_cols = chosen_cols.astype(str)
        chosen_vals = np.array([data_choices, cancer_test, grid_search, use_single, n_split, balance_strategy, thold,
                                comb_type, process])
        chosen_vals = chosen_vals.astype(str)
        if (chosen_cols == chosen_vals).all(1).any():
            print(f'{data_choices} is already calculated!')
            return 0
    else:
        result_df = pd.DataFrame(
            columns=['datasets', 'cancer_test', 'AUROC_m', 'AUROC_std', 'AUPRC_m', 'AUPRC_std',
                     'MC_m', 'MC_std', 'grid_search', 'use_single', 'n_split',
                     'balance_strat', 'threshold', 'comb_type', 'process'])

    use_comb = False
    use_all_comb = True
    model = ELRRF(boosting_type=boosting, use_single=use_single, grid_searched=grid_search, balance_strategy=balance_strategy,
                  use_comb=use_comb, thold=thold, process=process, cancer=cancer_test)
    results = {}
    for i in range(n_split):
        results[i] = {}
    dataset_names = data_choice_list.copy()
    dataset_names.append('&'.join(data_choice_list))
    for cancer_train in cancer_trains:
        model_loc = data_choices + '_' + cancer_train + '_' + cancer_test + "_" + str(grid_search) + '_' + str(
            use_single) + '_' + comb_type + '_' + str(n_split) + \
                    '_' + balance_strategy + '.pickle'
        contr_loc = config.ROOT_DIR / 'results' / model_type / 'models_cc_train_train_test_train' / model_loc
        model_loc = config.ROOT_DIR / 'results' / model_type / 'models_cc_train_train_test_test' / model_loc
        cancer_res = dfnc.load_pickle(model_loc)
        if contr:
            cancer_contr_res = dfnc.load_pickle(contr_loc)
            AUROC_contr, AUPRC_contr, MC_contr = model.evaluate_folds(cancer_contr_res, report=False)
        for i in range(n_split):
            for key in dataset_names:
                try:
                    cancer_res[i][cancer_train + '_' + key] = cancer_res[i].pop(key)
                    if contr:
                        cancer_res[i][cancer_train + '_' + key].update(
                            {'tr_auc': cancer_res[i][cancer_train + '_' + key]*AUPRC_contr[i]})
                except:
                    cancer_res[str(i)][cancer_train + '_' + key] = cancer_res[str(i)].pop(key)
                    if contr:
                        cancer_res[str(i)][cancer_train + '_' + key]['tr_auc'] = \
                            cancer_res[str(i)][cancer_train + '_' + key]['tr_auc']*AUPRC_contr[i]

            try:
                results[i].update(cancer_res[i])
            except:
                results[i].update(cancer_res[str(i)])


    AUROC, AUPRC, MC = model.evaluate_folds(results)
    res_dict = {'datasets': data_choices, 'cancer_test': cancer_test,
            'AUROC_m': np.around(AUROC.mean(), 2), 'AUROC_std': np.around(np.std(AUROC), 2),
            'AUPRC_m': np.around(AUPRC.mean(), 2), 'AUPRC_std': np.around(np.std(AUPRC), 2),
            'MC_m': np.around(MC.mean(), 2), 'MC_std': np.around(np.std(MC), 2),
            'grid_search': grid_search, 'use_single': use_single, 'n_split': n_split,
                'balance_strat': balance_strategy,
            'threshold': thold, 'comb_type': comb_type, 'process': process}

    if os.path.isfile(res_loc):
        result_df = pd.read_csv(res_loc)
    result_df = result_df.append(res_dict, ignore_index=True)

    result_df.to_csv(res_loc, index=False)

    # fold_predictions = elrrf.fit_predict_all_datasets_fold(train_test_sample_dict=train_test_sample_names[cancer])
    # elrrf.evaluate_folds(fold_predictions)

    print(f'Experiment with {data_choice_list} finished for {cancer_train} to {cancer_test}...')
    print()


def single_cancer_dataset_imp_detail_experiment(data_choice_list, cancer, grid_search, thold=None):
    model_type='ELRRF'
    comb_type='type2'
    boosting='rf'
    use_single=True
    if 'GBDT' in task:
        model_type='ELGBDT'
        boosting='gbdt'

    data_choices = '|'.join(data_choice_list)
    print(f'Experiment: {data_choice_list} for {cancer} with grid_search={grid_search}')
    #print(f'Experiment {model_type}: for {cancer}, grid_search={grid_search}')
    if 'trus' in task:
        balance_strategy = 'undersample_train'
    if 'trteus' in task:
        balance_strategy = 'undersample_train_test'

    n_split=10
    process=True
    if thold == None:
        thold = 0.5

    cols = ['datasets', 'cancer', 'fold', 'perm_id', 'perm_ds', 'mean', 'std']
    #for d_n in data_choice_list:
    #    cols.append('mean')
    #    cols.append(d_n + '_std')
    #cols.append('&'.join(data_choice_list) + '_m')
    #cols.append('&'.join(data_choice_list) + '_std')
    loc_dict = get_locs()
    res_loc = config.ROOT_DIR / 'results' / model_type / ('single_cancer_'+cancer+'_set_imp_test2_ratio.csv')
    config.ensure_dir(res_loc)
    '''
    if os.path.isfile(res_loc):
        result_df = pd.read_csv(res_loc)
        chosen_cols = result_df[
            ['datasets', 'cancer']]
        chosen_cols = chosen_cols.astype(str)
        chosen_vals = np.array([data_choices, cancer])
        chosen_vals = chosen_vals.astype(str)

        if (chosen_cols == chosen_vals).all(1).any():
            print(f'{data_choices} is already calculated!')
            return 0
    else:
        result_df = pd.DataFrame(columns=cols)
    '''
    use_comb = False
    use_all_comb = True
    model = ELRRF(boosting_type=boosting, use_single=use_single, grid_searched=grid_search, balance_strategy=balance_strategy,
                  use_comb=use_comb, thold=thold, process=process, cancer=cancer)

    tr_samples_loc = 'labels/train_pairs.csv'
    tr_samples_loc = config.DATA_DIR / tr_samples_loc
    tr_samples = pd.read_csv(tr_samples_loc)
    te_samples_loc = 'labels/test_pairs.csv'
    te_samples_loc = config.DATA_DIR / te_samples_loc
    te_samples = pd.read_csv(te_samples_loc)
    samples_tr = prepare_cancer_dataset(tr_samples, cancer=cancer)
    samples_te = prepare_cancer_dataset(te_samples, cancer=cancer)
    #If model is Early Late
    if comb_type == 'type2' and len(data_choice_list) > 1:
        final_trdata = samples_tr.copy()
        final_tedata = samples_te.copy()
    for data_choice in data_choice_list:
        tr_data = pd.read_csv(loc_dict['train_' + data_choice + '_data_loc'])
        tr_data = tr_data.fillna(0)
        te_data = pd.read_csv(loc_dict['test_' + data_choice + '_data_loc'])
        te_data = te_data.fillna(0)
        processed_data_tr = prepare_cancer_dataset(tr_data, cancer=cancer)
        processed_data_te = prepare_cancer_dataset(te_data, cancer=cancer)
        model.add_dataset(data_choice, processed_data_tr, processed_data_te)
        if comb_type == 'type2' and len(data_choice_list) > 1:
            processed_data_tr = processed_data_tr.drop(columns='class')
            final_trdata = pd.merge(final_trdata, processed_data_tr, how='inner', on=['pair_name'])
            processed_data_te = processed_data_te.drop(columns='class')
            final_tedata = pd.merge(final_tedata, processed_data_te, how='inner', on=['pair_name'])
    if comb_type=='type2' and len(data_choice_list) > 1:
        model.add_dataset('&'.join(data_choice_list), final_trdata,final_tedata)

    best_params = {}
    param_list = ['boosting_type', 'num_leaves', 'max_depth', 'learning_rate', 'n_estimators',
                  'subsample_for_bin', 'objective', 'class_weight', 'min_split_gain',
                  'min_child_weight', 'min_child_samples', 'subsample', 'subsample_freq',
                  'colsample_bytree', 'reg_alpha', 'reg_lambda', 'random_state', 'n_jobs',
                  'verbose', 'thold', 'process', 'is_ready_data',
                  'use_single', 'use_comb', 'grid_searched', 'sep_train_test', 'balance',
                  'balance_strategy', 'return_model', 'contr']

    results = calc_dataset_importance(model, samples_tr, samples_te, n_split=n_split, typex='cancer_set2')
    all_params = model.get_params(deep=False)
    selected_params = {}
    for key in param_list:
        selected_params[key] = all_params.pop(key)
    model_dict = {'params': selected_params}
    if best_params:
        model_dict['best_params'] = best_params
    model_dict.update(results)
    cols = ['datasets', 'cancer', 'fold', 'perm_id', 'perm_ds', 'val']
    result_df = pd.DataFrame(columns=cols)

    if os.path.isfile(res_loc):
        result_df = pd.read_csv(res_loc)

    try:
        for fold_no in range(n_split):
            for ds in ['seq_1024', 'ppi_ec', 'crispr_dependency_mut', 'crispr_dependency_expr', 'tissue']:
                for perm_id in range(20):
                    res_dict = {'datasets': data_choices, 'cancer': cancer, 'fold':fold_no, 'perm_id':perm_id,
                                'perm_ds': ds, 'val': results[fold_no][ds][perm_id]}
                    chosen_cols = result_df[['datasets', 'cancer', 'perm_ds']]
                    chosen_cols = chosen_cols.astype(str)
                    chosen_vals = np.array([data_choices, cancer, ds])
                    chosen_vals = chosen_vals.astype(str)
                    if (chosen_cols == chosen_vals).all(1).any():
                        continue
                    else:
                        result_df = result_df.append(res_dict, ignore_index=True)

    except:
        for fold_no in range(n_split):
            fold_no=str(fold_no)
            for ds in ['seq_1024', 'ppi_ec', 'crispr_dependency_mut', 'crispr_dependency_expr', 'tissue']:
                for perm_id in range(20):
                    res_dict = {'datasets': data_choices, 'cancer': cancer, 'fold':fold_no, 'perm_id':perm_id,
                                'perm_ds': ds, 'val': results[fold_no][ds][perm_id]}
                    chosen_cols = result_df[['datasets', 'cancer', 'fold', 'perm_ds', 'perm_id']]
                    chosen_cols = chosen_cols.astype(str)
                    chosen_vals = np.array([data_choices, cancer, fold_no, ds, perm_id])
                    chosen_vals = chosen_vals.astype(str)
                    if (chosen_cols == chosen_vals).all(1).any():
                        pass
                    else:
                        result_df = result_df.append(res_dict, ignore_index=True)

    #result_df = result_df.append(res_dict, ignore_index=True)

    result_df.to_csv(res_loc, index=False)

    # fold_predictions = elrrf.fit_predict_all_datasets_fold(train_test_sample_dict=train_test_sample_names[cancer])
    # elrrf.evaluate_folds(fold_predictions)

    print(f'Experiment with {data_choice_list} finished for {cancer}...')
    print()


def single_cancer_dataset_imp_experiment(data_choice_list, cancer, grid_search, thold=None):
    model_type='ELRRF'
    comb_type='type2'
    boosting='rf'
    use_single=True
    if 'GBDT' in task:
        model_type='ELGBDT'
        boosting='gbdt'

    data_choices = '|'.join(data_choice_list)
    print(f'Experiment: {data_choice_list} for {cancer} with grid_search={grid_search}')
    #print(f'Experiment {model_type}: for {cancer}, grid_search={grid_search}')
    if 'trus' in task:
        balance_strategy = 'undersample_train'
    if 'trteus' in task:
        balance_strategy = 'undersample_train_test'

    n_split=5
    process=True
    if thold == None:
        thold = 0.5

    cols = ['datasets', 'cancer']
    for d_n in data_choice_list:
        cols.append(d_n + '_m')
        cols.append(d_n + '_std')
    cols.append('&'.join(data_choice_list) + '_m')
    cols.append('&'.join(data_choice_list) + '_std')
    loc_dict = get_locs()
    res_loc = config.ROOT_DIR / 'results' / model_type / 'single_cancer_feat_imp_test2.csv'
    config.ensure_dir(res_loc)
    '''
    if os.path.isfile(res_loc):
        result_df = pd.read_csv(res_loc)
        chosen_cols = result_df[
            ['datasets', 'cancer']]
        chosen_cols = chosen_cols.astype(str)
        chosen_vals = np.array([data_choices, cancer])
        chosen_vals = chosen_vals.astype(str)

        if (chosen_cols == chosen_vals).all(1).any():
            print(f'{data_choices} is already calculated!')
            return 0
    else:
        result_df = pd.DataFrame(columns=cols)
    '''
    use_comb = False
    use_all_comb = True
    model = ELRRF(boosting_type=boosting, use_single=use_single, grid_searched=grid_search, balance_strategy=balance_strategy,
                  use_comb=use_comb, thold=thold, process=process, cancer=cancer)

    tr_samples_loc = 'labels/train_pairs.csv'
    tr_samples_loc = config.DATA_DIR / tr_samples_loc
    tr_samples = pd.read_csv(tr_samples_loc)
    te_samples_loc = 'labels/test_pairs.csv'
    te_samples_loc = config.DATA_DIR / te_samples_loc
    te_samples = pd.read_csv(te_samples_loc)
    samples_tr = prepare_cancer_dataset(tr_samples, cancer=cancer)
    samples_te = prepare_cancer_dataset(te_samples, cancer=cancer)
    #If model is Early Late
    if comb_type == 'type2' and len(data_choice_list) > 1:
        final_trdata = samples_tr.copy()
        final_tedata = samples_te.copy()
    for data_choice in data_choice_list:
        tr_data = pd.read_csv(loc_dict['train_' + data_choice + '_data_loc'])
        tr_data = tr_data.fillna(0)
        te_data = pd.read_csv(loc_dict['test_' + data_choice + '_data_loc'])
        te_data = te_data.fillna(0)
        processed_data_tr = prepare_cancer_dataset(tr_data, cancer=cancer)
        processed_data_te = prepare_cancer_dataset(te_data, cancer=cancer)
        model.add_dataset(data_choice, processed_data_tr, processed_data_te)
        if comb_type == 'type2' and len(data_choice_list) > 1:
            processed_data_tr = processed_data_tr.drop(columns='class')
            final_trdata = pd.merge(final_trdata, processed_data_tr, how='inner', on=['pair_name'])
            processed_data_te = processed_data_te.drop(columns='class')
            final_tedata = pd.merge(final_tedata, processed_data_te, how='inner', on=['pair_name'])
    if comb_type=='type2' and len(data_choice_list) > 1:
        model.add_dataset('&'.join(data_choice_list), final_trdata,final_tedata)

    best_params = {}
    param_list = ['boosting_type', 'num_leaves', 'max_depth', 'learning_rate', 'n_estimators',
                  'subsample_for_bin', 'objective', 'class_weight', 'min_split_gain',
                  'min_child_weight', 'min_child_samples', 'subsample', 'subsample_freq',
                  'colsample_bytree', 'reg_alpha', 'reg_lambda', 'random_state', 'n_jobs',
                  'verbose', 'thold', 'process', 'is_ready_data',
                  'use_single', 'use_comb', 'grid_searched', 'sep_train_test', 'balance',
                  'balance_strategy', 'return_model', 'contr']

    results = calc_dataset_importance(model, samples_tr, samples_te, n_split=n_split, typex='cancer_feature2')
    all_params = model.get_params(deep=False)
    selected_params = {}
    for key in param_list:
        selected_params[key] = all_params.pop(key)
    model_dict = {'params': selected_params}
    if best_params:
        model_dict['best_params'] = best_params
    model_dict.update(results)
    res_dict = {'datasets': data_choices, 'cancer': cancer}
    try:
        results[0].pop('time_all')
        all_res_cols = list(results[0].keys())
    except:
        results['0'].pop('time_all')
        all_res_cols = list(results['0'].keys())
    for d_n in all_res_cols:
        try:
            res_dict[d_n + '_m']= np.mean([results[i][d_n] for i in range(n_split)])
            res_dict[d_n + '_std']= np.std([results[i][d_n] for i in range(n_split)])
        except:
            res_dict[d_n + '_m']= np.mean([results[str(i)][d_n] for i in range(n_split)])
            res_dict[d_n + '_std']= np.std([results[str(i)][d_n] for i in range(n_split)])
    #try:
    #    res_dict['&'.join(data_choice_list) + '_m'] = np.mean([results[i]['&'.join(data_choice_list)] for i in range(n_split)])
    #    res_dict['&'.join(data_choice_list) + '_std'] = np.std([results[i]['&'.join(data_choice_list)] for i in range(n_split)])
    #except:
    #    res_dict['&'.join(data_choice_list) + '_m'] = np.mean([results[str(i)]['&'.join(data_choice_list)] for i in range(n_split)])
    #    res_dict['&'.join(data_choice_list) + '_std'] = np.std([results[str(i)]['&'.join(data_choice_list)] for i in range(n_split)])
    result_df = pd.DataFrame(columns=all_res_cols)

    if os.path.isfile(res_loc):
        result_df = pd.read_csv(res_loc)
    result_df = result_df.append(res_dict, ignore_index=True)

    result_df.to_csv(res_loc, index=False)

    # fold_predictions = elrrf.fit_predict_all_datasets_fold(train_test_sample_dict=train_test_sample_names[cancer])
    # elrrf.evaluate_folds(fold_predictions)

    print(f'Experiment with {data_choice_list} finished for {cancer}...')
    print()


def single_cancer_GCATSL_test_experiment(cancer, grid_search, thold=None):
    data_choice_list=['PPI', 'CC', 'BP']#, 'GO1', 'GO2']
    print(f'Experiment GCATSL: for {cancer}, grid_search={grid_search}')
    if 'trus' in task:
        balance_strategy = 'undersample_train'
    if 'trteus' in task:
        balance_strategy = 'undersample_train_test'
    feature_dim=128
    n_feature=len(data_choice_list)
    n_split=5
    process=True
    if thold == None:
        thold = 0.5

    loc_dict = get_locs()
    res_loc = config.ROOT_DIR / 'results' / 'GCATSL' / 'single_cancer_test.csv'
    config.ensure_dir(res_loc)
    grid_search_settings_loc = config.ROOT_DIR / 'results' / 'GCATSL' / 'result.json'
    data_choices = '|'.join(data_choice_list)
    if os.path.isfile(res_loc):
        result_df = pd.read_csv(res_loc)
    else:
        result_df = pd.DataFrame(
            columns=['datasets', 'cancer', 'AUROC_m', 'AUROC_std', 'AUPRC_m', 'AUPRC_std',
                     'MC_m', 'MC_std', 'grid_search', 'n_split',
                     'balance_strat', 'feature_dim', 'n_feature'])
    chosen_cols = result_df[
        ['datasets', 'cancer', 'grid_search', 'n_split', 'balance_strat', 'feature_dim', 'n_feature']]
    chosen_cols = chosen_cols.astype(str)
    chosen_vals = np.array([data_choices, cancer, grid_search, n_split, balance_strategy, feature_dim, n_feature])
    chosen_vals = chosen_vals.astype(str)
    if False:#(chosen_cols == chosen_vals).all(1).any():
        print(f'{data_choices} is already calculated!')
        return 0

    model = GCATSL(random_state=124, n_jobs=- 1,
                 verbose=False, thold=0.5, process=True, is_ready_data=False,
                 use_single=True, use_comb=True, grid_searched=False, sep_train_test=False, balance=True,
                 balance_strategy=balance_strategy, return_model=False, pretrained_model=None, cancer=cancer,
                   feature_dim=feature_dim, n_feature= n_feature, dataset_name = data_choices)

    model_loc = data_choices + '_' + cancer + '_' + str(grid_search) + '_' + balance_strategy + '_' + str(feature_dim) + '.pickle'
    model_loc = config.ROOT_DIR / 'results' / 'GCATSL' / 'models_test' / model_loc
    config.ensure_dir(model_loc)
    if False:#os.path.isfile(model_loc):
        results = dfnc.load_pickle(model_loc)
    else:
        tr_samples_loc = 'labels/train_pairs.csv'
        tr_samples_loc = config.DATA_DIR / tr_samples_loc
        tr_samples = pd.read_csv(tr_samples_loc)
        te_samples_loc = 'labels/test_pairs.csv'
        te_samples_loc = config.DATA_DIR / te_samples_loc
        te_samples = pd.read_csv(te_samples_loc)
        samples_tr = prepare_cancer_dataset(tr_samples, cancer=cancer)
        samples_te = prepare_cancer_dataset(te_samples, cancer=cancer)
        #for data_choice in data_choice_list:
        #    data = pd.read_csv(loc_dict['GCATSL_' + data_choice + '_data_loc'])
        #    data = data.fillna(0)
        #    model.add_dataset(data_choice, data)


        model.args_dict['input_dir'] = GCATSL_root+'data/SL_'+cancer+'/'
        model.args_dict['output_dir'] = GCATSL_root+'output/SL_'+cancer+'/'
        model.args_dict['log_dir'] = GCATSL_root+'logs/SL_'+cancer+'/'
        config.ensure_dir(model.args_dict['input_dir']+'/deneme.csv')
        config.ensure_dir(model.args_dict['output_dir']+'/deneme.csv')
        config.ensure_dir(model.args_dict['log_dir']+'/deneme.csv')
        model_dict = {'GCATSL_params': model.args_dict}
        results = fit_predict_2set(model, samples_tr, samples_te, n_split=n_split)
        model_dict.update(results)
        #dfnc.save_pickle(model_loc, model_dict)

    AUROC, AUPRC, MC = model.evaluate_folds(results)

    res_dict = {'datasets': data_choices, 'cancer': cancer,
                'AUROC_m': np.around(AUROC.mean(), 2), 'AUROC_std': np.around(np.std(AUROC), 2),
                'AUPRC_m': np.around(AUPRC.mean(), 2), 'AUPRC_std': np.around(np.std(AUPRC), 2),
                'MC_m': np.around(MC.mean(), 2), 'MC_std': np.around(np.std(MC), 2),
                'grid_search': grid_search, 'n_split': str(int(n_split)),
                'balance_strat': balance_strategy, 'feature_dim':feature_dim, 'n_feature': n_feature}

    if os.path.isfile(res_loc):
        result_df = pd.read_csv(res_loc)
    result_df = result_df.append(res_dict, ignore_index=True)

    #result_df.to_csv(res_loc, index=False)

    # fold_predictions = elrrf.fit_predict_all_datasets_fold(train_test_sample_dict=train_test_sample_names[cancer])
    # elrrf.evaluate_folds(fold_predictions)

    print(f'Experiment with {data_choice_list} finished for {cancer}...')
    print()


def single_cancer_GCATSL_dho_test_experiment(cancer, grid_search, thold=None):
    data_choice_list=['PPI', 'CC', 'BP']#, 'GO1', 'GO2']
    print(f'Experiment GCATSL: for {cancer}, grid_search={grid_search}')
    if 'trus' in task:
        balance_strategy = 'undersample_train'
    if 'trteus' in task:
        balance_strategy = 'undersample_train_test'
    feature_dim=128
    n_feature=len(data_choice_list)
    n_split=5
    process=True
    if thold == None:
        thold = 0.5

    loc_dict = get_locs()
    res_loc = config.ROOT_DIR / 'results' / 'GCATSL' / 'single_cancer_dho2_test.csv'
    config.ensure_dir(res_loc)
    grid_search_settings_loc = config.ROOT_DIR / 'results' / 'GCATSL_dho2_test' / 'result.json'
    data_choices = '|'.join(data_choice_list)
    if os.path.isfile(res_loc):
        result_df = pd.read_csv(res_loc)
    else:
        result_df = pd.DataFrame(
            columns=['datasets', 'cancer', 'AUROC_m', 'AUROC_std', 'AUPRC_m', 'AUPRC_std',
                     'MC_m', 'MC_std', 'grid_search', 'n_split',
                     'balance_strat', 'feature_dim', 'n_feature'])
    chosen_cols = result_df[
        ['datasets', 'cancer', 'grid_search', 'n_split', 'balance_strat', 'feature_dim', 'n_feature']]
    chosen_cols = chosen_cols.astype(str)
    chosen_vals = np.array([data_choices, cancer, grid_search, n_split, balance_strategy, feature_dim, n_feature])
    chosen_vals = chosen_vals.astype(str)
    if False:#(chosen_cols == chosen_vals).all(1).any():
        print(f'{data_choices} is already calculated!')
        return 0

    model = GCATSL(random_state=124, n_jobs=- 1,
                 verbose=False, thold=0.5, process=True, is_ready_data=False,
                 use_single=True, use_comb=True, grid_searched=False, sep_train_test=False, balance=True,
                 balance_strategy=balance_strategy, return_model=False, pretrained_model=None, cancer=cancer,
                   feature_dim=feature_dim, n_feature= n_feature, dataset_name = data_choices)

    model_loc = data_choices + '_' + cancer + '_' + str(grid_search) + '_' + balance_strategy + '_' + str(feature_dim) + '.pickle'
    model_loc = config.ROOT_DIR / 'results' / 'GCATSL' / 'models_dho2_test' / model_loc
    config.ensure_dir(model_loc)
    if False:#os.path.isfile(model_loc):
        results = dfnc.load_pickle(model_loc)
    else:
        tr_samples_loc = 'labels/train_pairs.csv'
        tr_samples_loc = config.DATA_DIR / tr_samples_loc
        tr_samples = pd.read_csv(tr_samples_loc)
        te_samples_loc = 'labels/test_pairs.csv'
        te_samples_loc = config.DATA_DIR / te_samples_loc
        te_samples = pd.read_csv(te_samples_loc)
        samples_tr = prepare_cancer_dataset(tr_samples, cancer=cancer)
        samples_te = prepare_cancer_dataset(te_samples, cancer=cancer)
        samples = pd.concat([samples_tr, samples_te])
        #for data_choice in data_choice_list:
        #    data = pd.read_csv(loc_dict['GCATSL_' + data_choice + '_data_loc'])
        #    data = data.fillna(0)
        #    model.add_dataset(data_choice, data)


        model.args_dict['input_dir'] = GCATSL_root+'data/SL_dho_'+cancer+'/'
        model.args_dict['output_dir'] = GCATSL_root+'output/SL_dho_'+cancer+'/'
        model.args_dict['log_dir'] = GCATSL_root+'logs/SL_dho_'+cancer+'/'
        config.ensure_dir(model.args_dict['input_dir']+'/deneme.csv')
        config.ensure_dir(model.args_dict['output_dir']+'/deneme.csv')
        config.ensure_dir(model.args_dict['log_dir']+'/deneme.csv')
        model_dict = {'GCATSL_params': model.args_dict}
        results = fit_predict_dho_cross_validation(model, samples, n_split=n_split)
        model_dict.update(results)
        #dfnc.save_pickle(model_loc, model_dict)

    AUROC, AUPRC, MC = model.evaluate_folds(results)

    res_dict = {'datasets': data_choices, 'cancer': cancer,
                'AUROC_m': np.around(AUROC.mean(), 2), 'AUROC_std': np.around(np.std(AUROC), 2),
                'AUPRC_m': np.around(AUPRC.mean(), 2), 'AUPRC_std': np.around(np.std(AUPRC), 2),
                'MC_m': np.around(MC.mean(), 2), 'MC_std': np.around(np.std(MC), 2),
                'grid_search': grid_search, 'n_split': str(int(n_split)),
                'balance_strat': balance_strategy, 'feature_dim':feature_dim, 'n_feature': n_feature}

    if os.path.isfile(res_loc):
        result_df = pd.read_csv(res_loc)
    result_df = result_df.append(res_dict, ignore_index=True)

    #result_df.to_csv(res_loc, index=False)

    # fold_predictions = elrrf.fit_predict_all_datasets_fold(train_test_sample_dict=train_test_sample_names[cancer])
    # elrrf.evaluate_folds(fold_predictions)

    print(f'Experiment with {data_choice_list} finished for {cancer}...')
    print()


def single_cancer_validation_experiment(data_choice_list, use_single, cancer, grid_search, thold=None):
    print(f'Experiment: {data_choice_list} for {cancer} with use_single:{use_single}, grid_search={grid_search}')
    start_time=None
    data_choices = '|'.join(data_choice_list)
    model_type='ELRRF'
    comb_type='type2'
    boosting='rf'
    if 'GCATSL' in task:
        model_type='GCATSL'
        data_choice_list=['PPI', 'CC', 'BP']#, 'GO1', 'GO2']
        feature_dim = 128
        n_feature = len(data_choice_list)
    else:
        use_single=True
        if 'GBDT' in task:
            model_type='ELGBDT'
            boosting='gbdt'
    print(f'Experiment {model_type}: for {cancer}, grid_search={grid_search}')
    if 'trus' in task:
        balance_strategy = 'undersample_train'
    if 'trteus' in task:
        balance_strategy = 'undersample_train_test'
    if 'early' in task:
        integration_type = 'early'
    elif 'late' in task:
        integration_type = 'late'
    else:
        integration_type = 'both'
    n_split: int = 5
    fold_type = 'stratified_shuffled'
    comb_type = 'type2'
    process=True
    if thold == None:
        thold = 0.5

    loc_dict = get_locs()
    if integration_type=='both':
        res_loc = config.ROOT_DIR / 'results' / model_type / ('single_cancer_val.csv')
    else:
        res_loc = config.ROOT_DIR / 'results' / model_type / ('single_cancer_val_'+integration_type+'.csv')
    grid_search_settings_loc = config.ROOT_DIR / 'results' / model_type / 'result.json'
    if os.path.isfile(res_loc):
        result_df = pd.read_csv(res_loc)
    else:
        result_df = pd.DataFrame(
            columns=['datasets', 'cancer', 'AUROC_m', 'AUROC_std', 'AUPRC_m', 'AUPRC_std',
                     'MC_m', 'MC_std', 'grid_search', 'use_single', 'fold_type', 'n_split',
                     'balance_strat', 'threshold', 'comb_type', 'process'])
    chosen_cols = result_df[
        ['datasets', 'cancer', 'grid_search', 'use_single', 'fold_type', 'n_split', 'balance_strat', 'threshold',
         'comb_type', 'process']]
    chosen_cols = chosen_cols.astype(str)
    chosen_vals = np.array([data_choices, cancer, grid_search, use_single, fold_type, n_split, balance_strategy, thold,
                            comb_type, process])
    chosen_vals = chosen_vals.astype(str)
    if (chosen_cols == chosen_vals).all(1).any():
        print(f'{data_choices} is already calculated!')
        return 0

    if comb_type=='type1':
        model = ELRRF(boosting_type=boosting, use_single=use_single, grid_searched=grid_search, balance_strategy=balance_strategy, thold=thold,
                      process=process, cancer=cancer)
    elif comb_type=='type2':
        use_comb = False
        use_all_comb = True
        model = ELRRF(boosting_type=boosting, use_single=use_single, grid_searched=grid_search, balance_strategy=balance_strategy,
                      use_comb=use_comb, thold=thold, process=process, cancer=cancer)

    model_loc = data_choices + '_' + cancer + '_' + str(grid_search) + '_' + str(
        use_single) + '_' + comb_type + '_' + fold_type + '_' + str(n_split) + \
                '_' + balance_strategy + '.pickle'
    if integration_type=='both':
        model_loc = config.ROOT_DIR / 'results' / model_type / 'models_val' / model_loc
    else:
        model_loc = config.ROOT_DIR / 'results' / model_type / ('models_val_'+integration_type) / model_loc
    config.ensure_dir(model_loc)
    if os.path.isfile(model_loc):
        model_dict = dfnc.load_pickle(model_loc)
    else:
        samples_loc = 'labels/train_pairs.csv'
        samples_loc = config.DATA_DIR / samples_loc
        samples = pd.read_csv(samples_loc)
        if cancer=='all':
            samples = prepare_cancer_dataset(samples, cancer=None)
        else:
            samples = prepare_cancer_dataset(samples, cancer=cancer)
        if comb_type == 'type2' and len(data_choice_list) > 1:
            final_data = samples.copy()
        for data_choice in data_choice_list:
            data = pd.read_csv(loc_dict['train_' + data_choice + '_data_loc'])
            data = data.fillna(0)
            if cancer=='all':
                processed_data = prepare_cancer_dataset(data, cancer=None)
            else:
                processed_data = prepare_cancer_dataset(data, cancer=cancer)
            if integration_type=='late' or integration_type=='both':
                model.add_dataset(data_choice, processed_data)
            if comb_type=='type2' and len(data_choice_list) > 1:
                processed_data = processed_data.drop(columns='class')
                final_data = pd.merge(final_data, processed_data, how='inner', on=['pair_name'])
        if comb_type=='type2' and len(data_choice_list) > 1 and (integration_type=='early' or integration_type=='both'):
            model.add_dataset('&'.join(data_choice_list), final_data)

        logging.info(f'single_cancer started for {cancer} with th={thold}')
        start_time = time.time()
        random_grid = model.get_tree_search_grid()
        best_params = {}
        if grid_search:
            search_space = list()
            for par_name, par_val in random_grid.items():
                search_space.append(Categorical(par_val, name=par_name))

            @use_named_args(search_space)
            def evaluate_model(**params):
                model.set_params(**params)
                result_one = fit_predict_cross_validation(samples, fold_type=fold_type, n_split=n_split)
                AUROC, AUPRC, MC = model.evaluate_folds(result_one, report=False)
                return 1 - np.mean(AUPRC)

            result_best = gp_minimize(evaluate_model, search_space, n_calls=20)
            for idx in range(len(search_space)):
                best_params[search_space[idx]._name] = result_best.x[idx]

        if best_params:
            model.set_params(**best_params)
        param_list = ['boosting_type', 'num_leaves', 'max_depth', 'learning_rate', 'n_estimators',
                      'subsample_for_bin', 'objective', 'class_weight', 'min_split_gain',
                      'min_child_weight', 'min_child_samples', 'subsample', 'subsample_freq',
                      'colsample_bytree', 'reg_alpha', 'reg_lambda', 'random_state', 'n_jobs',
                      'verbose', 'thold', 'process', 'is_ready_data',
                      'use_single', 'use_comb', 'grid_searched', 'sep_train_test', 'balance',
                      'balance_strategy', 'return_model']
        all_params = model.get_params(deep=False)
        selected_params = {}
        for key in param_list:
            selected_params[key] = all_params.pop(key)
        model_dict = {'params': selected_params}
        if best_params:
            model_dict['best_params']=best_params
        results = fit_predict_cross_validation(model, samples, fold_type=fold_type, n_split=n_split)
        model_dict.update(results)

        dfnc.save_pickle(model_loc, model_dict)

    eval_start_time = time.time()
    AUROC, AUPRC, MC = model.evaluate_folds(model_dict)
    eval_time = str(time.time() - eval_start_time)
    if start_time is not None:
        full_time=str(time.time() - start_time)
        model_dict['full_time']=full_time
        logging.info(f'single_cancer for {cancer} with th={thold}, model={integration_type}, cv={n_split}, data={data_choices} lasted {(time.time() - start_time)} seconds.')
    model_dict['eval_time']=eval_time
    dfnc.save_pickle(model_loc, model_dict)


    res_dict = {'datasets': data_choices, 'cancer': cancer,
                'AUROC_m': np.around(AUROC.mean(), 2), 'AUROC_std': np.around(np.std(AUROC), 2),
                'AUPRC_m': np.around(AUPRC.mean(), 2), 'AUPRC_std': np.around(np.std(AUPRC), 2),
                'MC_m': np.around(MC.mean(), 2), 'MC_std': np.around(np.std(MC), 2),
                'grid_search': grid_search, 'use_single': use_single,
                'fold_type': fold_type, 'n_split': n_split, 'balance_strat': balance_strategy,
                'threshold': thold, 'comb_type': comb_type, 'process': process}

    if os.path.isfile(res_loc):
        result_df = pd.read_csv(res_loc)
    result_df = result_df.append(res_dict, ignore_index=True)

    result_df.to_csv(res_loc, index=False)

    # fold_predictions = elrrf.fit_predict_all_datasets_fold(train_test_sample_dict=train_test_sample_names[cancer])
    # elrrf.evaluate_folds(fold_predictions)

    print(f'Experiment with {data_choice_list} finished for {cancer}...')
    print()


def single_cancer_ho2_validation_experiment(data_choice_list, use_single, cancer, grid_search, thold=None, ho='dho2'):
    print(f'Experiment: {data_choice_list} for {cancer} with use_single:{use_single}, grid_search={grid_search}')
    start_time=None
    if 'trus' in task:
        balance_strategy = 'undersample_train'
    if 'trteus' in task:
        balance_strategy = 'undersample_train_test'
    if 'early' in task:
        integration_type = 'early'
    elif 'late' in task:
        integration_type = 'late'
    else:
        integration_type = 'both'
    n_split: int = 5
    fold_type = 'stratified_shuffled'
    comb_type = 'type2'
    process=True
    if thold == None:
        thold = 0.5

    loc_dict = get_locs()
    if integration_type=='both':
        res_loc = config.ROOT_DIR / 'results' / 'elrrf' / ('single_cancer_'+ho+'_gbdt_validation.csv')
    else:
        res_loc = config.ROOT_DIR / 'results' / 'elrrf' / ('single_cancer_'+ho+'_validation_'+integration_type+'.csv')
    grid_search_settings_loc = config.ROOT_DIR / 'results' / 'elrrf' / 'result.json'
    data_choices = '|'.join(data_choice_list)
    if os.path.isfile(res_loc):
        result_df = pd.read_csv(res_loc)
    else:
        result_df = pd.DataFrame(
            columns=['datasets', 'cancer', 'AUROC_m', 'AUROC_std', 'AUPRC_m', 'AUPRC_std',
                     'MC_m', 'MC_std', 'grid_search', 'use_single', 'fold_type', 'n_split',
                     'balance_strat', 'threshold', 'comb_type', 'process'])
    chosen_cols = result_df[
        ['datasets', 'cancer', 'grid_search', 'use_single', 'fold_type', 'n_split', 'balance_strat', 'threshold',
         'comb_type', 'process']]
    chosen_cols = chosen_cols.astype(str)
    chosen_vals = np.array([data_choices, cancer, grid_search, use_single, fold_type, n_split, balance_strategy, thold,
                            comb_type, process])
    chosen_vals = chosen_vals.astype(str)
    if (chosen_cols == chosen_vals).all(1).any():
        print(f'{data_choices} is already calculated!')
        return 0

    if comb_type=='type1':
        elrrf = ELRRF(use_single=use_single, grid_searched=grid_search, balance_strategy=balance_strategy, thold=thold,
                      process=process)
    elif comb_type=='type2':
        use_comb = False
        use_all_comb = True
        elrrf = ELRRF(use_single=use_single, grid_searched=grid_search, balance_strategy=balance_strategy,
                      use_comb=use_comb, thold=thold, process=process)

    model_loc = data_choices + '_' + cancer + '_' + str(grid_search) + '_' + str(
        use_single) + '_' + comb_type + '_' + fold_type + '_' + str(n_split) + \
                '_' + balance_strategy + '.pickle'
    if integration_type=='both':
        model_loc = config.ROOT_DIR / 'results' / 'elrrf' / ('models_'+ho+'_gbdt') / model_loc
    else:
        model_loc = config.ROOT_DIR / 'results' / 'elrrf' / ('models_'+ho+'_'+integration_type) / model_loc
    config.ensure_dir(model_loc)
    if os.path.isfile(model_loc):
        model_dict = dfnc.load_pickle(model_loc)
    else:
        samples_loc = 'labels/train_pairs.csv'
        samples_loc = config.DATA_DIR / samples_loc
        samples = pd.read_csv(samples_loc)
        samples = prepare_cancer_dataset(samples, cancer=cancer)
        if comb_type == 'type2' and len(data_choice_list) > 1:
            final_data = samples.copy()
        for data_choice in data_choice_list:
            data = pd.read_csv(loc_dict['train_' + data_choice + '_data_loc'])
            data = data.fillna(0)
            processed_data = prepare_cancer_dataset(data, cancer=cancer)
            if integration_type=='late' or integration_type=='both':
                elrrf.add_dataset(data_choice, processed_data)
            if comb_type=='type2' and len(data_choice_list) > 1:
                processed_data = processed_data.drop(columns='class')
                final_data = pd.merge(final_data, processed_data, how='inner', on=['pair_name'])
        if comb_type=='type2' and len(data_choice_list) > 1 and (integration_type=='early' or integration_type=='both'):
            elrrf.add_dataset('&'.join(data_choice_list), final_data)

        logging.info(f'single_cancer started for {cancer} with th={thold}')
        start_time = time.time()
        random_grid = elrrf.get_tree_search_grid()
        best_params = {}
        if grid_search:
            search_space = list()
            for par_name, par_val in random_grid.items():
                search_space.append(Categorical(par_val, name=par_name))

            @use_named_args(search_space)
            def evaluate_model(**params):
                elrrf.set_params(**params)
                result_one = elrrf.fit_predict_cross_validation(samples, fold_type=fold_type, n_split=n_split)
                AUROC, AUPRC, MC = elrrf.evaluate_folds(result_one, report=False)
                return 1 - np.mean(AUPRC)

            result_best = gp_minimize(evaluate_model, search_space, n_calls=20)
            for idx in range(len(search_space)):
                best_params[search_space[idx]._name] = result_best.x[idx]

        if best_params:
            elrrf.set_params(**best_params)
        param_list = ['boosting_type', 'num_leaves', 'max_depth', 'learning_rate', 'n_estimators',
                      'subsample_for_bin', 'objective', 'class_weight', 'min_split_gain',
                      'min_child_weight', 'min_child_samples', 'subsample', 'subsample_freq',
                      'colsample_bytree', 'reg_alpha', 'reg_lambda', 'random_state', 'n_jobs',
                      'verbose', 'thold', 'process', 'is_ready_data',
                      'use_single', 'use_comb', 'grid_searched', 'sep_train_test', 'balance',
                      'balance_strategy', 'return_model']
        all_params = elrrf.get_params(deep=False)
        selected_params = {}
        for key in param_list:
            selected_params[key] = all_params.pop(key)
        model_dict = {'params': selected_params}
        if best_params:
            model_dict['best_params']=best_params
        results = elrrf.fit_predict_ho_cross_validation(samples, fold_type=fold_type, n_split=n_split, ho=ho)
        model_dict.update(results)

        #dfnc.save_pickle(model_loc, model_dict)

    eval_start_time = time.time()
    AUROC, AUPRC, MC = elrrf.evaluate_folds(model_dict)
    eval_time = str(time.time() - eval_start_time)
    if start_time is not None:
        full_time=str(time.time() - start_time)
        model_dict['full_time']=full_time
        logging.info(f'single_cancer for {cancer} with th={thold}, model={integration_type}, cv={n_split}, data={data_choices} lasted {(time.time() - start_time)} seconds.')
    model_dict['eval_time']=eval_time
    dfnc.save_pickle(model_loc, model_dict)


    res_dict = {'datasets': data_choices, 'cancer': cancer,
                'AUROC_m': np.around(AUROC.mean(), 2), 'AUROC_std': np.around(np.std(AUROC), 2),
                'AUPRC_m': np.around(AUPRC.mean(), 2), 'AUPRC_std': np.around(np.std(AUPRC), 2),
                'MC_m': np.around(MC.mean(), 2), 'MC_std': np.around(np.std(MC), 2),
                'grid_search': grid_search, 'use_single': use_single,
                'fold_type': fold_type, 'n_split': n_split, 'balance_strat': balance_strategy,
                'threshold': thold, 'comb_type': comb_type, 'process': process}

    if os.path.isfile(res_loc):
        result_df = pd.read_csv(res_loc)
    result_df = result_df.append(res_dict, ignore_index=True)

    result_df.to_csv(res_loc, index=False)

    # fold_predictions = elrrf.fit_predict_all_datasets_fold(train_test_sample_dict=train_test_sample_names[cancer])
    # elrrf.evaluate_folds(fold_predictions)

    print(f'Experiment with {data_choice_list} finished for {cancer}...')
    print()


def single_cancer_ho_validation_experiment(data_choice_list, use_single, cancer, grid_search, thold=None,ho='dho'):
    print(f'Experiment: {data_choice_list} for {cancer} with use_single:{use_single}, grid_search={grid_search}')
    start_time=None
    if 'trus' in task:
        balance_strategy = 'undersample_train'
    if 'trteus' in task:
        balance_strategy = 'undersample_train_test'
    if 'early' in task:
        integration_type = 'early'
    elif 'late' in task:
        integration_type = 'late'
    else:
        integration_type = 'both'
    n_split: int = 5
    fold_type = 'stratified_shuffled'
    comb_type = 'type2'
    process=True
    if thold == None:
        thold = 0.5

    loc_dict = get_locs()
    if integration_type=='both':
        res_loc = config.ROOT_DIR / 'results' / 'elrrf' / ('single_cancer_'+ho+'_validation.csv')
    else:
        res_loc = config.ROOT_DIR / 'results' / 'elrrf' / ('single_cancer_'+ho+'_validation_'+integration_type+'.csv')
    grid_search_settings_loc = config.ROOT_DIR / 'results' / 'elrrf' / 'result.json'

    config.ensure_dir(res_loc)
    data_choices = '|'.join(data_choice_list)
    if os.path.isfile(res_loc):
        result_df = pd.read_csv(res_loc)
    else:
        result_df = pd.DataFrame(
            columns=['datasets', 'cancer', 'AUROC_m', 'AUROC_std', 'AUPRC_m', 'AUPRC_std',
                     'MC_m', 'MC_std', 'grid_search', 'use_single', 'fold_type', 'n_split',
                     'balance_strat', 'threshold', 'comb_type', 'process'])
    chosen_cols = result_df[
        ['datasets', 'cancer', 'grid_search', 'use_single', 'fold_type', 'n_split', 'balance_strat', 'threshold',
         'comb_type', 'process']]
    chosen_cols = chosen_cols.astype(str)
    chosen_vals = np.array([data_choices, cancer, grid_search, use_single, fold_type, n_split, balance_strategy, thold,
                            comb_type, process])
    chosen_vals = chosen_vals.astype(str)
    if (chosen_cols == chosen_vals).all(1).any():
        print(f'{data_choices} is already calculated!')
        return 0

    if comb_type=='type1':
        elrrf = ELRRF(use_single=use_single, grid_searched=grid_search, balance_strategy=balance_strategy, thold=thold,
                      process=process)
    elif comb_type=='type2':
        use_comb = False
        use_all_comb = True
        elrrf = ELRRF(use_single=use_single, grid_searched=grid_search, balance_strategy=balance_strategy,
                      use_comb=use_comb, thold=thold, process=process)

    model_loc = data_choices + '_' + cancer + '_' + str(grid_search) + '_' + str(
        use_single) + '_' + comb_type + '_' + fold_type + '_' + str(n_split) + \
                '_' + balance_strategy + '.pickle'
    if integration_type=='both':
        model_loc = config.ROOT_DIR / 'results' / 'elrrf' / ('models_'+ho) / model_loc
    else:
        model_loc = config.ROOT_DIR / 'results' / 'elrrf' / ('models_'+ho+'_'+integration_type) / model_loc
    config.ensure_dir(model_loc)
    if os.path.isfile(model_loc):
        results = dfnc.load_pickle(model_loc)
    else:
        samples_loc = 'labels/train_pairs.csv'
        samples_loc = config.DATA_DIR / samples_loc
        samples = pd.read_csv(samples_loc)
        samples = prepare_cancer_dataset(samples, cancer=cancer)
        genes = samples['pair_name'].str.split('|', expand=True)
        genes['SL'] = samples['class']

        if 'dho' in ho:
            all_genes = np.union1d(genes[0].values, genes[1].values)
            adj_genes = pd.DataFrame(index=all_genes, columns=all_genes)
            for ind, row in genes.iterrows():
                adj_genes.loc[row[0], row[1]] = row['SL']
                adj_genes.loc[row[1], row[0]] = row['SL']
            train_size, test_size = 1.0, 1.0
            train_genes, test_genes = [], []
            seed_id = 0
            while (len(all_genes) > 0):
                if (train_size / test_size <= 4) & (len(all_genes) > 0):
                    np.random.seed(seed_id)
                    chosen_gene = np.random.choice(all_genes, 1)
                    all_genes = np.delete(all_genes, np.where(all_genes == chosen_gene))
                    train_genes.append(chosen_gene[0])
                    train_size = adj_genes.loc[train_genes, train_genes].notna().sum().sum() / 2
                if (train_size / test_size > 4) & (len(all_genes) > 0):
                    np.random.seed(seed_id)
                    chosen_gene = np.random.choice(all_genes, 1)
                    all_genes = np.delete(all_genes, np.where(all_genes == chosen_gene))
                    test_genes.append(chosen_gene[0])
                    test_size = adj_genes.loc[test_genes, test_genes].notna().sum().sum() / 2
                seed_id += 1

            print('double')
        samples_tr = samples[(genes[0].isin(train_genes))&(genes[1].isin(train_genes))]
        samples_te = samples[(genes[0].isin(test_genes))&(genes[1].isin(test_genes))]
        if comb_type == 'type2' and len(data_choice_list) > 1:
            final_trdata = samples_tr.copy()
            final_tedata = samples_te.copy()
        for data_choice in data_choice_list:
            data = pd.read_csv(loc_dict['train_' + data_choice + '_data_loc'])
            data = data.fillna(0)
            processed_data = prepare_cancer_dataset(data, cancer=cancer)
            processed_data_tr = processed_data[(genes[0].isin(train_genes)) & (genes[1].isin(train_genes))]
            processed_data_te = processed_data[(genes[0].isin(test_genes)) & (genes[1].isin(test_genes))]
            if integration_type=='late' or integration_type=='both':
                elrrf.add_dataset(data_choice, processed_data_tr, processed_data_te)
            if comb_type=='type2' and len(data_choice_list) > 1:
                processed_data_tr = processed_data_tr.drop(columns='class')
                final_trdata = pd.merge(final_trdata, processed_data_tr, how='inner', on=['pair_name'])
                processed_data_te = processed_data_te.drop(columns='class')
                final_tedata = pd.merge(final_tedata, processed_data_te, how='inner', on=['pair_name'])
        if comb_type=='type2' and len(data_choice_list) > 1 and (integration_type=='early' or integration_type=='both'):
            elrrf.add_dataset('&'.join(data_choice_list), final_trdata, final_tedata)

        logging.info(f'single_cancer started for {cancer} with th={thold}')
        start_time = time.time()
        random_grid = elrrf.get_tree_search_grid()
        best_params = {}
        if grid_search:
            search_space = list()
            for par_name, par_val in random_grid.items():
                search_space.append(Categorical(par_val, name=par_name))

            @use_named_args(search_space)
            def evaluate_model(**params):
                elrrf.set_params(**params)
                result_one = elrrf.fit_predict_cross_validation(samples, fold_type=fold_type, n_split=n_split)
                AUROC, AUPRC, MC = elrrf.evaluate_folds(result_one, report=False)
                return 1 - np.mean(AUPRC)

            result_best = gp_minimize(evaluate_model, search_space, n_calls=20)
            for idx in range(len(search_space)):
                best_params[search_space[idx]._name] = result_best.x[idx]

        if best_params:
            elrrf.set_params(**best_params)
        param_list = ['boosting_type', 'num_leaves', 'max_depth', 'learning_rate', 'n_estimators',
                      'subsample_for_bin', 'objective', 'class_weight', 'min_split_gain',
                      'min_child_weight', 'min_child_samples', 'subsample', 'subsample_freq',
                      'colsample_bytree', 'reg_alpha', 'reg_lambda', 'random_state', 'n_jobs',
                      'verbose', 'thold', 'process', 'is_ready_data',
                      'use_single', 'use_comb', 'grid_searched', 'sep_train_test', 'balance',
                      'balance_strategy', 'return_model']
        all_params = elrrf.get_params(deep=False)
        selected_params = {}
        for key in param_list:
            selected_params[key] = all_params.pop(key)
        model_dict = {'params': selected_params}
        if best_params:
            model_dict['best_params']=best_params
        results = elrrf.fit_predict_2set(samples_tr, samples_te, n_split=n_split)
        model_dict.update(results)

        #dfnc.save_pickle(model_loc, model_dict)

    eval_start_time = time.time()
    AUROC, AUPRC, MC = elrrf.evaluate_folds(results)
    eval_time = str(time.time() - eval_start_time)
    if start_time is not None:
        full_time=str(time.time() - start_time)
        model_dict['full_time']=full_time
        logging.info(f'single_cancer for {cancer} with th={thold}, model={integration_type}, cv={n_split}, data={data_choices} lasted {(time.time() - start_time)} seconds.')
    model_dict['eval_time']=eval_time
    dfnc.save_pickle(model_loc, model_dict)


    res_dict = {'datasets': data_choices, 'cancer': cancer,
                'AUROC_m': np.around(AUROC.mean(), 2), 'AUROC_std': np.around(np.std(AUROC), 2),
                'AUPRC_m': np.around(AUPRC.mean(), 2), 'AUPRC_std': np.around(np.std(AUPRC), 2),
                'MC_m': np.around(MC.mean(), 2), 'MC_std': np.around(np.std(MC), 2),
                'grid_search': grid_search, 'use_single': use_single,
                'fold_type': fold_type, 'n_split': n_split, 'balance_strat': balance_strategy,
                'threshold': thold, 'comb_type': comb_type, 'process': process}

    if os.path.isfile(res_loc):
        result_df = pd.read_csv(res_loc)
    result_df = result_df.append(res_dict, ignore_index=True)

    result_df.to_csv(res_loc, index=False)

    # fold_predictions = elrrf.fit_predict_all_datasets_fold(train_test_sample_dict=train_test_sample_names[cancer])
    # elrrf.evaluate_folds(fold_predictions)

    print(f'Experiment with {data_choice_list} finished for {cancer}...')
    print()


def single_cancer_test_experimentxx(data_choice_list, use_single, cancer, grid_search, thold=None):
    print(f'Experiment: {data_choice_list} for {cancer}, grid_search={grid_search}')
    if 'trus' in task:
        balance_strategy = 'undersample_train'
    if 'trteus' in task:
        balance_strategy = 'undersample_train_test'
    comb_type = 'type2'
    n_split=5
    process=True
    if thold == None:
        thold = 0.5

    loc_dict = get_locs()

    res_loc = config.ROOT_DIR / 'results' / 'elrrf' / 'single_cancer_test.csv'
    grid_search_settings_loc = config.ROOT_DIR / 'results' / 'elrrf' / 'result.json'
    data_choices = '|'.join(data_choice_list)
    if os.path.isfile(res_loc):
        result_df = pd.read_csv(res_loc)
    else:
        result_df = pd.DataFrame(
            columns=['datasets', 'cancer', 'AUROC_m', 'AUROC_std', 'AUPRC_m', 'AUPRC_std',
                     'MC_m', 'MC_std', 'grid_search', 'use_single', 'n_split',
                     'balance_strat', 'threshold', 'comb_type', 'process'])
    chosen_cols = result_df[
        ['datasets', 'cancer', 'grid_search', 'use_single', 'n_split', 'balance_strat', 'threshold',
         'comb_type', 'process']]
    chosen_cols = chosen_cols.astype(str)
    chosen_vals = np.array([data_choices, cancer, grid_search, use_single, n_split, balance_strategy, thold,
                            comb_type, process])
    chosen_vals = chosen_vals.astype(str)
    if (chosen_cols == chosen_vals).all(1).any():
        print(f'{data_choices} is already calculated!')
        return 0

    if comb_type=='type1':
        elrrf = ELRRF(use_single=use_single, grid_searched=grid_search, balance_strategy=balance_strategy, thold=thold,
                      process=process)
    elif comb_type=='type2':
        use_comb = False
        use_all_comb = True
        elrrf = ELRRF(use_single=use_single, grid_searched=grid_search, balance_strategy=balance_strategy,
                      use_comb=use_comb, thold=thold, process=process)

    model_loc = data_choices + '_' + cancer + '_' + str(grid_search) + '_' + str(
        use_single) + '_' + comb_type + \
                '_' + balance_strategy + '.pickle'
    model_loc = config.ROOT_DIR / 'results' / 'elrrf' / 'models_test' / model_loc
    if os.path.isfile(model_loc):
        results = dfnc.load_pickle(model_loc)
    else:
        tr_samples_loc = 'labels/train_pairs.csv'
        tr_samples_loc = config.DATA_DIR / tr_samples_loc
        tr_samples = pd.read_csv(tr_samples_loc)
        te_samples_loc = 'labels/test_pairs.csv'
        te_samples_loc = config.DATA_DIR / te_samples_loc
        te_samples = pd.read_csv(te_samples_loc)
        samples_tr = prepare_cancer_dataset(tr_samples, cancer=cancer)
        samples_te = prepare_cancer_dataset(te_samples, cancer=cancer)
        if comb_type == 'type2' and len(data_choice_list) > 1:
            final_trdata = samples_tr.copy()
            final_tedata = samples_te.copy()
        for data_choice in data_choice_list:
            tr_data = pd.read_csv(loc_dict['train_' + data_choice + '_data_loc'])
            tr_data = tr_data.fillna(0)
            te_data = pd.read_csv(loc_dict['test_' + data_choice + '_data_loc'])
            te_data = te_data.fillna(0)
            processed_data_tr = prepare_cancer_dataset(tr_data, cancer=cancer)
            processed_data_te = prepare_cancer_dataset(te_data, cancer=cancer)
            elrrf.add_dataset(data_choice, processed_data_tr,processed_data_te)
            if comb_type=='type2' and len(data_choice_list) > 1:
                processed_data_tr = processed_data_tr.drop(columns='class')
                final_trdata = pd.merge(final_trdata, processed_data_tr, how='inner', on=['pair_name'])
                processed_data_te = processed_data_te.drop(columns='class')
                final_tedata = pd.merge(final_tedata, processed_data_te, how='inner', on=['pair_name'])
        if comb_type=='type2' and len(data_choice_list) > 1:
            elrrf.add_dataset('&'.join(data_choice_list), final_trdata,final_tedata)

        random_grid = elrrf.get_tree_search_grid()
        best_params = {}
        if grid_search:
            best_val_model_loc = data_choices + '_' + cancer + '_' + str(grid_search) + '_' + str(use_single) + \
                                 '_' + comb_type + '_' + 'stratified_shuffled' + '_' + str(n_split) + \
                                 '_' + balance_strategy + '.pickle'
            best_val_model_loc = config.ROOT_DIR / 'results' / 'elrrf' / 'models' / best_val_model_loc
            best_val_model = dfnc.load_pickle(best_val_model_loc)

            best_params = best_val_model['best_params']

        if best_params:
            elrrf.set_params(**best_params)
        param_list = ['boosting_type', 'num_leaves', 'max_depth', 'learning_rate', 'n_estimators',
                      'subsample_for_bin', 'objective', 'class_weight', 'min_split_gain',
                      'min_child_weight', 'min_child_samples', 'subsample', 'subsample_freq',
                      'colsample_bytree', 'reg_alpha', 'reg_lambda', 'random_state', 'n_jobs',
                      'verbose', 'thold', 'process', 'is_ready_data',
                      'use_single', 'use_comb', 'grid_searched', 'sep_train_test', 'balance',
                      'balance_strategy', 'return_model']
        all_params = elrrf.get_params(deep=False)
        selected_params = {}
        for key in param_list:
            selected_params[key] = all_params.pop(key)
        model_dict = {'params': selected_params}
        if best_params:
            model_dict['best_params']=best_params
        results = elrrf.fit_predict_2set(samples_tr, samples_te, n_split=n_split)
        model_dict.update(results)
        dfnc.save_pickle(model_loc, model_dict)

    AUROC, AUPRC, MC = elrrf.evaluate_folds(results)

    res_dict = {'datasets': data_choices, 'cancer': cancer,
                'AUROC_m': np.around(AUROC.mean(), 2), 'AUROC_std': np.around(np.std(AUROC), 2),
                'AUPRC_m': np.around(AUPRC.mean(), 2), 'AUPRC_std': np.around(np.std(AUPRC), 2),
                'MC_m': np.around(MC.mean(), 2), 'MC_std': np.around(np.std(MC), 2),
                'grid_search': grid_search, 'use_single': use_single, 'n_split': str(int(n_split)),
                'balance_strat': balance_strategy, 'threshold': thold, 'comb_type': comb_type, 'process': process}

    if os.path.isfile(res_loc):
        result_df = pd.read_csv(res_loc)
    result_df = result_df.append(res_dict, ignore_index=True)

    result_df.to_csv(res_loc, index=False)

    # fold_predictions = elrrf.fit_predict_all_datasets_fold(train_test_sample_dict=train_test_sample_names[cancer])
    # elrrf.evaluate_folds(fold_predictions)

    print(f'Experiment with {data_choice_list} finished for {cancer}...')
    print()


def cross_cancer_validation_experiment(data_choice_list, use_single, cancer_train, cancer_test, grid_search, thold=None):
    print(f'Experiment: {data_choice_list} from {cancer_train} to {cancer_test}, grid_search={grid_search}')
    if 'trus' in task:
        balance_strategy = 'undersample_train'
    if 'trteus' in task:
        balance_strategy = 'undersample_train_test'
    comb_type = 'type2'
    n_split=5
    process=True
    if thold == None:
        thold = 0.5

    loc_dict = get_locs()

    res_loc = config.ROOT_DIR / 'results' / 'elrrf' / 'cross_cancer_gbdt_validation.csv'
    grid_search_settings_loc = config.ROOT_DIR / 'results' / 'elrrf' / 'result.json'
    data_choices = '|'.join(data_choice_list)
    if os.path.isfile(res_loc):
        result_df = pd.read_csv(res_loc)
    else:
        result_df = pd.DataFrame(
            columns=['datasets', 'cancer_train', 'cancer_test', 'AUROC_m', 'AUROC_std', 'AUPRC_m', 'AUPRC_std',
                     'MC_m', 'MC_std', 'grid_search', 'use_single', 'n_split',
                     'balance_strat', 'threshold', 'comb_type', 'process'])
    chosen_cols = result_df[
        ['datasets', 'cancer_train', 'cancer_test', 'grid_search', 'use_single', 'n_split', 'balance_strat', 'threshold',
         'comb_type', 'process']]
    chosen_cols = chosen_cols.astype(str)
    chosen_vals = np.array([data_choices, cancer_train, cancer_test, grid_search, use_single, n_split, balance_strategy, thold,
                            comb_type, process])
    chosen_vals = chosen_vals.astype(str)
    if (chosen_cols == chosen_vals).all(1).any():
        print(f'{data_choices} is already calculated!')
        return 0

    if comb_type=='type1':
        elrrf = ELRRF(use_single=use_single, grid_searched=grid_search, balance_strategy=balance_strategy, thold=thold,
                      process=process)
    elif comb_type=='type2':
        use_comb = False
        use_all_comb = True
        elrrf = ELRRF(use_single=use_single, grid_searched=grid_search, balance_strategy=balance_strategy,
                      use_comb=use_comb, thold=thold, process=process)

    model_loc = data_choices + '_' + cancer_train + '_' + cancer_test + '_' + str(grid_search) + '_' + str(
        use_single) + '_' + comb_type + '_' + \
                '_' + balance_strategy + '.pickle'
    model_loc = config.ROOT_DIR / 'results' / 'elrrf' / 'models_gbdt_cc' / model_loc
    config.ensure_dir(model_loc)
    if os.path.isfile(model_loc):
        results = dfnc.load_pickle(model_loc)
    else:
        samples_loc = 'labels/train_pairs.csv'
        samples_loc = config.DATA_DIR / samples_loc
        samples = pd.read_csv(samples_loc)
        samples_tr = prepare_cancer_dataset(samples, cancer=cancer_train)
        samples_te = prepare_cancer_dataset(samples, cancer=cancer_test)
        if comb_type == 'type2' and len(data_choice_list) > 1:
            final_trdata = samples_tr.copy()
            final_tedata = samples_te.copy()
        for data_choice in data_choice_list:
            data = pd.read_csv(loc_dict['train_' + data_choice + '_data_loc'])
            data = data.fillna(0)
            processed_data_tr = prepare_cancer_dataset(data, cancer=cancer_train)
            processed_data_te = prepare_cancer_dataset(data, cancer=cancer_test)
            elrrf.add_dataset(data_choice, processed_data_tr,processed_data_te)
            if comb_type=='type2' and len(data_choice_list) > 1:
                processed_data_tr = processed_data_tr.drop(columns='class')
                final_trdata = pd.merge(final_trdata, processed_data_tr, how='inner', on=['pair_name'])
                processed_data_te = processed_data_te.drop(columns='class')
                final_tedata = pd.merge(final_tedata, processed_data_te, how='inner', on=['pair_name'])
        if comb_type=='type2' and len(data_choice_list) > 1:
            elrrf.add_dataset('&'.join(data_choice_list), final_trdata,final_tedata)

        random_grid = elrrf.get_tree_search_grid()
        best_params = {}
        if grid_search:
            search_space = list()
            for par_name, par_val in random_grid.items():
                search_space.append(Categorical(par_val, name=par_name))

            @use_named_args(search_space)
            def evaluate_model(**params):
                elrrf.set_params(**params)
                result_one = elrrf.fit_predict_2set(samples_tr, samples_te, n_split=n_split)
                AUROC, AUPRC, MC = elrrf.evaluate_folds(result_one, report=False)
                return 1 - np.mean(AUPRC)

            result_best = gp_minimize(evaluate_model, search_space, n_calls=10)
            for idx in range(len(search_space)):
                best_params[search_space[idx]._name] = result_best.x[idx]

        if best_params:
            elrrf.set_params(**best_params)
        param_list = ['boosting_type', 'num_leaves', 'max_depth', 'learning_rate', 'n_estimators',
                      'subsample_for_bin', 'objective', 'class_weight', 'min_split_gain',
                      'min_child_weight', 'min_child_samples', 'subsample', 'subsample_freq',
                      'colsample_bytree', 'reg_alpha', 'reg_lambda', 'random_state', 'n_jobs',
                      'verbose', 'thold', 'process', 'is_ready_data',
                      'use_single', 'use_comb', 'grid_searched', 'sep_train_test', 'balance',
                      'balance_strategy', 'return_model']
        all_params = elrrf.get_params(deep=False)
        selected_params = {}
        for key in param_list:
            selected_params[key] = all_params.pop(key)
        model_dict = {'params': selected_params}
        if best_params:
            model_dict['best_params'] = best_params
        results = elrrf.fit_predict_2set(samples_tr, samples_te, n_split=n_split)
        model_dict.update(results)
        dfnc.save_pickle(model_loc, model_dict)

    AUROC, AUPRC, MC = elrrf.evaluate_folds(results)

    res_dict = {'datasets': data_choices, 'cancer_train': cancer_train, 'cancer_test': cancer_test,
                'AUROC_m': np.around(AUROC.mean(), 2), 'AUROC_std': np.around(np.std(AUROC), 2),
                'AUPRC_m': np.around(AUPRC.mean(), 2), 'AUPRC_std': np.around(np.std(AUPRC), 2),
                'MC_m': np.around(MC.mean(), 2), 'MC_std': np.around(np.std(MC), 2),
                'grid_search': grid_search, 'use_single': use_single, 'n_split': n_split,
                'balance_strat': balance_strategy, 'threshold': thold, 'comb_type': comb_type, 'process': process}

    if os.path.isfile(res_loc):
        result_df = pd.read_csv(res_loc)
    result_df = result_df.append(res_dict, ignore_index=True)

    result_df.to_csv(res_loc, index=False)

    # fold_predictions = elrrf.fit_predict_all_datasets_fold(train_test_sample_dict=train_test_sample_names[cancer])
    # elrrf.evaluate_folds(fold_predictions)

    print(f'Experiment with {data_choice_list} finished from {cancer_train} to {cancer_test}...')
    print()


def cross_multi_cancer_validation_experiment(data_choice_list, use_single, cancer_trains, cancer_test, grid_search, thold=None):
    print(f'Experiment: {data_choice_list} from all to {cancer_test}, grid_search={grid_search}')
    if 'trus' in task:
        balance_strategy = 'undersample_train'
    if 'trteus' in task:
        balance_strategy = 'undersample_train_test'
    comb_type = 'type2'
    n_split=5
    process=True
    if thold == None:
        thold = 0.5

    loc_dict = get_locs()

    res_loc = config.ROOT_DIR / 'results' / 'elrrf' / 'cross_multi_cancer_validation.csv'
    grid_search_settings_loc = config.ROOT_DIR / 'results' / 'elrrf' / 'result.json'
    data_choices = '|'.join(data_choice_list)
    if os.path.isfile(res_loc):
        result_df = pd.read_csv(res_loc)
    else:
        result_df = pd.DataFrame(
            columns=['datasets', 'cancer_test', 'AUROC_m', 'AUROC_std', 'AUPRC_m', 'AUPRC_std',
                     'MC_m', 'MC_std', 'grid_search', 'use_single', 'n_split',
                     'balance_strat', 'threshold', 'comb_type', 'process'])
    chosen_cols = result_df[
        ['datasets', 'cancer_test', 'grid_search', 'use_single', 'n_split', 'balance_strat', 'threshold',
         'comb_type', 'process']]
    chosen_cols = chosen_cols.astype(str)
    chosen_vals = np.array([data_choices, cancer_test, grid_search, use_single, n_split, balance_strategy, thold,
                            comb_type, process])
    chosen_vals = chosen_vals.astype(str)
    if (chosen_cols == chosen_vals).all(1).any():
        print(f'{data_choices} is already calculated!')
        return 0

    if comb_type=='type1':
        elrrf = ELRRF(use_single=use_single, grid_searched=grid_search, balance_strategy=balance_strategy, thold=thold,
                      process=process)
    elif comb_type=='type2':
        use_comb = False
        use_all_comb = True
        elrrf = ELRRF(use_single=use_single, grid_searched=grid_search, balance_strategy=balance_strategy,
                      use_comb=use_comb, thold=thold, process=process)

    results = {}
    for i in range(n_split):
        results[i]={}
    dataset_names = data_choice_list.copy()
    dataset_names.append('&'.join(data_choice_list))
    for cancer_train in cancer_trains:
        model_loc = data_choices + '_' + cancer_train + '_' + cancer_test + '_' + str(grid_search) + '_' + str(
            use_single) + '_' + comb_type + '_' + \
                    '_' + balance_strategy + '.pickle'
        model_loc = config.ROOT_DIR / 'results' / 'elrrf' / 'models_cc' / model_loc
        cancer_res = dfnc.load_pickle(model_loc)
        for i in range(n_split):
            for key in dataset_names:
                cancer_res[i][cancer_train+'_'+key] = cancer_res[i].pop(key)
            results[i].update(cancer_res[i])

    AUROC, AUPRC, MC = elrrf.evaluate_folds(results)

    res_dict = {'datasets': data_choices,  'cancer_test': cancer_test,
                'AUROC_m': np.around(AUROC.mean(), 2), 'AUROC_std': np.around(np.std(AUROC), 2),
                'AUPRC_m': np.around(AUPRC.mean(), 2), 'AUPRC_std': np.around(np.std(AUPRC), 2),
                'MC_m': np.around(MC.mean(), 2), 'MC_std': np.around(np.std(MC), 2),
                'grid_search': grid_search, 'use_single': use_single, 'n_split': n_split,
                'balance_strat': balance_strategy, 'threshold': thold, 'comb_type': comb_type, 'process': process}

    if os.path.isfile(res_loc):
        result_df = pd.read_csv(res_loc)
    result_df = result_df.append(res_dict, ignore_index=True)

    result_df.to_csv(res_loc, index=False)

    # fold_predictions = elrrf.fit_predict_all_datasets_fold(train_test_sample_dict=train_test_sample_names[cancer])
    # elrrf.evaluate_folds(fold_predictions)

    print(f'Experiment with {data_choice_list} finished from all to {cancer_test}...')
    print()


def pretrain_validation_experiment(data_choice_list, use_single, cancer_pretrain, cancer_test, grid_search, thold=None):
    print(f'Experiment: {data_choice_list} from {cancer_pretrain} to {cancer_test}, grid_search={grid_search}')
    if 'trus' in task:
        balance_strategy = 'undersample_train'
    if 'trteus' in task:
        balance_strategy = 'undersample_train_test'
    comb_type = 'type2'
    fold_type = 'stratified_shuffled'
    n_split=5
    process=False
    if thold == None:
        thold = 0.5

    loc_dict = get_locs()

    res_loc = config.ROOT_DIR / 'results' / 'elrrf' / 'pretrain_validation.csv'
    grid_search_settings_loc = config.ROOT_DIR / 'results' / 'elrrf' / 'result.json'
    data_choices = '|'.join(data_choice_list)
    if os.path.isfile(res_loc):
        result_df = pd.read_csv(res_loc)
    else:
        result_df = pd.DataFrame(
            columns=['datasets', 'cancer_pretrain', 'cancer_test', 'AUROC_m', 'AUROC_std', 'AUPRC_m', 'AUPRC_std',
                     'MC_m', 'MC_std', 'grid_search', 'use_single', 'n_split',
                     'balance_strat', 'threshold', 'comb_type', 'process'])
    chosen_cols = result_df[
        ['datasets', 'cancer_pretrain', 'cancer_test', 'grid_search', 'use_single', 'n_split', 'balance_strat', 'threshold',
         'comb_type', 'process']]
    chosen_cols = chosen_cols.astype(str)
    chosen_vals = np.array([data_choices, cancer_pretrain, cancer_test, grid_search, use_single, n_split, balance_strategy, thold,
                            comb_type, process])
    chosen_vals = chosen_vals.astype(str)
    if (chosen_cols == chosen_vals).all(1).any():
        print(f'{data_choices} is already calculated!')
        return 0

    if comb_type=='type1':
        elrrf_pre = ELRRF(use_single=use_single, grid_searched=grid_search, balance_strategy=balance_strategy, thold=thold,
                      process=process, return_model=True)
        elrrf_post = ELRRF(use_single=use_single, grid_searched=grid_search, balance_strategy=balance_strategy, thold=thold,
                      process=process)
    elif comb_type=='type2':
        use_comb = False
        use_all_comb = True
        elrrf_pre = ELRRF(use_single=use_single, grid_searched=grid_search, balance_strategy=balance_strategy,
                      use_comb=use_comb, thold=thold, process=process, return_model=True)
        elrrf_post = ELRRF(use_single=use_single, grid_searched=grid_search, balance_strategy=balance_strategy,
                      use_comb=use_comb, thold=thold, process=process, return_model=False)

    model_loc = data_choices + '_' + cancer_pretrain + '_' + cancer_test + '_' + str(grid_search) + '_' + str(
        use_single) + '_' + comb_type + '_' + \
                '_' + balance_strategy + '.pickle'
    model_loc = config.ROOT_DIR / 'results' / 'elrrf' / 'models_pretrain' / model_loc
    config.ensure_dir(model_loc)
    if os.path.isfile(model_loc):
        results_post = dfnc.load_pickle(model_loc)
    else:
        samples_loc = 'labels/train_pairs.csv'
        samples_loc = config.DATA_DIR / samples_loc
        samples = pd.read_csv(samples_loc)
        samples_tr = prepare_cancer_dataset(samples, cancer=cancer_pretrain)
        samples_te = prepare_cancer_dataset(samples, cancer=cancer_test)
        if comb_type == 'type2' and len(data_choice_list) > 1:
            final_trdata = samples_tr.copy()
            final_tedata = samples_te.copy()
        for data_choice in data_choice_list:
            data = pd.read_csv(loc_dict['train_' + data_choice + '_data_loc'])
            data = data.fillna(0)
            processed_data_tr = prepare_cancer_dataset(data, cancer=cancer_pretrain)
            processed_data_te = prepare_cancer_dataset(data, cancer=cancer_test)
            elrrf_pre.add_dataset(data_choice, processed_data_tr)
            elrrf_post.add_dataset(data_choice, processed_data_te)
            #elrrf.add_dataset(data_choice, processed_data_tr,processed_data_te)
            if comb_type=='type2' and len(data_choice_list) > 1:
                processed_data_tr = processed_data_tr.drop(columns='class')
                final_trdata = pd.merge(final_trdata, processed_data_tr, how='inner', on=['pair_name'])
                processed_data_te = processed_data_te.drop(columns='class')
                final_tedata = pd.merge(final_tedata, processed_data_te, how='inner', on=['pair_name'])
        if comb_type=='type2' and len(data_choice_list) > 1:
            elrrf_pre.add_dataset('&'.join(data_choice_list), final_trdata)
            elrrf_post.add_dataset('&'.join(data_choice_list), final_tedata)
            #elrrf.add_dataset('&'.join(data_choice_list), final_trdata,final_tedata)

        model_dict={}
        results_pre = elrrf_pre.fit_set(samples_tr, n_split=n_split)
        elrrf_post.set_params(**{'pretrained_model':results_pre})
        results_post = elrrf_post.fit_predict_cross_validation(samples_te, fold_type=fold_type, n_split=n_split)
        model_dict.update(results_post)
        print()

        dfnc.save_pickle(model_loc, model_dict)

    AUROC, AUPRC, MC = elrrf_post.evaluate_folds(results_post)

    res_dict = {'datasets': data_choices, 'cancer_pretrain': cancer_pretrain, 'cancer_test': cancer_test,
                'AUROC_m': np.around(AUROC.mean(), 2), 'AUROC_std': np.around(np.std(AUROC), 2),
                'AUPRC_m': np.around(AUPRC.mean(), 2), 'AUPRC_std': np.around(np.std(AUPRC), 2),
                'MC_m': np.around(MC.mean(), 2), 'MC_std': np.around(np.std(MC), 2),
                'grid_search': grid_search, 'use_single': use_single, 'n_split': n_split,
                'balance_strat': balance_strategy, 'threshold': thold, 'comb_type': comb_type, 'process': process}

    if os.path.isfile(res_loc):
        result_df = pd.read_csv(res_loc)
    result_df = result_df.append(res_dict, ignore_index=True)

    result_df.to_csv(res_loc, index=False)

    # fold_predictions = elrrf.fit_predict_all_datasets_fold(train_test_sample_dict=train_test_sample_names[cancer])
    # elrrf.evaluate_folds(fold_predictions)

    print(f'Experiment with {data_choice_list} finished from {cancer_pretrain} to {cancer_test}...')
    print()


def LOCO_validation_experiment(data_choice_list, use_single, cancer, grid_search, thold=None):
    print(f'Experiment: {data_choice_list} all to {cancer}, grid_search={grid_search}')
    if 'trus' in task:
        balance_strategy = 'undersample_train'
    if 'trteus' in task:
        balance_strategy = 'undersample_train_test'
    comb_type = 'type2c'
    is_cancer=False
    if comb_type=='type2c':
        is_cancer=True
    n_split=5
    process=True
    if thold == None:
        thold = 0.5

    loc_dict = get_locs()

    res_loc = config.ROOT_DIR / 'results' / 'elrrf' / 'loco_gbdt_validation.csv'
    grid_search_settings_loc = config.ROOT_DIR / 'results' / 'elrrf' / 'result.json'
    data_choices = '|'.join(data_choice_list)
    if os.path.isfile(res_loc):
        result_df = pd.read_csv(res_loc)
    else:
        result_df = pd.DataFrame(
            columns=['datasets', 'cancer', 'AUROC_m', 'AUROC_std', 'AUPRC_m', 'AUPRC_std',
                     'MC_m', 'MC_std', 'grid_search', 'use_single', 'n_split',
                     'balance_strat', 'threshold', 'comb_type', 'process'])
    chosen_cols = result_df[
        ['datasets', 'cancer', 'grid_search', 'use_single', 'n_split', 'balance_strat', 'threshold',
         'comb_type', 'process']]
    chosen_cols = chosen_cols.astype(str)
    chosen_vals = np.array([data_choices, cancer, grid_search, use_single, n_split, balance_strategy, thold,
                            comb_type, process])
    chosen_vals = chosen_vals.astype(str)
    if (chosen_cols == chosen_vals).all(1).any():
        print(f'{data_choices} is already calculated!')
        return 0

    if comb_type=='type1':
        elrrf = ELRRF(use_single=use_single, grid_searched=grid_search, balance_strategy=balance_strategy, thold=thold,
                      process=process, return_model=False)
    elif comb_type=='type2' or comb_type=='type2c':
        use_comb = False
        use_all_comb = True
        elrrf = ELRRF(use_single=use_single, grid_searched=grid_search, balance_strategy=balance_strategy,
                      use_comb=use_comb, thold=thold, process=process, return_model=False)

    model_loc = data_choices + '_' + cancer + '_' + str(grid_search) + '_' + str(
        use_single) + '_' + comb_type + \
                '_' + balance_strategy + '.pickle'
    model_loc = config.ROOT_DIR / 'results' / 'elrrf' / 'models_gbdt_loco' / model_loc
    config.ensure_dir(model_loc)
    if os.path.isfile(model_loc):
        results = dfnc.load_pickle(model_loc)
    else:
        samples_loc = 'labels/train_pairs.csv'
        samples_loc = config.DATA_DIR / samples_loc
        samples = pd.read_csv(samples_loc)
        samples_tr = prepare_cancer_dataset(samples, cancer=cancer, reverse=True, is_cancer=is_cancer, reduce_min=False)
        samples_te = prepare_cancer_dataset(samples, cancer=cancer)
        if 'type2' in comb_type and len(data_choice_list) > 1:
            final_trdata = samples_tr.copy()
            if 'cancer' in final_trdata.columns:
                final_trdata = final_trdata.drop(columns='cancer')
            final_tedata = samples_te.copy()
        for data_choice in data_choice_list:
            data = pd.read_csv(loc_dict['train_' + data_choice + '_data_loc'])
            data = data.fillna(0)
            processed_data_tr = prepare_cancer_dataset(data, cancer=cancer, reverse=True)
            processed_data_te = prepare_cancer_dataset(data, cancer=cancer)
            elrrf.add_dataset(data_choice, processed_data_tr,processed_data_te)
            if 'type2' in comb_type and len(data_choice_list) > 1:
                processed_data_tr = processed_data_tr.drop(columns='class')
                final_trdata = pd.merge(final_trdata, processed_data_tr, how='inner', on=['pair_name'])
                processed_data_te = processed_data_te.drop(columns='class')
                final_tedata = pd.merge(final_tedata, processed_data_te, how='inner', on=['pair_name'])
        if 'type2' in comb_type and len(data_choice_list) > 1:
            elrrf.add_dataset('&'.join(data_choice_list), final_trdata,final_tedata)

        random_grid = elrrf.get_tree_search_grid()
        best_params = {}
        if grid_search:
            search_space = list()
            for par_name, par_val in random_grid.items():
                search_space.append(Categorical(par_val, name=par_name))

            @use_named_args(search_space)
            def evaluate_model(**params):
                elrrf.set_params(**params)
                result_one = elrrf.fit_predict_2set(samples_tr, samples_te, n_split=n_split)
                AUROC, AUPRC, MC = elrrf.evaluate_folds(result_one, report=False)
                return 1 - np.mean(AUPRC)

            result_best = gp_minimize(evaluate_model, search_space, n_calls=10)
            for idx in range(len(search_space)):
                best_params[search_space[idx]._name] = result_best.x[idx]

        if best_params:
            elrrf.set_params(**best_params)
        param_list = ['boosting_type', 'num_leaves', 'max_depth', 'learning_rate', 'n_estimators',
                      'subsample_for_bin', 'objective', 'class_weight', 'min_split_gain',
                      'min_child_weight', 'min_child_samples', 'subsample', 'subsample_freq',
                      'colsample_bytree', 'reg_alpha', 'reg_lambda', 'random_state', 'n_jobs',
                      'verbose', 'thold', 'process', 'is_ready_data',
                      'use_single', 'use_comb', 'grid_searched', 'sep_train_test', 'balance',
                      'balance_strategy', 'return_model']
        all_params = elrrf.get_params(deep=False)
        selected_params = {}
        for key in param_list:
            selected_params[key] = all_params.pop(key)
        model_dict = {'params': selected_params}
        if best_params:
            model_dict['best_params'] = best_params
        results = elrrf.fit_predict_2set(samples_tr, samples_te, n_split=n_split)
        model_dict.update(results)
        dfnc.save_pickle(model_loc, model_dict)

    AUROC, AUPRC, MC = elrrf.evaluate_folds(results)

    res_dict = {'datasets': data_choices, 'cancer': cancer,
                'AUROC_m': np.around(AUROC.mean(), 2), 'AUROC_std': np.around(np.std(AUROC), 2),
                'AUPRC_m': np.around(AUPRC.mean(), 2), 'AUPRC_std': np.around(np.std(AUPRC), 2),
                'MC_m': np.around(MC.mean(), 2), 'MC_std': np.around(np.std(MC), 2),
                'grid_search': grid_search, 'use_single': use_single, 'n_split': n_split,
                'balance_strat': balance_strategy, 'threshold': thold, 'comb_type': comb_type, 'process': process}

    if os.path.isfile(res_loc):
        result_df = pd.read_csv(res_loc)
    result_df = result_df.append(res_dict, ignore_index=True)

    result_df.to_csv(res_loc, index=False)

    # fold_predictions = elrrf.fit_predict_all_datasets_fold(train_test_sample_dict=train_test_sample_names[cancer])
    # elrrf.evaluate_folds(fold_predictions)

    print(f'Experiment with {data_choice_list} finished from all to {cancer}...')
    print()


def get_locs():
    loc_dict = coll.OrderedDict()
    for t_set in ['train', 'test', 'isle', 'dsl', 'lu15', 'exp2sl', 'unknown_cancer_BRCA', 'unknown_repair_cancer_BRCA',
                  'unknown_families_BRCA', 'negative_families_LUAD', 'unknown_families_extra_BRCA', 'negative_families_extra_LUAD']:
        loc_dict[t_set + '_colm_data_loc'] = config.DATA_DIR / ('feature_sets/colm_full_' + t_set +'.csv')

        for i in range(4, 11, 1):
            dim = str(int(np.power(2, i)))
            loc_dict[t_set + '_seq_' + dim + '_data_loc'] = config.DATA_DIR / (
                        'feature_sets/' + t_set + '_seq_' + dim + '.csv')
        for other in ['ppi_ec']:
            loc_dict[t_set + '_'+other+'_data_loc'] = config.DATA_DIR / ('feature_sets/' + t_set + '_'+other+'.csv.gz')
        for other in ['onehot']:
            loc_dict[t_set + '_'+other+'_data_loc'] = config.DATA_DIR / ('feature_sets/' + t_set + '_'+other+'.csv')
        for dep_type in ['crispr', 'd2', 'comb']:
            for omic in ['mut', 'cnv', 'comb', 'muex']:
                loc_dict[t_set + '_' + dep_type + '_dependency_' + omic + '_data_loc'] = \
                    config.DATA_DIR / ('feature_sets/' + t_set + '_' + dep_type + '_dependency_' + omic + '.csv.gz')
            for omic in ['expr', 'any']:
                for extra in ['co=1.96']:
                    loc_dict[t_set + '_' + dep_type + '_dependency_' + omic + '_data_loc'] = \
                        config.DATA_DIR / (
                                    'feature_sets/' + t_set + '_' + dep_type + '_dependency_' + omic + '_' + extra + '.csv.gz')
        for tis_type in ['surv', 'tcoexp', 'hcoexp', 'coexp', 'diff_expr']:
            for extra in ['co=1.96']:
                loc_dict[t_set + '_tissue_' + tis_type + '_data_loc'] = \
                    config.DATA_DIR / ('feature_sets/' + t_set + '_tissue_' + tis_type + '_' + extra + '.csv.gz')
        for tis_type in ['diff_expr']:
            loc_dict[t_set + '_tissue_' + tis_type + '_data_loc'] = \
                config.DATA_DIR / ('feature_sets/' + t_set + '_tissue_' + tis_type + '.csv.gz')
        loc_dict[t_set + '_tissue_data_loc'] = \
            config.DATA_DIR / ('feature_sets/' + t_set + '_tissue.csv.gz')

    for dt in ['PPI', 'GO1', 'GO2']:
        loc_dict['GCATSL_' + dt + '_data_loc'] = \
            config.DATA_DIR / 'feature_sets' / 'GCATSL' / 'ppi_128dim.csv'

    return loc_dict


def get_data_choices(seq=None, singles=False, onehot=False, dep_comb=False, all_comb=False, final_comb=False,
                     best_comb=False, final_with_tissue_comb=False, final_with_without_tissue_comb=False,
                     chosen_singles=False, only_seq=False, seq_colm=False, extra_single=False):
    if seq==None:
        seq = ['seq_1024']#, 'seq_512', 'seq_256', 'seq_128', 'seq_64', 'seq_32', 'seq_16']
    choices = []
    # choices.append([seq])
    # choices.append([seq])
    # choices.append([seq])
    # choices.append([seq])
    if only_seq:
        choices.append([seq])
    if chosen_singles:
        choices.append([seq])
        choices.append(['ppi_ec'])
        choices.append(['crispr_dependency_mut'])
        choices.append(['d2_dependency_mut'])
        choices.append(['crispr_dependency_expr'])
        choices.append(['d2_dependency_expr'])
        choices.append(['tissue'])
    if singles:
        choices.append([seq])
        choices.append(['ppi_ec'])
        choices.append(['crispr_dependency_mut'])
        choices.append(['d2_dependency_mut'])
        choices.append(['comb_dependency_mut'])
        choices.append(['crispr_dependency_cnv'])
        choices.append(['d2_dependency_cnv'])
        choices.append(['comb_dependency_cnv'])
        choices.append(['crispr_dependency_expr'])
        choices.append(['d2_dependency_expr'])
        choices.append(['comb_dependency_expr'])
        choices.append(['crispr_dependency_comb'])
        choices.append(['d2_dependency_comb'])
        choices.append(['comb_dependency_comb'])
        choices.append(['crispr_dependency_any'])
        choices.append(['d2_dependency_any'])
        choices.append(['comb_dependency_any'])
        choices.append(['crispr_dependency_muex'])
        choices.append(['d2_dependency_muex'])
        choices.append(['comb_dependency_muex'])
        #choices.append(['tissue_tcoexp'])
        #choices.append(['tissue_hcoexp'])
        #choices.append(['tissue_coexp'])
        #choices.append(['tissue_diff_expr'])
        #choices.append(['tissue_surv'])
        choices.append(['tissue'])
    if onehot:
        choices.append(['onehot'])
    if dep_comb:
        choices.append(['crispr_dependency_mut', 'd2_dependency_mut'])
        choices.append(['crispr_dependency_cnv', 'd2_dependency_cnv'])
        choices.append(['crispr_dependency_expr', 'd2_dependency_expr'])
        choices.append(['crispr_dependency_comb', 'd2_dependency_comb'])
        choices.append(['crispr_dependency_mut', 'crispr_dependency_expr'])
        choices.append(['d2_dependency_mut', 'd2_dependency_expr'])
        choices.append(['comb_dependency_mut', 'comb_dependency_expr'])
        choices.append(['crispr_dependency_muex', 'd2_dependency_muex'])
        choices.append(
            ['crispr_dependency_mut', 'crispr_dependency_expr', 'd2_dependency_mut', 'd2_dependency_expr'])
        choices.append(['crispr_dependency_mut', 'crispr_dependency_cnv', 'crispr_dependency_expr'])
        choices.append(['d2_dependency_mut', 'd2_dependency_cnv', 'd2_dependency_expr'])
        choices.append(['crispr_dependency_mut', 'crispr_dependency_cnv', 'crispr_dependency_expr',
                        'd2_dependency_mut', 'd2_dependency_cnv', 'd2_dependency_expr'])
        choices.append(['comb_dependency_mut', 'comb_dependency_cnv', 'comb_dependency_expr'])
        # choices.append(['original'])
    if all_comb:
        choices.append([seq, 'ppi_ec'])
        choices.append([seq, 'ppi_ec', 'crispr_dependency_mut'])
        choices.append([seq, 'ppi_ec', 'd2_dependency_mut'])
        choices.append([seq, 'crispr_dependency_mut', 'd2_dependency_mut'])
        choices.append([seq, 'ppi_ec', 'crispr_dependency_mut', 'd2_dependency_mut'])
        choices.append([seq, 'ppi_ec', 'crispr_dependency_comb', 'd2_dependency_comb'])
        choices.append([seq, 'ppi_ec', 'comb_dependency_mut'])
        choices.append([seq, 'ppi_ec', 'comb_dependency_comb'])
        choices.append([seq, 'ppi_ec', 'crispr_dependency_muex', 'd2_dependency_muex'])
        choices.append([seq, 'crispr_dependency_mut', 'crispr_dependency_expr'])
        choices.append([seq, 'crispr_dependency_muex', 'd2_dependency_muex'])
        choices.append(
            [seq, 'crispr_dependency_mut', 'crispr_dependency_expr', 'd2_dependency_mut', 'd2_dependency_expr'])
    if final_comb:
        choices.append([seq, 'ppi_ec', 'crispr_dependency_mut', 'crispr_dependency_expr'])
        choices.append([seq, 'ppi_ec', 'crispr_dependency_muex', 'd2_dependency_muex'])
        #choices.append([seq, 'ppi_ec', 'crispr_dependency_mut', 'crispr_dependency_expr', 'd2_dependency_mut',
        #                'd2_dependency_expr'])
    if final_with_tissue_comb:
        choices.append([seq, 'ppi_ec', 'crispr_dependency_mut', 'crispr_dependency_expr', 'tissue'])
        choices.append([seq, 'ppi_ec', 'crispr_dependency_muex', 'd2_dependency_muex', 'tissue'])
    if final_with_without_tissue_comb:
        choices.append([seq, 'ppi_ec', 'crispr_dependency_mut', 'crispr_dependency_expr'])
        choices.append([seq, 'ppi_ec', 'crispr_dependency_muex', 'd2_dependency_muex'])
        choices.append([seq, 'ppi_ec', 'crispr_dependency_mut', 'crispr_dependency_expr', 'tissue'])
        choices.append([seq, 'ppi_ec', 'crispr_dependency_muex', 'd2_dependency_muex', 'tissue'])
    if best_comb:
        choices.append([seq, 'ppi_ec', 'crispr_dependency_mut', 'crispr_dependency_expr', 'tissue'])
    if seq_colm:
        choices.append([seq, 'colm'])
    if extra_single:
        choices.append(['crispr_dependency_mut', 'crispr_dependency_expr'])


    return choices

def main_single_contr_test(choices, thold):
    cancers = ['BRCA', 'CESC', 'KIRC', 'LAML', 'LUAD', 'OV', 'SKCM', 'COAD']
    #cancers = ['BRCA', 'COAD', 'LUAD', 'OV']
    #cancers = ['COAD', 'SKCM']
    #cancers=['KIRC']
    cancers=[args.cancer]
    use_single = True
    gs=False
    if 'gs' in task:
        gs=True
    for cancer in cancers:
        for choice_list in choices:
            single_contr_test_experiment(choice_list, cancer=cancer,
                                                grid_search=gs, thold=thold)

def main_single_cancer_test(choices, thold):
    cancers = ['BRCA', 'CESC', 'KIRC', 'LAML', 'LUAD', 'OV', 'SKCM', 'COAD']
    #cancers = ['BRCA', 'COAD', 'LUAD', 'OV']
    #cancers = ['COAD', 'SKCM']
    #cancers=['KIRC']
    cancers=[args.cancer]
    use_single = True
    gs=False
    if 'gs' in task:
        gs=True
    for cancer in cancers:
        for choice_list in choices:
            #if 'GCATSL' in task:
            #    single_cancer_GCATSL_test_experiment(cancer, gs, thold)
            #else:
            single_cancer_test_experiment(choice_list, cancer=cancer,
                                                grid_search=gs, thold=thold)

def main_permut_ds_detail_test(choices, thold):
    all_cancers = ['BRCA', 'CESC', 'KIRC', 'LAML', 'LUAD', 'OV', 'SKCM', 'COAD']
    #cancers = ['BRCA', 'COAD', 'LUAD', 'OV']
    #cancers = ['COAD', 'SKCM']
    #cancers=['KIRC']
    use_single = True
    gs=False
    if 'gs' in task:
        gs=True
    canc_list = all_cancers
    if args.cancer != 'any':
        canc_list = [args.cancer]
    for cancer in canc_list:
        for choice_list in choices:
            #if 'GCATSL' in task:
            #    single_cancer_GCATSL_test_experiment(cancer, gs, thold)
            #else:
            single_cancer_dataset_imp_detail_experiment(choice_list, cancer=cancer,
                                                grid_search=gs, thold=thold)

def main_permut_ds_test(choices, thold):
    cancers = ['BRCA', 'CESC', 'KIRC', 'LAML', 'LUAD', 'OV', 'SKCM', 'COAD']
    #cancers = ['BRCA', 'COAD', 'LUAD', 'OV']
    #cancers = ['COAD', 'SKCM']
    #cancers=['KIRC']
    cancers=[args.cancer]
    use_single = True
    gs=False
    if 'gs' in task:
        gs=True
    for cancer in cancers:
        for choice_list in choices:
            #if 'GCATSL' in task:
            #    single_cancer_GCATSL_test_experiment(cancer, gs, thold)
            #else:
            single_cancer_dataset_imp_experiment(choice_list, cancer=cancer,
                                                grid_search=gs, thold=thold)

def main_cross_cancer_test(choices, thold):
    cancers = ['BRCA', 'CESC', 'COAD', 'KIRC', 'LAML', 'LUAD', 'OV', 'SKCM']
    #cancers = ['BRCA', 'COAD', 'LUAD', 'OV']
    #cancers = ['KIRC', 'CESC']
    use_single = True
    gs=False
    if 'gs' in task:
        gs=True
    for tr_cancer in [args.cancer]:
        for te_cancer in cancers:
            if tr_cancer != te_cancer:
                for choice_list in choices:
                    cross_cancer_test_experiment(choice_list,  cancer_train=tr_cancer,
                                                       cancer_test = te_cancer, grid_search=gs, thold=thold)

def main_cross_ds_test(choices, thold):
    cancers = ['BRCA', 'CESC', 'COAD', 'KIRC', 'LAML', 'LUAD', 'OV', 'SKCM']
    #cancers = ['BRCA', 'COAD', 'LUAD', 'OV']
    #cancers = ['KIRC', 'CESC']
    use_single = True
    train_test_dict = {'BRCA': ['dsl', 'isle'], 'LUAD': ['exp2sl', 'dsl']}
    gs=False
    if 'gs' in task:
        gs=True
    for tr_cancer in [args.cancer]:
        for choice_list in choices:
            cross_dataset_test_experiment(choice_list,  cancer=tr_cancer, train_ds = train_test_dict[tr_cancer][0],
                                          test_ds = train_test_dict[tr_cancer][1], grid_search=gs, thold=thold)

def main_unknown_test(choices, thold):
    cancers = ['BRCA', 'CESC', 'COAD', 'KIRC', 'LAML', 'LUAD', 'OV', 'SKCM']
    #cancers = ['BRCA', 'COAD', 'LUAD', 'OV']
    #cancers = ['KIRC', 'CESC']
    use_single = True
    cancers=['BRCA']
    train_test_dict = {'BRCA': ['train', 'unknown_families_extra_BRCA'],
                       'LUAD': ['train', 'negative_families_extra_LUAD']}
    gs=False
    if 'gs' in task:
        gs=True
    for tr_cancer in [args.cancer]:
        for choice_list in choices:
            cross_dataset_test_experiment(choice_list,  cancer=tr_cancer, train_ds = train_test_dict[tr_cancer][0],
                                          test_ds = train_test_dict[tr_cancer][1], grid_search=gs, thold=thold)


def main_cross_mcancer_test(choices, thold):
    all_cancers = ['BRCA', 'CESC', 'COAD', 'KIRC', 'LAML', 'LUAD', 'OV', 'SKCM']
    #cancers = ['BRCA', 'COAD', 'LUAD', 'OV']
    use_single = True
    gs=False
    if 'gs' in task:
        gs=True
    canc_list = all_cancers
    if args.cancer != 'any':
        canc_list = [args.cancer]
    for te_cancer in canc_list:
        tr_cancers = list(set(all_cancers)-set([te_cancer]))
        tr_cancers.sort()
        for choice_list in choices:
            multi_cross_cancer_test_experiment(choice_list, cancer_trains=tr_cancers,
                                               cancer_test = te_cancer, grid_search=gs, thold=thold)

def main_loco_test(choices, thold):
    cancers = ['BRCA', 'CESC', 'COAD', 'KIRC', 'LAML', 'LUAD', 'OV', 'SKCM']
    #cancers = ['BRCA', 'COAD', 'LUAD', 'OV']
    #cancers = ['CESC']
    use_single = True
    gs=False
    if 'gs' in task:
        gs=True
    for cancer in cancers:
        for choice_list in choices:
            LOCO_validation_experiment(choice_list, use_single=use_single, cancer=cancer,
                                                grid_search=gs, thold=thold)

def main_single_cancer_val(choices, thold):
    cancers = ['BRCA', 'CESC', 'COAD', 'KIRC', 'LAML', 'LUAD', 'OV', 'SKCM']#
    #cancers = ['BRCA', 'COAD', 'LAML', 'LUAD', 'OV']
    #cancers = ['CESC', 'KIRC', 'LAML', 'SKCM']
    #cancers=['OV']
    use_single = True
    gs=False
    if 'gs' in task:
        gs=True
    for cancer in cancers:
        for choice_list in choices:
            single_cancer_validation_experiment(choice_list, use_single=use_single, cancer=cancer,
                                                grid_search=gs, thold=thold)

def main_dho_cancer_test(choices, thold):
    cancers = ['BRCA', 'CESC', 'COAD', 'KIRC', 'LAML', 'LUAD', 'OV', 'SKCM']#
    #cancers = ['BRCA', 'COAD', 'LAML', 'LUAD', 'OV']
    cancers = ['BRCA', 'OV']#, 'LUAD', 'OV']
    use_single = True
    gs=False
    if 'gs' in task:
        gs=True
    for cancer in [args.cancer]:
        for choice_list in choices:
            single_cancer_dho_test_experiment(data_choice_list=choice_list, cancer=cancer, grid_search=gs, thold=thold)
            #if 'GCATSL' in task:
            #    single_cancer_GCATSL_dho_test_experiment(cancer, gs, thold)
            #else:
            #    single_cancer_dho_test_experiment(choice_list, use_single=use_single, cancer=cancer,
            #                                    grid_search=gs, thold=thold, ho='dho2')


def main_double_ho_cancer_val(choices, thold):
    cancers = ['BRCA', 'CESC', 'COAD', 'KIRC', 'LAML', 'LUAD', 'OV', 'SKCM']#
    #cancers = ['BRCA', 'COAD', 'LAML', 'LUAD', 'OV']
    cancers = ['BRCA', 'LUAD', 'OV']
    use_single = True
    gs=False
    if 'gs' in task:
        gs=True
    for cancer in cancers:
        for choice_list in choices:
            single_cancer_ho2_validation_experiment(choice_list, use_single=use_single, cancer=cancer,
                                                grid_search=gs, thold=thold, ho='dho2')

def main_cross_cancer_val(choices, thold):
    cancers = ['BRCA', 'CESC', 'COAD', 'KIRC', 'LAML', 'LUAD', 'OV', 'SKCM']
    #cancers = ['BRCA', 'COAD', 'LUAD', 'OV']
    #cancers = ['OV']
    use_single = True
    gs=False
    if 'gs' in task:
        gs=True
    for tr_cancer in cancers:
        for te_cancer in cancers:
            if tr_cancer != te_cancer:
                for choice_list in choices:
                    cross_cancer_validation_experiment(choice_list, use_single=use_single, cancer_train=tr_cancer,
                                                       cancer_test = te_cancer, grid_search=gs, thold=thold)

def main_cross_multi_cancer_val(choices, thold):
    all_cancers = ['BRCA', 'CESC', 'COAD', 'KIRC', 'LAML', 'LUAD', 'OV', 'SKCM']
    #cancers = ['BRCA', 'COAD', 'LUAD', 'OV']
    cancers = ['KIRC']
    use_single = True
    gs=False
    if 'gs' in task:
        gs=True
    for te_cancer in cancers:
        tr_cancers = list(set(all_cancers)-set([te_cancer]))
        tr_cancers.sort()
        for choice_list in choices:
            cross_multi_cancer_validation_experiment(choice_list, use_single=use_single, cancer_trains=tr_cancers,
                                               cancer_test = te_cancer, grid_search=gs, thold=thold)

def main_pretrain_val(choices, thold):
    cancers = ['BRCA', 'CESC', 'COAD', 'KIRC', 'LAML', 'LUAD', 'OV', 'SKCM']
    #cancers = ['BRCA', 'COAD', 'LUAD', 'OV']
    #cancers = ['OV']
    #cancers=['CESC','KIRC']
    use_single = True
    gs=False
    if 'gs' in task:
        gs=True
    for tr_cancer in cancers:
        for te_cancer in cancers:
            if tr_cancer != te_cancer:
                for choice_list in choices:
                    pretrain_validation_experiment(choice_list, use_single=use_single, cancer_pretrain=tr_cancer,
                                                       cancer_test = te_cancer, grid_search=gs, thold=thold)

def main_loco_val(choices, thold):
    cancers = ['BRCA', 'CESC', 'COAD', 'KIRC', 'LAML', 'LUAD', 'OV', 'SKCM']
    #cancers = ['BRCA', 'COAD', 'LUAD', 'OV']
    cancers = ['CESC']
    use_single = True
    gs=False
    if 'gs' in task:
        gs=True
    for cancer in cancers:
        for choice_list in choices:
            LOCO_validation_experiment(choice_list, use_single=use_single, cancer=cancer,
                                                grid_search=gs, thold=thold)

def main():
    logging.info(f'{task} starts')
    print(f'{task} starts')
    warnings.filterwarnings("ignore")
    seqs = ['seq_1024']#, 'seq_512', 'seq_256', 'seq_128','seq_64','seq_32']#, 'seq_128', 'seq_64','seq_32','seq_16']
    for thold in [0.5]:#0.4, 0.425, 0.45, 0.5, 0.525, 0.55, 0.575, 0.6, 0.625, 0.65, 0.7, 0.75, 0.8]:
        for seq in seqs:
            choices = get_data_choices(seq, best_comb=True)
            if 'single' in task and 'val' in task:
                main_single_cancer_val(choices, thold)
            elif 'single' in task and 'test' in task:
                main_single_cancer_test(choices, thold)
            elif 'contr' in task and 'test' in task:
                main_single_contr_test(choices, thold)
            elif 'crossc' in task and 'val' in task:
                main_cross_cancer_val(choices, thold)
            elif 'cross_mc' in task and 'val' in task:
                main_cross_multi_cancer_val(choices, thold)
            elif 'loco' in task and 'val' in task:
                main_loco_val(choices, thold)
            elif 'pretrain' in task and 'val' in task:
                main_pretrain_val(choices, thold)
            elif 'holdout_d' in task and 'val' in task:
                main_double_ho_cancer_val(choices, thold)
            elif 'dho' in task and 'test' in task:
                main_dho_cancer_test(choices, thold)
            elif 'crossc' in task and 'test' in task:
                main_cross_cancer_test(choices, thold)
            elif 'crossds' in task and 'test' in task:
                main_cross_ds_test(choices, thold)
            elif 'crossmc' in task and 'test' in task:
                main_cross_mcancer_test(choices, thold)
            elif 'crossallmc' in task and 'test' in task:
                main_cross_mcancer_test(choices, thold)
            elif 'permut_detail' in task and 'test' in task:
                main_permut_ds_detail_test(choices, thold)
            elif 'permut' in task and 'test' in task:
                main_permut_ds_test(choices, thold)
            elif 'unknown' in task and 'test' in task:
                main_unknown_test(choices, thold)



if __name__ == '__main__':
    main()
