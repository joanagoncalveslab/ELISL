import os
import sys
path2this = os.path.dirname(os.path.abspath(__file__)).split('/')
for i, folder in enumerate(path2this):
    if folder.lower()=='elisl':
        project_path = '/'.join(path2this[:i+1])
sys.path.insert(0,project_path)
from src.models.ELRRF import *
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

GCATSL_root = str(config.ROOT_DIR / 'src' / 'comparison' / 'GCATSL/')

PROJECT_LOC = config.ROOT_DIR

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
        train_test_ind_loc = config.DATA_DIR / 'feature_sets' / f'train_test_{model.cancer}_{n_split}.json'
    else:
        train_test_ind_loc = config.DATA_DIR / 'feature_sets' / \
                             f'train_test_{model.cancer}_{ds_names[0]}_{ds_names[1]}_{n_split}.json'

    if ds_names is not None and 'train' == ds_names[0]:
        train_ind_loc = config.DATA_DIR / 'feature_sets' / f'train_test_{model.cancer}_{n_split}.json'
        with open(train_ind_loc, 'r') as fpp:
            train_inds = json.load(fpp)

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
    train_test_ind_loc = config.DATA_DIR / 'feature_sets' / f'train_test_{model.cancer}_{n_split}.json'
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