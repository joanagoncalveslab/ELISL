import warnings
import os
import sys
path2this = os.path.dirname(os.path.abspath(__file__)).split('/')
for i, folder in enumerate(path2this):
    if folder.lower()=='elisl':
        project_path = '/'.join(path2this[:i+1])
sys.path.insert(0,project_path)
import collections as coll
from collections import defaultdict
import pandas as pd
import numpy as np
from src import config
# import pyreadr
from sklearn.metrics import auc, roc_auc_score, average_precision_score, make_scorer, precision_recall_curve, \
    matthews_corrcoef
from sklearn import preprocessing
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, KFold
import src.data_functions as dfnc
from glmnet import LogitNet
from skopt.space import Real
from skopt.space import Categorical
from skopt.utils import use_named_args
from skopt import gp_minimize

import logging


#task='crossc_val_trteus'
task = 'cross_mc_val_trus'
#task='single_val_trteus'
#task='loco_val_trteus'
log_name = 'ELGLMNET_' +task
logging.basicConfig(level=logging.DEBUG, filename=log_name, filemode="a+",
                    format="%(asctime)-15s %(levelname)-8s %(message)s")

PROJECT_LOC = config.ROOT_DIR
RANDOMIZING=False


class ELGLMNET:
    def __init__(self, *, alpha=1, n_lambda=100, min_lambda_ratio=1e-4, lambda_path=None, standardize=True,
                 fit_intercept=True, lower_limits=-np.inf, upper_limits=np.inf, cut_point=1, n_splits=3,
                 scoring='average_precision', n_jobs=1, tol=1e-7, max_iter=100000, max_features=False, random_state=124,
                 thold=0.5, process=True, is_ready_data=False, use_single=True, use_comb=True,
                 grid_searched=False, sep_train_test=False, balance=True, balance_strategy='class_weight_train',
                 return_model=False):
        self.alpha = alpha
        self.n_lambda = n_lambda
        self.min_lambda_ratio = min_lambda_ratio
        self.lambda_path = lambda_path
        self.standardize = standardize
        self.fit_intercept = fit_intercept
        self.lower_limits = lower_limits
        self.upper_limits = upper_limits
        self.cut_point = cut_point
        self.n_splits = n_splits
        self.scoring = scoring
        self.n_jobs = n_jobs
        self.tol = tol
        self.max_iter = max_iter
        self.max_features = max_features
        self.random_state = random_state
        self.datasets_dict = coll.OrderedDict()
        self.train_datasets_dict = coll.OrderedDict()
        self.test_datasets_dict = coll.OrderedDict()
        self.both_datasets_dict = coll.OrderedDict()
        self.no_of_datasets = 0
        self.thold = thold
        self.datasets_dict = coll.OrderedDict()
        self.no_of_datasets = 0
        self.process = process
        self.is_ready_data = is_ready_data
        self.ready_datasets = coll.OrderedDict()
        self.no_of_ready_datasets = 0
        self.use_single = use_single
        self.use_comb = use_comb
        self.grid_searched = grid_searched
        self.sep_train_test = sep_train_test
        self.balance = balance
        self.balance_strategy = balance_strategy
        self.return_model=return_model

    def get_params(self, deep=True):
        """
        Get parameters for this estimator.
        Parameters
        ----------
        deep : bool, default=True
            If True, will return the parameters for this estimator and
            contained subobjects that are estimators.
        Returns
        -------
        params : dict
            Parameter names mapped to their values.
        """
        out = dict()
        for key in dir(self):
            value = getattr(self, key)
            if deep and hasattr(value, 'get_params'):
                deep_items = value.get_params().items()
                out.update((key + '__' + k, val) for k, val in deep_items)
            out[key] = value
        return out

    def set_params(self, **params):
        """
        Set the parameters of this estimator.
        The method works on simple estimators as well as on nested objects
        (such as :class:`~sklearn.pipeline.Pipeline`). The latter have
        parameters of the form ``<component>__<parameter>`` so that it's
        possible to update each component of a nested object.
        Parameters
        ----------
        **params : dict
            Estimator parameters.
        Returns
        -------
        self : estimator instance
            Estimator instance.
        """
        if not params:
            # Simple optimization to gain speed (inspect is slow)
            return self
        valid_params = self.get_params(deep=False)

        nested_params = defaultdict(dict)  # grouped by prefix
        for key, value in params.items():
            key, delim, sub_key = key.partition('__')
            if key not in valid_params:
                raise ValueError('Invalid parameter %s for estimator %s. '
                                 'Check the list of available parameters '
                                 'with `estimator.get_params().keys()`.' %
                                 (key, self))

            if delim:
                nested_params[key][sub_key] = value
            else:
                setattr(self, key, value)
                valid_params[key] = value

        for key, sub_params in nested_params.items():
            valid_params[key].set_params(**sub_params)

        return self

    def add_dataset(self, dataset_name, dataset, dataset_te=None):
        """
        Parameters
        ----------
        dataset_name:
            Name of the dataset
        dataset:
            pandas dataframe with columns = [sample_name, label, features0, ..., featuresn]
        """
        if dataset_te is None:
            self.datasets_dict[dataset_name] = dataset
            self.no_of_datasets = self.no_of_datasets + 1
        else:
            self.both_datasets_dict[dataset_name] = {'train': dataset, 'test':dataset_te}
            self.no_of_datasets = self.no_of_datasets + 1

    def add_datasets(self, datasets_to_add=coll.OrderedDict()):
        """
        Parameters
        ----------
        datasets_to_add:
            Ordered dict where
                key=dataset_name,
                value= pandas dataframe with columns = [sample_name, label, features0, ..., featuresn]
        """
        for ds_name, ds in datasets_to_add.items():
            self.add_dataset(ds_name, ds)

    def balance_set(self, x, y, names, n=None, rand_state=-1):
        if rand_state == -1:
            rand_state = self.random_state
        n = np.unique(y, return_counts=True)[1].min() if n is None else n
        neg_indices = np.where(y == 0)[0]
        pos_indices = np.where(y == 1)[0]

        np.random.seed(rand_state)
        neg_sample_ind = np.random.choice(np.unique(y, return_counts=True)[1][0], n, replace=False)
        pos_sample_ind = np.random.choice(np.unique(y, return_counts=True)[1][1], n, replace=False)

        chosen_neg_ind = neg_indices[neg_sample_ind]
        chosen_pos_ind = pos_indices[pos_sample_ind]

        x_neg = x[chosen_neg_ind, :]
        x_pos = x[chosen_pos_ind, :]
        new_x = np.concatenate((x_neg, x_pos))

        y_neg = y[chosen_neg_ind]
        y_pos = y[chosen_pos_ind]
        new_y = np.concatenate((y_neg, y_pos))

        names_neg = names[chosen_neg_ind]
        names_pos = names[chosen_pos_ind]
        new_names = np.concatenate((names_neg, names_pos))

        return new_x, new_y, new_names

    def find_comb(self, choice=None):
        comb_dict = coll.OrderedDict()
        for ds1_name, ds1_data in self.datasets_dict.items():
            if self.use_single:
                comb_dict[ds1_name] = ds1_data
            if self.use_comb:
                for ds2_name in self.datasets_dict.keys():
                    ds_pair_name = min(ds1_name, ds2_name) + '|' + max(ds1_name, ds2_name)
                    if ds1_name != ds2_name and ds_pair_name not in comb_dict.keys():
                        comb_dict[ds_pair_name] = self.comb_2datasets(ds_pair_name)
        return comb_dict

    def comb_2datasets(self, ds_pair_name):
        df1 = self.datasets_dict[ds_pair_name.split('|')[0]]
        df2 = self.datasets_dict[ds_pair_name.split('|')[1]]
        df2 = df2.drop(columns='class')
        comb_df = pd.merge(df1, df2, how='inner', on=['pair_name'])
        return comb_df

    def prepare_train_test(self, df, train_test_ind=[]):
        train_ind = train_test_ind['train']
        test_ind = train_test_ind['test']

        if type(df) is dict:
            train_x = df['train'].iloc[train_ind].drop(columns=['pair_name', 'class']).values
            test_x = df['test'].iloc[test_ind].drop(columns=['pair_name', 'class']).values
        else:
            train_x = df.iloc[train_ind].drop(columns=['pair_name', 'class']).values
            test_x = df.iloc[test_ind].drop(columns=['pair_name', 'class']).values

        # test_names = df.loc[test_ind]['pair_name'].values
        # test_y = df.loc[test_ind]['SL'].values

        if RANDOMIZING:
            for idx in range(len(train_x)):
                np.random.seed(idx)
                np.random.shuffle(train_x[idx])
            lst_idx = idx
            for idx in range(len(test_x)):
                np.random.seed(lst_idx+idx)
                np.random.shuffle(test_x[idx])


        if self.process:
            fitted_scale = preprocessing.StandardScaler().fit(train_x)
            train_x = fitted_scale.transform(train_x)
            test_x = fitted_scale.transform(test_x)
            try:
                train_fits = VarianceThreshold(threshold=1e-04).fit(train_x)
                train_x = train_fits.transform(train_x)
                test_x = train_fits.transform(test_x)
            except:
                print('Variance could not be applied')
            # train_x = VarianceThreshold(threshold=1e-04).fit_transform(train_x)
            # test_x = VarianceThreshold(threshold=1e-04).fit_transform(test_x)

        # return train_x, train_y, test_x, test_y, train_names, test_names
        return train_x, test_x,

    def precision_recall_scorer(self, truth, pred):
        precision, recall, thresholds = precision_recall_curve(truth, pred)
        return auc(recall, precision)

    def get_tree_search_grid(self):
        random_grid = {'alpha': [x for x in np.linspace(start=0, stop=1, num=5)]}
        search_space = list()
        search_space.append(Real(0, 1, 'uniform', name='alpha'))
        return random_grid

    def fit_predict_report(self, train_x, train_y, test_x):
        pos = sum(train_y)
        neg = len(train_y) - pos
        ratio = neg / pos
        weights = None
        if 'class_weight' in self.balance_strategy:
            fraction_0 = np.repeat(1 - sum(train_y == 0) / len(train_y), sum(train_y == 0))
            fraction_1 = np.repeat(1 - sum(train_y == 1) / len(train_y), sum(train_y == 1))
            # assign that value to a "weights" vector
            weights = np.ones(len(train_y))
            weights[train_y == 0] = fraction_0
            weights[train_y == 1] = fraction_1
        if False:#self.grid_searched:
            clf = LogitNet(standardize=self.standardize, fit_intercept=self.fit_intercept,
                           lambda_path=[x for x in np.linspace(start=1, stop=0.0001, num=50)],
                           lower_limits=self.lower_limits, upper_limits=self.upper_limits, cut_point=self.cut_point,
                           n_splits=10, scoring=self.scoring, n_jobs=-1, tol=self.tol,
                           max_iter=self.max_iter, random_state=self.random_state, max_features=self.max_features)
            scorer = make_scorer(self.precision_recall_scorer, greater_is_better=True, needs_threshold=True)
            cv = 5
            clf = RandomizedSearchCV(estimator=clf, param_distributions=self.get_tree_search_grid(),
                                     random_state=self.random_state, n_jobs=-1, verbose=-1, n_iter=5, cv=cv,
                                     scoring=scorer)
        else:
            clf = LogitNet(alpha=self.alpha, n_lambda=self.n_lambda, min_lambda_ratio=self.min_lambda_ratio,
                           lambda_path=self.lambda_path, standardize=self.standardize, fit_intercept=self.fit_intercept,
                           lower_limits=self.lower_limits, upper_limits=self.upper_limits, cut_point=self.cut_point,
                           n_splits=self.n_splits, scoring=self.scoring, n_jobs=self.n_jobs, tol=self.tol,
                           max_iter=self.max_iter, random_state=self.random_state, max_features=self.max_features)
        clf.fit(train_x, train_y, sample_weight=weights)
        zero_ind = np.where(clf.classes_ == 0)[0][0]
        one_ind = np.where(clf.classes_ == 1)[0][0]
        tr_probs = clf.predict_proba(train_x)[:, one_ind]
        tr_auc = average_precision_score(train_y, tr_probs)
        probs = clf.predict_proba(test_x)[:, one_ind]
        preds = clf.predict(test_x)
        return probs, preds, tr_auc, clf

    def evaluate_all_models(self, all_models):
        label_dict = coll.OrderedDict()
        pred_df = pd.DataFrame(columns=['pair_name', 'model', 'prediction', 'probability'])
        for model_name, model_res in all_models.items():
            model_pred_df = pd.DataFrame(columns=['pair_name', 'model', 'prediction', 'probability'])
            model_pred_df['model'] = np.repeat([model_name], len(model_res['predictions']))
            model_pred_df['pair_name'] = model_res['test_names']
            model_pred_df['prediction'] = model_res['predictions']
            model_pred_df['probability'] = model_res['probabilities']
            model_pred_df['tr_auc'] = model_res['tr_auc']
            pred_df = pred_df.append(model_pred_df, ignore_index=True)
            model_labels = dict(zip(model_res['test_names'], model_res['labels']))
            label_dict.update(model_labels)
        # grouped_preds = pred_df.groupby(['pair_name'])['prediction'].mean().reset_index()
        grouped_preds = pred_df.groupby(['pair_name']).apply(combine_preds).reset_index()
        grouped_preds.loc[grouped_preds['probability'] >= self.thold, 'pred'] = 1
        grouped_preds.loc[grouped_preds['probability'] < self.thold, 'pred'] = 0
        final_preds = grouped_preds.rename(columns={'probability': 'prob'})
        final_preds['label'] = final_preds['pair_name'].map(label_dict)
        scores = {}
        scores['AUROC'] = roc_auc_score(final_preds['label'], final_preds['prob'])
        scores['AUPRC'] = average_precision_score(final_preds['label'], final_preds['prob'])
        scores['MC'] = matthews_corrcoef(final_preds['label'], final_preds['pred'])

        return scores

    def evaluate_folds(self, folds_prediction, report=True):
        AUROC = []
        AUPRC = []
        MC = []
        for fold, models in folds_prediction.items():
            if type(fold)==int:
                scores = self.evaluate_all_models(models)
                AUROC.append(scores['AUROC'])
                AUPRC.append(scores['AUPRC'])
                MC.append(scores['MC'])
            # print(f'Fold={fold}, AUROC={scores["AUROC"]}, AUPRC={scores["AUPRC"]}')
        AUROC = np.array(AUROC)
        AUPRC = np.array(AUPRC)
        MC = np.array(MC)
        if report:
            print(f'Average AUROC score: {AUROC.mean()}, with Std.Dev.: {np.std(AUROC)}')
            print(f'Average AUPRC score: {AUPRC.mean()}, with Std.Dev.: {np.std(AUPRC)}')
            print(f'Average MC score: {MC.mean()}, with Std.Dev.: {np.std(MC)}')
        return AUROC, AUPRC, MC

    def fit_predict_all_datasets(self, sample_dict=[], samples=None):
        combinations = self.find_comb()
        predictions = {}
        train_samples = samples.iloc[sample_dict['train']]
        test_samples = samples.iloc[sample_dict['test']]
        for comb_name, comb_data in combinations.items():
            train_x, test_x = self.prepare_train_test(comb_data, sample_dict)
            # train_x, train_y, train_names = self.balance_set(train_x, train_y, train_names)
            predictions[comb_name] = {'train_names': train_samples['pair_name'].values,
                                      'test_names': test_samples['pair_name'].values}
            comb_probs, comb_preds, tr_auc, model = self.fit_predict_report(train_x, train_samples['class'].values, test_x)
            predictions[comb_name]['probabilities'] = comb_probs
            predictions[comb_name]['predictions'] = comb_preds
            predictions[comb_name]['labels'] = test_samples['class'].values
            predictions[comb_name]['tr_auc'] = tr_auc
            if self.return_model:
                predictions[comb_name]['model'] = model
        return predictions


    def fit_predict_all_datasets_2sets(self, sample_dict, samples_tr=None, samples_te=None):
        predictions = {}
        train_samples = samples_tr.iloc[sample_dict['train']]
        test_samples = samples_te.iloc[sample_dict['test']]
        for set_name, set_both_data in self.both_datasets_dict.items():
            train_x, test_x = self.prepare_train_test(set_both_data, sample_dict)
            # train_x, train_y, train_names = self.balance_set(train_x, train_y, train_names)
            predictions[set_name] = {'train_names': train_samples['pair_name'].values,
                                      'test_names': test_samples['pair_name'].values}
            comb_probs, comb_preds, tr_auc, model = self.fit_predict_report(train_x, train_samples['class'].values, test_x)
            predictions[set_name]['probabilities'] = comb_probs
            predictions[set_name]['predictions'] = comb_preds
            predictions[set_name]['labels'] = test_samples['class'].values
            predictions[set_name]['tr_auc'] = tr_auc
            if self.return_model:
                predictions[set_name]['model'] = model
        return predictions


    def fit_predict_cross_validation(self, samples, fold_type, n_split):
        if 'stratified' in fold_type and 'shuffle' in fold_type:
            skf = StratifiedKFold(n_splits=n_split, shuffle=True, random_state=self.random_state)
        elif 'stratified' in fold_type:
            skf = StratifiedKFold(n_splits=n_split, shuffle=False, random_state=self.random_state)
        elif 'shuffle' in fold_type:
            skf = KFold(n_splits=n_split, shuffle=True, random_state=self.random_state)
        else:
            skf = KFold(n_splits=n_split, shuffle=False, random_state=self.random_state)

        indices = skf.split(samples, samples['class'].values)
        fold_ind = coll.OrderedDict()
        fold_samples = coll.OrderedDict()
        for i, (tr_indices, te_indices) in enumerate(indices):
            if 'undersample' in self.balance_strategy and 'train' in self.balance_strategy:
                tr_indices = balance_by_index(samples, tr_indices, rand_seed=124)
            if 'undersample' in self.balance_strategy and 'test' in self.balance_strategy:
                te_indices = balance_by_index(samples, te_indices, rand_seed=124)
            fold_ind[i] = {'train': tr_indices, 'test': te_indices}
            fold_samples[i] = {'train': samples.iloc[tr_indices], 'test': samples.iloc[te_indices]}

        results = coll.OrderedDict()
        for fold, sample_dict in fold_ind.items():
            results[fold] = self.fit_predict_all_datasets(sample_dict, samples)

        return results


    def fit_predict_2set(self, samples_train, samples_test, n_split=5):
        fold_ind = coll.OrderedDict()
        fold_samples = coll.OrderedDict()
        tr_all_indices = np.array(list(range(len(samples_train))))
        te_all_indices = np.array(list(range(len(samples_test))))
        for i in range(n_split):
            tr_indices = tr_all_indices.copy()
            te_indices = te_all_indices.copy()
            if 'undersample' in self.balance_strategy and 'train' in self.balance_strategy:
                tr_indices = balance_by_index(samples_train, tr_all_indices, rand_seed=124+i)
            if 'undersample' in self.balance_strategy and 'test' in self.balance_strategy:
                te_indices = balance_by_index(samples_test, te_all_indices, rand_seed=124+i)
            fold_ind[i] = {'train': tr_indices, 'test': te_indices}
            fold_samples[i] = {'train': samples_train.iloc[tr_indices], 'test': samples_test.iloc[te_indices]}

        results = coll.OrderedDict()
        for fold, sample_dict in fold_ind.items():
            results[fold] = self.fit_predict_all_datasets_2sets(sample_dict, samples_train, samples_test)

        return results


def balance_by_index(samples, indices, rand_seed=-1):
    tr_samples = samples.iloc[indices]
    pos_indices = indices[tr_samples['class'].values == 1]
    neg_indices = indices[tr_samples['class'].values == 0]
    n = min(len(pos_indices), len(neg_indices))
    np.random.seed(rand_seed)
    if n == len(neg_indices):
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


def prepare_cancer_dataset(df, cancer='BRCA', reverse=False):
    df_cancer = df.copy()
    df_cancer.insert(loc=0, column='pair_name', value=df[['gene1', 'gene2']].agg('|'.join, axis=1))
    if cancer == None:
        df_cancer = df_cancer.reset_index()
    elif reverse:
        df_cancer = df_cancer[~(df_cancer['cancer'] == cancer)].reset_index()
    else:
        df_cancer = df_cancer[df_cancer['cancer'] == cancer].reset_index()

    df_cancer = df_cancer.drop(columns=['gene1', 'gene2', 'cancer', 'index'])
    df_cancer = df_cancer.sort_values(by=['pair_name'])
    return df_cancer


def single_cancer_validation_experiment(data_choice_list, use_single, cancer, grid_search, thold=None):
    print(f'Experiment: {data_choice_list} for {cancer} with use_single:{use_single}, grid_search={grid_search}')
    if 'trus' in task:
        balance_strategy = 'undersample_train'
    if 'trteus' in task:
        balance_strategy = 'undersample_train_test'
    n_split: int = 5
    fold_type = 'stratified_shuffled'
    comb_type = 'type2'
    process=True
    if thold == None:
        thold = 0.5

    loc_dict = get_locs()

    res_loc = config.ROOT_DIR / 'results' / 'elglmnet' / 'single_cancer_validation.csv'
    grid_search_settings_loc = config.ROOT_DIR / 'results' / 'elglmnet' / 'result.json'
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
    ##############Model################
    if comb_type=='type1':
        elglmnet = ELGLMNET(use_single=use_single, grid_searched=grid_search, balance_strategy=balance_strategy,
                            thold=thold, process=process)
    elif comb_type=='type2':
        use_comb = False
        use_all_comb = True
        elglmnet = ELGLMNET(use_single=use_single, grid_searched=grid_search, balance_strategy=balance_strategy,
                      use_comb=use_comb, thold=thold, process=process)

    model_loc = data_choices + '_' + cancer + '_' + str(grid_search) + '_' + str(
        use_single) + '_' + comb_type + '_' + fold_type + '_' + str(n_split) + \
                '_' + balance_strategy + '.pickle'
    model_loc = config.ROOT_DIR / 'results' / 'elglmnet' / 'models' / model_loc
    if os.path.isfile(model_loc):
        results = dfnc.load_pickle(model_loc)
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
            elglmnet.add_dataset(data_choice, processed_data)
            if comb_type=='type2' and len(data_choice_list) > 1:
                processed_data = processed_data.drop(columns='class')
                final_data = pd.merge(final_data, processed_data, how='inner', on=['pair_name'])
        if comb_type=='type2' and len(data_choice_list) > 1:
            elglmnet.add_dataset('&'.join(data_choice_list), final_data)

        best_params = {}
        if grid_search:
            search_space = list()
            search_space.append(Real(0, 1, 'uniform', name='alpha'))

            @use_named_args(search_space)
            def evaluate_model(**params):
                elglmnet.set_params(**params)
                result_one = elglmnet.fit_predict_cross_validation(samples, fold_type=fold_type, n_split=n_split)
                AUROC, AUPRC, MC = elglmnet.evaluate_folds(result_one, report=False)
                return 1 - np.mean(AUPRC)

            result_best = gp_minimize(evaluate_model, search_space, n_calls=20)
            for idx in range(len(search_space)):
                best_params[search_space[idx]._name] = result_best.x[idx]

        if best_params:
            elglmnet.set_params(**best_params)
        param_list = ['alpha', 'n_lambda', 'min_lambda_ratio', 'lambda_path', 'standardize',
                      'fit_intercept', 'lower_limits', 'upper_limits', 'cut_point', 'n_splits',
                      'scoring', 'n_jobs', 'tol', 'max_iter', 'max_features', 'random_state',
                      'thold', 'process', 'is_ready_data', 'use_single', 'use_comb',
                      'grid_searched', 'sep_train_test', 'balance', 'balance_strategy', 'return_model']
        all_params = elglmnet.get_params(deep=False)
        selected_params = {}
        for key in param_list:
            selected_params[key] = all_params.pop(key)
        model_dict = {'params': selected_params}
        if best_params:
            model_dict['best_params'] = best_params
        results = elglmnet.fit_predict_cross_validation(samples, fold_type=fold_type, n_split=n_split)
        model_dict.update(results)

        dfnc.save_pickle(model_loc, model_dict)

    AUROC, AUPRC, MC = elglmnet.evaluate_folds(results)
    ###########Model############

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

    # fold_predictions = elglmnet.fit_predict_all_datasets_fold(train_test_sample_dict=train_test_sample_names[cancer])
    # elglmnet.evaluate_folds(fold_predictions)

    print(f'Experiment with {data_choice_list} finished for {cancer}...')
    print()


def single_cancer_test_experiment(data_choice_list, use_single, cancer, grid_search, thold=None):
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

    res_loc = config.ROOT_DIR / 'results' / 'elglmnet' / 'single_cancer_test.csv'
    grid_search_settings_loc = config.ROOT_DIR / 'results' / 'elglmnet' / 'result.json'
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
        elglmnet = ELGLMNET(use_single=use_single, grid_searched=grid_search, balance_strategy=balance_strategy, thold=thold,
                      process=process)
    elif comb_type=='type2':
        use_comb = False
        use_all_comb = True
        elglmnet = ELGLMNET(use_single=use_single, grid_searched=grid_search, balance_strategy=balance_strategy,
                      use_comb=use_comb, thold=thold, process=process)

    model_loc = data_choices + '_' + cancer + '_' + str(grid_search) + '_' + str(
        use_single) + '_' + comb_type + \
                '_' + balance_strategy + '.pickle'
    model_loc = config.ROOT_DIR / 'results' / 'elglmnet' / 'models_test' / model_loc
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
            elglmnet.add_dataset(data_choice, processed_data_tr,processed_data_te)
            if comb_type=='type2' and len(data_choice_list) > 1:
                processed_data_tr = processed_data_tr.drop(columns='class')
                final_trdata = pd.merge(final_trdata, processed_data_tr, how='inner', on=['pair_name'])
                processed_data_te = processed_data_te.drop(columns='class')
                final_tedata = pd.merge(final_tedata, processed_data_te, how='inner', on=['pair_name'])
        if comb_type=='type2' and len(data_choice_list) > 1:
            elglmnet.add_dataset('&'.join(data_choice_list), final_trdata,final_tedata)

        random_grid = elglmnet.get_tree_search_grid()
        best_params = {}
        if grid_search:
            best_val_model_loc = data_choices + '_' + cancer + '_' + str(grid_search) + '_' + str(use_single) + \
                                 '_' + comb_type + '_' + 'stratified_shuffled' + '_' + str(n_split) + \
                                 '_' + balance_strategy + '.pickle'
            best_val_model_loc = config.ROOT_DIR / 'results' / 'elglmnet' / 'models' / best_val_model_loc
            best_val_model = dfnc.load_pickle(best_val_model_loc)

            best_params = best_val_model['best_params']

        if best_params:
            elglmnet.set_params(**best_params)
        param_list = ['alpha', 'n_lambda', 'min_lambda_ratio', 'lambda_path', 'standardize',
                      'fit_intercept', 'lower_limits', 'upper_limits', 'cut_point', 'n_splits',
                      'scoring', 'n_jobs', 'tol', 'max_iter', 'max_features', 'random_state',
                      'thold', 'process', 'is_ready_data', 'use_single', 'use_comb',
                      'grid_searched', 'sep_train_test', 'balance', 'balance_strategy', 'return_model']
        all_params = elglmnet.get_params(deep=False)
        selected_params = {}
        for key in param_list:
            selected_params[key] = all_params.pop(key)
        model_dict = {'params': selected_params}
        if best_params:
            model_dict['best_params']=best_params
        results = elglmnet.fit_predict_2set(samples_tr, samples_te, n_split=n_split)
        model_dict.update(results)
        dfnc.save_pickle(model_loc, model_dict)

    AUROC, AUPRC, MC = elglmnet.evaluate_folds(results)

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

    # fold_predictions = elglmnet.fit_predict_all_datasets_fold(train_test_sample_dict=train_test_sample_names[cancer])
    # elglmnet.evaluate_folds(fold_predictions)

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

    res_loc = config.ROOT_DIR / 'results' / 'elglmnet' / 'cross_cancer_validation.csv'
    grid_search_settings_loc = config.ROOT_DIR / 'results' / 'elglmnet' / 'result.json'
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
        elglmnet = ELGLMNET(use_single=use_single, grid_searched=grid_search, balance_strategy=balance_strategy, thold=thold,
                      process=process)
    elif comb_type=='type2':
        use_comb = False
        use_all_comb = True
        elglmnet = ELGLMNET(use_single=use_single, grid_searched=grid_search, balance_strategy=balance_strategy,
                      use_comb=use_comb, thold=thold, process=process)

    model_loc = data_choices + '_' + cancer_train + '_' + cancer_test + '_' + str(grid_search) + '_' + str(
        use_single) + '_' + comb_type + '_' + \
                '_' + balance_strategy + '.pickle'
    model_loc = config.ROOT_DIR / 'results' / 'elglmnet' / 'models_cc' / model_loc
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
            elglmnet.add_dataset(data_choice, processed_data_tr,processed_data_te)
            if comb_type=='type2' and len(data_choice_list) > 1:
                processed_data_tr = processed_data_tr.drop(columns='class')
                final_trdata = pd.merge(final_trdata, processed_data_tr, how='inner', on=['pair_name'])
                processed_data_te = processed_data_te.drop(columns='class')
                final_tedata = pd.merge(final_tedata, processed_data_te, how='inner', on=['pair_name'])
        if comb_type=='type2' and len(data_choice_list) > 1:
            elglmnet.add_dataset('&'.join(data_choice_list), final_trdata,final_tedata)

        random_grid = elglmnet.get_tree_search_grid()
        best_params = {}
        if grid_search:
            search_space = list()
            for par_name, par_val in random_grid.items():
                search_space.append(Categorical(par_val, name=par_name))

            @use_named_args(search_space)
            def evaluate_model(**params):
                elglmnet.set_params(**params)
                result_one = elglmnet.fit_predict_2set(samples_tr, samples_te, n_split=n_split)
                AUROC, AUPRC, MC = elglmnet.evaluate_folds(result_one, report=False)
                return 1 - np.mean(AUPRC)

            result_best = gp_minimize(evaluate_model, search_space, n_calls=10)
            for idx in range(len(search_space)):
                best_params[search_space[idx]._name] = result_best.x[idx]

        if best_params:
            elglmnet.set_params(**best_params)
        param_list = ['alpha', 'n_lambda', 'min_lambda_ratio', 'lambda_path', 'standardize',
                      'fit_intercept', 'lower_limits', 'upper_limits', 'cut_point', 'n_splits',
                      'scoring', 'n_jobs', 'tol', 'max_iter', 'max_features', 'random_state',
                      'thold', 'process', 'is_ready_data', 'use_single', 'use_comb',
                      'grid_searched', 'sep_train_test', 'balance', 'balance_strategy', 'return_model']
        all_params = elglmnet.get_params(deep=False)
        selected_params = {}
        for key in param_list:
            selected_params[key] = all_params.pop(key)
        model_dict = {'params': selected_params}
        if best_params:
            model_dict['best_params'] = best_params
        results = elglmnet.fit_predict_2set(samples_tr, samples_te, n_split=n_split)
        model_dict.update(results)
        dfnc.save_pickle(model_loc, model_dict)

    AUROC, AUPRC, MC = elglmnet.evaluate_folds(results)

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

    # fold_predictions = elglmnet.fit_predict_all_datasets_fold(train_test_sample_dict=train_test_sample_names[cancer])
    # elglmnet.evaluate_folds(fold_predictions)

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

    res_loc = config.ROOT_DIR / 'results' / 'elglmnet' / 'cross_multi_cancer_validation.csv'
    grid_search_settings_loc = config.ROOT_DIR / 'results' / 'elglmnet' / 'result.json'
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
        elglmnet = ELGLMNET(use_single=use_single, grid_searched=grid_search, balance_strategy=balance_strategy, thold=thold,
                      process=process)
    elif comb_type=='type2':
        use_comb = False
        use_all_comb = True
        elglmnet = ELGLMNET(use_single=use_single, grid_searched=grid_search, balance_strategy=balance_strategy,
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
        model_loc = config.ROOT_DIR / 'results' / 'elglmnet' / 'models_cc' / model_loc
        cancer_res = dfnc.load_pickle(model_loc)
        for i in range(n_split):
            for key in dataset_names:
                cancer_res[i][cancer_train+'_'+key] = cancer_res[i].pop(key)
            results[i].update(cancer_res[i])

    AUROC, AUPRC, MC = elglmnet.evaluate_folds(results)

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

    # fold_predictions = elglmnet.fit_predict_all_datasets_fold(train_test_sample_dict=train_test_sample_names[cancer])
    # elglmnet.evaluate_folds(fold_predictions)

    print(f'Experiment with {data_choice_list} finished from all to {cancer_test}...')
    print()



def LOCO_validation_experiment(data_choice_list, use_single, cancer, grid_search, thold=None):
    print(f'Experiment: {data_choice_list} all to {cancer}, grid_search={grid_search}')
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

    res_loc = config.ROOT_DIR / 'results' / 'elglmnet' / 'loco_validation.csv'
    grid_search_settings_loc = config.ROOT_DIR / 'results' / 'elglmnet' / 'result.json'
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
        elglmnet = ELGLMNET(use_single=use_single, grid_searched=grid_search, balance_strategy=balance_strategy, thold=thold,
                      process=process, return_model=False)
    elif comb_type=='type2':
        use_comb = False
        use_all_comb = True
        elglmnet = ELGLMNET(use_single=use_single, grid_searched=grid_search, balance_strategy=balance_strategy,
                      use_comb=use_comb, thold=thold, process=process, return_model=False)

    model_loc = data_choices + '_' + cancer + '_' + str(grid_search) + '_' + str(
        use_single) + '_' + comb_type + \
                '_' + balance_strategy + '.pickle'
    model_loc = config.ROOT_DIR / 'results' / 'elglmnet' / 'models_loco' / model_loc
    if os.path.isfile(model_loc):
        results = dfnc.load_pickle(model_loc)
    else:
        samples_loc = 'labels/train_pairs.csv'
        samples_loc = config.DATA_DIR / samples_loc
        samples = pd.read_csv(samples_loc)
        samples_tr = prepare_cancer_dataset(samples, cancer=cancer, reverse=True)
        samples_te = prepare_cancer_dataset(samples, cancer=cancer)
        if comb_type == 'type2' and len(data_choice_list) > 1:
            final_trdata = samples_tr.copy()
            final_tedata = samples_te.copy()
        for data_choice in data_choice_list:
            data = pd.read_csv(loc_dict['train_' + data_choice + '_data_loc'])
            data = data.fillna(0)
            processed_data_tr = prepare_cancer_dataset(data, cancer=cancer, reverse=True)
            processed_data_te = prepare_cancer_dataset(data, cancer=cancer)
            elglmnet.add_dataset(data_choice, processed_data_tr,processed_data_te)
            if comb_type=='type2' and len(data_choice_list) > 1:
                processed_data_tr = processed_data_tr.drop(columns='class')
                final_trdata = pd.merge(final_trdata, processed_data_tr, how='inner', on=['pair_name'])
                processed_data_te = processed_data_te.drop(columns='class')
                final_tedata = pd.merge(final_tedata, processed_data_te, how='inner', on=['pair_name'])
        if comb_type=='type2' and len(data_choice_list) > 1:
            elglmnet.add_dataset('&'.join(data_choice_list), final_trdata,final_tedata)

        random_grid = elglmnet.get_tree_search_grid()
        best_params = {}
        if grid_search:
            search_space = list()
            for par_name, par_val in random_grid.items():
                search_space.append(Categorical(par_val, name=par_name))

            @use_named_args(search_space)
            def evaluate_model(**params):
                elglmnet.set_params(**params)
                result_one = elglmnet.fit_predict_2set(samples_tr, samples_te, n_split=n_split)
                AUROC, AUPRC, MC = elglmnet.evaluate_folds(result_one, report=False)
                return 1 - np.mean(AUPRC)

            result_best = gp_minimize(evaluate_model, search_space, n_calls=10)
            for idx in range(len(search_space)):
                best_params[search_space[idx]._name] = result_best.x[idx]

        if best_params:
            elglmnet.set_params(**best_params)
        param_list = ['alpha', 'n_lambda', 'min_lambda_ratio', 'lambda_path', 'standardize',
                      'fit_intercept', 'lower_limits', 'upper_limits', 'cut_point', 'n_splits',
                      'scoring', 'n_jobs', 'tol', 'max_iter', 'max_features', 'random_state',
                      'thold', 'process', 'is_ready_data', 'use_single', 'use_comb',
                      'grid_searched', 'sep_train_test', 'balance', 'balance_strategy', 'return_model']
        all_params = elglmnet.get_params(deep=False)
        selected_params = {}
        for key in param_list:
            selected_params[key] = all_params.pop(key)
        model_dict = {'params': selected_params}
        if best_params:
            model_dict['best_params'] = best_params
        results = elglmnet.fit_predict_2set(samples_tr, samples_te, n_split=n_split)
        model_dict.update(results)
        dfnc.save_pickle(model_loc, model_dict)

    AUROC, AUPRC, MC = elglmnet.evaluate_folds(results)

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

    # fold_predictions = elglmnet.fit_predict_all_datasets_fold(train_test_sample_dict=train_test_sample_names[cancer])
    # elglmnet.evaluate_folds(fold_predictions)

    print(f'Experiment with {data_choice_list} finished from all to {cancer}...')
    print()


def get_locs():
    loc_dict = coll.OrderedDict()
    for t_set in ['train', 'test']:
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

    return loc_dict


def get_data_choices(seq=None, singles=False, onehot=False, dep_comb=False, all_comb=False, final_comb=False,
                     best_comb=False, final_with_tissue_comb=False, final_with_without_tissue_comb=False):
    if seq==None:
        seq = ['seq_1024']#, 'seq_512', 'seq_256', 'seq_128', 'seq_64', 'seq_32', 'seq_16']
    choices = []
    # choices.append([seq])
    # choices.append([seq])
    # choices.append([seq])
    # choices.append([seq])
    if singles:
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
        choices.append(['tissue_tcoexp'])
        choices.append(['tissue_hcoexp'])
        choices.append(['tissue_coexp'])
        choices.append(['tissue_diff_expr'])
        choices.append(['tissue_surv'])
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
        choices.append([seq, 'ppi_ec', 'crispr_dependency_mut', 'crispr_dependency_expr'])

    return choices

def main_single_cancer_test(choices, thold):
    cancers = ['BRCA', 'CESC', 'COAD', 'KIRC', 'LAML', 'LUAD', 'OV', 'SKCM']
    #cancers = ['BRCA', 'COAD', 'LUAD', 'OV']
    #cancers = ['OV']
    use_single = True
    gs=False
    if 'gs' in task:
        gs=True
    for cancer in cancers:
        for choice_list in choices:
            single_cancer_test_experiment(choice_list, use_single=use_single, cancer=cancer,
                                                grid_search=gs, thold=thold)

def main_cross_cancer_test(choices, thold):
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

def main_loco_test(choices, thold):
    cancers = ['BRCA', 'CESC', 'COAD', 'KIRC', 'LAML', 'LUAD', 'OV', 'SKCM']
    #cancers = ['BRCA', 'COAD', 'LUAD', 'OV']
    #cancers = ['COAD']
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
    #cancers = ['OV']
    use_single = True
    gs=False
    if 'gs' in task:
        gs=True
    for cancer in cancers:
        for choice_list in choices:
            single_cancer_validation_experiment(choice_list, use_single=use_single, cancer=cancer,
                                                grid_search=gs, thold=thold)

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
    cancers = ['BRCA', 'CESC', 'COAD', 'KIRC', 'LAML', 'LUAD', 'OV', 'SKCM']
    #cancers = ['BRCA', 'COAD', 'LUAD', 'OV']
    #cancers = ['OV']
    use_single = True
    gs=False
    if 'gs' in task:
        gs=True
    for te_cancer in cancers:
        tr_cancers = list(set(cancers)-set([te_cancer]))
        tr_cancers.sort()
        for choice_list in choices:
            cross_multi_cancer_validation_experiment(choice_list, use_single=use_single, cancer_trains=tr_cancers,
                                               cancer_test = te_cancer, grid_search=gs, thold=thold)

def main_loco_val(choices, thold):
    cancers = ['BRCA', 'CESC', 'COAD', 'KIRC', 'LAML', 'LUAD', 'OV', 'SKCM']
    #cancers = ['BRCA', 'COAD', 'LUAD', 'OV']
    #cancers = ['COAD']
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
    seqs = ['seq_1024', 'seq_512', 'seq_256', 'seq_128','seq_64','seq_32']#, 'seq_128', 'seq_64','seq_32','seq_16']
    for thold in [0.4, 0.425, 0.45, 0.5, 0.525, 0.55, 0.575, 0.6, 0.625, 0.65, 0.7, 0.75, 0.8]:
        for seq in seqs:
            choices = get_data_choices(seq, final_with_tissue_comb=True)
            if 'single' in task and 'val' in task:
                main_single_cancer_val(choices, thold)
            if 'single' in task and 'test' in task:
                main_single_cancer_test(choices, thold)
            if 'crossc' in task and 'val' in task:
                main_cross_cancer_val(choices, thold)
            if 'cross_mc' in task and 'val' in task:
                main_cross_multi_cancer_val(choices, thold)
            if 'loco' in task and 'val' in task:
                main_loco_val(choices, thold)



if __name__ == '__main__':
    main()