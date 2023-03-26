import os
import sys
path2this = os.path.dirname(os.path.abspath(__file__)).split('/')
for i, folder in enumerate(path2this):
    if folder.lower()=='elisl':
        project_path = '/'.join(path2this[:i+1])
sys.path.insert(0,project_path)
from src.models.ELRRF import *
#from src.comparison.GRSMF.GRSMF import *
import pandas as pd
import numpy as np
import json
import src.data_functions as dfnc
from src import config
from src import experiment
import logging
#GCATSL_root = str(config.ROOT_DIR / 'src' / 'comparison' / 'GCATSL/')

PROJECT_LOC = config.ROOT_DIR
import argparse

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

parser = argparse.ArgumentParser(description='Run Single Cancer Experiment')
parser.add_argument('--model', '-m', metavar='the-model', dest='model', type=str, help='Choose model', default='ELRRF')
parser.add_argument('--balance_strategy', '-bs', metavar='the-balance_strategy', dest='balance_strategy', type=str, help='Choose balancing', default='undersample_train_test')
parser.add_argument('--grid_search', '--gs', metavar='the-grid_search', dest='grid_search', type=str2bool, nargs='?', const=True, help='Grid searched?', default=False)
parser.add_argument('--cancer', '-c', metavar='the-cancer', dest='cancer', type=str, help='Choose cancer', default='BRCA')
parser.add_argument('--n_split', '-ns', metavar='the-no-of-split', dest='n_split', type=int, help='Choose number of split', default=10)
parser.add_argument('--process', '--p', metavar='the-process', dest='process', type=str2bool, nargs='?', const=True, help='Processing?', default=False)
parser.add_argument('--threshold', '-th', metavar='the-threshold', dest='threshold', type=float, help='Choose threshold', default=0.5)
parser.add_argument('--exp_file', '-ef', metavar='the-experiment_file', dest='exp_file', type=str, help='Choose experiment description file', default=config.DATA_DIR/"experiment_configs"/"double_holdout_config.json")

args = parser.parse_args()
print(f'Running args:{args}')
task = f'DHO_{args.model}_{args.cancer}_{args.grid_search}_{args.balance_strategy}'
log_name = config.ROOT_DIR / 'logs' / f'{task}.txt'
config.ensure_dir(log_name)
logging.basicConfig(level=logging.DEBUG, filename=log_name, filemode="a+",
                    format="%(asctime)-15s %(levelname)-8s %(message)s")
logging.info(f'Double holdout experiment started for with arguments\n{args}')

PROJECT_LOC = config.ROOT_DIR


def dho_cancer_test_experiment():
    model_type = args.model
    n_split = args.n_split
    process = args.process
    balance_strategy = args.balance_strategy
    cancer = args.cancer
    grid_search = args.grid_search
    comb_type = 'type2'
    use_single = True
    if model_type == 'ELRRF':
        boosting = 'rf'
    elif model_type == 'ELGBDT':
        boosting = 'gbdt'

    with open(args.exp_file) as f:
        exp_config = json.load(f)

    data_choice_list = list(exp_config['feature_sets'].keys())
    data_choices = '|'.join(data_choice_list)
    logging.info(f'Experiment configuration:\n{exp_config}')

    threshold = args.threshold

    res_loc = config.ROOT_DIR / 'results' / model_type / 'dho_cancer_experiment.csv'
    config.ensure_dir(res_loc)
    all_res_col_names = ['datasets', 'cancer', 'AUROC_m', 'AUROC_std', 'AUPRC_m', 'AUPRC_std',
                 'MC_m', 'MC_std', 'grid_search', 'use_single', 'n_split',
                 'balance_strat', 'threshold', 'comb_type', 'process']
    res_col_names = ['datasets', 'cancer', 'grid_search', 'use_single', 'n_split', 'balance_strat', 'threshold',
                     'comb_type', 'process']
    if os.path.isfile(res_loc):
        result_df = pd.read_csv(res_loc)
        chosen_cols = result_df[res_col_names].astype(str)
        chosen_vals = np.array([data_choices, cancer, grid_search, use_single, n_split, balance_strategy, threshold,
                                comb_type, process]).astype(str)

        if (chosen_cols == chosen_vals).all(1).any():
            logging.info(f'{data_choices} is already calculated!')
            print(f'{data_choices} is already calculated!')
            return 0
    else:
        result_df = pd.DataFrame(columns=all_res_col_names)
    model = ELRRF(boosting_type=boosting, use_single=True, grid_searched=grid_search, balance_strategy=balance_strategy,
                  use_comb=False, thold=threshold, process=process, cancer=cancer)
    model_loc = f'{data_choices}_{cancer}_{grid_search}_{n_split}_{balance_strategy}.pickle'

    model_loc = config.ROOT_DIR / 'results' / model_type / 'models_dho_cancer' / model_loc
    config.ensure_dir(model_loc)
    if os.path.isfile(model_loc):
        results = dfnc.load_pickle(model_loc)
    else:
        tr_samples_loc = config.DATA_DIR / exp_config['labels']['train']
        tr_samples = pd.read_csv(tr_samples_loc)
        te_samples_loc = config.DATA_DIR / exp_config['labels']['test']
        te_samples = pd.read_csv(te_samples_loc)
        samples_tr = experiment.prepare_cancer_dataset(tr_samples, cancer=cancer)
        samples_te = experiment.prepare_cancer_dataset(te_samples, cancer=cancer)
        samples = pd.concat([samples_tr, samples_te])
        #If model is Early Late
        if len(data_choice_list) > 1:
            final_trdata = samples_tr.copy()
            final_tedata = samples_te.copy()
        for data_choice, data_choice_dict in exp_config['feature_sets'].items():
            tr_data = pd.read_csv(config.DATA_DIR / data_choice_dict['train'])
            tr_data = tr_data.fillna(0)
            te_data = pd.read_csv(config.DATA_DIR / data_choice_dict['test'])
            te_data = te_data.fillna(0)
            processed_data_tr = experiment.prepare_cancer_dataset(tr_data, cancer=cancer)
            processed_data_te = experiment.prepare_cancer_dataset(te_data, cancer=cancer)
            processed_data = pd.concat([processed_data_tr, processed_data_te])
            model.add_dataset(data_choice, processed_data)
            if len(data_choice_list) > 1:
                processed_data_tr = processed_data_tr.drop(columns='class')
                final_trdata = pd.merge(final_trdata, processed_data_tr, how='inner', on=['pair_name'])
                processed_data_te = processed_data_te.drop(columns='class')
                final_tedata = pd.merge(final_tedata, processed_data_te, how='inner', on=['pair_name'])
        if len(data_choice_list) > 1:
            model.add_dataset('&'.join(data_choice_list), pd.concat([final_trdata, final_tedata]))

        param_list = ['boosting_type', 'num_leaves', 'max_depth', 'learning_rate', 'n_estimators',
                      'subsample_for_bin', 'objective', 'class_weight', 'min_split_gain',
                      'min_child_weight', 'min_child_samples', 'subsample', 'subsample_freq',
                      'colsample_bytree', 'reg_alpha', 'reg_lambda', 'random_state', 'n_jobs',
                      'verbose', 'thold', 'process', 'is_ready_data',
                      'use_single', 'use_comb', 'grid_searched', 'sep_train_test', 'balance',
                      'balance_strategy', 'return_model', 'contr']

        results = experiment.fit_predict_dho_cross_validation(model, samples, n_split=n_split)
        all_params = model.get_params(deep=False)
        selected_params = {}
        for key in param_list:
            selected_params[key] = all_params.pop(key)
        model_dict = {'params': selected_params}
        model_dict.update(results)
        dfnc.save_pickle(model_loc, model_dict)

    AUROC, AUPRC, MC = model.evaluate_folds(results)
    res_dict = {'datasets': data_choices, 'cancer': cancer,
                'AUROC_m': np.around(AUROC.mean(), 2), 'AUROC_std': np.around(np.std(AUROC), 2),
                'AUPRC_m': np.around(AUPRC.mean(), 2), 'AUPRC_std': np.around(np.std(AUPRC), 2),
                'MC_m': np.around(MC.mean(), 2), 'MC_std': np.around(np.std(MC), 2),
                'grid_search': grid_search, 'use_single': use_single, 'n_split': n_split,
                    'balance_strat': balance_strategy,
                'threshold': threshold, 'comb_type': comb_type, 'process': process}

    if os.path.isfile(res_loc):
        result_df = pd.read_csv(res_loc)
    result_df = result_df.append(res_dict, ignore_index=True)

    result_df.to_csv(res_loc, index=False)

    # fold_predictions = elrrf.fit_predict_all_datasets_fold(train_test_sample_dict=train_test_sample_names[cancer])
    # elrrf.evaluate_folds(fold_predictions)

    print(f'Experiment dho with {data_choice_list} finished for {cancer}...')
    logging.info(f'Experiment dho with {data_choice_list} finished for {cancer}...')
    print()

def main():
    dho_cancer_test_experiment()

if __name__ == '__main__':
    main()