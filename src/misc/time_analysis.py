import os
import sys
path2this = os.path.dirname(os.path.abspath(__file__)).split('/')
for i, folder in enumerate(path2this):
    if folder.lower()=='elisl':
        project_path = '/'.join(path2this[:i+1])
sys.path.insert(0,project_path)
from src import config, data_functions as dfnc
import pandas as pd
import numpy as np

cancer_list = ['BRCA', 'CESC', 'COAD', 'KIRC', 'LAML', 'LUAD', 'OV', 'SKCM']
#cancer_list = ['BRCA', 'COAD', 'LAML', 'LUAD', 'OV']
chosen_th_train = {'BRCA':{'1024':0.475, '512':0.475, '256':0.475, '128':0.45, '64':0.475, '32':0.475},
            'CESC':{'1024':0.5, '512':0.5, '256':0.5, '128':0.55, '64':0.5, '32':0.575},
            'COAD':{'1024':0.525, '512':0.5, '256':0.5, '128':0.5, '64':0.5, '32':0.5},
            'KIRC':{'1024':0.5, '512':0.525, '256':0.525, '128':0.525, '64':0.525, '32':0.525},
            'LAML':{'1024':0.575, '512':0.575, '256':0.575, '128':0.575, '64':0.575, '32':0.575},
            'LUAD':{'1024':0.525, '512':0.55, '256':0.55, '128':0.55, '64':0.55, '32':0.55},
            'OV':{'1024':0.5, '512':0.525, '256':0.525, '128':0.525, '64':0.525, '32':0.55},
            'SKCM':{'1024':0.575, '512':0.575, '256':0.575, '128':0.575, '64':0.575, '32':0.55}}
chosen_th_train_test = {'BRCA':{'1024':0.475, '512':0.475, '256':0.475, '128':0.525, '64':0.5, '32':0.5},
            'CESC':{'1024':0.525, '512':0.525, '256':0.525, '128':0.525, '64':0.525, '32':0.525},
            'COAD':{'1024':0.5, '512':0.5, '256':0.5, '128':0.5, '64':0.5, '32':0.525},
            'KIRC':{'1024':0.525, '512':0.525, '256':0.475, '128':0.5, '64':0.5, '32':0.5},
            'LAML':{'1024':0.525, '512':0.525, '256':0.525, '128':0.525, '64':0.525, '32':0.525},
            'LUAD':{'1024':0.55, '512':0.5, '256':0.525, '128':0.5, '64':0.5, '32':0.55},
            'OV':{'1024':0.5, '512':0.5, '256':0.5, '128':0.5, '64':0.525, '32':0.5},
            'SKCM':{'1024':0.525, '512':0.525, '256':0.475, '128':0.525, '64':0.525, '32':0.475}}


def get_model_name(data_list=['seq_1024','ppi_ec','crispr_dependency_mut','crispr_dependency_expr','tissue'],
                   cancer='BRCA', grid_search='False', process='True', comb_type='type2', fold_type='stratified_shuffled',
                   n_split='5', balance_strat='undersample_train_test'):
    data_name = '|'.join(data_list)
    name = data_name+'_'+cancer+'_'+grid_search+'_'+process+'_'+comb_type+'_'+fold_type+'_'+n_split+'_'+balance_strat+\
           '.pickle'
    return name


def get_avg_time(loc='', folds=5, is_full=False):
    loc = config.ROOT_DIR / 'results' / loc
    model = dfnc.load_pickle(loc)
    if is_full:
        return float(model['full_time'])
    total_time=0.0
    for i in range(folds):
        total_time += float(model[i]['time'])
    avg_time = total_time/folds
    #print(avg_time)
    return avg_time

def main():
    cancers = ['BRCA', 'COAD', 'LAML', 'LUAD', 'OV']
    seqs = ['seq_1024', 'seq_512', 'seq_256', 'seq_128', 'seq_64', 'seq_32']
    seq_ind = []
    is_full=True
    for seq in seqs:
        seq_ind.append(seq)
        seq_ind.append('l_'+seq)
    res_df = pd.DataFrame(columns=cancers, index=seq_ind)
    for cancer in cancers:
        for seq in seqs:
            data_list = [seq,'ppi_ec','crispr_dependency_mut','crispr_dependency_expr','tissue']
            n_split='5'
            balance_strat = 'undersample_train_test'
            model_name = get_model_name(data_list=data_list, cancer=cancer, n_split=n_split, balance_strat=balance_strat)
            loc = 'elrrf/models/'+model_name
            avg_time = get_avg_time(loc, folds=int(n_split), is_full=is_full)
            res_df.loc[seq, cancer]=str(np.around(avg_time, 1))+' sec'
            loc_late = 'elrrf/models_late/'+model_name
            avg_l_time = get_avg_time(loc_late, folds=int(n_split), is_full=is_full)
            res_df.loc['l_'+seq, cancer]=str(np.around(avg_l_time, 1))+' sec'
    print(res_df)
    tmp_loc = config.ROOT_DIR / ('results/elrrf/time_tmp.csv')
    res_df.to_csv(tmp_loc)

if __name__ == '__main__':
    main()