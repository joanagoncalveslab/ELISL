import os
import sys
path2this = os.path.dirname(os.path.abspath(__file__)).split('/')
for i, folder in enumerate(path2this):
    if folder.lower()=='elisl':
        project_path = '/'.join(path2this[:i+1])
sys.path.insert(0,project_path)
from src.models.ELRRF import *
from src.comparison.GCATSL.source.GCATSL import *
from src.comparison.GRSMF.GRSMF import *

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

def get_result_loc(loc='results/elrrf/single_cancer_validation.csv', res_names=None, cancer='BRCA', selections={},
                   out_cols=['datasets'], process_text=True, bestMC=False, chosen_th=None):
    print(f'Looking for results of {cancer}')
    if res_names==None:
        res_names = ['seq_1024', 'seq_512', 'seq_256', 'seq_128', 'seq_64', 'seq_32', 'seq_16', 'ppi_ec',
                     'crispr_dependency_mut', 'crispr_dependency_expr', 'crispr_dependency_cnv', 'crispr_dependency_any', 'crispr_dependency_muex', 'crispr_dependency_comb',
                     'd2_dependency_mut', 'd2_dependency_expr', 'd2_dependency_cnv', 'd2_dependency_any', 'd2_dependency_muex', 'd2_dependency_comb',
                     'comb_dependency_mut', 'comb_dependency_expr', 'comb_dependency_cnv', 'comb_dependency_any', 'comb_dependency_muex', 'comb_dependency_comb']
    loc = config.ROOT_DIR / loc
    res_df = pd.read_csv(loc)
    selection = res_df.copy()
    #res_df = res_df.drop_duplicates()
    #selection = res_df[res_df['cancer'] == cancer]
    #selection = selection[selection['balance_strat'] == 'undersample_train']
    selection['grid_search'] = selection['grid_search'].astype(str)
    for s_name, s_vals in selections.items():
        selection = selection[selection[s_name].isin(s_vals)]
    #selection = selection[selection.datasets.isin(res_names)]
    selection = selection.set_index('datasets').loc[res_names]
    if 'cancer_test' in selection:
        selection = selection.sort_values(by=['datasets', 'cancer_test'])
    elif False:
        selection = selection.sort_values(by=['datasets', 'cancer'])

    if bestMC:
        a = pd.DataFrame(index=selection.index.unique(), columns=selection.columns.values)
        for ind_n, row in a.iterrows():
            tmp_best = selection.copy().loc[[ind_n]]
            tmp_best = tmp_best[tmp_best['MC_m']==tmp_best['MC_m'].max()]
            tmp_best = tmp_best[tmp_best['MC_std']==tmp_best['MC_std'].min()]
            a.loc[ind_n] = tmp_best.iloc[0].values
        selection=a.copy()

    if chosen_th is not None:
        a = pd.DataFrame(index=selection.index.unique(), columns=selection.columns.values)
        for ind_n, row in a.iterrows():
            tmp_best = selection.copy().loc[ind_n]
            sdim = ind_n.split('|')[0].split('_')[1]
            tmp_best = tmp_best[tmp_best['threshold']==chosen_th[cancer][sdim]]
            a.loc[ind_n] = tmp_best.iloc[0].values
        selection=a.copy()

    if process_text:
        selection[['AUROC_m', 'AUROC_std', 'AUPRC_m', 'AUPRC_std', 'MC_m', 'MC_std']] = selection[
            ['AUROC_m', 'AUROC_std', 'AUPRC_m', 'AUPRC_std', 'MC_m', 'MC_std']].astype(str)
        selection['AUROC_m'] = selection['AUROC_m'].str[1:]
        selection['AUROC_std'] = selection['AUROC_std'].str[1:]
        selection['AUPRC_m'] = selection['AUPRC_m'].str[1:]
        selection['AUPRC_std'] = selection['AUPRC_std'].str[1:]
        selection['MC_m'] = selection['MC_m'].str[1:]
        selection['MC_std'] = selection['MC_std'].str[1:]

        selection.insert(loc=2, column='AUROC', value=selection[['AUROC_m', 'AUROC_std']].agg(' +- '.join, axis=1))
        selection.insert(loc=3, column='AUPRC', value=selection[['AUPRC_m', 'AUPRC_std']].agg(' +- '.join, axis=1))
        selection.insert(loc=3, column='MC', value=selection[['MC_m', 'MC_std']].agg(' +- '.join, axis=1))
    #ret = selection.set_index('datasets').loc[res_names].reset_index(inplace=False)
    return selection[out_cols]

def add_col2_res(loc,cancer=None, col_name='', col_val=np.nan):
    loc = config.ROOT_DIR / loc
    res_df = pd.read_csv(loc)
    if cancer is not None:
        res_df = res_df[res_df['cancer'] == cancer]
    res_df[col_name] = col_val
    res_df.to_csv(loc, index=False)


def get_res_name(res_id=0):
    if res_id==0:
        res_names = ['crispr_dependency_mut|crispr_dependency_cnv|crispr_dependency_expr', 'crispr_dependency_comb',
                     'd2_dependency_mut|d2_dependency_cnv|d2_dependency_expr', 'd2_dependency_comb',
                     'crispr_dependency_mut|d2_dependency_mut', 'comb_dependency_mut',
                     'crispr_dependency_expr|d2_dependency_expr', 'comb_dependency_expr',
                     'crispr_dependency_comb|d2_dependency_comb',
                     'comb_dependency_mut|comb_dependency_cnv|comb_dependency_expr', 'comb_dependency_comb']
    elif res_id==1:
        res_names = ['crispr_dependency_mut|crispr_dependency_expr', 'crispr_dependency_muex', 'crispr_dependency_comb',
                     'd2_dependency_mut|d2_dependency_expr', 'd2_dependency_muex', 'd2_dependency_comb',
                     'crispr_dependency_mut|d2_dependency_mut', 'comb_dependency_mut',
                     'crispr_dependency_expr|d2_dependency_expr', 'comb_dependency_expr',
                     'crispr_dependency_comb|d2_dependency_comb', 'crispr_dependency_muex|d2_dependency_muex',
                     'comb_dependency_mut|comb_dependency_expr', 'comb_dependency_muex', 'comb_dependency_comb']
    elif res_id==2:
        res_names = ['seq_1024', 'ppi_ec', 'crispr_dependency_mut', 'crispr_dependency_expr',
                     'd2_dependency_mut', 'd2_dependency_expr', 'tissue', ]
    elif res_id==3:
        res_names = ['seq_1024', 'seq_512', 'seq_256', 'seq_128', 'seq_64', 'seq_32', 'seq_16', 'ppi_ec',
                     'tissue', #'tissue_surv', 'tissue_coexp',
                         'crispr_dependency_mut', 'crispr_dependency_expr', 'crispr_dependency_cnv', 'crispr_dependency_any', 'crispr_dependency_muex', 'crispr_dependency_comb',
                         'd2_dependency_mut', 'd2_dependency_expr', 'd2_dependency_cnv', 'd2_dependency_any', 'd2_dependency_muex', 'd2_dependency_comb',
                         'comb_dependency_mut', 'comb_dependency_expr', 'comb_dependency_cnv', 'comb_dependency_any', 'comb_dependency_muex', 'comb_dependency_comb',]
    elif res_id==4:
        res_names = ['crispr_dependency_mut|d2_dependency_mut','crispr_dependency_cnv|d2_dependency_cnv',
                     'crispr_dependency_expr|d2_dependency_expr','crispr_dependency_comb|d2_dependency_comb',
                     'crispr_dependency_mut', 'crispr_dependency_expr',
                     'crispr_dependency_mut|crispr_dependency_expr', 'crispr_dependency_muex',
                     'crispr_dependency_mut|crispr_dependency_cnv|crispr_dependency_expr', 'crispr_dependency_comb',
                     'comb_dependency_mut', 'comb_dependency_expr', 'comb_dependency_cnv',
                     'crispr_dependency_muex|d2_dependency_muex', 'comb_dependency_mut|comb_dependency_expr',
                     'crispr_dependency_mut|crispr_dependency_expr|d2_dependency_mut|d2_dependency_expr',
                     'comb_dependency_muex',
                     'crispr_dependency_comb|d2_dependency_comb', 'comb_dependency_mut|comb_dependency_cnv|comb_dependency_expr',
                     'crispr_dependency_mut|crispr_dependency_cnv|crispr_dependency_expr|d2_dependency_mut|d2_dependency_cnv|d2_dependency_expr',
                     'comb_dependency_comb']
    elif res_id==5:
        res_names = ['tissue_coexp']
    elif res_id==6:
        res_names = ['seq_1024|crispr_dependency_mut|crispr_dependency_expr',
                     'seq_512|crispr_dependency_mut|crispr_dependency_expr',
                     'seq_256|crispr_dependency_mut|crispr_dependency_expr',
                     'seq_1024|crispr_dependency_muex|d2_dependency_muex',
                     'seq_512|crispr_dependency_muex|d2_dependency_muex',
                     'seq_256|crispr_dependency_muex|d2_dependency_muex',
                     'seq_1024|crispr_dependency_mut|crispr_dependency_expr|d2_dependency_mut|d2_dependency_expr',
                     'seq_512|crispr_dependency_mut|crispr_dependency_expr|d2_dependency_mut|d2_dependency_expr',
                     'seq_256|crispr_dependency_mut|crispr_dependency_expr|d2_dependency_mut|d2_dependency_expr',
                     'seq_1024|ppi_ec|crispr_dependency_mut|crispr_dependency_expr',
                     'seq_512|ppi_ec|crispr_dependency_mut|crispr_dependency_expr',
                     'seq_256|ppi_ec|crispr_dependency_mut|crispr_dependency_expr',
                     'seq_1024|ppi_ec|crispr_dependency_muex|d2_dependency_muex',
                     'seq_512|ppi_ec|crispr_dependency_muex|d2_dependency_muex',
                     'seq_256|ppi_ec|crispr_dependency_muex|d2_dependency_muex',
                     'seq_1024|ppi_ec|crispr_dependency_mut|crispr_dependency_expr|d2_dependency_mut|d2_dependency_expr',
                     'seq_512|ppi_ec|crispr_dependency_mut|crispr_dependency_expr|d2_dependency_mut|d2_dependency_expr',
                     'seq_256|ppi_ec|crispr_dependency_mut|crispr_dependency_expr|d2_dependency_mut|d2_dependency_expr',]
    elif res_id==7:
        res_names = ['seq_1024|ppi_ec|crispr_dependency_mut|crispr_dependency_expr',
                     'seq_1024|ppi_ec|crispr_dependency_muex|d2_dependency_muex',]
                     #'seq_1024|ppi_ec|crispr_dependency_mut|crispr_dependency_expr|d2_dependency_mut|d2_dependency_expr']
    elif res_id==8:
        res_names = ['seq_1024|ppi_ec|crispr_dependency_mut|crispr_dependency_expr']

    elif res_id==9:
        res_names = ['onehot']
    elif res_id==10:
        res_names = ['seq_1024|ppi_ec|crispr_dependency_mut|crispr_dependency_expr',
                     'seq_1024|ppi_ec|crispr_dependency_muex|d2_dependency_muex',
                     'seq_512|ppi_ec|crispr_dependency_mut|crispr_dependency_expr',
                     'seq_512|ppi_ec|crispr_dependency_muex|d2_dependency_muex',
                     'seq_256|ppi_ec|crispr_dependency_mut|crispr_dependency_expr',
                     'seq_256|ppi_ec|crispr_dependency_muex|d2_dependency_muex',
                     ]
    elif res_id==11:
        res_names = []
        for i in ['1024', '512', '256', '128', '64', '32']:
            res_names.append('seq_'+i+'|ppi_ec|crispr_dependency_mut|crispr_dependency_expr')
            res_names.append('seq_'+i+'|ppi_ec|crispr_dependency_muex|d2_dependency_muex')
    elif res_id==12:
        res_names = []
        for i in ['1024', '512', '256', '128', '64', '32']:
            res_names.append('seq_'+i+'|ppi_ec|crispr_dependency_mut|crispr_dependency_expr')
            res_names.append('seq_'+i+'|ppi_ec|crispr_dependency_mut|crispr_dependency_expr|tissue')
            res_names.append('seq_'+i+'|ppi_ec|crispr_dependency_muex|d2_dependency_muex')
            res_names.append('seq_'+i+'|ppi_ec|crispr_dependency_muex|d2_dependency_muex|tissue')
    elif res_id==13:
        res_names = []
        for i in ['1024', '512', '256', '128', '64', '32']:
            res_names.append('seq_'+i+'|ppi_ec|crispr_dependency_mut|crispr_dependency_expr|tissue')
            #res_names.append('seq_'+i+'|ppi_ec|crispr_dependency_muex|d2_dependency_muex|tissue')
    elif res_id==14:
        res_names = []
        for i in ['1024']:#, '512', '256', '128', '64', '32']:
            res_names.append('seq_'+i+'|ppi_ec|crispr_dependency_mut|crispr_dependency_expr|tissue')
    elif res_id==15:
        res_names = []
        for i in ['32']:
            res_names.append('seq_'+i+'|ppi_ec|crispr_dependency_muex|d2_dependency_muex|tissue')
    return res_names

def report_cancer_res(res_id=0, loc='results/elrrf/single_cancer_validation.csv', selections=None, isMC=False, chosen_th=None):
    if selections is None:
        selections = {'threshold': [0.5], 'balance_strat': ['undersample_train_test'],
                      'comb_type':['type2']}
    res_names = get_res_name(res_id)
    all_res_list = []
    for cancer in cancer_list:
        selections.update({'cancer':[cancer]})
        if isMC:
            res = get_result_loc(loc=loc, cancer=cancer, res_names=res_names, selections=selections,
                                 out_cols=['AUROC', 'AUPRC', 'MC'], bestMC=isMC, chosen_th=chosen_th)
        else:
            res = get_result_loc(loc=loc, cancer=cancer, res_names=res_names, selections=selections,
                                 out_cols=['AUROC', 'AUPRC'], bestMC=isMC, chosen_th=chosen_th)
        res_cancer = res.copy()
        res_cancer['cancer']=cancer
        all_res_list.append(res_cancer)
        tmp_loc = config.ROOT_DIR / ('results/ELRRF/temp_'+cancer+'.csv')
        res.to_csv(tmp_loc, index=True)
    all_res = pd.concat(all_res_list)
    return all_res



def report_cancer_res_1_score(res_id=0, loc='results/elrrf/loco_validation.csv', selections=None, isMC=False,
                              chosen_th=None, out_cols=['AUROC_m']):
    if selections is None:
        selections = {'threshold': [0.5], 'balance_strat': ['undersample_train_test'],
                      'comb_type':['type2']}
    res_names = get_res_name(res_id)
    for cancer in cancer_list:
        if 'loco' in loc:
            selections.update({'cancer':[cancer]})
        if 'cross' in loc:
            selections.update({'cancer_test':[cancer]})
        if isMC:
            res = get_result_loc(loc=loc, cancer=cancer, res_names=res_names, selections=selections,
                                 out_cols=out_cols, bestMC=isMC, chosen_th=chosen_th)
        else:
            res = get_result_loc(loc=loc, cancer=cancer, res_names=res_names, selections=selections,
                                 out_cols=out_cols, bestMC=isMC, chosen_th=chosen_th)
        tmp_loc = config.ROOT_DIR / ('results/elrrf/temp_'+cancer+'.csv')
        res.to_csv(tmp_loc, index=None)

def report_mult_cancer_res(res_id=0, locs=['results/elrrf/single_cancer_validation.csv'], selections=None, isMC=False):
    if selections is None:
        selections = {'threshold': [0.5], 'balance_strat': ['undersample_train_test'],
                      'comb_type':['type2']}
    res_names_all = get_res_name(res_id)
    for cancer in cancer_list:
        data_list = []
        selections.update({'cancer_test':[cancer]})
        for res_names in res_names_all:
            for loc in locs:
                if isMC:
                    res = get_result_loc(loc=loc, cancer=cancer, res_names=res_names, selections=selections,
                                         out_cols=['AUROC', 'AUPRC', 'MC', 'threshold'], bestMC=True)
                else:
                    res = get_result_loc(loc=loc, cancer=cancer, res_names=res_names, selections=selections,
                                         out_cols=['AUROC', 'AUPRC', 'MC', 'threshold'], bestMC=False)
                data_list.append(res)
        final_res=pd.concat(data_list)
        tmp_loc = config.ROOT_DIR / ('results/elrrf/temp_'+cancer+'.csv')
        final_res.to_csv(tmp_loc)


def report_cross_cancer_res(res_id=0, loc='results/elrrf/single_cancer_validation.csv', selections=None, score='AUROC'):
    if selections is None:
        selections = {'threshold': [0.5], 'balance_strat': ['undersample_train_test'],
                      'comb_type':['type2']}
    res_names = get_res_name(res_id)
    dim= len(cancer_list)
    cross_table = pd.DataFrame(index=list(reversed(cancer_list)), columns=cancer_list)
    for cancer in reversed(cancer_list):
        selections.update({'cancer_train':[cancer]})
        res = get_result_loc(loc=loc, cancer=cancer, res_names=res_names, selections=selections, out_cols=['cancer_test', 'AUROC', 'AUPRC','threshold'])
        tmp_loc = config.ROOT_DIR / ('results/elrrf/temp_'+cancer+'.csv')
        res.to_csv(tmp_loc)
        res[score+'_mean'] = '0'+res[score].str.split(' ', n=1, expand=True)[0].values
        tmp_series = res.set_index('cancer_test')[score+'_mean']
        cross_table.loc[cancer] = tmp_series.loc[cross_table.loc[cancer].index.drop_duplicates()]
        print(f'{cancer} Done')
    if 'elrrf' in loc:
        tmp_loc = config.ROOT_DIR / ('results/elrrf/temp.csv')
    elif 'ELGBDT' in loc:
        tmp_loc = config.ROOT_DIR / ('results/ELGBDT/temp.csv')
    cross_table.to_csv(tmp_loc)
    return cross_table


def get_best_res(cancer, res_id=0, loc='results/elrrf/single_cancer_validation.csv', selections=None):
    if selections is None:
        selections = {'threshold': [0.5], 'balance_strat': ['undersample_train_test'],
                      'comb_type':['type2']}
    res_names = get_res_name(res_id)
    out_cols = ['AUROC_m', 'AUROC_std', 'AUPRC_m', 'AUPRC_std', 'MC_m', 'MC_std', 'threshold']
    res = get_result_loc(loc=loc, cancer=cancer, res_names=res_names, selections=selections,
                         out_cols=out_cols, process_text=False)
    #res_bests = res[res['MC_m']==res['MC_m'].max()]
    #res_bests = res_bests[res_bests['MC_std']==res_bests['MC_std'].min()]
    res_bests = res[res['AUPRC_m']==res['AUPRC_m'].max()]
    res_bests = res_bests[res_bests['AUPRC_std']==res_bests['AUPRC_std'].min()]
    res_bests = res_bests[res_bests['MC_m']==res_bests['MC_m'].max()]
    res_bests = res_bests[res_bests['MC_std']==res_bests['MC_std'].min()]
    print(res_bests)


def analyze_single_model_res(cancer='BRCA'):
    elrrf_loc = 'results/ELRRF/models_test/seq_1024|ppi_ec|crispr_dependency_mut|crispr_dependency_expr|tissue_'+cancer+'_True_True_type2_10_undersample_train_test.pickle'
    elgbdt_loc = 'results/ELGBDT/models_test/seq_1024|ppi_ec|crispr_dependency_mut|crispr_dependency_expr|tissue_'+cancer+'_True_True_type2_10_undersample_train_test.pickle'
    gcatsl_loc = 'results/GCATSL/models_test/PPI|CC|BP_'+cancer+'_True_10_undersample_train_test_128.pickle'
    grsmf_loc = 'results/GRSMF/models_test/BP_'+cancer+'_True_10_undersample_train_test.pickle'
    r_res_loc = '~/r_projects/msc-thesis-project/r/experiments/1.Performance/1.2 Per Cancer/artifacts/'+cancer+'_single_final_yasin_10.Rdata'
    r_base_res_loc = '~/r_projects/msc-thesis-project/r/experiments/1.Performance/1.2 Per Cancer/artifacts/'+cancer+'_baselines_final_yasin_10.Rdata'
    elrrf_loc = config.ROOT_DIR / elrrf_loc
    elgbdt_loc = config.ROOT_DIR / elgbdt_loc
    gcatsl_loc = config.ROOT_DIR / gcatsl_loc
    grsmf_loc = config.ROOT_DIR / grsmf_loc
    res_elrrf = dfnc.load_pickle(elrrf_loc)
    res_elgbdt = dfnc.load_pickle(elgbdt_loc)
    try:
        res_gcatsl = dfnc.load_pickle(gcatsl_loc)
    except:
        gcatsl_loc = 'results/GCATSL/models_test/PPI|CC|BP_'+cancer+'_True_10_undersample_train_test_128.pickle'
        gcatsl_loc = config.ROOT_DIR / gcatsl_loc
        res_gcatsl = dfnc.load_pickle(gcatsl_loc)

    res_grsmf = dfnc.load_pickle(grsmf_loc)
    res_colm = {}
    res_colm_rdata = dfnc.load_rdata_file(r_res_loc, isRDS=True, is_abs_loc=True)
    res_colm_dict = dict(zip(res_colm_rdata.names, list(res_colm_rdata)))
    labels_colm = {}
    for idx, labels in enumerate(res_colm_dict['labels']):
        labels_colm[idx] = np.array(labels).astype(int)
    res_colm_pred_dict = dict(zip(res_colm_dict['preds'].names, list(res_colm_dict['preds'])))
    preds_colm = {}
    for method, info in res_colm_pred_dict.items():
        preds_colm[method] = coll.OrderedDict()
        for idx, preds in enumerate(info):
            preds_colm[method][idx] = {'predictions': preds, 'labels':labels_colm[idx]}
    try:
        res_pca_gcmf_rdata = dfnc.load_rdata_file(r_base_res_loc, isRDS=True, is_abs_loc=True)
        res_pca_gcmf_dict = dict(zip(res_pca_gcmf_rdata.names, list(res_pca_gcmf_rdata)))
        labels_pca_gcmf = {}
        for idx, labels in enumerate(res_pca_gcmf_dict['labels']):
            labels_pca_gcmf[idx] = np.array(labels).astype(int)
        res_pca_gcmf_pred_dict = dict(zip(res_pca_gcmf_dict['preds'].names, list(res_pca_gcmf_dict['preds'])))
        preds_pca_gcmf = {}
        for method, info in res_pca_gcmf_pred_dict.items():
            preds_pca_gcmf[method] = coll.OrderedDict()
            for idx, preds in enumerate(info):
                preds_pca_gcmf[method][idx] = {'predictions': preds, 'labels':labels_pca_gcmf[idx]}
    except:
        pass

    model_elrrf= ELRRF()
    model_elgbdt= ELRRF()
    model_gcatsl= GCATSL()
    model_grsmf= GRSMF()
    all_res = {'GCATSL':{},  'pca-gCMF':{}, 'GRSMF':{},'Seale_EN':{}, 'Seale_L0L2':{}, 'Seale_RRF':{}, 'Seale_MUVR':{},
               'ELRRF':{}, 'ELGBDT':{}, }
    colm2name= {'Elastic Net': 'Seale_EN', 'L0L2': 'Seale_L0L2',  'MUVR': 'Seale_MUVR',  'Random Forest': 'Seale_RRF',
                'pca-gCMF':'pca-gCMF'}

    all_res['ELRRF']['AUROC'], all_res['ELRRF']['AUPRC'], all_res['ELRRF']['MCC'] = model_elrrf.evaluate_folds(res_elrrf)
    all_res['ELGBDT']['AUROC'], all_res['ELGBDT']['AUPRC'], all_res['ELGBDT']['MCC'] = model_elgbdt.evaluate_folds(res_elgbdt)
    all_res['GCATSL']['AUROC'], all_res['GCATSL']['AUPRC'], all_res['GCATSL']['MCC'] = model_gcatsl.evaluate_folds(res_gcatsl)
    all_res['GRSMF']['AUROC'], all_res['GRSMF']['AUPRC'], all_res['GRSMF']['MCC'] = model_grsmf.evaluate_folds(res_grsmf)
    for colm_name, py_name in colm2name.items():
        try:
            all_res[py_name]['AUROC'],all_res[py_name]['AUPRC'],all_res[py_name]['MCC'] = model_grsmf.evaluate_folds(preds_colm[colm_name])
        except:
            if colm_name=='pca-gCMF':
                all_res[py_name]['AUROC'],all_res[py_name]['AUPRC'],all_res[py_name]['MCC'] = model_grsmf.evaluate_folds(preds_pca_gcmf[colm_name])
    return all_res


def analyze_cd_model_res(cancer='BRCA', train_ds='isle', test_ds='dsl'):
    elrrf_loc = 'results/ELRRF/models_cross_ds/seq_1024|ppi_ec|crispr_dependency_mut|crispr_dependency_expr|tissue_'+cancer+'_'+train_ds+'_'+test_ds+'_True_True_type2_10_undersample_train_test.pickle'
    elgbdt_loc = 'results/ELGBDT/models_cross_ds/seq_1024|ppi_ec|crispr_dependency_mut|crispr_dependency_expr|tissue_'+cancer+'_'+train_ds+'_'+test_ds+'_True_True_type2_10_undersample_train_test.pickle'
    gcatsl_loc = 'results/GCATSL/models_cross_ds/PPI|CC|BP_'+cancer+'_'+train_ds+'_'+test_ds+'_True_10_undersample_train_test_128.pickle'
    grsmf_loc = 'results/GRSMF/models_cross_ds/BP_'+cancer+'_'+train_ds+'_'+test_ds+'_True_10_undersample_train_test.pickle'
    r_res_loc = '~/r_projects/msc-thesis-project/r/experiments/1.Performance/1.3 Dataset Cross Comparison/artifacts/'+cancer+'_'+train_ds+'_'+test_ds+'_final_yasin_10.Rdata'
    elrrf_loc = config.ROOT_DIR / elrrf_loc
    elgbdt_loc = config.ROOT_DIR / elgbdt_loc
    gcatsl_loc = config.ROOT_DIR / gcatsl_loc
    grsmf_loc = config.ROOT_DIR / grsmf_loc
    res_elrrf = dfnc.load_pickle(elrrf_loc)
    res_elgbdt = dfnc.load_pickle(elgbdt_loc)
    try:
        res_gcatsl = dfnc.load_pickle(gcatsl_loc)
    except:
        gcatsl_loc = 'results/GCATSL/models_dho2_test/PPI|CC|BP_'+cancer+'_True_10_undersample_train_test_128.pickle'
        gcatsl_loc = config.ROOT_DIR / gcatsl_loc
        res_gcatsl = dfnc.load_pickle(gcatsl_loc)

    res_grsmf = dfnc.load_pickle(grsmf_loc)
    res_colm = {}
    res_colm_rdata = dfnc.load_rdata_file(r_res_loc, isRDS=True, is_abs_loc=True)
    res_colm_dict = dict(zip(res_colm_rdata.names, list(res_colm_rdata)))
    labels_colm = {}
    for idx, labels in enumerate(res_colm_dict['labels']):
        labels_colm[idx] = np.array(labels).astype(int)
    res_colm_pred_dict = dict(zip(res_colm_dict['preds'].names, list(res_colm_dict['preds'])))
    preds_colm = {}
    for method, info in res_colm_pred_dict.items():
        preds_colm[method] = coll.OrderedDict()
        for idx, preds in enumerate(info):
            preds_colm[method][idx] = {'predictions': preds, 'labels':labels_colm[idx]}
        print()
    model_elrrf= ELRRF()
    model_elgbdt= ELRRF()
    model_gcatsl= GCATSL()
    model_grsmf= GRSMF()
    all_res = {'GCATSL':{},  'pca-gCMF':{}, 'GRSMF':{},'Seale_EN':{}, 'Seale_L0L2':{}, 'Seale_RRF':{}, 'Seale_MUVR':{},
               'ELRRF':{}, 'ELGBDT':{}, }
    colm2name= {'Elastic Net': 'Seale_EN', 'L0L2': 'Seale_L0L2',  'MUVR': 'Seale_MUVR',  'Random Forest': 'Seale_RRF',
                'pca-gCMF':'pca-gCMF'}

    all_res['ELRRF']['AUROC'], all_res['ELRRF']['AUPRC'], all_res['ELRRF']['MCC'] = model_elrrf.evaluate_folds(res_elrrf)
    all_res['ELGBDT']['AUROC'], all_res['ELGBDT']['AUPRC'], all_res['ELGBDT']['MCC'] = model_elgbdt.evaluate_folds(res_elgbdt)
    all_res['GCATSL']['AUROC'], all_res['GCATSL']['AUPRC'], all_res['GCATSL']['MCC'] = model_gcatsl.evaluate_folds(res_gcatsl)
    all_res['GRSMF']['AUROC'], all_res['GRSMF']['AUPRC'], all_res['GRSMF']['MCC'] = model_grsmf.evaluate_folds(res_grsmf)
    for colm_name, py_name in colm2name.items():
        all_res[py_name]['AUROC'],all_res[py_name]['AUPRC'],all_res[py_name]['MCC'] = model_grsmf.evaluate_folds(preds_colm[colm_name])

    return all_res


def analyze_dho_model_res(cancer='BRCA'):
    elrrf_loc = 'results/ELRRF/models_dho2_test/seq_1024|ppi_ec|crispr_dependency_mut|crispr_dependency_expr|tissue_'+cancer+'_True_True_type2_10_undersample_train_test.pickle'
    elgbdt_loc = 'results/ELGBDT/models_dho2_test/seq_1024|ppi_ec|crispr_dependency_mut|crispr_dependency_expr|tissue_'+cancer+'_True_True_type2_10_undersample_train_test.pickle'
    gcatsl_loc = 'results/GCATSL/models_dho2_test/PPI|CC|BP_'+cancer+'_True_10_undersample_train_test_128.pickle'
    grsmf_loc = 'results/GRSMF/models_dho2_test/BP_'+cancer+'_True_10_undersample_train_test.pickle'
    r_res_loc = '~/r_projects/msc-thesis-project/r/experiments/1.Performance/1.9 Gene Dropout/artifacts/'+cancer+'_dho_final_yasin_10.Rdata'
    elrrf_loc = config.ROOT_DIR / elrrf_loc
    elgbdt_loc = config.ROOT_DIR / elgbdt_loc
    gcatsl_loc = config.ROOT_DIR / gcatsl_loc
    grsmf_loc = config.ROOT_DIR / grsmf_loc
    res_elrrf = dfnc.load_pickle(elrrf_loc)
    res_elgbdt = dfnc.load_pickle(elgbdt_loc)
    try:
        res_gcatsl = dfnc.load_pickle(gcatsl_loc)
    except:
        gcatsl_loc = 'results/GCATSL/models_dho2_test/PPI|CC|BP_'+cancer+'_True_10_undersample_train_test_128.pickle'
        gcatsl_loc = config.ROOT_DIR / gcatsl_loc
        res_gcatsl = dfnc.load_pickle(gcatsl_loc)

    res_grsmf = dfnc.load_pickle(grsmf_loc)
    res_colm = {}
    res_colm_rdata = dfnc.load_rdata_file(r_res_loc, isRDS=True, is_abs_loc=True)
    res_colm_dict = dict(zip(res_colm_rdata.names, list(res_colm_rdata)))
    labels_colm = {}
    for idx, labels in enumerate(res_colm_dict['labels']):
        labels_colm[idx] = np.array(labels).astype(int)
    res_colm_pred_dict = dict(zip(res_colm_dict['preds'].names, list(res_colm_dict['preds'])))
    preds_colm = {}
    for method, info in res_colm_pred_dict.items():
        preds_colm[method] = coll.OrderedDict()
        for idx, preds in enumerate(info):
            preds_colm[method][idx] = {'predictions': preds, 'labels':labels_colm[idx]}
        print()
    model_elrrf= ELRRF()
    model_elgbdt= ELRRF()
    model_gcatsl= GCATSL()
    model_grsmf= GRSMF()
    all_res = {'GCATSL':{},  'pca-gCMF':{}, 'GRSMF':{},'Seale_EN':{}, 'Seale_L0L2':{}, 'Seale_RRF':{}, 'Seale_MUVR':{},
               'ELRRF':{}, 'ELGBDT':{}, }
    colm2name= {'Elastic Net': 'Seale_EN', 'L0L2': 'Seale_L0L2',  'MUVR': 'Seale_MUVR',  'Random Forest': 'Seale_RRF',
                'pca-gCMF':'pca-gCMF'}

    all_res['ELRRF']['AUROC'], all_res['ELRRF']['AUPRC'], all_res['ELRRF']['MCC'] = model_elrrf.evaluate_folds(res_elrrf)
    all_res['ELGBDT']['AUROC'], all_res['ELGBDT']['AUPRC'], all_res['ELGBDT']['MCC'] = model_elgbdt.evaluate_folds(res_elgbdt)
    all_res['GCATSL']['AUROC'], all_res['GCATSL']['AUPRC'], all_res['GCATSL']['MCC'] = model_gcatsl.evaluate_folds(res_gcatsl)
    all_res['GRSMF']['AUROC'], all_res['GRSMF']['AUPRC'], all_res['GRSMF']['MCC'] = model_grsmf.evaluate_folds(res_grsmf)
    for colm_name, py_name in colm2name.items():
        all_res[py_name]['AUROC'],all_res[py_name]['AUPRC'],all_res[py_name]['MCC'] = model_grsmf.evaluate_folds(preds_colm[colm_name])

    return all_res


def analyze_model(loc, task='avg_samples'):
    n_split = int(loc.split('|')[-1].split('_')[7])
    loc = config.ROOT_DIR / loc
    models = dfnc.load_pickle(loc)
    tr_pairs = 0
    te_pairs = 0
    tr_gene_size= 0
    te_gene_size = 0
    for i in range(n_split):
        first_key = list(models[i].keys())[0]
        tr_samples = models[i][first_key]['train_names']
        te_samples = models[i][first_key]['test_names']
        df_tr = pd.DataFrame(tr_samples, columns=['genes'])
        df_te = pd.DataFrame(te_samples, columns=['genes'])
        df_tr = df_tr['genes'].str.split('|', expand=True)
        df_te = df_te['genes'].str.split('|', expand=True)
        tr_genes = np.union1d(df_tr[0], df_tr[1])
        te_genes = np.union1d(df_te[0], df_te[1])
        tr_gene_size = tr_gene_size + len(tr_genes)
        te_gene_size = te_gene_size + len(te_genes)
        tr_pairs = tr_pairs+df_tr.shape[0]
        te_pairs = te_pairs+df_te.shape[0]
    avg_tr_pairs = tr_pairs/n_split
    avg_te_pairs = te_pairs/n_split
    print(avg_tr_pairs)
    print(avg_te_pairs)

    avg_tr_genes = tr_gene_size/n_split
    avg_te_genes = te_gene_size/n_split
    print(avg_tr_genes)
    print(avg_te_genes)


def find_top_n_genes(loc='results/ELRRF/models_test/seq_1024|ppi_ec|crispr_dependency_mut|crispr_dependency_expr|tissue_BRCA_True_True_type2_10_undersample_train_test.pickle', n=10):
    loc = config.ROOT_DIR / loc
    save_loc = '.'.join(str(loc).split('.')[:-1]) + '.csv'
    if os.path.exists(save_loc):
        return pd.read_csv(save_loc, index_col=0)
    models = dfnc.load_pickle(loc)
    tr_pairs = 0
    te_pairs = 0
    tr_gene_size= 0
    te_gene_size = 0
    rows= []
    all_labels = {}
    for i in range(10):
        if i in models.keys():
            i_s = i
        else:
            i_s = str(i)

        label_dict = coll.OrderedDict()
        pred_df = pd.DataFrame(columns=['pair_name', 'model', 'prediction', 'probability'])
        for model_name, model_res in models[i_s].items():
            if 'time' in model_name:
                continue
            model_pred_df = pd.DataFrame(columns=['pair_name', 'model', 'prediction', 'probability'])
            model_pred_df['pair_name'] = model_res['test_names']
            model_pred_df['prediction'] = model_res['predictions']
            model_pred_df['probability'] = model_res['probabilities']
            model_pred_df['tr_auc'] = model_res['tr_auc']
            model_pred_df['model'] = model_name
            pred_df = pred_df.append(model_pred_df, ignore_index=True)
            model_labels = dict(zip(model_res['test_names'], model_res['labels']))
            label_dict.update(model_labels)
        #grouped_preds = pred_df.groupby(['pair_name'])['probability'].mean().reset_index()
        grouped_preds = pred_df.groupby(['pair_name']).apply(combine_preds).reset_index()
        rows.append(grouped_preds[['pair_name', 'probability']].values)
        all_labels.update(label_dict)

    all_scores = pd.DataFrame(np.concatenate(rows, axis=0), columns=['pair_name', 'score'])
    all_scores['score'] = all_scores['score'].astype(float)
    grouped_scores = all_scores.groupby('pair_name')['score'].agg(['count', 'mean'])
    sorted_res = grouped_scores.sort_values(['mean'], ascending=False)
    sorted_res['labels'] = sorted_res.index.map(all_labels)

    sorted_res.to_csv(save_loc)
    print()
    return sorted_res


def analyze_save_colm_res(task='single', cancer='BRCA'):
    columns = ['method', 'fold_no', 'auroc', 'auprc', 'ap_3']
    r_project_folder = '~/r_projects/msc-thesis-project/r/experiments/1.Performance/'
    if task=='single':
        folder='models_test'
        r_res_loc = r_project_folder+'1.2 Per Cancer/artifacts/'+cancer+'_single_forcolm.Rdata'
        r_base_loc = r_project_folder+'1.2 Per Cancer/artifacts/'+cancer+'_baselines_forcolm.Rdata'
    elif task=='dho':
        folder='models_dho2_test'
        r_res_loc = r_project_folder+'1.9 Gene Dropout/artifacts/'+cancer+'_predictions_forcolm_dho.Rdata'
        r_base_loc = r_project_folder+'1.9 Gene Dropout/artifacts/'+cancer+'_predictions_forcolm_dho.Rdata'
    elif task=='sho':
        folder='models_sho_test'
        r_res_loc = r_project_folder+'1.9 Gene Dropout/artifacts/'+cancer+'_predictions_forcolm_sho.Rdata'
        r_base_loc = r_project_folder+'1.9 Gene Dropout/artifacts/'+cancer+'_predictions_forcolm_sho.Rdata'
    elif task=='cross_ds':
        folder='models_cross_ds'
        r_res_loc = r_project_folder+'1.3 Dataset Cross Comparison/artifacts/'+cancer+'_forcolm_data.Rdata'
        r_base_loc = r_project_folder+'1.3 Dataset Cross Comparison/artifacts/'+cancer+'_forcolm_data.Rdata'

    gcatsl_loc = config.ROOT_DIR / 'results_colm' / 'GCATSL' / folder / ('PPI|CC|BP_'+cancer+'_True_undersample_train_test_128.pickle')
    grsmf_loc = config.ROOT_DIR / 'results_colm' / 'GRSMF' / folder / ('BP_'+cancer+'_True_undersample_train_test.pickle')

    res_gcatsl = dfnc.load_pickle(gcatsl_loc)
    res_grsmf = dfnc.load_pickle(grsmf_loc)
    res_colm = {}
    res_colm_rdata = dfnc.load_rdata_file(r_res_loc, isRDS=True, is_abs_loc=True)
    res_colm_dict = dict(zip(res_colm_rdata.names, list(res_colm_rdata)))
    labels_colm = {}
    for idx, labels in enumerate(res_colm_dict['labels']):
        labels_colm[idx] = np.array(labels).astype(int)
    res_colm_pred_dict = dict(zip(res_colm_dict['preds'].names, list(res_colm_dict['preds'])))
    preds_colm = {}
    for method, info in res_colm_pred_dict.items():
        preds_colm[method] = coll.OrderedDict()
        for idx, preds in enumerate(info):
            preds_colm[method][idx] = {'predictions': preds, 'labels':labels_colm[idx]}
        print()

    try:
        res_base_rdata = dfnc.load_rdata_file(r_base_loc, isRDS=True, is_abs_loc=True)
        res_base_dict = dict(zip(res_base_rdata.names, list(res_base_rdata)))
        labels_base = {}
        for idx, labels in enumerate(res_base_dict['labels']):
            labels_base[idx] = np.array(labels).astype(int)
        res_base_pred_dict = dict(zip(res_base_dict['preds'].names, list(res_base_dict['preds'])))
        preds_base = {}
        for method, info in res_base_pred_dict.items():
            preds_base[method] = coll.OrderedDict()
            for idx, preds in enumerate(info):
                preds_base[method][idx] = {'predictions': preds, 'labels':labels_base[idx]}
    except:
        pass


    model_gcatsl= GCATSL()
    model_grsmf= GRSMF()
    if task=='single':
        all_res = {'GCATSL':{},  'pca-gCMF':{}, 'GRSMF':{},'Elastic Net':{}, 'L0L2':{}, 'Random Forest':{}, 'MUVR':{},
                   'DAISY':{}, 'DiscoverSL':{}, }
        colm2name= {'Elastic Net': 'Elastic Net', 'L0L2': 'L0L2',  'MUVR': 'MUVR',  'Random Forest': 'Random Forest',
                    'pca-gCMF':'pca-gCMF', 'DAISY':'DAISY', 'DiscoverSL':'DiscoverSL'}
    else:
        all_res = {'GCATSL':{},  'pca-gCMF':{}, 'GRSMF':{},'Elastic Net':{}, 'L0L2':{}, 'Random Forest':{}, 'MUVR':{} }
        colm2name= {'Elastic Net': 'Elastic Net', 'L0L2': 'L0L2',  'MUVR': 'MUVR',  'Random Forest': 'Random Forest',
                    'pca-gCMF':'pca-gCMF'}


    all_res['GCATSL']['AUROC'], all_res['GCATSL']['AUPRC'], all_res['GCATSL']['MCC'], all_res['GCATSL']['AP3'] = model_gcatsl.evaluate_folds(res_gcatsl, return_ap3=True)
    all_res['GRSMF']['AUROC'], all_res['GRSMF']['AUPRC'], all_res['GRSMF']['MCC'], all_res['GRSMF']['AP3'] = model_grsmf.evaluate_folds(res_grsmf, return_ap3=True)
    for colm_name, py_name in colm2name.items():
        try:
            all_res[py_name]['AUROC'],all_res[py_name]['AUPRC'],all_res[py_name]['MCC'], all_res[py_name]['AP3'] = model_grsmf.evaluate_folds(preds_colm[colm_name], return_ap3=True)
        except:
            if colm_name in ['pca-gCMF', 'DAISY', 'DiscoverSL']:
                all_res[py_name]['AUROC'],all_res[py_name]['AUPRC'],all_res[py_name]['MCC'], all_res[py_name]['AP3'] = model_grsmf.evaluate_folds(preds_base[colm_name], return_ap3=True)

    all_rows = np.zeros(shape=(0,5))
    for name, res in all_res.items():
        names = np.array([name]*len(res['AUROC']))
        folds = np.arange(1,len(res['AUROC'])+1)
        rows = np.stack([names, folds, res['AUROC'], res['AUPRC'], res['AP3']], axis=1)
        all_rows = np.concatenate([all_rows, rows])

    df_res = pd.DataFrame(all_rows, columns=columns)
    res_loc = config.ROOT_DIR / 'results_colm' / (cancer+'_'+task+'.csv')
    df_res.to_csv(res_loc, index=None)
    print()


def main():
    loc = 'results/ELRRF/single_cancer_validation.csv'
    if False:
        report_cancer_res(res_id=2, loc=loc, selections={'balance_strat': ['undersample_train_test'],
                                                         'comb_type': ['type2'],  'process':[True],'n_split':[10],
                                                         'grid_search':['False']}, isMC=True)#,chosen_th=chosen_th_train)
    #loc = 'results/ELRRF/loco_gbdt_validation.csv'
    loc = 'results/ELRRF/cross_multi_cancer_gbdt_validation.csv'
    if False:
        report_cancer_res_1_score(res_id=13, loc=loc, selections={'balance_strat': ['undersample_train_test'],
                                                         'comb_type': ['type2'],  'process':[True],'n_split':[10],
                                                         'grid_search':['False']}, isMC=False, out_cols=['AUPRC_m'])#,chosen_th=chosen_th_train)

    locs = ['results/ELRRF/single_cancer_validation.csv',
            'results/ELRRF/single_cancer_validation_early.csv',
            'results/ELRRF/single_cancer_validation_late.csv']
    if False:
        report_mult_cancer_res(res_id=13, locs=locs, selections={'balance_strat': ['undersample_train_test'],
                                                         'comb_type': ['type2'],  'process':[True],'n_split':[10],
                                                         'grid_search':['False']}, isMC=True, score='AUPRC')

    #loc = 'results/ELRRF/cross_cancer_test_all_train.csv'
    loc = 'results/ELGBDT/cross_cancer_train_train_test_test.csv'
    #loc = 'results/ELRRF/pretrain_validation.csv'
    if False:
        report_cross_cancer_res(res_id=14, loc=loc, selections={'balance_strat': ['undersample_train_test'],
                                                         'comb_type': ['type2'], 'threshold': [0.5], 'process':[True], 'n_split':[10],
                                                         'grid_search':['True']}, score='AUROC')
    #add_col2_res(loc, col_name='n_split', col_val=str(int(5)))
    for cancer in cancer_list:
        if False:
            print(f'For cancer {cancer}')
            get_best_res(cancer=cancer, res_id=11, loc=loc, selections={'balance_strat': ['undersample_train_test'],'n_split':[10],
                                                                    'comb_type': ['type2'], 'process':[True]})
    loc = 'results/ELRRF/models_dho2_test/seq_1024|ppi_ec|crispr_dependency_mut|crispr_dependency_expr|tissue_BRCA_True_True_type2_5_undersample_train_test.pickle'
    loc = 'results/GCATSL/models_dho2_test/PPI|CC|BP_BRCA_False_undersample_train_test_128.pickle'
    #analyze_model(loc)
    #all_res = analyze_model_res()
    #find_top_n_genes()

    analyze_save_colm_res(task='cross_ds', cancer='BRCA_isle_discoversl')
if __name__ == '__main__':
    main()