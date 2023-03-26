#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Survival Analysis
"""
import json
from src import config

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import multivariate_logrank_test
from functools import reduce


def get_gene_set(gene1, gene2):
    gene1_set = [gene1]
    gene2_set = [gene2]
    if gene1 =='BRCA':
        gene1_set=['BRCA1', 'BRCA2']
    if gene1 =='MAP2KFAM':
        gene1_set=['MAP2K1', 'MAP2K2', 'MAP2K3', 'MAP2K4', 'MAP2K5', 'MAP2K6', 'MAP2K7']
    if gene1 =='MAP3KFAM':
        gene1_set=['MAP3K1', 'MAP3K2', 'MAP3K3', 'MAP3K4', 'MAP3K5', 'MAP3K6', 'MAP3K7', 'MAP3K8', 'MAP3K9', 'MAP3K10',
                   'MAP3K11', 'MAP3K12', 'MAP3K13', 'MAP3K14', 'MAP3K15', 'TAOK1', 'TAOK2', 'RAF1', 'BRAF', 'ARAF', 'MAP3K20']
    if gene1 =='DAPKFAM':
        gene1_set=['DAPK1', 'DAPK2', 'DAPK3', 'STK17A', 'STK17B']
    if gene1 =='RIPKFAM':
        gene1_set=['RIPK1', 'RIPK2', 'RIPK3', 'RIPK4', 'DSTYK']
    if gene2 =='WNT':
        gene2_set=['WNT3A', 'WNT7A', 'WNT6', 'WNT16']
    elif gene2 =='HHIP':
        gene2_set=['IHH', 'SHH', 'DHH', 'PTCH1']
    elif gene2 =='FGF':
        gene2_set=['FGF6', 'FGF8']
    elif gene2 =='FGFFAM':
        gene2_set=['FGF1', 'FGF2', 'FGF3', 'FGF4', 'FGF5', 'FGF6', 'FGF7', 'FGF8', 'FGF9', 'FGF10', 'FGF11', 'FGF12', 'FGF13',
                   'FGF14', 'FGF16', 'FGF17', 'FGF18', 'FGF19', 'FGF20', 'FGF21', 'FGF22', 'FGF23']
    elif gene2 =='MRPL':
        gene2_set=['MRPL1', 'MRPL2', 'MRPL3', 'MRPL4', 'MRPL9', 'MRPL10', 'MRPL11', 'MRPL12', 'MRPL13', 'MRPL14',
                   'MRPL15', 'MRPL16', 'MRPL17', 'MRPL18', 'MRPL19', 'MRPL20', 'MRPL21', 'MRPL22', 'MRPL23', 'MRPL24',
                   'MRPL27', 'MRPL28', 'MRPL30', 'MRPL32', 'MRPL33', 'MRPL34', 'MRPL35', 'MRPL36', 'MRPL37', 'MRPL38',
                   'MRPL39', 'MRPL40', 'MRPL41', 'MRPL42', 'MRPL43', 'MRPL44', 'MRPL45', 'MRPL46', 'MRPL47', 'MRPL48',
                   'MRPL49', 'MRPL50', 'MRPL51', 'MRPL52', 'MRPL53', 'MRPL54', 'MRPL55', 'MRPL57', 'MRPL58']
    elif gene2 =='POLX':
        gene2_set=['POLL', 'POLB', 'POLM', 'TENT4A', 'DNTT']
    elif gene2 =='NR1':
        gene2_set=['THRA', 'THRB', 'RARA', 'RARB', 'RARG', 'PPARA', 'PPARD', 'PPARG', 'NR1D1', 'NR1D2', 'RORA', 'RORB', 'RORC', 'NR1H4', 'NR1H5P', 'NR1H3', 'NR1H2', 'VDR', 'NR1I2', 'NR1I3']
    elif gene2 =='IL6FAM':
        gene2_set=['IL6', 'IL11', 'IL27', 'IL31', 'CNTF', 'LIF', 'OSM']
    elif gene2 =='PARP12':
        gene2_set=['PARP1', 'PARP2']
    elif gene2 =='PARPFAM':
        gene2_set=['PARP1', 'PARP2', 'PARP3', 'PARP4', 'TNKS', 'TNKS2', 'PARP6', 'TIPARP', 'PARP8', 'PARP9', 'PARP10', 'PARP11', 'PARP12', 'ZC3HAV1', 'PARP14', 'PARP15', 'PARP16']

    return gene1_set, gene2_set


def get_clinical(study_name='pancancer_study_clinical_data.tsv'):
    clinical_loc = config.DATA_DIR / 'pair_analysis' / study_name
    if study_name.split('_')[0] == 'Pancancer':
        clinical_df = pd.read_csv(clinical_loc, sep='\t')[['Study ID', 'Sample ID', 'Diagnosis Age', 'Sex',
                                                           'TCGA PanCanAtlas Cancer Type Acronym',
                                                           'Overall Survival (Months)', 'Overall Survival Status', ]]
        clinical_df['Overall Survival Status'] = clinical_df['Overall Survival Status'].str.split(':', expand=True).values[:,0]
        clinical_df['Cancer Type'] = clinical_df['TCGA PanCanAtlas Cancer Type Acronym']
        clinical_df = clinical_df.drop(columns=['TCGA PanCanAtlas Cancer Type Acronym'])
    elif study_name.split('_')[0] == 'Breast':
        clinical_df = pd.read_csv(clinical_loc, sep='\t')[
            ['Study ID', 'Sample ID', 'Diagnosis Age', "Age at Diagnosis", 'Sex', 'Cancer Type',
             'Overall Survival (Months)', 'Overall Survival Status', "Patient's Vital Status"]]
        clinical_df['Overall Survival Status'] = clinical_df['Overall Survival Status'].str.split(':', expand=True).values[:,0]
        #clinical_df.loc[clinical_df["Patient's Vital Status"].dropna().isin(['Alive','Living']).index, "Patient's Vital Status"] = 0
        #clinical_df.loc[~clinical_df["Patient's Vital Status"].dropna().isin(['Alive','Living']).index, "Patient's Vital Status"] = 1
        clinical_df["Patient's Vital Status2"] = np.where(clinical_df["Patient's Vital Status"].isin(['Alive','Living']), 0, 1)
        clinical_df["Patient's Vital Status"] = np.where(clinical_df["Patient's Vital Status"].isna(), clinical_df["Patient's Vital Status"], clinical_df["Patient's Vital Status2"])
        clinical_df["Overall Survival Status"] = np.where(clinical_df["Overall Survival Status"].isna(),
                                                         clinical_df["Patient's Vital Status"],
                                                         clinical_df["Overall Survival Status"])
        clinical_df["Diagnosis Age"] = np.where(clinical_df["Diagnosis Age"].isna(),
                                                         clinical_df["Age at Diagnosis"],
                                                         clinical_df["Diagnosis Age"])
        clinical_df = clinical_df.drop(columns=["Patient's Vital Status2", "Patient's Vital Status", "Age at Diagnosis"])
    elif study_name.split('_')[0] == 'Lung':
        clinical_df = pd.read_csv(clinical_loc, sep='\t')[
            ['Study ID', 'Sample ID', 'Diagnosis Age', "Patient Current Age", "Age (yrs)", 'Sex', 'Cancer Type',
             'Overall Survival (Months)', 'Overall Survival Status', "Patient's Vital Status"]]
        clinical_df['Overall Survival Status'] = clinical_df['Overall Survival Status'].str.split(':', expand=True).values[:,0]
        #clinical_df.loc[clinical_df["Patient's Vital Status"].dropna().isin(['Alive','Living']).index, "Patient's Vital Status"] = 0
        #clinical_df.loc[~clinical_df["Patient's Vital Status"].dropna().isin(['Alive','Living']).index, "Patient's Vital Status"] = 1
        clinical_df["Patient's Vital Status2"] = np.where(clinical_df["Patient's Vital Status"].isin(['Alive','Living']), 0, 1)
        clinical_df["Patient's Vital Status"] = np.where(clinical_df["Patient's Vital Status"].isna(), clinical_df["Patient's Vital Status"], clinical_df["Patient's Vital Status2"])
        clinical_df["Overall Survival Status"] = np.where(clinical_df["Overall Survival Status"].isna(),
                                                         clinical_df["Patient's Vital Status"],
                                                         clinical_df["Overall Survival Status"])
        clinical_df["Diagnosis Age"] = np.where(clinical_df["Diagnosis Age"].isna(),
                                                         clinical_df["Patient Current Age"],
                                                         clinical_df["Diagnosis Age"])
        clinical_df["Diagnosis Age"] = np.where(clinical_df["Diagnosis Age"].isna(),
                                                         clinical_df["Age (yrs)"],
                                                         clinical_df["Diagnosis Age"])
        clinical_df = clinical_df.drop(columns=["Patient's Vital Status2", "Patient's Vital Status", "Patient Current Age", "Age (yrs)"])

    return clinical_df

def find_mutated_samples(in_file, gene1s, gene2s, group_names='Both_OneSide_Unaltered_AtMost1', cancer=None):
    file_loc = config.DATA_DIR / 'pair_analysis' / (in_file+'.txt')
    alterations = pd.read_csv(file_loc, sep='\t')
    #alterations = filter_alterations(alterations, study_name = study_name)
    for col in alterations.columns:
        if ':' in col:
            alterations[col.split(':')] = alterations[col].str.split(":", expand=True)
            alterations = alterations.drop(columns=[col])
    groups=[]
    gene1_mut_samples = []
    gene2_mut_samples = []
    for gene1 in gene1s:
        gene1_mut_samples.append(list(alterations[alterations[gene1] == 1]['sampleId'].values))
    for gene2 in gene2s:
        gene2_mut_samples.append(list(alterations[alterations[gene2] == 1]['sampleId'].values))
    gene1_mut_samples = reduce(np.union1d, tuple(gene1_mut_samples)) if len(gene1s)>1 else gene1_mut_samples[0]
    gene2_mut_samples = reduce(np.union1d, tuple(gene2_mut_samples)) if len(gene2s)>1 else gene2_mut_samples[0]
    both_mut_samples = np.intersect1d(gene1_mut_samples, gene2_mut_samples)
    for group_name in group_names.split('_'):
        if 'Both' == group_name:
            groups.append(both_mut_samples)
        if 'OneSide' == group_name:
            any_mut_samples = np.union1d(gene1_mut_samples, gene2_mut_samples)
            groups.append(np.setdiff1d(any_mut_samples, both_mut_samples))
        if 'Unaltered' == group_name:
            groups.append(alterations[alterations['Altered'] == 0]['sampleId'].values)
        if 'AtMost1' == group_name:
            groups.append(np.setdiff1d(alterations['sampleId'].values, both_mut_samples))

    return groups


def calc_logrank_p_value(days, status, labels):
    """
    Calculates p value for given labels using clinical data
    Parameters
    ----------
    labels: iterable labels, must have same size as self.exp_pats_with_data
    Returns
    -------
    """
    return multivariate_logrank_test(days, labels, status).p_value


def calc_coxph(df, day, status, strata=[]):
    """
    Calculates p value for given labels using clinical data
    Parameters
    ----------
    labels: iterable labels, must have same size as self.exp_pats_with_data
    Returns
    -------
    """
    cph = CoxPHFitter()
    cph.fit(df, day, status, strata=strata, robust=True)
    cph.print_summary()
    return cph

def km_analysis(genes, group_names, out_filename, vers=1):
    """
    Parameters
    ----------
    labels:
        Array patient dicts, which includes pat_id, label, status, days
    out_filename:
        Name of file to save Kaplan Meier Graph
    Return
    -----------
    [0]:
        Multivariate log-rank test p values
    [1]:
        One-vs-All result of log-rank test (array of p values)
    """

    gene1_set, gene2_set = get_gene_set(gene1, gene2)
    gene_groups = (gene1_set, gene2_set)
    #plt.clf()

    #plt.figure(figsize=(8, 10))
    #ax0 = plt.subplot(2,1,1)
    #ax1 = plt.subplot(4,1,2)
    ax0 = plt.subplot2grid((16, 10), (0, 0),    rowspan=9,  colspan=7)
    ax1 = plt.subplot2grid((16, 10), (12, 1),   rowspan=1,  colspan=9)
    ax2 = plt.subplot2grid((16, 10), (15, 1),   rowspan=1,  colspan=9)
    ax3 = plt.subplot2grid((16, 10), (0, 9),    rowspan=10,  colspan=1)
    ax0.set_xlabel('Time (day)')
    ax0.set_ylabel('Survival Probability')

    in_file = out_filename.split('_')[0] + '_' + genes[0]+'_'+genes[1]
    groups = find_mutated_samples(in_file, gene_groups[0], gene_groups[1], group_names)
    cli_df = get_clinical(study_name=out_filename.split('_')[0]+'_study_clinical_data.tsv').set_index('Sample ID')
    all_days, all_status, all_groups = [], [], []
    table_rows, table_cells, table_cols = [], [], ['# of Cases', '# of Events', 'Median Survival Time']
    all_df = []
    both_cancers_size = []
    for group_id, group in enumerate(groups):  # get unique cluster labels
        group_name = group_names.split('_')[group_id]
        cli_group = cli_df.loc[group, :]
        cli_group['group'] = 'others'
        if group_name == 'Both':
            group_name = genes[0]+' and '+genes[1]
            cli_group['group']=group_name
            for cancer in cli_group['Cancer Type'].unique():
                both_cancers_size.append([cancer, np.sum(cli_group['Cancer Type']==cancer),  np.sum((cli_group['Cancer Type']==cancer)&(cli_group['Overall Survival Status']=='1'))])
        if group_name == 'OneSide':
            group_name = genes[0]+' xor '+genes[1]
        if group_name == 'AtMost1':
            group_name = '~('+genes[0]+' and '+genes[1]+')'
        all_df.append(cli_group)
        days = cli_df.loc[group,'Overall Survival (Months)']
        status = cli_df.loc[group,'Overall Survival Status']
        non_nas = list(status.notna().values) and list(days.notna().values)
        days = days[non_nas].values
        status = status[non_nas].astype(int).values
        names = np.repeat(group_name, len(days))
        all_days.append(days)
        all_status.append(status)
        all_groups.append(names)

        if vers == 2:
            if 'xor' in group_name or 'naltered' in group_name:
                alpha = 0.1
                censor_alpha = 0.1
            else:
                alpha = 1
                censor_alpha = 1
        else:
            alpha = 1
            censor_alpha = 1
        table_rows.append(group_name)
        kmf = KaplanMeierFitter()
        fitted = kmf.fit(days, status, label=group_name)
        table_cells.append([len(status), sum(status==1), f'{fitted.median_survival_time_:.4}'])
        kmf.plot(show_censors=True, censor_styles={'ms': 3, 'marker': '|', 'alpha':censor_alpha}, ci_show=False, ax=ax0, alpha=alpha)

    all_df = pd.concat(all_df).drop(columns=['Study ID']).dropna()
    all_df['group'] = all_df['group'].astype('category').cat.codes
    age_strata = pd.qcut(x=all_df['Diagnosis Age'], q=[0, .25, .5, .75, 1.])
    all_df['AGE_STRATA'] = age_strata
    all_df = all_df.drop(columns='Diagnosis Age')
    cph=calc_coxph(all_df, 'Overall Survival (Months)', 'Overall Survival Status', ['AGE_STRATA', 'Sex', 'Cancer Type'])

    all_days=np.concatenate(all_days)
    all_status=np.concatenate(all_status)
    all_groups=np.concatenate(all_groups)

    # all
    p_val = calc_logrank_p_value(all_days, all_status, all_groups)
    print(p_val)
    # one vs all
    vs_p = [f"({lb}) vs Rest: {calc_logrank_p_value(all_days, all_status, all_groups == lb):.2e}" for lb in np.unique(all_groups)]
    print(vs_p)
    multivariate_p_txt = f'Multivariate: {p_val:.2e}'
    single_p_txt = ''
    for p in vs_p:
        if (' and ' in p) and ('~' not in p):
            single_p_txt += p+'\n'

    #ax0.text(1, 0.72, multivariate_p_txt, horizontalalignment='right', verticalalignment='top', transform=ax0.transAxes)
    ax0.text(1, 0.63, single_p_txt, horizontalalignment='right', verticalalignment='top', transform=ax0.transAxes,
             fontdict={'size':9})
    ax0.legend(prop={'size': 8})

    the_table1 = ax1.table(cellText=table_cells,
                          rowLabels=table_rows,
                          colLabels=table_cols, loc='center',cellLoc='center')
    ax1.set_axis_off()

    coxph_sum = cph.summary[['coef', 'exp(coef)', 'z','p']]
    coxph_sum[['coef', 'exp(coef)', 'z']] = coxph_sum[['coef', 'exp(coef)', 'z']].applymap('{:.2f}'.format)
    coxph_sum['p'] = coxph_sum['p'].apply('{:.2e}'.format)
    table2_cells = list(coxph_sum.values)

    both_cancers_size = pd.DataFrame(both_cancers_size, columns=['Cancer', '#Cases', '#Events'])
    both_cancers_size = both_cancers_size.sort_values(by=['#Cases', '#Events'], ascending=False)
    if len(both_cancers_size)>8:
        both_cancers_size = both_cancers_size.head(8)
    #cancer_size_row = []
    #for row in both_cancers_size.values:
    #    cancer_size_row.append(f'{row[0]}-{row[1]}({row[2]})')
    #table2_cells.append(cancer_size_row)

    the_table2 = ax2.table(cellText=table2_cells,
                          rowLabels=['Mutation'],
                          colLabels=['coef', 'exp(coef)', 'z','p'], loc='center',cellLoc='center')

    the_table3 = ax3.table(cellText=both_cancers_size[['#Cases', '#Events']].values,
                          rowLabels=both_cancers_size['Cancer'].values,colWidths=[1.2] * 2,
                          colLabels=['#Cases', '#Events'], loc='center',cellLoc='center')
    the_table3.set_fontsize(8)

    ax1.set_axis_off()
    ax2.set_axis_off()
    ax3.set_axis_off()

    out_path = config.DATA_DIR / 'pair_analysis' / 'images' / f'{out_filename}_v{vers}.png'
    plt.savefig(out_path, bbox_inches='tight')
    plt.show()

    return p_val

gene1='BRCA'
gene2='PARPFAM'
km_analysis(genes=(gene1,gene2), group_names='Both_OneSide_Unaltered_AtMost1', out_filename=f'Pancancer_{gene1}_{gene2}_both_oneside_nonalter', vers=2)