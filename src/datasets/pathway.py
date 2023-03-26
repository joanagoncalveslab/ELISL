import os
print(os.getcwd())
import sys
sys.path.insert(0,"/Users/yitepeli/PycharmProjects/ELISL")
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
import numpy as np
import pandas as pd
from src import config
import src.data_functions as dfnc
from builtins import any as b_any

def get_pathway_dict(aim=None):
    pathway_folder = config.DATA_DIR / 'pathway'
    pws = {}
    for file in os.listdir(pathway_folder):
        if aim is None or aim in file:
            pw_loc = pathway_folder / file
            pws.update(dfnc.load_csv(pw_loc, sep='\t'))
    return pws

def get_pathway_df(aim='all'):
    pathway_folder = config.DATA_DIR / 'pathway'
    if aim=='all':
        loc = pathway_folder / 'c2.all.v7.4.symbols.gmt'
        return pd.read_csv(loc)
    for file in os.listdir(pathway_folder):
        if aim in file:
            pw_loc = pathway_folder / file
            return pd.read_csv(pw_loc)
    return None

def get_pws(pw_dict, replication_b=True, repair_b=True, cancer_b=True, additionals_b=True):
    selected_keywords = []
    replication= ['KEGG_DNA_REPLICATION', 'WP_DNA_REPLICATION', 'KAUFFMANN_DNA_REPLICATION_GENES']
    repair = ['KEGG_BASE_EXCISION_REPAIR', 'KEGG_BASE_EXCISION_REPAIR', 'REACTOME_BASE_EXCISION_REPAIR',
              'WP_NUCLEOTIDE_EXCISION_REPAIR', 'KEGG_NUCLEOTIDE_EXCISION_REPAIR', 'REACTOME_NUCLEOTIDE_EXCISION_REPAIR',
              'KEGG_MISMATCH_REPAIR', 'REACTOME_MISMATCH_REPAIR', 'WP_DNA_MISMATCH_REPAIR',
              'WP_HOMOLOGOUS_RECOMBINATION', 'KEGG_HOMOLOGOUS_RECOMBINATION',
              '_NON_HOMOLOGOUS_END_JOINING', '_FANCONI_PATHWAY']
    cancer=['PATHWAYS_IN_CANCER']#, 'BREAST_CANCER']
    additionals= ['KEGG_CELL_CYCLE', 'WP_CELL_CYCLE',
                  '_SIGNALING_BY_HIPPO', '_MYC_PATHWAY', '_NOTCH_PATHWAY',
                  'WP_NOTCH_SIGNALING', 'KEGG_NOTCH_SIGNALING_PATHWAY',
                  '_NRF2_PATHWAY', '_PI3KAKT_', '_TGFB_FAMILY', '_TGFB_PATHWAY',
                  'BIOCARTA_P53_PATHWAY', 'KEGG_P53_SIGNALING_PATHWAY',
                  'PID_BETA_CATENIN_DEG_PATHWAY', 'PID_BETA_CATENIN_NUC_PATHWAY']
    if replication_b:
        for item in replication:
            selected_keywords.append(item)
    if repair_b:
        for item in repair:
            selected_keywords.append(item)
    if cancer_b:
        for item in cancer:
            selected_keywords.append(item)
    if additionals_b:
        for item in additionals:
            selected_keywords.append(item)

    selected_pws = [key for key, item in pw_dict.items() if b_any(x in key for x in selected_keywords)]
    selected_gene_lists = [item for key, item in pw_dict.items() if b_any(x in key for x in selected_keywords)]
    selected_dict_items = {key: item for key, item in pw_dict.items() if b_any(x in key for x in selected_keywords)}
    all_genes = list(set([item for sublist in selected_gene_lists for item in sublist]))
    return all_genes
    print()

def create_pairs(genes, cancer='BRCA', out='cancer', alreadys=[], check_trte=True):
    tr_loc = config.DATA_DIR / ('labels/train_pairs.csv')
    te_loc = config.DATA_DIR / ('labels/test_pairs.csv')
    tr_data = pd.read_csv(tr_loc)
    te_data = pd.read_csv(te_loc)
    if check_trte:
        prev_list = [tr_data, te_data]
    else:
        prev_list = []
    for already in alreadys:
        already_loc = config.DATA_DIR / 'labels' / ('unknown_'+already + '_' + cancer + '_pairs.csv')
        already_data = pd.read_csv(already_loc)
        prev_list.append(already_data)


    #already_loc = config.DATA_DIR / 'labels' / ('unknown_'+already + '_' + cancer + '_pairs.csv')
    #already_data = pd.read_csv(already_loc)

    pairs = []
    if type(genes) is dict:
        for id1, id2_list in genes.items():
            for id2 in id2_list:
                pair_row = [min(id1, id2), max(id1, id2)]
                pairs.append(pair_row)
    else:
        genes = np.sort(genes)
        for id1 in range(len(genes)):
            for id2 in range(id1+1, len(genes)):
                pair_row=[genes[id1], genes[id2]]
                pairs.append(pair_row)
    new_data = pd.DataFrame(pairs, columns=['gene1', 'gene2'])

    if len(prev_list)>0:
        prev_pairs = pd.concat(prev_list)
        prev_pairs = prev_pairs[prev_pairs['cancer'] == cancer]
        diff_df = new_data.merge(prev_pairs.drop(columns=['cancer', 'class']), on=['gene1', 'gene2'], how='outer',
                                    indicator=True).loc[lambda x: x['_merge'] == 'left_only'].drop(columns='_merge')
    else:
        diff_df = new_data.copy()
    diff_df['cancer']=cancer
    diff_df['class'] = -1
    diff_df = diff_df.sort_values(by=['gene1', 'gene2'])
    save_loc = config.DATA_DIR / 'labels' / (out + '_' + cancer + '_pairs.csv')
    diff_df.to_csv(save_loc, index=None)
    print()

    #df[df['A'].str.contains("hello")]


all_pw_dict = get_pathway_dict(aim='all')
all_genes = get_pws(all_pw_dict, replication_b=False, repair_b=True, cancer_b=True, additionals_b=False)
pairs = create_pairs(all_genes, cancer='SKCM', out='unknown_repair_cancer', alreadys=[], check_trte=True)

all_genes = {'KRAS': ['POLL', 'POLB', 'POLM', 'TENT4A', 'DNTT',
                      'THRA', 'THRB', 'RARA', 'RARB', 'RARG', 'PPARA', 'PPARD', 'PPARG', 'NR1D1', 'NR1D2', 'RORA', 'RORB', 'RORC',
                      'NR1H4', 'NR1H5P', 'NR1H3', 'NR1H2', 'VDR', 'NR1I2', 'NR1I3',
                      'IL6', 'IL11', 'IL27', 'IL31', 'CNTF', 'LIF', 'OSM',
                      'MRPL1', 'MRPL2', 'MRPL3', 'MRPL4', 'MRPL9', 'MRPL10', 'MRPL11', 'MRPL12', 'MRPL13', 'MRPL14',
                      'MRPL15', 'MRPL16', 'MRPL17', 'MRPL18', 'MRPL19', 'MRPL20', 'MRPL21', 'MRPL22', 'MRPL23',
                      'MRPL24',
                      'MRPL27', 'MRPL28', 'MRPL30', 'MRPL32', 'MRPL33', 'MRPL34', 'MRPL35', 'MRPL36', 'MRPL37',
                      'MRPL38',
                      'MRPL39', 'MRPL40', 'MRPL41', 'MRPL42', 'MRPL43', 'MRPL44', 'MRPL45', 'MRPL46', 'MRPL47', 'MRPL48',
                      'MRPL49', 'MRPL50', 'MRPL51', 'MRPL52', 'MRPL53', 'MRPL54', 'MRPL55', 'MRPL57', 'MRPL58'
                      ]}
'''
'PARP1', 'PARP2', 'PARP3', 'PARP4', 'TNKS', 'TNKS2', 'PARP6', 'TIPARP', 'PARP8', 'PARP9', 'PARP10', 'PARP11',
'PARP12', 'ZC3HAV1', 'PARP14', 'PARP15', 'PARP16',
'''
'''
all_genes = {'BRCA1':['WNT3A', 'WNT7A', 'WNT6', 'WNT16', 'IHH', 'SHH', 'DHH', 'PTCH1',
                      'FGF1', 'FGF2', 'FGF3', 'FGF4', 'FGF5', 'FGF6', 'FGF7', 'FGF8', 'FGF9', 'FGF10', 'FGF11', 'FGF12', 'FGF13',
                      'FGF14', 'FGF16', 'FGF17', 'FGF18', 'FGF19', 'FGF20', 'FGF21', 'FGF22', 'FGF23'],
             'BRCA2': ['WNT3A', 'WNT7A', 'WNT6', 'WNT16', 'IHH', 'SHH', 'DHH', 'PTCH1',
                      'FGF1', 'FGF2', 'FGF3', 'FGF4', 'FGF5', 'FGF6', 'FGF7', 'FGF8', 'FGF9', 'FGF10', 'FGF11', 'FGF12', 'FGF13',
                      'FGF14', 'FGF16', 'FGF17', 'FGF18', 'FGF19', 'FGF20', 'FGF21', 'FGF22', 'FGF23']}
'''
#all_genes_extra_b = {'BRCA1':['FGF12'],
#             'BRCA2': ['PTCH1']}
#all_genes_extra_l = {'KRAS':['IL6', 'CNTF', 'THRA', 'RARA', 'POLM', 'MRPL30', 'MRPL49']}
#pairs = create_pairs(all_genes_extra_l, cancer='LUAD', out='negative_families_extra', alreadys=[], check_trte=False)
'''
pairs=[(['BRCA1'],['FGF8']),(['BRCA2'],['FGF6']),(['BRCA1','BRCA2'],['HHIP', 'IHH', 'DHH', 'SHH', 'PTCH1'])]
pairs=[(['KRAS', 'K-RAS'],['MRPL28', 'MAAT1', 'L28MT', 'P15', 'MRP-L28']), (['KRAS'],['POLL']), (['KRAS'],['NR1D2']), (['KRAS'],['OSM'])]
pairs=[(['BRCA1','BRCA2'],['WNT3A', 'WNT7A', 'WNT16'])]

for pair in pairs:
    print(f'\nAnalysis of genes:{pair[0]} and genes:{pair[1]}')
    pair0_pws = [key for key,item in all_pw_dict.items() if len(set(pair[0]).intersection(item))>0]
    pair1_pws = [key for key,item in all_pw_dict.items() if len(set(pair[1]).intersection(item))>0]
    intersection_pws = set(pair0_pws).intersection(pair1_pws)
    print(intersection_pws)
print()
'''

