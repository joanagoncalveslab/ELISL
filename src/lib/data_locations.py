from src import config
import collections as coll

FEATURE_SET = 'feature_sets'
CUTOFF = 'co=1.96'
UNKNOWN = 'unknown_repair_cancer'


def get_unk_locs():
    loc_dict = coll.OrderedDict()
    for cancer in {'BRCA', 'CESC', 'KIRC', 'LAML', 'LUAD', 'OV', 'SKCM'}:
        loc_dict[f'unk_seq_1024_{cancer}_data_loc'] = \
            config.DATA_DIR / FEATURE_SET / f'{UNKNOWN}_{cancer}_seq_1024.csv'
        loc_dict[f'unk_crispr_dependency_expr_{cancer}_data_loc'] = \
            config.DATA_DIR / FEATURE_SET / f'{UNKNOWN}_{cancer}_crispr_dependency_expr_{CUTOFF}.csv.gz'
        for ds_name in ['ppi_ec', 'crispr_dependency_mut', 'tissue']:
            loc_dict[f'unk_{ds_name}_{cancer}_data_loc'] = \
                config.DATA_DIR / FEATURE_SET / f'{UNKNOWN}_{cancer}_{ds_name}.csv.gz'
    return loc_dict

def get_other_locs():
    loc_dict = coll.OrderedDict()
    for dt in ['PPI', 'GO1', 'GO2']:
        loc_dict['GCATSL_' + dt + '_data_loc'] = \
            config.DATA_DIR / 'feature_sets' / 'GCATSL' / 'ppi_128dim.csv'
    return loc_dict


def get_main_locs():
    loc_dict = coll.OrderedDict()
    for t_set in ['train', 'unknown', 'test', 'isle', 'dsl', 'lu15', 'exp2sl', 'unknown_cancer_BRCA',
                  'unknown_repair_cancer_BRCA',
                  'unknown_families_BRCA', 'negative_families_LUAD', 'unknown_families_extra_BRCA',
                  'negative_families_extra_LUAD']:
        loc_dict[f'{t_set}_colm_data_loc'] = config.DATA_DIR / FEATURE_SET / f'colm_full_{t_set}.csv'

        for other in ['seq_1024', 'seq_512', 'seq_256', 'seq_128', 'seq_64', 'seq_32', 'onehot']:
            loc_dict[f'{t_set}_{other}_data_loc'] = config.DATA_DIR / FEATURE_SET / f'{t_set}_{other}.csv'
        for other in ['ppi_ec', 'tissue', 'tissue_diff_expr']:
            loc_dict[f'{t_set}_{other}_data_loc'] = config.DATA_DIR / FEATURE_SET / f'{t_set}_{other}.csv.gz'
        for dep_type in ['crispr', 'd2', 'comb']:
            for omic in ['mut', 'cnv', 'comb', 'muex']:
                loc_dict[f'{t_set}_{dep_type}_dependency_{omic}_data_loc'] = \
                    config.DATA_DIR / FEATURE_SET / f'{t_set}_{dep_type}_dependency_{omic}.csv.gz'
            for omic in ['expr', 'any']:
                loc_dict[f'{t_set}_{dep_type}_dependency_{omic}_data_loc'] = \
                    config.DATA_DIR / FEATURE_SET / f'{t_set}_{dep_type}_dependency_{omic}_{CUTOFF}.csv.gz'
        for tis_type in ['surv', 'tcoexp', 'hcoexp', 'coexp', 'diff_expr']:
            loc_dict[f'{t_set}_tissue_{tis_type}_data_loc'] = \
                config.DATA_DIR / 'feature_sets' / f'{t_set}_tissue_{tis_type}_{CUTOFF}.csv.gz'
    return loc_dict

def get_all_locs():
    main_loc = get_main_locs()
    unk_loc = get_unk_locs()
    other_loc = get_other_locs()
    all_locs = {**main_loc, **unk_loc, **other_loc}
    return all_locs


