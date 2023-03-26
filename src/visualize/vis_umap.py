"""
Visualizing RNA sequencing data using tmap.

Data Source:
https://gdc.cancer.gov/about-data/publications/pancanatlas
"""
import os
import sys
import argparse

path2this = os.path.dirname(os.path.abspath(__file__)).split('/')
project_path = ''
for i, folder in enumerate(path2this):
    if folder.lower() == 'elisl':
        project_path = '/'.join(path2this[:i + 1])
sys.path.insert(0, project_path)
import numpy as np
import pandas as pd
from timeit import default_timer as timer
from src import config
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import os
import json
import umap
from src.lib import data_locations as dl
from src import data_functions as dfnc

parser = argparse.ArgumentParser(description='Self-training arguments')
parser.add_argument('--nneighbors', '-nn', metavar='the-n-neighbors', dest='nneighbors', type=int,
                    help='Choose the max iteration', default=100)
parser.add_argument('--mindist', '-md', metavar='the-min-dist', dest='mindist', type=float,
                    help='Choose the min-dist', default=0.001)
parser.add_argument('--target_weight', '-tw', metavar='the-target-weight', dest='target_weight', type=float, help='Choose the target_weight', default=0.9)
args = parser.parse_args()
print(f'Running args:{args}')

def umap_ds_cancer(ds_aims=['seq_1024'], cancer_aims=['BRCA'], chosen_gene=None, label_usage='None'):
    nneighbors = args.nneighbors
    mindist = args.mindist
    target_weight = args.target_weight
    unk_fold = 1
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
                for unk_fold in [1, 2, 3]:
                    png_loc = config.RESULT_DIR / 'sample_vis' / \
                        f'UMAP_nn={nneighbors}_md={mindist}_{cancer}{fold_id}_train_{ds_name}_{label_usage}_{chosen_gene}_tw{target_weight}_unk{unk_fold}.png'
                    pdf_loc = config.RESULT_DIR / 'sample_vis' / \
                        f'UMAP_nn={nneighbors}_md={mindist}_{cancer}{fold_id}_train_{ds_name}_{label_usage}_{chosen_gene}_tw{target_weight}_unk{unk_fold}.pdf'
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

                    if 'semisupervised' in label_usage:
                        processed_data_unk = dfnc.prepare_cancer_dataset(unk_data, cancer=cancer)
                        if unk_fold != -1:
                            processed_data_unk = processed_data_unk.sample(n=10000, replace=False,
                                                                           random_state=unk_fold)
                        all_x_unk = processed_data_unk.drop(columns=['pair_name', 'class']).values.copy()
                        y_unk = processed_data_unk['class'].values.copy()
                        raw_y_unk = processed_data_unk['class'].values.copy()

                    scaler = StandardScaler()
                    if label_usage == 'fakesemisupervised':
                        scaler.fit(all_x)
                        x = scaler.transform(all_x)
                        x_unk = scaler.transform(all_x_unk)
                    elif label_usage == 'semisupervised':
                        all_x = np.concatenate((all_x, all_x_unk))
                        raw_y = np.concatenate((raw_y, raw_y_unk))
                        y = np.concatenate((y, y_unk))
                        x = scaler.fit_transform(all_x)

                    print("Running UMAP ...")
                    start = timer()
                    mapper = umap.UMAP(n_neighbors=nneighbors, min_dist=mindist, target_weight=target_weight)
                    if label_usage == 'None':
                        x_transformed = mapper.fit_transform(x)
                    elif label_usage in ['supervised', 'semisupervised']:
                        x_transformed = mapper.fit_transform(x, y=raw_y)
                    elif 'fakesemisupervised' == label_usage:
                        mapper.fit(x, y=raw_y)
                        x_transformed = mapper.transform(x)
                        x_unk_transformed = mapper.transform(x_unk)
                    algo_time = timer() - start
                    print(f"UMAP: {algo_time}")

                    negatives = y == 0
                    positives = y == 1
                    unknowns = y == -1
                    negatives_chosen = y == 3
                    positives_chosen = y == 2
                    if 'semisupervised' == label_usage:
                        plt.scatter(x_transformed[unknowns, 0], x_transformed[unknowns, 1], c="#FF8000",
                                    s=0.25, label='Unknown')
                    plt.scatter(x_transformed[negatives, 0], x_transformed[negatives, 1], c="#EE99AA",
                                s=0.25, label='Negative')
                    plt.scatter(x_transformed[positives, 0], x_transformed[positives, 1], c="#6699CC",
                                s=0.25, label='Positive')
                    if chosen_gene is not None:
                        plt.scatter(x_transformed[negatives_chosen, 0], x_transformed[negatives_chosen, 1], c="#994455",
                                    s=0.25, label=f'Negative_{chosen_gene}')
                        plt.scatter(x_transformed[positives_chosen, 0], x_transformed[positives_chosen, 1], c="#004488",
                                    s=0.25, label=f'Positive_{chosen_gene}')

                    if 'fakesemisupervised' == label_usage:
                        plt.scatter(x_unk_transformed[:, 0], x_unk_transformed[:, 1], c="#FF8000",
                                    s=0.25, label='Unknown')

                    # ax[int(fold_id), ds_id].legend()
                    plt.xticks([])
                    plt.yticks([])
                    plt.legend()
                    plt.title(
                        f'UMAP({nneighbors},{mindist}) of {cancer}{all_x.shape}-{ds_name} in {format(algo_time, ".2f")} sec')

                    plt.savefig(pdf_loc, type='pdf', dpi=300,
                                bbox_inches='tight')
                    plt.savefig(png_loc, type='png', dpi=300,
                                bbox_inches='tight')
                    plt.show()


if __name__ == "__main__":
    ds_aims = ['seq_1024', 'ppi_ec', 'crispr_dependency_mut', 'crispr_dependency_expr', 'tissue']
    cancer_aims = ['BRCA', 'CESC', 'KIRC', 'LAML', 'LUAD', 'OV', 'SKCM']
    # umap_ds_cancer(ds_aim=None, cancer_aim=None, chosen_gene = None, label_usage='supervised')
    # umap_ds_cancer(ds_aims=ds_aims, cancer_aims=['BRCA'], chosen_gene = 'BRCA1', label_usage='supervised')
    # umap_ds_cancer(ds_aims=ds_aims, cancer_aims=['BRCA'], chosen_gene = 'BRCA2', label_usage='supervised')
    # umap_ds_cancer(ds_aims=ds_aims, cancer_aims=['BRCA'], chosen_gene = 'PARP1', label_usage='supervised')
    # umap_ds_cancer(ds_aims=ds_aims, cancer_aims=['BRCA'], chosen_gene = 'TP53', label_usage='supervised')
    # umap_ds_cancer(ds_aims=ds_aims, cancer_aims=['LUAD'], chosen_gene = 'KRAS', label_usage='supervised')
    umap_ds_cancer(ds_aims=ds_aims, cancer_aims=cancer_aims, chosen_gene=None, label_usage='fakesemisupervised')
