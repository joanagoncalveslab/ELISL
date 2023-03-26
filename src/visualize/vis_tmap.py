"""
Visualizing RNA sequencing data using tmap.

Data Source:
https://gdc.cancer.gov/about-data/publications/pancanatlas
"""

import numpy as np
import pandas as pd
from faerun import Faerun
import tmap as tm
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

# Configuration for the tmap layout
CFG_TMAP = tm.LayoutConfiguration()
CFG_TMAP.k = 100
CFG_TMAP.kc = 50
CFG_TMAP.node_size = 1 / 60

def tmap_ds_cancer_fold():
    loc_dict = dl.get_all_locs()
    for ds_name in ['seq_1024', 'ppi_ec', 'crispr_dependency_mut', 'crispr_dependency_expr', 'tissue']:
        tr_data = pd.read_csv(loc_dict[f'train_{ds_name}_data_loc'])
        tr_data = tr_data.fillna(0)
        for cancer in ['BRCA', 'CESC', 'KIRC', 'LAML', 'LUAD', 'OV', 'SKCM']:
            processed_data_tr = dfnc.prepare_cancer_dataset(tr_data, cancer=cancer)
            train_ind_loc = config.DATA_DIR / 'feature_sets' / f'train_test_{cancer}_{10}.json'
            with open(train_ind_loc, 'r') as fp:
                fold_dict = json.load(fp)
            for fold_id, items in fold_dict.items():
                print(f'Fold: {fold_id}')
                y = processed_data_tr['class'].values
                labels = processed_data_tr['pair_name'].values
                all_x = processed_data_tr.drop(columns=['pair_name', 'class']).values
                y = y[items['train']]
                labels = labels[items['train']]
                all_x = all_x[items['train']]
                print(all_x.shape)
                x = StandardScaler().fit_transform(all_x)

                colors = ['red', "blue"]
                cmap = ListedColormap(colors)

                # Initialize and configure tmap
                dims = 1024
                enc = tm.Minhash(x.shape[1], 42, dims)
                lf = tm.LSHForest(dims * 2, 128, store=True)

                fps = []
                for _, row in enumerate(x):
                    fps.append(tm.VectorFloat(list(row)))

                nneighbors = CFG_TMAP.k
                print("Running TMAP ...")
                start = timer()
                lf.batch_add(enc.batch_from_weight_array(fps, method="I2CWS"))
                lf.index()
                x, y, s, t, _ = tm.layout_from_lsh_forest(lf, CFG_TMAP)
                algo_time = timer() - start
                print(f"TMAP: {algo_time}")
                lf.clear()

                legend_labels = {(1, "Positive"), (0, "Negative"), (2, "Positive-KRAS"), (3, "Negative-KRAS")}

                # Create the plot
                faerun = Faerun(clear_color='#ffffff', view="front", coords=False, legend_title="")
                faerun.add_scatter(
                    f"SL_{cancer}",
                    {"x": x, "y": y, "c": y, "labels": labels},
                    colormap=cmap,
                    point_scale=5.0,
                    max_point_size=10,
                    shader="smoothCircle",
                    has_legend=True,
                    categorical=True,
                    legend_labels=legend_labels,
                    legend_title="SL-BRCA",
                )
                faerun.add_tree(
                    f"SL_{cancer}_tree", {"from": s, "to": t}, point_helper=f"SL_{cancer}", color="#666666"
                )
                faerun.plot(
                    f'TMAP_nneighbors={nneighbors}_time={format(algo_time, ".2f")}_{cancer}_train_{ds_name}_{fold_id}')


def tmap_ds_cancer(ds_aim=None, cancer_aim=None, chosen_gene=None):
    loc_dict = dl.get_all_locs()
    for ds_name in ['seq_1024', 'ppi_ec', 'crispr_dependency_mut', 'crispr_dependency_expr', 'tissue']:
        if ds_aim is not None and ds_aim != ds_name:
            continue
        tr_data = pd.read_csv(loc_dict[f'train_{ds_name}_data_loc'])
        tr_data = tr_data.fillna(0)
        for cancer in ['BRCA', 'CESC', 'KIRC', 'LAML', 'LUAD', 'OV', 'SKCM']:
            if cancer_aim is not None and cancer_aim != cancer:
                continue
            processed_data_tr = dfnc.prepare_cancer_dataset(tr_data, cancer=cancer)
            # processed_data_tr = dfnc.prepare_cancer_dataset(tr_data, cancer=cancer)
            # processed_data_te = dfnc.prepare_cancer_dataset(te_data, cancer=cancer)
            # processed_data = pd.concat([processed_data_tr, processed_data_te])
            Y = processed_data_tr['class'].values
            labels = processed_data_tr['pair_name'].values
            if chosen_gene is not None:
                chosen_gene_pairs = [chosen_gene in label for label in labels]
                Y[chosen_gene_pairs & (Y == 1)] = 2
                Y[chosen_gene_pairs & (Y == 0)] = 3
            all_X = processed_data_tr.drop(columns=['pair_name', 'class']).values
            # Y = Y[items['train']]
            # labels = labels[items['train']]
            # all_X = all_X[items['train']]
            print(all_X.shape)
            scaler = StandardScaler().fit(all_X)
            X = scaler.transform(all_X)

            # Coniguration for the tmap layout
            CFG_TMAP = tm.LayoutConfiguration()
            CFG_TMAP.k = 100
            CFG_TMAP.kc = 50
            CFG_TMAP.node_size = 1 / 80

            # DATA = pd.read_csv("data.csv.xz", index_col=0, sep=",")
            # LABELS = pd.read_csv("labels.csv", index_col=0, sep=",")

            # LABELMAP = {"PRAD": 1, "LUAD": 2, "BRCA": 3, "KIRC": 4, "COAD": 5}
            # LABELS = np.array([int(LABELMAP[v]) for v in LABELS["Class"]], dtype=np.int)
            # LABELS = np.array(labels, dtype='str')
            from matplotlib.colors import ListedColormap

            colors = ['#EE99AA', "#6699CC", "#004488", "#994455"]
            cmap = ListedColormap(colors)

            # Initialize and configure tmap
            dims = 1024
            enc = tm.Minhash(X.shape[1], 42, dims)
            lf = tm.LSHForest(dims * 2, 128, store=True)

            fps = []
            for _, row in enumerate(X):
                fps.append(tm.VectorFloat(list(row)))

            nneighbors = CFG_TMAP.k
            print("Running TMAP ...")
            start = timer()
            lf.batch_add(enc.batch_from_weight_array(fps, method="I2CWS"))
            lf.index()
            x, y, s, t, _ = tm.layout_from_lsh_forest(lf, CFG_TMAP)
            algo_time = timer() - start
            print(f"TMAP: {algo_time}")
            lf.clear()
            legend_labels = {(1, "Positive"), (0, "Negative"), (2, f"Positive-{chosen_gene}"),
                             (3, f"Negative-{chosen_gene}")}

            # Create the plot
            faerun = Faerun(clear_color='#ffffff', view="front", coords=False, legend_title="")
            faerun.add_scatter(
                f"SL_{cancer}",
                {"x": x, "y": y, "c": Y, "labels": labels},
                colormap=cmap,
                point_scale=5.0,
                max_point_size=10,
                shader="smoothCircle",
                has_legend=True,
                categorical=True,
                legend_labels=legend_labels,
                legend_title="SL-BRCA",
            )
            faerun.add_tree(
                f"SL_{cancer}_tree", {"from": s, "to": t}, point_helper=f"SL_{cancer}", color="#666666"
            )
            faerun.plot(
                f'd2TMAP_nneighbors={nneighbors}_time={format(algo_time, ".2f")}_{cancer}_train_{ds_name}_{chosen_gene}')


if __name__ == "__main__":
    # umap_ds_cancer(ds_aim=None, cancer_aim=None, chosen_gene = None, label_usage='supervised')
    # umap_ds_cancer(ds_aim=None, cancer_aim='BRCA', chosen_gene = 'BRCA1', label_usage='supervised')
    # umap_ds_cancer(ds_aim=None, cancer_aim='BRCA', chosen_gene = 'BRCA2', label_usage='supervised')
    # umap_ds_cancer(ds_aim=None, cancer_aim='BRCA', chosen_gene = 'PARP1', label_usage='supervised')
    # umap_ds_cancer(ds_aim=None, cancer_aim='BRCA', chosen_gene = 'TP53', label_usage='supervised')
    # umap_ds_cancer(ds_aim=None, cancer_aim='LUAD', chosen_gene = 'KRAS', label_usage='supervised')
    umap_ds_cancer(ds_aim=None, cancer_aim=None, chosen_gene=None, label_usage='fakesemisupervised')
