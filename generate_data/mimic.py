"""Implements dataloaders for generic MIMIC tasks."""
from tqdm import tqdm
import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
from utils.utils import save_data

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.getcwd())), "MultiBench"))

from robustness.tabular_robust import add_tabular_noise
from robustness.timeseries_robust import add_timeseries_noise
import sys

import numpy as np
from torch.utils.data import DataLoader
import random
import pickle
import copy
import os
import sys
from pathlib import Path
import argparse
import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import pandas as pd
from sklearn.model_selection import train_test_split
import umap
from generate_data import visualize_umap, format_split



sys.path.append(os.path.dirname(os.path.dirname(os.getcwd())))


def get_dataset(task, imputed_path='im.pk', flatten_time_series=False, tabular_robust=False, timeseries_robust=False):
    """Get dataloaders for MIMIC dataset.

    Args:
        task (int): Integer between -1 and 19 inclusive, -1 means mortality task, 0-19 means icd9 task.
        imputed_path (str, optional): Datafile location. Defaults to 'im.pk'.
        flatten_time_series (bool, optional): Whether to flatten time series data or not. Defaults to False.
        tabular_robust (bool, optional): Whether to apply tabular robustness as dataset augmentation or not. Defaults to True.
        timeseries_robust (bool, optional): Whether to apply timeseries robustness noises as dataset augmentation or not. Defaults to True.

    Returns:
        tuple: Tuple of training dataset, validation dataset, and test dataset
    """
    f = open(imputed_path, 'rb')
    datafile = pickle.load(f)
    f.close()
    X_t = datafile['ep_tdata']
    X_s = datafile['adm_features_all']

    X_t[np.isinf(X_t)] = 0
    X_t[np.isnan(X_t)] = 0
    X_s[np.isinf(X_s)] = 0
    X_s[np.isnan(X_s)] = 0

    X_s_avg = np.average(X_s, axis=0)
    X_s_std = np.std(X_s, axis=0)
    X_t_avg = np.average(X_t, axis=(0, 1))
    X_t_std = np.std(X_t, axis=(0, 1))

    for i in range(len(X_s)):
        X_s[i] = (X_s[i] - X_s_avg) / X_s_std
        for j in range(len(X_t[0])):
            X_t[i][j] = (X_t[i][j] - X_t_avg) / X_t_std

    static_dim = len(X_s[0])
    timestep = len(X_t[0])
    series_dim = len(X_t[0][0])
    if flatten_time_series:
        X_t = X_t.reshape(len(X_t), timestep * series_dim)
    if task < 0:
        y = datafile['adm_labels_all'][:, 1]
        admlbl = datafile['adm_labels_all']
        le = len(y)
        for i in range(0, le):
            if admlbl[i][1] > 0:
                y[i] = 1
            elif admlbl[i][2] > 0:
                y[i] = 2
            elif admlbl[i][3] > 0:
                y[i] = 3
            elif admlbl[i][4] > 0:
                y[i] = 4
            elif admlbl[i][5] > 0:
                y[i] = 5
            else:
                y[i] = 0
    else:
        y = datafile['y_icd9'][:, task]
        le = len(y)
    datasets = [[X_s[i], X_t[i]] for i in range(le)]
    labels = [[y[i]] for i in range(le)]
    random.seed(10)

    random.shuffle(datasets)

    valid_X = datasets[0:le // 10]
    train_X = datasets[le // 5:]
    valid_labels = labels[0:le // 10]
    train_labels = labels[le // 5:]

    if tabular_robust or timeseries_robust:
        test_X = []
        y_robust = []
        for noise_level in tqdm(range(11)):
            dataset_robust = copy.deepcopy(datasets[le // 10:le // 5])
            test_labels = copy.deepcopy(labels[le // 10:le // 5])
            if tabular_robust:
                X_s_robust = add_tabular_noise([dataset_robust[i][0] for i in range(
                    len(dataset_robust))], noise_level=noise_level / 10)
            else:
                X_s_robust = [dataset_robust[i][0]
                              for i in range(len(dataset_robust))]
            if timeseries_robust:
                X_t_robust = add_timeseries_noise([[dataset_robust[i][1] for i in range(
                    len(dataset_robust))]], noise_level=noise_level / 10)[0]
            else:
                X_t_robust = [dataset_robust[i][1]
                              for i in range(len(dataset_robust))]
            y_robust = [test_labels[i] for i in range(len(dataset_robust))]
            if flatten_time_series:
                 test_X.extend([(X_s_robust[i], X_t_robust[i].reshape(timestep * series_dim))
                         for i in range(len(y_robust))])
            else:
                test_X.extend([(X_s_robust[i], X_t_robust[i]) for i in range(
                    len(y_robust))])
            y_robust.extend([y_robust[i] for i in range(
                    len(y_robust))])
    else:
        test_X = copy.deepcopy(datasets[le // 10:le // 5])
        y_robust = copy.deepcopy(labels[le // 10:le // 5])



    return train_X, valid_X, test_X, train_labels, valid_labels,y_robust

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_path", type=str,
                        default="/home/lw754/masterproject/real_world_data/",
                        help="Path to save the file")
    parser.add_argument("--imputed_path", type=str,
                        default='/home/lw754/masterproject/MultiBench/datasets/mimic/im.pk',
                        help="Path where data is stored")
    parser.add_argument("--task", type=int, default=7, help="-1 means mortality task, 1 means icd9 10-19 task, 7 means ic9 70-79 task")
    parser.add_argument("--noise",action='store_true',
                        help="If add noise to test set")

    args = parser.parse_args()

    save_path = Path(args.save_path)
    save_path.mkdir(parents=True, exist_ok=True)


    print('create mimic')
    X_train,X_valid,X_test,y_train,y_valid,y_test = get_dataset(task=args.task, imputed_path=args.imputed_path, flatten_time_series=True, tabular_robust=args.noise, timeseries_robust=args.noise)
    mods = 2
    labels = ['Class 1', 'Class 2' ]
    save_name = f"mimic_DATA_task{args.task}"
    if args.noise:
        save_name = f"mimic_DATA_task{args.task}_noise"

    data = format_split(X_train,X_valid,X_test,y_train,y_valid,y_test,n_modalities=mods)



    visualize_umap(data,mods ,save_path/save_name,legend_labels = labels)
    # Save the data
    filename = save_path/f"{save_name}.pickle"  # Using setting variable directly
    save_data(data, filename)