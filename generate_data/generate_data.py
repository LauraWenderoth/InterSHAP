import os
import sys
from pathlib import Path
import argparse

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

from sklearn.model_selection import train_test_split
import umap

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
from utils.utils import save_data


def visualize_umap(data, n_modality,save_path,legend_labels = ['0', '1']):
    # Concatenate the modalities for the training dataset
    try:
        X_train_concatenated = np.concatenate([data['train'][str(i)] for i in range(n_modality)], axis=1)


        y_train = data['train']['label']

        # Apply UMAP for dimensionality reduction
        umap_reducer = umap.UMAP(n_neighbors=50, min_dist=0.5)
        umap_result = umap_reducer.fit_transform(X_train_concatenated)

        # Plotting
        plt.figure(figsize=(6, 6))
        plt.scatter(umap_result[:, 0], umap_result[:, 1], c=y_train, cmap='coolwarm', s=5)
        unique_labels = np.unique(y_train)

        norm = Normalize(vmin=np.min(y_train), vmax=np.max(y_train))
        # Create empty legend handles
        legend_handles = [plt.Line2D([], [], marker='o', markersize=5, linestyle='None', color=plt.cm.coolwarm(norm(label))) for label in unique_labels]

        # Plot the legend
        plt.legend(legend_handles, legend_labels, loc='upper right')
        plt.axis('off')
        plt.tight_layout(pad=0)
        plt.savefig(str(save_path) + '_scatter_plot_with_legend.pdf')
        plt.show()
    except:
        print('Data has do be 1D')


def generate_equally_distributed_data(num_samples, n_modalities):
    data = np.random.randint(0, 2, size=(num_samples * (n_modalities-1), n_modalities))
    labels = np.where(np.sum(data, axis=1) == 1, 1, 0)
    zero_indices = np.where(labels == 0)[0]
    one_indices = np.where(labels == 1)[0]
    min_samples = min(len(one_indices), num_samples // 2)
    filtered_data = np.concatenate((data[zero_indices[:(num_samples-min_samples)]], data[one_indices[:min_samples]]))
    np.random.shuffle(filtered_data)

    return filtered_data[:num_samples]

def generate_random_data(num_samples, dimensions, d):
    n_modalities = len(dimensions)
    data = np.random.normal(loc=0.5, scale=1.0, size=(num_samples, d))
    label = np.random.randint(0, 2, size=(num_samples, 1))
    M = [np.random.uniform(-0.5, 0.5, size=(dimensions[i], d)) for i in range(n_modalities)]
    result = [[np.dot(row, M[i].T)  for i in range(n_modalities)] for row in data]
    return result,label

def generate_synthetic_data(num_samples, dimensions, d, run_name = 'XOR',setting = 'redundancy'):
    """
    """
    n_modalities = len(dimensions)
    assert n_modalities > 1, 'Choose at least 2 modalities'
    X = []

    # Step 1: Sample random projection V and T from U(-0.5, 0.5)
    M = [np.random.uniform(-0.5, 0.5, size=(dimensions[i], d)) for i in range(n_modalities)]

    data = np.random.randint(0, 2, size=(num_samples, 2))
    clear_data = []
    if setting == "synergy" and n_modalities > 2:
        data = generate_equally_distributed_data(num_samples, n_modalities)
    if run_name == 'AND':
        label = np.prod(data, axis=1)
    elif run_name == 'OR':
        label = np.any(data == 1, axis=1).astype(int)
    elif run_name == 'XOR':
        label = np.where(np.sum(data, axis=1) == 1, 1, 0)
    else:
        print('False setting selected: selcect XOR, OR, AND for label')
        label = None

    for sample_i in range(num_samples):
        # Step 2: Sample v and t from N(0, 1) and normalize to unit length
        #m = [np.random.normal( size=(d,)) for i in range(n_modalities)]
        #m = [m1 / np.linalg.norm(m1) for m1 in m]

        x_end = 0
        if setting == "redundancy":
            vector = []
            for mi in range(2):
                if data[sample_i][mi] == 0:
                    loc =-0.25
                elif data[sample_i][mi] == 1:
                    loc = 0.25
                vector.append(np.random.normal(loc=loc, scale=1.0,size=(d//2,)))
            x_end = [np.concatenate(vector) for i in range(n_modalities)]

                
        elif "uniqueness" in setting:
            assert setting and setting[-1].isdigit(), 'Last char has to be the number of the modality that should determine the label (starts with 0)'
            unique_modality = int(setting[-1])
            assert unique_modality <= n_modalities, f'Setting {setting} not defined for, check number of dimensions (not enough dims)'
            vector = []
            for mi in range(2):
                if data[sample_i][mi] == 0:
                    loc = -0.35
                elif data[sample_i][mi] == 1:
                    loc = 0.35
                vector.append(np.random.normal(loc=loc, scale=1.0, size=(d // 2,)))

            x_end = [np.random.normal(loc=0.0, scale=1.0, size=(d,)) for i in range(n_modalities)]
            x_end[unique_modality] = np.concatenate(vector)

            x_org = [np.random.randint(0, 2, size=(2)) for i in range(n_modalities)]
            x_org[unique_modality] = data[sample_i]
            clear_data.append(x_org)
            
        elif setting == "synergy":
            x_end = []
            for mi in range(n_modalities):
                if data[sample_i][mi] == 0:
                    loc = -0.35
                elif data[sample_i][mi] == 1:
                    loc = 0.35
                x_end.append(np.random.normal(loc=loc, scale=2.0, size=(d,)))
        X.append([np.dot(M1, m1) for M1, m1 in zip(M, x_end)])
    print(f'Proportion of the label 1: {np.sum(label) / len(label)}')
    if setting == "synergy" or setting == "redundancy":
        clear_data = [[np.array([item]) for item in row] for row in data]
    return X,clear_data, label

def create_split(X,y,n_modalities):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y)
    X_valid, X_test, y_valid, y_test = train_test_split(X_test, y_test, test_size=0.5, stratify=y_test)

    data = format_split(X_train,X_valid,X_test,y_train,y_valid,y_test,n_modalities)

    return data


def format_split(X_train,X_valid,X_test,y_train,y_valid,y_test,n_modalities):
    data = dict()
    data['train'] = dict()
    data['valid'] = dict()
    data['test'] = dict()

    for i in range(n_modalities):
        data['train'][str(i)] = [sample[i] for sample in X_train]
        data['valid'][str(i)] = [sample[i] for sample in X_valid]
        data['test'][str(i)] = [sample[i] for sample in X_test]

    data['train']['label'] = np.array(y_train)
    data['valid']['label'] = np.array(y_valid)
    data['test']['label'] = np.array(y_test)
    return data

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--perturbation", type=bool, default=True, help="Whether to apply perturbation")
    parser.add_argument("--number_samples", type=int, default=20000, help="Number of samples")
    parser.add_argument("--save_path", type=str,
                        default="/home/lw754/masterproject/synthetic_data/",
                        help="Path to save the file")
    parser.add_argument("--setting", type=str, default='random',
                        choices=['redundancy', 'uniqueness0', 'uniqueness1','uniqueness2','uniqueness3','uniqueness4', 'synergy','random'], help="Data generation setting")
    parser.add_argument('--dim_modalities', nargs='+', type=int, default=[200, 100], help='List of dim for modalities')
    parser.add_argument("--label", type=str, default='XOR', choices=['XOR', 'OR', 'AND'],  help="Number of modalities")
    args = parser.parse_args()
    save_path = Path(args.save_path)
    save_path.mkdir(parents=True, exist_ok=True)
    print(args.setting, args.number_samples)

    d = 200
    if args.setting =='random':
        X, y = generate_random_data(args.number_samples,  args.dim_modalities, d)
    else:
        X,org_data, y = generate_synthetic_data(args.number_samples, args.dim_modalities, d,setting=args.setting)
        if args.setting == 'redundancy':
            org_data = create_split(org_data, y, 2)
        else:
            org_data = create_split(org_data, y, len(args.dim_modalities))
        filename = save_path / f"VEC{len(args.dim_modalities)}{args.label}_org_DATA_{args.setting}.pickle"  # Using setting variable directly
        save_data(org_data, filename)

    data = create_split(X, y, len(args.dim_modalities))



    visualize_umap(data, len(args.dim_modalities),save_path/f"VEC{len(args.dim_modalities)}{args.label}_DATA_{args.setting}")
    # Save the data
    filename = save_path/f"VEC{len(args.dim_modalities)}{args.label}_DATA_{args.setting}.pickle"  # Using setting variable directly
    save_data(data, filename)

