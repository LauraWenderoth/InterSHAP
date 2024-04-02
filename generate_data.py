import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from pathlib import Path
import argparse
import umap
import matplotlib.pyplot as plt
def generate_data_AND(setting, perturbation, number_features=8, number_modalities=2, number_samples=100, label='AND'):
    number_features = number_features // 2
    # Generate random data and labels
    data = np.random.randint(0, 2, size=(number_samples, number_modalities))
    if label =='AND':
        labels = np.prod(data, axis=1)
    elif label == 'OR':
        labels = np.any(data == 1, axis=1).astype(int)
    elif label == 'XOR':
        labels = np.where(np.sum(data, axis=1) == 1, 1, 0)
    else:
        print('False setting selected: selcect XOR, OR, AND for label')
        labels = None

    # Extend each number by a factor of 4
    data = np.repeat(data, number_features, axis=1)
    data = data.astype(float)

    if setting == "redundancy":
        # Write 01 in each modality
        data = [data, data]

    elif setting == "uniqueness0":
        # Write data only in the first modality and 0.5 in the second modality
        data = [data, np.full_like(data, 0.5)]

    elif setting == "uniqueness1":
        # Write data only in the second modality and 0.5 in the first modality
        data = [np.full_like(data, 0.5), data]

    elif setting == "synergy":
        data = [ np.repeat(data[:, :number_features], 2, axis=1) ,  np.repeat( data[:, number_features:],2, axis=1)]

    # Apply perturbation
    if perturbation:
        perturbed_data = []
        for modality in data:
            noise = np.random.normal(0, 0.3, size=modality.shape)  # Example of adding Gaussian noise
            perturbed_data.append(modality + noise)
        data = perturbed_data

    return data, labels

def save_data(data, filename):
    with open(filename, 'wb') as f:
        pickle.dump(data, f)
def visualize_umap(data, n_modality, label, setting):
    # Concatenate the modalities for the training dataset
    X_train_concatenated = np.concatenate([data['train'][str(i)] for i in range(n_modality)], axis=1)
    y_train = data['train']['label']

    # Apply UMAP for dimensionality reduction
    umap_reducer = umap.UMAP()
    umap_result = umap_reducer.fit_transform(X_train_concatenated)

    # Plotting
    plt.figure(figsize=(10, 8))
    plt.scatter(umap_result[:, 0], umap_result[:, 1], c=y_train, cmap='viridis', s=5)
    plt.colorbar()
    plt.title(f'UMAP Visualization for {label} Data ({setting})')
    plt.xlabel('UMAP Dimension 1')
    plt.ylabel('UMAP Dimension 2')
    plt.show()


if __name__ == "__main__":
    # Example usage:
    parser = argparse.ArgumentParser()
    parser.add_argument("--perturbation", type=bool, default=True, help="Whether to apply perturbation")
    parser.add_argument("--number_samples", type=int, default=20000, help="Number of samples")
    parser.add_argument("--save_path", type=str,
                        default="/home/lw754/masterproject/PID/synthetic_data/",
                        help="Path to save the file")
    parser.add_argument("--setting", type=str, default='uniqueness1',
                        choices=['redundancy', 'uniqueness0', 'uniqueness1', 'synergy'], help="Data generation setting")
    parser.add_argument("--n_modality", type=int, default=2, help="Number of modalities")
    parser.add_argument("--label", type=str, default='OR', help="Number of modalities")

    args = parser.parse_args()

    save_path = Path(args.save_path)
    X, y = generate_data_AND(args.setting, args.perturbation, number_samples=args.number_samples,
                                 number_modalities=args.n_modality, label=args.label)

    # Assuming you have the necessary data ready

    data = dict()
    data['train'] = dict()
    data['valid'] = dict()
    data['test'] = dict()

    # Assuming you already have X_train, X_valid, X_test, y_train, y_valid, y_test, and n_modality defined

    X_train, X_test, y_train, y_test = train_test_split(np.array(X).transpose((1,0,2)), y, test_size=0.3, stratify=y) #(modalities, samples, features) --> (samples, modalities, features)
    X_valid, X_test, y_valid, y_test = train_test_split(X_test, y_test, test_size=0.5, stratify=y_test)

    for i in range(args.n_modality):
        data['train'][str(i)] = np.array(X_train)[:, i, :]
        data['valid'][str(i)] = np.array(X_valid)[:, i, :]
        data['test'][str(i)] = np.array(X_test)[:, i, :]

    data['train']['label'] = np.array(y_train)
    data['valid']['label'] = np.array(y_valid)
    data['test']['label'] = np.array(y_test)

    visualize_umap(data, args.n_modality, args.label, args.setting)
    # Save the data
    filename = save_path/f"{args.label}_DATA_{args.setting}.pickle"  # Using setting variable directly
    save_data(data, filename)
    print(f'Saved data to: {filename}')
