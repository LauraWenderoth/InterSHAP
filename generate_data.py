import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from pathlib import Path
import argparse
import umap
from utils import save_data
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




def generate_synthetic_data(num_samples, d1, d2, d, delta,setting = 'redundancy'):
    """
    Generate synthetic data (v, t, y) according to the specified process.

    Parameters:
    num_samples (int): Number of synthetic data points to generate.
    d1 (int): Number of rows for matrix V.
    d2 (int): Number of rows for matrix T.
    d (int): Number of columns for matrices V and T.
    delta (float): Threshold for |v · t|.

    Returns:
    data (list): List of tuples, each containing (V v, T t, y).
    """
    X = []
    label = []
    base_ratio =  0.40
    d_base = int(d * base_ratio)
    d_modality = d - d_base

    # Step 1: Sample random projection V and T from U(-0.5, 0.5)
    M1 = np.random.uniform(-0.5, 0.5, size=(d1, d))
    M2 = np.random.uniform(-0.5, 0.5, size=(d2, d))
    while len(X) < num_samples:



        # Step 2: Sample v and t from N(0, 1) and normalize to unit length
        m1 = np.random.normal( size=(d_modality,))
        m2 = np.random.normal( size=(d_modality,))
        base = np.random.normal( size=(d_base,))
        m1 = m1 / np.linalg.norm(m1)
        m2 = m2 / np.linalg.norm(m2)
        base = base / np.linalg.norm(base)

        
        #check label
        label_vector1 = 0
        label_vector2 = 0
        if setting == "redundancy":
            label_vector1 = base[:len(base) // 2]
            label_vector2 = base[len(base) // 2:]
                
        elif setting == "uniqueness0":
            label_vector1 = m1[:len(m1) // 2]
            label_vector2 = m1[len(m1) // 2:]

        elif setting == "uniqueness1":
            label_vector1 = m2[:len(m2) // 2]
            label_vector2 = m2[len(m2) // 2:]
            
        elif setting == "synergy":
            label_vector1 = m1
            label_vector2 = m2

            
        # Step 3: Check if |v · t| > delta
        if abs(np.dot(label_vector1, label_vector2)) > delta:
            # Step 4: Determine y based on the sign of v · t
            if np.dot(label_vector1, label_vector2) > 0:
                y = 1
            else:
                y = 0
            # Step 5: Add data point to the list
            X.append([np.dot(M1, np.concatenate((m1, base))), np.dot(M2, np.concatenate((m2, base)))])
            label.append(y)
    print(np.sum(label)/len(label))

    return X,label


if __name__ == "__main__":
    # Example usage:
    parser = argparse.ArgumentParser()
    parser.add_argument("--perturbation", type=bool, default=True, help="Whether to apply perturbation")
    parser.add_argument("--number_samples", type=int, default=200, help="Number of samples")
    parser.add_argument("--save_path", type=str,
                        default="/home/lw754/masterproject/PID/synthetic_data/",
                        help="Path to save the file")
    parser.add_argument("--setting", type=str, default='uniqueness1',
                        choices=['redundancy', 'uniqueness0', 'uniqueness1', 'synergy'], help="Data generation setting")
    parser.add_argument("--n_modality", type=int, default=2, help="Number of modalities")
    parser.add_argument("--label", type=str, default='AND', help="Number of modalities")

    args = parser.parse_args()

    save_path = Path(args.save_path)

    if args.label == 'OR' or args.label == 'XOR' or args.label == 'AND':
        X, y = generate_data_AND(args.setting, args.perturbation, number_samples=args.number_samples,
                                 number_modalities=args.n_modality, label=args.label)
        X = np.array(X).transpose((1, 0, 2))  # (modalities, samples, features) --> (samples, modalities, features)


    elif args.label == 'VEC': # synergy d = 200
        d = 200
        d1 = 2000
        d2 = 1000
        delta = 0.25

        # Generate synthetic data
        X, y = generate_synthetic_data(args.number_samples, d1, d2, d, delta)


    data = dict()
    data['train'] = dict()
    data['valid'] = dict()
    data['test'] = dict()

    # Assuming you already have X_train, X_valid, X_test, y_train, y_valid, y_test, and n_modality defined

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y)
    X_valid, X_test, y_valid, y_test = train_test_split(X_test, y_test, test_size=0.5, stratify=y_test)

    ''' 
    for i in range(args.n_modality):
        data['train'][str(i)] = np.array(X_train)[:, i, :]
        data['valid'][str(i)] = np.array(X_valid)[:, i, :]
        data['test'][str(i)] = np.array(X_test)[:, i, :]
    '''
    for i in range(args.n_modality):
        data['train'][str(i)] = [sample[i] for sample in X_train]
        data['valid'][str(i)] = [sample[i] for sample in X_valid]
        data['test'][str(i)] = [sample[i] for sample in X_test]

    data['train']['label'] = np.array(y_train)
    data['valid']['label'] = np.array(y_valid)
    data['test']['label'] = np.array(y_test)

    visualize_umap(data, args.n_modality, args.label, args.setting)
    # Save the data
    filename = save_path/f"{args.label}_DATA_{args.setting}.pickle"  # Using setting variable directly
    save_data(data, filename)
    print(f'Saved data to: {filename}')
