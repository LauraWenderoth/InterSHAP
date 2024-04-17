import numpy as np
from sklearn.model_selection import train_test_split
from pathlib import Path
import argparse
import umap
from utils.utils import save_data
import matplotlib.pyplot as plt
import math

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

def check_dot_product(vectors, delta):
    for i in range(len(vectors)):
        for j in range(i + 1, len(vectors)):
            dot_product = np.dot(vectors[i], vectors[j])
            if abs(dot_product) <= delta:
                return False
    return True

def calc_label(vectors):
    count_true = 0
    total_counts = math.comb(len(vectors), 2)
    for i in range(len(vectors)):
        for j in range(i + 1, len(vectors)):
            dot_product = np.dot(vectors[i], vectors[j])
            if dot_product > 0:
                count_true += 1
                if count_true > (total_counts//2):
                    return 0
    return 1


def generate_synthetic_data(num_samples, dimensions, d, delta, setting='redundancy'):
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
    n_modalities = len(dimensions)
    assert n_modalities > 1, 'Choose at least 2 modalities'
    X = []
    label = []
    base_ratio = 0.40
    d_base = int(d * base_ratio)
    d_modality = d - d_base

    # Step 1: Sample random projection V and T from U(-0.5, 0.5)
    M = [np.random.uniform(-0.5, 0.5, size=(dimensions[i], d)) for i in range(n_modalities)]

    while len(X) < num_samples:

        # Step 2: Sample v and t from N(0, 1) and normalize to unit length
        m = [np.random.normal(size=(d_modality,)) for i in range(n_modalities)]
        base = np.random.normal(size=(d_base,))
        m = [m1 / np.linalg.norm(m1) for m1 in m]
        base = base / np.linalg.norm(base)

        # check label
        label_vector = 0
        if setting == "redundancy":
            label_vector = [base[:len(base) // 2], base[len(base) // 2:]]

        elif "uniqueness" in setting:
            assert setting and setting[
                -1].isdigit(), 'Last char has to be the number of the modality that should determine the label (starts with 0)'
            unique_modality = int(setting[-1])
            m1 = m[unique_modality]
            label_vector = [m1[:len(m1) // 2], m1[len(m1) // 2:]]

        elif setting == "synergy":
            label_vector = m

        # Step 3: Check if |v · t| > delta
        if len(label_vector) == 2:
            if abs(np.dot(label_vector[0], label_vector[1])) > delta:
                # Step 4: Determine y based on the sign of v · t
                if np.dot(label_vector[0], label_vector[1]) > 0:
                    y = 1
                else:
                    y = 0
                # Step 5: Add data point to the list
                X.append([np.dot(M1, np.concatenate((m1, base))) for M1, m1 in zip(M, m)])
                label.append(y)
        else:
            if check_dot_product(label_vector, delta):
                y = calc_label(m)
                if y:
                    if np.sum(label) <= num_samples // 2:
                        X.append([np.dot(M1, np.concatenate((m1, base))) for M1, m1 in zip(M, m)])
                        label.append(y)
                else:
                    X.append([np.dot(M1, np.concatenate((m1, base))) for M1, m1 in zip(M, m)])
                    label.append(y)


def generate_equally_distributed_data(num_samples, n_modalities):
    data = np.random.randint(0, 2, size=(num_samples * (n_modalities-1), n_modalities))
    labels = np.where(np.sum(data, axis=1) == 1, 1, 0)
    # Find indices where labels are 0 and 1
    zero_indices = np.where(labels == 0)[0]
    one_indices = np.where(labels == 1)[0]
    min_samples = min(len(one_indices), num_samples // 2)
    filtered_data = np.concatenate((data[zero_indices[:(num_samples-min_samples)]], data[one_indices[:min_samples]]))

    # Shuffle the filtered data
    np.random.shuffle(filtered_data)

    return filtered_data[:num_samples]  # Trim the data to ensure total number of samples


def generate_synthetic_data_XOR(num_samples, dimensions, d, delta, run_name = 'XOR',setting = 'redundancy'):
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
    n_modalities = len(dimensions)
    assert n_modalities > 1, 'Choose at least 2 modalities'
    X = []

    # Step 1: Sample random projection V and T from U(-0.5, 0.5)
    M = [np.random.uniform(-0.5, 0.5, size=(dimensions[i], d)) for i in range(n_modalities)]

    data = np.random.randint(0, 2, size=(num_samples, 2))
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
            vector = []
            for mi in range(2):
                if data[sample_i][mi] == 0:
                    loc = -0.35
                elif data[sample_i][mi] == 1:
                    loc = 0.35
                vector.append(np.random.normal(loc=loc, scale=1.0, size=(d // 2,)))

            x_end = [np.random.normal(loc=0.0, scale=1.0, size=(d,)) for i in range(n_modalities)]
            x_end[unique_modality] = np.concatenate(vector)
            
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

    return X,label


if __name__ == "__main__":
    # Example usage:
    parser = argparse.ArgumentParser()
    parser.add_argument("--perturbation", type=bool, default=True, help="Whether to apply perturbation")
    parser.add_argument("--number_samples", type=int, default=20000, help="Number of samples")
    parser.add_argument("--delta", type=float, default=0.10, help="redundancy d = 200 + delta = 0.25; synergy d = 200 delte = 0.105 # uniqueness delta = 0.1  synergy mod = 3: 0.10")

    parser.add_argument("--save_path", type=str,
                        default="/home/lw754/masterproject/PID/synthetic_data/",
                        help="Path to save the file")
    parser.add_argument("--setting", type=str, default='uniqueness3',
                        choices=['redundancy', 'uniqueness0', 'uniqueness1', 'synergy'], help="Data generation setting")
    parser.add_argument('--dim_modalities', nargs='+', type=int, default=[200, 100, 150, 100], help='List of dim for modalities')

    parser.add_argument("--label", type=str, default='VEC', help="Number of modalities")

    args = parser.parse_args()

    save_path = Path(args.save_path)
    print(args.setting, args.number_samples)

    if args.label == 'OR' or args.label == 'XOR' or args.label == 'AND':
        X, y = generate_data_AND(args.setting, args.perturbation, number_samples=args.number_samples,
                                 number_modalities=len(args.dim_modalities), label=args.label)
        X = np.array(X).transpose((1, 0, 2))  # (modalities, samples, features) --> (samples, modalities, features)


    elif args.label == 'VEC': # synergy d = 200
        d = 200


        # Generate synthetic data
        X, y = generate_synthetic_data_XOR(args.number_samples, args.dim_modalities, d, args.delta,setting=args.setting)


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
    for i in range(len(args.dim_modalities)):
        data['train'][str(i)] = [sample[i] for sample in X_train]
        data['valid'][str(i)] = [sample[i] for sample in X_valid]
        data['test'][str(i)] = [sample[i] for sample in X_test]

    data['train']['label'] = np.array(y_train)
    data['valid']['label'] = np.array(y_valid)
    data['test']['label'] = np.array(y_test)

    visualize_umap(data, len(args.dim_modalities), args.label, args.setting)
    # Save the data
    filename = save_path/f"{args.label}{len(args.dim_modalities)}_DATA_{args.setting}.pickle"  # Using setting variable directly
    save_data(data, filename)
    print(f'Saved data to: {filename}')
