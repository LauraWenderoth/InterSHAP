import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from pathlib import Path
import argparse
def generate_data_AND(setting, perturbation, number_features=8, number_modalities=2, number_samples=100):
    number_features = number_features // 2
    # Generate random data and labels
    data = np.random.randint(0, 2, size=(number_samples, number_modalities))
    labels = np.prod(data, axis=1)

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
            noise = np.random.normal(0, 0.1, size=modality.shape)  # Example of adding Gaussian noise
            perturbed_data.append(modality + noise)
        data = perturbed_data

    return data, labels

def save_data(data, filename):
    with open(filename, 'wb') as f:
        pickle.dump(data, f)

if __name__ == "__main__":
    # Example usage:
    parser = argparse.ArgumentParser()
    parser.add_argument("--perturbation", type=bool, default=True, help="Whether to apply perturbation")
    parser.add_argument("--number_samples", type=int, default=20000, help="Number of samples")
    parser.add_argument("--save_path", type=str,
                        default="/home/lw754/masterproject/PID/synthetic_data/",
                        help="Path to save the file")
    parser.add_argument("--setting", type=str, default='synergy',
                        choices=['redundancy', 'uniqueness0', 'uniqueness1', 'synergy'], help="Data generation setting")
    parser.add_argument("--n_modality", type=int, default=2, help="Number of modalities")
    args = parser.parse_args()

    save_path = Path(args.save_path)
    # Generate data
    X, y  = generate_data_AND(args.setting, args.perturbation, number_samples=args.number_samples,
                                                 number_modalities=args.n_modality)




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

    # Save the data
    filename = save_path/"myDATA_{}.pickle".format(args.setting)  # Using setting variable directly
    save_data(data, filename)
