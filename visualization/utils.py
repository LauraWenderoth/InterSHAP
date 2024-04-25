import shap
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from shap import TreeExplainer, Explanation
from pathlib import Path
import math

def split_acc_class(values_array,base_values):
    split_arrays = []

    unique_base_values = set(base_values)
    for base_value in unique_base_values:
        mask = np.array(base_values) == base_value
        split_array = values_array[mask]
        split_arrays.append(split_array)

    return split_arrays, unique_base_values
def sum_pairs(array, n_mod, features):
    result_dict = {}
    for n_sample in range(array.shape[0]):
        a = 0
        for i in range(n_mod):
            a += 1
            for j in range(a,n_mod):
                if i != j:
                    key = "$I_{" + features[i]+ " + " + features[j]+ "}$"
                    if key not in result_dict.keys():
                        result_dict[key] = []
                    value = array[n_sample][i][j] + array[n_sample][j][i]
                    result_dict[key].append(value)
    return result_dict

def explainer_split_interactions(shaply_values, interactions_2D,base_values, n_modalities,modality_names):
    interaction_split = sum_pairs(interactions_2D, n_modalities, modality_names)
    interaction_split_data = np.array(list(interaction_split.values()))
    interaction_split_data = np.transpose(interaction_split_data)
    shap_data = np.hstack((shaply_values, interaction_split_data))
    exp = Explanation(shap_data,
                      base_values,
                      feature_names=modality_names + list(interaction_split.keys()))
    return exp

def map_to_2D(data,n_samples,n_modalities):
    shaply_values = data[:, :n_modalities]
    interactions = data[:, n_modalities:]
    interactions_2D  = np.zeros((n_samples,n_modalities, n_modalities))
    interactions_2D[:,~np.eye(n_modalities, dtype=bool)] = interactions
    return shaply_values,interactions,interactions_2D

def load_class_base_value_dic(explaination_path,n_classes,n_modalities):
    class_dict = dict()

    for i in range(n_classes):
        df = pd.read_csv(explaination_path / f'class_{i}.csv')
        base_values = list(df[str(np.zeros(n_modalities, dtype=int))])
        class_dict[base_values[0]] = f'Class {i}'
    return class_dict

def get_explanations(explaination_path,modality_names,n_classes):
    results_dict = dict()
    explaination_path = Path(explaination_path)
    file_name_base_value = 'best.csv'
    file_name_interactions = 'interaction_index_best_with_ii.csv'
    data = pd.read_csv(explaination_path/file_name_interactions)
    data = data.values
    df = pd.read_csv(explaination_path/file_name_base_value)

    n_modalities = math.sqrt(data.shape[1])
    assert n_modalities % 1 == 0, 'Interactions have wrong format!'
    n_modalities = int(n_modalities)
    assert n_modalities == len(modality_names), 'Not enough or to many names for modalities'

    base_values = list(df[str(np.zeros(n_modalities,dtype=int))])
    n_samples = len(base_values)
    shaply_values, interactions, interactions_2D = map_to_2D(data, n_samples, n_modalities)



    exp = explainer_split_interactions(shaply_values, interactions_2D,base_values,n_modalities,modality_names)
    results_dict['Single Interaction Split'] = exp

    class_dict = load_class_base_value_dic(explaination_path, n_classes,n_modalities)

    split_array, unique_base_values = split_acc_class(data,base_values)
    for data_split, base_value_split in zip(split_array,unique_base_values):
        n_samples_split = data_split.shape[0]
        shaply_values_split, interactions_split, interactions_2D_split = map_to_2D(data_split,n_samples_split , n_modalities)
        exp_split = explainer_split_interactions(shaply_values_split, interactions_2D_split, [base_value_split] * n_samples_split, n_modalities, modality_names)
        class_name = class_dict[base_value_split]
        results_dict[f'Single {class_name}'] = exp_split


    shap_data = np.concatenate((np.mean(shaply_values, axis=0), [np.mean(interactions)]))
    shap_data = shap_data.reshape(1,shap_data.shape[0])
    exp = Explanation(shap_data,
                      [np.mean(base_values)],
                      feature_names=modality_names+['Interactions'])
    results_dict['Mean'] = exp

    shap_data = np.hstack((shaply_values,np.sum(interactions,axis=1)[:, np.newaxis]))
    exp = Explanation(shap_data,
                      (base_values),
                      feature_names=modality_names + ['Interactions'])
    results_dict['Single'] = exp
    return results_dict

if __name__ == "__main__":
    explaination_path = '/cross-modal-interaction/results/VEC2XOR_redundancy_epochs_1_concat_early/seed_1/'
    split = True
    modality_names = ['RNA', 'Protein']
    indiviuell_samples = [1,2]
    overall = True
    plot_mode = 'waterfall' # 'force' 'waterfall'
    n_classes = 2

    results = get_explanations(explaination_path,modality_names,n_classes)
    exp = results['Mean']
    shap.force_plot(exp[0], show=True, matplotlib=True)
        #ax = shap.plots.waterfall(exp[0],show=False)
        # ax = shap.force_plot(exp[0],show=False,matplotlib=True)
        #plt.show()
        #plt.savefig('/home/lw754/masterproject/dummy_plot.pdf')