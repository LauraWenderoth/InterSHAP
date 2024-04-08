import numpy as np
import pandas as pd
import math


def geometric_scaling( pi,value_ij):
    # Dot product of sij and pi
    dot_product = np.dot(value_ij, pi)

    # Euclidean norm of pi
    norm_pi = np.linalg.norm(pi)** 2

    # Scaling calculation
    scaled_value = dot_product / norm_pi

    return scaled_value


def ratio_of_squared_norms( pi,sij):
    # Squared Euclidean norm of sij
    squared_norm_sij = np.linalg.norm(sij) ** 2

    # Squared Euclidean norm of pi
    squared_norm_pi = np.linalg.norm(pi) ** 2

    # Ratio of the squared norms
    ratio = squared_norm_sij / squared_norm_pi

    return ratio

def vector_projection(p_i, p_ij):
    p_i = np.array(p_i)
    p_ij = np.array(p_ij)
    # Dot product of pi and pij
    dot_product = np.dot(p_i, p_ij)

    # Euclidean norm of pij squared
    norm_pij_squared = np.linalg.norm(p_ij) ** 2

    # Synergy vector calculation
    synergy_vector = (dot_product / norm_pij_squared) * p_ij

    return synergy_vector
def SRI(shaply_interaction_values,n_modalities):
    P_ij = dict()
    P_i = dict()
    synergy_values = dict()
    synergy_vec = dict()
    autonomy_vec = dict()
    redundancy_vec = dict()
    independence_vec = dict()
    redundancy_values = dict()
    independence_values = dict()

    for sample in shaply_interaction_values:
        interactions = np.array(sample)
        for i in range(n_modalities):
            for j in range(n_modalities):
                ij = interactions[i][j]
                key = f'{i}{j}'
                if key not in P_ij:
                    P_ij[key] = []
                P_ij[key].append(ij)
    for i in range(n_modalities):
        modality_key = f'{i}'
        P_i[modality_key] = np.sum([P_ij[key] for key in P_ij.keys() if key.startswith(modality_key)],axis=0)
    #synergy
    for i in range(n_modalities):
        for j in range(n_modalities):
            if i != j:
                p_i = P_i[f'{i}']
                p_ij = P_ij[f'{i}{j}']
                syn_ij = vector_projection(p_i, p_ij)
                autonomy_vec[f'{i}{j}'] = p_i - syn_ij
                syn_value = ratio_of_squared_norms( p_i,syn_ij)
                synergy_values[f'{i}{j}'] = syn_value
                synergy_vec[f'{i}{j}'] = syn_ij
    #redundancy + independence
    for i in range(n_modalities):
        for j in range(n_modalities):
            if i != j:
                p_i = P_i[f'{i}']
                a_ij = autonomy_vec[f'{i}{j}']
                a_ji = autonomy_vec[f'{j}{i}']
                redundancy_ij = vector_projection(a_ij, a_ji)
                red_value = ratio_of_squared_norms(p_i, redundancy_ij)
                independence_ij = a_ij - redundancy_ij
                indep_value = ratio_of_squared_norms(p_i, independence_ij )

                redundancy_values[f'{i}{j}'] = red_value
                independence_values[f'{i}{j}'] = indep_value

                redundancy_vec[f'{i}{j}'] = redundancy_ij
                independence_vec[f'{i}{j}'] = independence_ij
    return synergy_values, redundancy_values, independence_values

def SRI_normalise(shaply_interaction_values,n_modalities):
    synergy_values, redundancy_values, independence_values = SRI(shaply_interaction_values, n_modalities)

    #normalise
    synergy_values = list(synergy_values.values())
    redundancy_values = list(redundancy_values.values())
    independence_values = list(independence_values.values())
    normalise_factor = np.sum([synergy_values, redundancy_values, independence_values])
    synergy = np.sum(synergy_values) /normalise_factor
    redundancy = np.sum(redundancy_values) / normalise_factor
    uniqueness = list(np.array(independence_values)/normalise_factor)
    uniqueness_dict = {f'SRI_uniqueness{i}': value for i, value in enumerate(uniqueness)}

    result = {'SRI_synergy': synergy, 'SRI_redundancy': redundancy}
    result.update(uniqueness_dict)
    return result


def powerset(lst):
    n = len(lst)
    indices = np.indices((2,) * n).reshape(n, -1).T
    powerset_masks = np.array([[lst[j] if i else 0 for i, j in zip(subset, range(n))] for subset in indices])
    # Sort the powerset masks in descending order
    sorted_indices = np.lexsort(np.rot90(powerset_masks))
    return powerset_masks[sorted_indices[::-1]]
def shaply_interaction_values(df_coalitions,number_modalities):
    modality_arrays = [np.eye(number_modalities, dtype=int)[i] for i in range(number_modalities)]
    interaction_values = []
    feature_names = [f'{i}' for i in range(number_modalities)]
    permutations = [1] * number_modalities
    powerset_masks = powerset(permutations)

    for sample_row in range(len(df_coalitions)):
        interactions = []
        for i in modality_arrays:
            interaction_row = []
            for j in modality_arrays:
                if np.any(i != j):
                    modality_index_i = np.where(i == 1)[0][0]
                    modality_index_j = np.where(j == 1)[0][0]
                    filtered_rows = powerset_masks[np.where((powerset_masks[:, modality_index_i] == 0) & (powerset_masks[:, modality_index_j] == 0))]
                    filtered_rows_as_strings = [row for row in filtered_rows]
                    interaction_value = 0
                    for np_row, name in zip(filtered_rows, filtered_rows_as_strings):
                        coalition_size = np.sum(np_row)
                        weight = (math.factorial(coalition_size) * math.factorial(
                            number_modalities - coalition_size - 2)) / (2*math.factorial(
                            number_modalities-1))
                        S_i = str(np_row + i)
                        S_j = str(np_row + j)
                        S_i_j = str(np_row + j + i)
                        S = str(np_row)
                        result_S_i = df_coalitions[S_i][sample_row]
                        result_S_j = df_coalitions[S_j][sample_row]
                        result_S_i_j = df_coalitions[S_i_j][sample_row]
                        result_S = df_coalitions[S][sample_row]
                        interaction_value += weight*(result_S_i_j+result_S-result_S_j-result_S_i)
                    interaction_row.append(interaction_value)
                else:
                    interaction_row.append(0)
            interactions.append(interaction_row)

        interaction_df = pd.DataFrame(interactions, columns=[feature_names], index=[feature_names])
        for idx,i in enumerate(modality_arrays):
            shaply_value_i = df_coalitions[f'shaply_value_{i}'][sample_row]
            interaction_i = np.sum(interaction_df[feature_names[idx]].values)
            interaction_df.loc[feature_names[idx],feature_names[idx]] = shaply_value_i - interaction_i
        interaction_values.append(interaction_df)
    return interaction_values

if __name__ == "__main__":
    number_modalities = 2
    path = '/home/lw754/masterproject/cross-modal-interaction/results/mix5_epochs_150_concat_True/seed_1/best.csv'
    df = pd.read_csv(path)
    df_coalitions = df.iloc[:, :(2**number_modalities+number_modalities)]
    shap_interactions = shaply_interaction_values(df_coalitions,number_modalities)
    synergy_values, redundancy_values, independence_values = SRI(shap_interactions,number_modalities)
    print(f"01 {synergy_values['01'] + redundancy_values['01'] +independence_values['01']}")
    print(f"10 {synergy_values['10'] + redundancy_values['10'] + independence_values['10']}")
    synergy, redundancy, uniqueness = SRI_normalise(synergy_values, redundancy_values, independence_values)
    print(f'Synergy {synergy}')
    print(f'Redundancy {redundancy}')
    print(f'Uniqueness {uniqueness}')

