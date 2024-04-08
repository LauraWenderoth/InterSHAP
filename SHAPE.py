import pandas as pd
import numpy as np
def cross_modal_SHAPE(coalition_values,number_modalities):
    modality_arrays = [np.eye(number_modalities, dtype=int)[i] for i in range(number_modalities)]
    key_all = str(np.array([1]*number_modalities))
    key_base  = str(np.array([0]*number_modalities))
    all_mods  = coalition_values[key_all]
    base =  coalition_values[key_base]

    modality_marginal_ditribution = np.zeros(all_mods.shape)
    for key_mod in modality_arrays:
        m_i = coalition_values[str(key_mod)]
        phi_mi = m_i - base
        modality_marginal_ditribution += phi_mi

    shape_cross_modal = all_mods -base  - modality_marginal_ditribution
    shape_cross_modal_mean = np.mean(shape_cross_modal)
    relative_shape_cross_modal_mean = shape_cross_modal_mean/np.mean(all_mods -base)
    result = {'SHAPE': shape_cross_modal_mean, 'SHAPE_rel':relative_shape_cross_modal_mean}
    return result

if __name__ == "__main__":
    number_modalities = 2
    path = '/home/lw754/masterproject/cross-modal-interaction/results/uniqueness0_epochs_150_concat_True/seed_1/best.csv'
    df = pd.read_csv(path)
    df_coalitions = df.iloc[:, :(2**number_modalities)]
    shape_cross_modal = cross_modal_SHAPE(df_coalitions,number_modalities)
    print(shape_cross_modal)
