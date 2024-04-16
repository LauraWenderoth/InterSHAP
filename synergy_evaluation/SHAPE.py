import pandas as pd
import numpy as np
from collections import Counter
from utils.utils import eval_model
from utils.dataset import MMDataset
from torch.utils.data import  DataLoader
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

def org_SHAPE(dataset,model,batch_size,device,metric='f1_macro'):
    X,y = dataset.get_data()
    concat = 'early' if dataset.get_concat() else False

    # 1. calulcate base values using dataset:
    counts = Counter(y)
    majority_number, majority_count = counts.most_common(1)[0]
    base_value = (majority_count / len(y))

    # 2. calculate F1 for all
    dataloader = DataLoader(dataset, batch_size=batch_size,shuffle=False)
    results,_ = eval_model(dataloader, model,device,title='')
    all_mods_value = results[f'_{metric}']

    # 3. calulate for each mod with maskin 0
    mod_contributions = []
    X_zero = np.zeros_like(X)
    for i in range(len(X)):
        X_mask = X_zero.copy()
        X_mask[i] = X[i]
        dataset_i = MMDataset(X_mask,y, concat=concat, device=device)
        dataloader = DataLoader(dataset_i, batch_size=batch_size, shuffle=False)
        results, _ = eval_model(dataloader, model, device, title='')
        mod_contribution = results[f'_{metric}'] - base_value
        mod_contributions.append(mod_contribution)

    #4. calc result
    shape_cross_modal = all_mods_value - base_value - np.sum(mod_contributions)
    return {'shape_org':shape_cross_modal}

if __name__ == "__main__":
    number_modalities = 2
    path = '/home/lw754/masterproject/cross-modal-interaction/results/uniqueness0_epochs_150_concat_True/seed_1/best.csv'
    df = pd.read_csv(path)
    df_coalitions = df.iloc[:, :(2**number_modalities)]
    shape_cross_modal = cross_modal_SHAPE(df_coalitions,number_modalities)
    print(shape_cross_modal)
