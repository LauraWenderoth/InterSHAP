import shap
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from shap import  Explanation
from pathlib import Path
import math
import argparse

def get_shaply_interactions(data,n_samples,n_modalities):
    shaply_values = data[:, :n_modalities]
    interactions = data[:, n_modalities:]
    return shaply_values,interactions

def split_values(data,df):
    data = data.values
    n_modalities = math.sqrt(data.shape[1])
    assert n_modalities % 1 == 0, 'Interactions have wrong format!'
    n_modalities = int(n_modalities)

    base_values = list(df[str(np.zeros(n_modalities,dtype=int))])
    n_samples = len(base_values)
    shaply_values, interactions = get_shaply_interactions(data, n_samples, n_modalities)
    return shaply_values, interactions

def load_data(explaination_path):
    explaination_path = Path(explaination_path)
    file_name_base_value = 'best.csv'
    file_name_interactions = 'interaction_index_best_with_ii.csv'
    data = pd.read_csv(explaination_path/file_name_interactions)
    data = data.values
    df = pd.read_csv(explaination_path/file_name_base_value)

    n_modalities = math.sqrt(data.shape[1])
    assert n_modalities % 1 == 0, 'Interactions have wrong format!'
    n_modalities = int(n_modalities)

    base_values = list(df[str(np.zeros(n_modalities,dtype=int))])
    n_samples = len(base_values)
    shaply_values, interactions = get_shaply_interactions(data, n_samples, n_modalities)
    return shaply_values, interactions

def calc_interaction_metric_over_dataset(shaply_values, interactions, local_return = False):
    interactions_dataset_abs = np.sum(np.abs(np.mean(interactions, axis=0)))
    shaply_values_dataset_abs = np.sum(np.abs(np.mean(shaply_values, axis=0)))

    local_interactions = np.sum(np.abs(interactions),axis=1)
    shaply_values_local = np.sum(np.abs(shaply_values),axis=1)
    local = local_interactions / (shaply_values_local + local_interactions)

    interactions_rel_abs = interactions_dataset_abs / (interactions_dataset_abs+shaply_values_dataset_abs)
    if local_return:
        return interactions_rel_abs, local
    else:
        return interactions_rel_abs


def calc_interaction_dataset_ensamble(results_path,dataset_label,epochs=200,concat='early',seeds = [1,42 ,113 ],settings=['uniqueness0', 'synergy',  'uniqueness1','redundancy']):
    results_path = Path(results_path)
    for setting in settings:
        local = []
        run_name = f'{dataset_label}{setting}_epochs_{epochs}_concat_{concat}'
        results_abs = []
        for seed in seeds:
            run_name_seed = Path(run_name)/f'seed_{seed}'
            run_path = results_path/run_name_seed
            shaply_values, interactions = load_data(run_path)
            interactions_rel_abs, local_scores = calc_interaction_metric_over_dataset(shaply_values, interactions, local_return = True)
            results_abs.append(interactions_rel_abs)
            local.append(local_scores)
        print(f'Result for {results_path/run_name}')
        local_mean = np.mean(local,axis=0)
        print(f'LOCAL Mean:{np.mean(local_mean)}, STD: {np.std(local_mean)} GLOBAL: Mean:{np.mean(results_abs)}, STD: {np.std(results_abs)}')



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate interaction dataset ensemble.")
    parser.add_argument('--results_path', type=str,
                        default='/home/lw754/masterproject/cross-modal-interaction/results/', help='Path for results')
    parser.add_argument('--dataset_label',nargs='+', type=str, default=['VEC2XOR_'], help='Label for dataset, VEC2XOR_org, single-cell')
    parser.add_argument('--settings', nargs='+', default=['synergy','uniqueness0','uniqueness1','redundancy','random'],
                        help='List of settings')
    parser.add_argument('--seeds', nargs='+', type=int, default=[1,42,113], help='List of seeds')
    parser.add_argument('--concat', nargs='+', type=str, default=['early'], help='Concatenation strategy function')
    parser.add_argument('--epochs', nargs='+', type=int, default=[200], help='Number of epochs')
    args = parser.parse_args()

    for conc, dataset_lab,epoch in zip(args.concat,args.dataset_label,args.epochs ):
        print('##################################################################')
        print(f'################ {conc} + { dataset_lab}  ##########################')
        calc_interaction_dataset_ensamble(args.results_path, dataset_lab, epoch, conc, args.seeds,
                                          args.settings)
