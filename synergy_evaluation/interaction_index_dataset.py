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

def calc_interaction_metric_over_dataset(explaination_path):
    shaply_values, interactions = load_data(explaination_path)
    interactions_dataset_abs = np.sum(np.abs(np.mean(interactions, axis=0)))
    shaply_values_dataset_abs = np.sum(np.abs(np.mean(shaply_values, axis=0)))

    interactions_dataset = np.sum(np.mean(interactions, axis=0))
    shaply_values_dataset = np.sum(np.mean(shaply_values, axis=0))

    interactions_rel_abs = interactions_dataset_abs / (interactions_dataset_abs+shaply_values_dataset_abs)
    interaction_rel2 = interactions_dataset / (interactions_dataset +shaply_values_dataset )
    return interaction_rel2, interactions_rel_abs

def calc_interaction_dataset_ensamble(results_path,dataset_label,epochs=200,concat='early',seeds = [1,42 ,113 ],settings=['uniqueness0', 'synergy',  'uniqueness1','redundancy']):
    results_path = Path(results_path)
    for setting in settings:
        run_name = f'{dataset_label}_{setting}_epochs_{epochs}_concat_{concat}'
        results_rel = []
        results_abs = []
        for seed in seeds:
            run_name_seed = Path(run_name)/f'seed_{seed}'
            run_path = results_path/run_name_seed
            interaction_rel2, interactions_rel_abs = calc_interaction_metric_over_dataset(run_path)
            results_rel.append(interaction_rel2)
            results_abs.append(interactions_rel_abs)
        print(f'Result for {results_path/run_name}')
        print(f'Mean:{np.mean(results_rel)}, STD: {np.std(results_rel)} ABSOLUTE: Mean:{np.mean(results_abs)}, STD: {np.std(results_abs)}')



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate interaction dataset ensemble.")
    parser.add_argument('--results_path', type=str,
                        default='/home/lw754/masterproject/cross-modal-interaction/results/', help='Path for results')
    parser.add_argument('--dataset_label',nargs='+', type=str, default=['single-cell','single-cell'], help='Label for dataset, VEC2XOR_org')
    parser.add_argument('--settings', nargs='+', default=['all'],
                        help='List of settings')
    parser.add_argument('--seeds', nargs='+', type=int, default=[1,42,113], help='List of seeds')
    parser.add_argument('--concat', nargs='+', type=str, default=['intermediate','late'], help='Concatenation strategy')
    parser.add_argument('--epochs', nargs='+', type=int, default=[200,200], help='Number of epochs')
    args = parser.parse_args()

    for conc, dataset_lab,epoch in zip(args.concat,args.dataset_label,args.epochs ):
        print('##################################################################')
        print(f'################ {conc} + { dataset_lab}  ##########################')
        calc_interaction_dataset_ensamble(args.results_path, dataset_lab, epoch, conc, args.seeds,
                                          args.settings)
