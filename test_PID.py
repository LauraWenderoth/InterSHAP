import wandb
import pickle
import torch
from pathlib import  Path
import pandas as pd
from utils.utils import save_data
from synergy_evaluation.PID import clustering, convert_data_to_distribution, get_measure

if __name__ == "__main__":
    # TODO add argparse
    use_wandb = True
    results = dict()
    data_root_path = Path('/home/lw754/masterproject/PID/synthetic_data/')
    results_path = Path('/home/lw754/masterproject/cross-modal-interaction/results')
    label = 'VEC2_'  # 'OR_' XOR_ VEC2_
    settings = ['redundancy', 'synergy', 'uniqueness0', 'uniqueness1'] #['syn_mix10-5-0','syn_mix9-10-0','syn_mix8-10-0','syn_mix7-10-0','syn_mix6-10-0', 'syn_mix5-10-0','syn_mix4-10-0','syn_mix3-10-0','syn_mix2-10-0','syn_mix1-10-0',]#['syn_mix9', 'syn_mix92' ]#['syn_mix2', 'syn_mix5', 'syn_mix10']#['redundancy', 'synergy', 'uniqueness0', 'uniqueness1'] #['redundancy', 'synergy', 'uniqueness0', 'uniqueness1', 'mix1', 'mix2', 'mix3', 'mix4', 'mix5', 'mix6'] #['redundancy', 'synergy', 'uniqueness0', 'uniqueness1'] #
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    if use_wandb:
        wandb.init(project="masterthesis", name=f"{label}DATA", group="shaply_interaction_index")
    for setting in settings:

        data_path = data_root_path/ f'{label}DATA_{setting}.pickle'
        dataset = pd.read_pickle(data_path)
        n_components = 2
        data_cluster = dict()
        for split in ['valid', 'test']:
            data_cluster[split] = dict()
            data = dataset[split]
            kmeans_0, data_0 = clustering(data['0'], pca=True, n_components=n_components, n_clusters=20)
            data_cluster[split]['0'] = kmeans_0.reshape(-1,1)
            kmeans_1, data_1 = clustering(data['1'], pca=True, n_components=n_components, n_clusters=20)
            data_cluster[split]['1'] = kmeans_1.reshape(-1,1)
            data_cluster[split]['label'] = data['label']
        with open(data_root_path/f'{label}DATA_{setting}_cluster.pickle', 'wb') as f:
            pickle.dump(data_cluster, f)

    for setting in settings:
        with open(data_root_path/f'{label}DATA_{setting}_cluster.pickle', 'rb') as f:
            dataset = pickle.load(f)
        print(setting)
        data = (dataset['test']['0'], dataset['test']['1'], dataset['test']['label'])
        P, maps = convert_data_to_distribution(*data)
        result = get_measure(P)
        results[setting] = result
        print()

    save_path = results_path / f'{label}DATA_PID_results.pickle'
    save_data(results, save_path)

    if use_wandb:
        wandb.log(results)
        df = pd.DataFrame.from_dict(results, orient='index')
        df.reset_index(drop=False, inplace=True)
        my_table = wandb.Table(dataframe=df)
        wandb.log({"PID results": my_table})

        wandb.finish()
