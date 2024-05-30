import pickle
import wandb
import pandas as pd

import random
import numpy as np
from pathlib import Path

import torch
from tqdm import tqdm
from torch.utils.data import  DataLoader
from utils.dataset import MMDataset
from utils.models import LateFusionFeedForwardNN,EarlyFusionFeedForwardNN,IntermediateFusionFeedForwardNN, OrginalFunctionXOR
from utils.unimodal import train_unimodal
from utils.utils import  eval_model, save_checkpoint, train_epoch, load_checkpoint
from synergy_evaluation.evaluation import eval_synergy
import argparse
import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.getcwd()), "MultiBench"))

def parse_args():
    parser = argparse.ArgumentParser(description='Argument Parser for your settings')
    # TODO boolean flags with action

    # Add arguments

    parser.add_argument('--number_of_classes', type=int, default=2, help='')
    parser.add_argument('--use_wandb', default=True, help='Whether to use wandb or not')
    parser.add_argument('--batch_size', type=int, default=400, help='Batch size for training')
    parser.add_argument('--n_samples_for_interaction', type=int, default=100, help='Number of samples for interaction')


    parser.add_argument('--settings', nargs='+', type=str, default=[ 'task7_2D'], #['redundancy','synergy', 'uniqueness0', 'uniqueness1','syn_mix5-10-0','syn_mix10-5-0'],#'uniqueness0', 'uniqueness1','syn_mix5-10-0','syn_mix10-5-0 , #['syn_mix9-10-0','syn_mix8-10-0','syn_mix7-10-0','syn_mix6-10-0', 'syn_mix5-10-0','syn_mix4-10-0','syn_mix3-10-0','syn_mix2-10-0','syn_mix1-10-0'],#['syn_mix9', 'syn_mix92' ],#['mix1', 'mix2', 'mix3', 'mix4','mix5', 'mix6'],#['redundancy', 'synergy', 'uniqueness0', 'uniqueness1'], ['syn_mix2', 'syn_mix5','syn_mix10' ]
                        choices=['taskmortality','task7','task1','taskmortality_2D','task7_2D','task1_2D'], help='List of settings')
    parser.add_argument('--concat', default = '2D', help='If you want to concat before input to model do early else always late')
    parser.add_argument('--label', type=str, default='mimic_', help='mimic_')
    parser.add_argument('--device', type=str, default='cuda:0' if torch.cuda.is_available() else 'cpu', help='Device for computation')
    parser.add_argument('--root_save_path', type=str, default='/home/lw754/masterproject/cross-modal-interaction/results/', help='Root save path')
    parser.add_argument('--data_path', type=str,
                        default='/home/lw754/masterproject/real_world_data/', help='Root save path to where the data is stored')
    parser.add_argument('--pretrained_paths',nargs='+', type=str,
                        default=['/home/lw754/masterproject/results_real_world/baselinge_task7_seed1.pt' , '/home/lw754/masterproject/results_real_world/baselinge_task7_seed42.pt' , '/home/lw754/masterproject/results_real_world/baselinge_task7_seed113.pt'],
                        help='Path to trained model')
    parser.add_argument('--wandb_name', type=str, default='MA Real World Results',
                        help='')
    parser.add_argument('--synergy_metrics', nargs='+', type=str, default= ['SHAPE','SRI','Interaction','PID','EMAP'], help="List of seed values ['SHAPE','SRI','Interaction','PID','EMAP'] ")
    parser.add_argument('--save_results', default=True, help='Whether to locally save results or not')

    args = parser.parse_args()
    return args








if __name__ == "__main__":
    args = parse_args()
    root_save_path = Path(args.root_save_path)
    print(f'Start experiments with setting {args.settings} on dataset {args.label}DATA'
          f'\nAll results will be saved to: {root_save_path}'
          f'\nweight and biases is turned on: {args.use_wandb}')
    for setting in args.settings:
        print('################################################')
        print(f'Start setting {setting}')
        final_results = dict()
        data_path = Path(args.data_path) /f'{args.label}DATA_{setting}.pickle'
        model_name = str(Path( args.pretrained_paths[0]).name).split('_')[0]
        experiment_name_run = f'{args.label}{setting}_{model_name}'
        save_path_run = root_save_path/experiment_name_run
        if args.use_wandb:
            wandb.init(project=args.wandb_name, name=f"{experiment_name_run}") #Final_MA
            wandb.config.update(
                {'batch_size': args.batch_size, 'n_samples_for_interaction': args.n_samples_for_interaction,
                  'setting': setting, 'data_path': data_path, 'save_path': save_path_run})


        for path in args.pretrained_paths:
            run_results = {}
            path = Path(path)
            experiment_name = f'{experiment_name_run}/{path.name}'
            save_path = root_save_path / experiment_name
            with open(data_path, 'rb') as f:
                data = pickle.load(f)


            # Datasets
            train_data = MMDataset(data['train'], concat=args.concat, device=args.device)
            val_dataset = MMDataset(data['valid'], concat=args.concat, device=args.device)
            test_dataset = MMDataset(data['test'],concat=args.concat,device=args.device,length=args.n_samples_for_interaction)
            # Dataloader
            train_loader = DataLoader(train_data,batch_size=args.batch_size,shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=args.batch_size,shuffle=False)
            test_loader = DataLoader(test_dataset, batch_size=args.batch_size,shuffle=False)


            model = torch.load(str(path))
            model.to(args.device)



            print(f'\nSynergy metric evaluation seed {path.name}')
            synergy_results = eval_synergy(model, val_dataset, test_dataset, device = args.device,
                         eval_metrics=args.synergy_metrics, batch_size=args.batch_size,
                         save_path=save_path, use_wandb=args.use_wandb, n_samples_for_interaction=args.n_samples_for_interaction,
                                           classes = args.number_of_classes)
            run_results.update(synergy_results)

            for key, score in run_results.items():
                if key not in final_results:
                    final_results[key] = []
                final_results[key].append(score)

        stats = {}
        for key, values in final_results.items():
            stats[key] = {
                'mean': np.mean(values),
                'std_dev': np.std(values) if len(values) > 1 else 0
            }
        print(stats)


        if args.use_wandb:
            wandb.log(stats)
            wandb.finish()

        if args.save_results:
            df = pd.DataFrame.from_dict(stats, orient='index')
            data_df = pd.DataFrame(final_results)
            df.to_csv(save_path_run/'mean_std_stats.csv')
            data_df.to_csv(save_path_run/'results_all_seeds.csv')



