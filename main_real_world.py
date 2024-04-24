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

def parse_args():
    parser = argparse.ArgumentParser(description='Argument Parser for your settings')
    # TODO boolean flags with action

    # Add arguments
    parser.add_argument('--train_model', default=True, help='Whether to train the model or just eval') #action='store_false'
    parser.add_argument('--seeds', nargs='+', type=int, default=[1], help='List of seed values, 113 ')
    parser.add_argument('--use_wandb', default=True, help='Whether to use wandb or not')
    parser.add_argument('--batch_size', type=int, default=5000, help='Batch size for training')
    parser.add_argument('--n_samples_for_interaction', type=int, default=3000, help='Number of samples for interaction')
    parser.add_argument('--epochs', type=int, default=0, help='Number of epochs for training')
    parser.add_argument('--test_inverval', type=int, default=10, help='Eval interval during traing (int = number of epochs)')
    parser.add_argument('--settings', nargs='+', type=str, default=[  'uniqueness0', 'uniqueness1','redundancy', 'synergy',], #['redundancy','synergy', 'uniqueness0', 'uniqueness1','syn_mix5-10-0','syn_mix10-5-0'],#'uniqueness0', 'uniqueness1','syn_mix5-10-0','syn_mix10-5-0 , #['syn_mix9-10-0','syn_mix8-10-0','syn_mix7-10-0','syn_mix6-10-0', 'syn_mix5-10-0','syn_mix4-10-0','syn_mix3-10-0','syn_mix2-10-0','syn_mix1-10-0'],#['syn_mix9', 'syn_mix92' ],#['mix1', 'mix2', 'mix3', 'mix4','mix5', 'mix6'],#['redundancy', 'synergy', 'uniqueness0', 'uniqueness1'], ['syn_mix2', 'syn_mix5','syn_mix10' ]
                        choices=['redundancy', 'synergy', 'uniqueness0', 'uniqueness1', 'uniqueness2','uniqueness3','uniqueness4','mix1', 'mix2', 'mix3', 'mix4', 'mix5', 'mix6','syn_mix5-10-0','syn_mix10-5-0'], help='List of settings')
    parser.add_argument('--concat', default = 'function', choices='early, intermediate, late', help='early, intermediate, late function')
    parser.add_argument('--label', type=str, default='VEC2XOR_org_', help='Can choose "" as PID synthetic data or VEC2XOR_org_ "OR_" "XOR_" "VEC3_" "VEC2_"')
    parser.add_argument('--device', type=str, default='cuda:0' if torch.cuda.is_available() else 'cpu', help='Device for computation')
    parser.add_argument('--root_save_path', type=str, default='/home/lw754/masterproject/cross-modal-interaction/results/', help='Root save path')
    parser.add_argument('--data_path', type=str,
                        default='/home/lw754/masterproject/synthetic_data/', help='Root save path')
    parser.add_argument('--wandb_name', type=str, default='MA Final Results',
                        help='Can choose "" as PID synthetic data or "OR_" "XOR_" "VEC3_" "VEC2_"')

    parser.add_argument('--train_uni_model',  action='store_true', help='Whether to train the model or just eval')
    parser.add_argument('--synergy_eval_epoch', default=False, help='Whether to eval synergy metrics during training')
    parser.add_argument('--synergy_metrics', nargs='+', type=str, default=['SHAPE','SRI','Interaction','EMAP'], help="List of seed values ['SHAPE','SRI','Interaction','PID','EMAP'] ")
    parser.add_argument('--save_results', default=True, help='Whether to locally save results or not')

    args = parser.parse_args()
    return args

def print_latex_results_table(stats_dict):
    # Generate LaTeX table
    latex_table = "\\begin{table}[htbp]\n"
    latex_table += "\\centering\n"
    latex_table += "\\begin{tabular}{|c|" + "c|" * len(stats_dict) + "}\n"
    latex_table += "\\hline\n"
    latex_table += "Metric "
    for metric in stats_dict.keys():
        metric_name = metric.replace('_', ' ').capitalize()
        latex_table += f"& {metric_name} "
    latex_table += "\\\\\n"
    latex_table += "\\hline\n"
    latex_table += "Mean $\\pm$ Std Dev & "
    for metric, values in stats_dict.items():
        mean_percent = f"{values['mean'] * 100:.2f}"
        std_percent = f"{values['std_dev'] * 100:.2f}"
        latex_table += f"{mean_percent}\\ $\\pm$ {std_percent}\\ & "
    latex_table = latex_table[:-2]  # Remove the trailing '& '
    latex_table += "\\\\\n"
    latex_table += "\\hline\n"
    latex_table += "\\end{tabular}\n"
    latex_table += "\\caption{Statistics}\n"
    latex_table += "\\label{tab:stats}\n"
    latex_table += "\\end{table}"

    print(latex_table)



def load_model(concat,modality_shape=None,output_size=None,setting=None):
    if concat=='early':
        model =EarlyFusionFeedForwardNN(np.sum(modality_shape), output_size)
    elif concat =='intermediate':
        model = IntermediateFusionFeedForwardNN(modality_shape, output_size)
    elif concat == 'late':
        model = LateFusionFeedForwardNN(modality_shape, output_size)
    elif concat == 'function':
        model = OrginalFunctionXOR(setting)
    else:
        print('No valid model selected')
        model = None
    return model


def train(device,train_dataloader, val_dataloader, save_path, use_wandb, experiment_name = 'redundancy', num_epochs=100, test_inverval = 10, modality_shape=None, output_size = 2, lr = 1e-4, step_size = 500 ,concat=True,seed=42,synergy_eval_epoch=False,dataset=None,args=None):

    # Set random seed for reproducibility
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


    model = load_model(concat,modality_shape,output_size)


    model = model.to(device)

    print("Model device:", next(model.parameters()).device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=0.2)
    print("\nMulti-modal Training")
    for epoch in tqdm(range(num_epochs ), desc="Epochs"):
        model, loss = train_epoch(train_dataloader, model, optimizer,device=device)
        scheduler.step()
        if epoch % test_inverval == 0:
            result, y_pred_test = eval_model(val_dataloader, model, device, use_wandb=args.use_wandb,
                                             return_predictions=True)
            result["epoch"] = epoch
            result['training_loss'] = loss
            if synergy_eval_epoch:
                val_dataset = dataset['valid']
                test_dataset = dataset['test']
                synergy_result = eval_synergy(model, val_dataset, test_dataset, device=device,
                                               eval_metrics=args.synergy_metrics, batch_size=args.batch_size,
                                               save_path=save_path, use_wandb=args.use_wandb,
                                               n_samples_for_interaction=args.n_samples_for_interaction)
                result.update(synergy_result)




            if use_wandb:
                wandb.log(result)
    save_checkpoint(model, save_path, filename=f"{experiment_name}.pt")
    return model

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
        experiment_name_run = f'{args.label}{setting}_epochs_{args.epochs}_concat_{args.concat}'
        save_path_run = root_save_path/experiment_name_run
        if args.use_wandb:
            wandb.init(project=args.wandb_name, name=f"{experiment_name_run}") #Final_MA
            wandb.config.update(
                {'batch_size': args.batch_size, 'n_samples_for_interaction': args.n_samples_for_interaction, 'epochs': args.epochs,
                 'concat': args.concat, 'setting': setting, 'data_path': data_path, 'save_path': save_path_run})


        for seed in args.seeds:
            run_results = {'seed': seed}
            experiment_name = f'{experiment_name_run}/seed_{seed}'
            save_path = root_save_path / experiment_name
            with open(data_path, 'rb') as f:
                data = pickle.load(f)

            if args.train_uni_model:
                train_unimodal(data, root_save_path, experiment_name_run, device=args.device, seed=seed, num_epochs
                =args.epochs, use_wandb=args.use_wandb, batch_size=args.batch_size, test_inverval=args.test_inverval, n_samples_for_eval=args.n_samples_for_interaction)

            # Datasets
            train_data = MMDataset(data['train'], concat=args.concat, device=args.device)
            val_dataset = MMDataset(data['valid'], concat=args.concat, device=args.device)
            test_dataset = MMDataset(data['test'],concat=args.concat,device=args.device,length=args.n_samples_for_interaction)
            # Dataloader
            train_loader = DataLoader(train_data,batch_size=args.batch_size,shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=args.batch_size,shuffle=False)
            test_loader = DataLoader(test_dataset, batch_size=args.batch_size,shuffle=False)

            if args.train_model:
                model = train(args.device, train_loader, val_loader, save_path, use_wandb=args.use_wandb, num_epochs=args.epochs,
                              test_inverval=args.test_inverval, modality_shape = train_data.get_modality_shapes(), output_size=train_data.get_number_classes(),
                              experiment_name=setting,concat=args.concat,seed=seed, synergy_eval_epoch=args.synergy_eval_epoch,dataset={'valid':val_dataset,'test':test_dataset },args=args)
            else:
                model_weights = save_path / f'{setting}.pt'
                model = load_model(args.concat,train_data.get_modality_shapes(),train_data.get_number_classes(),setting)
                if args.concat != 'function':
                    model = load_checkpoint(model,model_weights)
                    model.to(args.device)




            print(f'\nSynergy metric evaluation seed {seed}')
            synergy_results = eval_synergy(model, val_dataset, test_dataset, device = args.device,
                         eval_metrics=args.synergy_metrics, batch_size=args.batch_size,
                         save_path=save_path, use_wandb=args.use_wandb, n_samples_for_interaction=args.n_samples_for_interaction)
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
        print_latex_results_table(stats)

        if args.use_wandb:
            wandb.log(stats)
            wandb.finish()

        if args.save_results:
            df = pd.DataFrame.from_dict(stats, orient='index')
            data_df = pd.DataFrame(final_results)
            df.to_csv(save_path_run/'mean_std_stats.csv')
            data_df.to_csv(save_path_run/'results_all_seeds.csv')



