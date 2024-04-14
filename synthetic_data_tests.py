import pickle
import wandb
import pandas as pd

import random
import numpy as np
from pathlib import Path

import torch
from tqdm import tqdm
from torch.utils.data import  DataLoader
from dataloader import MMDataset
from models import LateFusionFeedForwardNN,EarlyFusionFeedForwardNN,IntermediateFusionFeedForwardNN
from utils.utils import  eval_model, save_checkpoint
from synergy_evaluation.evaluation import eval_synergy
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Argument Parser for your settings')

    # Add arguments
    parser.add_argument('--train_model', default=True, help='Whether to train the model or just eval')
    parser.add_argument('--seeds', nargs='+', type=int, default=[1, 42,113 ], help='List of seed values')
    parser.add_argument('--use_wandb', default=False, help='Whether to use wandb or not')
    parser.add_argument('--batch_size', type=int, default=5000, help='Batch size for training')
    parser.add_argument('--n_samples_for_interaction', type=int, default=100, help='Number of samples for interaction')
    parser.add_argument('--epochs', type=int, default=1, help='Number of epochs for training')
    parser.add_argument('--n_modality', type=int, default=2, help='Number of modalities')
    parser.add_argument('--settings', nargs='+', type=str, default= [ 'redundancy', 'synergy' ],#'uniqueness0', 'uniqueness1','syn_mix5-10-0','syn_mix10-5-0 , #['syn_mix9-10-0','syn_mix8-10-0','syn_mix7-10-0','syn_mix6-10-0', 'syn_mix5-10-0','syn_mix4-10-0','syn_mix3-10-0','syn_mix2-10-0','syn_mix1-10-0'],#['syn_mix9', 'syn_mix92' ],#['mix1', 'mix2', 'mix3', 'mix4','mix5', 'mix6'],#['redundancy', 'synergy', 'uniqueness0', 'uniqueness1'], ['syn_mix2', 'syn_mix5','syn_mix10' ]
                        choices=['redundancy', 'synergy', 'uniqueness0', 'uniqueness1', 'mix1', 'mix2', 'mix3', 'mix4', 'mix5', 'mix6'], help='List of settings')
    parser.add_argument('--concat', default = 'late', choices='early, intermediate, late', help='early, intermediate, late')
    parser.add_argument('--label', type=str, default='', help='Can choose "" as PID synthetic data or "OR_" "XOR_" "VEC3_" "VEC2_"')
    parser.add_argument('--device', type=str, default='cuda:0' if torch.cuda.is_available() else 'cpu', help='Device for computation')
    parser.add_argument('--root_save_path', type=str, default='/home/lw754/masterproject/cross-modal-interaction/results/', help='Root save path')
    parser.add_argument('--cross_modal_scores_during_training', default=False, help='early, intermediate, late')
    parser.add_argument('--train_uni_model', default=False, help='Whether to train the model or just eval')
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



def train_epoch(train_dataloader, model, optimizer,device):
    model.train()
    epoch_loss = 0
    criterion = torch.nn.CrossEntropyLoss(reduction='none')
    for i, data in enumerate(train_dataloader):
        features, labels = data
        labels = labels.to(device)
        try:
            features = features.to(device)
        except:
            features = [feat.to(device) for feat in features]
        optimizer.zero_grad()
        logits = model(features)
        loss = torch.mean(criterion(logits, labels))
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    return model, epoch_loss


def train(device,train_dataloader, val_dataloader, save_path, use_wandb, experiment_name = 'redundancy', num_epochs=100, test_inverval = 10, input_size = 10, output_size = 2, lr = 1e-4, step_size = 500 ,concat=True,seed=42):

    # Set random seed for reproducibility
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


    if concat=='early':
        model =EarlyFusionFeedForwardNN(input_size, output_size)
    elif concat =='intermediate':
        model = IntermediateFusionFeedForwardNN(input_size, output_size)
    elif concat == 'late':
        model = LateFusionFeedForwardNN(input_size, output_size)
    else:
        print('No valid model selected')
        model = None


    #model = TabularCrossmodalMultiheadAttention( prediction_task='binary', data_dims=[64,64,None], multiclass_dimensions=None)
    model = model.to(device)

    print("Model device:", next(model.parameters()).device)
    best_model = model
    best_f1_macro = 0
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=0.2)
    print("\nTraining")
    for epoch in tqdm(range(1,num_epochs + 1), desc="Epochs"):
        model, loss = train_epoch(train_dataloader, model, optimizer,device=device)
        if use_wandb:
            wandb.log({"training_loss": loss})
        scheduler.step()
        if epoch % test_inverval == 0:
            result,cfm = eval_model(val_dataloader, model,device)
            result["epoch"] = epoch
            if use_wandb:
                wandb.log(result)

            if best_f1_macro < result['Val_f1_macro']:
                best_f1_macro = result['Val_f1_macro']
                best_model = model
    save_checkpoint(best_model, save_path, filename=f"{experiment_name}.pt")
    return best_model

if __name__ == "__main__":
    args = parse_args()
    # args.train_model = False
    root_save_path = Path(args.root_save_path)
    print(f'Start experiments with setting {args.settings} on dataset {args.label}DATA'
          f'\nAll results will be saved to: {root_save_path}'
          f'\nweight and biases is turned on: {args.use_wandb}')
    for setting in args.settings:
        print('################################################')
        print(f'Start setting {setting}')

        data_path = f'/home/lw754/masterproject/PID/synthetic_data/{args.label}DATA_{setting}.pickle'
        experiment_name_run = f'{args.label}{setting}_epochs_{args.epochs}_concat_{args.concat}'
        save_path_run = root_save_path/experiment_name_run
        if args.use_wandb:
            wandb.init(project="masterthesis", name=f"{experiment_name_run}", group="shaply_interaction_index")
            wandb.config.update(
                {'batch_size': args.batch_size, 'n_samples_for_interaction': args.n_samples_for_interaction, 'epochs': args.epochs,
                 'concat': args.concat, 'setting': setting, 'data_path': data_path, 'save_path': save_path_run})


        for seed in args.seeds:
            run_results = {'seed': seed}
            print(f'Seed {seed}')
            if args.train_uni_model:

                for i_modality in range(args.n_modality):
                    print(f'Start uni-modal with modality {i_modality}')
                    experiment_name = f'{experiment_name_run}/unimodal_{i_modality}'
                    save_path = root_save_path / experiment_name
                    with open(data_path, 'rb') as f:
                        data = pickle.load(f)
                    # visualize_umap(data, n_modality, args.label, setting)
                    test_data = data['test'][str(i_modality)][:args.n_samples_for_interaction]
                    val_data = data['valid'][str(i_modality)]
                    train_data = data['train'][str(i_modality)]
                    train_label = np.squeeze(data['train']['label'])

                    test_dataset = MMDataset(test_data,
                                             np.squeeze(data['test']['label'][:args.n_samples_for_interaction]),
                                             device=args.device, concat='early')
                    val_dataset = MMDataset(val_data, np.squeeze(data['valid']['label']), concat='early',
                                            device=args.device)
                    train_loader = DataLoader(MMDataset(train_data, train_label, concat='early', device=args.device),
                                              batch_size=args.batch_size,
                                              shuffle=True, num_workers=0)
                    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                                            shuffle=True, num_workers=0)

                    input_size = train_data.shape[1]

                    model = train(args.device, train_loader, val_loader, save_path, use_wandb=args.use_wandb,
                                  num_epochs=args.epochs,
                                  test_inverval=10, input_size=input_size, output_size=len(np.unique(train_label)),
                                  experiment_name=setting, concat='early', seed=seed)
                    if args.use_wandb:
                        print(f'\nEvaluation Unimodal Seed {seed}')

                        test_loader = DataLoader(test_dataset, batch_size=args.batch_size,
                                                 shuffle=True, num_workers=0)
                        test_results, _ = eval_model(test_loader, model, args.device, title=f'Test_unimodal_{i_modality}', use_wandb=True)
                        run_results.update(test_results)

            print('Start multi-modal')
            experiment_name = f'{experiment_name_run}/seed_{seed}'
            save_path = root_save_path / experiment_name
            with open(data_path, 'rb') as f:
                data = pickle.load(f)
            # visualize_umap(data, n_modality, args.label, setting)
            if args.concat == 'early':
                test_data = np.concatenate(
                    (data['test']['0'][:args.n_samples_for_interaction], data['test']['1'][:args.n_samples_for_interaction]),
                    axis=1)
                val_data = np.concatenate((data['valid']['0'], data['valid']['1']), axis=1)
                train_data = np.concatenate((data['train']['0'], data['train']['1']), axis=1)
                input_size = train_data.shape[1]
            else:
                test_data = [data['test']['0'][:args.n_samples_for_interaction],
                             data['test']['1'][:args.n_samples_for_interaction]]
                val_data = [data['valid']['0'], data['valid']['1']]
                train_data = [data['train']['0'], data['train']['1']]
                input_size = [mod.shape[1] for mod in train_data]
            train_label = np.squeeze(data['train']['label'])

            test_dataset = MMDataset(test_data, np.squeeze(data['test']['label'][:args.n_samples_for_interaction]),
                                     device=args.device, concat=args.concat)
            val_dataset = MMDataset(val_data, np.squeeze(data['valid']['label']), concat=args.concat, device=args.device)
            train_loader = DataLoader(MMDataset(train_data, train_label, concat=args.concat, device=args.device),
                                      batch_size=args.batch_size,
                                      shuffle=True, num_workers=0)
            val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                                    shuffle=True, num_workers=0)
            if args.train_model:
                model = train(args.device, train_loader, val_loader, save_path, use_wandb=args.use_wandb, num_epochs=args.epochs,
                              test_inverval=10, input_size=input_size, output_size=len(np.unique(train_label)),
                              experiment_name=setting,concat=args.concat,seed=seed)
            else:
                model_weights = save_path / f'{setting}.pt'
                model = torch.load(model_weights)
                model.to(args.device)
            if args.concat == 'early':
                #TODO wofür brauche ich das??
                input_size = [input_size // 2, input_size // 2]


            if args.use_wandb:
                print(f'\nEvaluation Seed {seed}')

                test_loader = DataLoader(test_dataset, batch_size=args.batch_size,
                                         shuffle=True, num_workers=0)
                test_results, _,y_pred_test = eval_model(test_loader, model, args.device, title='Test', use_wandb=True,return_predictions=True)
                run_results.update(test_results)
                test_data_model_pred = {'0':data['test']['0'][:args.n_samples_for_interaction], '1':
                             data['test']['1'][:args.n_samples_for_interaction],'label':y_pred_test}
                final_results = eval_synergy(model, val_dataset, test_dataset, test_data_model_pred, run_results, save_path, input_size,
                             n_modalites=args.n_modality, device=args.device, test_results=test_results, concat=args.concat, use_wandb=args.use_wandb,
                             n_samples_for_interaction=args.n_samples_for_interaction)

        if args.use_wandb:
            # calculate mean and std for each eval metric
            stats = {}
            for key, values in final_results.items():
                stats[key] = {
                    'mean': np.mean(values),
                    'std_dev': np.std(values) if len(values) > 1 else 0
                    # Avoiding division by zero for single-value lists
                }
            print(stats)
            print_latex_results_table(stats)
            wandb.log(stats)

            # save results
            df = pd.DataFrame.from_dict(stats, orient='index')
            data_df = pd.DataFrame(final_results)
            df.to_csv(save_path_run/'mean_std_stats.csv')
            data_df.to_csv(save_path_run/'results_all_seeds.csv')

            wandb.finish()
    print('Finish')
    # test



