import pickle
import wandb
import fusilli
from fusilli.fusionmodels.tabularfusion.crossmodal_att import TabularCrossmodalMultiheadAttention
import pandas as pd
from interaction_values import MultiModalExplainer
import random
import numpy as np
from pathlib import Path
from EMAP import evaluate_emap, calculate_EMAP
import torch
from tqdm import tqdm
from torch.utils.data import  DataLoader
from dataloader import MMDataset
from models import LateFusionFeedForwardNN,EarlyFusionFeedForwardNN,IntermediateFusionFeedForwardNN
from utils import  eval_model, save_checkpoint
import argparse
from SRI import SRI_normalise
from SHAPE import cross_modal_SHAPE
def parse_args():
    parser = argparse.ArgumentParser(description='Argument Parser for your settings')

    # Add arguments
    parser.add_argument('--train_model', default=True, help='Whether to train the model or just eval')
    parser.add_argument('--seeds', nargs='+', type=int, default=[1, 42, 113], help='List of seed values')
    parser.add_argument('--use_wandb', default=False, help='Whether to use wandb or not')
    parser.add_argument('--batch_size', type=int, default=5000, help='Batch size for training')
    parser.add_argument('--n_samples_for_interaction', type=int, default=100, help='Number of samples for interaction')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs for training')
    parser.add_argument('--n_modality', type=int, default=2, help='Number of modalities')
    parser.add_argument('--settings', nargs='+', type=str, default= ['syn_mix9-10-0','syn_mix8-10-0','syn_mix7-10-0','syn_mix6-10-0', 'syn_mix5-10-0','syn_mix4-10-0','syn_mix3-10-0','syn_mix2-10-0','syn_mix1-10-0'],#['syn_mix9', 'syn_mix92' ],#['mix1', 'mix2', 'mix3', 'mix4','mix5', 'mix6'],#['redundancy', 'synergy', 'uniqueness0', 'uniqueness1'], ['syn_mix2', 'syn_mix5','syn_mix10' ]
                        choices=['redundancy', 'synergy', 'uniqueness0', 'uniqueness1', 'mix1', 'mix2', 'mix3', 'mix4', 'mix5', 'mix6'], help='List of settings')
    parser.add_argument('--concat', default = 'late',  help='early, intermediate, late')
    parser.add_argument('--label', type=str, default='', help='Can choose "" as PID synthetic data or "OR_" "XOR_" "VEC3_" "VEC2_"')
    parser.add_argument('--device', type=str, default='cuda:0' if torch.cuda.is_available() else 'cpu', help='Device for computation')
    parser.add_argument('--root_save_path', type=str, default='/home/lw754/masterproject/cross-modal-interaction/results/', help='Root save path')

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
        final_results = dict()
        data_path = f'/home/lw754/masterproject/PID/synthetic_data/{args.label}DATA_{setting}.pickle'
        experiment_name_run = f'{args.label}{setting}_epochs_{args.epochs}_concat_{args.concat}'
        save_path_run = root_save_path/experiment_name_run
        if args.use_wandb:
            wandb.init(project="masterthesis", name=f"{experiment_name_run}", group="shaply_interaction_index")
            wandb.config.update(
                {'batch_size': args.batch_size, 'n_samples_for_interaction': args.n_samples_for_interaction, 'epochs': args.epochs,
                 'concat': args.concat, 'setting': setting, 'data_path': data_path, 'save_path': save_path_run})

        for seed in args.seeds:
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
                input_size = [input_size // 2, input_size // 2]


            if args.use_wandb:
                print(f'\nEvaluation Seed {seed}')
                run_results = {'seed': seed}
                test_loader = DataLoader(test_dataset, batch_size=args.batch_size,
                                         shuffle=True, num_workers=0)
                test_results, _ = eval_model(test_loader, model, args.device, title='Test', use_wandb=True)
                run_results.update(test_results)

                # cross-modal interaction value
                #### cross modal

                explainer = MultiModalExplainer(model=model, data=val_dataset, modality_shapes=input_size,
                                                feature_names=['0', '1'], classes=2, concat=args.concat)
                explaination = explainer(test_dataset)
                shaply_values = explaination.values
                interaction_values = explainer.shaply_interaction_values()

                cross_modal_results = dict()
                for output_class in explainer.coalitions.keys() :
                    cm_result = explainer.interaction_metric(output_class=output_class)
                    for key in cm_result.keys():
                        cross_modal_results[f'{output_class}_{key}'] = np.mean(cm_result[key], axis=0)
                run_results.update(cross_modal_results)
                wandb.log(cross_modal_results)

                #### EMAP
                projection_logits, y = calculate_EMAP(model, test_dataset, device=args.device, concat=args.concat,
                                                      len_modality=input_size[0])
                results_emap,_ = evaluate_emap(projection_logits, y, title='Emap', use_wandb=True)
                run_results.update(results_emap)
                emap_gap = {}
                for (key_results, value_results), (key_results_emap, value_results_emap) in zip(test_results.items(),
                                                                                                results_emap.items()):
                    assert key_results.split("_")[1] == key_results_emap.split("_")[1], "Keys do not match!"
                    key = '_'.join(['emap_gap'] + key_results.split("_")[1:])
                    emap_gap[key] = abs(value_results - value_results_emap)
                run_results.update(emap_gap)
                wandb.log(emap_gap)


                ### SRI
                # TODO adapt for different output_classes
                SRI_results = dict()
                for output_class in explainer.coalitions.keys() :
                    SRI_result = SRI_normalise(interaction_values[output_class], args.n_modality)
                    for key in SRI_result.keys():
                        SRI_results[f'{output_class}_{key}'] = SRI_result[key]
                run_results.update(SRI_results)
                wandb.log(SRI_results)

                #### SHAPE
                SHAPE_results = dict()
                for output_class in explainer.coalitions.keys():
                    coalition_values = explainer.coalitions[output_class]
                    coalition_values= coalition_values.iloc[:, :(2 ** args.n_modality)]
                    SHAPE_result = cross_modal_SHAPE(coalition_values,args.n_modality)
                    for key in SHAPE_result.keys():
                        SHAPE_results[f'{output_class}_{key}'] = SHAPE_result[key]
                run_results.update(SHAPE_results)
                wandb.log(SHAPE_results)

                ######## store interactions ######
                for key, score in run_results.items():
                    if key not in final_results:
                        final_results[key] = []
                    final_results[key].append(score)

                for key, df in explainer.coalitions.items():
                    df.to_csv(save_path / f"{key}.csv", index=False)
                shaply_values.to_csv(save_path / f"shap_values_best.csv", index=False)
                interaction_values = np.array(interaction_values['best'])
                mask = np.eye(interaction_values.shape[1], dtype=bool)
                result = interaction_values[:, ~mask]
                df_interaction_values = pd.DataFrame(result)
                df_interaction_values.to_csv(save_path / f"interaction_index.csv", index=False)





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



