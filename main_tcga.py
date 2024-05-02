import pickle
import wandb
import pandas as pd

import random
import numpy as np
from pathlib import Path

import torch
from tqdm import tqdm
from torch.utils.data import  DataLoader
from utils.dataset import TCGADataset
from utils.models import LateFusionFeedForwardNN,EarlyFusionFeedForwardNN,IntermediateFusionFeedForwardNN, OrginalFunctionXOR
from models.intermediate_fusion import HealNet
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
    parser.add_argument('--use_wandb', default=False, help='Whether to use wandb or not')
    parser.add_argument('--batch_size', type=int, default=100, help='Batch size for training')
    parser.add_argument('--n_samples_for_interaction', type=int, default=3000, help='Number of samples for interaction')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs for training')
    parser.add_argument('--test_inverval', type=int, default=10, help='Eval interval during traing (int = number of epochs)')
    parser.add_argument('--model', default='healnet', choices=['healnet'], help='early, intermediate, late function')


    parser.add_argument('--settings', default=[['omic','slides']], help='')

    parser.add_argument('--label', type=str, default='brca', help='Can choose "" as PID synthetic data or VEC2XOR_org_ "OR_" "XOR_" "VEC3_" "VEC2_"')
    parser.add_argument('--device', type=str, default='cuda:0' if torch.cuda.is_available() else 'cpu', help='Device for computation')
    parser.add_argument('--root_save_path', type=str, default='/home/lw754/masterproject/cross-modal-interaction/results/', help='Root save path')
    parser.add_argument('--data_path', type=str,
                        default='/net/archive/export/tcga/tcga', help='Data path')
    parser.add_argument('--wandb_name', type=str, default='MA Final Results',
                        help='Can choose "" as PID synthetic data or "OR_" "XOR_" "VEC3_" "VEC2_"')

    parser.add_argument('--train_uni_model',  action='store_true', help='Whether to train the model or just eval')
    parser.add_argument('--synergy_eval_epoch', default=False, help='Whether to eval synergy metrics during training')
    parser.add_argument('--synergy_metrics', nargs='+', type=str, default=['SHAPE','SRI','Interaction','EMAP'], help="List of seed values ['SHAPE','SRI','Interaction','PID','EMAP'] ")
    parser.add_argument('--save_results', default=True, help='Whether to locally save results or not')

    args = parser.parse_args()
    return args


def load_model(model_name, train_data: DataLoader,device,modality_names,number_of_classes):
    """
    Instantiates model and moves to CUDA device if available
    Args:
        train_data:

    Returns:
        nn.Module: model used for training
    """
    feat, _ = next(iter(train_data))
    if model_name == "healnet_early":
        # early fusion healnet (concatenation, so just one modality)
        modalities = 1  # same model just single modality
        input_channels = [feat[0].shape[2]]
        input_axes = [1]

    if model_name == "healnet":
        num_sources = len(modality_names)
        input_channels = []
        input_axes = []
        for source_idx in range(num_sources):
            channels = feat[source_idx].shape[2]
            input_channels.append(channels)
            input_axes.append(1)
        modalities = num_sources

    if model_name in ["healnet", "healnet_early"]:
        model = HealNet(
            modalities=modalities,
            input_channels=input_channels,  # number of features as input channels
            input_axes=input_axes,  # second axis (b n_feats c)
            num_classes=number_of_classes,
            num_freq_bands=2,
            depth=2,
            max_freq=2,
            num_latents=17,
            latent_dim=126,
            cross_dim_head=63,
            latent_dim_head=20,
            cross_heads=1,
            latent_heads=8,
            attn_dropout=0.45,
            ff_dropout=0.35,
            weight_tie_layers=False,
            fourier_encode_data=True,
            self_per_cross_attn=0,  # if 0, no self attention at all
            final_classifier_head=True,
            snn=True,
        )
        model.float()
        model.to(device)
    '''
    elif model_name == "mcat":
        if len(modality_names) == 2:
            if "slides" in modality_names:
                model = MCAT(
                    n_classes=self.output_dims,
                    omic_shape=feat[0].shape[1:],
                    wsi_shape=feat[1].shape[1:]
                )
            else:
                model = MCATomics(
                    n_classes=self.output_dims,
                    omic_shape1=feat[0].shape[1:],
                    omic_shape2=feat[1].shape[1:]
                )
        elif modality_names[0] in ["omic", "rna-sequence", "mutation", "copy-number"]:
            model = SNN(
                n_classes=self.output_dims,
                input_dim=feat[0].shape[1]
            )
        elif modality_names[0] == "slides":
            model = MILAttentionNet(
                input_dim=feat[0].shape[1:],
                n_classes=self.output_dims
            )
        model.float()
        model.to(device)
        '''

    return model


def train(device,train_dataloader, val_dataloader, save_path, use_wandb, model_name, modality_names, number_of_classes, experiment_name = 'redundancy', num_epochs=100, test_inverval = 10, lr = 1e-4, step_size = 500 ,seed=42,synergy_eval_epoch=False,dataset=None,args=None):

    # Set random seed for reproducibility
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


    model = load_model(model_name, train_dataloader,device,modality_names,number_of_classes)


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
    number_of_classes = 4
    setting = 'brca'
    level = 2

    root_save_path = Path(args.root_save_path)
    print(f'Start experiments with setting {args.settings} on dataset {args.label}'
          f'\nAll results will be saved to: {root_save_path}'
          f'\nweight and biases is turned on: {args.use_wandb}')


    final_results = dict()
    data_path = Path(args.data_path)
    for setting in args.settings:
        experiment_name_run = f'{args.label}{setting}_epochs_{args.epochs}_model_{args.model}'
        save_path_run = root_save_path/experiment_name_run
        if args.use_wandb:
            wandb.init(project=args.wandb_name, name=f"{experiment_name_run}") #Final_MA
            wandb.config.update(
                {'batch_size': args.batch_size, 'n_samples_for_interaction': args.n_samples_for_interaction, 'epochs': args.epochs,
                 'model': args.model, 'setting': setting, 'data_path': data_path, 'save_path': save_path_run})


        for seed in args.seeds:
            run_results = {'seed': seed}
            experiment_name = f'{experiment_name_run}/seed_{seed}'
            save_path = root_save_path / experiment_name

            #Todo add train uni modal agian
            #if args.train_uni_model:
             #   train_unimodal(data, root_save_path, experiment_name_run, device=args.device, seed=seed, num_epochs =args.epochs, use_wandb=args.use_wandb, batch_size=args.batch_size, test_inverval=args.test_inverval, n_samples_for_eval=args.n_samples_for_interaction)

            # Datasets
            data = TCGADataset(data_path,
                               args.label,
                               setting,
                               args.model,
                               level=level,
                               n_bins = number_of_classes,
                               model = args.model)
            train_size = 0.7
            test_size = 0.15
            val_size = 0.15

            print(f"Train samples: {int(train_size * len(data))}, Val samples: {int(val_size * len(data))}, "
                  f"Test samples: {int(test_size * len(data))}")
            train_data, test_dataset, val_dataset = torch.utils.data.random_split(data, [train_size, test_size, val_size])

            # Dataloader
            train_loader = DataLoader(train_data,batch_size=args.batch_size,shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=args.batch_size,shuffle=False)
            test_loader = DataLoader(test_dataset, batch_size=args.batch_size,shuffle=False)

            if args.train_model:

                model = train(device=args.device, train_dataloader=train_loader, val_dataloader=val_loader, save_path=save_path, use_wandb=args.use_wandb, model_name=args.model, modality_names=setting,
                 number_of_classes=number_of_classes, experiment_name = setting, num_epochs=args.epochs, test_inverval = args.test_inverval, seed=seed,  dataset={'valid':val_dataset,'test':test_dataset }, args=args)

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


        if args.use_wandb:
            wandb.log(stats)
            wandb.finish()

        if args.save_results:
            df = pd.DataFrame.from_dict(stats, orient='index')
            data_df = pd.DataFrame(final_results)
            df.to_csv(save_path_run/'mean_std_stats.csv')
            data_df.to_csv(save_path_run/'results_all_seeds.csv')



