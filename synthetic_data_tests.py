import pickle
import wandb
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
from models import MMFeedForwardNN,FeedForwardNN
from utils import  eval_model, save_checkpoint





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


def train(device,train_dataloader, val_dataloader, save_path, use_wandb, experiment_name = 'redundancy', num_epochs=100, test_inverval = 10, input_size = 10, output_size = 2, lr = 1e-4, step_size = 500 ,concat=True):

    # Set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    torch.backends.cudnn.deterministic = True


    if concat:
        model =FeedForwardNN(input_size, output_size)
    else:
        model = MMFeedForwardNN(input_size, output_size)

    model = model.to(device)
    print("Model device:", next(model.parameters()).device)
    best_model = model
    best_f1_macro = 0
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=0.2)
    print("\nTraining...")
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

def run(train_model = True):
    use_wandb = True
    batch_size = 5000
    n_samples_for_interaction = 100
    epochs = 100
    n_modality = 2
    setting = 'uniqueness1'  # 'redundancy' 'synergy' uniqueness0 uniqueness1
    concat = True
    label = ''  # 'OR_' XOR_
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    for setting in ['redundancy', 'synergy', 'uniqueness0', 'uniqueness1']:
        data_path = f'/home/lw754/masterproject/PID/synthetic_data/{label}DATA_{setting}.pickle'
        experiment_name = f'{label}{setting}_epochs_{epochs}_concat_{concat}'
        save_path = Path(f'/home/lw754/masterproject/cross-modal-interaction/results/{experiment_name}')

        if use_wandb:
            wandb.init(project="masterthesis", name=f"{experiment_name}", group="shaply_interaction_index")
            wandb.config.update(
                {'batch_size': batch_size, 'n_samples_for_interaction': n_samples_for_interaction, 'epochs': epochs,
                 'concat': concat, 'setting': setting, 'data_path': data_path, 'save_path': save_path})
        with open(data_path, 'rb') as f:
            results = pickle.load(f)
        # visualize_umap(results, n_modality, label, setting)
        if concat:
            test_data = np.concatenate(
                (results['test']['0'][:n_samples_for_interaction], results['test']['1'][:n_samples_for_interaction]),
                axis=1)
            val_data = np.concatenate((results['valid']['0'], results['valid']['1']), axis=1)
            train_data = np.concatenate((results['train']['0'], results['train']['1']), axis=1)
            input_size = train_data.shape[1]
        else:
            test_data = [results['test']['0'][:n_samples_for_interaction],
                         results['test']['1'][:n_samples_for_interaction]]
            val_data = [results['valid']['0'], results['valid']['1']]
            train_data = [results['train']['0'], results['train']['1']]
            input_size = [mod.shape[1] for mod in train_data]
        train_label = np.squeeze(results['train']['label'])

        test_dataset = MMDataset(test_data, np.squeeze(results['test']['label'][:n_samples_for_interaction]),
                                 device=device, concat=concat)
        val_dataset = MMDataset(val_data, np.squeeze(results['valid']['label']), concat=concat, device=device)
        train_loader = DataLoader(MMDataset(train_data, train_label, concat=concat, device=device),
                                  batch_size=batch_size,
                                  shuffle=True, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=batch_size,
                                shuffle=True, num_workers=0)
        if train_model:
            model = train(device, train_loader, val_loader, save_path, use_wandb=use_wandb, num_epochs=epochs,
                          test_inverval=10, input_size=input_size, output_size=len(np.unique(train_label)),
                          experiment_name=setting,concat=concat)
        else:
            model_weights = save_path / f'{setting}.pt'
            model = torch.load(model_weights)
            model.to(device)
        if concat:
            input_size = [input_size // 2, input_size // 2]
        explainer = MultiModalExplainer(model=model, data=val_dataset, modality_shapes=input_size,
                                        feature_names=['0', '1'], classes=2, concat=concat)
        explaination = explainer(test_dataset)
        shaply_values = explaination.values
        interaction_values = explainer.shaply_interaction_values()
        interaction_score = explainer.interaction_metric()
        our_metric = [np.sum(interaction.values) for interaction in interaction_values]
        if use_wandb:
            test_loader = DataLoader(test_dataset, batch_size=batch_size,
                                    shuffle=True, num_workers=0)
            results,_=  eval_model(test_loader, model, device, title='Test',use_wandb=True)
            projection_logits, y = calculate_EMAP(model, test_dataset, device=device, concat=concat,
                                                  len_modality=input_size[0])
            results_emap,_ = evaluate_emap(projection_logits, y, title='Emap', use_wandb=True)
            emap_gap = {}
            for (key_results, value_results), (key_results_emap, value_results_emap) in zip(results.items(),
                                                                                            results_emap.items()):
                assert key_results.split("_")[1] == key_results_emap.split("_")[1], "Keys do not match!"
                key = '_'.join(['emap_gap'] + key_results.split("_")[1:])
                emap_gap[key] = abs(value_results - value_results_emap)
            wandb.log(emap_gap)
            wandb.log({"cross_modal": np.mean(np.abs(our_metric)),'cross_modal_rel':np.mean(interaction_score)})
            wandb.log(shaply_values.mean().to_dict())
        print("cross_modal", np.mean(np.abs(our_metric)),'cross_modal_rel', np.mean(interaction_score))

        for key, df in explainer.coalitions.items():
            df.to_csv(save_path / f"{key}.csv", index=False)
        shaply_values.to_csv(save_path / f"shap_values_best.csv", index=False)
        interaction_values = np.array(interaction_values)
        mask = np.eye(interaction_values.shape[1], dtype=bool)
        result = interaction_values[:, ~mask]
        df_interaction_values = pd.DataFrame(result)
        df_interaction_values.to_csv(save_path / f"interaction_index.csv", index=False)
        # After your training loop or wherever appropriate
        if use_wandb:
            wandb.finish()

    print('Finish')
    # test
if __name__ == "__main__":
    run(train_model=False)

