import os
import pandas as pd
from torchvision.io import read_image
from torch.utils.data import Dataset, DataLoader
import torch
import copy
import numpy as np
from pathlib import Path
from torchvision import transforms
from sklearn.model_selection import train_test_split
import itertools
import wandb
from tqdm import tqdm
import random
from .models import EarlyFusionFeedForwardNN
from .utils import train_epoch, save_checkpoint, eval_model
from PIL import Image
class UnimodalDataset(Dataset):
    def __init__(self, X,y=None, device = 'cuda:0' if torch.cuda.is_available() else 'cpu',length=None):
        assert isinstance(X, list) or isinstance(X, dict) or isinstance(X,np.ndarray), "X must be a list or dict with one modality (+ label)"
        self.X = X
        self.y = y
        if y is None:
            try:
                self.y = X['label']
                self.X = [value for key, value in X.items() if key != 'label']
            except KeyError:
                raise ValueError("'label' key is not present in the input dictionary X. Store y in 'label'")
        self.device = device

        ## preprocess data
        self.y = np.squeeze(self.y)

        assert len(self.X) == len(self.y), \
            "Inconsistent data: X must be with the same length as y"
        self.len = len(self.y) if length is None else length

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        label = torch.tensor(self.y[idx])
        features = torch.tensor(self.X[idx], dtype=torch.float)
        return features, label
    def set_length(self,length):
        self.len = length

    def get_number_classes(self):
        return len(np.unique(self.y))

def train_unimodal(data,root_save_path, experiment_name_run,device,seed,num_epochs=100,use_wandb=False,batch_size= 100,test_inverval =10, n_samples_for_eval=None, lr = 1e-4, step_size = 500):
    assert all(key in data for key in ['test', 'train', 'valid']), "One or more keys are missing in the data dictionary: test, train, valid"
    assert all('label' in data[subset].keys() for subset in ['test', 'train', 'valid'] ), \
        "One or more elements in data['test'], data['train'], or data['valid'] are missing the key 'label'"
    n_modalities = sum(1 for key, value in data['train'].items() if key != 'label')
    output_size = len(np.unique(data['train']['label']))
    run_results = dict()

    n_samples_for_eval = len(data['test']['label']) if n_samples_for_eval is None else n_samples_for_eval

    print(f'Start uni-modal training')
    for i_modality in range(n_modalities):
        print(f'Unimodal training: modality {i_modality}')
        experiment_name = f'{experiment_name_run}/unimodal_{i_modality}'
        save_path = root_save_path / experiment_name

        test_data = data['test'][str(i_modality)][:n_samples_for_eval]
        val_data = data['valid'][str(i_modality)]
        train_data = data['train'][str(i_modality)]

        input_size = train_data.shape[1]

        train_dataset = UnimodalDataset(train_data, data['train']['label'], device=device)
        val_dataset = UnimodalDataset(val_data, data['valid']['label'],device=device)
        test_dataset = UnimodalDataset(test_data,data['test']['label'][:n_samples_for_eval], device=device)

        train_loader = DataLoader(train_dataset,  batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)



        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True

        model = EarlyFusionFeedForwardNN(input_size, output_size)
        model = model.to(device)

        best_model = model
        best_f1_macro = 0
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=0.2)
        for epoch in tqdm(range(num_epochs), desc="Epochs"):
            model, loss = train_epoch(train_loader, model, optimizer, device=device)
            if use_wandb:
                wandb.log({"training_loss": loss})
            scheduler.step()
            if epoch % test_inverval == 0:
                result, _ = eval_model(val_loader, model, device, use_wandb=use_wandb)
                result["epoch"] = epoch
                if use_wandb:
                    wandb.log(result)

                if best_f1_macro < result['Val_f1_macro']:
                    best_f1_macro = result['Val_f1_macro']
                    best_model = model
        save_checkpoint(best_model, save_path, filename=f"uni_modal_{i_modality}.pt")

        test_loader = DataLoader(test_dataset, batch_size=batch_size,
                                 shuffle=True, num_workers=0)
        test_results, _ = eval_model(test_loader, model, device, title=f'Test_unimodal_{i_modality}',use_wandb=use_wandb)
        run_results.update(test_results)
    return run_results
