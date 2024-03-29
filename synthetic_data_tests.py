import pickle
import os
import wandb
import shap
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.metrics import f1_score, confusion_matrix, recall_score, precision_score, accuracy_score,balanced_accuracy_score
from functools import reduce
import pandas as pd
from interaction_values import MultiModalExplainer
import random
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.metrics import f1_score, confusion_matrix, recall_score, precision_score, accuracy_score,balanced_accuracy_score
from functools import reduce
import torch
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import  DataLoader
from dataloader import MMDataset


class MMFeedForwardNN(nn.Module):
    def __init__(self, input_size, output_size):
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        super(MMFeedForwardNN, self).__init__()
        hidden_size = [dim // 2 for dim in input_size]
        self.fc_modality_layers = [nn.Linear(input, hid).to(device) for input,hid in zip(input_size,hidden_size)]
        self.fc_final = nn.Linear(np.sum(hidden_size), output_size)
        self.float()

    def forward(self, features):
        # Pass each modality through its respective hidden layer
        x_hidden = [torch.relu(layer(x_modality))for layer,x_modality in zip(self.fc_modality_layers,features)]
        # Concatenate hidden representations
        x_concatenated = torch.cat(x_hidden, dim=-1)
        # Pass concatenated hidden representations through the final fully connected layer
        x_output = self.fc_final(x_concatenated)

        return x_output
class FeedForwardNN(nn.Module):
    def __init__(self, input_size, output_size):
        super(FeedForwardNN, self).__init__()
        hidden_size = input_size // 2
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.float()  # Set dtype of the model to float

    def forward(self, x):
        x = torch.relu(self.fc1(x))  # Apply ReLU activation to output of first layer
        x = self.fc2(x)  # Output of second layer
        return x



def save_checkpoint(model, checkpoint_path, filename="checkpoint.pt"):
    os.makedirs(checkpoint_path, exist_ok=True)
    filename = os.path.join(checkpoint_path, filename)
    torch.save(model, filename)
    print(f'Model checkpoint saved to {filename}')


def load_checkpoint(model, path):
    best_checkpoint = torch.load(path)
    model.load_state_dict(best_checkpoint)
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

def eval_model(dataloader, model,device,title='Val'):
    model.eval()
    probas = []
    y_true = []
    with torch.no_grad():
        for i, data in enumerate(dataloader):
            features, label = data
            try:
                features = features.to(device)
            except:
                features = [feat.to(device) for feat in features]
            logit = model(features)
            prob = F.softmax(logit, dim=1).data.cpu().numpy()
            probas.extend(prob)
            y_true.extend(label)
    predictions = np.argmax(probas,axis=1)

    f1 = f1_score(y_true, predictions, average='weighted')
    f1_macro = f1_score(y_true, predictions, average='macro')
    recall = recall_score(y_true, predictions, average='weighted')
    precision = precision_score(y_true, predictions, average='weighted')
    accuracy = accuracy_score(y_true, predictions)
    balanced_accuracy = balanced_accuracy_score(y_true, predictions)
    print(f"Evaluation of {title} split: F1 Score: {f1},F1 Score macro: {f1_macro}, Recall: {recall}, Precision: {precision}, Accuracy: {accuracy}, balanced Acc {balanced_accuracy}")

    # Calculate confusion matrix
    conf_matrix = confusion_matrix(y_true, predictions)
    results = {f'{title}_f1':f1,f'{title}_f1_macro':f1_macro,f'{title}_recall':recall,f'{title}_precision':precision,f'{title}_accuracy':accuracy,f'{title}_balanced_accuracy':balanced_accuracy}
    return results,conf_matrix

def train(device,train_dataloader, val_dataloader, save_path, use_wandb, experiment_name = 'redundancy', num_epochs=100, test_inverval = 10, input_size = 10, output_size = 5, lr = 1e-4, step_size = 500 ):

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
            if wandb:
                wandb.log(result)

            if best_f1_macro < result['Val_f1_macro']:
                best_f1_macro = result['Val_f1_macro']
                best_model = model
    save_checkpoint(best_model, save_path, filename=f"{experiment_name}.pt")
    return best_model

    # test
if __name__ == "__main__":
    use_wandb = True
    batch_size = 1000
    n_samples_for_interaction = 100
    epochs = 100
    setting = 'uniqueness1' #'redundancy' 'synergy' uniqueness0 uniqueness1
    concat = True
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    data_path = f'/home/lw754/masterproject/PID/synthetic_data/myDATA_{setting}.pickle'
    experiment_name = f'{setting}_epochs_{epochs}_concat_{concat}'
    save_path = Path(f'/home/lw754/masterproject/cross-modal-interaction/results/{experiment_name}')

    if use_wandb:
        wandb.init(project="masterthesis", name=f"{experiment_name}", group="shaply_interaction_index")
        wandb.config.update({'batch_size':batch_size,'n_samples_for_interaction':n_samples_for_interaction,'epochs':epochs,
                             'concat':concat,'setting':setting,'data_path':data_path,'save_path':save_path})
    with open(data_path, 'rb') as f:
        results = pickle.load(f)
    if concat:
        test_data = np.concatenate((results['test']['0'][:n_samples_for_interaction],results['test']['1'][:n_samples_for_interaction]), axis=1)
        val_data = np.concatenate((results['valid']['0'],results['valid']['1']), axis=1)
        train_data = np.concatenate((results['train']['0'],results['train']['1']), axis=1)
        input_size = train_data.shape[1]
    else:
        test_data =[results['test']['0'][:n_samples_for_interaction], results['test']['1'][:n_samples_for_interaction]]
        val_data = [results['valid']['0'], results['valid']['1']]
        train_data = [results['train']['0'], results['train']['1']]
        input_size = [mod.shape[1] for mod in train_data]
    train_label = np.squeeze(results['train']['label'])

    test_dataset = MMDataset(test_data,np.squeeze(results['test']['label'][:n_samples_for_interaction]),device=device,concat=concat)
    val_dataset = MMDataset(val_data,np.squeeze(results['valid']['label']),concat=concat,device=device)
    train_loader =  DataLoader(MMDataset(train_data,train_label, concat=concat,device=device),batch_size=batch_size,
                                  shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size,
                              shuffle=True, num_workers=0)

    model = train(device, train_loader, val_loader, save_path,use_wandb=use_wandb, num_epochs=epochs, test_inverval=10, input_size=input_size, output_size=len(np.unique(train_label)),experiment_name=setting)
    if concat:
        input_size = [input_size//2,input_size//2]
    explainer = MultiModalExplainer(model=model, data= val_dataset , modality_shapes=input_size, feature_names=['0','1'],classes=2,concat=concat)
    explaination  = explainer(test_dataset)
    shaply_values = explaination.values
    interaction_values = explainer.shaply_interaction_values()
    our_metric = [np.sum(interaction.values) for interaction in interaction_values]
    if use_wandb:
        wandb.log({"cross_modal": np.mean(np.abs(our_metric))})
    print(our_metric)
    print(np.mean(np.abs(our_metric)))

    for key, df in explainer.coalitions.items():
        df.to_csv(save_path/f"{key}.csv", index=False)
    shaply_values.to_csv(save_path/f"shap_values_best.csv", index=False)
    interaction_values = np.array(interaction_values)
    mask = np.eye(interaction_values.shape[1], dtype=bool)
    result = interaction_values[:, ~mask]
    df_interaction_values = pd.DataFrame(result)
    df_interaction_values.to_csv(save_path/f"interaction_index.csv", index=False)
    print('Finish')
