from torch.utils.data import  DataLoader
import numpy as np
from scipy.special import softmax
from sklearn.metrics import f1_score, confusion_matrix, recall_score, precision_score, accuracy_score,balanced_accuracy_score
import copy
import wandb
import collections
import torch
import pickle
from pathlib import Path
from utils.dataset import MMDataset
from utils.models import LateFusionFeedForwardNN,EarlyFusionFeedForwardNN, IntermediateFusionFeedForwardNN
from tqdm.auto import tqdm

def evaluate_emap(logits, y_true,title='Val',use_wandb=False):
    probas = softmax(logits, axis=1)
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
    if use_wandb:
        class_names = list(np.unique(y_true))
        class_names = [str(class_name) for class_name in class_names]
        wandb.log(results)
        cm = wandb.plot.confusion_matrix(
            y_true=np.array(y_true),
            preds=np.array(predictions),
            class_names=class_names)

        wandb.log({"emap_conf_mat": cm})
    return results,conf_matrix
def emap(idx2logits):
    '''Example implementation of EMAP (more efficient ones exist)

    inputs:
      idx2logits: This nested dictionary maps from image/text indices
        function evals, i.e., idx2logits[i][j] = f(t_i, v_j)

    returns:
      projected_preds: a numpy array where projected_preds[i]
        corresponds to \hat f(t_i, v_i).
    '''
    all_logits = []
    for k, v in idx2logits.items():
        all_logits.extend(v.values())
    all_logits = np.vstack(all_logits)
    logits_mean = np.mean(all_logits, axis=0)

    reversed_idx2logits = collections.defaultdict(dict)
    for i in range(len(idx2logits)):
        for j in range(len(idx2logits[i])):
            reversed_idx2logits[j][i] = idx2logits[i][j]

    projected_preds = []
    for idx in range(len(idx2logits)):
        pred = np.mean(np.vstack(list(idx2logits[idx].values())), axis=0)
        pred += np.mean(np.vstack(list(reversed_idx2logits[idx].values())), axis=0)
        pred -= logits_mean
        projected_preds.append(pred)

    projected_preds = np.vstack(projected_preds)
    return projected_preds
def calculate_EMAP(model,X,device,concat=False,len_modality=None):
    # TODO rewrite modality for mods with different shapes
    assert not (concat and len_modality is None), 'Need number (len_modality) to split tensor'
    results_emap = collections.defaultdict(lambda: collections.defaultdict(dict))
    y = []
    for i,data_i in enumerate(tqdm(X,desc="EMAP")):
        X_i, y_i = data_i
        y.append(y_i)
        for j,data_j in enumerate(copy.deepcopy(X)):
            X_j, _ = data_j
            X_ij = copy.deepcopy(X_j)
            if concat:
                X_ij[:len_modality] = X_i[:len_modality]
                X_ij = X_ij.to(device)
            else:
                X_ij[0] = X_i[0]
                X_ij = [x[np.newaxis, ...] for x in X_ij]
                X_ij = [x.to(device) for x in X_ij]
            logits = model.forward(X_ij)
            logits = logits.tolist()
            for class_index, logit in enumerate(logits):
                results_emap[class_index][i][j] = logits[class_index]
    projection_logits = []
    for class_index in results_emap.keys():
        projection_logits.append(emap(results_emap[class_index]))
    projection_logits = np.concatenate(projection_logits, axis=1)
    return projection_logits, y

if __name__ == "__main__":
    batch_size = 5000
    n_samples_for_interaction = 100
    epochs = 100
    n_modality = 2
    setting = 'synergy'  # 'redundancy' 'synergy' uniqueness0 uniqueness1
    concat = True
    label = 'XOR_' #'XOR_'  # 'OR_'
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    data_path = f'/home/lw754/masterproject/PID/synthetic_data/{label}DATA_{setting}.pickle'
    experiment_name = f'{label}{setting}_epochs_{epochs}_concat_{concat}'
    print(f'Eval EMAP Experiment {experiment_name}')
    save_path = Path(f'/home/lw754/masterproject/cross-modal-interaction/results/{experiment_name}')

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
    test_loader = DataLoader(test_dataset, batch_size=batch_size,
                            shuffle=True, num_workers=0)


    model_weights = save_path / f'{setting}.pt'
    model = torch.load(model_weights)
    model.to(device)
    if concat:
        input_size = input_size // 2

    projection_logits, y = calculate_EMAP(model, test_dataset,device=device,concat=concat,len_modality=input_size)
    evaluate_emap(projection_logits, y,  title='Val', use_wandb=False)

