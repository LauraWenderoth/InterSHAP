import os
import wandb
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import f1_score, confusion_matrix, recall_score, precision_score, accuracy_score,balanced_accuracy_score
import torch


def save_checkpoint(model, checkpoint_path, filename="checkpoint.pt"):
    os.makedirs(checkpoint_path, exist_ok=True)
    filename = os.path.join(checkpoint_path, filename)
    torch.save(model, filename)
    print(f'Model checkpoint saved to {filename}')


def load_checkpoint(model, path):
    best_checkpoint = torch.load(path)
    model.load_state_dict(best_checkpoint)
    return best_checkpoint

def eval_model(dataloader, model,device,title='Val',use_wandb=False):
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
    if use_wandb:
        class_names = list(np.unique(y_true))
        class_names = [str(class_name) for class_name in class_names]
        wandb.log(results)
        cm = wandb.plot.confusion_matrix(
            y_true=np.array(y_true),
            preds=np.array(predictions),
            class_names=class_names)

        wandb.log({"conf_mat": cm})
    return results,conf_matrix
