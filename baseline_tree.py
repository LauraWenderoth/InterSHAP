import pickle
import shap
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, recall_score, precision_score, accuracy_score
import torch

if __name__ == "__main__":
    batch_size = 1000
    n_samples_for_interaction = 10
    setting = 'synergy' #'redundancy' 'synergy' uniqueness0
    concat = True
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    data_path = f'/home/lw754/masterproject/PID/synthetic_data/DATA_{setting}.pickle'
    save_path = Path('/home/lw754/masterproject/cross-modal-interaction/results')
    with open(data_path, 'rb') as f:
        results = pickle.load(f)

    test_data = np.concatenate((results['test']['0'][:n_samples_for_interaction],results['test']['1'][:n_samples_for_interaction]), axis=1)
    val_data = np.concatenate((results['valid']['0'],results['valid']['1']), axis=1)
    train_data = np.concatenate((results['train']['0'],results['train']['1']), axis=1)
    input_size = train_data.shape[1]
    train_label = np.squeeze(results['train']['label'])
    y_test = np.squeeze(results['test']['label'][:n_samples_for_interaction])


    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(train_data, train_label)

    y_pred = model.predict(test_data)

    # Calculate evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    # Print the evaluation metrics
    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1)


    explainer = shap.TreeExplainer(model, train_data,feature_perturbation="tree_path_dependent")
    shap_values = explainer(test_data)
    shap.summary_plot(shap_values, test_data) #feature_dependence

    shap_interaction_values = explainer.shap_interaction_values(test_data)
    shap.summary_plot(shap_interaction_values[0][0], test_data, plot_type="bar", show=True)