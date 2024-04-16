import pandas as pd
from .EMAP import evaluate_emap, calculate_EMAP
from .PID import clustering, convert_data_to_distribution, get_measure
from .SRI import SRI_normalise
from .SHAPE import cross_modal_SHAPE, org_SHAPE
from .interaction_values import MultiModalExplainer
from utils.utils import eval_model
from torch.utils.data import  DataLoader
import wandb
import numpy as np
from pathlib import Path

def eval_synergy(model,val_dataset,test_dataset,device, eval_metrics = ['PID','SHAPE','EMAP','SRI','Interaction'],batch_size=100,save_path =None,use_wandb=False,n_samples_for_interaction=100 ):
    eval_metrics = [metric.lower() for metric in eval_metrics]
    run_results = dict()
    mod_shape =  test_dataset.get_modality_shapes()

    concat = test_dataset.get_concat()
    n_modalites = len(mod_shape)
    # eval on standard metrics
    test_loader  = DataLoader(test_dataset, batch_size=batch_size,shuffle=False)
    test_results, y_pred_test = eval_model(test_loader, model, device, title='Test', use_wandb=False,
                                           return_predictions=True)
    run_results.update(test_results)

    if 'interaction' or 'sri' in eval_metrics:
        explainer = MultiModalExplainer(model=model, data=val_dataset, modality_shapes=mod_shape,
                                        feature_names=['0', '1'], classes=2, concat=concat)
        explaination = explainer(test_dataset)
        shaply_values = explaination.values
        interaction_values = explainer.shaply_interaction_values()

        if save_path is not None:
            save_path = Path(save_path)
            save_path.mkdir(parents=True, exist_ok=True)
            ######## store interactions ######
            for key, df in explainer.coalitions.items():
                df.to_csv(save_path / f"{key}.csv", index=False)
            shaply_values.to_csv(save_path / f"shap_values_best.csv", index=False)
            interaction_value = np.array(interaction_values['best'])
            mask = np.eye(interaction_value.shape[1], dtype=bool)
            result = interaction_value[:, ~mask]
            df_interaction_values = pd.DataFrame(result).astype(float)
            df_interaction_values.to_csv(save_path / f"interaction_index.csv", index=False,float_format='%.16f')

    ##### PID
    if 'pid' in eval_metrics:
        X_test, y_test = test_dataset.get_data()
        assert len(X_test) == 2, "Length of X_test must be 2, as PID is only defined for 2 modalities"

        n_components = 2
        n_clusters = 20 if n_samples_for_interaction >= 20 else n_samples_for_interaction
        kmeans_0, _ = clustering(X_test[0][:n_samples_for_interaction], pca=True, n_components=n_components,
                                 n_clusters=n_clusters)
        kmeans_0 = kmeans_0.reshape(-1, 1)
        kmeans_1, _ = clustering(X_test[1][:n_samples_for_interaction], pca=True, n_components=n_components,
                                 n_clusters=n_clusters)
        kmeans_1 = kmeans_1.reshape(-1, 1)
        PID_label = y_test.reshape(-1, 1)
        PID_data = (kmeans_0, kmeans_1, PID_label)

        P, _ = convert_data_to_distribution(*PID_data)
        PID_result = get_measure(P)
        run_results.update(PID_result)

    #### cross modal
    if 'interaction' in eval_metrics:
        cross_modal_results = dict()
        for output_class in explainer.coalitions.keys():
            cm_result = explainer.interaction_metric(output_class=output_class)
            for key in cm_result.keys():
                cross_modal_results[f'{output_class}_{key}'] = np.mean(cm_result[key], axis=0)
        run_results.update(cross_modal_results)


    #### EMAP
    if 'emap' in eval_metrics:
        # TODO rewrite for different mod shapes
        # TODO calc relative emap
        projection_logits, y = calculate_EMAP(model, test_dataset, device=device, concat=concat,
                                              len_modality=mod_shape[0])
        results_emap, _ = evaluate_emap(projection_logits, y, title='Emap', use_wandb=False)
        run_results.update(results_emap)
        emap_gap = {}
        for (key_results, value_results), (key_results_emap, value_results_emap) in zip(test_results.items(),
                                                                                        results_emap.items()):
            assert key_results.split("_")[1] == key_results_emap.split("_")[1], "Keys do not match!"
            key = '_'.join(['emap_gap'] + key_results.split("_")[1:])
            emap_gap[key] = abs(value_results - value_results_emap)
        run_results.update(emap_gap)


    ### SRI
    if 'sri' in eval_metrics:
        SRI_results = dict()
        for output_class in explainer.coalitions.keys():
            SRI_result = SRI_normalise(interaction_values[output_class], n_modalites)
            for key in SRI_result.keys():
                SRI_results[f'{output_class}_{key}'] = SRI_result[key]
        run_results.update(SRI_results)


    #### SHAPE
    if 'shape' in eval_metrics:
        # TODO make independent from interaction values recalc it for acc
        SHAPE_results = dict()
        for output_class in explainer.coalitions.keys():
            coalition_values = explainer.coalitions[output_class]
            coalition_values = coalition_values.iloc[:, :(2 ** n_modalites)]
            SHAPE_result = cross_modal_SHAPE(coalition_values, n_modalites)
            for key in SHAPE_result.keys():
                SHAPE_results[f'{output_class}_{key}'] = SHAPE_result[key]
        run_results.update(SHAPE_results)
        org_SHAPE_result = org_SHAPE(test_dataset,model,batch_size,device,metric='f1_macro')
        run_results.update(org_SHAPE_result)


    if use_wandb:
        wandb.log(run_results)


    return run_results
