import pandas as pd
from .EMAP import evaluate_emap, calculate_EMAP
from .PID import clustering, convert_data_to_distribution, get_measure
from .SRI import SRI_normalise
from .SHAPE import cross_modal_SHAPE
from .interaction_values import MultiModalExplainer
import wandb
import numpy as np
def eval_synergy(model,val_dataset,test_dataset, test_data_model_pred, run_results,save_path,input_size,n_modalites,device,test_results,concat=False,use_wandb=False,n_samples_for_interaction=100 ):
    ##### PID
    n_components = 2
    kmeans_0, _ = clustering(test_data_model_pred['0'][:n_samples_for_interaction], pca=True, n_components=n_components,
                             n_clusters=20)
    kmeans_0 = kmeans_0.reshape(-1, 1)
    kmeans_1, _ = clustering(test_data_model_pred['1'][:n_samples_for_interaction], pca=True, n_components=n_components,
                             n_clusters=20)
    kmeans_1 = kmeans_1.reshape(-1, 1)
    PID_label = test_data_model_pred['label'].reshape(-1, 1)
    PID_data = (kmeans_0, kmeans_1, PID_label)

    P, _ = convert_data_to_distribution(*PID_data)
    PID_result = get_measure(P)
    run_results.update(PID_result)


    # cross-modal interaction value
    #### cross modal

    explainer = MultiModalExplainer(model=model, data=val_dataset, modality_shapes=input_size,
                                    feature_names=['0', '1'], classes=2, concat=concat)
    explaination = explainer(test_dataset)
    shaply_values = explaination.values
    interaction_values = explainer.shaply_interaction_values()

    cross_modal_results = dict()
    for output_class in explainer.coalitions.keys():
        cm_result = explainer.interaction_metric(output_class=output_class)
        for key in cm_result.keys():
            cross_modal_results[f'{output_class}_{key}'] = np.mean(cm_result[key], axis=0)
    run_results.update(cross_modal_results)


    #### EMAP
    concat = True if concat == 'early' else False
    projection_logits, y = calculate_EMAP(model, test_dataset, device=device, concat=concat,
                                          len_modality=input_size[0])
    results_emap, _ = evaluate_emap(projection_logits, y, title='Emap', use_wandb=use_wandb)
    run_results.update(results_emap)
    emap_gap = {}
    for (key_results, value_results), (key_results_emap, value_results_emap) in zip(test_results.items(),
                                                                                    results_emap.items()):
        assert key_results.split("_")[1] == key_results_emap.split("_")[1], "Keys do not match!"
        key = '_'.join(['emap_gap'] + key_results.split("_")[1:])
        emap_gap[key] = abs(value_results - value_results_emap)
    run_results.update(emap_gap)


    ### SRI
    SRI_results = dict()
    for output_class in explainer.coalitions.keys():
        SRI_result = SRI_normalise(interaction_values[output_class], n_modalites)
        for key in SRI_result.keys():
            SRI_results[f'{output_class}_{key}'] = SRI_result[key]
    run_results.update(SRI_results)


    #### SHAPE
    SHAPE_results = dict()
    for output_class in explainer.coalitions.keys():
        coalition_values = explainer.coalitions[output_class]
        coalition_values = coalition_values.iloc[:, :(2 ** n_modalites)]
        SHAPE_result = cross_modal_SHAPE(coalition_values, n_modalites)
        for key in SHAPE_result.keys():
            SHAPE_results[f'{output_class}_{key}'] = SHAPE_result[key]
    run_results.update(SHAPE_results)


    ######## store interactions ######
    for key, df in explainer.coalitions.items():
        df.to_csv(save_path / f"{key}.csv", index=False)
    shaply_values.to_csv(save_path / f"shap_values_best.csv", index=False)
    interaction_values = np.array(interaction_values['best'])
    mask = np.eye(interaction_values.shape[1], dtype=bool)
    result = interaction_values[:, ~mask]
    df_interaction_values = pd.DataFrame(result)
    df_interaction_values.to_csv(save_path / f"interaction_index.csv", index=False)

    if use_wandb:
        wandb.log(PID_result)
        wandb.log(cross_modal_results)
        wandb.log(emap_gap)
        wandb.log(SRI_results)
        wandb.log(SHAPE_results)

    final_results = dict()
    for key, score in run_results.items():
        if key not in final_results:
            final_results[key] = []
        final_results[key].append(score)

    return final_results
