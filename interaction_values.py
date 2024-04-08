from shap.explainers._explainer import Explainer
from shap._explanation import Explanation
from torch.utils.data import  DataLoader
import numpy as np
import pandas as pd
import random
import time

import math
import torch

import itertools
from tqdm.auto import tqdm
def reduce_data(data_point):
    #should return only the data as tensor
    data, y = data_point

    return data


def reverse_reduce_data(reduced_data, original_sample: list):
    # Reconstruct the original data structure
    data_point = []
    index = 0
    for tensor in original_sample:
        if isinstance(tensor,torch.Tensor):
            tensor_size = np.prod(tensor.shape[:])
            np_reshaped = reduced_data[index:index + tensor_size].reshape(tensor.shape[:]) # add patch
            tensor_reshaped = torch.tensor(np_reshaped)
            #tensor_reshaped = tensor_reshaped.to(torch.double)
            data_point.append(tensor_reshaped)
            index += tensor_size
    return data_point
def powerset(lst):
    n = len(lst)
    indices = np.indices((2,) * n).reshape(n, -1).T
    powerset_masks = np.array([[lst[j] if i else 0 for i, j in zip(subset, range(n))] for subset in indices])
    # Sort the powerset masks in descending order
    sorted_indices = np.lexsort(np.rot90(powerset_masks))
    return powerset_masks[sorted_indices[::-1]]
class MultiModalExplainer(Explainer):
    def __init__(self, model, data, modality_shapes, classes = 2, max_samples = 100, feature_names=None,concat =False, device='cuda:0' if torch.cuda.is_available() else 'cpu',random_masking=100):
        self.concat = concat

        self.device = device
        self.model = model
        self.data = data
        self.modalities = modality_shapes # tupel of how many feature per modatlity
        self.number_modalities = len(modality_shapes)
        permutations = [1]*self.number_modalities
        self.powerset_masks = powerset(permutations)
        self.expected_value = None
        self.masks = []
        for powerset_mask in self.powerset_masks:
            mask = []
            for i in range(self.number_modalities):
                mod_mask = [powerset_mask[i]] * self.modalities[i]
                mask.append(mod_mask)
            if self.concat:
                mask = np.array(mask).flatten()
            self.masks.append(mask)
        self.coalitions = {}
        self.number_output = classes
        for c in range(classes):
            self.coalitions[f'class_{c}'] = pd.DataFrame(columns=[str(subset) for subset in self.powerset_masks])
        self.coalitions['best'] = pd.DataFrame(columns=[str(subset) for subset in self.powerset_masks])
        self.random_masking = random_masking
        self.max_samples = max_samples
        self.explain_data = []
        self.base_values = self.calc_base_values(data,model)
        self.interaction_values = dict()
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        if feature_names is not None:
            self.feature_names = feature_names
        else:
            self.feature_names = list(self.coalitions['best'].columns)



        # find E_x[f(x)]
        # for all samples in data: calculate probas
        # calculate mean axis = 1

    def calculate_coalition_value(self,sample,mask,model):
        model_predictions = []
        for i in range(self.random_masking):
            random_index = random.randint(0, len(self.data) - 1)
            random_data_point = self.data[random_index] # to masked modalitiy
            random_data_point = reduce_data(random_data_point)
            if self.concat:
                if not isinstance(mask, torch.Tensor):
                    mask = torch.tensor(mask)
                mask_random_data_point = mask ^ 1
                masked_data_point = mask_random_data_point * random_data_point
                masked_sample = mask.clone() * sample
                masked_sample = masked_sample + masked_data_point
                masked_sample = masked_sample.to(self.device,dtype=torch.float)
            else:
                mask_random_data_point = [mask_ ^ 1 for mask_ in mask]
                masked_data_point = [mask_random_data_point[i] * random_data_point[i] for i in range(len(sample))]
                masked_sample = [mask[i] * sample[i] for i in range(len(sample))]
                masked_sample = [masked_sample[i] + masked_data_point[i] for i in range(len(sample))]
                masked_sample = [masked_sample[i].to(self.device,dtype=torch.float) for i in range(len(sample))]
            logits = model.forward(masked_sample)
            probabilities = torch.nn.functional.softmax(logits, dim=-1)
            logits = logits.detach().cpu().numpy()
            probabilities = probabilities.detach().cpu().numpy()
            model_predictions.append(probabilities)
        #print(masked_sample_tensor)
        return np.mean(model_predictions,axis=0).squeeze()

    def calculate_coaliton_values(self,X,model):
        for i, sample in enumerate(tqdm(X,desc="Coalition Values")):
            self.explain_data.append(sample)
            data = reduce_data(sample)
            best = -1
            for mask,subset in zip(self.masks,self.powerset_masks):


                if self.concat == False: #prepare mask
                    mask = list(itertools.chain.from_iterable(mask))
                    mask = np.squeeze(mask)
                    mask = reverse_reduce_data(mask, data)
                if np.all(subset == 0):
                    for c in range(self.number_output):
                        self.coalitions[f'class_{c}'].loc[i, str(subset)] = self.base_values[c]
                    self.coalitions['best'].loc[i, str(subset)] = self.base_values[best]
                else:
                    predictions = self.calculate_coalition_value(data, mask, model)
                    if np.all(subset == 1):
                        best = np.argmax(predictions)
                    #print(f'coaltion values{data}')
                    for c in range(len(predictions)):
                        self.coalitions[f'class_{c}'].loc[i,  str(subset)] = predictions[c]
                    self.coalitions['best'].loc[i, str(subset)] = predictions[best]

    def calc_base_values(self,X, model):
        model_predictions = []
        X = DataLoader(X,
                              batch_size=1,
                              shuffle=False,
                              num_workers=1,
                              pin_memory=True,
                              multiprocessing_context="fork",
                              persistent_workers=True,
                              prefetch_factor=2)
        for sample in tqdm(X, desc="base value extraction", total=self.max_samples):
            features = reduce_data(sample)
            # only move to GPU now (use CPU for preprocessing)
            if self.concat:
                features = features.to(self.device, dtype=torch.float)
            else:
                features = [feat.to(self.device, dtype=torch.float) for feat in features]
            logits = model.forward(features)
            probabilities = torch.nn.functional.softmax(logits, dim=1)
            logits = logits.detach().cpu().numpy()
            probabilities = probabilities.detach().cpu().numpy()
            model_predictions.append(probabilities)
        return np.mean(model_predictions,axis=0).squeeze()

    def shaply_values(self):
        modality_arrays = [np.eye(self.number_modalities, dtype=int)[i] for i in range(self.number_modalities)]
        powerset_array = np.array(self.powerset_masks)
        for output_class in self.coalitions.keys():

            for modality_array in modality_arrays:
                shaply_value_column = f'shaply_value_{modality_array}'
                self.coalitions[output_class][shaply_value_column] = 0
                for sample_row in range(len(self.coalitions[output_class])):
                    modality_index = np.where(modality_array == 1)[0][0]
                    filtered_rows = powerset_array[np.where(powerset_array[:, modality_index] == 0)]
                    filtered_rows_as_strings = [row for row in filtered_rows]
                    shaply_value = 0
                    for np_row, name in  zip(filtered_rows, filtered_rows_as_strings): #np row is output without i
                        coalition_size = np.sum(np_row)
                        weight = (math.factorial(coalition_size) * math.factorial(
                            self.number_modalities - coalition_size - 1)) / math.factorial(
                            self.number_modalities)
                        S_i = str(np_row + modality_array)
                        S = str(np_row)
                        result_S_i = self.coalitions[output_class][S_i][sample_row]
                        result_S = self.coalitions[output_class][S][sample_row]
                        shaply_value += weight * (result_S_i-result_S)
                    self.coalitions[output_class].loc[sample_row, shaply_value_column] = shaply_value

    def shaply_interaction_values(self):
        modality_arrays = [np.eye(self.number_modalities, dtype=int)[i] for i in range(self.number_modalities)]
        output_class = 'best'
        interaction_column = 'interaction_sum_abs'
        for output_class in self.coalitions.keys():
            df_coalitions = self.coalitions[output_class]
            self.coalitions[output_class][interaction_column] = 0
            for sample_row in range(len(df_coalitions)):
                interactions = []
                for i in modality_arrays:
                    interaction_row = []
                    for j in modality_arrays:
                        if np.any(i != j):
                            modality_index_i = np.where(i == 1)[0][0]
                            modality_index_j = np.where(j == 1)[0][0]
                            filtered_rows = self.powerset_masks[np.where((self.powerset_masks[:, modality_index_i] == 0) & (self.powerset_masks[:, modality_index_j] == 0))]
                            filtered_rows_as_strings = [row for row in filtered_rows]
                            interaction_value = 0
                            for np_row, name in zip(filtered_rows, filtered_rows_as_strings):
                                coalition_size = np.sum(np_row)
                                weight = (math.factorial(coalition_size) * math.factorial(
                                    self.number_modalities - coalition_size - 2)) / (2*math.factorial(
                                    self.number_modalities-1))
                                S_i = str(np_row + i)
                                S_j = str(np_row + j)
                                S_i_j = str(np_row + j + i)
                                S = str(np_row)
                                result_S_i = df_coalitions[S_i][sample_row]
                                result_S_j = df_coalitions[S_j][sample_row]
                                result_S_i_j = df_coalitions[S_i_j][sample_row]
                                result_S = df_coalitions[S][sample_row]
                                interaction_value += weight*(result_S_i_j+result_S-result_S_j-result_S_i)
                            interaction_row.append(interaction_value)
                        else:
                            interaction_row.append(0)
                    interactions.append(interaction_row)

                self.coalitions[output_class].loc[sample_row, interaction_column] = np.sum(np.abs(interactions))
                # calculate phi_ii
                interaction_df = pd.DataFrame(interactions, columns=[self.feature_names], index=[self.feature_names])
                for idx,i in enumerate(modality_arrays):
                    shaply_value_i = df_coalitions[f'shaply_value_{i}'][sample_row]
                    interaction_i = np.sum(interaction_df[self.feature_names[idx]].values)
                    interaction_df.loc[self.feature_names[idx],self.feature_names[idx]] = shaply_value_i - interaction_i
                self.interaction_values.append(interaction_df)
        return self.interaction_values



    def interaction_metric(self):
        '''
        best = self.coalitions['best']
        ones_columns = [col for col in best.columns if '1' in col and '0' not in col]
        zeros_columns = [col for col in best.columns if '0' in col and '1' not in col]
        ones_values = best[ones_columns].values
        zeros_values = best[zeros_columns].values
        interaction = best['interaction_sum'].values
        interaction_score = np.abs(interaction).flatten() / (ones_values - zeros_values).flatten()
        self.coalitions['best']['cross_modal_interaction'] = interaction_score
        '''
        interaction_score = []
        for interaction_df in self.interaction_values:
            identity_mask = np.eye(interaction_df.shape[0])
            shaply_values = np.sum(np.abs(interaction_df.values))
            interaction_values = np.sum(np.abs((1 - identity_mask) * interaction_df.values))
            interaction_score.append(interaction_values/shaply_values)
        self.coalitions['best']['cross_modal_interaction'] = interaction_score
        return interaction_score

    def __call__(self, X):
        start_time = time.time()
        self.calculate_coaliton_values(X,self.model)
        self.shaply_values()
        columns_with_shap = list(filter(lambda x: 'shap' in x, self.coalitions['best'].columns))
        v = self.coalitions['best'][columns_with_shap]
        #TODO hard coded!!!
        ev_tiled = self.coalitions['best']['[0 0]'].values
        self.expected_value = ev_tiled

        explanation = Explanation(
            v, #values
            base_values=ev_tiled,
            #data=X.to_numpy() if isinstance(X, pd.DataFrame) else X,
            feature_names=self.feature_names,
            compute_time=time.time() - start_time
            )
        return explanation