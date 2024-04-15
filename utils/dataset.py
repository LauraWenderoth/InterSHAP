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
import random
from PIL import Image

class MMDataset(Dataset):
    def __init__(self, X,y=None,concat='early', device = 'cuda:0' if torch.cuda.is_available() else 'cpu',length=None):
        assert isinstance(X, list) or isinstance(X, dict), "X must be a list or dict with modalities"
        self.X = X
        self.y = y
        if y is None:
            try:
                self.y = X['label']
                self.X = [value for key, value in X.items() if key != 'label']
            except KeyError:
                raise ValueError("'label' key is not present in the input dictionary X. Store y in 'label'")
        assert len(self.X) >= 2, "X must have at least 2 modalities"
        self.X_org = copy.deepcopy(self.X)
        self.modality_shape = [mod.shape[1] for mod in self.X]
        self.device = device
        self.concat = True if concat == 'early' else False

        ## preprocess data
        self.y = np.squeeze(self.y)
        if self.concat:
            self.X = np.concatenate(self.X,axis=1)

        assert len(self.X) == len(self.y) or all(isinstance(sublist, list) and len(sublist) == len(self.y) for sublist in self.X), \
            "Inconsistent data: X must be either a single list with the same length as y or a list of lists with each sublist having the same length as y"
        if length is not None:
            self.set_length(length)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        label = torch.tensor(self.y[idx])
        if self.concat:
            features = torch.tensor(self.X[idx], dtype=torch.float)
        else:
            features = [torch.tensor(modality[idx], dtype=torch.float) for modality in self.X]
        return features, label

    def get_data(self):
        # get unconcatinated data
        return self.X_org, self.y

    def set_length(self,length):
        if length <=  len(self.y):
            self.y = self.y[:length]
            try:
                self.X = self.X[:length]
            except:
                self.X = [mod[:length] for mod in self.X]



    def get_modality_shapes(self):
        return self.modality_shape

    def get_number_classes(self):
        return len(np.unique(self.y))

    def get_concat(self):
        return self.concat