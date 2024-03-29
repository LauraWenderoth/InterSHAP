import os
import pandas as pd
from torchvision.io import read_image
from torch.utils.data import Dataset, DataLoader
import torch
from pathlib import Path
from torchvision import transforms
from sklearn.model_selection import train_test_split
import itertools
import random
from PIL import Image

class MMDataset(Dataset):
    def __init__(self, X,y,concat=True, device = 'cuda:0' if torch.cuda.is_available() else 'cpu'):
        self.X = X
        self.y = y
        self.device = device
        self.concat = concat


    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        label = torch.tensor(self.y[idx])
        if self.concat:
            features = torch.tensor(self.X[idx], dtype=torch.float)
        else:
            features = [torch.tensor(modality[idx], dtype=torch.float) for modality in self.X]
        return features, label