from torch.utils.data import Dataset
from typing import List, Dict, Tuple
import albumentations as albu
import pandas as pd
from catalyst.utils import get_one_hot
import cv2
import numpy as np


class ALASKAData(Dataset):
    def __init__(self, df, augmentations=None):

        self.data = df
        self.tfms = augmentations

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        path, label = self.data.loc[idx]
        image = cv2.imread(path)[:, :, ::-1]
        if self.tfms:
            # Apply transformations
            image = self.tfms(image=image)['image']
            return image, label


class ALASKATestData(Dataset):
    def __init__(self, df, augmentations=None):

        self.data = df
        self.tfms = augmentations

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        path = self.data.loc[idx][0]
        image = cv2.imread(path)[:, :, ::-1]
        if self.tfms:
            # Apply transformations
            image = self.tfms(image=image)['image']
            return image
