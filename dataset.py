from skimage.exposure import equalize_adapthist
from albumentations.pytorch import ToTensor
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Tuple
import albumentations as albu
import cv2
import numpy as np


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
            image = self.tfms(image=image)['image']
            return image


class ALASKAData2(Dataset):
    def __init__(self, ids: List[str],
                 labels: Dict[str, int],
                 tfms: albu.Compose,
                 stage: str = 'train'
                 ):
        super().__init__()
        self.ids = ids
        self.labels = labels
        self.tfms = tfms
        self.stage = stage

    def __getitem__(self, idx):
        if self.stage != 'test':
            image = cv2.imread(self.ids[idx])[..., ::-1]
            image = cv2.cvtColor(
                (image/255).astype(np.float32), cv2.COLOR_RGB2HSV)
            labels = self.labels[self.ids[idx]]
            if self.tfms:
                image = self.tfms(image=image)['image']
            else:
                image = image
            return image, labels
        else:
            image = cv2.imread(self.ids[idx])[..., ::-1]
            if self.tfms:
                image = self.tfms(image=image)['image']
            else:
                image = image
            return image

    def __len__(self):
        return len(self.ids)


def to_tensor(x):
    return np.transpose(x, (2, 0, 1))


def get_train_augm(size=Tuple[int, int],
                   p=0.5):
    return albu.Compose([
        albu.Resize(*size),
        albu.OneOf([albu.CLAHE(6, (4, 4), always_apply=True),
                    albu.Equalize(always_apply=True)], p=0.99),
        albu.HorizontalFlip(p=p),
        albu.VerticalFlip(p=p),
        albu.ToFloat(255),
        ToTensor()  # albu.Lambda(image=to_tensor)
    ])


def get_valid_augm(size=Tuple[int, int],
                   p=0.5):
    return albu.Compose([
        albu.Resize(*size),
        albu.ToFloat(255),
        ToTensor()  # albu.Lambda(image=to_tensor)
    ])
