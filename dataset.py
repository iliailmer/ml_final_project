from skimage.exposure import equalize_adapthist
from albumentations.pytorch import ToTensor
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Tuple
import albumentations as albu
import cv2
import numpy as np
#from jpegio import jpegio as jio
from catalyst.utils import get_one_hot
import jpegio as jio


class ALASKATestData(Dataset):
    def __init__(self, df, augmentations=None, combine_dct=False):
        self.data = df
        self.tfms = augmentations
        self.dct = combine_dct

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        path = self.data.loc[idx][0]
        image = cv2.cvtColor(cv2.imread(
            path), cv2.COLOR_BGR2RGB).astype(np.float32)/255
        if self.dct:
            dct_data = np.array(jio.read(self.ids[idx]).coef_arrays)\
                .transpose(1, 2, 0).astype(np.float32)
            dct_data = (dct_data-dct_data.min()) / \
                (dct_data.max()-dct_data.min())
            image = np.concatenate([image, dct_data], axis=-1)
        if self.tfms:
            image = self.tfms(image=image)['image']
            return image


class ALASKAData2(Dataset):
    def __init__(self, ids: List[str],
                 labels: Dict[str, int],
                 tfms: albu.Compose,
                 stage: str = 'train',
                 combine_dct: bool = False
                 ):
        super().__init__()
        self.ids = ids
        self.labels = labels
        self.tfms = tfms
        self.stage = stage
        self.dct = combine_dct

    def __getitem__(self, idx):
        if self.stage != 'test':
            image = cv2.cvtColor(cv2.imread(
                self.ids[idx]), cv2.COLOR_BGR2RGB).astype(np.float32)/255
            if self.dct:
                dct_data = np.array(jio.read(self.ids[idx]).coef_arrays)\
                    .transpose(1, 2, 0).astype(np.float32)
                dct_data = (dct_data-dct_data.min()) / \
                    (dct_data.max()-dct_data.min())
                if np.isnan(dct_data).any():
                    raise ValueError(f"Encountered nan in {self.ids[idx]}")
                image = np.concatenate([image, dct_data], axis=2)
            labels = self.labels[self.ids[idx]]
            if self.tfms:
                image = self.tfms(image=image)['image']
            else:
                image = image
            return {"features": image,
                    "targets": labels,
                    "targets_one_hot": get_one_hot(labels, 4)}
        else:
            image = cv2.imread(self.ids[idx])[..., ::-1]
            #image = np.array(jio.read(self.ids[idx]).coef_arrays).transpose(1,2,0).astype(np.float32)
            if self.tfms:
                image = self.tfms(image=image)['image']
            else:
                image = image
            return {"features": image}

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
