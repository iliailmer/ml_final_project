from torch.utils.data import Dataset
from typing import List, Dict, Tuple
import albumentations as albu

from catalyst.utils import get_one_hot
import cv2
import numpy as np


class ALASKAData(Dataset):
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
            labels = self.labels[self.ids[idx]]
            if self.tfms:
                image = self.tfms(image=image)['image']
            else:
                image = image
            return {'image': image,
                    'targets_one_hot': get_one_hot(labels, 4).astype(np.int64),
                    'targets': labels}
        else:
            image = cv2.imread(self.ids[idx])[..., ::-1]
            if self.tfms:
                image = self.tfms(image=image)['image']
            else:
                image = image
            return {'image': image}

    def __len__(self):
        return len(self.ids)


def get_none_files():
    nones_jmipod = []
    from tqdm import auto
    for each in auto.tqdm([0]):
        try:
            cv2.imread(each)[..., ::-1]
        except TypeError as t:
            nones_jmipod.append(each)
