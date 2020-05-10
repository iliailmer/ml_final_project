from torchvision.datasets import CocoDetection
from torch.utils.data import Dataset, DataLoader
from pycocotools.coco import COCO


class CocoDataset(Dataset):
    def __init__(self, root, annFile, transforms):
        self.coco = COCO(annFile)
        self.ids = list(sorted(self.coco.imgs.keys()))
