from sklearn.model_selection import GroupKFold
import random
from models import LabelSmoothing
from torch.utils.data import Dataset
import cv2
import argparse
import os
import warnings
from glob import glob
from typing import Iterable, List, Union

import albumentations as albu
import kornia
import numpy as np
import pandas as pd
import torch
from albumentations.pytorch import ToTensorV2

from catalyst import utils

from catalyst.contrib.nn import Lookahead, RAdam
from catalyst.dl import SupervisedRunner  # , Runner
from catalyst.dl.callbacks import (
    AccuracyCallback,
    CriterionCallback,
    MetricAggregationCallback,
    AUCCallback,
    OptimizerCallback
)
from torch import nn, optim
# from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

# from dataset import ALASKAData2, ALASKATestData
from models import ENet
from utils import AlaskaAUCCallback


SEED = 42
utils.set_global_seed(SEED)
utils.prepare_cudnn(deterministic=True)

parser = argparse.ArgumentParser()

# size of data subsample
parser.add_argument("-sample", type=int, default=8192)
# size of image
parser.add_argument("-img", type=int, default=512)
# size of batch
parser.add_argument("-bs", type=int, default=8)
# learning rate
parser.add_argument("-lr", type=float, default=1e-4)
# weight decay rate
parser.add_argument("-wd", type=float, default=1e-2)
# number of cores to load data
parser.add_argument("-nw", type=int, default=4)
# number of epochs
parser.add_argument("-e", type=int, default=10)
# number of accumulation steps
parser.add_argument("-acc", type=int, default=1)
parser.add_argument("-test", type=int, default=0)
parser.add_argument("-train", type=int, default=1)
parser.add_argument("-offset", type=int, default=0)
parser.add_argument("-pseudo", type=int, default=0)
args = parser.parse_args()

data_dir = f'../input/alaska2-image-steganalysis'
sample_size = args.sample
val_size = int(sample_size*0.25)

offset = args.offset  # 20000


# def get_key(x):
#     return x.split('/')[-1].split('.')[0]


# jmipod = sorted([f'{data_dir}/JMiPOD/{x}'
#                  for x in os.listdir(f'{data_dir}/JMiPOD/')],
#                 key=get_key)[offset:offset+sample_size]
# juniward = sorted([f'{data_dir}/JUNIWARD/{x}'
#                    for x in os.listdir(
#                        f'{data_dir}/JUNIWARD/')])[offset:offset+sample_size]
# uerd = sorted([f'{data_dir}/UERD/{x}'
#                for x in os.listdir(
#                    f'{data_dir}/UERD/')])[offset:offset+sample_size]
# covers = sorted([f'{data_dir}/Cover/{x}'
#                  for x in os.listdir(
#                      f'{data_dir}/Cover/')])[offset:offset+sample_size]
# test = [f'{data_dir}/Test/{x}'
#         for x in os.listdir(f'{data_dir}/Test/')]

# labels = {f'{id}': 0 for id in covers}
# labels.update({f'{id}': 1 for id in jmipod})
# labels.update({f'{id}': 2 for id in juniward})
# labels.update({f'{id}': 3 for id in uerd})

# items = np.array(list(labels.items()))
# np.random.shuffle(items)
# labels = {idx_: int(label) for (idx_, label) in items}

# val_offset = offset+sample_size
# jmipod_val = sorted([f'{data_dir}/JMiPOD/{x}'
#                      for x in os.listdir(
#                          f'{data_dir}/JMiPOD/')],
#                     key=get_key)[val_offset:val_size+val_offset]
# juniward_val = sorted(
#     [f'{data_dir}/JUNIWARD/{x}'
#      for x in os.listdir(
#          f'{data_dir}/JUNIWARD/')])[val_offset:val_size+val_offset]
# uerd_val = sorted([f'{data_dir}/UERD/{x}'
#                    for x in os.listdir(
#                        f'{data_dir}/UERD/')])[val_offset:val_size+val_offset]
# covers_val = sorted(
#     [f'{data_dir}/Cover/{x}'
#      for x in os.listdir(
#          f'{data_dir}/Cover/')])[val_offset:val_size+val_offset]

# labels_val = {f'{id}': 0 for id in covers_val}
# labels_val.update({f'{id}': 1 for id in jmipod_val})
# labels_val.update({f'{id}': 2 for id in juniward_val})
# labels_val.update({f'{id}': 3 for id in uerd_val})

# items_val = np.array(list(labels_val.items()))
# np.random.shuffle(items_val)
# labels_val = {idx_: int(label) for (idx_, label) in items_val}
# train_keys = list(labels.keys())
# val_keys = list(labels_val.keys())

dataset = []

for label, kind in enumerate(['Cover', 'JMiPOD', 'JUNIWARD', 'UERD']):
    for path in glob('../input/alaska2-image-steganalysis/Cover/*.jpg'):
        dataset.append({
            'kind': kind,
            'image_name': path.split('/')[-1],
            'label': label
        })


random.shuffle(dataset)
dataset = pd.DataFrame(dataset)
gkf = GroupKFold(n_splits=5)

dataset.loc[:, 'fold'] = 0
for fold_number, (train_index, val_index) in enumerate(
        gkf.split(X=dataset.index, y=dataset['label'],
                  groups=dataset['image_name'])):
    dataset.loc[dataset.iloc[val_index].index, 'fold'] = fold_number

warnings.filterwarnings("ignore")
p = 0.5
dct = False
# train_data = DataLoader(
#     ALASKAData2(
#         train_keys, labels, albu.Compose([
#             albu.HorizontalFlip(p=p),
#             albu.VerticalFlip(p=p),
#             # albu.Normalize(),
#             ToTensorV2()
#         ], p=1.0), combine_dct=dct,
#     ), batch_size=args.bs,
#     shuffle=True, num_workers=args.nw)
# val_data = DataLoader(
#     ALASKAData2(
#         val_keys, labels_val, albu.Compose([
#             # albu.Normalize(),
#             ToTensorV2()
#         ]), combine_dct=dct,
#     ), batch_size=args.bs, shuffle=False, num_workers=args.nw
# )
# test_filenames = sorted(glob(f"{data_dir}/Test/*.jpg"))
# test_df = pd.DataFrame({'ImageFileName': list(
#     test_filenames)}, columns=['ImageFileName'])

# test_dataset = DataLoader(
#     ALASKATestData(test_df, augmentations=albu.Compose([
#         # albu.Normalize(),
#         ToTensorV2()
#     ]), combine_dct=dct,
#     ),
#     batch_size=8, shuffle=False, num_workers=args.nw)

DATA_ROOT_PATH = '../input/alaska2-image-steganalysis'


def onehot(size, target):
    vec = torch.zeros(size, dtype=torch.float32)
    vec[target] = 1.
    return vec


class DatasetRetriever(Dataset):

    def __init__(self, kinds, image_names, labels, transforms=None):
        super().__init__()
        self.kinds = kinds
        self.image_names = image_names
        self.labels = labels
        self.transforms = transforms

    def __getitem__(self, index: int):
        kind, image_name, label = self.kinds[index], \
            self.image_names[index], self.labels[index]
        image = cv2.imread(
            f'{DATA_ROOT_PATH}/{kind}/{image_name}', cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image /= 255.0
        if self.transforms:
            image = self.transforms(image=image)['image']

        target = onehot(4, label)
        return {"features": image,
                "targets_one_hot": target,
                "targets": label}

    def __len__(self) -> int:
        return self.image_names.shape[0]

    def get_labels(self):
        return list(self.labels)


class PseudoDatasetRetriever(Dataset):

    def __init__(self, image_names, labels, transforms=None):
        super().__init__()
        self.image_names = image_names
        self.labels = labels
        self.transforms = transforms

    def __getitem__(self, index: int):
        image_name, label = self.image_names[index], self.labels[index]
        image = cv2.imread(
            image_name, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image /= 255.0
        if self.transforms:
            image = self.transforms(image=image)['image']

        target = onehot(4, label)
        return {"features": image,
                "targets_one_hot": target,
                "targets": label}

    def __len__(self) -> int:
        return len(self.image_names)

    def get_labels(self):
        return list(self.labels)


fold_number = 0


def get_train_transforms():
    return albu.Compose([
        albu.HorizontalFlip(p=0.5),
        albu.VerticalFlip(p=0.5),
        # albu.Resize(height=512, width=512, p=1.0),
        ToTensorV2(p=1.0),
    ], p=1.0)


def get_valid_transforms():
    return albu.Compose([
        # albu.Resize(height=512, width=512, p=1.0),
        ToTensorV2(p=1.0),
    ], p=1.0)


if args.pseudo > 0:

    pseudo_label_df = pd.read_csv('pseudo_labeled_data.csv')
    train_data = DataLoader(
        PseudoDatasetRetriever(
            image_names=pseudo_label_df.Id.values[:4000],
            labels=pseudo_label_df.Label.values[:4000],
            transforms=get_train_transforms(),
        ), batch_size=args.bs, shuffle=True, pin_memory=False, num_workers=args.nw)

    val_data = DataLoader(
        PseudoDatasetRetriever(
            image_names=pseudo_label_df.Id.values[4000:],
            labels=pseudo_label_df.Label.values[4000:],
            transforms=get_train_transforms(),
        ), batch_size=args.bs, shuffle=False, pin_memory=False, num_workers=args.nw)
else:
    train_data = DataLoader(DatasetRetriever(
        kinds=dataset[dataset['fold'] != fold_number].kind.values,
        image_names=dataset[dataset['fold'] != fold_number].image_name.values,
        labels=dataset[dataset['fold'] != fold_number].label.values,
        transforms=get_train_transforms(),
    ), batch_size=args.bs, shuffle=True, pin_memory=False, num_workers=args.nw)

    val_data = DataLoader(DatasetRetriever(
        kinds=dataset[dataset['fold'] == fold_number].kind.values,
        image_names=dataset[dataset['fold'] == fold_number].image_name.values,
        labels=dataset[dataset['fold'] == fold_number].label.values,
        transforms=get_valid_transforms(),
    ), batch_size=16, shuffle=False, pin_memory=True, num_workers=args.nw)

print(len(train_data))
print(len(val_data))

loaders = {'train': train_data,
           'valid': val_data}
criterion = {"label_smooth": LabelSmoothing(),
             "bce": nn.BCEWithLogitsLoss(),
             "ce": nn.CrossEntropyLoss()}
model = ENet('efficientnet-b2', dct=dct)
logdir = "./logs/efficientnet-b2"
model.load_state_dict(torch.load(
    f'{logdir}/checkpoints/last.pth')['model_state_dict'])
optimizer = Lookahead(optim.AdamW(  # Lookahead(RAdam(
    model.parameters(), lr=args.lr, weight_decay=args.wd, eps=1e-8))

scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, factor=0.5, patience=2)
num_epochs = args.e

runner = SupervisedRunner(device='cuda')
if args.train > 0:
    runner.train(
        model=model,
        criterion=criterion,
        scheduler=scheduler,
        optimizer=optimizer,
        loaders=loaders,
        callbacks=[
            CriterionCallback("targets", prefix="loss_ce", criterion_key="ce"),
            CriterionCallback(input_key="targets_one_hot",
                              prefix="loss_label",
                              criterion_key="label_smooth"),
            CriterionCallback(input_key="targets_one_hot",
                              prefix="loss_bce",
                              criterion_key="bce"),
            MetricAggregationCallback("loss_total", metrics=[
                                      "loss_ce", "loss_label", "loss_bce"]),
            AlaskaAUCCallback(),
            AUCCallback(input_key="targets_one_hot"),
            AccuracyCallback(),
            OptimizerCallback(metric_key="loss_total",
                              accumulation_steps=args.acc)],
        logdir=logdir,
        main_metric="auc/class_0",
        # resume=f'{logdir}/checkpoints/last_full.pth',
        num_epochs=num_epochs,
        minimize_metric=False,
        verbose=True,
        fp16=dict(opt_level="O0")
    )


class DatasetSubmissionRetriever(Dataset):

    def __init__(self, image_names, transforms=None):
        super().__init__()
        self.image_names = image_names
        self.transforms = transforms

    def __getitem__(self, index: int):
        image_name = self.image_names[index]
        image = cv2.imread(
            f'{DATA_ROOT_PATH}/Test/{image_name}', cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image /= 255.0
        if self.transforms:
            sample = {'image': image}
            sample = self.transforms(**sample)
            image = sample['image']

        return image_name, image

    def __len__(self) -> int:
        return self.image_names.shape[0]


dataset = DatasetSubmissionRetriever(
    image_names=np.array([path.split(
        '/')[-1] for path in glob(
            '../input/alaska2-image-steganalysis/Test/*.jpg')]),
    transforms=get_valid_transforms(),
)


data_loader = DataLoader(
    dataset,
    batch_size=1,
    shuffle=False,
    num_workers=args.nw,
    drop_last=False,
)

if args.test > 0:
    model.cuda()
    model.load_state_dict(torch.load(
        f'{logdir}/checkpoints/last.pth')['model_state_dict'])
    test_preds_proba: Union[List, Iterable, np.ndarray] = []
    model.eval()
    progress_bar_test = tqdm(data_loader)
    result = {'Id': [], 'Label': []}
    pseudo_labeled = {'Id': [], 'Label': []}
    for step, (image_names, images) in enumerate(progress_bar_test):
        #print(step, end='\r')
        im = images.cuda()

        #im_h = kornia.augmentation.F.hflip(im)

        #im_v = kornia.augmentation.F.vflip(im)
        y_pred = 1*model(im)  # + 0.3*model(im_v)+0.3*model(im_h)

        # pseudo_labeled['Id'].extend(image_names)
        # pseudo_labeled['Label'].extend(y_pred.data.cpu().numpy().argmax(axis=1))
        y_pred = 1 - \
            nn.functional.softmax(y_pred, dim=1).data.cpu().numpy()[:, 0]

        result['Id'].extend(image_names)
        result['Label'].extend(y_pred)
    # pseudo_labeled['Id'] = [f'../input/alaska2-image-steganalysis/Test/{x}' for x in pseudo_labeled['Id'] ]
    # print(pseudo_labeled)
    # pseudo_labeled = pd.DataFrame(pseudo_labeled)
    # pseudo_labeled.to_csv(
    #     f'pseudo_labeled_data.csv', index=False)
    submission = pd.DataFrame(result)
    submission.to_csv(
        f'submission_new_dataset_efficient-netb2-pseudo.csv', index=False)
    submission.head()
    # with torch.no_grad():
    #     for i, im in enumerate(progress_bar_test):
    #         inputs = im.to('cuda')
    #         # flip horizontal
    #         #im = kornia.augmentation.F.hflip(inputs)
    #         outputs = model(im)
    #         # flip vertical
    #         #im = kornia.augmentation.F.vflip(inputs)
    #         #outputs = (0.25*outputs + 0.25*model(im))
    #         #outputs = (outputs + 0.5*model(inputs))
    #         test_preds_proba.extend(F.softmax(outputs, 1).cpu().numpy())
    # test_preds_proba = np.array(test_preds_proba)
    # labels = test_preds_proba.argmax(1)
    # bin_proba_test = np.zeros((len(test_preds_proba),))
    # temp = test_preds_proba[labels != 0, 1:]
    # bin_proba_test[labels != 0] = temp.sum(1)
    # bin_proba_test[labels == 0] = test_preds_proba[labels == 0, 0]
    # sub = pd.read_csv(
    #     f'{data_dir}/sample_submission.csv')
    # sub['Label'] = bin_proba_test
    # name = logdir.split('/')[-1]
    # sub.to_csv(f'submission_catalyst_{name}.csv', index=False)
os.system(
    f"kaggle competitions submit -c alaska2-image-steganalysis -f" +
    " submission_new_dataset_efficient-netb2-2.csv " +
    "-m 'Efficient Net B2 June 15 No TTA'")
