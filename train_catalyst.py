from catalyst.contrib.nn import FocalLossMultiClass
from torch import nn
import albumentations as albu
from torch import optim
import warnings
from torch.utils.data import DataLoader
import argparse
import kornia
from torch.nn import functional as F
from tqdm.auto import tqdm
from utils import alaska_weighted_auc_modified
from glob import glob
from albumentations import (
    Compose, Resize, VerticalFlip, HorizontalFlip, ImageCompression,
    ToFloat, ToGray, ToSepia, Normalize
)
from albumentations.pytorch import ToTensorV2
from models import ENet, SRNet, WideOrthoResNet
from resnest.torch import resnest101
from dataset import ALASKAData2, ALASKATestData
from catalyst import utils
import torch
import seaborn as sns
from catalyst.contrib.nn import RAdam, Lookahead
from catalyst.dl.callbacks import (
    AccuracyCallback,
    OptimizerCallback,
    MetricCallback
)
from catalyst.dl import SupervisedRunner
import pandas as pd
import cv2
import numpy as np
from scipy import fftpack
import os
import matplotlib.pyplot as plt

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
parser.add_argument("-nw", type=int, default=2)
# number of epochs
parser.add_argument("-e", type=int, default=10)
# number of accumulation steps
parser.add_argument("-acc", type=int, default=1)
parser.add_argument("-test", type=int, default=0)
args = parser.parse_args()

data_dir = '../input/alaska2-image-steganalysis'
sample_size = args.sample
val_size = int(sample_size*0.25)


jmipod = sorted([f'../input/alaska2-image-steganalysis/JMiPOD/{x}'
                 for x in os.listdir('../input/alaska2-image-steganalysis/JMiPOD/')],
                key=lambda x: x.split('/')[-1].split('.')[0])[:sample_size]
juniward = sorted([f'../input/alaska2-image-steganalysis/JUNIWARD/{x}'
                   for x in os.listdir('../input/alaska2-image-steganalysis/JUNIWARD/')])[:sample_size]
uerd = sorted([f'../input/alaska2-image-steganalysis/UERD/{x}'
               for x in os.listdir('../input/alaska2-image-steganalysis/UERD/')])[:sample_size]
covers = sorted([f'../input/alaska2-image-steganalysis/Cover/{x}'
                 for x in os.listdir('../input/alaska2-image-steganalysis/Cover/')])[:sample_size]
test = [f'../input/alaska2-image-steganalysis/Test/{x}'
        for x in os.listdir('../input/alaska2-image-steganalysis/Test/')]

labels = {f'{id}': 0 for id in covers}
labels.update({f'{id}': 1 for id in jmipod})
labels.update({f'{id}': 2 for id in juniward})
labels.update({f'{id}': 3 for id in uerd})

items = np.array(list(labels.items()))
np.random.shuffle(items)
labels = {idx_: int(label) for (idx_, label) in items}


jmipod_val = sorted([f'../input/alaska2-image-steganalysis/JMiPOD/{x}'
                     for x in os.listdir('../input/alaska2-image-steganalysis/JMiPOD/')],
                    key=lambda x: x.split('/')[-1].split('.')[0])[sample_size:val_size+sample_size]
juniward_val = sorted([f'../input/alaska2-image-steganalysis/JUNIWARD/{x}'
                       for x in os.listdir('../input/alaska2-image-steganalysis/JUNIWARD/')])[sample_size:val_size+sample_size]
uerd_val = sorted([f'../input/alaska2-image-steganalysis/UERD/{x}'
                   for x in os.listdir('../input/alaska2-image-steganalysis/UERD/')])[sample_size:val_size+sample_size]
covers_val = sorted([f'../input/alaska2-image-steganalysis/Cover/{x}'
                     for x in os.listdir('../input/alaska2-image-steganalysis/Cover/')])[sample_size:val_size+sample_size]

labels_val = {f'{id}': 0 for id in covers_val}
labels_val.update({f'{id}': 1 for id in jmipod_val})
labels_val.update({f'{id}': 2 for id in juniward_val})
labels_val.update({f'{id}': 3 for id in uerd_val})

items_val = np.array(list(labels_val.items()))
np.random.shuffle(items_val)
labels_val = {idx_: int(label) for (idx_, label) in items_val}

train_keys = list(labels.keys())
val_keys = list(labels_val.keys())

# from utils import wAUC
warnings.filterwarnings("ignore")

size = (args.img, args.img)
p = 0.5
train_data = DataLoader(
    ALASKAData2(
        train_keys, labels, albu.Compose([
            albu.Resize(*size),
            albu.HorizontalFlip(p=p),
            albu.VerticalFlip(p=p),
            ToTensorV2()
        ])
    ), batch_size=args.bs, shuffle=True, num_workers=args.nw)
val_data = DataLoader(
    ALASKAData2(
        val_keys, labels_val, albu.Compose([
            albu.Resize(*size),
            ToTensorV2()
        ])
    ), batch_size=16, shuffle=False, num_workers=args.nw
)
test_filenames = sorted(glob(f"{data_dir}/Test/*.jpg"))
test_df = pd.DataFrame({'ImageFileName': list(
    test_filenames)}, columns=['ImageFileName'])

test_dataset = DataLoader(
    ALASKATestData(test_df, augmentations=albu.Compose([
        albu.Resize(*size),
        albu.ToFloat(),
        ToTensorV2()
    ])
    ),
    batch_size=1, shuffle=False, num_workers=args.nw)
print(len(train_data))
print(len(val_data))

SEED = 2020
utils.set_global_seed(SEED)
utils.prepare_cudnn(deterministic=True)
loaders = {'train': train_data,
           'valid': val_data}
criterion = nn.CrossEntropyLoss()
model = ENet('efficientnet-b0')
print(model)
optimizer = optim.AdamW(
    model.parameters(), lr=args.lr, weight_decay=args.wd)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, factor=0.25, patience=2)
num_epochs = args.e
logdir = "./logs"
fp16_params = dict(opt_level="O1")
runner = SupervisedRunner(device='cuda')


runner.train(model=model,
             criterion=criterion,
             # scheduler=scheduler,
             optimizer=optimizer,
             loaders=loaders,
             callbacks=[
                 # wAUC(),
                 AccuracyCallback(prefix='ACC'),
                 OptimizerCallback(accumulation_steps=args.acc)],
             logdir='./logs',
             num_epochs=num_epochs,
             fp16=None,  # fp16_params,
             verbose=True
             )
if args.test > 0:
    test_preds_proba = []
    model.eval()
    progress_bar_test = tqdm(test_dataset)
    with torch.no_grad():
        for i, im in enumerate(progress_bar_test):
            inputs = im.to('cuda')
            # flip vertical
            im = kornia.augmentation.F.hflip(inputs)
            outputs = model(im)
            # fliplr
            im = kornia.augmentation.F.vflip(inputs)
            outputs = (0.25*outputs + 0.25*model(im))
            outputs = (outputs + 0.5*model(inputs))
            test_preds_proba.extend(F.softmax(outputs, 1).cpu().numpy())

    model.load_state_dict(torch.load(
        'logs/checkpoints/best.pth')['model_state_dict'])
    test_preds_proba = np.array(test_preds_proba)
    labels = test_preds_proba.argmax(1)
    bin_proba_test = np.zeros((len(test_preds_proba),))
    temp = test_preds_proba[labels != 0, 1:]
    bin_proba_test[labels != 0] = temp.sum(1)
    bin_proba_test[labels == 0] = test_preds_proba[labels == 0, 0]
    sub = pd.read_csv(
        '../input/alaska2-image-steganalysis/sample_submission.csv')
    sub['Label'] = bin_proba_test
    sub.to_csv('submission_catalyst_b0.csv', index=False)

# train_fn, val_fn = [], []
# train_labels, val_labels = [], []

# cover_filenames = sorted(glob(f"{data_dir}/Cover/*.jpg")[:sample_size])
# np.random.shuffle(cover_filenames)

# train_fn.extend(cover_filenames[val_size:])
# train_labels.extend(np.zeros(len(cover_filenames[val_size:],)))

# val_fn.extend(cover_filenames[:val_size])
# val_labels.extend(np.zeros(len(cover_filenames[:val_size],)))

# folder_names = ['JMiPOD/', 'JUNIWARD/', 'UERD/']
# for label, folder in enumerate(folder_names):
#     cover_filenames = sorted(glob(f"{data_dir}/{folder}/*.jpg")[:sample_size])
#     np.random.shuffle(cover_filenames)
#     train_fn.extend(cover_filenames[val_size:])
#     train_labels.extend(np.zeros(len(cover_filenames[val_size:],))+label+1)
#     val_fn.extend(cover_filenames[:val_size])
#     val_labels.extend(np.zeros(len(cover_filenames[:val_size],))+label+1)

# assert len(train_labels) == len(train_fn), "wrong labels"
# assert len(val_labels) == len(val_fn), "wrong labels"

# train_df = pd.DataFrame({'ImageFileName': train_fn, 'Label': train_labels},
#                         columns=['ImageFileName', 'Label'])
# train_df['Label'] = train_df['Label'].astype(int)
# val_df = pd.DataFrame({'ImageFileName': val_fn, 'Label': val_labels}, columns=[
#                       'ImageFileName', 'Label'])
# val_df['Label'] = val_df['Label'].astype(int)
# print(train_df.sample(10))


# img_size = (args.img, args.img)
# train_aug = Compose([
#     Resize(*img_size, p=1),
#     VerticalFlip(p=0.5),
#     HorizontalFlip(p=0.5),
#     Normalize(),
#     ToTensorV2()
# ], p=1)
# valid_aug = Compose([
#     Resize(*img_size, p=1),  # does nothing if it's alread 512.
#     # ToFloat(max_value=255),
#     Normalize(),
#     ToTensorV2()
# ], p=1)


# batch_size = args.bs
# num_workers = args.nw
# lr = args.lr
# # train_dataset = ALASKAData(train_df, augmentations=train_aug)
# # valid_dataset = ALASKAData(val_df, augmentations=valid_aug)
# train_loader = DataLoader(ALASKAData(train_df,
#                                      augmentations=train_aug),
#                           batch_size=batch_size,
#                           num_workers=num_workers,
#                           shuffle=True)

# valid_loader = DataLoader(ALASKAData(val_df,
#                                      augmentations=valid_aug),
#                           batch_size=batch_size,
#                           num_workers=num_workers,
#                           shuffle=False)

# loaders = {'train': train_loader,
#            'valid': valid_loader}
# device = 'cuda'
# model = ENet(name='efficientnet-b0').cuda()
# optimizer = Lookahead(RAdam(model.parameters(),
#                             lr=lr,
#                             weight_decay=args.wd))
# criterion = torch.nn.CrossEntropyLoss()

# scheduler = torch.optim.lr_scheduler\
#     .ReduceLROnPlateau(optimizer, mode='max', factor=0.75, patience=5)
# num_epochs = args.e

# # wAUCCallback = MetricCallback(prefix='wAUC',
# #                               metric_fn=alaska_weighted_auc_modified,
# #                               input_key='targets',
# #                               output_key='logits'
# #                               )
# runner = SupervisedRunner(model, 'cuda')
# runner.train(model=model,
#              criterion=criterion,
#              scheduler=scheduler,
#              optimizer=optimizer,
#              loaders=loaders,
#              callbacks=[
#                  wAUC(),
#                  AccuracyCallback(num_classes=4),
#                  OptimizerCallback(accumulation_steps=1)],
#              logdir='./logs',
#              num_epochs=num_epochs,
#              fp16=dict(opt_level='O1'),
#              verbose=True
#              )

# test_filenames = sorted(glob(f"{data_dir}/Test/*.jpg"))
# test_df = pd.DataFrame({'ImageFileName': list(
#     test_filenames)}, columns=['ImageFileName'])

# batch_size = 1
# num_workers = 4
# test_dataset = ALASKATestData(test_df, augmentations=valid_aug)
# test_loader = torch.utils.data.DataLoader(test_dataset,
#                                           batch_size=batch_size,
#                                           num_workers=num_workers,
#                                           shuffle=False,
#                                           drop_last=False)
# model.eval()

# test_preds_proba = []
# progress_bar_test = tqdm(test_loader)
# with torch.no_grad():
#     for i, im in enumerate(progress_bar_test):
#         inputs = im.to(device)
#         # flip vertical
#         im = kornia.augmentation.F.hflip(inputs)
#         outputs = model(im)
#         # fliplr
#         im = kornia.augmentation.F.vflip(inputs)
#         outputs = (0.25*outputs + 0.25*model(im))
#         outputs = (outputs + 0.5*model(inputs))
#         test_preds_proba.extend(F.softmax(outputs, 1).cpu().numpy())

# test_preds_proba = np.array(test_preds_proba)
# labels = test_preds_proba.argmax(1)
# bin_proba_test = np.zeros((len(test_preds_proba),))
# temp = test_preds_proba[labels != 0, 1:]
# bin_proba_test[labels != 0] = temp.sum(1)
# bin_proba_test[labels == 0] = test_preds_proba[labels == 0, 0]
# test_df['Id'] = test_df['ImageFileName'].apply(lambda x: x.split(os.sep)[-1])
# test_df['Label'] = bin_proba_test

# test_df = test_df.drop('ImageFileName', axis=1)
# test_df.to_csv('submission_eb3.csv', index=False)
# print(test_df.head())
