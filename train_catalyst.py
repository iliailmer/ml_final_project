# TODO Pretrain on data again and SAVE THIS TIME

from typing import Iterable, Union, List
import argparse
import os
import warnings
from glob import glob

import albumentations as albu
import numpy as np
import pandas as pd
import torch
from albumentations.pytorch import ToTensorV2
from catalyst import utils
from catalyst.contrib.nn import Lookahead, RAdam
from catalyst.dl import SupervisedRunner
from catalyst.dl.callbacks import AccuracyCallback, OptimizerCallback
from resnest.torch import resnest50
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

import kornia
from dataset import ALASKAData2, ALASKATestData
from models import ENet, Model, Swish
from utils import alaska_weighted_auc_modified, replace

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

data_dir = f'../input/alaska2-image-steganalysis'
sample_size = args.sample
val_size = int(sample_size*0.25)

offset = 0  # 20000


def get_key(x):
    return x.split('/')[-1].split('.')[0]


jmipod = sorted([f'{data_dir}/JMiPOD/{x}'
                 for x in os.listdir(f'{data_dir}/JMiPOD/')],
                key=get_key)[offset:offset+sample_size]
juniward = sorted([f'{data_dir}/JUNIWARD/{x}'
                   for x in os.listdir(
                       f'{data_dir}/JUNIWARD/')])[offset:offset+sample_size]
uerd = sorted([f'{data_dir}/UERD/{x}'
               for x in os.listdir(
                   f'{data_dir}/UERD/')])[offset:offset+sample_size]
covers = sorted([f'{data_dir}/Cover/{x}'
                 for x in os.listdir(
                     f'{data_dir}/Cover/')])[offset:offset+sample_size]
test = [f'{data_dir}/Test/{x}'
        for x in os.listdir(f'{data_dir}/Test/')]

labels = {f'{id}': 0 for id in covers}
labels.update({f'{id}': 1 for id in jmipod})
labels.update({f'{id}': 2 for id in juniward})
labels.update({f'{id}': 3 for id in uerd})

items = np.array(list(labels.items()))
np.random.shuffle(items)
labels = {idx_: int(label) for (idx_, label) in items}

val_offset = offset+sample_size
jmipod_val = sorted([f'{data_dir}/JMiPOD/{x}'
                     for x in os.listdir(
                         f'{data_dir}/JMiPOD/')],
                    key=get_key)[val_offset:val_size+val_offset]
juniward_val = sorted([f'{data_dir}/JUNIWARD/{x}'
                       for x in os.listdir(
                           f'{data_dir}/JUNIWARD/')])[val_offset:val_size+val_offset]
uerd_val = sorted([f'{data_dir}/UERD/{x}'
                   for x in os.listdir(
                       f'{data_dir}/UERD/')])[val_offset:val_size+val_offset]
covers_val = sorted([f'{data_dir}/Cover/{x}'
                     for x in os.listdir(
                         f'{data_dir}/Cover/')])[val_offset:val_size+val_offset]

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
            # albu.Resize(*size),
            albu.HorizontalFlip(p=p),
            albu.VerticalFlip(p=p),
            albu.Normalize(),
            ToTensorV2()
        ])
    ), batch_size=args.bs, shuffle=True, num_workers=args.nw)
val_data = DataLoader(
    ALASKAData2(
        val_keys, labels_val, albu.Compose([
            # albu.Resize(*size),
            albu.Normalize(),
            ToTensorV2()
        ])
    ), batch_size=16, shuffle=False, num_workers=args.nw
)
test_filenames = sorted(glob(f"{data_dir}/Test/*.jpg"))
test_df = pd.DataFrame({'ImageFileName': list(
    test_filenames)}, columns=['ImageFileName'])

test_dataset = DataLoader(
    ALASKATestData(test_df, augmentations=albu.Compose([
        # albu.CenterCrop()
        albu.Normalize(),
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

model = Model(resnest50(True))  # SRNet(3)  # ENet('efficientnet-b0')
replace(model, nn.ReLU, Swish, 'relu')
# for i, layer in enumerate(effnet.layers):
#     if "batch_normalization" in layer.name:
#         effnet.layers[i] = GroupNormalization(groups=2, axis=-1, epsilon=0.1)
# model.load_state_dict(torch.load(
#     'logs/checkpoints/best.pth')['model_state_dict'])
print(model)
optimizer = Lookahead(RAdam(
    model.parameters(), lr=args.lr, weight_decay=args.wd))
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, factor=0.25, patience=3)
num_epochs = args.e
logdir = "./logs/resnest50"
fp16_params = None  # dict(opt_level="O1")
runner = SupervisedRunner(device='cuda')


runner.train(
    model=model,
    criterion=criterion,
    scheduler=scheduler,
    optimizer=optimizer,
    loaders=loaders,
    callbacks=[
        # wAUC(),
        AccuracyCallback(prefix='ACC'),
        OptimizerCallback(accumulation_steps=args.acc)],
    logdir=logdir,
    num_epochs=num_epochs,
    fp16=fp16_params,
    verbose=True
)
if args.test > 0:
    test_preds_proba: Union[List, Iterable, np.ndarray] = []
    model.eval()
    progress_bar_test = tqdm(test_dataset)
    with torch.no_grad():
        for i, im in enumerate(progress_bar_test):
            inputs = im.to('cuda')
            # flip horizontal
            im = kornia.augmentation.F.hflip(inputs)
            outputs = model(im)
            # flip vertical
            im = kornia.augmentation.F.vflip(inputs)
            outputs = (0.25*outputs + 0.25*model(im))
            outputs = (outputs + 0.5*model(inputs))
            test_preds_proba.extend(F.softmax(outputs, 1).cpu().numpy())

    model.load_state_dict(torch.load(
        f'{logdir}/checkpoints/best.pth')['model_state_dict'])
    test_preds_proba = np.array(test_preds_proba)
    labels = test_preds_proba.argmax(1)
    bin_proba_test = np.zeros((len(test_preds_proba),))
    temp = test_preds_proba[labels != 0, 1:]
    bin_proba_test[labels != 0] = temp.sum(1)
    bin_proba_test[labels == 0] = test_preds_proba[labels == 0, 0]
    sub = pd.read_csv(
        f'{data_dir}/sample_submission.csv')
    sub['Label'] = bin_proba_test
    sub.to_csv('submission_catalyst_srnet.csv', index=False)
