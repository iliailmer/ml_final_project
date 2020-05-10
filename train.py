"""Main Training Code."""

import pandas as pd
import numpy as np
from utils import alaska_weighted_auc
from pytorch_toolbelt import utils as u
from scipy.special import softmax
from catalyst.dl import utils
from catalyst.dl import (
    AUCCallback,
    AccuracyCallback,
    OptimizerCallback,
    CheckpointCallback
)
from catalyst.contrib.nn import RAdam, Lookahead
from torch import optim
from catalyst.dl import SupervisedRunner
from torchvision.models import resnet50, resnext101_32x8d, resnext50_32x4d
from efficientnet_pytorch import EfficientNet
from torch import nn
import os
from utils import get_loader, get_train_augm, get_valid_augm
from sklearn.model_selection import train_test_split, KFold
import argparse
from models import Model
parser = argparse.ArgumentParser()

parser.add_argument("--batch-size", "-bs", type=int, default=8)
parser.add_argument("--image-size", type=int, default=224)
parser.add_argument("-lr", type=float, default=0.01)
parser.add_argument("--feat-lr", type=int, default=0.0005)
parser.add_argument("--wd", type=int, default=0.00003)


args = parser.parse_args()
size = (args.image_size, args.image_size)
jmipod = [f'../input/alaska2-image-steganalysis/JMiPOD/{x}'
          for x in os.listdir('../input/alaska2-image-steganalysis/JMiPOD/')]
juniward = [f'../input/alaska2-image-steganalysis/JUNIWARD/{x}'
            for x in os.listdir(
                '../input/alaska2-image-steganalysis/JUNIWARD/')]
uerd = [f'../input/alaska2-image-steganalysis/UERD/{x}'
        for x in os.listdir('../input/alaska2-image-steganalysis/UERD/')]
covers = [f'../input/alaska2-image-steganalysis/Cover/{x}'
          for x in os.listdir('../input/alaska2-image-steganalysis/Cover/')]
test = [f'../input/alaska2-image-steganalysis/Test/{x}'
        for x in os.listdir('../input/alaska2-image-steganalysis/Test/')]

labels = {f'{id}': 0 for id in covers}
labels.update({f'{id}': 1 for id in jmipod})
labels.update({f'{id}': 2 for id in juniward})
labels.update({f'{id}': 3 for id in uerd})

ids = list(labels.keys())
np.random.shuffle(ids)


learning_rate = args.lr
feature_learning_rate = args.feat_lr
layerwise_params = {"features": dict(
    lr=feature_learning_rate, weight_decay=args.wd)}
criterion = nn.CrossEntropyLoss()


SEED = 2020
utils.set_global_seed(SEED)
utils.prepare_cudnn(deterministic=True)


folds = KFold(n_splits=3, shuffle=True, random_state=2020)
fold_n = 0
# base_optimizer = RAdam(model_params, lr=learning_rate, weight_decay=0.0003)
# optimizer = Lookahead(base_optimizer)

num_epochs = 1
logdir = "./logs/alaska"
fp16_params = dict(opt_level="O1")


callbacks = [
    AUCCallback(input_key='targets_one_hot',
                num_classes=4),
    AccuracyCallback(),
    CheckpointCallback(),
    OptimizerCallback(accumulation_steps=1)
]

runner = SupervisedRunner(
    device='cuda', input_key="image", input_target_key="targets")

fold_auc = []
for train_ids, val_ids in folds.split(ids[:100]):
    print(f"Fold {fold_n}")
    fold_n += 1
    train_keys = [ids[i] for i in train_ids]
    val_keys = [ids[i] for i in val_ids]

    loaders = {'train': get_loader(train_keys, labels, get_train_augm,
                                   size=size, stage='train',
                                   bs=args.batch_size, shuffle=True),
               'valid': get_loader(val_keys, labels, get_valid_augm,
                                   size=size, stage='val', bs=args.batch_size,
                                   shuffle=False)}

    model = Model(resnet50(True))

    optimizer = optim.Adam(model.parameters(),
                           lr=learning_rate,
                           weight_decay=0.0001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.25, patience=2)

    runner.train(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        loaders=loaders,
        callbacks=callbacks,
        logdir=logdir,
        main_metric="auc/_mean",
        minimize_metric=False,
        num_epochs=5,
        # fp16=fp16_params,
        verbose=False,
    )
    # TODO: Add custom validation callback, in which 4-class -> binary class
    # problem
    # TODO: Add


sub = pd.read_csv('../input/alaska2-image-steganalysis/sample_submission.csv')

sub['Label'] = probabilities

sub.to_csv("prediction_resnext.csv", index=False)
