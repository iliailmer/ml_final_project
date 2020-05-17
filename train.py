from torch.utils.data import DataLoader
from catalyst.contrib.nn import RAdam, Lookahead
import argparse
import os
import kornia
from torch.nn import functional as F
from tqdm.auto import tqdm
from utils import alaska_weighted_auc
from glob import glob
import numpy as np
from albumentations import (
    Compose, Resize, VerticalFlip, HorizontalFlip, ImageCompression,
    ToFloat, ToGray, ToSepia, Normalize
)

from albumentations.pytorch import ToTensorV2
import pandas as pd
from models import ENet, SRNet, WideOrthoResNet, Model
from torchvision.models import resnext50_32x4d, resnet101
from resnest.torch import resnest50
from dataset import ALASKAData, ALASKATestData
from catalyst.utils import set_global_seed
import torch
set_global_seed(2020)
torch.cuda.manual_seed(2020)
torch.backends.cudnn.deterministic = True
# TODO: make notebooks with image processing:
# different color channels, dct, fft, etc

# TODO: test out EfficientNet-B0 with a different data sampling.
# possibly random train_test_split (stratified by labels.)
# TODO: Run Leaning rate finder on all models?
# TODO: add mixed precision training
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
parser.add_argument("-acc", type=int, default=4)
args = parser.parse_args()

data_dir = '../input/alaska2-image-steganalysis'
sample_size = args.sample
val_size = int(sample_size*0.25)

train_fn, val_fn = [], []
train_labels, val_labels = [], []

cover_filenames = sorted(glob(f"{data_dir}/Cover/*.jpg")[:sample_size])
np.random.shuffle(cover_filenames)

train_fn.extend(cover_filenames[val_size:])
train_labels.extend(np.zeros(len(cover_filenames[val_size:],)))

val_fn.extend(cover_filenames[:val_size])
val_labels.extend(np.zeros(len(cover_filenames[:val_size],)))

folder_names = ['JMiPOD/', 'JUNIWARD/', 'UERD/']
for label, folder in enumerate(folder_names):
    cover_filenames = sorted(glob(f"{data_dir}/{folder}/*.jpg")[:sample_size])
    np.random.shuffle(cover_filenames)
    train_fn.extend(cover_filenames[val_size:])
    train_labels.extend(np.zeros(len(cover_filenames[val_size:],))+label+1)
    val_fn.extend(cover_filenames[:val_size])
    val_labels.extend(np.zeros(len(cover_filenames[:val_size],))+label+1)

assert len(train_labels) == len(train_fn), "wrong labels"
assert len(val_labels) == len(val_fn), "wrong labels"

train_df = pd.DataFrame({'ImageFileName': train_fn, 'Label': train_labels},
                        columns=['ImageFileName', 'Label'])
train_df['Label'] = train_df['Label'].astype(int)
val_df = pd.DataFrame({'ImageFileName': val_fn, 'Label': val_labels}, columns=[
                      'ImageFileName', 'Label'])
val_df['Label'] = val_df['Label'].astype(int)
print(train_df.sample(10))


img_size = (args.img, args.img)
train_aug = Compose([
    Resize(*img_size, p=1),
    VerticalFlip(p=0.5),
    HorizontalFlip(p=0.5),
    Normalize(),
    ToTensorV2()
], p=1)
valid_aug = Compose([
    Resize(*img_size, p=1),  # does nothing if it's alread 512.
    # ToFloat(max_value=255),
    Normalize(),
    ToTensorV2()
], p=1)


batch_size = args.bs
num_workers = args.nw
lr = args.lr
# train_dataset = ALASKAData(train_df, augmentations=train_aug)
# valid_dataset = ALASKAData(val_df, augmentations=valid_aug)
train_loader = DataLoader(ALASKAData(train_df,
                                     augmentations=train_aug),
                          batch_size=batch_size,
                          num_workers=num_workers,
                          shuffle=True)

valid_loader = DataLoader(ALASKAData(val_df,
                                     augmentations=valid_aug),
                          batch_size=batch_size,
                          num_workers=num_workers,
                          shuffle=False)
device = 'cuda'
model = ENet(name='efficientnet-b0').cuda()
optimizer = Lookahead(RAdam(model.parameters(),
                            lr=lr,
                            weight_decay=args.wd))
criterion = torch.nn.CrossEntropyLoss()
# model.load_state_dict(torch.load(
# 'epoch_9_val_loss_0.991_auc_0.792.pth'))
scheduler = torch.optim.lr_scheduler\
    .ReduceLROnPlateau(optimizer, mode='max', factor=0.75, patience=5)
num_epochs = args.e
train_loss, val_loss = [], []
best_score = 0.6
optimizer.zero_grad()
for epoch in range(num_epochs):
    print('Epoch {}/{}'.format(epoch+1, num_epochs))
    print('-' * 10)
    model.train()
    step = 0
    running_loss = 0.
    progress_bar_train = tqdm(train_loader, total=int(len(train_loader)))
    for im, labels in progress_bar_train:
        inputs = im.to(device, dtype=torch.float)
        labels = labels.to(device, dtype=torch.long)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        progress_bar_train.set_postfix(loss=(loss.item()))

    epoch_loss = running_loss / (len(train_loader))
    train_loss.append(epoch_loss)
    print('Training Loss: {:.8f}'.format(epoch_loss))

    progress_bar_valid = tqdm(valid_loader, total=int(len(valid_loader)))
    model.eval()
    running_loss = 0.
    y_true, val_preds_proba = [], []
    with torch.no_grad():
        for (im, labels) in progress_bar_valid:
            inputs = im.to(device, dtype=torch.float)
            labels = labels.to(device, dtype=torch.long)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            y_true.extend(labels.cpu().numpy().astype(int))
            val_preds_proba.extend(F.softmax(outputs, 1).cpu().numpy())
            running_loss += loss.item()
            progress_bar_valid.set_postfix(loss=(loss.item()))

        epoch_loss = running_loss / (len(valid_loader))
        val_loss.append(epoch_loss)
        val_preds_proba = np.array(val_preds_proba)
        # convert multiclass labels to binary class
        labels = val_preds_proba.argmax(1)
        acc = (labels == y_true).mean()*100
        bin_proba = np.zeros((len(val_preds_proba),))  # binary probability
        temp = val_preds_proba[labels != 0, 1:]
        bin_proba[labels != 0] = temp.sum(1)
        bin_proba[labels == 0] = val_preds_proba[labels == 0, 0]
        y_true = np.array(y_true[:])
        y_true[y_true != 0] = 1
        auc_score = alaska_weighted_auc(y_true, bin_proba)
    print(
        f'Val Loss: {epoch_loss:.3}, Weighted AUC:{auc_score:.3}, Acc: {acc:.3}')
    scheduler.step(auc_score)
    if auc_score > best_score:
        best_score = auc_score
        torch.save(model.state_dict(),
                   f"effnet0_epoch_{epoch}_auc_{auc_score:.3}.pth")


test_filenames = sorted(glob(f"{data_dir}/Test/*.jpg"))
test_df = pd.DataFrame({'ImageFileName': list(
    test_filenames)}, columns=['ImageFileName'])

batch_size = 1
num_workers = 4
test_dataset = ALASKATestData(test_df, augmentations=valid_aug)
test_loader = torch.utils.data.DataLoader(test_dataset,
                                          batch_size=batch_size,
                                          num_workers=num_workers,
                                          shuffle=False,
                                          drop_last=False)
model.eval()

test_preds_proba = []
progress_bar_test = tqdm(test_loader)
with torch.no_grad():
    for i, im in enumerate(progress_bar_test):
        inputs = im.to(device)
        # flip vertical
        im = kornia.augmentation.F.hflip(inputs)
        outputs = model(im)
        # fliplr
        im = kornia.augmentation.F.vflip(inputs)
        outputs = (0.25*outputs + 0.25*model(im))
        outputs = (outputs + 0.5*model(inputs))
        test_preds_proba.extend(F.softmax(outputs, 1).cpu().numpy())

test_preds_proba = np.array(test_preds_proba)
labels = test_preds_proba.argmax(1)
bin_proba_test = np.zeros((len(test_preds_proba),))
temp = test_preds_proba[labels != 0, 1:]
bin_proba_test[labels != 0] = temp.sum(1)
bin_proba_test[labels == 0] = test_preds_proba[labels == 0, 0]
test_df['Id'] = test_df['ImageFileName'].apply(lambda x: x.split(os.sep)[-1])
test_df['Label'] = bin_proba_test

test_df = test_df.drop('ImageFileName', axis=1)
test_df.to_csv('submission_eb3.csv', index=False)
print(test_df.head())
