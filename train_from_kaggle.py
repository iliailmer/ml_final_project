import torch.nn.functional as F
import gc
import cv2
from sklearn import metrics
from tqdm.auto import tqdm
import time
from torch.utils.data import Dataset
import torchvision
from glob import glob
import matplotlib.pyplot as plt
import torch.nn as nn
import random
import numpy as np
import pandas as pd
import torch
import os
from albumentations import (
    Compose, HorizontalFlip, CLAHE, HueSaturationValue,
    RandomBrightness, RandomContrast, RandomGamma, OneOf, Resize,
    ToFloat, ShiftScaleRotate, GridDistortion, ElasticTransform, JpegCompression, HueSaturationValue,
    RGBShift, RandomBrightness, RandomContrast, Blur, MotionBlur, MedianBlur, GaussNoise, CenterCrop,
    IAAAdditiveGaussianNoise, GaussNoise, OpticalDistortion, RandomSizedCrop, VerticalFlip
)
from albumentations.pytorch import ToTensor
from efficientnet_pytorch import EfficientNet
seed = 42
print(f'setting everything to seed {seed}')
random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
# from tqdm import tqdm


data_dir = '../input/alaska2-image-steganalysis'
sample_size = 8192
val_size = int(sample_size*0.25)
train_fn, val_fn = [], []
train_labels, val_labels = [], []
train_filenames = sorted(glob(f"{data_dir}/Cover/*.jpg")[:sample_size])
np.random.shuffle(train_filenames)
train_fn.extend(train_filenames[val_size:])
train_labels.extend(np.zeros(len(train_filenames[val_size:],)))
val_fn.extend(train_filenames[:val_size])
val_labels.extend(np.zeros(len(train_filenames[:val_size],)))

folder_names = ['JMiPOD/', 'JUNIWARD/', 'UERD/']
for label, folder in enumerate(folder_names):
    train_filenames = sorted(glob(f"{data_dir}/{folder}/*.jpg")[:sample_size])
    np.random.shuffle(train_filenames)
    train_fn.extend(train_filenames[val_size:])
    train_labels.extend(np.zeros(len(train_filenames[val_size:],))+label+1)
    val_fn.extend(train_filenames[:val_size])
    val_labels.extend(np.zeros(len(train_filenames[:val_size],))+label+1)

assert len(train_labels) == len(train_fn), "wrong labels"
assert len(val_labels) == len(val_fn), "wrong labels"

train_df = pd.DataFrame({'ImageFileName': train_fn, 'Label': train_labels},
                        columns=['ImageFileName', 'Label'])
train_df['Label'] = train_df['Label'].astype(int)
val_df = pd.DataFrame({'ImageFileName': val_fn, 'Label': val_labels}, columns=[
                      'ImageFileName', 'Label'])
val_df['Label'] = val_df['Label'].astype(int)
print(train_df.sample(10))


class Alaska2Dataset(Dataset):

    def __init__(self, df, augmentations=None):

        self.data = df
        self.augment = augmentations

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        fn, label = self.data.loc[idx]
        im = cv2.imread(fn)[:, :, ::-1]
        if self.augment:
            # Apply transformations
            im = self.augment(image=im)
        return im, label


img_size = 512
AUGMENTATIONS_TRAIN = Compose([
    # few images are not 512x512. does nothing if it's alread 512.
    Resize(img_size, img_size, p=1),
    VerticalFlip(p=0.5),
    HorizontalFlip(p=0.5),
    JpegCompression(quality_lower=75, quality_upper=100,
                    p=0.5),
    ToFloat(max_value=255),
    ToTensor()
], p=1)


AUGMENTATIONS_TEST = Compose([
    Resize(img_size, img_size, p=1),  # does nothing if it's alread 512.
    ToFloat(max_value=255),
    ToTensor()
], p=1)


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = EfficientNet.from_pretrained('efficientnet-b0')
        # 1280 is the number of neurons in last layer. is diff for diff. architecture
        self.dense_output = nn.Linear(1280, 4)

    def forward(self, x):
        feat = self.model.extract_features(x)
        feat = F.avg_pool2d(feat, feat.size()[2:]).reshape(-1, 1280)
        return self.dense_output(feat)


batch_size = 8
num_workers = 2

train_dataset = Alaska2Dataset(train_df, augmentations=AUGMENTATIONS_TRAIN)
valid_dataset = Alaska2Dataset(val_df, augmentations=AUGMENTATIONS_TEST)

train_loader = torch.utils.data.DataLoader(train_dataset,
                                           batch_size=batch_size,
                                           num_workers=num_workers,
                                           shuffle=True)

valid_loader = torch.utils.data.DataLoader(valid_dataset,
                                           batch_size=batch_size*2,
                                           num_workers=num_workers,
                                           shuffle=False)

device = 'cuda'
model = Net().to(device)
# model.load_state_dict(torch.load(
#     '../input/alaska2-cnn-multiclass-classifier/epoch_16_val_loss_8.3_auc_0.729.pth'))
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)


def alaska_weighted_auc(y_true, y_valid):
    tpr_thresholds = [0.0, 0.4, 1.0]
    weights = [2,   1]

    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_valid, pos_label=1)

    # size of subsets
    areas = np.array(tpr_thresholds[1:]) - np.array(tpr_thresholds[:-1])

    # The total area is normalized by the sum of weights such that the final weighted AUC is between 0 and 1.
    normalization = np.dot(areas, weights)

    competition_metric = 0
    for idx, weight in enumerate(weights):
        y_min = tpr_thresholds[idx]
        y_max = tpr_thresholds[idx + 1]
        mask = (y_min < tpr) & (tpr < y_max)
        # pdb.set_trace()

        x_padding = np.linspace(fpr[mask][-1], 1, 100)

        x = np.concatenate([fpr[mask], x_padding])
        y = np.concatenate([tpr[mask], [y_max] * len(x_padding)])
        y = y - y_min  # normalize such that curve starts at y=0
        score = metrics.auc(x, y)
        submetric = score * weight
        best_subscore = (y_max - y_min) * weight
        competition_metric += submetric

    return competition_metric / normalization


# TRAINING
criterion = torch.nn.CrossEntropyLoss()
num_epochs = 8
train_loss, val_loss = [], []

for epoch in range(num_epochs):
    print('Epoch {}/{}'.format(epoch, num_epochs - 1))
    print('-' * 10)
    model.train()
    running_loss = 0
    tk0 = tqdm(train_loader, total=int(len(train_loader)))
    for im, labels in tk0:
        inputs = im["image"].to(device, dtype=torch.float)
        labels = labels.to(device, dtype=torch.long)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        tk0.set_postfix(loss=(loss.item()))

    epoch_loss = running_loss / (len(train_loader)/batch_size)
    train_loss.append(epoch_loss)
    print('Training Loss: {:.8f}'.format(epoch_loss))

    tk1 = tqdm(valid_loader, total=int(len(valid_loader)))
    model.eval()
    running_loss = 0
    y, preds = [], []
    with torch.no_grad():
        for (im, labels) in tk1:
            inputs = im["image"].to(device, dtype=torch.float)
            labels = labels.to(device, dtype=torch.long)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            y.extend(labels.cpu().numpy().astype(int))
            preds.extend(F.softmax(outputs, 1).cpu().numpy())
            running_loss += loss.item()
            tk1.set_postfix(loss=(loss.item()))

        epoch_loss = running_loss / (len(valid_loader)/batch_size)
        val_loss.append(epoch_loss)
        preds = np.array(preds)
        # convert multiclass labels to binary class
        labels = preds.argmax(1)
        acc = (labels == y).mean()*100
        new_preds = np.zeros((len(preds),))
        temp = preds[labels != 0, 1:]
#         new_preds[labels != 0] = [temp[i, val]
#                                   for i, val in enumerate(temp.argmax(1))]
        new_preds[labels != 0] = temp.sum(1)
        new_preds[labels == 0] = preds[labels == 0, 0]
        y = np.array(y)
        y[y != 0] = 1
        auc_score = alaska_weighted_auc(y, new_preds)
        print(
            f'Val Loss: {epoch_loss:.3}, Weighted AUC:{auc_score:.3}, Acc: {acc:.3}')

    torch.save(model.state_dict(),
               f"epoch_{epoch+11}_val_loss_{epoch_loss:.3}_auc_{auc_score:.3}.pth")


# # Inference
class Alaska2TestDataset(Dataset):

    def __init__(self, df, augmentations=None):

        self.data = df
        self.augment = augmentations

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        fn = self.data.loc[idx][0]
        im = cv2.imread(fn)[:, :, ::-1]

        if self.augment:
            # Apply transformations
            im = self.augment(image=im)

        return im


test_filenames = sorted(glob(f"{data_dir}/Test/*.jpg"))
test_df = pd.DataFrame({'ImageFileName': list(
    test_filenames)}, columns=['ImageFileName'])

batch_size = 16
num_workers = 4
test_dataset = Alaska2TestDataset(test_df, augmentations=AUGMENTATIONS_TEST)
test_loader = torch.utils.data.DataLoader(test_dataset,
                                          batch_size=batch_size,
                                          num_workers=num_workers,
                                          shuffle=False,
                                          drop_last=False)
model.eval()

preds = []
tk0 = tqdm(test_loader)
with torch.no_grad():
    for i, im in enumerate(tk0):
        inputs = im["image"].to(device)
        # flip vertical
        im = inputs.flip(2)
        outputs = model(im)
        # fliplr
        im = inputs.flip(3)
        outputs = (0.25*outputs + 0.25*model(im))
        outputs = (outputs + 0.5*model(inputs))
        preds.extend(F.softmax(outputs, 1).cpu().numpy())

preds = np.array(preds)
labels = preds.argmax(1)
new_preds = np.zeros((len(preds),))
temp = preds[labels != 0, 1:]
# new_preds[labels != 0] = [temp[i, val] for i, val in enumerate(temp.argmax(1))]
new_preds[labels != 0] = temp.sum(1)
new_preds[labels == 0] = preds[labels == 0, 0]

test_df['Id'] = test_df['ImageFileName'].apply(lambda x: x.split(os.sep)[-1])
test_df['Label'] = new_preds

test_df = test_df.drop('ImageFileName', axis=1)
test_df.to_csv('submission_eb3.csv', index=False)
print(test_df.head())
