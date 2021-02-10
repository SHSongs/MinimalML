##
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from model import UNet

from torch.utils.data import DataLoader
# from torch.utils.tensorboard import SummaryWriter

from torchvision import transforms, datasets

##
lr = 1e-4
batch_size = 4
num_epoch = 100

data_dir = './datasets'
ckpt_dir = './checkpoint'
log_dir = './log'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


## data loader

class Dataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform

        lst_data = os.listdir(self.data_dir)

        lst_label = [f for f in lst_data if f.startswith('label')]
        lst_input = [f for f in lst_data if f.startswith('input')]

        lst_label.sort()
        lst_input.sort()

        self.lst_label = lst_label
        self.lst_input = lst_input

    def __len__(self):
        return len(self.lst_label)

    def __getitem__(self, index):
        label = np.load(os.path.join(self.data_dir, self.lst_label[index]))
        input = np.load(os.path.join(self.data_dir, self.lst_input[index]))

        label = label / 255.0
        input = input / 255.0

        if label.ndim == 2:
            label = label[:, :, np.newaxis]
        if input.ndim == 2:
            input = input[:, :, np.newaxis]

        data = {'input': input, 'label': label}

        if self.transform:
            data = self.transform(data)

        return data


##


class ToTensor(object):
    def __call__(self, data):
        label, input = data['label'], data['input']

        label = label.transpose((2, 0, 1)).astype(np.float32)
        input = input.transpose((2, 0, 1)).astype(np.float32)

        data = {'label': torch.from_numpy(label), 'input': torch.from_numpy(input)}

        return data


class Normalization(object):
    def __init__(self, mean=0.5, std=0.5):
        self.mean = mean
        self.std = std

    def __call__(self, data):
        label, input = data['label'], data['input']

        input = (input - self.mean) / self.std

        data = {'label': label, 'input': input}

        return data


class RandomFlip(object):
    def __call__(self, data):
        label, input = data['label'], data['input']

        if np.random.rand() > 0.5:
            label = np.fliplr(label)
            input = np.fliplr(input)

        if np.random.rand() > 0.5:
            label = np.flipud(label)
            input = np.flipud(input)
        data = {'label': label, 'input': input}

        return data


##


##
transform = transforms.Compose([Normalization(), RandomFlip(), ToTensor()])
dataset_train = Dataset(data_dir='./datasets/train', transform=transform)

##
data = dataset_train.__getitem__(0)

input = data['input']
label = data['label']

##
plt.subplot(121)
plt.imshow(label.squeeze())
plt.title('label')

plt.subplot(122)
plt.imshow(input.squeeze())
plt.title('input')

plt.show()

