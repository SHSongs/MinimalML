##
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from model import UNet

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

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
    def __init__(self, data_dir, tarnsform=None):
        self.data_dir = data_dir
        self.transform = tarnsform

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

dataset_train = Dataset(data_dir='./datasets/train')

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

##

