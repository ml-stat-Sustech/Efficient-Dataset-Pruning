'''
load full datasets
'''

import os
import random

import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder, DTD, FGVCAircraft
from torch.utils.data import DataLoader


def init_seed(seed):
    random.seed(seed)


def get_dataset(root, name, mode, transforms):
    # train, val, and test split

    # [DeepWeeds, FGVCAIRCRAFT, RESISC45, Sketch, DTD, CXRB10]
    # load dataset
    if name == 'FGVCAircraft':
        dataset = FGVCAircraft(root, split=mode, transform=transforms, download=True)
        dataset.num_classes = len(dataset.classes) # 100
        # print(dataset.num_classes)
    elif name == 'DTD':
        dataset = DTD(root, split=mode, transform=transforms, download=True)
        dataset.num_classes = len(dataset.classes)
    elif name in ['DeepWeeds', 'RESISC45', 'CXRB10', 'Sketch']:
        dataset = ImageFolder(os.path.join(root, name, mode), transform=transforms)
        dataset.num_classes = len(dataset.classes)
    else:
        raise Exception('INVALID DATASET')

    return dataset


def get_mu_std(root, name, mode):

    dataset = get_dataset(root, name, mode, transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor()]))

    loader = DataLoader(dataset, batch_size=10, num_workers=0, shuffle=False, drop_last=False)

    mean = 0.0
    for images, _ in loader:
        batch_samples = images.size(0)
        images = images.view(batch_samples, images.size(1), -1)
        mean += images.mean(2).sum(0)
    mean = mean / len(loader.dataset)

    var = 0.0
    pixel_count = 0
    for images, _ in loader:
        batch_samples = images.size(0)
        images = images.view(batch_samples, images.size(1), -1)
        var += ((images - mean.unsqueeze(1))**2).sum([0,2])
        pixel_count += images.nelement()
    std = torch.sqrt(var / pixel_count)

    return mean, std