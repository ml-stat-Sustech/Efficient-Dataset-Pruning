'''
dataset selection with pretrained model
'''
import os
import random
import argparse
import numpy as np

import torch
from torch.utils.data import DataLoader, ConcatDataset
import torchvision.transforms as transforms

from datasets import get_dataset
from algorithms import get_selector


def init_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main(args):

    # init seed
    init_seed(args.seed)

    # specify the data selected indexes storage location
    # idxs_dir = os.path.join(args.output_dir, 'seed_'+str(args.seed), args.dataset)
    idxs_dir = os.path.join(args.idxs_dir, 'seed_'+str(args.seed), args.dataset, args.principle, 'pt_'+args.pretrain+'-arch_'+str(args.arch))
    os.makedirs(idxs_dir, exist_ok=True)

    # load dataset
    mean, std =  [0.485, 0.456, 0.406], [0.229, 0.224, 0.225] # IN-1K

    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ]),
        'test': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    }

    data_sets = {mode: get_dataset(args.datasets_dir, args.dataset, mode, data_transforms[mode]) for mode in ['train', 'val', 'test']}
    num_classes = data_sets['train'].num_classes

    data_loaders = {
        'train': DataLoader(ConcatDataset([data_sets['train'], data_sets['val']]), batch_size=args.batch_size, shuffle=False, num_workers=8),
        # 'train': DataLoader(data_sets['train'], batch_size=args.batch_size, shuffle=False, num_workers=8),
        'test': DataLoader(data_sets['test'], batch_size=args.batch_size, shuffle=False, num_workers=8)
    }

    # ckpt_path
    # ./outputs/seed_1/CXRB10-fully-vit_base/all
    ckpt_dir = os.path.join(args.ckpts_dir, 'seed_'+str(args.seed), '-'.join([args.dataset, args.pretrain, args.arch]), 'all')

    models = []
    if os.path.isdir(ckpt_dir):
        # record the statistics during the classifier training
        ckpts = [f for f in os.listdir(ckpt_dir) if f.split('.')[0].isdigit()]
        ckpts = sorted(ckpts, key= lambda x: int(x.split('.')[0]))

        for ckpt in ckpts:
            # print(os.path.join(ckpt_dir, ckpt))
            # exit()
            models.append((args.arch, os.path.join(ckpt_dir, ckpt)))
    else:
        raise ValueError('INVALID CKPT DIR')

    # select
    selector = get_selector(args.principle, num_classes, data_loaders['train'], models)
    idxs_list = selector.select()

    # save to local
    for i in range(1, 10):

        storage_path = os.path.join(idxs_dir, str(i / 10))
        np.save(storage_path, idxs_list[i-1])


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Other pruning')
    parser.add_argument('--seed', default=1, type=int)
    parser.add_argument('--idxs_dir', type=str, default='./idxs')
    parser.add_argument('--datasets_dir', default='../datasets', type=str)
    parser.add_argument('--dataset', type=str, default='CXRB10', choices=['CXRB10', 'DeepWeeds', 'DTD', 'FGVCAircraft', 'Sketch'])
    parser.add_argument('--batch_size', type=int, default=16)

    # parser.add_argument('--principle', type=str, default='random', choices=['random', 'herding', 'kcentergreedy', 'maha_a', 'maha_t', 'leastconfidence', 'entropy', 'margin', 'forgetting', 'grand', 'el2n', 'contextualdiversity'])
    parser.add_argument('--principle', type=str, default='random', choices=['random', 'herding', 'kcentergreedy', 'leastconfidence', 'entropy', 'margin', 'forgetting', 'grand', 'el2n', 'contextualdiversity', 'cle', 'clh'])

    parser.add_argument('--arch', type=str, default='resnet18', choices=['resnet18', 'resnet50', 'vit_small', 'vit_base'])
    parser.add_argument('--pretrain', type=str, default='weakly', choices=['weakly', 'fully'])
    parser.add_argument('--ckpts_dir', type=str, default='./ckpts')

    args = parser.parse_args()

    main(args)
