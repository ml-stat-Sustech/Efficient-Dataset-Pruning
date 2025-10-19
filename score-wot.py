'''
Learning Complexity Scoring function (without tuning)
'''

import os
import time
import random
import argparse
import numpy as np

import torch
import torch.nn.functional as F
import torch.nn.utils.prune as prune

from torch.autograd import Variable
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader, ConcatDataset
import torchvision.transforms as transforms

from datasets import get_dataset
from models import get_cclf
from utils import euclidean_dist


def init_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def extract_ys(dataloader):

    ys = torch.zeros([0], requires_grad=False, dtype=torch.long).cuda()

    for data in dataloader:
        _, labels = data
        labels = Variable(labels.long().cuda())

        ys = torch.cat((ys, labels), dim=0)
    
    return ys


def extract_zs(cclf, dataloader):

    zs = None

    for data in dataloader:
        inputs, _ = data
        inputs = Variable(inputs.float().cuda())

        with torch.no_grad():
            z = cclf.forward_features(inputs)
        
        if zs is None:
            zs = torch.flatten(F.adaptive_avg_pool2d(z, (1, 1)), start_dim=1)
        else:
            # z_all = torch.cat((z_all, z), dim=0)
            zs = torch.cat((zs, torch.flatten(F.adaptive_avg_pool2d(z, (1, 1)), start_dim=1)), dim=0)
    
    return zs


def cal_loss(class_centroids, czs, ys, num_classes):

    ds = euclidean_dist(czs, class_centroids) # [NUM_SAMPLES, NUM_CLASSES]
    assert ds.size(1) == num_classes
    loss = F.cross_entropy(1/ds, ys, reduction='none')

    return loss

def select_idxs_ead(ys, num_classes, ls, budget_ratio, mode):

    idxs_target = np.array([], dtype=np.int64)

    for c in range(num_classes):
        idxs_c = torch.where(ys == c)[0].cpu().numpy()
        num_c = len(idxs_c)

        ls_c = ls[idxs_c]
        num_selected_c = round(num_c * budget_ratio)

        if mode == 'e':
            idxs_target = np.append(idxs_target, idxs_c[np.argsort(ls_c)[:num_selected_c]])
        else:
            idxs_target = np.append(idxs_target, idxs_c[np.argsort(ls_c)[::-1][:num_selected_c]])

    return idxs_target


def select_idxs(ys, num_classes, ls, budget_ratio, split_ratio):

    idxs_target = np.array([], dtype=np.int64)

    for c in range(num_classes):
        idxs_c_target = np.array([], dtype=np.int64)
        idxs_c = torch.where(ys == c)[0].cpu().numpy()
        num_c = len(idxs_c)

        ls_c = ls[idxs_c]
        # num_selected_c = round(num_c * budget_ratio)

        if split_ratio <= 0.5:
            easy_ratio = min(split_ratio, 0.5 * budget_ratio)
            hard_ratio = budget_ratio - easy_ratio
        else:
            hard_ratio = min(1-split_ratio, 0.5 * budget_ratio)
            easy_ratio = budget_ratio - hard_ratio
        
        # easy part
        idxs_c_target = np.append(idxs_c_target, idxs_c[np.random.choice(np.argsort(ls_c)[:round(num_c * split_ratio)], size=round(num_c * easy_ratio), replace=False)])

        # hard part
        idxs_c_target = np.append(idxs_c_target, idxs_c[np.random.choice(np.argsort(ls_c)[round(num_c * split_ratio):], size=round(num_c * hard_ratio), replace=False)])

        idxs_target = np.append(idxs_target, idxs_c_target)
    
    return idxs_target


def main(args):
    
    init_seed(args.seed)
    # specify the data selected indexes storage location
    # i: capacity interval; f: subnet frequency; pt: pretraining paradigm
    # [10, 1]; [5, 2]; [2, 5]; [1, 10]
    inter = args.interval
    freq = args.frequency

    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225] # IN-1K
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

    # load the checkpoint model
    # ./ckpts/seed_1/CXRB10-fully-vit_base/all
    ckpts_dir = os.path.join(args.ckpts_dir, 'seed_'+str(args.seed), '-'.join([args.dataset, args.pretrain, args.arch]), 'all')

    models = []
    if os.path.isdir(ckpts_dir):
        # record the statistics during the classifier training
        ckpts = [f for f in os.listdir(ckpts_dir) if f.split('.')[0].isdigit()]
        ckpts = sorted(ckpts, key= lambda x: int(x.split('.')[0]))

        for ckpt in ckpts:
            # print(os.path.join(ckpt_dir, ckpt))
            # exit()
            models.append((args.arch, os.path.join(ckpts_dir, ckpt)))
    else:
        raise ValueError('INVALID CKPT DIR')
    
    data_sets = {mode: get_dataset(args.datasets_dir, args.dataset, mode, data_transforms[mode]) for mode in ['train', 'val', 'test']}
    num_classes = data_sets['train'].num_classes

    cclf = get_cclf(models[0][0], num_classes, models[0][1])
    if torch.cuda.is_available():
        cclf.cuda()
    cudnn.benchmark = True
    cclf.eval()

    train_val_loader = DataLoader(ConcatDataset([data_sets['train'], data_sets['val']]), batch_size=args.batch_size, shuffle=False, num_workers=8, pin_memory=True)
    
    # Step 1: calculate the pruning score
    loss_list = []

    ys = extract_ys(train_val_loader)
    zs = extract_zs(cclf, train_val_loader)

    class_centroids =  torch.zeros([num_classes, zs.size(1)], requires_grad=False).cuda()
    for c in range(num_classes):
        category_mask = ys == c
        idxs_c = category_mask.nonzero()
        class_centroids[c] = torch.mean(zs[idxs_c], dim=0)
    
    parameters_to_prune = []
    if args.arch in ['vit_small', 'vit_base']:
        for name, module in cclf.named_modules():
            if 'head' not in name and isinstance(module, torch.nn.Linear):
                parameters_to_prune.append((module, 'weight'))
    elif args.arch in ['resnet18', 'resnet50']:
        for name, module in cclf.named_modules():
            if isinstance(module, torch.nn.Conv2d):
                parameters_to_prune.append((module, 'weight'))
    else:
        raise Exception('NOT SUPPORTED ARCH')
    
    t1 = time.time()
    czs = zs
    for _ in range(freq):
    
        loss = cal_loss(class_centroids, czs, ys, num_classes)
        loss_list.append(loss.cpu().numpy())

        if args.masking == 'l1':
            prune.global_unstructured(parameters_to_prune, pruning_method=prune.L1Unstructured, amount=args.interval)
        else:
            prune.global_unstructured(parameters_to_prune, pruning_method=prune.RandomUnstructured, amount=args.interval)
        
        czs = extract_zs(cclf, train_val_loader)

    lss = np.stack(loss_list, axis=0)
    ls = np.average(lss, axis=0)
    print('Time (%s - %s - %s - %ss): ' % (args.dataset, args.pretrain, args.arch, str(time.time() - t1)))

    # scoring func with easy and hard sampling
    for mode in ['e', 'h']:
        idxs_dir = os.path.join(args.idxs_dir, 'seed_'+str(args.seed), args.dataset, 'lwot-%s-%s-i%s_f%s' % (args.masking, mode, str(int(inter * 1000)), str(freq)), 'pt_'+args.pretrain+'-arch_'+str(args.arch))
        os.makedirs(idxs_dir, exist_ok=True)

        np.save(os.path.join(idxs_dir, 'lss'), lss)
        np.save(os.path.join(idxs_dir, 'ls'), ls)

        for i in range(1, 10):
            budget = i / 10
            
            idxs = select_idxs_ead(ys, num_classes, ls, budget, mode)

            np.save(os.path.join(idxs_dir, str(budget)), idxs)
    
    for split in [0.05 * i for i in range(1, 20)]:

        idxs_dir = os.path.join(args.idxs_dir, 'seed_'+str(args.seed), args.dataset, 'lwot-%s-s%s-i%s_f%s' % (args.masking, str(int(split * 100)), str(int(inter * 1000)), str(freq)), 'pt_'+args.pretrain+'-arch_'+str(args.arch))
        os.makedirs(idxs_dir, exist_ok=True)

        np.save(os.path.join(idxs_dir, 'lss'), lss)
        np.save(os.path.join(idxs_dir, 'ls'), ls)

        for i in range(1, 10):
            budget = i / 10
            
            idxs = select_idxs(ys, num_classes, ls, budget, split)

            np.save(os.path.join(idxs_dir, str(budget)), idxs)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Pruning Score')
    parser.add_argument('--seed', default=1, type=int)
    parser.add_argument('--datasets_dir', default='../datasets', type=str)
    parser.add_argument('--ckpts_dir', type=str, default='./ckpts')
    parser.add_argument('--idxs_dir', type=str, default='./idxs')
    parser.add_argument('--dataset', type=str, default='CXRB10', choices=['CXRB10', 'DeepWeeds', 'DTD', 'FGVCAircraft', 'RESISC45', 'Sketch'])
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--arch', type=str, default='resnet18', choices=['resnet18', 'resnet50', 'vit_small', 'vit_base'])
    parser.add_argument('--pretrain', type=str, default='fully', choices=['fully', 'weakly'])

    # parser.add_argument('--split', type=float, default=0.5)
    parser.add_argument('--masking', type=str, default='l1', choices=['random', 'l1'])
    parser.add_argument('--interval', type=float, default=0.02)
    parser.add_argument('--frequency', type=int, default=50)

    args = parser.parse_args()
    main(args)