import os
import time
import random
import shutil
import argparse

import numpy as np
import pandas as pd

import torch
torch.multiprocessing.set_sharing_strategy('file_system')
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader, ConcatDataset, Subset
import torchvision.transforms as transforms

from datasets import get_dataset, get_mu_std
from models import get_pclf
from utils import setup_logger


def init_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_in_epoch(data_loader, model, criterion, optimizer):

    since = time.time()

    model.train()

    running_loss, correct, total = 0.0, 0, 0

    for data in data_loader:
        # dynamic batch pruning

        inputs, labels = data
        inputs, labels = Variable(inputs.float().cuda()), Variable(labels.long().cuda())

        # forward
        logits = model(inputs)
        loss = criterion(logits, labels)

        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # evaluate
        with torch.no_grad():
            _, preds = logits.max(dim=1)
            running_loss += loss.item()
            correct += preds.eq(labels).sum().item()
        total += inputs.size(0)

    # print on average
    time_elapsed = time.time() - since
    # print('[loss: {:.8f} | acc: {:.4f}% | time: {:.0f}m {:.0f}s]'.format(running_loss / total, 100. * correct / total, time_elapsed // 60, time_elapsed % 60))
    
    return time_elapsed


def validate(data_loader, model):
    # evaluate the model
    model.eval()

    correct, total = 0, 0

    for data in data_loader:
        inputs, labels = data
        inputs, labels = Variable(inputs.float().cuda()), Variable(labels.long().cuda())

        with torch.no_grad():
            logits = model(inputs)
        
            _, preds = logits.max(dim=1)
            correct += preds.eq(labels).sum().item()
            total += inputs.size(0)

    return 100. * correct / total


def individual_acc(num_classes, data_laoder, model):

    confusion_matrix = torch.zeros(num_classes, num_classes)
    model.eval()

    # for i, (inputs, classes) in enumerate(data_laoder):
    for data in data_laoder:
        inputs, labels = data
        inputs, labels = Variable(inputs.float().cuda()), Variable(labels.long().cuda())

        with torch.no_grad():
            logits = model(inputs)

            _, preds = logits.max(dim=1)

        for t, p in zip(labels.view(-1), preds.view(-1)):
            confusion_matrix[t.long(), p.long()] += 1

    # return torch.sort(confusion_matrix.diag()/confusion_matrix.sum(1))[0]
    # return confusion_matrix.diag()/confusion_matrix.sum(1)
    accs = pd.DataFrame((confusion_matrix.diag() / confusion_matrix.sum(1)).numpy())
    
    return accs


def main(args):

    # init seed
    init_seed(args.seed)

    pclf_name = args.pretrain + '-' + args.arch
    print(pclf_name)

    args.ratio /= 100
    # specify output dir
    if args.pruning:
        # training with part samples
        # exp_dir = os.path.join(args.principle+'_'+args.pruning_mode+'-'+str(args.ratio), 'seed_'+str(args.seed))
        exp_dir = os.path.join(args.ckpts_dir, 'seed_'+str(args.seed), args.dataset+'-'+pclf_name, args.principle+'-'+str(args.ratio))
    else:
        # training with all samples
        # exp_dir = os.path.join(args.output_dir, args.dataset, configs['encoder']+'-'+args.paradigm+'-'+args.pretrain, args.exp_name, 'seed_'+str(args.seed))
        exp_dir = os.path.join(args.ckpts_dir, 'seed_'+str(args.seed), args.dataset+'-'+pclf_name, 'all')
    
    os.makedirs(exp_dir, exist_ok=True)

    ckpt_path = os.path.join(exp_dir, '50.pth')
    if os.path.isfile(ckpt_path):
        print('Done.')
        exit()
    
    setup_logger(str(exp_dir), 'outputs.log')

    # load dataset
    # dataset preprocessing
    # mean, std = get_mu_std(args.datasets_dir, args.dataset, 'train')
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

    data_sets = {mode : get_dataset(args.datasets_dir, args.dataset, mode, data_transforms[mode]) for mode in ['train', 'val', 'test']}
    num_classes = data_sets['train'].num_classes

    # instantiate model
    clf = get_pclf(args.arch, args.pretrain, num_classes)
    # clf = nn.DataParallel(clf)

    if torch.cuda.is_available():
        clf.cuda()
    cudnn.benchmark = True

    # criterion
    criterion = nn.CrossEntropyLoss()

    # set optimizer and scheduler
    lr_stones = [args.epochs * float(lr_stone) for lr_stone in args.lr_stones]
    optimizer = optim.SGD(clf.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=args.momentum, nesterov=True)
    # optimizer = optim.Adam(clf.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    print('LR: {:.5f} - WD: {:.5f} - Mom: {:.2f} - Nes: True - LMS: {}'.format(args.lr, args.weight_decay, args.momentum, args.lr_stones))
    if args.scheduler == 'multistep':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=lr_stones, gamma=0.1)
    else:
        raise Exception('NOT SUPPORTED SCHEDULER')
    
    # offline pruning
    if args.pruning:

        idxs_path = os.path.join(args.idxs_dir, 'seed_'+str(args.seed), args.dataset, args.principle, 'pt_'+args.pretrain+'-arch_'+args.arch, str(args.ratio)+'.npy')
        idxs = np.load(idxs_path).astype(int)
        
        data_loaders = {
            'train': DataLoader(Subset(ConcatDataset([data_sets['train'], data_sets['val']]), idxs), batch_size=args.batch_size, shuffle=True, num_workers=8),
            'test': DataLoader(data_sets['test'], batch_size=args.batch_size, shuffle=False, num_workers=8)
        }
    else:
        data_loaders = {
            'train': DataLoader(ConcatDataset([data_sets['train'], data_sets['val']]), batch_size=args.batch_size, shuffle=True, num_workers=8),
            'test': DataLoader(data_sets['test'], batch_size=args.batch_size, shuffle=False, num_workers=8)
        }

    duration = 0.0
    start_epoch, best_epoch = 1, 0
    cla_acc, best_acc = 0.0, 0.0
    
    for epoch in range(start_epoch, args.epochs + 1):
    
        # record the training process of full dataset training
        if not args.pruning:
            ckpt_name = str(epoch-1) + '.pth'
            cla_path = os.path.join(exp_dir, ckpt_name)
            torch.save(clf.state_dict(), str(cla_path))
        
        duration += train_in_epoch(data_loaders['train'], clf, criterion, optimizer)
        scheduler.step()
        
        cla_acc = validate(data_loaders['test'], clf)

        if cla_acc > best_acc:
            best_acc = cla_acc
            best_epoch = epoch

        if epoch == args.epochs:
            ckpt_name = str(epoch) + '.pth'
            # ckpt_name = 'val_best.pth'
            cla_path = os.path.join(exp_dir, ckpt_name)
            torch.save(clf.state_dict(), str(cla_path))

            # save the accs from confusion matrix
            individual_acc(num_classes, data_loaders['test'], clf).to_csv(os.path.join(exp_dir, 'accs.csv'))
            
            print('---> Acc: {:.4f}% - {:.4f}% e_{} | Time: {:.0f}h {:.0f}m {:.0f}s'.format(cla_acc, best_acc, str(best_epoch), duration // 3600, (duration % 3600) // 60, duration % 60))
            exit()
        
        

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='fully fine-tuning')
    parser.add_argument('--seed', default=1, type=int)
    parser.add_argument('--ckpts_dir', type=str, default='./ckpts')

    parser.add_argument('--datasets_dir', default='../datasets')
    parser.add_argument('--dataset', type=str, default='DeepWeeds', choices=['CXRB10', 'DeepWeeds', 'DTD', 'FGVCAircraft', 'Sketch'])

    # pruning
    parser.add_argument('--pruning', action='store_true', help='prune the full dataset (default: false)')
    parser.add_argument('--principle', type=str)
    parser.add_argument('--idxs_dir', type=str, default='./idxs')
    parser.add_argument('--ratio', type=float, default=10)

    parser.add_argument('--pretrain', type=str, default='weakly', choices=['weakly', 'fully'])
    parser.add_argument('--arch', type=str, default='vit_tiny', choices=['resnet18', 'resnet50', 'vit_small', 'vit_base'])
    
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--scheduler', type=str, default='multistep', choices=['multistep'])
    parser.add_argument('--lr_stones', nargs='+', default=[0.5, 0.75]) # specify for multistep scheduler
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--batch_size', type=int, default=128)
    
    args = parser.parse_args()

    main(args)
