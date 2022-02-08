"""
https://github.com/Originofamonia/vinbigdata_yolov5/blob/main/yolov5/train.py
unsuccessful attempt to learn DDP from previous yolo
https://github.com/yangkky/distributed_tutorial/blob/master/src/mnist-distributed.py
dataset
"""
from cProfile import label
import os
import argparse
from pathlib import Path
import random
import math
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader, DistributedSampler
from torchvision import datasets, transforms
from torch.cuda import amp
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm

from ddp_torch import setup, cleanup, run_ddp


def create_data_loaders(rank: int,
                        world_size: int,
                        batch_size: int):
    
    dataset_loc = './'

    train_dataset = datasets.CIFAR10(dataset_loc,
                                   download=True,
                                   train=True,
                                transform=transforms.ToTensor(),)
    sampler = DistributedSampler(train_dataset,
                                 num_replicas=world_size,  # Number of GPUs
                                 rank=rank,  # GPU where process is running
                                 shuffle=True,  # Shuffling is done by Sampler
                                 seed=444)
    train_loader = DataLoader(train_dataset,
                              batch_size=batch_size,
                              shuffle=False,  # This is mandatory to set this to False here, shuffling is done by Sampler
                              num_workers=4,
                              sampler=sampler,
                              pin_memory=True)

    # This is not necessary to use distributed sampler for the test or validation sets.
    test_dataset = datasets.CIFAR10(dataset_loc,
                                  download=True,
                                  train=False,
                                  transform=transforms.ToTensor(),)
    test_loader = DataLoader(test_dataset,
                             batch_size=batch_size,
                             shuffle=False,
                             num_workers=4,
                             pin_memory=True)

    return train_loader, test_loader


class ConvNet(nn.Module):
    def __init__(self, num_classes=10):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.fc = nn.Linear(2048, num_classes)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        return out


def train(rank, world_size, opt):
    print(f'run DDP example on cifar10 rank: {rank}')
    setup(rank, world_size)

    model = ConvNet().to(rank)
    ddp_model = DDP(model, device_ids=[rank])

    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.SGD(ddp_model.parameters(), lr=1e-3)

    train_loader, test_loader = create_data_loaders(rank, world_size, opt.batch_size)

    for e in range(opt.epochs):
        pbar = tqdm(train_loader)
        for j, batch in enumerate(pbar):
            batch = tuple(item.to(opt.device) for item in batch)
            imgs, labels = batch
            
            optimizer.zero_grad()
            outputs = ddp_model(imgs)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

            if j % 100 == 0:
                desc = f'Epoch: {e}/{opt.epochs}, step: {j}, loss: {loss.item():.3f}'
                pbar.set_description(desc)

    cleanup()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='yolov5x.pt', help='initial weights path')
    parser.add_argument('--cfg', type=str, default='models/yolov5x.yaml', help='model.yaml path')
    parser.add_argument('--data', type=str, default='/home/qiyuan/2021summer/vinbigdata/vinbigdata.yaml', help='data.yaml path')
    parser.add_argument('--model', type=str, default='yolo',
                        help='model for datasets')
    parser.add_argument('--hyp', type=str, default='data/hyp.scratch.yaml', help='hyperparameters path')
    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=64, help='total batch size for all GPUs')
    parser.add_argument('--img-size', nargs='+', type=int, default=[640, 640], help='[train, test] image sizes')
    parser.add_argument('--rect', action='store_true', help='rectangular training')
    parser.add_argument('--resume', nargs='?', const=True, default=False, help='resume most recent training')
    parser.add_argument('--nosave', action='store_true', help='only save final checkpoint')
    parser.add_argument('--notest', action='store_true', help='only test final epoch')
    parser.add_argument('--noautoanchor', action='store_true', help='disable autoanchor check')
    parser.add_argument('--evolve', action='store_true', help='evolve hyperparameters')
    parser.add_argument('--augment', default=True,
                        help='augment data')
    parser.add_argument('--verbose', default=True,
                        help='print results per class')
    parser.add_argument('--save_txt', default=False,
                        help='auto labelling')
    parser.add_argument('--save_hybrid', default=False,
                        help='auto labelling')
    parser.add_argument('--save_conf', default=False,
                        help='save auto-label confidences')
    parser.add_argument('--bucket', type=str, default='', help='gsutil bucket')
    parser.add_argument('--cache-images', action='store_true', help='cache images for faster training')
    parser.add_argument('--image-weights', action='store_true', help='use weighted image selection for training')
    parser.add_argument('--device', default='cuda', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--multi-scale', action='store_true', help='vary img-size +/- 50%%')
    parser.add_argument('--single-cls', action='store_true', help='train multi-class data as single-class')
    parser.add_argument('--adam', action='store_true', help='use torch.optim.Adam() optimizer')
    parser.add_argument('--sync-bn', action='store_true', help='use SyncBatchNorm, only available in DDP mode')
    parser.add_argument('--local_rank', type=int, default=-1, help='DDP parameter, do not modify')
    parser.add_argument('--log-imgs', type=int, default=16, help='number of images for W&B logging, max 100')
    parser.add_argument('--log-artifacts', action='store_true', help='log artifacts, i.e. final trained model')
    parser.add_argument('--workers', type=int, default=4, help='maximum number of dataloader workers')
    parser.add_argument('--project', default='runs/train', help='save to project/name')
    parser.add_argument('--name', default='exp', help='save to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    opt = parser.parse_args()
    n_gpus = torch.cuda.device_count()
    assert n_gpus >= 2, f"Requires at least 2 GPUs to run, but got {n_gpus}"
    opt.world_size = n_gpus

    run_ddp(train, opt)


if __name__ == '__main__':
    main()
