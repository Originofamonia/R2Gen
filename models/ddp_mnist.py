"""
https://github.com/Originofamonia/vinbigdata_yolov5/blob/main/yolov5/train.py
unsuccessful attempt to learn DDP from previous yolo
"""
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


def select_device(device='', batch_size=None):
    # device = 'cpu' or '0' or '0,1,2,3'
    cpu_request = device.lower() == 'cpu'
    if device and not cpu_request:  # if device requested other than 'cpu'
        os.environ['CUDA_VISIBLE_DEVICES'] = device  # set environment variable
        assert torch.cuda.is_available(), f'CUDA unavailable, invalid device {device} requested'  # check availablity

    cuda = False if cpu_request else torch.cuda.is_available()
    if cuda:
        c = 1024 ** 2  # bytes to MB
        ng = torch.cuda.device_count()
        if ng > 1 and batch_size:  # check that batch_size is compatible with device_count
            assert batch_size % ng == 0, f'batch-size {batch_size} not multiple of GPU count {ng}'
        x = [torch.cuda.get_device_properties(i) for i in range(ng)]
        s = f'Using torch {torch.__version__} '
        for i, d in enumerate((device or '0').split(',')):
            if i == 1:
                s = ' ' * len(s)
            # logger.info(f"{s}CUDA:{d} ({x[i].name}, {x[i].total_memory / c}MB)")
    # else:
        # logger.info(f'Using torch {torch.__version__} CPU')

    # logger.info('')  # skip a line
    return torch.device(f'cuda:0' if cuda else 'cpu')


def init_seeds(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    init_torch_seeds(seed)


def init_torch_seeds(seed=0):
    # Speed-reproducibility tradeoff https://pytorch.org/docs/stable/notes/randomness.html
    torch.manual_seed(seed)
    if seed == 0:  # slower, more reproducible
        cudnn.deterministic = True
        cudnn.benchmark = False
    else:  # faster, less reproducible
        cudnn.deterministic = False
        cudnn.benchmark = True


@contextmanager
def torch_distributed_zero_first(local_rank: int):
    """
    Decorator to make all processes in distributed training wait for each local_master to do something.
    """
    if local_rank not in [-1, 0]:
        torch.distributed.barrier()
    yield
    if local_rank == 0:
        torch.distributed.barrier()


def create_data_loaders(rank: int,
                        world_size: int,
                        batch_size: int) -> Tuple[DataLoader, DataLoader]:
    
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


def train(hyp, opt, device):
    save_dir, epochs, batch_size, total_batch_size, weights, rank = \
        Path(opt.save_dir), opt.epochs, opt.batch_size, opt.total_batch_size, opt.weights, opt.global_rank
    
    cuda = device.type != 'cpu'
    pretrained = weights.endswith('.pt')

    model = ConvNet().to(device)

    if opt.adam:
        optimizer = optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.999))  # adjust beta1 to momentum
    else:
        optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.9, nesterov=True)
    
    lf = lambda x: ((1 + math.cos(x * math.pi / epochs)) / 2) * (1 - 0.2) + 0.2  # cosine
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

    # SyncBatchNorm
    if opt.sync_bn and cuda and rank != -1:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(device)
        print('Using SyncBatchNorm()')



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='yolov5x.pt', help='initial weights path')
    parser.add_argument('--cfg', type=str, default='models/yolov5x.yaml', help='model.yaml path')
    parser.add_argument('--data', type=str, default='/home/qiyuan/2021summer/vinbigdata/vinbigdata.yaml', help='data.yaml path')
    parser.add_argument('--model', type=str, default='yolo',
                        help='model for datasets')
    parser.add_argument('--hyp', type=str, default='data/hyp.scratch.yaml', help='hyperparameters path')
    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--batch-size', type=int, default=8, help='total batch size for all GPUs')
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
    parser.add_argument('--device', default='0,1', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
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

    # Set DDP variables
    opt.total_batch_size = opt.batch_size
    opt.world_size = int(os.environ['WORLD_SIZE']) if 'WORLD_SIZE' in os.environ else 1
    opt.global_rank = int(os.environ['RANK']) if 'RANK' in os.environ else -1
    # set_logging(opt.global_rank)
    # if opt.global_rank in [-1, 0]:
    #     check_git_status()

    # DDP mode
    device = select_device(opt.device, batch_size=opt.batch_size)
    if opt.local_rank != -1:
        assert torch.cuda.device_count() > opt.local_rank
        torch.cuda.set_device(opt.local_rank)
        device = torch.device('cuda', opt.local_rank)
        dist.init_process_group(backend='nccl', init_method='env://')  # distributed backend
        assert opt.batch_size % opt.world_size == 0, '--batch-size must be multiple of CUDA device count'
        opt.batch_size = opt.total_batch_size // opt.world_size
    
    train(None, opt, device)
