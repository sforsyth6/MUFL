import argparse
import os
import sys
import shutil
import time

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms

from MalConv import MalConv
from binaryLoader import BinaryDataset, RandomChunkSampler, pad_collate_func
import multiprocessing
from torch.utils.data import Dataset, DataLoader, Subset

import datasets
import models
import math

from lib.NCEAverage import NCEAverage
from lib.LinearAverage import LinearAverage
from lib.NCECriterion import NCECriterion
from lib.utils import AverageMeter
from test import NN, kNN

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet18',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--lr', '--learning-rate', default=0.03, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--test-only', action='store_true', help='test only')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--world-size', default=1, type=int,
                    help='number of distributed processes')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='gloo', type=str,
                    help='distributed backend')
parser.add_argument('--low-dim', default=128, type=int,
                    metavar='D', help='feature dimension')
parser.add_argument('--nce-t', default=0.07, type=float, 
                    metavar='T', help='temperature parameter for softmax')
parser.add_argument('--nce-m', default=0.5, type=float,
                    help='momentum for non-parametric updates')
parser.add_argument('--iter_size', default=1, type=int,
                    help='caffe style iter size')

parser.add_argument('--filter_size', type=int, default=512, help='How wide should the filter be')
parser.add_argument('--filter_stride', type=int, default=512, help='Filter Stride')
parser.add_argument('--embd_size', type=int, default=8, help='Size of embedding layer')
parser.add_argument('--num_channels', type=int, default=128, help='Total number of channels in output')
parser.add_argument('--non-neg', type=bool, default=False, help='Should non-negative training be used')
parser.add_argument('-b','--batch_size', type=int, default=128, help='Batch size during training')
#Default is set ot 16 MB! 
parser.add_argument('--max_len', type=int, default=16000000, help='Maximum length of input file in bytes, at which point files will be truncated')

parser.add_argument('--gpus', nargs='+', type=int)
parser.add_argument('--outfile', type=str, default='checkpoints/checkpoint.pth.tar')
parser.add_argument('--results', type=str, default='default_out.txt')

best_prec1 = 0
args = parser.parse_args()

def main():
    global args, best_prec1
    args = parser.parse_args()

    GPUS = args.gpus

    NON_NEG = args.non_neg
    EMBD_SIZE = args.embd_size
    FILTER_SIZE = args.filter_size
    FILTER_STRIDE = args.filter_stride
    NUM_CHANNELS= args.num_channels
    EPOCHS = args.epochs
    MAX_FILE_LEN = args.max_len

    BATCH_SIZE = args.batch_size

    args.distributed = args.world_size > 1

    if args.distributed:
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size)

        # create model
    model = MalConv(out_size=args.low_dim, channels=NUM_CHANNELS, window_size=FILTER_SIZE, stride=FILTER_STRIDE, embd_size=EMBD_SIZE).cuda()

    model = torch.nn.DataParallel(model).cuda()

    # Data loading code
    whole_dataset = BinaryDataset(args.data + "train/", sort_by_size=True, max_len=MAX_FILE_LEN )
    test_dataset = BinaryDataset(args.data + "test/", sort_by_size=True, max_len=MAX_FILE_LEN )

    loader_threads =  max(multiprocessing.cpu_count()-4, multiprocessing.cpu_count()//2+1) - 8
    loader_threads =  8


    train_loader = DataLoader(whole_dataset, batch_size=BATCH_SIZE, num_workers=loader_threads, collate_fn=pad_collate_func, 
                            sampler=RandomChunkSampler(whole_dataset,BATCH_SIZE))

    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, num_workers=loader_threads, collate_fn=pad_collate_func, 
                            sampler=RandomChunkSampler(test_dataset,BATCH_SIZE))


    # define lemniscate and loss function (criterion)
    ndata = whole_dataset.__len__()

    lemniscate = NCEAverage(args.low_dim, ndata, args.nce_t, args.nce_m, batchSize=args.batch_size).cuda()
    criterion = NCECriterion().cuda()


    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            lemniscate = checkpoint['lemniscate']
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    if args.evaluate:
        kNN(0, model, lemniscate, train_loader, test_loader, 200, args.nce_t, args.results)
        return

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        adjust_learning_rate(optimizer, epoch)

        # train for one epoch
        train(train_loader, model, lemniscate, criterion, optimizer, epoch)

        # evaluate on validation set
        prec1 = NN(epoch, model, lemniscate, train_loader, test_loader, args.results)

        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'lemniscate': lemniscate,
            'best_prec1': best_prec1,
            'optimizer' : optimizer.state_dict(),
        }, is_best)
    # evaluate KNN after last epoch
    kNN(0, model, lemniscate, train_loader, test_loader, 2, args.nce_t, args.results)


def train(train_loader, model, lemniscate, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    optimizer.zero_grad()
    for i, (inputs, targets, index) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        # compute output
        features = model(inputs)
        x, penult, post_conv = features
        index = torch.tensor(index).cuda()
        output = lemniscate(x, index)
        loss = criterion(output, index) / args.iter_size

        loss.backward()

        # measure accuracy and record loss
        losses.update(loss.item() * args.iter_size, inputs.size(0))

        if (i+1) % args.iter_size == 0:
            # compute gradient and do SGD step
            optimizer.step()
            optimizer.zero_grad()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            with open("results/" + args.results, '+a') as f:
                print('Epoch: [{0}][{1}/{2}]\t'
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                    'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                    epoch, i, len(train_loader), batch_time=batch_time,
                    data_time=data_time, loss=losses), file=f)


def save_checkpoint(state, is_best, filename=args.outfile):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')

def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 100 epochs"""
    lr = args.lr
    if epoch < 120:
        lr = args.lr
    elif epoch >= 120 and epoch < 160:
        lr = args.lr * 0.1
    else:
        lr = args.lr * 0.01
    #lr = args.lr * (0.1 ** (epoch // 100))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


if __name__ == '__main__':
    main()
