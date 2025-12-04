'''
Training script for CIFAR-10/100.
Copyright (c) Wei YANG, 2017
Modified by Vineeth S. Bhaskara and Sneha Desai (Winter 2019),
Included MNIST dataset in Training + Few Cosmetic changes.
'''
from __future__ import print_function

import argparse
import os
#os.environ['CUDA_VISIBLE_DEVICES']='0'
import shutil
import time
import random

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import models.cifar as models
from adam_all import *


from utils import Bar, Logger, AverageMeter, accuracy, mkdir_p, savefig

num_updates = 0
model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch CIFAR10/100, MNIST Training')
# Datasets
parser.add_argument('-d', '--dataset', default='cifar10', type=str)
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
# Optimization options
parser.add_argument('--epochs', default=300, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--updates', default=300, type=int, metavar='N',
                    help='number of total weight updates to run (if this is specified then overrides epochs). to turn off say -1')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--train-batch', default=128, type=int, metavar='N',
                    help='train batchsize')
parser.add_argument('--test-batch', default=512, type=int, metavar='N',
                    help='test batchsize')
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--eta1', '--eta1', default=1.0, type=float,
                    metavar='eta1', help='eta1')
parser.add_argument('--eta2', '--eta2', default=1.0, type=float,
                    metavar='eta2', help='eta2')
parser.add_argument('--drop', '--dropout', default=0, type=float,
                    metavar='Dropout', help='Dropout ratio')
parser.add_argument('--usecpu', '--usecpu', default=0, type=float,
                    metavar='1 if you want CPU', help='CPU usage')
parser.add_argument('--schedule', type=int, nargs='+', default=[150, 225],
                        help='Decrease learning rate at these epochs.')
parser.add_argument('--gamma', type=float, default=0.1, help='LR is multiplied by gamma on schedule.')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
# Checkpoints
parser.add_argument('-c', '--checkpoint', default='checkpoint', type=str, metavar='PATH',
                    help='path to save checkpoint (default: checkpoint)')
parser.add_argument('--opt', '--opt', default='adam', type=str, metavar='PATH',
                    help='opt to use (def adam). options: adam/adamucb/sgd/adamcb/adams')
parser.add_argument('--valevery', '--valevery', default=0, type=int, metavar='PATH',
                    help='non-zero to evaluate every <given> iterations evaluated only on the first mini-batch of the validation set. values saved under "iter" column in log.')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
# Architecture
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet20',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')
parser.add_argument('--depth', type=int, default=29, help='Model depth.')
parser.add_argument('--block-name', type=str, default='BasicBlock',
                    help='the building block for Resnet and Preresnet: BasicBlock, Bottleneck (default: Basicblock for cifar10/cifar100)')
parser.add_argument('--cardinality', type=int, default=8, help='Model cardinality (group).')
parser.add_argument('--widen-factor', type=int, default=4, help='Widen factor. 4 -> 64, 8 -> 128, ...')
parser.add_argument('--growthRate', type=int, default=12, help='Growth rate for DenseNet.')
parser.add_argument('--compressionRate', type=int, default=2, help='Compression Rate (theta) for DenseNet.')
# Miscs
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
#Device options
parser.add_argument('--gpu-id', default='0', type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')

args = parser.parse_args()
state = {k: v for k, v in args._get_kwargs()}

# Validate dataset
assert args.dataset == 'cifar10' or args.dataset == 'cifar100' or args.dataset == 'mnist' # 'Dataset can only be cifar10 or cifar100 or mnist.'

# Use CUDA
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
use_cuda = torch.cuda.is_available()

if args.usecpu == 1:
    use_cuda = False


# Random seed
if args.manualSeed is None:
    args.manualSeed = random.randint(1, 10000)
random.seed(args.manualSeed)
torch.manual_seed(args.manualSeed)
if use_cuda:
    torch.cuda.manual_seed_all(args.manualSeed)

best_acc = 0  # best test accuracy

def main():
    global best_acc
    start_epoch = args.start_epoch  # start from epoch 0 or last checkpoint epoch
    checkpoint_folder_suffix = '_' + str(args.opt) + '_bsz' + str(args.train_batch)
    if ('adamucb' in args.opt) or ('adams' in args.opt) or ('adamcb' in args.opt):
        checkpoint_folder_suffix = checkpoint_folder_suffix + '_' + str(args.eta1) + '_' + str(args.eta2)
    
    
    args.checkpoint = args.checkpoint + checkpoint_folder_suffix

    if not os.path.isdir(args.checkpoint):
        mkdir_p(args.checkpoint)



    # Data
    print('==> Preparing dataset %s' % args.dataset)
    if args.dataset == 'mnist':
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ])
            
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ])
       
    else:
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        
        
    if args.dataset == 'cifar10':
        dataloader = datasets.CIFAR10
        num_classes = 10
    elif args.dataset == 'mnist':
        dataloader = datasets.MNIST
        num_classes = 10
    else:
        dataloader = datasets.CIFAR100
        num_classes = 100
       


    trainset = dataloader(root='./data', train=True, download=True, transform=transform_train)
    trainloader = data.DataLoader(trainset, batch_size=args.train_batch, shuffle=True, num_workers=args.workers)

    testset = dataloader(root='./data', train=False, download=False, transform=transform_test)
    testloader = data.DataLoader(testset, batch_size=args.test_batch, shuffle=False, num_workers=args.workers)

    # Model
    print("==> creating model '{}'".format(args.arch))
    if args.arch.startswith('resnext'):
        model = models.__dict__[args.arch](
                    cardinality=args.cardinality,
                    num_classes=num_classes,
                    depth=args.depth,
                    widen_factor=args.widen_factor,
                    dropRate=args.drop,
                )
    elif args.arch.startswith('densenet'):
        model = models.__dict__[args.arch](
                    num_classes=num_classes,
                    depth=args.depth,
                    growthRate=args.growthRate,
                    compressionRate=args.compressionRate,
                    dropRate=args.drop,
                )
    elif args.arch.startswith('wrn'):
        model = models.__dict__[args.arch](
                    num_classes=num_classes,
                    depth=args.depth,
                    widen_factor=args.widen_factor,
                    dropRate=args.drop,
                )
    elif args.arch.endswith('resnet'):
        model = models.__dict__[args.arch](
                    num_classes=num_classes,
                    depth=args.depth,
                    block_name=args.block_name,
                )
    elif args.arch.startswith('cifar10cnn'):
        print('Adam paper CIFAR model with Dropout (Architecture same as in Original Adam Paper)')
        if args.drop == 1:
            model = models.__dict__[args.arch](num_classes=num_classes, dropout=True)
        else:
            model = models.__dict__[args.arch](num_classes=num_classes)
    
    elif args.arch.startswith('mnistmlp'):
        print('MNIST model with Dropout (Architecture same as in Original Adam Paper)')
        if args.drop == 1:
            model = models.__dict__[args.arch](num_classes=num_classes, dropout=True)
        else:
            model = models.__dict__[args.arch](num_classes=num_classes)

    else:
        model = models.__dict__[args.arch](num_classes=num_classes)

    if use_cuda:
        model = torch.nn.DataParallel(model).cuda()
    else:
        model = torch.nn.DataParallel(model)
    cudnn.benchmark = True
    print('    Total params: %.2fM' % (sum(p.numel() for p in model.parameters())/1000000.0))
    criterion = nn.CrossEntropyLoss()
    
    if args.opt == 'sgd':
        print('Doing SGD')
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    elif args.opt == 'adamucb':
        print('Doing AdamUCB with etas: {}, {}'.format(args.eta1, args.eta2))
        optimizer = AdamALL(model.parameters(), lr=args.lr, etas=(args.eta1, args.eta2), weight_decay=args.weight_decay)
    elif args.opt == 'adam':
        print('Doing Adam')
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.opt == 'adamcb':
        print('Doing AdamCB with Bounded StdDev with etas: {}, {}'.format(args.eta1, args.eta2))
        optimizer = AdamALL(model.parameters(), lr=args.lr, etas=(args.eta1, args.eta2), bound_stddev=True, weight_decay=args.weight_decay)
    elif args.opt == 'adams':
        print('Doing AdamS')
        optimizer = AdamALL(model.parameters(), lr=args.lr, etas=(args.eta1, args.eta2), bound_stddev=False, weight_decay=args.weight_decay, randomize_eta2=True)
        

    # Resume
    title = args.dataset + '-' + args.arch
    if args.resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isfile(args.resume), 'Error: no checkpoint directory found!'
        args.checkpoint = os.path.dirname(args.resume)
        checkpoint = torch.load(args.resume)
        best_acc = checkpoint['best_acc']
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        logger = Logger(os.path.join(args.checkpoint, 'log.txt'), title=title, resume=True)
    else:
        logger = Logger(os.path.join(args.checkpoint, 'log.txt'), title=title)
        logger.set_names(['Epoch', 'Learning Rate', 'Train Loss', 'Valid Loss', 'Train Acc.', 'Valid Acc.'])
        logger2 = Logger(os.path.join(args.checkpoint, 'log_iter.txt'), title=title)
        logger2.set_names(['Iter','Learning Rate', 'Train Loss', 'Valid Loss', 'Train Acc.', 'Valid Acc.'])


    if args.evaluate:
        print('\nEvaluation only')
        test_loss, test_acc = test(testloader, model, criterion, start_epoch, use_cuda)
        print(' Test Loss:  %.8f, Test Acc:  %.2f' % (test_loss, test_acc))
        return

    # Train and val
    for epoch in range(start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch)

        print('\nEpoch: [%d | %d] LR: %f' % (epoch + 1, args.epochs, state['lr']))
        if args.valevery != 0:
            train_loss, train_acc = train(trainloader, model, criterion, optimizer, epoch, use_cuda, testloader=testloader, logger2=logger2)
        else:
            train_loss, train_acc = train(trainloader, model, criterion, optimizer, epoch, use_cuda, logger2=logger2)
        test_loss, test_acc = test(testloader, model, criterion, epoch, use_cuda)

        # append logger file
        logger.append([epoch, state['lr'], train_loss, test_loss, train_acc, test_acc])

        # save model
        is_best = test_acc > best_acc
        best_acc = max(test_acc, best_acc)
        save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'acc': test_acc,
                'best_acc': best_acc,
                'optimizer' : optimizer.state_dict(),
            }, is_best, checkpoint=args.checkpoint)
                
        if (args.updates != -1) and (args.updates == num_updates):
            break

    logger.close()
    logger2.close()
    logger2.plot(imgname='iter.png')
    logger.plot(imgname='epoch.png')
    savefig(os.path.join(args.checkpoint, 'log.eps'))

    print('Best acc:')
    print(best_acc)

def train(trainloader, model, criterion, optimizer, epoch, use_cuda, testloader=None, logger2=None):
    global num_updates
    # switch to train mode
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    end = time.time()

    bar = Bar('Processing', max=len(trainloader))
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        # measure data loading time
        data_time.update(time.time() - end)

        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda(non_blocking=True)
        inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)

        # compute output
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        if torch.isnan(loss):
            print('Loss for current iteration {} is NaN!'.format(num_updates+1))

        # measure accuracy and record loss
        prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
        losses.update(loss.item(), inputs.size(0))
        top1.update(float(prec1), inputs.size(0))
        top5.update(float(prec5), inputs.size(0))

        # compute gradient and do SGD step
        def closure():
            return loss
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step(closure)
        
        num_updates = num_updates + 1

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # plot progress
        bar.suffix  = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f}'.format(
                    batch=batch_idx + 1,
                    size=len(trainloader),
                    data=data_time.avg,
                    bt=batch_time.avg,
                    total=bar.elapsed_td,
                    eta=bar.eta_td,
                    loss=losses.avg,
                    top1=top1.avg,
                    top5=top5.avg,
                    )
        bar.next()
        
        # append logger 2 file
        if testloader is not None and (num_updates % args.valevery == 0):
            test_loss, test_acc = test(testloader, model, criterion, epoch, use_cuda, silent=True)
            model.train()
        else:
            test_loss = 0
            test_acc = 0
        logger2.append([num_updates, state['lr'], loss.item(), test_loss, float(prec1), test_acc])
        
        if (args.updates != -1) and (args.updates == num_updates):
            break
    bar.finish()
    return (losses.avg, top1.avg)

def test(testloader, model, criterion, epoch, use_cuda, silent=False):
    global best_acc

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    
    if not silent:
        bar = Bar('Processing', max=len(testloader))
        
    for batch_idx, (inputs, targets) in enumerate(testloader):
        # measure data loading time
        data_time.update(time.time() - end)

        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = torch.autograd.Variable(inputs, volatile=True), torch.autograd.Variable(targets)

        # compute output
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
        losses.update(loss.item(), inputs.size(0))
        top1.update(float(prec1), inputs.size(0))
        top5.update(float(prec5), inputs.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        


        # plot progress
        if not silent:
            bar.suffix  = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f}'.format(
                        batch=batch_idx + 1,
                        size=len(testloader),
                        data=data_time.avg,
                        bt=batch_time.avg,
                        total=bar.elapsed_td,
                        eta=bar.eta_td,
                        loss=losses.avg,
                        top1=top1.avg,
                        top5=top5.avg,
                        )
            bar.next()
            
            
        # if silent then exit
        if silent:
            #break # testset minibatches are not shuffled!
#             with open('./test2.txt', 'a') as f:
#                 f.write('\n\n')
#                 f.write(str(targets))
            break
            
    if not silent:
        bar.finish()
    return (losses.avg, top1.avg)

def save_checkpoint(state, is_best, checkpoint='checkpoint', filename='checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'model_best.pth.tar'))

def adjust_learning_rate(optimizer, epoch):
    global state
    if epoch in args.schedule:
        state['lr'] *= args.gamma
        for param_group in optimizer.param_groups:
            param_group['lr'] = state['lr']

if __name__ == '__main__':
    num_updates = 0
    main()
