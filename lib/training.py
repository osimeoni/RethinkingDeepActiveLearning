import os
import torch
import random
import numpy as np
import torchvision.datasets

from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from lib import samplers, data, utils

def get_train_loader(args, dataset, weights=None):
    '''
    Loader creation in case of label propagation
    '''

    if args.labeled_batch_size:
        if weights:
            batch_sampler = samplers.WeightedTwoStreamBatchSampler(
                                    dataset.pseudo_label_idx, dataset.labeled_idxs,
                                    args.batch_size, args.labeled_batch_size,
                                    weights)
        else:
            batch_sampler = data.TwoStreamBatchSampler(
                                    dataset.pseudo_label_idx, dataset.labeled_idxs,
                                    args.batch_size, args.labeled_batch_size)
    elif args.exclude_unlabeled:
        sampler = SubsetRandomSampler(dataset.labeled_idxs)
        batch_sampler = BatchSampler(sampler, args.batch_size, drop_last=True)
    else:
        raise ValueError("Should only be used in the case of label propagation")

    train_loader = torch.utils.data.DataLoader(dataset,
                                               batch_sampler=batch_sampler,
                                               num_workers=args.workers,
                                               pin_memory=True,
                                               worker_init_fn=np.random.seed(7))

    return train_loader

def update_batch_sampler(args, dataset, weights=None):
    '''
    Loader creation in case of label propagation
    '''

    if args.labeled_batch_size:
        if weights:
            batch_sampler = samplers.WeightedTwoStreamBatchSampler(
                                    dataset.pseudo_label_idx, dataset.labeled_idxs,
                                    args.batch_size, args.labeled_batch_size,
                                    weights)
        else:
            batch_sampler = data.TwoStreamBatchSampler(
                                    dataset.pseudo_label_idx, dataset.labeled_idxs,
                                    args.batch_size, args.labeled_batch_size)
    elif args.exclude_unlabeled:
        sampler = SubsetRandomSampler(dataset.labeled_idxs)
        batch_sampler = BatchSampler(sampler, args.batch_size, drop_last=True)
    else:
        raise ValueError("Should only be used in the case of label propagation")

    return batch_sampler

def create_data_loaders(args,
                        train_transformation,
                        eval_transformation,
                        datadir):

    traindir = os.path.join(datadir, args.train_subdir)
    if args.eval_subdir:
        evaldir = os.path.join(datadir, args.eval_subdir)
    testdir = os.path.join(datadir, args.test_subdir)

    utils.assert_exactly_one([args.exclude_unlabeled, args.labeled_batch_size])
    dataset = data.DatasetFolder(traindir, train_transformation)

    if args.split is not None:
        with open(args.labels) as f:
            labels = dict(line.split(' ') for line in f.read().splitlines())
        labeled_idxs, unlabeled_idxs = data.relabel_dataset(dataset, labels)
    else:
        raise ValueError('Not implemented.')

    if args.exclude_unlabeled:
        sampler = SubsetRandomSampler(labeled_idxs)
        batch_sampler = BatchSampler(sampler, args.batch_size, drop_last=True)

    # Construct batch_sampler
    elif args.add_lp:
        # train_loader will be defined in case of label propagation later
        batch_sampler = None
        if args.lp_mode == 'full':
            if args.weighted_unlabeled_batch:
                print('Initialize with weights at 1')
                weights = [1]*len(unlabeled_idxs)
                batch_sampler = samplers.WeightedTwoStreamBatchSampler(
                                        unlabeled_idxs, dataset.labeled_idxs,
                                        args.batch_size, args.labeled_batch_size,
                                        weights)
            else:
                batch_sampler = data.TwoStreamBatchSampler(
                    unlabeled_idxs, dataset.labeled_idxs, args.batch_size,
                    args.labeled_batch_size)
        else:
            frag_N = int(round(len(unlabeled_idxs) * args.lp_percent))
            print('Label propagation: initialize partial selection with %d random indices'%frag_N)
            sel = np.random.permutation(unlabeled_idxs)[:frag_N]
            batch_sampler = data.TwoStreamBatchSampler(
                                    sel, dataset.labeled_idxs,
                                    args.batch_size, args.labeled_batch_size)
    else:
        assert False, "labeled batch size {}".format(args.labeled_batch_size)

    print('\nNumber of unlabeled images: %d'%len(unlabeled_idxs))
    print('Number of already labeled images: %d\n'%len(labeled_idxs))

    if batch_sampler is not None:
        train_loader = torch.utils.data.DataLoader(dataset,
                                                   batch_sampler=batch_sampler,
                                                   num_workers=args.workers,
                                                   pin_memory=True,
                                                   worker_init_fn=np.random.seed(7))
    else:
        train_loader = None

    train_loader_noshuff = torch.utils.data.DataLoader(dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,  # Needs images twice as fast
        pin_memory=True,
        drop_last=False,
        worker_init_fn=np.random.seed(7))

    eval_loader = None
    if args.eval_subdir:
        eval_loader = torch.utils.data.DataLoader(
            torchvision.datasets.ImageFolder(evaldir, eval_transformation),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.workers,  # Needs images twice as fast
            pin_memory=True,
            drop_last=False,
            worker_init_fn=np.random.seed(7))

    test_loader = torch.utils.data.DataLoader(
        torchvision.datasets.ImageFolder(testdir, eval_transformation),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,  # Needs images twice as fast
        pin_memory=True,
        drop_last=False,
        worker_init_fn=np.random.seed(7))

    return train_loader, train_loader_noshuff, eval_loader, \
           labeled_idxs, unlabeled_idxs, dataset, test_loader, batch_sampler
