# Based on the code from Antti Tarvainen and Harri Valpola
# Published in https://github.com/CuriousAI/mean-teacher
# Modified by O. Simeoni, M. Budnik, Y. Avrithis, G. Gravier, 2019


import re
import argparse
import logging

from . import architectures, datasets


LOG = logging.getLogger('main')

__all__ = ['parse_cmd_args', 'parse_dict_args']


def create_parser():
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
    parser.add_argument('--dataset', metavar='DATASET', default='cifar10',
                        choices=datasets.__all__,
                        help='dataset: ' +
                            ' | '.join(datasets.__all__) +
                            ' (default: imagenet)')
    parser.add_argument('--train-subdir', type=str, default='train',
                        help='the subdirectory inside the data directory that contains the training data')
    parser.add_argument('--eval-subdir', type=str, default='val',
                        help='the subdirectory inside the data directory that contains the evaluation data')
    parser.add_argument('--test-subdir', type=str, default='test',
                        help='the subdirectory inside the data directory that contains the test data')
    parser.add_argument('--exclude-unlabeled', default=True, type=str2bool, metavar='BOOL',
                        help='exclude unlabeled examples from the training set')
    parser.add_argument('--arch', '-a', metavar='ARCH', default='cifar_cnn',
                        choices=architectures.__all__,
                        help='model architecture: ' +
                            ' | '.join(architectures.__all__))
    parser.add_argument('-j', '--workers', default=0, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--epochs', default=200, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('-b', '--batch-size', default=32, type=int,
                        metavar='N', help='mini-batch size (default: 32)')
    parser.add_argument('--labeled-batch-size', '--labeled-batch-size', default=None, type=int,
                        metavar='N', help="labeled examples per minibatch (default: no constrain)")
    parser.add_argument('--lr', '--learning-rate', default=0.2, type=float,
                        metavar='LR', help='max learning rate')
    parser.add_argument('--initial-lr', default=0.0, type=float,
                        metavar='LR', help='initial learning rate when using linear rampup')
    parser.add_argument('--lr-rampup', default=0, type=int, metavar='EPOCHS',
                        help='length of learning rate rampup in the beginning')
    parser.add_argument('--lr-rampdown-epochs', default=210, type=int, metavar='EPOCHS',
                        help='length of learning rate cosine rampdown (>= length of training)')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--nesterov', default=True, type=str2bool,
                        help='use nesterov momentum', metavar='BOOL')
    parser.add_argument('--weight-decay', '--wd', default=2e-4, type=float,
                        metavar='W', help='weight decay (default: 2e-4)')
    parser.add_argument('--ema-decay', default=0.999, type=float, metavar='ALPHA', 
                        help='ema variable decay rate (default: 0.999)')
    parser.add_argument('-e', '--evaluate', type=str2bool,
                        help='evaluate model on evaluation set')
    parser.add_argument('--isL2', default=True, type=str2bool, metavar='BOOL',
                        help='is l2 normalized features')
    parser.add_argument('--loss_type', type=str, default="ce",
                        help='loss type')
    parser.add_argument('--sobel', type=str2bool, default=False,
                        help='Use sobel filters for preprocessing. Used with pretraining.')

    parser.add_argument('--exp_name', default='', type=str,
                        help='name of the current experiment')

    # Active learning parameters
    parser.add_argument('--split', default=None, type=int, metavar='FILE',
                        help='split to be used')
    parser.add_argument('--al-nb-cycles', default=10, type=int, metavar='N',
                        help='number of active learning steps (default: 10')
    parser.add_argument('--al-budget', default=100, type=int, metavar='N',
                        help='number of annotations per active learning step (default: 100)')
    parser.add_argument('--al-method', type=str, metavar='METH',
                        choices=['random', 'uncertainty_entropy', 'coreset', 'jlp'],
                        help='selection method used for active learning (default: Random)')
    parser.add_argument('--initial_set_size', default=1000, type=int, metavar='N',
                        help='cold start set size (default: 1000')

    parser.add_argument('--use_val_set', default=False, type=bool, metavar='N',
                        help='cold start set size (default: 1000')

    # Remove?
    parser.add_argument('--finetuning', type=bool, default=False,
                        help='Finetuning - use the trained model from the previous step')
    
    # CEAL
    parser.add_argument('--add-ceal', type=str2bool, default=False,
                        help='Use CEAL')
    parser.add_argument('--ceal-th', type=float, default=0.1,
                        help='CEAL threshold')
    parser.add_argument('--ceal-repeat', type=str2bool, default=False,
                        help='Use CEAL after every epoch.')

    # Pretraining
    parser.add_argument('--add-unsupervised-pretraining', type=str2bool, default=False,
                        help='Use pretrained model or not')
    parser.add_argument('--learn-unsupervised-pretraining', type=str2bool, default=False,
                        help='Create pretrained model or not')

    # Semi supervised label-propagation
    parser.add_argument('--add-lp', type=str2bool, default=False,
                        help='Label propagation enabled.')  
    parser.add_argument('--lp-mode', type=str, choices=['full', 'certain', 'prob', 'uniform'],
                        default='prob', help='Method for pseudo selection')
    parser.add_argument('--lp-percent', type=float, default=0.5,
                        help='Percentage of added images of entire dataset')
    parser.add_argument('--lp-step', type=int, default=1,
                        help='Every how many epoch should diffusion be applied ?')

    parser.add_argument('--is_pW', default=0, type=int,
                        help='weight for pseudo loss')
    parser.add_argument('--is_cW', default=True, type=bool,
                        help='Use class weight')

    parser.add_argument('--tau', type=float, default=0,
                        help='diffusion tau')
    parser.add_argument('--weighted_unlabeled_batch', type=str2bool, default=False,
                        help='Use weighting while creating batches ?', metavar='BOOL')
    
    # Training
    parser.add_argument('--resume', type=str2bool, default=True,
                        help='Resume from previous checkpoint')
    parser.add_argument('--checkpoint-epochs', default=25, type=int,
                        metavar='EPOCHS', help='checkpoint frequency in epochs, 0 to turn checkpointing off (default: 1)')
    parser.add_argument('--evaluation-epochs', default=10, type=int,
                        metavar='EPOCHS', help='evaluation frequency in epochs, 0 to turn evaluation off (default: 1)')
    parser.add_argument('--print-freq', '-p', default=10, type=int,
                        metavar='N', help='print frequency (default: 10)')
                  
    parser.add_argument('--seed', type=int, default=7,
                        help='Seed')
    return parser


def parse_commandline_args():
    return create_parser().parse_args()


def parse_dict_args(**kwargs):
    def to_cmdline_kwarg(key, value):
        if len(key) == 1:
            key = "-{}".format(key)
        else:
            key = "--{}".format(re.sub(r"_", "-", key))
        value = str(value)
        return key, value

    kwargs_pairs = (to_cmdline_kwarg(key, value)
                    for key, value in kwargs.items())
    cmdline_args = list(sum(kwargs_pairs, ()))

    LOG.info("Using these command line args: %s", " ".join(cmdline_args))

    return create_parser().parse_args(cmdline_args)


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def str2epochs(v):
    try:
        if len(v) == 0:
            epochs = []
        else:
            epochs = [int(string) for string in v.split(",")]
    except:
        raise argparse.ArgumentTypeError(
            'Expected comma-separated list of integers, got "{}"'.format(v))
    if not all(0 < epoch1 < epoch2 for epoch1, epoch2 in zip(epochs[:-1], epochs[1:])):
        raise argparse.ArgumentTypeError(
            'Expected the epochs to be listed in increasing order')
    return epochs
