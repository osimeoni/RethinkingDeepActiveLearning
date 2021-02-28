# Authors: O. Simeoni, M. Budnik, Y. Avrithis, G. Gravier, 2019

import time
import numpy as np
import torch
torch.manual_seed(7)
torch.cuda.manual_seed(7)
np.random.seed(7)
torch.backends.cudnn.deterministic=True
torch.backends.cudnn.benchmark = False
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from . import architectures, ramps
from .utils import AverageMeterSet
from .data import NO_LABEL


def create_model(args, num_classes, ema=False):
    model_factory = architectures.__dict__[args.arch]
    model_params = dict(num_classes=num_classes, isL2 = args.isL2, sobel = args.sobel)
    model = model_factory(**model_params)
    model = nn.DataParallel(model).cuda()

    if ema:
        for param in model.parameters():
            param.detach_()

    return model

def train(train_loader, model, optimizer, epoch, args):
    loss_type = args.loss_type
    if loss_type == 'ce':
        class_criterion = nn.CrossEntropyLoss(size_average=False,
                                              ignore_index=NO_LABEL, reduce=False).cuda()
    elif loss_type == 'kl':
        class_criterion = nn.KLDivLoss(size_average=False, reduce = False).cuda()
    elif loss_type == 'mse':
        class_criterion = nn.CrossEntropyLoss(size_average=False,
                                              ignore_index=NO_LABEL, reduce=False).cuda()

    meters = AverageMeterSet()

    # switch to train mode
    model.train()

    end = time.time()
    for i, ((input, ema_input), target, weight, one_hot, c_weight, index) in enumerate(train_loader):
        meters.update('data_time', time.time() - end)
        adjust_learning_rate(optimizer, epoch, i, len(train_loader), args)

        meters.update('lr', optimizer.param_groups[0]['lr'])

        input_var = torch.autograd.Variable(input.to('cuda'))
        target_var = torch.autograd.Variable(target.cuda())
        weight_var = torch.autograd.Variable(weight.cuda())
        c_weight_var = torch.autograd.Variable(c_weight.cuda(async=True))
        one_hot_var = torch.autograd.Variable(one_hot.cuda()) #async=True))


        minibatch_size = len(target_var)
        labeled_minibatch_size = target_var.data.ne(NO_LABEL).sum()
        assert labeled_minibatch_size > 0

        meters.update('labeled_minibatch_size', labeled_minibatch_size)
        model_out, feats = model(input_var)
        class_logit = model_out

        if loss_type == 'kl':
            valid_idx = target.ne(-1)
            class_loss = class_criterion(F.log_softmax(class_logit[valid_idx,:]), one_hot_var[valid_idx,:])
            if args.is_pW:
                class_loss = class_loss.sum(1) * weight_var[valid_idx].float()

            class_loss = class_loss.sum() / minibatch_size
        elif loss_type == 'ce':
            class_loss = class_criterion(class_logit, target_var)
            if args.is_pW:
                class_loss = class_loss * weight_var.float()
            if args.is_cW == 4:
                class_loss = class_loss * c_weight_var
            class_loss = class_loss.sum() / minibatch_size

        elif loss_type == 'mse':
            valid_idx = range(50,100)
            class_loss = class_criterion(class_logit[valid_idx,:], target_var[valid_idx])
            class_loss = class_loss * weight_var[valid_idx].float()
            class_loss = class_loss.sum() / len(valid_idx)

            mse_idx = range(0,50)
            num_classes = class_logit.size()[1]
            mse_loss = F.mse_loss(F.softmax(class_logit[mse_idx,:]), one_hot_var[mse_idx,:], size_average=False, reduce = False)
            mse_loss = mse_loss.sum(1) * weight_var[mse_idx].float()
            mse_loss = mse_loss.sum() / num_classes
            mse_loss = mse_loss / len(mse_idx)

            class_loss = class_loss + mse_loss

        meters.update('class_loss', class_loss.item())

        loss = class_loss

        assert not (np.isnan(loss.item()) or loss.item() > 1e5), 'Loss explosion: {}'.format(loss.item())
        meters.update('loss', loss.item())

        prec1, prec5 = accuracy(class_logit.data, target_var.data, topk=(1, 5))
        meters.update('top1', prec1[0], labeled_minibatch_size)
        meters.update('error1', 100. - prec1[0], labeled_minibatch_size)
        meters.update('top5', prec5[0], labeled_minibatch_size)
        meters.update('error5', 100. - prec5[0], labeled_minibatch_size)

        # compute gradient and do SGD step
        optimizer.zero_grad()

        loss.backward()
        optimizer.step()

        # measure elapsed time
        meters.update('batch_time', time.time() - end)
        end = time.time()

    return meters


def evaluate(eval_loader, model):

    meters = AverageMeterSet()

    # switch to evaluate mode
    model.eval()

    for i, (input, target) in enumerate(eval_loader):
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target.cuda())

        labeled_minibatch_size = target_var.data.ne(NO_LABEL).sum()
        assert labeled_minibatch_size > 0

        # compute output
        output1, output2 = model(input_var)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output1.data, target_var.data, topk=(1, 5))
        meters.update('top1', prec1[0], labeled_minibatch_size)
        meters.update('top5', prec5[0], labeled_minibatch_size)

    return meters['top1'].avg.cpu().numpy(), meters['top5'].avg.cpu().numpy()

def validate(eval_loader, model, epoch, log, step, results, set_string):

    class_criterion = nn.CrossEntropyLoss(size_average=False, ignore_index=NO_LABEL).cuda()
    meters = AverageMeterSet()

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():

        end = time.time()
        for i, (input, target) in enumerate(eval_loader):
            meters.update('data_time', time.time() - end)

            input_var = torch.autograd.Variable(input)
            target_var = torch.autograd.Variable(target.cuda())

            minibatch_size = len(target_var)
            labeled_minibatch_size = target_var.data.ne(NO_LABEL).sum()
            assert labeled_minibatch_size > 0
            meters.update('labeled_minibatch_size', labeled_minibatch_size)

            # compute output
            output1, output2 = model(input_var)
            class_loss = class_criterion(output1, target_var) / minibatch_size

            # measure accuracy and record loss
            prec1, prec5 = accuracy(output1.data, target_var.data, topk=(1, 5))
            meters.update('class_loss', class_loss.item(), labeled_minibatch_size)
            meters.update('top1', prec1[0], labeled_minibatch_size)
            meters.update('error1', 100.0 - prec1[0], labeled_minibatch_size)
            meters.update('top5', prec5[0], labeled_minibatch_size)
            meters.update('error5', 100.0 - prec5[0], labeled_minibatch_size)

            # measure elapsed time
            meters.update('batch_time', time.time() - end)
            end = time.time()

    result = '{} step {}, epoch {} Prec@1 {top1.avg:.3f}\tPrec@5 {top5.avg:.3f}'.format(
                                                                      set_string,
                                                                      step,
                                                                      epoch,
                                                                      top1=meters['top1'],
                                                                      top5=meters['top5'])
    print(result)
    log.write(str(step)+' '+str(epoch)+' '+result+'\n')
    results.write(str(step)+';'+str(epoch)+';'+\
                  str(meters['top1'].avg.cpu().numpy())+\
                  ';'+str(meters['top5'].avg.cpu().numpy())+';\n')

    return meters['top1'].avg, meters['top5'].avg


def adjust_learning_rate(optimizer, epoch, step_in_epoch, total_steps_in_epoch, args):
    lr = args.lr
    epoch = epoch + step_in_epoch / total_steps_in_epoch

    # LR warm-up to handle large minibatch sizes from https://arxiv.org/abs/1706.02677
    lr = ramps.linear_rampup(epoch, args.lr_rampup) * (args.lr - args.initial_lr) + args.initial_lr

    # Cosine LR rampdown from https://arxiv.org/abs/1608.03983 (but one cycle only)
    if args.lr_rampdown_epochs:
        assert args.lr_rampdown_epochs >= args.epochs
        lr *= ramps.cosine_rampdown(epoch, args.lr_rampdown_epochs)

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    labeled_minibatch_size = max(target.ne(NO_LABEL).sum(), 1e-8)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()

    correct = pred.eq(target.view(1, -1).expand_as(pred))
    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / float(labeled_minibatch_size)))
    return res


def extract_features(train_loader, model):
    model.eval()
    predictions_all, labels_all, embeddings_all = [], [], []

    for batch_idx, batch in enumerate(train_loader):

        # Data.
        X = batch[0][0]
        y = batch[1]

        # Data.
        X = torch.autograd.Variable(X.cuda())
        y = torch.autograd.Variable(y.cuda())

        # Prediction.
        preds, feats = model(X)

        predictions_all.append(preds.data.cpu())
        labels_all.append(y.data.cpu())
        embeddings_all.append(feats.data.cpu())

    predictions_all = torch.cat(predictions_all)
    embeddings_all = np.asarray(torch.cat(embeddings_all).numpy())
    labels_all = torch.cat(labels_all).numpy()

    return (embeddings_all, labels_all, predictions_all)
