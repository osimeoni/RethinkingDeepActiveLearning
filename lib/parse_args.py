# Authors: O. Simeoni, M. Budnik, Y. Avrithis, G. Gravier, 2019

import lib.selection_methods as selection_methods

def args_def(args):

    # When not using validation 
    if not args.use_val_set: 
        args.train_subdir = 'train+val'
        args.eval_subdir = None

    # Types of label propagation 
    if args.add_lp and args.add_ceal:
        raise ValueError('Label propagation and CEAL can t be used together.')

    if args.add_ceal and args.resume:
        args.resume = False
        print('Resume not implemented for CEAL')        

    # Set default parameters for label propagation 
    if args.add_lp:
        args.is_pW = True
        args.exclude_unlabeled = False
        if args.batch_size == 64:
            args.labeled_batch_size = 10
        elif args.batch_size == 128:
            args.labeled_batch_size = 50

    # -------------
    # Checks
    if args.add_lp:
        if args.labeled_batch_size is None or args.labeled_batch_size >= args.batch_size:
            raise ValueError('Labeled batch size should be defined and smaller than batch_size')
        if args.is_pW != 1:
            print ('CAREFULL : Label propagation better with is_pW')
        if args.exclude_unlabeled:
            raise ValueError('Unlabeled are needed')
        if args.lp_percent > 1 or args.lp_percent <= 0:
            raise ValueError('args.lp_percent should be between 0 and 1')
        if args.weighted_unlabeled_batch and args.pseudo_method != 'full':
            raise ValueError('Batch weighting can be done only with full')

    if args.lr_rampdown_epochs < args.epochs:
        raise ValueError('lr_rampdown_epochs should be larger than epochs')

    return args

def from_args_to_string(args):

    args = args_def(args)

    str_ = 'bs%d_nepochs%d' % (args.batch_size, args.epochs)

    if args.exclude_unlabeled != True:
        str_ += '_keepUnlabeled'
    if args.isL2 != True:
        str_ += '_L2%d' % (args.isL2)
    if args.is_pW != 0:
        str_ += '_ispW%d' % (args.is_pW)

    # Network
    if args.loss_type != 'ce':
        str_ += '_loss_%s' % (args.loss_type)
    if args.is_cW != 3:
        str_ += '_iscW%d' % (args.is_cW)
    if args.lr != 0.2:
        str_ += '_lr%.3f' % (args.lr)
    if args.lr_rampdown_epochs != 210:
        str_ += '_lrRampdown%d' % (args.lr_rampdown_epochs)
    if args.lr_rampup != 0:
        str_ += '_lrRampup%d' % (args.lr_rampup)
    if args.momentum != 0.9:
        str_ += '_momentum%.2f' % (args.momentum)
    if args.nesterov != True:
        str_ += '_nesterov%d' % (args.nesterov)
    if args.tau != 0:
        str_ += '_tau%.1f' % (args.tau)
    if args.weight_decay != 2e-4:
        str_ += '_wDecay%.5f' % (args.weight_decay)
  
    if args.add_unsupervised_pretraining:
        str_ += '_with_U_pretraining'

    if args.use_val_set:
        str_ += '_useBestModelNew'
        
    # CEAL
    if args.ceal_repeat:
        str_ += '_CEAL_REPEAT'
    if args.ceal_th != 0.1:
        str_ += '_ceal_%.2f' % (args.ceal_th)

    # Label propagation
    if args.add_lp:
        str_ += '_lp_%s'% (args.lp_mode)

        if args.lp_mode != 'full':
            str_ += '_%.2f' % (args.lp_percent)
        if args.lp_step != 1:
            str_ += '_step%d' % (args.lp_step)
        if args.labeled_batch_size != None:
            str_ += '_Lbs%d' % (args.labeled_batch_size)
        if args.weighted_unlabeled_batch:
            str_ += '_weighted_UB'

    if args.ema_decay != 0.999:
        str_ += '_ema_decay%.3f' % (args.ema_decay)

    if args.exp_name != '':
        str_ += '_%s' % (args.exp_name)

    return str_

def get_method(args):
    if args.al_method == 'random':
        method = selection_methods.RandomSampling
    elif args.al_method == 'uncertainty_entropy':
        method = selection_methods.UncertaintyEntropySampling
    elif args.al_method == 'jlp':
        method = selection_methods.JLPSelection
    elif args.al_method == 'coreset':
        method = selection_methods.CoreSetSampling
    else:
        raise ValueError('Method %s non existing'%args.al_method)
    return method
