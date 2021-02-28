# Authors: O. Simeoni, M. Budnik, Y. Avrithis, G. Gravier, 2019

import os
import time
import random
import pickle
import numpy as np
import torch

import pdb

from tqdm import tqdm

from time import gmtime, strftime
from lib import checkpoints, parse_args, training, label_propagation, pretraining
from lib import models, datasets, cli, data
import lib.selection_methods as selection_methods

torch.manual_seed(7)
torch.cuda.manual_seed(7)
np.random.seed(7)
random.seed(7)
torch.backends.cudnn.deterministic=True
torch.backends.cudnn.benchmark = False

args = None

def apply_random_seed(seed, log_file):
    if seed != 7:
        print_log('SEED: changing seed to %d'%seed, log_file)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

def print_log(msg, log):
    print(msg)
    with open(log, 'a') as log:
        log.write('%s\n' % msg)

def write_log_dist_class(dataset, labeled_idxs, log):
    targets = [targ for i, targ in enumerate(dataset.targets) if i in labeled_idxs]
    unique, counts = np.unique(targets, return_counts=True)
    dic_target = dict(zip(unique, counts))
    log.write('targets %s \n'% str(dic_target))

def create_folder(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)
    return folder

def load_unsupervised_pretrained_model(args):
    root_dir = './models/pretrained/'

    if 'mnist' in args.dataset:
        pretraining_path = 'pretrained_mnist_cifar_cnn_lr0.02_batch_128_final.pickle'
    elif 'svhn' in args.dataset:
        pretraining_path = 'pretrained_svhn_cifar_cnn_lr0.02_batch_128_final.pickle'
    elif 'cifar' in args.dataset:
        pretraining_path = 'pretrained_cifar_cifar_cnn_lr0.02_batch_128_final.pickle'
    else:
        raise ValueError('No pretraining for this dataset')

    pretraining_path = os.path.join(root_dir, pretraining_path)

    with open(pretraining_path, 'rb') as f:
        pretrain_w = pickle.load(f)

    return pretrain_w

def main():

    # -------------------------------------------------------------------------------
    # Directories
    dir_path = 'exps/'
    method_str = args.al_method
    if args.add_lp:
        method_str += '_withLP'
    if args.add_ceal:
        method_str += '_withCEAL'
    dir_path = create_folder(os.path.join(dir_path, args.dataset,
                                'budget%d'% args.al_budget,
                                args.arch, method_str, 
                                parse_args.from_args_to_string(args), 'split%d' % args.split))
    ckpt_dir    = create_folder(os.path.join(dir_path, 'ckpt'))
    label_dir   = create_folder(os.path.join(dir_path, 'used_labels'))
    log_dir     = create_folder(os.path.join(dir_path, 'logs'))
    weight_dir  = create_folder(os.path.join(log_dir, 'weights'))

    # -------------------------------------------------------------------------------
    # Check dataset exists
    data_root = 'data/'
    labels_file = os.path.join(data_root,
                               'labels/%s/%d_balanced_labels/0%d.txt' % (args.dataset,
                                                                         args.al_budget,
                                                                         args.split))
    args.labels = labels_file
    if not os.path.exists(labels_file):
        raise ValueError('Non existing label file %s' % labels_file)

    # Dataset
    dataset_config = datasets.__dict__[args.dataset]()
    num_classes = dataset_config.pop('num_classes')
    train_loader, train_loader_noshuff, eval_loader,\
        labeled_idxs, unlabeled_idxs, dataset, \
        test_loader, batch_sampler = training.create_data_loaders(args=args, \
                                                                  **dataset_config)
    if args.add_ceal:
        ceal_pseudo_labels_idxs = list()

    # Logging
    results_file = os.path.join(log_dir, 'results.csv')
    test_results_file = os.path.join(log_dir, 'test_results.csv')
    log_file = os.path.join(log_dir, '%s_%s.txt' % (strftime("%Y-%m-%d_%H-%M-%S", gmtime()),
                                                    args.exp_name))
    print_log('Log will be saved to %s' % log_file, log_file)
    print_log('Results will be saved to %s' % results_file, log_file)

    print_log('\nArgs: ', log_file)
    print_log(str(args) + '\n', log_file)
    with open(log_file, 'a') as log:
        write_log_dist_class(dataset, labeled_idxs, log)

    # Select random seed
    apply_random_seed(args.seed, log_file)

    # AL method
    al_method = parse_args.get_method(args)
    
    # ------------------------------------------------------------------------
    # Pretrained model in an unsupervised fashion
    if args.add_unsupervised_pretraining:
        # Pretrain a model from scratch
        if args.learn_unsupervised_pretraining:
            pretrain_w = pretraining.pretrain(args, dataset, num_classes, train_loader_noshuff)

        # Select one of the pretrained models used in the paper.
        else:
            pretrain_w = load_unsupervised_pretrained_model(args)
        
    
    # ------------------------------------------------------------------------
    # RESUMING
    r_cycle = -1

    if args.resume:
        r_cycle, r_epoch = checkpoints.get_checkpoint_cycle_epoch(ckpt_dir)
        r_labels = checkpoints.load_labels(r_cycle, label_dir)
        continue_training = True

        if r_cycle == -1 or r_epoch == -1:
            continue_training = False
        else:
            print_log('----------------------------------', log_file)
            print_log('RESUMING: from cycle %d and epoch %d'%(r_cycle, r_epoch), log_file)

            # In case training of a cycle was finished, check if labels were generated properly
            if r_epoch == (args.epochs - 1):
                print_log('Going to next cycle', log_file)
                r_labels_next_cycle = checkpoints.load_labels(r_cycle+1, label_dir)

                # If resuming labels exist go to next cycle
                if r_labels_next_cycle:
                    r_cycle += 1
                    r_labels = r_labels_next_cycle
                    continue_training = False
                    print_log('RESUMING: Going to next cycle %d'%(r_cycle), log_file)

            if r_labels:
                print_log('RESUMING: Updating the dataset and generating new train_loader', log_file)
                labeled_idxs = r_labels
                unlabeled_idxs = data.update_dataset_resuming(dataset, labeled_idxs)
                
                if args.add_lp:
                    print_log('Create dummy pseudo_label_idx', log_file)
                    dataset.pseudo_label_idx = dataset.unlabeled_idxs

                train_loader = training.get_train_loader(args, dataset)

                if args.add_lp and args.lp_mode != 'full' and \
                        (r_epoch > 0 and continue_training):
                    raise ValueError('Not implemented.')

            elif r_cycle > 0:
                raise ValueError('Should have new set of labels')

    # ------------------------------------------------------------------------
    # CYCLE
    for cycle in range(max(r_cycle, 0), args.al_nb_cycles):
        print_log('AL cycle %d' % cycle, log_file)

        model = models.create_model(args, num_classes)
        optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay,
                                nesterov=args.nesterov)

        # Pretraining
        if args.add_unsupervised_pretraining:
            print_log('Use model pretrained in an unsupervised fashion', log_file)
            model_dict = model.state_dict()
            model_dict.update(pretrain_w)
            model.load_state_dict(model_dict)

        # Resuming from previous checkpoint if exists
        start_epoch = 0
        if args.resume:
            if continue_training:
                start_epoch = r_epoch

            if r_epoch > 0 and r_cycle == cycle and continue_training:
                r_cycle, r_epoch, model, optimizer = checkpoints.load_checkpoint(model, optimizer,
                                                                                ckpt_dir, cycle)
                print_log('RESUMING: weights model from cycle %d and epoch %d'% (r_cycle,
                                                                                r_epoch),
                                                                                log_file)
                epoch = r_epoch
                start_epoch = r_epoch + 1

        if args.finetuning and cycle != 0:
            model = checkpoints.load_checkpoint_finetuning(model, ckpt_dir, cycle-1)
            print('Using pretrained model')

        # Apply first label propagation
        if args.add_lp:
            print_log('Label propagation: Starting diffusion', log_file)
           
            # Extract features
            feats, labels, preds = models.extract_features(train_loader_noshuff, model)
            
            # Apply label propagation
            lp = label_propagation.LP()
            sel_acc, sel_n = lp.update_lp(feats, preds, dataset, thresh=args.tau,
                                                  args=args, w_mode=args.is_cW)
            
            if args.weighted_unlabeled_batch:
                batch_sampler.update_weights(np.array(dataset.p_weights)[unlabeled_idxs])
            elif args.lp_mode != 'full':
                print_log('Updating %d pseudo labels in the batch' % len(dataset.pseudo_label_idx), log_file)
                batch_sampler.update_pseudo_indices(dataset.pseudo_label_idx, same_length=False)

        # ------------------------------------------------------------------------
        # CYCLE TRAINING
        for epoch in tqdm(range(start_epoch, args.epochs)):

            # Train the model
            models.train(train_loader, model, optimizer, epoch, args)

            # Apply label propagation if needed
            if args.add_lp and epoch > args.start_epoch and epoch % args.lp_step == 0:
                print_log('Label propagation applied epoch {}'.format(epoch), log_file)
                
                # Extract features
                feats, labels, preds = models.extract_features(train_loader_noshuff, model)

                # Apply label propagation
                sel_acc, sel_n = lp.update_lp(feats, preds, dataset, thresh=args.tau,
                                                        args=args, w_mode=args.is_cW)

                if args.weighted_unlabeled_batch:
                    batch_sampler.update_weights(np.array(dataset.p_weights)[unlabeled_idxs])
                elif args.lp_mode != 'full':
                    print_log('Updating %d pseudo labels in the batch'%len(dataset.pseudo_label_idx), log_file)
                    batch_sampler.update_pseudo_indices(dataset.pseudo_label_idx)

            # Save models
            if epoch % args.checkpoint_epochs == 0 or epoch == args.epochs-1:
                checkpoints.save_checkpoint(model, optimizer, ckpt_dir, cycle, epoch)

        #---------------------------
        #------- Evaluation -------
        #---------------------------
        with open(log_file, 'a') as log:
            write_log_dist_class(dataset, labeled_idxs, log)

            # TODO Change validation
            if args.use_val_set:
                with open(results_file, 'a') as log_results:
                    models.validate(eval_loader, model, epoch, log, cycle, log_results, 'Val set')
            
            # Model Evaluation
            with open(test_results_file, 'a') as log_test_results:
                models.validate(test_loader, model, epoch, log, cycle, log_test_results, 'Test set')

        # Removing labels used for ceal before the selection
        if args.add_ceal:
            data.remove_semi_labels(dataset, ceal_pseudo_labels_idxs)

        # Perform the selection using the selected AL method
        selection_method = al_method(model)
        selected = selection_method.select(train_loader_noshuff, dataset, 
                                                args.al_budget, args=args)

        # Check
        if set(selected).intersection(set(labeled_idxs)) or len(set(selected)) != args.al_budget:
            raise ValueError("Selection is not correct")

        # Update the dataset with the newly selected images
        labeled_idxs, unlabeled_idxs = data.update_dataset(dataset, selected, labeled_idxs)

        # Save the selected images - used for next cycle 
        print_log('%d selected images saved'%(len(selected)), log_file)
        checkpoints.write_labels(labeled_idxs, cycle+1, label_dir)
        continue_training = False

        # Apply CEAL - used for next cycle
        if args.add_ceal:
            print_log('Applying CEAL', log_file)
            ceal_method = selection_methods.CEAL(model)
            ceal_pseudo_labels_idxs, ceal_pseudo_labels = ceal_method.select(train_loader_noshuff, dataset,
                                                               args.ceal_th, 0.00033, cycle)
            # Update the dataset
            data.update_dataset_semi(dataset, ceal_pseudo_labels_idxs, ceal_pseudo_labels)


if __name__ == '__main__':
    args = cli.parse_commandline_args()
    args.test_batch_size = args.batch_size

    main()
