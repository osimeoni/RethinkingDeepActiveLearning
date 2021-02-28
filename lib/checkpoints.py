# Authors: O. Simeoni, M. Budnik, Y. Avrithis, G. Gravier, 2019

import os
import csv
import torch
import pickle
import pdb

def print_log(msg, log):
    print(msg)
    with open(log, 'a') as log:
        log.write('%s\n' % msg)

def load_best_model(model, cycle, dirpath):
    filename = 'checkpoint_cycle%d_best.ckpt' % cycle
    best_ckpt_path = os.path.join(dirpath, filename)

    best_ckpt = torch.load(best_ckpt_path)
    model.load_state_dict(best_ckpt['model'])

    return model, best_ckpt['cycle'], best_ckpt['epoch']


def save_best_model(model, optimizer, dirpath, cycle, epoch, validation_res):
    state = {'epoch': epoch, 'cycle': cycle, 'model': model.state_dict(),
            'optimizer': optimizer.state_dict()}

    best_model_file = os.path.join(dirpath, 'bestmodel_cycle%d.csv' % cycle)
    best_info = [cycle, epoch, validation_res.cpu().numpy()]

    filename = 'checkpoint_cycle%d_best.ckpt' % cycle
    best_ckpt_path = os.path.join(dirpath, filename)

    if os.path.exists(best_model_file):
        with open(best_model_file, 'r') as f:
            reader = csv.reader(f)
            lines = list(reader)
            best_old = float(lines[-1][2])

    if not os.path.exists(best_model_file) or validation_res > best_old:
        print('New best %.3f' % validation_res)
        with open(best_model_file, 'a') as f:
            writer = csv.writer(f)
            writer.writerow(best_info)

        torch.save(state, best_ckpt_path)
        print("--- best checkpoint saved to %s ---" % (best_ckpt_path))
        print("--- best info saved to %s ---" % (best_model_file))

def ckpt_to_remove(dirpath, cycle, epoch):
    for path in os.listdir(dirpath):
        splt = [sub for s in path.split('_') for sub in s.split('.')]
        if splt[1].isdigit() and float(splt[1]) == cycle:
            if splt[2].isdigit() and float(splt[2]) < epoch:
                os.remove(os.path.join(dirpath, path))


def save_checkpoint(model, optimizer, dirpath, cycle, epoch):
    state = {'epoch': epoch, 'cycle': cycle, 'model': model.state_dict(),
            'optimizer': optimizer.state_dict()}

    # Checkpoint per cycle
    filename = 'checkpoint.{}_{}.ckpt'.format(cycle, epoch)
    ckpt_path = os.path.join(dirpath, filename)
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)
    torch.save(state, ckpt_path)

    filename = 'checkpoint_last.ckpt'
    last_ckpt_path = os.path.join(dirpath, filename)
    torch.save(state, last_ckpt_path)

    # Remove previously saved ckpt
    ckpt_to_remove(dirpath, cycle, epoch)


def get_checkpoint_cycle_epoch(dirpath):
    filename = 'checkpoint_last.ckpt'
    last_ckpt_path = os.path.join(dirpath, filename)

    if os.path.exists(last_ckpt_path):
        print('Loading state from checkpoint %s' % last_ckpt_path)
        checkpoint = torch.load(last_ckpt_path)
        return checkpoint['cycle'], checkpoint['epoch']

    return -1, -1

def load_checkpoint(model, optimizer, dirpath, current_cycle):
    filename = 'checkpoint_last.ckpt'
    last_ckpt_path = os.path.join(dirpath, filename)

    if os.path.exists(last_ckpt_path):
        checkpoint = torch.load(last_ckpt_path)

        if current_cycle == checkpoint['cycle']:
            print ('Resuming from checkpoint %s' % last_ckpt_path)
            model.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])

        return checkpoint['cycle'], checkpoint['epoch'], model, optimizer
    else:
        return -1, -1, model, optimizer
        print ('Did not find any checkpoint')


def load_checkpoint_finetuning(model, dirpath, previous_cycle):
    filename = 'checkpoint.%d_199.ckpt'%previous_cycle
    last_ckpt_path = os.path.join(dirpath, filename)
    print(last_ckpt_path)
    if os.path.exists(last_ckpt_path):
        print ('Loading from checkpoint %s' % last_ckpt_path)
        checkpoint = torch.load(last_ckpt_path)
        if previous_cycle != checkpoint['cycle']:
            raise ValueError('Resuming from checkpoint %s failed' % last_ckpt_path)
        model.load_state_dict(checkpoint['model'])
    else:
        print ('Did not find any checkpoint')

    return model


def load_cycle_model(model, dirpath, current_cycle):
    filename = 'checkpoint_cycle%d_best.ckpt'%current_cycle
    last_ckpt_path = os.path.join(dirpath, filename)

    if os.path.exists(last_ckpt_path):
        print ('Loading from checkpoint %s' % last_ckpt_path)
        checkpoint = torch.load(last_ckpt_path)
        if current_cycle != checkpoint['cycle']:
            raise ValueError('Resuming from checkpoint %s failed' % last_ckpt_path)
        model.load_state_dict(checkpoint['model'])
    else:
        print ('Did not find any checkpoint')

    return model

def load_cycle_epoch_model(model, dirpath, current_cycle, nb_epochs):
    filename = 'checkpoint.%d_%d.ckpt'%(current_cycle, nb_epochs)
    last_ckpt_path = os.path.join(dirpath, filename)

    if os.path.exists(last_ckpt_path):
        print ('Loading from checkpoint %s' % last_ckpt_path)
        checkpoint = torch.load(last_ckpt_path)
        if current_cycle != checkpoint['cycle']:
            raise ValueError('Resuming from checkpoint %s failed' % last_ckpt_path)
        model.load_state_dict(checkpoint['model'])
    else:
        print ('Did not find any checkpoint')

    return model

def load_last_model(model, dirpath, current_cycle):
    filename = 'checkpoint_last.ckpt'
    last_ckpt_path = os.path.join(dirpath, filename)

    if os.path.exists(last_ckpt_path):
        print ('Loading from checkpoint %s' % last_ckpt_path)
        checkpoint = torch.load(last_ckpt_path)
        pdb.set_trace()
        if current_cycle == checkpoint['cycle']:
            print ('Resuming from checkpoint %s' % last_ckpt_path)
            model.load_state_dict(checkpoint['model'])
    else:
        print ('Did not find any checkpoint')

    return model

def write_labels(labels, cycle, dirpath):
    filename = 'labels_cycle%d.txt' % cycle
    label_path = os.path.join(dirpath, filename)

    if not os.path.exists(dirpath):
        os.makedirs(dirpath)

    with open(label_path, 'wb') as fp:
        pickle.dump(labels, fp)


def load_labels(cycle, dirpath):
    filename = 'labels_cycle%d.txt' % cycle
    label_path = os.path.join(dirpath, filename)

    if os.path.exists(label_path):
        with open(label_path, 'rb') as fp:
            print('Loading labels from %s'%label_path)
            labels = pickle.load(fp)
        return labels

    return None
    
def load_cycle_epoch_model(model, dirpath, current_cycle, nb_epochs):
    filename = 'checkpoint.%d_%d.ckpt'%(current_cycle, nb_epochs)
    last_ckpt_path = os.path.join(dirpath, filename)

    if os.path.exists(last_ckpt_path):
        print ('Loading from checkpoint %s' % last_ckpt_path)
        checkpoint = torch.load(last_ckpt_path)
        if current_cycle != checkpoint['cycle']:
            raise ValueError('Resuming from checkpoint %s failed' % last_ckpt_path)
        model.load_state_dict(checkpoint['model'])
    else:
        print ('Did not find any checkpoint')

    return model