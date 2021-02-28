import os
import pdb
import pickle
import argparse
import numpy as np
import scipy.io as sio

from PIL import Image
from glob import glob
from tqdm import tqdm
from shutil import copyfile
from torchvision import datasets


np.random.seed(7)

parser = argparse.ArgumentParser(description='Extracting a given dataset')
parser.add_argument('--dataset', metavar='DATASET', default='cifar10',                
                        help='Dataset to generate splits from(default: cifar10)')
parser.add_argument('--val_size', default=50, type=int, metavar='N',
                        help='Validation size (images per class) (default: 50')


def cifar_extraction(path_in, path_out, lbs, batch, k=1, dataset='cifar10'):
    if dataset == 'cifar10':
        if batch == 'train+val':
            fpath = os.path.join(path_in, 'data_batch_'+str(k))
        else:
            fpath = os.path.join(path_in, 'test_batch')
    else: 
        if batch == 'train+val':
            fpath = os.path.join(path_in, 'train')
        else:
            fpath = os.path.join(path_in, 'test')
    fo = open(fpath, 'rb')
    d = pickle.load(fo, encoding='bytes')
    d_decoded = {}
    for j, v in d.items():
        d_decoded[j.decode('utf8')] = v
    d = d_decoded
    fo.close()
    itr = 0
    for i, filename in enumerate(tqdm(d['filenames'])):
        if dataset == 'cifar10':
            lb = lbs['label_names'][d['labels'][i]]
        else: 
            lb = lbs['fine_label_names'][d['fine_labels'][i]]
            if '_' in lb:
                lb = lb.split('_')[0]
        folder = os.path.join(path_out, batch, lb)
        os.makedirs(folder, exist_ok=True)
        q_ = d['data'][i]
        itr = itr + 1
        f_name = filename.decode().split('_')[-1].split('.')[0]+'_'+lb+'.png'
        ad_it = 1
        while os.path.isfile(os.path.join(folder, f_name)) == True:
            f_name = filename.decode().split('_')[-1].split('.')[0]+'-'+str(ad_it)+'_'+lb+'.png'     
            ad_it = ad_it + 1
        outfile = os.path.join(folder, f_name)
        A = q_.reshape((32, 32, 3), order='F').swapaxes(0,1)
        im = Image.fromarray(A)
        im.save(outfile)
        

def main():
    args = parser.parse_args()

    if args.dataset in ['svhn', 'mnist']:
        path_in  = './data/'
        path_out = './data/images/%s/by-image/' % (args.dataset)
    elif 'cifar' in args.dataset:
        if args.dataset == 'cifar100':
            path_in = './data/cifar-100-python'
        else: 
            path_in = './data/cifar-10-batches-py'
        path_out = './data/images/cifar/%s/by-image/' % (args.dataset)
    else:
        raise ValueError('Dataset %s is not supported.' % args.dataset)

    path_all = path_out+'/train+val/'
    val_size = args.val_size

    if not os.path.exists(path_out):
        os.makedirs(path_out)
    for batch in ('train','test'): 
        if args.dataset in ['svhn', 'mnist']:
            if args.dataset == 'svhn':
                if batch == 'train':
                    d_set = datasets.SVHN(path_in, split=batch, download=True)
                    batch = 'train+val'
                else: 
                    d_set = datasets.SVHN(path_in, split=batch, download=True)
            else:
                if batch == 'train':
                    d_set = datasets.MNIST(path_in, train=True, download=True)
                    batch = 'train+val'
                else: 
                    d_set = datasets.MNIST(path_in, train=False, download=True)
                    
            for i in tqdm(range(len(d_set))):
                lbls = str(d_set[i][1])
                folder = os.path.join(path_out, batch, lbls)
                os.makedirs(folder, exist_ok=True)
                f_name = 'image_%s.png' % i
                outfile = os.path.join(folder, f_name)
                im = d_set[i][0]
                im.save(outfile)
                
        elif args.dataset == 'cifar10':
            lbs = pickle.load(open(os.path.join(path_in, 'batches.meta'), 'rb'), encoding="ASCII")
            if batch == 'train':
                batch = 'train+val'
                for k in range(1,6):
                    cifar_extraction(path_in, path_out, lbs, batch, k)
            else: 
                cifar_extraction(path_in, path_out, lbs, batch)
        else:
            labels = pickle.load(open(os.path.join(path_in, 'meta'), 'rb'), encoding="ASCII")
            if batch == 'train':
                batch = 'train+val'
            cifar_extraction(path_in, path_out, labels, batch, dataset = 'cifar100')
            

    # Train and validation split is created here based on the images in train+val
    
    all_classes = glob(path_all+'*')

    for cl in all_classes:
        all_images = glob(cl+'/*')
        val_idxs = np.random.choice(len(all_images), val_size, replace=False)
        directory = path_out+'/val/'+cl.split('/')[-1]
        if not os.path.exists(directory):
            os.makedirs(directory)
        itr = 0
        for x in val_idxs:
            copyfile(all_images[x], directory+'/'+os.path.basename(all_images[x]))
            itr = itr + 1

        mask = np.ones(len(all_images), bool)
        mask[val_idxs] = False
        directory = path_out+'/train/'+cl.split('/')[-1]
        if not os.path.exists(directory):
            os.makedirs(directory)
        for x in np.array(all_images)[mask]:
            copyfile(x, directory+'/'+os.path.basename(x))
            
            
if __name__ == '__main__':
    main()
