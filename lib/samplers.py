# Samplers implementated by: A. Iscen, G. Tolias, Y. Avrithis, O. Chum. 2018.
# Published in https://github.com/ahmetius/LP-DeepSSL

import sys
import torch
import pickle
import random
import os.path
import itertools
import numpy as np

from PIL import Image
import torch.utils.data
from torch.utils.data.sampler import Sampler

class WeightedTwoStreamBatchSampler(Sampler):
    """Iterate two sets of indices

    An 'epoch' is one iteration through the primary indices.
    During the epoch, the secondary indices are iterated through
    as many times as needed.
    """
    def __init__(self, primary_indices, secondary_indices, batch_size, secondary_batch_size,
                 weights, replacement=True):

        self.primary_indices = primary_indices
        self.secondary_indices = secondary_indices
        self.secondary_batch_size = secondary_batch_size
        self.primary_batch_size = batch_size - secondary_batch_size

        assert len(self.primary_indices) >= self.primary_batch_size > 0
        assert len(self.secondary_indices) >= self.secondary_batch_size > 0

        # Weighted
        if not isinstance(replacement, bool):
            raise ValueError("replacement should be a boolean value, but got "
                             "replacement={}".format(replacement))

        if not len(weights) == len(self.primary_indices):
            raise ValueError("weights vector should be the same lenght as indices")

        self.weights = np.array(weights)
        self.replacement = replacement

    def __iter__(self):
        print('iter')
        print(self.weights[1:10])
        primary_iter = iterate_once_weighted(self.primary_indices, self.weights, self.replacement)
        secondary_iter = iterate_eternally(self.secondary_indices)
        return (
            primary_batch + secondary_batch
            for (primary_batch, secondary_batch)
            in  zip(grouper(primary_iter, self.primary_batch_size),
                    grouper(secondary_iter, self.secondary_batch_size))
        )

    def __len__(self):
        return len(self.primary_indices) // self.primary_batch_size

    def update_weights(self, weights):
        print('Update weights')
        if not len(weights) == len(self.weights):
            raise ValueError("weights vector should be the same lenght \
                              as indices (%d vs %d)"%(len(weights), len(self.weights)))
        self.weights = weights

def iterate_once(iterable):
    return np.random.permutation(iterable)

def iterate_once_weighted(indices, distribution, replacement):
    print('Iterate once weighthed')
    distribution = distribution / np.sum(distribution)
    sel = np.random.choice(range(len(indices)), len(indices),
            p=distribution)
    return np.array(indices)[sel]

def iterate_eternally_weighted(indices, distribution, replacement):
    print('Iterate eternally weighthed')
    def infinite_shuffles(distribution):
        while True:
            distribution = distribution / np.sum(distribution)
            sel = np.random.choice(range(len(indices)), len(indices),
                    p=distribution) #, replacement=replacement)
            yield np.array(indices)[sel]
    return itertools.chain.from_iterable(infinite_shuffles(distribution))


def iterate_eternally(indices):
    def infinite_shuffles():
        while True:
            yield np.random.permutation(indices)
    return itertools.chain.from_iterable(infinite_shuffles())


def grouper(iterable, n):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3) --> ABC DEF"
    args = [iter(iterable)] * n
    return zip(*args)

IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif']

def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)


def has_file_allowed_extension(filename, extensions):
    """Checks if a file is an allowed extension.

    Args:
        filename (string): path to a file
        extensions (iterable of strings): extensions to consider (lowercase)

    Returns:
        bool: True if the filename ends with one of given extensions
    """
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in extensions)


def is_image_file(filename):
    """Checks if a file is an allowed image extension.

    Args:
        filename (string): path to a file

    Returns:
        bool: True if the filename ends with a known image extension
    """
    return has_file_allowed_extension(filename, IMG_EXTENSIONS)


def make_dataset(dir, class_to_idx, extensions):
    images = []
    dir = os.path.expanduser(dir)
    for target in sorted(class_to_idx.keys()):
        d = os.path.join(dir, target)
        if not os.path.isdir(d):
            continue

        for root, _, fnames in sorted(os.walk(d)):
            for fname in sorted(fnames):
                if has_file_allowed_extension(fname, extensions):
                    path = os.path.join(root, fname)
                    item = (path, class_to_idx[target])
                    images.append(item)

    return images

class DatasetFolder(torch.utils.data.Dataset):
    """A generic data loader where the samples are arranged in this way: ::

        root/class_x/xxx.ext
        root/class_x/xxy.ext
        root/class_x/xxz.ext

        root/class_y/123.ext
        root/class_y/nsdf3.ext
        root/class_y/asd932_.ext

    Args:
        root (string): Root directory path.
        loader (callable): A function to load a sample given its path.
        extensions (list[string]): A list of allowed extensions.
        transform (callable, optional): A function/transform that takes in
            a sample and returns a transformed version.
            E.g, ``transforms.RandomCrop`` for images.
        target_transform (callable, optional): A function/transform that takes
            in the target and transforms it.

     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        samples (list): List of (sample path, class_index) tuples
        targets (list): The class_index value for each image in the dataset
    """

    def __init__(self, root, transform=None, target_transform=None, loader=default_loader,
        extensions=IMG_EXTENSIONS):
        classes, class_to_idx = self._find_classes(root)
        samples = make_dataset(root, class_to_idx, extensions)
        if len(samples) == 0:
            raise(RuntimeError("Found 0 files in subfolders of: " + root + "\n"
                               "Supported extensions are: " + ",".join(extensions)))

        self.root = root
        self.loader = loader
        self.extensions = extensions

        self.classes = classes
        self.class_to_idx = class_to_idx
        self.samples = samples
        self.imgs = self.samples
        self.targets = [s[1] for s in samples]
        self.labeled_idxs = []
        self.unlabeled_idxs = []
        self.all_labels = []

        self.p_labels = []#[-1] * len(self.imgs)
        self.p_weights = [0.0] * len(self.imgs)
        self.class_weights = np.ones((len(self.classes),),dtype = np.float32)

        self.transform = transform
        self.target_transform = target_transform
        self.images_lists = [[] for i in range(len(self.classes))]


        imfile_name = '%s/images.pkl' % self.root
        if os.path.isfile(imfile_name):
            print('Loading images in ram: %s' % imfile_name)
            with open(imfile_name, 'rb') as f:
                self.images = pickle.load(f)
        else:
            print('Not loading images in ram: %s' % imfile_name)
            self.images = None

    def _find_classes(self, dir):
        """
        Finds the class folders in a dataset.

        Args:
            dir (string): Root directory path.

        Returns:
            tuple: (classes, class_to_idx) where classes are relative to (dir), and class_to_idx is a dictionary.

        Ensures:
            No class is a subdirectory of another.
        """
        if sys.version_info >= (3, 5):
            # Faster and available in Python 3.5 and above
            classes = [d.name for d in os.scandir(dir) if d.is_dir()]
        else:
            classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        return classes, class_to_idx

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """

        path, target = self.samples[index]

        if (index not in self.labeled_idxs) and not isinstance(target,list):
            target = self.p_labels[index]

        if not self.p_weights:
            weight = 0.0
        elif index in self.labeled_idxs:
            weight = 1.0
        else:
            weight = self.p_weights[index]

        if self.images is not None:
            sample = Image.fromarray(self.images[index,:,:,:])
        else:
            sample = self.loader(path)

        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        if not isinstance(target,list):
            one_hot = [0]*len(self.classes)
            if target != -1:
                one_hot[target] = 1
            one_hot = torch.Tensor(one_hot)
        else:
            one_hot = torch.Tensor(target)
            target = self.p_labels[index]

        c_weight = self.class_weights[target]

        return sample, target, weight, one_hot, c_weight, index

    def __len__(self):
        return len(self.samples)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str

