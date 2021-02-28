# Data loading implemented by Antti Tarvainen and Harri Valpola
# Published in https://github.com/CuriousAI/mean-teacher

"""Functions to load data from folders and augment it"""

import itertools
import logging
import os.path
import sys
import pickle

from PIL import Image
import numpy as np
from torch.utils.data.sampler import Sampler

LOG = logging.getLogger('main')
NO_LABEL = -1

import torch.utils.data

class RandomTranslateWithReflect:
    """Translate image randomly

    Translate vertically and horizontally by n pixels where
    n is integer drawn uniformly independently for each axis
    from [-max_translation, max_translation].

    Fill the uncovered blank area with reflect padding.
    """

    def __init__(self, max_translation):
        self.max_translation = max_translation

    def __call__(self, old_image):
        xtranslation, ytranslation = np.random.randint(-self.max_translation,
                                                       self.max_translation + 1,
                                                       size=2)
        xpad, ypad = abs(xtranslation), abs(ytranslation)
        xsize, ysize = old_image.size

        flipped_lr = old_image.transpose(Image.FLIP_LEFT_RIGHT)
        flipped_tb = old_image.transpose(Image.FLIP_TOP_BOTTOM)
        flipped_both = old_image.transpose(Image.ROTATE_180)

        new_image = Image.new("RGB", (xsize + 2 * xpad, ysize + 2 * ypad))

        new_image.paste(old_image, (xpad, ypad))

        new_image.paste(flipped_lr, (xpad + xsize - 1, ypad))
        new_image.paste(flipped_lr, (xpad - xsize + 1, ypad))

        new_image.paste(flipped_tb, (xpad, ypad + ysize - 1))
        new_image.paste(flipped_tb, (xpad, ypad - ysize + 1))

        new_image.paste(flipped_both, (xpad - xsize + 1, ypad - ysize + 1))
        new_image.paste(flipped_both, (xpad + xsize - 1, ypad - ysize + 1))
        new_image.paste(flipped_both, (xpad - xsize + 1, ypad + ysize - 1))
        new_image.paste(flipped_both, (xpad + xsize - 1, ypad + ysize - 1))

        new_image = new_image.crop((xpad - xtranslation,
                                    ypad - ytranslation,
                                    xpad + xsize - xtranslation,
                                    ypad + ysize - ytranslation))

        return new_image


class TransformTwice:
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, inp):
        out1 = self.transform(inp)
        out2 = self.transform(inp)
        return out1, out2


def update_dataset_semi(dataset, semi_labels_idxs, semi_labels):
    it = 0
    for idx in semi_labels_idxs:
        path, _ = dataset.imgs[idx]
        dataset.imgs[idx] = path, semi_labels[it]
        #labeled_idxs.append(idx)
        it += 1

def remove_semi_labels(dataset, semi_labels):
    for idx in semi_labels:
        path, _ = dataset.imgs[idx]
        dataset.imgs[idx] = path, -1

def update_dataset_resuming(dataset, labeled_idxs):
    for idx in labeled_idxs:
        path, _ = dataset.imgs[idx]
        label = path.split('/')[-2]
        label_idx = dataset.class_to_idx[label]
        dataset.imgs[idx] = path, label_idx

    unlabeled_idxs = sorted(set(range(len(dataset.imgs))) - set(labeled_idxs))
    dataset.labeled_idxs = labeled_idxs
    dataset.unlabeled_idxs = unlabeled_idxs
    return unlabeled_idxs

def update_dataset(dataset, to_label_idx, labeled_idxs):
    for idx in to_label_idx:
        path, _ = dataset.imgs[idx]
        label = path.split('/')[-2]
        label_idx = dataset.class_to_idx[label]
        dataset.imgs[idx] = path, label_idx
        labeled_idxs.append(idx)

    unlabeled_idxs = sorted(set(range(len(dataset.imgs))) - set(labeled_idxs))
    dataset.labeled_idxs = labeled_idxs
    dataset.unlabeled_idxs = unlabeled_idxs
    return labeled_idxs, unlabeled_idxs


def relabel_dataset(dataset, labels):
    unlabeled_idxs = []
    for idx in range(len(dataset.imgs)):
        path, orig_label = dataset.imgs[idx]

        filename = os.path.basename(path)
        dataset.all_labels.append(orig_label)
        dataset.p_labels.append(-1)
        if filename in labels:
            label_idx = dataset.class_to_idx[labels[filename]]
            dataset.imgs[idx] = path, label_idx
            del labels[filename]
        else:
            dataset.imgs[idx] = path, NO_LABEL
            unlabeled_idxs.append(idx)

    if len(labels) != 0:
        message = "List of unlabeled contains {} unknown files: {}, ..."
        some_missing = ', '.join(list(labels.keys())[:5])
        raise LookupError(message.format(len(labels), some_missing))

    labeled_idxs = sorted(set(range(len(dataset.imgs))) - set(unlabeled_idxs))
    dataset.labeled_idxs = labeled_idxs
    dataset.unlabeled_idxs = unlabeled_idxs
    return labeled_idxs, unlabeled_idxs


def relabel_dataset_preprocessing(dataset, labels):
    unlabeled_idxs = []
    for idx in range(len(dataset.imgs)):
        path, orig_label = dataset.imgs[idx]

        dataset.all_labels.append(orig_label)
        dataset.p_labels.append(-1)

        dataset.imgs[idx] = path, labels[idx]

    labeled_idxs = sorted(set(range(len(dataset.imgs))) - set(unlabeled_idxs))
    dataset.labeled_idxs = labeled_idxs
    dataset.unlabeled_idxs = unlabeled_idxs
    return labeled_idxs, unlabeled_idxs

class TwoStreamBatchSampler(Sampler):
    """Iterate two sets of indices

    An 'epoch' is one iteration through the primary indices.
    During the epoch, the secondary indices are iterated through
    as many times as needed.
    """
    def __init__(self, primary_indices, secondary_indices, batch_size, secondary_batch_size):
        self.primary_indices = primary_indices
        self.secondary_indices = secondary_indices
        self.secondary_batch_size = secondary_batch_size
        self.primary_batch_size = batch_size - secondary_batch_size

        assert len(self.primary_indices) >= self.primary_batch_size > 0
        assert len(self.secondary_indices) >= self.secondary_batch_size > 0

    def __iter__(self):
        primary_iter = iterate_once(self.primary_indices)
        secondary_iter = iterate_eternally(self.secondary_indices)
        return (
            primary_batch + secondary_batch
            for (primary_batch, secondary_batch)
            in  zip(grouper(primary_iter, self.primary_batch_size),
                    grouper(secondary_iter, self.secondary_batch_size))
        )

    def __len__(self):
        return len(self.primary_indices) // self.primary_batch_size

    def update_pseudo_indices(self, indices, same_length = True):
        print('Update pseudo indices')
        if same_length and (not len(indices) == len(self.primary_indices)):
            raise ValueError("weights vector should be the same lenght \
                              as indices (%d vs %d)"%(len(indices), len(self.primary_indices)))
        self.primary_indices = indices

def iterate_once(iterable):
    return np.random.permutation(iterable)


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

        self.p_labels = [] #[-1] * len(self.imgs)
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

