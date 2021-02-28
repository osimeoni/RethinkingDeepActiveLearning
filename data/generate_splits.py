from glob import glob
import random
import pdb
import os
import argparse



parser = argparse.ArgumentParser(description='Generating splits for training')
parser.add_argument('--dataset', metavar='DATASET', default='cifar10',                
                        help='Dataset to generate splits from(default: cifar10)')
parser.add_argument('--nr-splits', default=5, type=int, metavar='N',
                        help='Number of splits to generate (default: 5)')                       
parser.add_argument('--split-size', default=100, type=int, metavar='N',
                        help='Split size (default: 100)')
args = parser.parse_args()
random.seed(7)

total_imgs = args.split_size
dataset = args.dataset

if 'cifar' in dataset:
    path = './data/images/cifar/%s/by-image/train' % (dataset)
else:
    path = './data/images/%s/by-image/train' % (dataset)
out_path = './data/labels/%s/%s_balanced_labels/' % (dataset, total_imgs)
if not os.path.exists(out_path):
        os.makedirs(out_path)

all_classes = set(glob(path+'/*')) - set(glob(path+'/*.pkl'))
splits = ["%.2d.txt" % i for i in range(args.nr_splits)]

 


per_class = int(total_imgs/len(all_classes))
print(per_class)
for x in splits:
	with open(out_path+x, 'w') as out:
		it = 0
		for cl in all_classes:
			c = cl.split('/')[-1].strip()
			all_images = glob(cl+'/*')
			nr_images = len(all_images)
			rand_subset = random.sample(range(nr_images), per_class)
			for idx in rand_subset:
				oy = all_images[idx].split('/')[-1].strip()
				print(it, oy)
				out.write(oy+' '+c+'\n')
				it = it + 1


