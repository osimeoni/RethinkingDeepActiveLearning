#!/bin/bash

# Download CIFAR10 and CIFAR100 dataset
cd data
wget https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
wget https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz
tar -xvf cifar-10-python.tar.gz
tar -xvf cifar-100-python.tar.gz

rm cifar-10-python.tar.gz
rm cifar-100-python.tar.gz

# Format
cd ../
python data_formatting.py --dataset cifar10
python generate_splits.py --dataset cifar10

python data_formatting.py --dataset cifar100
python generate_splits.py --dataset cifar100

# Format MNIST and SVHN datasets 
python data/data_formatting.py --dataset mnist 
python data/generate_splits.py --dataset mnist

rm data/train_32x32.mat
rm data/test_32x32.mat

python data/data_formatting.py --dataset svhn
python data/generate_splits.py --dataset svhn
