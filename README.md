# Rethinking Deep Active Learning

This repository provides a pytorch framework that can be used to compare different active learning methods under different setups. We additionally propose to use knowledge acquired from unlabeled data by adding unsupervised and semi-supervised methods. Among others, this code allows you to reproduce the following results:

![alt text](https://github.com/osimeoni/RethinkingDeepActiveLearning/blob/main/al_results.PNG?raw=true)

Those results were published in the paper:

**Rethinking deep active learning: Using unlabeled data at model training** 
[Siméoni O.](http://people.rennes.inria.fr/Oriane.Simeoni/), [Budnik M.](https://scholar.google.com/citations?user=t00kcd0AAAAJ&hl=en&oi=ao), [Avrithis Y.](https://avrithis.net/), [Gravier G.](https://scholar.google.com/citations?user=MbFGBKwAAAAJ&hl=fr&oi=ao)
ICPR 2020 [[arXiv](https://arxiv.org/abs/1911.08177)]


## Dependencies 

1. Python3 (tested on version 3.6)
1. PyTorch (tested on version 1.4)
1. [FAISS](https://github.com/facebookresearch/faiss) 

If using conda, you can run the following commands:
```
conda create -n rethinkingAL python=3.6
conda activate rethinkingAL
```
And then install the python packages using
```
pip install -r requirements.txt
conda install -c pytorch faiss-gpu
```

## Setting up the datasets 

In order to download and format the datasets **CIFAR10**, **CIFAR100**, **MNIST** and **SVHN**, please run the following script:

```
sh data/install_datasets.sh 
```

You can create additional splits with different sizes by running the script below. By default the script creates 5 splits with 100 balanced labels. For example, 10 splits with the size of 1000 for **CIFAR10**:

```
python data/generate_splits.py --dataset cifar10 --nr-splits 10 --split-size 1000
```

## Usage

### Launching
In our paper, we propose to test different AL baselines method with the addition of unsupervised pre-training and semi-supervised methods. 

Following are the command lines to launch learning for one split (here *slit 0* - for more repetitions, run different split) with method *uncertainty_entropy* on the dataset *cifar10*. The *--dataset* can currently take as input *cifar10*, *cifar100*, *svhn* or *mnist*. 

```
python main_al.py --dataset cifar10 --al-method uncertainty_entropy --split 0 --al-budget 100 
```

It is possible to add unsupervised pretraining (*--add-unsupervised-pretraining True*). We follow the unsupervised method [DeepCluster](https://arxiv.org/pdf/1807.05520.pdf). 
```
python main_al.py --dataset cifar10 --al-method uncertainty_entropy --split 0 --al-budget 100 --add-unsupervised-pretraining True
```

We also implemented the addition of the label-propagation method semi-supervised (*--add-lp True*) following [Iscen et al.](https://arxiv.org/pdf/1904.04717.pdf)
```
python main_al.py --dataset cifar10 --al-method uncertainty_entropy --split 0 --al-budget 100 --add-lp True --b 128 --labeled-batch-size 50
```

The *al-method* argument can take any of the four following values corresponding to the different methods we have implemented:
- random
- uncertainty_entropy 
- [coreset](https://arxiv.org/abs/1708.00489)
- jlp (our AL acquisition function based on label-propagation)

The semi-supervised method [CEAL](https://arxiv.org/abs/1701.03551) can be added to any previous acquisition functions using the *add-ceal* parameter. Following is an example:

```
python main_al.py --dataset cifar10 --al-method uncertainty_entropy --split 0 --al-budget 100 --add-ceal True
```

## Citation

If you use our work, please cite us using:

```
@conference{SMAG20,
   title = {Rethinking deep active learning: Using unlabeled data at model training},
   author = {O. Sim\'eoni and M. Budnik and Y. Avrithis and G. Gravier},
   booktitle = {Proceedings of International Conference on Pattern Recognition (ICPR)},
   month = {12},
   address = {Virtual},
   year = {2020}
}
```

## Acknowledgements

The code is based on the [Mean Teacher Pytorch](https://github.com/CuriousAI/mean-teacher/tree/master/pytorch), the [LabelProp-SSDL](https://github.com/ahmetius/LP-DeepSSL) and the [DeepCluster](https://github.com/facebookresearch/deepcluster) implementations.
