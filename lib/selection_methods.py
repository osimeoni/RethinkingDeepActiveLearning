# Authors: O. Simeoni, M. Budnik, Y. Avrithis, G. Gravier, 2019

import gc
import math
import random
import numpy as np

import torch
import torch.nn.functional as F
import torch.utils.data

import matplotlib
matplotlib.use('Agg')

import faiss
from faiss import normalize_L2

import scipy
import scipy.stats
from scipy.spatial import distance_matrix
from scipy.sparse import eye

import lib.diffusion as diff
from lib.models import extract_features

def get_unlabeled_idx(X_train, labeled_idx):
    """
    Given the training set and the indices of the labeled examples, return the indices of the unlabeled examples.
    """
    return np.arange(X_train.shape[0])[np.logical_not(np.in1d(np.arange(X_train.shape[0]), labeled_idx))]


class SelectionMethod:
    """
    A general class for selection methods.
    """

    def __init__(self, model):
        self.model = model

    def select(self, X_train, Y_train, labeled_idx, budget, args=None):
        """
        get the indices of labeled examples after the given budget have been queried by the query strategy.
        :param labeled_idx: the indices of the labeled examples
        :param budget: the budget of examples to query
        :return: the new labeled indices (including the ones queried)
        """
        return NotImplemented

    def update_model(self, new_model):
        del self.model
        gc.collect()
        self.model = new_model


class RandomSampling(SelectionMethod):
    """
    Random sampling baseline.
    """

    def __init__(self, model):
        super().__init__(model)

    def select(self, dataset, dl, budget, args=None):
        return np.random.choice(dl.unlabeled_idxs, budget, replace=False)


class UncertaintyEntropySampling(SelectionMethod):
    """
    Basic uncertainty sampling query strategy, querying the examples with the top entropy.
    """

    def __init__(self, model):
        super().__init__(model)

    def select(self, dataset, dl, budget, args=None):

        _, _, preds = extract_features(dataset, self.model)

        predictions = F.softmax(preds, dim=1).cuda()
        predictions = np.asarray(predictions.cpu().numpy())

        unlabeled_entropy = np.sum(predictions * np.log(predictions + 1e-10), axis=1)
        selected_indices = np.argpartition(unlabeled_entropy[dl.unlabeled_idxs], budget)[:budget]

        # unlabeled_idx --> dl.unlabeled_idxs
        return np.array(dl.unlabeled_idxs)[selected_indices]

class JLPSelection(SelectionMethod):
    """
    JLP acquisition function introduced in this work. 
    """

    def __init__(self, model):
        super().__init__(model)

    def select(self, dataset, dl, budget, thresh = 0.9, k = 50, w_mode = 3,
               w_criterion = "entropyl1", temp = 1, y_pooling = 'mean', args=None):
        
        # Extract features
        X, _, preds = extract_features(dataset, self.model)

        # Perform Diffusion
        alpha = 0.99
        d = X.shape[1]
        labels = np.asarray(dl.all_labels)
        labeled_idx = np.asarray(dl.labeled_idxs)

        res = faiss.StandardGpuResources()
        flat_config = faiss.GpuIndexFlatConfig()
        flat_config.device = 0
        index = faiss.GpuIndexFlatIP(res,d,flat_config)

        normalize_L2(X)
        index.add(X)
        N = X.shape[0]
        D, I = index.search(X, k + 1)
        D = D[:,1:] ** 3
        I = I[:,1:]
        row_idx = np.arange(N)
        row_idx_rep = np.tile(row_idx,(k,1)).T

        Wknn = scipy.sparse.csr_matrix((D.flatten('F'),
                                       (row_idx_rep.flatten('F'), I.flatten('F'))),
                                       shape=(N, N))

        qsim = np.zeros((N))
        for i in range(len(dl.classes)):
            cur_idx = labeled_idx[np.where(labels[labeled_idx] == i)]
            qsim[cur_idx] = 1.0 #/ cur_idx.shape[0]

        W = Wknn
        W = W + W.T

        Wn = diff.normalize_connection_graph(W)
        Wnn = eye(Wn.shape[0]) - alpha * Wn
        selected_indices = list()

        # Perform diffusion after adding the newly selected images
        for x in range(budget):
            cg_ranks, cg_sims =  diff.cg_diffusion_sel(qsim, Wnn, tol = 1e-6)

            cg_sims[labeled_idx] = 1.0
            if selected_indices:
                cg_sims[selected_indices] = 1.0
            ranks = np.argsort(cg_sims, axis = 0)

            it = 0
            while True:
                sel_id = ranks[it]
                it += 1
                if sel_id not in selected_indices and sel_id not in labeled_idx:
                    break

            qsim[sel_id] = 1.0
            selected_indices.append(sel_id)

        return selected_indices

class CoreSetSampling(SelectionMethod):
    """
    An implementation of the greedy core set query strategy published in https://arxiv.org/abs/1708.00489
    """

    def __init__(self, model):
        super().__init__(model)

    def greedy_k_center(self, labeled, unlabeled, budget):

        greedy_indices = []

        # get the minimum distances between the labeled and unlabeled examples (iteratively, to avoid memory issues):
        min_dist = np.min(distance_matrix(labeled[0, :].reshape((1, labeled.shape[1])), unlabeled),
                          axis=0)
        min_dist = min_dist.reshape((1, min_dist.shape[0]))
        for j in range(1, labeled.shape[0], 100):
            if j + 100 < labeled.shape[0]:
                dist = distance_matrix(labeled[j:j+100, :], unlabeled)
            else:
                dist = distance_matrix(labeled[j:, :], unlabeled)
            min_dist = np.vstack((min_dist, np.min(dist, axis=0).reshape((1, min_dist.shape[1]))))
            min_dist = np.min(min_dist, axis=0)
            min_dist = min_dist.reshape((1, min_dist.shape[0]))

        # iteratively insert the farthest index and recalculate the minimum distances:
        farthest = np.argmax(min_dist)
        greedy_indices.append(farthest)
        for i in range(budget-1):
            dist = distance_matrix(unlabeled[greedy_indices[-1], :].reshape((1,unlabeled.shape[1])),
                                   unlabeled)
            min_dist = np.vstack((min_dist, dist.reshape((1, min_dist.shape[1]))))
            min_dist = np.min(min_dist, axis=0)
            min_dist = min_dist.reshape((1, min_dist.shape[0]))
            farthest = np.argmax(min_dist)
            greedy_indices.append(farthest)

        return np.array(greedy_indices)

    def select(self, dataset, dl, budget, args=None):
        # Extract features
        representation, _, preds = extract_features(dataset, self.model)

        # Select the core sets
        new_indices = self.greedy_k_center(representation[dl.labeled_idxs, :],
                                           representation[dl.unlabeled_idxs, :], budget)
        
        return np.array(dl.unlabeled_idxs)[new_indices]

class CEAL():
    """
    An implementation of the CEAL acquisition function introduced in https://arxiv.org/abs/1701.03551
    """

    def __init__(self, model):
        self.model = model

    def select(self, dataset, dl, threshold, decay, cycle, args=None):

        _, _, preds = extract_features(dataset, self.model)

        predictions = F.softmax(preds, dim=1).cuda()
        predictions = np.asarray(predictions.cpu().numpy())

        threshold = threshold - decay*cycle
        
        pred_labels = np.argmax(predictions, axis=1)
        unlabeled_predictions = -np.sum((predictions[dl.unlabeled_idxs]*np.log(predictions[dl.unlabeled_idxs]+1e-10))/np.log(predictions.shape[1]), axis=1)
        new_labels = np.where(unlabeled_predictions < threshold)

        return np.array(dl.unlabeled_idxs)[new_labels], pred_labels[new_labels]
