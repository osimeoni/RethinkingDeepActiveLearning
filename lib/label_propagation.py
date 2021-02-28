# Label propagation implementated by: A. Iscen, G. Tolias, Y. Avrithis, O. Chum. 2018.
# Published in https://github.com/ahmetius/LP-DeepSSL

import faiss
import pdb
import scipy
import time
import torch
import random
import numpy as np

from tqdm import tqdm

from faiss import normalize_L2
from scipy.sparse import eye
from scipy.sparse import linalg as s_linalg
import torch.nn.functional as F
import lib.diffusion as diff

class LP():

    def __init__(self):
        super(LP, self).__init__()
        self.p_labels = []

    def update_lp(self, X, preds, dl, thresh = 0.9, k = 50, w_mode = True, w_criterion = "entropyl1",
                    temp = 1, y_pooling = 'mean', weight_path=None, args=None):
        alpha = 0.99
        labels = np.asarray(dl.all_labels)
        labeled_idx = np.asarray(dl.labeled_idxs)
        unlabeled_idx = np.asarray(dl.unlabeled_idxs)

        # kNN search for the graph
        d = X.shape[1]
        res = faiss.StandardGpuResources()
        flat_config = faiss.GpuIndexFlatConfig()
        flat_config.device = 0
        index = faiss.GpuIndexFlatIP(res,d,flat_config)

        normalize_L2(X)
        index.add(X)
        N = X.shape[0]

        c = time.time()
        D, I = index.search(X, k + 1)
        elapsed = time.time() - c

        # Create the graph
        D = D[:,1:] ** 3
        I = I[:,1:]
        row_idx = np.arange(N)
        row_idx_rep = np.tile(row_idx,(k,1)).T
        Wknn = scipy.sparse.csr_matrix((D.flatten('F'), (row_idx_rep.flatten('F'),
                                        I.flatten('F'))), shape=(N, N))

        W = Wknn
        W = W + W.T
        
        # Normalize the graph
        Wn = diff.normalize_connection_graph(W)

        # Initiliaze the y vector for each class and apply label propagation
        qsim = np.zeros((len(dl.classes),N))
        for i in range(len(dl.classes)):
            cur_idx = labeled_idx[np.where(labels[labeled_idx] == i)]
            if y_pooling == 'mean':
                qsim[i,cur_idx] = 1.0 / cur_idx.shape[0]
            elif y_pooling == 'sum':
                qsim[i,cur_idx] = 1.0
            else:
                raise ValueError("y_pooling method not defined.")

        cg_ranks, cg_sims =  diff.cg_diffusion(qsim, Wn, alpha , tol = 1e-6)

        if temp > 0:
            cg_sims_temp = cg_sims * temp

        p_labels = np.argmax(cg_sims,1)
        probs = F.softmax(torch.tensor(cg_sims)).numpy()
        probs_temp = F.softmax(torch.tensor(cg_sims_temp)).numpy()
        probs_temp[probs_temp <0] = 0

        prob_sort = np.amax(probs_temp, axis=1)
        p_labels = np.argmax(probs_temp,1)

        if w_criterion == "entropy":
            entropy = scipy.stats.entropy(probs_temp.T)
            weights = 1 - entropy / np.log(len(dl.classes))
        elif w_criterion == 'entropyl1':
            cg_sims[cg_sims < 0] = 0
            probs_l1 = F.normalize(torch.tensor(cg_sims),1).numpy()
            probs_l1[probs_l1 <0] = 0

            probs_temp = F.softmax(torch.tensor(probs_l1 * temp)).numpy()

            entropy = scipy.stats.entropy(probs_l1.T)
            weights = 1 - entropy / np.log(len(dl.classes))
            weights = weights / np.max(weights)

            p_labels = np.argmax(probs_l1,1)

        else:
            raise ValueError("weight criterion not defined.")

        # Compute the accuracy of pseudolabels for statistical purposes
        valid_idx = np.where(weights >= thresh)[0]
        correct_idx=(p_labels[valid_idx] == labels[valid_idx])
        acc = correct_idx.sum() / len(valid_idx)

        weight_idxs = np.argsort(-weights)

        # Number of images to be selected for pseudo labeling
        frag_N = int(round(len(unlabeled_idx) * args.lp_percent))

        if args.lp_mode == 'full':
            sub_weights_idx = weight_idxs
        elif args.lp_mode == 'uniform':
            selected_indices = np.random.choice(range(len(unlabeled_idx)), frag_N, replace=False)
            sub_weights_idx = unlabeled_idx[selected_indices]
        elif args.lp_mode == 'prob':
            p = weights[unlabeled_idx] / np.sum(weights[unlabeled_idx])

            # Keep only unlabeled images for pseudo label selection
            # Replace false to ensure unique selection
            selected_indices = np.random.choice(range(len(unlabeled_idx)), frag_N, p=p, replace=False)
            sub_weights_idx = unlabeled_idx[selected_indices]

            # Save histograms of weight into log dir
            if weight_path is not None:
                with open(weight_path, 'a') as f:
                    for w in weights[unlabeled_idx]:
                        f.write('%.4f;'%w)
                    f.write('\n')

                    for w in weights[sub_weights_idx]:
                        f.write('%.4f;'%w)

        dl.pseudo_label_idx = []
        for i in range(N):
            if i in valid_idx and (not i in labeled_idx) and (i in sub_weights_idx):
                dl.pseudo_label_idx.append(i)
                dl.p_labels[i] = p_labels[i]
                dl.p_weights[i] = weights[i]
                dl.samples[i] = (dl.samples[i][0], probs_temp[i,:].tolist())
            elif i in labeled_idx:
                dl.p_labels[i] = labels[i]
                dl.p_weights[i] = 1.0
            else:
                dl.p_labels[i] = -1
                dl.p_weights[i] = 0.0

        # Count of the pseudo and gt labels used from the dataset
        all_sel = np.sum(np.asarray(dl.p_labels) != -1)

        # Compute the weight for each class
        if w_mode:
            for i in range(len(dl.classes)):
                # Selection of instances of current class i
                cur_idx = np.where(np.asarray(dl.p_labels) == i)[0]
                dl.class_weights[i] = (float(all_sel) / len(dl.classes)) / cur_idx.size

        return acc, len(valid_idx)

def apply_LP(X, preds, dl, k = 50):
    # Apply Label Propagation

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

    Wknn = scipy.sparse.csr_matrix((D.flatten('F'), (row_idx_rep.flatten('F'), I.flatten('F'))),
                                    shape=(N, N))

    qsim = np.zeros((len(dl.classes),N))
    for i in range(len(dl.classes)):
        cur_idx = labeled_idx[np.where(labels[labeled_idx] == i)]
        qsim[i,cur_idx] = 1.0

    W = Wknn
    W = W + W.T
    Wn = diff.normalize_connection_graph(W)

    cg_ranks, cg_sims =  diff.cg_diffusion(qsim, Wn, alpha , tol = 1e-6)

    return cg_sims