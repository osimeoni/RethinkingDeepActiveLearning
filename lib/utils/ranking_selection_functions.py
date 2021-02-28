import numpy as np
import lib.selection_methods as sel_methods
import matplotlib.pyplot as plt

def for_rank(alg1, alg2, rank1, rank2, fig_name, amt):
    re_rank_2 = list()
    for it in range(amt):
        re_rank_2.append(int(np.where(rank2 == rank1[it])[0]))
    re_rank_2 = np.array(re_rank_2)
    plt.scatter(rank1, re_rank_2, s=0.01)
    plt.xlabel(str(alg1))
    plt.ylabel(str(alg2))
    plt.gca().set_aspect('equal', adjustable='box')
    plt.savefig(fig_name)
    plt.clf()

# FINISH WRITING 

def perform_ranking(model):
    print('Start ranking')
    alg_1 = sel_methods.UncertaintySamplingRank(model)
    alg_2 = sel_methods.UncertaintyEntropySamplingRank(model)

    amount = len(unlabeled_idx)
    rank_1 = alg_1.select(train_loader_noshuff, unlabeled_idx, amount)
    rank_2 = alg_2.select(train_loader_noshuff, unlabeled_idx, amount)

    re_rank_2 = list()
    for r in rank_1:
        re_rank_2.append(int(np.where(rank_2 == r)[0]))

    re_rank_2 = np.array(re_rank_2)
    plt.scatter(rank_1, re_rank_2, s=0.01)
    plt.xlabel(str(alg_1))
    plt.ylabel(str(alg_2))
    plt.gca().set_aspect('equal', adjustable='box')
    plt.savefig('ranking_uncer_uncer_entropy')
    plt.clf()


    alg_3 = sel_methods.CoreSetSampling(model)
    rank_3 = alg_3.select(train_loader_noshuff, labeled_idxs, unlabeled_idx, amount)
    for_rank(alg_1, alg_3, rank_1, rank_3, 'ranking_core_set_sampling', amount)

    alg_4 = sel_methods.DiffusionSelectionEntropy()
    feats, labels, preds = models.extract_features(train_loader_noshuff, model)
    rank_4 = alg_4.select(feats, preds, dataset, amount, step)
    for_rank(alg_1, alg_4, rank_1, rank_4, 'ranking_diffusion_selection_entropy', amount)

    alg_5 = sel_methods.DiffusionSelectionPerClass()
    rank_5 = alg_5.select(feats, preds, dataset, amount, step)
    for_rank(alg_1, alg_5, rank_1, rank_5, 'ranking_diffusion_selection_per_class', amount)

    alg_6 = sel_methods.LossSelection()
    rank_6 = alg_6.select(feats, preds, dataset, amount, step)
    for_rank(alg_1, alg_6, rank_1, rank_6, 'ranking_loss_selection', amount)
