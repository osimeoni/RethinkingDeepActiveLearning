import numpy as np
import faiss
import torch
import time
import pickle
from lib import models, data
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler


def run_kmeans(x, nmb_clusters, verbose=False):
    """Runs kmeans on 1 GPU.
    Args:
        x: data
        nmb_clusters (int): number of clusters
    Returns:
        list: ids of data in each cluster
    """
    n_data, d = x.shape

    # faiss implementation of k-means
    clus = faiss.Clustering(d, nmb_clusters)
    clus.niter = 10
    clus.max_points_per_centroid = 10000000
    res = faiss.StandardGpuResources()
    flat_config = faiss.GpuIndexFlatConfig()
    flat_config.useFloat16 = False
    flat_config.device = 0
    index = faiss.GpuIndexFlatL2(res, d, flat_config)

    # perform the training
    clus.train(x, index)
    _, I = index.search(x, 1)
    
    stats = clus.iteration_stats
    losses = np.array([
        stats.at(i).obj for i in range(stats.size())
    ])
    if verbose:
        print('k-means loss evolution: {0}'.format(losses))

    return [int(n[0]) for n in I], losses[-1]

def preprocess_features(npdata, pca=128):
    """Preprocess an array of features.
    Args:
        npdata (np.array N * ndim): features to preprocess
        pca (int): dim of output
    Returns:
        np.array of dim N * pca: data PCA-reduced, whitened and L2-normalized
    """
    _, ndim = npdata.shape
    npdata =  npdata.astype('float32')

    # Using PCA didn't help in our case.
    
    # Apply PCA-whitening with Faiss
    #mat = faiss.PCAMatrix (ndim, pca, eigen_power=-0.9)
    #mat.train(npdata)
    #assert mat.is_trained
    #npdata = mat.apply_py(npdata)


    # L2 normalization
    row_sums = np.linalg.norm(npdata, axis=1)
    npdata = npdata / row_sums[:, np.newaxis]

    return npdata


def arrange_clustering(images_lists):
    pseudolabels = []
    image_indexes = []
    for cluster, images in enumerate(images_lists):
        image_indexes.extend(images)
        pseudolabels.extend([cluster] * len(images))
    indexes = np.argsort(image_indexes)
    return np.asarray(pseudolabels)[indexes]
    
   
def pretrain(args, dataset, num_classes, train_loader_noshuff,  
             pre_epochs = 400, class_mltp = 5, pretrain_eval = True,
             save_every = 10, eval_every = 20):
    
    out_path = './models/pretrained/'
    # PRETRAINING CREATION    
    model = models.create_model(args, class_mltp * num_classes)
    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                        momentum=args.momentum,
                        weight_decay=args.weight_decay,
                        nesterov=args.nesterov)

    dataset.classes = range(class_mltp * num_classes)
    dataset.class_weights = np.ones((len(dataset.classes),),dtype = np.float32)
    
    
    print('Pretraining model...')
    for epoch in range(pre_epochs):
        print('Pretrain epoch: ', epoch)
        step_t = time.time()
        feats, labels, preds = models.extract_features(train_loader_noshuff, model)
        feats = preprocess_features(feats)
        km_I, km_losses = run_kmeans(feats, class_mltp * num_classes)
        l_idx, unl_idx = data.relabel_dataset_preprocessing(dataset, km_I)
        dataset.labeled_idxs = l_idx

        sampler = SubsetRandomSampler(np.array(l_idx))
        batch_sampler = BatchSampler(sampler, args.batch_size, drop_last=True) # Hard coded batch size!!!
        train_loader_pretrain = torch.utils.data.DataLoader(dataset,
                                               batch_sampler=batch_sampler,
                                               num_workers=args.workers,
                                               pin_memory=True)

        train_meter = models.train(train_loader_pretrain, model, optimizer, epoch, args)

        if epoch % save_every == 0 and (epoch > 0):
            stored_model_keys = list(model.state_dict().keys())[:-6]
            msd =  model.state_dict()
            pretrained_dict = {key:msd[key] for key in stored_model_keys}
            with open('%s/pretrained_%s_%s_lr_%s_batch_%s_%s.pickle' % 
                      (out_path, args.dataset, args.arch, args.lr, args.batch_size, args.epoch), 'wb') as handle:
                pickle.dump(pretrained_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
            print('Pretrained model saved after '+str(epoch))
        
        if epoch % eval_every == 0  and (epoch > 0) and pretrain_eval == True:
            stored_model_keys = list(model.state_dict().keys())[:-6]
            msd =  model.state_dict()
            pretrained_dict = {key:msd[key] for key in stored_model_keys}
            print('Evaluating pretraining...')
            model_eval = models.create_model(args, num_classes)
            optimizer_eval = torch.optim.SGD(model_eval.parameters(), args.lr,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay,
                                    nesterov=args.nesterov)

            model_dict = model_eval.state_dict()
            model_dict.update(pretrained_dict)
            model_eval.load_state_dict(model_dict)

            for epoch_eval in range(args.epochs):
                models.train(train_loader, model_eval, optimizer_eval, epoch_eval, args)

            with open(log_file, 'a') as log:
                log.write('Pretraining results after %s pretraining epochs\n' % epoch)
                with open(results_file, 'a') as log_results:
                    prec1, prec5 = models.validate(eval_loader, model_eval, epoch_eval,
                                                   log, 0, log_results, 'Val set')
        print('It took: ', time.time()-step_t)
    stored_model_keys = list(model.state_dict().keys())[:-6]
    msd =  model.state_dict()

    pretrained_dict = {key:msd[key] for key in stored_model_keys}
    
    with open('%s/pretrained_%s_%s_lr_%s_batch_%s_final.pickle' % 
              (out_path, args.dataset, args.arch, args.lr, args.batch_size), 'wb') as handle:
        pickle.dump(pretrained_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
    return pretrained_dict
