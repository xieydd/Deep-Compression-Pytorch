import torch
import numpy as np
from sklearn.cluster import KMeans
from scipy.sparse import csc_matrix, csr_matrix


def apply_weight_sharing(model, bits=2):
    """ 
    Applies weight sharing to the given model
    """
    params = list(model.parameters())
    k = 0
    for i in params:
        l = 1
        print("The layer frame is " + str(list(i.size())))
        for j in i.size():
            l *= j
        print("This Layer parameter nums " + str(l))
        k = k + l
    print("All Layers parameter nums " + str(k))
    for name, param in model.named_parameters():
        print(name)
        if 'mask' in name:
            continue
        if 'weight' in name:
            #dev = module.weight.device
            #weights = module.weight.data.cpu().numpy()
            weights = param.data.cpu().numpy()
            dev = param.data.device
            shape = weights.shape
            #print(shape)
            if weights.ndim == 4:
                for i in range(weights.shape[0]):
                    if shape[3]  == 1:
                        continue
                    for j in range(weights.shape[1]):
                        mat = csr_matrix(weights[i,j,:,:]) if shape[2] < shape[3] else csc_matrix(weights[i,j,:,:])
                        min_ = min(mat.data)
                        max_ = max(mat.data)
                        space = np.linspace(min_, max_, num=2**bits)
                        kmeans = KMeans(n_clusters=len(space), init=space.reshape(-1,1), n_init=1, precompute_distances=True, algorithm="full")
                        kmeans.fit(mat.data.reshape(-1,1))
                        new_weight = kmeans.cluster_centers_[kmeans.labels_].reshape(-1)
                        print(new_weight)
                        mat.data = new_weight
                        param.data[i,j,:,:] = torch.from_numpy(mat.toarray()).to(dev)
        # elif 'bias' in name:
        #     dev = param.data.device
        #     bias = param.data.cpu().numpy()
        #     mat = np.zeros(bias.shape)
        #     #mat = mat.type(torch.cuda.FloatTensor)
        #     param.data = torch.from_numpy(mat).type(torch.cuda.FloatTensor).to(dev)

    return model

