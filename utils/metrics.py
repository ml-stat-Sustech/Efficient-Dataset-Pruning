import numpy as np

import torch


# def cd_dist(x, y):
#     # metric function for contextual diversity
#     m, n = x.size(0), y.size(0)
#     i, j = x.size(1), y.size(1)
#     xx = x.view(m, 1, i).repeat(1, n, 1)
#     yy = y.view(1, n, j).repeat(m, 1, 1)
    
#     return torch.sum(0.5 * xx * torch.log(xx / yy) + 0.5 * yy * torch.log(yy / xx), dim=2)

def euclidean_dist(x, y):
    m, n = x.size(0), y.size(0)
    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
    yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
    dist = xx + yy
    # dist.addmm_(1, -2, x, y.t())
    dist = torch.addmm(dist, x, y.t(), beta=1, alpha=-2)
    dist = dist.clamp(min=1e-12).sqrt()

    return dist

def euclidean_dist_pair_np(x):
    (rowx, colx) = x.shape
    xy = np.dot(x, x.T)
    x2 = np.repeat(np.reshape(np.sum(np.multiply(x, x), axis=1), (rowx, 1)), repeats=rowx, axis=1)
    return np.sqrt(np.clip(x2 + x2.T - 2. * xy, 1e-12, None))

# def cossim_pair_np(v1):
#     num = np.dot(v1, v1.T)
#     norm = np.linalg.norm(v1, axis=1)
#     denom = norm.reshape(-1, 1) * norm
#     res = num / denom
#     res[np.isneginf(res)] = 0.
#     return 0.5 + 0.5 * res

def cossim_np(v1, v2):
    num = np.dot(v1, v2.T)
    denom = np.linalg.norm(v1, axis=1).reshape(-1, 1) * np.linalg.norm(v2, axis=1)
    res = num / denom
    res[np.isneginf(res)] = 0.

    return 0.5 + 0.5 * res

