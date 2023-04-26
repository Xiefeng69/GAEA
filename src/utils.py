import math
import numpy as np
import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn import preprocessing
import scipy
import scipy.sparse as sp
import random
import faiss
import json

def set_random_seed(seed_value=0):
    print(f"- current seed is {seed_value}\n")
    torch.manual_seed(seed_value)
    np.random.seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)
    random.seed(seed_value)

def save_args(args):
    '''save training arguments'''
    args_dict = args.__dict__
    if args.model == "gaea":
        with open(f'args/{args.task}.json', 'w', encoding='utf-8') as f:
            json.dump(args_dict, f)
    else:
        with open(f'args/{args.model}_{args.task}.json', 'w', encoding='utf-8') as f:
            json.dump(args_dict, f)

def normalize_adj(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv_sqrt = np.power(rowsum, -0.5).flatten()
    r_inv_sqrt[np.isinf(r_inv_sqrt)] = 0.
    r_mat_inv_sqrt = sp.diags(r_inv_sqrt)
    return mx.dot(r_mat_inv_sqrt).transpose().dot(r_mat_inv_sqrt)

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor"""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.FloatTensor(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def csls_sim(sim_mat, csls_k):
    """
    https://github.com/cambridgeltl/eva/blob/948ffbf3bf70cd3db1d3b8fcefc1660421d87a1f/src/utils.py#L287
    Compute pairwise csls similarity based on the input similarity matrix.
    Parameters
    ----------
    sim_mat : matrix-like
        A pairwise similarity matrix.
    csls_k : int
        The number of nearest neighbors.
    Returns
    -------
    csls_sim_mat : A csls similarity matrix of n1*n2.
    """

    nearest_values1 = torch.mean(torch.topk(sim_mat, csls_k)[0], 1)
    nearest_values2 = torch.mean(torch.topk(sim_mat.t(), csls_k)[0], 1)
    sim_mat = 2 * sim_mat.t() - nearest_values1
    sim_mat = sim_mat.t() - nearest_values2
    return sim_mat

def cal_distance(Lvec, Rvec, eval_metric, csls_k, eval_normalize=True):
    '''calculate distance/similarity matrix'''
    if eval_normalize:
        Lvec = preprocessing.normalize(Lvec)
        Rvec = preprocessing.normalize(Rvec)
    if eval_metric == "cityblock":
        sim_mat = scipy.spatial.distance.cdist(Lvec, Rvec, metric="cityblock")
    elif eval_metric == "euclidean":
        sim_mat = scipy.spatial.distance.cdist(Lvec, Rvec, metric="euclidean")
    elif eval_metric == "cosine":
        sim_mat = scipy.spatial.distance.cdist(Lvec, Rvec, metric="cosine") # 注意这里计算的余弦距离，即1-cos()，而不是余弦相似度
    elif eval_metric == "inner":
        sim_mat = 1 - np.matmul(Lvec, Rvec.T)
    elif eval_metric == "csls":
        sim_mat = scipy.spatial.distance.cdist(Lvec, Rvec, metric="cosine") # calculate pair-wise similarity
        del Lvec, Rvec
        sim_mat = 1 - csls_sim(1 - torch.from_numpy(sim_mat), csls_k=csls_k) # calculate csls similarity and transfer to distance form
        sim_mat = sim_mat.numpy()
    return sim_mat

def cal_metrics(sim_mat, Lvec, Rvec, data, k):
    '''calculate evaluation metrics: hit@1. hit@k, MRR, Mean etc.'''
    test_num = len(data)
    assert test_num == Lvec.shape[0]
    mrr = 0
    mean = 0
    hit_1_score = 0
    hit_k_score = 0
    for idx in range(Lvec.shape[0]):
        # np.argsort generate ascending sort: smaller is more similar
        rank = sim_mat[idx,:].argsort()
        assert idx in rank
        rank_index = np.where(rank==idx)[0][0]
        rank_index += 1
        mean += (rank_index)
        mrr += 1.0 / (rank_index)
        if rank_index <= 1: # hit@1
            hit_1_score += 1
        if rank_index <= k: # hit@k
            hit_k_score += 1
    mrr = mrr / test_num
    hit_1_score = hit_1_score / test_num
    hit_k_score = hit_k_score / test_num
    mean = mean / test_num
    return hit_1_score, hit_k_score, mrr, mean

def cal_metrics_faiss(Lvec, Rvec, data, k, eval_normalize=True):
    '''
        calculate evaluation metrics: hit@1, hit@k, MRR, Mean etc.
        using faiss accelerate alignment inference: https://github.com/facebookresearch/faiss
        faiss: Faiss is an effective similarity search tool that retrieves the top-k nearest neighbors for each source entity by Approximate Nearest Neighbor algorithm
    '''
    if eval_normalize:
        Lvec = preprocessing.normalize(Lvec)
        Rvec = preprocessing.normalize(Rvec)
    test_num = len(data)
    assert test_num == Lvec.shape[0]
    mrr = 0
    mean = 0
    hit_1_score = 0
    hit_k_score = 0
    index = faiss.IndexFlatL2(Rvec.shape[1]) # create index base with fixed dimension
    index.add(np.ascontiguousarray(Rvec)) # add key to index base
    _, I = index.search(np.ascontiguousarray(Lvec), test_num) # search query in index base
    for idx in range(Lvec.shape[0]):
        rank_index = np.where(I[idx,:]==idx)[0][0]
        rank_index += 1
        mean += (rank_index)
        mrr += 1.0 / (rank_index)
        if rank_index <= 1: # hit@1
            hit_1_score += 1
        if rank_index <= k: # hit@k
            hit_k_score += 1
    mrr = mrr / test_num
    hit_1_score = hit_1_score / test_num
    hit_k_score = hit_k_score / test_num
    mean = mean / test_num
    return hit_1_score, hit_k_score, mrr, mean