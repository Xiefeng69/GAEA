import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def margin_based_loss(embeddings1, embeddings2, a1_align, a2_align, neg1_left, neg1_right, neg2_left, neg2_right, neg_samples_size, loss_norm, pos_margin=0.01, neg_margin=3, neg_param=0.2, only_pos=False):
    # process the ground truth
    a1_align = np.array(a1_align)
    a2_align = np.array(a2_align)
    t = len(a1_align)
    L = np.ones((t, neg_samples_size)) * (a1_align.reshape((t,1)))
    a1_align = L.reshape((t*neg_samples_size,))
    R = np.ones((t, neg_samples_size)) * (a2_align.reshape((t,1)))
    a2_align = R.reshape((t*neg_samples_size,))
    del L, R;
    # convert to tensor
    a1_align = torch.tensor(a1_align)
    a2_align = torch.tensor(a2_align)
    neg1_left = torch.tensor(neg1_left)
    neg1_right = torch.tensor(neg1_right)
    neg2_left = torch.tensor(neg2_left)
    neg2_right = torch.tensor(neg2_right)
    # positive pair loss computation
    left_x = embeddings1[a1_align.long()]
    right_x = embeddings2[a2_align.long()]
    if loss_norm == "l1": # L1 normal
        pos_loss = torch.abs(left_x - right_x)
    elif loss_norm == 'l2': # L2 normal
        pos_loss = torch.square(left_x - right_x)
    pos_loss = torch.sum(pos_loss, dim=1)
    # negative pair loss computation on neg_1
    left_x = embeddings1[neg1_left.long()]
    right_x = embeddings2[neg1_right.long()]
    if loss_norm == "l1":
        neg_loss_1 = torch.abs(left_x - right_x)
    elif loss_norm == 'l2':
        neg_loss_1 = torch.square(left_x - right_x)
    neg_loss_1 = torch.sum(neg_loss_1, dim=1)
    # negative pair loss computation on neg_2
    left_x = embeddings1[neg2_left.long()]
    right_x = embeddings2[neg2_right.long()]
    if loss_norm == "l1":
        neg_loss_2 = torch.abs(left_x - right_x)
    elif loss_norm == 'l2':
        neg_loss_2 = torch.square(left_x - right_x)
    neg_loss_2 = torch.sum(neg_loss_2, dim=1)
    # loss summation
    loss1 = F.relu(pos_loss + neg_margin - neg_loss_1)
    loss2 = F.relu(pos_loss + neg_margin - neg_loss_2)
    loss1 = torch.sum(loss1)
    loss2 = torch.sum(loss2)
    return loss1 + loss2

def limited_based_loss(embeddings1, embeddings2, a1_align, a2_align, neg1_left, neg1_right, neg2_left, neg2_right, neg_samples_size, loss_norm, pos_margin=0.01, neg_margin=3, neg_param=0.2, only_pos=False):
    # process the ground truth
    a1_align = np.array(a1_align)
    a2_align = np.array(a2_align)
    t = len(a1_align)
    L = np.ones((t, neg_samples_size)) * (a1_align.reshape((t,1)))
    a1_align = L.reshape((t*neg_samples_size,))
    L = np.ones((t, neg_samples_size)) * (a2_align.reshape((t,1)))
    a2_align = L.reshape((t*neg_samples_size,))
    # convert to tensor
    a1_align = torch.tensor(a1_align)
    a2_align = torch.tensor(a2_align)
    neg1_left = torch.tensor(neg1_left)
    neg1_right = torch.tensor(neg1_right)
    neg2_left = torch.tensor(neg2_left)
    neg2_right = torch.tensor(neg2_right)
    # positive pair loss computation
    left_x = embeddings1[a1_align.long()]
    right_x = embeddings2[a2_align.long()]
    if loss_norm == "l1": # L1 normal
        pos_loss = torch.abs(left_x - right_x)
    elif loss_norm == 'l2': # L2 normal
        pos_loss = torch.square(left_x - right_x)
    pos_loss = torch.sum(pos_loss, dim=1)
    # negative pair loss computation on neg_1
    left_x = embeddings1[neg1_left.long()]
    right_x = embeddings2[neg1_right.long()]
    if loss_norm == "l1":
        neg_loss_1 = torch.abs(left_x - right_x)
    elif loss_norm == 'l2':
        neg_loss_1 = torch.square(left_x - right_x)
    neg_loss_1 = torch.sum(neg_loss_1, dim=1)
    # negative pair loss computation on neg_2
    left_x = embeddings1[neg2_left.long()]
    right_x = embeddings2[neg2_right.long()]
    if loss_norm == "l1":
        neg_loss_2 = torch.abs(left_x - right_x)
    elif loss_norm == 'l2':
        neg_loss_2 = torch.square(left_x - right_x)
    neg_loss_2 = torch.sum(neg_loss_2, dim=1)
    # loss summation
    loss1 = F.relu(pos_loss - pos_margin) + neg_param * F.relu(neg_margin - neg_loss_1)
    loss2 = F.relu(pos_loss - pos_margin) + neg_param * F.relu(neg_margin - neg_loss_2)
    loss1 = torch.sum(loss1)
    loss2 = torch.sum(loss2)
    return loss1 + loss2

def customized_loss_for_aug(embeddings1, embeddings2, a1_align, a2_align, neg_samples_size, pos_margin=0.01, neg_margin=3, neg_param=0.2, only_pos=False):
    # convert to tensor
    a1_align = torch.tensor(a1_align)
    a2_align = torch.tensor(a2_align)
    # positive pair loss computation
    left_x = embeddings1[a1_align.long()]
    right_x = embeddings2[a2_align.long()]
    pos_loss = torch.abs(left_x - right_x)
    pos_loss = torch.sum(pos_loss, dim=1)
    loss1 = F.relu(pos_loss)
    loss2 = F.relu(pos_loss)
    loss1 = torch.sum(loss1)
    loss2 = torch.sum(loss2)
    return loss1+loss2