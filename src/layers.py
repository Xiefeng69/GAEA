import math
import torch
import torch.nn as nn
from torch import sparse
import numpy as np
import torch.nn.functional as F

class GraphConvLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout=0.2, bias=True, act=True, transform=True, residual=False, mean=False):
        super(GraphConvLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(in_features, out_features))
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.residual = residual
        self.mean = mean
        self.has_act = act
        self.has_transform = transform

        nn.init.xavier_uniform_(self.weight)
        
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
            stdv = 1. / math.sqrt(self.bias.size(0))
            self.bias.data.uniform_(-stdv, stdv)
        else:
            self.register_parameter('bias', None)

    def forward(self, feature, adj):
        if self.has_transform:
            support = torch.matmul(feature, self.weight) # when remove the linear transformation ==> RREA
        else:
            support = feature
        output = torch.matmul(adj, support)

        if self.residual and self.in_features == self.out_features: # add residual link
            output = output + feature
        
        if self.mean: # MeanAggregator
            degree = torch.sparse.sum(adj, dim=1).to_dense().reshape(-1, 1)
            output = output / degree

        if self.bias is not None:
            if self.has_act:
                return self.dropout(self.act(output + self.bias))
            else:
                return self.dropout(output+self.bias)
        else:
            if self.has_act:
                return self.dropout(self.act(output))
            else:
                return self.dropout(output)

class SpGraphAttentionLayer(nn.Module):
    """
    Sparse version GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, in_features, out_features, dropout, alpha=0.2, act=True, transform=True, w_adj=True, cuda=True, residual=False):
        super(SpGraphAttentionLayer, self).__init__()
        assert in_features == out_features
        self.w_adj = w_adj
        self.is_cuda = cuda
        self.has_act = act
        self.has_transform = transform
        self.residual = residual
        self.in_features = in_features
        self.out_features = out_features
        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features), dtype=torch.float))
        self.a = nn.Parameter(torch.zeros(size=(1, 2 * out_features), dtype=torch.float32))
        # nn.init.ones_(self.W.data)
        # stdv = 1. / math.sqrt(in_features * 2)
        # nn.init.uniform_(self.a.data, -stdv, stdv)
        nn.init.xavier_normal_(self.W.data, gain=1.414)
        nn.init.xavier_normal_(self.a.data, gain=1.414)

        self.dropout = nn.Dropout(dropout)
        self.leakyrelu = nn.LeakyReLU(alpha)

    def forward(self, inputs, adj):
        N = inputs.size()[0]
        ones = torch.ones(size=(N, 1), dtype=torch.float32)
        if self.is_cuda:
            ones = ones.to(inputs.device)

        edge = adj._indices()
        h = torch.mm(inputs, self.W)  # transformation

        assert not torch.isnan(h).any()
        edge_h = torch.cat((h[edge[0, :], :], h[edge[1, :], :]), dim=1).t()
        edge_e = torch.exp(self.leakyrelu(self.a.mm(edge_h).squeeze()))
        assert not torch.isnan(edge_e).any()

        e_rowsum = sparse.mm(torch.sparse_coo_tensor(edge, edge_e), ones)
        if self.has_transform:
            h_prime = sparse.mm(torch.sparse_coo_tensor(edge, edge_e), h)
        else:
            h_prime = sparse.mm(torch.sparse_coo_tensor(edge, edge_e), inputs)

        assert not torch.isnan(h_prime).any()
        h_prime = h_prime.div(e_rowsum)
        assert not torch.isnan(h_prime).any()
        
        if self.has_act:
            output = F.elu(h_prime)
        else:
            output = h_prime

        if self.residual:
            output = inputs + output
            assert output.size() == inputs.size()
        return output

class DoubleEmbedding(nn.Module):
    def __init__(self, num_sr, num_tg, embedding_dim, init_type='xavier'):
        super(DoubleEmbedding, self).__init__()
        self.embedding_sr = nn.Embedding(num_sr, embedding_dim,
                                         _weight=torch.zeros((num_sr, embedding_dim), dtype=torch.float))
        self.embedding_tg = nn.Embedding(num_tg, embedding_dim,
                                         _weight=torch.zeros((num_tg, embedding_dim), dtype=torch.float))

        if init_type == 'xavier':
            nn.init.xavier_uniform_(self.embedding_sr.weight.data)
            nn.init.xavier_uniform_(self.embedding_tg.weight.data)
        elif init_type == 'normal':
            nn.init.normal_(self.embedding_sr.weight.data, std=1. / math.sqrt(num_sr))
            nn.init.normal_(self.embedding_tg.weight.data, std=1. / math.sqrt(num_tg))
        else:
            raise NotImplementedError

    def normalize(self):
        self.embedding_sr.weight.data = F.normalize(self.embedding_sr.weight, dim=-1, p=2)
        self.embedding_tg.weight.data = F.normalize(self.embedding_tg.weight, dim=-1, p=2)

    def forward(self, sr_data, tg_data):
        return self.embedding_sr(sr_data), self.embedding_tg(tg_data)

    @property
    def weight(self):
        return self.embedding_sr.weight, self.embedding_tg.weight

class GraphMultiHeadAttLayer(nn.Module):
    def __init__(self, in_features, out_features, n_head=2, dropout=0.2, alpha=0.2, cuda=True):
        super(GraphMultiHeadAttLayer, self).__init__()
        self.attentions = nn.ModuleList([AttentiveAggregator(in_features, out_features, alpha, dropout, cuda) for _ in range(n_head)])

    def forward(self, inputs, adj):
        # inputs shape = [num, ent_dim]
        outputs = torch.cat(tuple([att(inputs, adj).unsqueeze(-1) for att in self.attentions]), dim=-1) # shape = [num, ent_dim, nheads]
        outputs = torch.mean(outputs, dim=-1)
        return outputs

class AttentiveAggregator(nn.Module):
    def __init__(self, in_features, out_features, alpha=0.2, dropout=0.2, cuda=True):
        super(AttentiveAggregator, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = nn.Dropout(dropout)
        self.leakyrelu = nn.LeakyReLU(alpha)
        self.W = nn.Parameter(torch.zeros(size=(out_features,), dtype=torch.float))
        # self.W = nn.Parameter(torch.zeros(size=(in_features, out_features), dtype=torch.float))
        self.a = nn.Parameter(torch.zeros(size=(1, 2 * out_features), dtype=torch.float32))
        nn.init.ones_(self.W.data)
        stdv = 1. / math.sqrt(in_features * 2)
        nn.init.uniform_(self.a.data, -stdv, stdv)
        self.is_cuda = cuda

    def forward(self, inputs, adj):
        N = inputs.size()[0]
        ones = torch.ones(size=(N, 1), dtype=torch.float32)
        if self.is_cuda:
            ones = ones.to(inputs.device)

        edge = adj._indices()
        h = torch.mul(inputs, self.W)  # transformation: h = torch.mm(inputs, self.W)

        assert not torch.isnan(h).any()
        edge_h = torch.cat((h[edge[0, :], :], h[edge[1, :], :]), dim=1).t()
        edge_e = torch.exp(self.leakyrelu(self.a.mm(edge_h).squeeze()))
        assert not torch.isnan(edge_e).any()

        e_rowsum = sparse.mm(torch.sparse_coo_tensor(edge, edge_e), ones)
        h_prime = sparse.mm(torch.sparse_coo_tensor(edge, edge_e), inputs)

        assert not torch.isnan(h_prime).any()
        h_prime = h_prime.div(e_rowsum)
        assert not torch.isnan(h_prime).any()

        return h_prime

class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.2):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.dropout = nn.Dropout(dropout)
        self.w_qs = nn.Linear(d_model, n_head * d_k)
        self.w_ks = nn.Linear(d_model, n_head * d_k)
        # self.w_vs = nn.Linear(d_model, n_head * d_v) # ablation
        nn.init.normal_(self.w_qs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_ks.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        # nn.init.normal_(self.w_vs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_v))) # ablation
        self.attention = ScaledDotProductAttention(temperature=np.power(d_k, 0.5), attn_dropout=dropout)

    def forward(self, q, k, v):

        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head

        sz_b, len_q, _ = q.size()
        sz_b, len_k, _ = k.size()
        sz_b, len_v, _ = v.size()
        
        # residual = v.repeat(1,1,n_head)
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = v.unsqueeze(2).repeat(1,1,n_head,1).view(sz_b, len_v, n_head, d_v)
        # v = self.w_vs(v).view(sz_b, len_v, n_head, d_v) # ablation

        q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_q, d_k)
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_k, d_k)
        v = v.permute(2, 0, 1, 3).contiguous().view(-1, len_v, d_v)

        output, attn = self.attention(q, k, v)

        output = output.view(n_head, sz_b, len_q, d_v)
        output = output.permute(1, 2, 0, 3).contiguous().view(sz_b, len_q, -1)

        return output, attn

class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.2):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v):

        attn = torch.bmm(q, k.transpose(1, 2))
        attn = attn / self.temperature

        attn = self.softmax(attn)
        attn = self.dropout(attn)
        output = torch.bmm(attn, v)

        return output, attn