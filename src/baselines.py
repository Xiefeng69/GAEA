import torch
import torch.nn as nn
import torch.nn.functional as F

from src.layers import DoubleEmbedding, GraphConvLayer, SpGraphAttentionLayer

class GCN(nn.Module):
    def __init__(self, num_sr, num_tg, adj_sr, adj_tg, embedding_dim, dropout=0.2, layer=2) -> None:
        super().__init__()
        self.num_sr = num_sr
        self.num_tg = num_tg
        self.adj_sr = adj_sr
        self.adj_tg = adj_tg
        self.embedding_dim = embedding_dim
        self.layer = layer
        self.dropout = dropout
        
        self.entity_embedding = DoubleEmbedding(num_sr=self.num_sr, num_tg=self.num_tg, embedding_dim=embedding_dim, init_type="xavier")
        self.gcnblocks = nn.ModuleList([GraphConvLayer(in_features=self.embedding_dim, out_features=self.embedding_dim, dropout=self.dropout, act=False, transform=False) for i in range(self.layer)])
    
    def forward(self, **args):
        sr_embedding, tg_embedding = self.entity_embedding.weight
        for layer in self.gcnblocks:
            sr_embedding = layer(sr_embedding, self.adj_sr)
            tg_embedding = layer(tg_embedding, self.adj_tg)
        sr_embedding = F.normalize(sr_embedding, dim=1, p=2)
        tg_embedding = F.normalize(tg_embedding, dim=1, p=2)
        return sr_embedding, tg_embedding

class GAT(nn.Module):
    def __init__(self, num_sr, num_tg, adj_sr, adj_tg, embedding_dim, dropout=0.2, layer=2) -> None:
        super().__init__()
        self.num_sr = num_sr
        self.num_tg = num_tg
        self.adj_sr = adj_sr
        self.adj_tg = adj_tg
        self.embedding_dim = embedding_dim
        self.layer = layer
        self.dropout = dropout

        self.entity_embedding = DoubleEmbedding(num_sr=self.num_sr, num_tg=self.num_tg, embedding_dim=embedding_dim, init_type="xavier")
        self.gatblocks = nn.ModuleList([SpGraphAttentionLayer(in_features=self.embedding_dim, out_features=self.embedding_dim, dropout=self.dropout, act=True, transform=True) for i in range(self.layer)])
    
    def forward(self, **args):
        sr_embedding, tg_embedding = self.entity_embedding.weight
        for layer in self.gatblocks:
            sr_embedding = layer(sr_embedding, self.adj_sr)
            tg_embedding = layer(tg_embedding, self.adj_tg)
        return sr_embedding, tg_embedding

class MTransE(nn.Module):
    def __init__(self, num_sr, num_tg, rel_num, embedding_dim, L1_flag=True) -> None:
        super().__init__()
        self.num_sr = num_sr
        self.num_tg = num_tg
        self.rel_num = rel_num
        self.embedding_dim = embedding_dim
        self.L1_flag = L1_flag

        self.sr_embedding = nn.Embedding(self.num_sr, self.embedding_dim)
        self.tg_embedding = nn.Embedding(self.num_tg, self.embedding_dim)
        self.rel_embedding = nn.Embedding(self.rel_num, self.embedding_dim)
        self.transformation = nn.Linear(self.embedding_dim, self.embedding_dim)
    
    def forward(self, **args):
        return self.sr_embedding.weight, self.tg_embedding.weight

    def km_loss(self, head_ids, rel_ids, tail_ids, type):
        '''Knowledge Model'''
        #print(len(head_ids), len(rel_ids), len(tail_ids))
        if type == "sr":
            head_embedding = self.sr_embedding.weight[head_ids]
            rel_embedding = self.rel_embedding.weight[rel_ids]
            tail_embedding = self.sr_embedding.weight[tail_ids]
        elif type == "tg":
            head_embedding = self.tg_embedding.weight[head_ids]
            rel_embedding = self.rel_embedding.weight[rel_ids]
            tail_embedding = self.tg_embedding.weight[tail_ids]
        if self.L1_flag:
            loss = torch.sum(torch.abs(head_embedding + rel_embedding - tail_embedding), 1)
            loss = torch.sum(loss)
        else:
            loss = torch.sum((head_embedding + rel_embedding - tail_embedding) ** 2, 1)
            loss = torch.sum(loss)
        return loss
    
    def am_loss(self, a1_align, a2_align):
        '''Alignment Model'''
        sr_embedding = self.sr_embedding.weight[a1_align]
        tg_embedding = self.tg_embedding.weight[a2_align]

        sr_embedding = self.transformation(sr_embedding)
        loss = torch.sum(torch.abs(sr_embedding - tg_embedding), 1)
        loss = torch.sum(loss)

        return loss

class GCN_Align(nn.Module):
    def __init__(self, num_sr, num_tg, adj_sr, adj_tg, attr_num1, attr_num2, attr_weight_sr, attr_weight_tg, embedding_dim=1000, embedding_dim_attr=100, dropout=0.2, layer=2) -> None:
        super().__init__()
        self.num_sr = num_sr
        self.num_tg = num_tg
        self.attr_num1 = attr_num1
        self.attr_num2 = attr_num2
        self.adj_sr = adj_sr
        self.adj_tg = adj_tg
        self.layer = layer
        self.dropout = dropout
        self.embedding_dim = embedding_dim # for structure
        self.embedding_dim_attr = embedding_dim_attr # for attribute

        self.attr_weight_sr = attr_weight_sr # fixed
        self.attr_weight_tg = attr_weight_tg # fixed
        self.entity_embedding = DoubleEmbedding(num_sr=self.num_sr, num_tg=self.num_tg, embedding_dim=embedding_dim, init_type="xavier")
        self.gcnblocks_s = nn.ModuleList([GraphConvLayer(in_features=self.embedding_dim, out_features=self.embedding_dim, dropout=self.dropout) for i in range(self.layer)])
        self.gcnblocks_a_11 = GraphConvLayer(in_features=self.attr_num1, out_features=self.embedding_dim_attr)
        self.gcnblocks_a_12 = GraphConvLayer(in_features=self.attr_num2, out_features=self.embedding_dim_attr)
        self.gcnblocks_a_2 = GraphConvLayer(in_features=self.embedding_dim_attr, out_features=self.embedding_dim_attr)

    def forward(self, **args):
        sr_embedding_s, tg_embedding_s = self.entity_embedding.weight
        for layer in self.gcnblocks_s:
            sr_embedding_s = layer(sr_embedding_s, self.adj_sr)
            tg_embedding_s = layer(tg_embedding_s, self.adj_tg)
        sr_embedding_s = F.normalize(sr_embedding_s, dim=1, p=2)
        tg_embedding_s = F.normalize(tg_embedding_s, dim=1, p=2)
        
        sr_embedding_a = self.gcnblocks_a_11(self.attr_weight_sr, self.adj_sr)
        sr_embedding_a = self.gcnblocks_a_2(sr_embedding_a, self.adj_sr)
        tg_embedding_a = self.gcnblocks_a_12(self.attr_weight_tg, self.adj_tg)
        tg_embedding_a = self.gcnblocks_a_2(tg_embedding_a, self.adj_tg)

        # sr_embedding = torch.cat([sr_embedding_s, sr_embedding_a], dim=-1)
        # tg_embedding = torch.cat([tg_embedding_s, tg_embedding_a], dim=-1)

        return sr_embedding_s, tg_embedding_s, sr_embedding_a, tg_embedding_a