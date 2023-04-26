import math
import torch
import torch.nn as nn
import numpy as np
import networkx as nx
import scipy
import scipy.sparse as sp
import random
import copy

from src.utils import sparse_mx_to_torch_sparse_tensor, normalize_adj, cal_distance

class KGData():

    def __init__(self, model, task, device, neg_samples_size, fold, train_ratio=0.3, val=False, direct=True, lang=[1,2], save=False):
        print('[loading KG data...]\n')
        # --- parameters ---#
        self.model = model
        self.task = task
        self.device = device
        self.neg_samples_size = neg_samples_size
        self.fold = fold # current fold
        self.lang = lang # language index
        self.val = val # has validation?
        self.direct = direct # use the direction of relation?
        self.file_dir = self.task2filedir(task)

        if "dbp15k" in self.file_dir:
            # --- load ids dict ---#
            self.ent2id, self.id2ent, [self.kg1_ent_ids, self.kg2_ent_ids] = self.load_dict(f'{self.file_dir}/0_{self.fold}/ent_ids_', file_num=2)
            self.rel2id, self.id2rel, [self.kg1_rel_ids, self.kg2_rel_ids] = self.load_dict(f'{self.file_dir}/0_{self.fold}/rel_ids_', file_num=2)

            # --- load triples ---#
            self.entity1, self.rel1, self.triples1 = self.load_triples(f"{self.file_dir}/0_{fold}/triples_{lang[0]}")
            self.entity2, self.rel2, self.triples2 = self.load_triples(f"{self.file_dir}/0_{fold}/triples_{lang[1]}")
        
        elif "OpenEA" in self.file_dir: # need to give the id manually
            ent2id1, id2ent1, rel2id1, id2rel1, self.kg1_ent_ids, self.kg1_rel_ids, self.entity1, self.rel1, self.triples1 = self.load_openea(f"{self.file_dir}/rel_triples_{lang[0]}", ent_begin_id=0, rel_begin_id=0)
            ent2id2, id2ent2, rel2id2, id2rel2, self.kg2_ent_ids, self.kg2_rel_ids, self.entity2, self.rel2, self.triples2 = self.load_openea(f"{self.file_dir}/rel_triples_{lang[1]}", ent_begin_id=len(self.kg1_ent_ids), rel_begin_id=len(self.kg1_rel_ids), rel2id=rel2id1)
            self.ent2id = {**ent2id1, **ent2id2}
            self.id2ent = {**id2ent1, **id2ent2}
            self.rel2id = {**rel2id1, **rel2id2}
            # self.id2rel = {**id2rel1, **id2rel2} this code is wrong!!!
            self.id2rel = {v:k for k,v in self.rel2id.items()}

            if save:
                with open("triples_1.txt", "w", encoding="utf-8") as f:
                    for item in self.triples1:
                        f.write(f"{item[0], item[1], item[2]}\n")
                with open("triples_2.txt", "w", encoding="utf-8") as f:
                    for item in self.triples2:
                        f.write(f"{item[0], item[1], item[2]}\n")
            
            self.triples1 = self.map_triples_openea(self.triples1, ent2id1, self.rel2id) # return form as [ent_id in kg field, rel_id in global filed, ent_id in kg field]
            self.triples2 = self.map_triples_openea(self.triples2, ent2id2, self.rel2id)
            del ent2id1, id2ent1, rel2id1, id2rel1, ent2id2, id2ent2, rel2id2, id2rel2
        
        self.ent_num = len(self.id2ent)
        self.rel_num = len(self.id2rel)
        self.kg1_ent_num = len(self.kg1_ent_ids)
        self.kg2_ent_num = len(self.kg2_ent_ids)

        # --- save current triples as txt file ---#
        if save:
            with open("mapped_triples_1.txt", "w", encoding="utf-8") as f:
                for item in self.triples1:
                    f.write(f"{item[0], item[1], item[2]}\n")
            with open("mapped_triples_2.txt", "w", encoding="utf-8") as f:
                for item in self.triples2:
                    f.write(f"{item[0], item[1], item[2]}\n")

        # --- test an example, uncommand them to show the example ---#
        # test_id = np.random.randint(low=0, high=len(self.triples1))
        # print(self.id2ent[self.triples1[test_id][0]], self.id2rel[self.triples1[test_id][1]], self.id2ent[self.triples1[test_id][2]])

        # --- load train, (val), and test pair ---#
        if "dbp15k" in self.file_dir:
            self.train_pair = self.load_alignment_pair(f"{self.file_dir}/0_{self.fold}/sup_ent_ids") # entity links for training/validation
            self.test_pair = self.load_alignment_pair(f"{self.file_dir}/0_{self.fold}/ref_ent_ids") # entity links for testing
        elif "OpenEA" in self.file_dir:
            self.train_pair = self.load_alignment_pair_openea(f"{self.file_dir}/721_5fold/{self.fold}/train_links", self.ent2id)
            self.val_pair = self.load_alignment_pair_openea(f"{self.file_dir}/721_5fold/{self.fold}/valid_links", self.ent2id)
            self.test_pair = self.load_alignment_pair_openea(f"{self.file_dir}/721_5fold/{self.fold}/test_links", self.ent2id)
        else:
            alignment_pair = self.load_alignment_pair(f"{self.file_dir}/0_{self.fold}/ill_ent_ids") # entity links encoded by ids
            np.random.shuffle(alignment_pair)
            train_pair_size = int(len(alignment_pair)*train_ratio)
            self.train_pair, self.test_pair = alignment_pair[0:train_pair_size], alignment_pair[train_pair_size:]
        if self.val and "OpenEA" not in self.file_dir:
            self.train_pair, self.val_pair = self.split_train_val_alignment_pair(self.train_pair)
        
        self.train_pair = np.array(self.train_pair)
        self.train_pair_size = len(self.train_pair)
        self.test_pair = np.array(self.test_pair)
        self.test_pair_size = len(self.test_pair)
        if self.val:
            self.val_pair = np.array(self.val_pair)
            self.val_pair_size = len(self.val_pair)

        # --- load adjacent matrix and degree vector ---#
        self.adj1, self.G1, self.ent2node1, self.node2ent1 = self.get_adj(ent_ids=self.kg1_ent_ids, triples=self.triples1)
        self.adj2, self.G2, self.ent2node2, self.node2ent2 = self.get_adj(ent_ids=self.kg2_ent_ids, triples=self.triples2)
        self.d_v1, self.d_v2 = self.get_degree_vector(self.adj1, self.adj2)

        # --- load relation matrix ---#
        self.rel_adj1 = self.get_rel_adj(self.triples1, self.ent2node1, self.kg1_ent_num, self.rel_num, self.direct) # shape as [ent_num, rel_num]
        self.rel_adj2 = self.get_rel_adj(self.triples2, self.ent2node2, self.kg2_ent_num, self.rel_num, self.direct)

        #self.ins_G_edges_idx, self.ins_G_values_idx, self.ent_rels, self.r_ij_idx = self.gen_sparse_graph_from_triples(self.triples1+self.triples2, self.ent_num, with_r=False)

        #  --- load attribute ---#
        # self.attr_adj1, self.kg1_attr_num = self.get_attribute(f'{self.file_dir}/training_attrs_1', self.kg1_ent_num, self.ent2id, self.ent2node1)
        # self.attr_adj2, self.kg2_attr_num = self.get_attribute(f'{self.file_dir}/training_attrs_2', self.kg2_ent_num, self.ent2id, self.ent2node2)

        self.new_train_pair = list()
    
    def task2filedir(self, task):
        '''map task to file direction'''
        task2filedir_dict = {
            'zh_en': 'data/dbp15k/zh_en',
            'fr_en': 'data/dbp15k/fr_en',
            'ja_en': 'data/dbp15k/ja_en',
            'en_fr_15k': 'data/OpenEA_dataset_v2.0/EN_FR_15K_V1',
            'en_de_15k': 'data/OpenEA_dataset_v2.0/EN_DE_15K_V1',
            'd_w_15k': 'data/OpenEA_dataset_v2.0/D_W_15K_V1',
            'd_y_15k': 'data/OpenEA_dataset_v2.0/D_Y_15K_V1',
            'en_fr_100k': 'data/OpenEA_dataset_v2.0/EN_FR_100K_V1',
            'en_de_100k': 'data/OpenEA_dataset_v2.0/EN_DE_100K_V1',
            'd_w_100k': 'data/OpenEA_dataset_v2.0/D_W_100K_V1',
            'd_y_100k': 'data/OpenEA_dataset_v2.0/D_Y_100K_V1',
        }
        return task2filedir_dict[task]

    def data_summary(self):
        print("--------------dataset summary--------------\n")
        print(f"current task: {self.task}, file direction: {self.file_dir}\n")
        print(f"current fold: {self.fold}\n")
        print(f"entity num: {self.ent_num}, duplicated entity num: {len(self.ent2id) - self.ent_num}\n")
        print(f"entity num of kg1: {self.kg1_ent_num}, min index: {min(self.kg1_ent_ids)}, max index: {max(self.kg1_ent_ids)}\n")
        print(f"entity num of kg2: {self.kg2_ent_num}, min index: {min(self.kg2_ent_ids)}, max index: {max(self.kg2_ent_ids)}\n")
        print(f"relation num: {self.rel_num}\n")
        print(f"triple num: {len(self.triples1+self.triples2)}\n")
        print(f"training samples: {self.train_pair_size}, test samples: {self.test_pair_size}\n")
        if self.val: print(f"validation samples: {self.val_pair_size}\n")
        print("-------------------------------------------\n")
    
    def load_openea(self, file_name, ent_begin_id, rel_begin_id, rel2id=dict()):
        '''
            Load OpenEA dataset: https://github.com/nju-websoft/OpenEA
            this function load the entity and relation information seperately
            but notice that when this is the second KG the rel2id is generated by the first KG, and is dict() when it is the first execution
            this is simply because the relations are shared between different KG, the entity is not.
        '''
        ent2id, id2ent, ent_ids = dict(), dict(), list()
        rel2id, id2rel, rel_ids = rel2id, dict(), list()
        triples, entity, rel = list(), set(), set()
        ent_id, rel_id = ent_begin_id, rel_begin_id
        with open(file_name, 'r', encoding='utf-8') as f:
            data = f.read().strip().split("\n")
            data = [i.split("\t") for i in data]
            for item in data:
                entity.add(item[0])
                entity.add(item[2])
                rel.add(item[1])
                triples.append([item[0], item[1], item[2]])
                if item[0] not in ent2id:
                    ent2id[item[0]] = ent_id
                    id2ent[ent_id] = item[0]
                    ent_ids.append(ent_id)
                    ent_id += 1
                if item[2] not in ent2id:
                    ent2id[item[2]] = ent_id
                    id2ent[ent_id] = item[2]
                    ent_ids.append(ent_id)
                    ent_id += 1
                if item[1] not in rel2id:
                    rel2id[item[1]] = rel_id
                    id2rel[rel_id] = item[1]
                    rel_ids.append(rel_id)
                    rel_id += 1
        return ent2id, id2ent, rel2id, id2rel, ent_ids, rel_ids, entity, rel, triples

    def map_triples_openea(self, triples, ent2id, rel2id):
        for triple in triples:
            triple[0], triple[1], triple[2] = ent2id[triple[0]], rel2id[triple[1]], ent2id[triple[2]]
        return triples

    def load_dict(self, data_dir, file_num):
        '''load the id mapping dictionary'''
        if file_num == 2:
            file_names = [data_dir + str(i) for i in range(1,3)]
        else:
            file_names = [data_dir]
        what2id, id2what, ids = dict(), dict(), list()
        for file_name in file_names:
            with open(file_name, 'r', encoding='utf-8') as f:
                data = f.read().strip().split("\n") # the instance in the list is formed like "32742\thttp://dbpedia.org/resource/Dae_Jang_Geum"
                data = [i.split("\t") for i in data]
                what2id = {**what2id, **dict([[i[1], int(i[0])] for i in data])}
                id2what = {**id2what, **dict([[int(i[0]), i[1]] for i in data])}
                ids.append(set([int(i[0]) for i in data]))
        return what2id, id2what, ids
    
    def load_triples(self, triple_file_name):
        '''
            Read triples file, like: data/DBP15K/fr_en/triples_1, data/DBP15K/fr_en/triples_2
            return entity, relation, and triples information
        '''
        triples = list()
        entity = set()
        rel = set()
        for line in open(triple_file_name, 'r'):
            head, relation, tail = [int(item) for item in line.split()] # the instance is formed as "head_id relation_id tail_id"
            entity.add(head)
            entity.add(tail)
            rel.add(relation)
            triples.append([head, relation, tail])
        return entity, rel, triples

    def load_alignment_pair(self, file_name):
        '''Load alignment pair information for training and tesing phase'''
        alignment_pair = list()
        for line in open(file_name, 'r'):
            e1, e2 = line.split()
            alignment_pair.append((int(e1), int(e2)))
        return alignment_pair

    def load_alignment_pair_openea(self, file_name, ent2id):
        '''Load alignment pair information for training and tesing phase'''
        alignment_pair = list()
        for line in open(file_name, 'r'):
            e1, e2 = line.split()
            alignment_pair.append((ent2id[e1], ent2id[e2]))
        return alignment_pair

    def split_train_val_alignment_pair(self, train_pair, val_ratio=0.33):
        train_size = len(self.train_pair)
        mid = math.ceil(train_size * (1-val_ratio))
        return train_pair[:mid], train_pair[mid:]

    def get_adj(self, ent_ids, triples, norm=True):
        '''
            from triples generate an adjacent matrix with shape (ent_size*ent_size)
            use scipy.coo_matrix generate a sparse matrix
        '''
        G = nx.Graph()
        ent2node = dict()
        node2ent = dict()
        index = 0
        for entid in ent_ids:
            G.add_node(entid) # the node sort is according to the .add_node() order by default.
            ent2node[int(entid)] = index
            node2ent[index] = int(entid)
            index += 1 # update the index!
        for item in triples:
            G.add_edge(item[0], item[2])
        if self.model in ['gcn', 'gat']:
            for index in ent_ids: # add self-loop
                G.add_edge(index, index)
        
        return nx.adjacency_matrix(G), G, ent2node, node2ent

    def get_degree_vector(self, adj1 ,adj2):
        rowsum1 = np.array(adj1.sum(1))
        rowsum2 = np.array(adj2.sum(1))
        return rowsum1, rowsum2
    
    def get_rel_adj(self, triples, ent2node, ent_num, rel_num, direct):
        '''
            generate relation adjacent like shape as [ent_num, rel_num]
            if use the direction of relation, we consider both in edge and out edge
        '''
        if direct:
            rel_adj_in = np.zeros(shape=(ent_num, rel_num))
            rel_adj_out = np.zeros(shape=(ent_num, rel_num))
            for triple in triples:
                head_id, relation_id, tail_id = triple[0], triple[1], triple[2]
                rel_adj_out[ent2node[head_id]][relation_id] = 1
                rel_adj_in[ent2node[tail_id]][relation_id] = 1
            return [sp.coo_matrix(rel_adj_in), sp.coo_matrix(rel_adj_out)]
        else:
            rel_adj = np.zeros(shape=(ent_num, rel_num))
            for triple in triples:
                head_id, relation_id, tail_id = triple[0], triple[1], triple[2]
                rel_adj[ent2node[head_id]][relation_id] = 1
                rel_adj[ent2node[tail_id]][relation_id] = 1
            return sp.coo_matrix(rel_adj)
    
    def gen_sparse_graph_from_triples(self, triples, ins_num, with_r=False):
        edge_dict = {}
        ent_rels = []
        for (h, r, t) in triples:
            if h != t:
                if (h, t) not in edge_dict:
                    edge_dict[(h, t)] = []
                    edge_dict[(t, h)] = []
                edge_dict[(h, t)].append(r)
                edge_dict[(t, h)].append(-r)
                ent_rels.append([h, r])
                ent_rels.append([t, r])
        if with_r:
            edges = [[h, t] for (h, t) in edge_dict for r in edge_dict[(h, t)]]
            values = [1 for (h, t) in edge_dict for r in edge_dict[(h, t)]]
            r_ij = [abs(r) for (h, t) in edge_dict for r in edge_dict[(h, t)]]
            edges = np.array(edges, dtype=np.int32)
            values = np.array(values, dtype=np.float32)
            r_ij = np.array(r_ij, dtype=np.float32)
            return edges, values, r_ij
        else:
            edges = [[h, t] for (h, t) in edge_dict]
            values = [1 for (h, t) in edge_dict]
        # add self-loop
        edges += [[e, e] for e in range(ins_num)]
        values += [1 for e in range(ins_num)]
        edges = np.array(edges, dtype=np.int32)
        values = np.array(values, dtype=np.float32)
        return edges, values, ent_rels, None
    
    def map_ent2node(self, entids, ent2node):
        '''for mapping the ent_id to the id in the adjacent matrix'''
        nodes = list()
        for i in entids:
            nodes.append(ent2node[i])
        return np.array(nodes)

    def generate_neg_sample(self, neg_samples_size):
        '''
            generate negtive samples for metrics learning
            [neg1_left, neg1_right] the type1 negative samples with changed right index
            [neg2_left, neg2_right] the type2 negative samples with changed left index
        '''
        
        t = self.train_pair_size
        # broadcast ground truth
        L = np.ones((t, neg_samples_size)) * (self.mapped_train_pair[:,0].reshape((t,1)))
        neg1_left = L.reshape((t*neg_samples_size,))

        R = np.ones((t, neg_samples_size)) * (self.mapped_train_pair[:,1].reshape((t,1)))
        neg2_right = R.reshape((t*neg_samples_size,))

        # generate random neg-counterparts: can also use np.random.choice
        neg1_right = np.random.randint(low=0, high=self.kg2_ent_num, size=(t*neg_samples_size))
        neg2_left = np.random.randint(low=0, high=self.kg1_ent_num, size=(t*neg_samples_size))

        return neg1_left, neg1_right, neg2_left, neg2_right
    
    def update_neg_sample(self, sr_embedding, tg_embedding, neg_samples_size, eval_metric, csls_k=0, e=0.9):
        '''update negative samples based on Epsilon-Truncated Uniform Negative Sampling'''
        print('[updating negative samples...]\n')

        if len(self.new_train_pair) != 0: # have iterative learning
            t = self.train_pair_size + len(self.new_train_pair)
            mapped_train_pair = np.concatenate((self.mapped_train_pair, self.new_train_pair), axis=0)
        else:
            t = self.train_pair_size
            mapped_train_pair = self.mapped_train_pair

        L = mapped_train_pair[:,0]
        sr_embedding = sr_embedding[L]
        sr_sim_mat = cal_distance(sr_embedding, sr_embedding, eval_metric, csls_k)
        L = L.reshape((t,1))
        L = np.ones((t,neg_samples_size)) * L
        neg1_left = L.reshape((t*neg_samples_size,))
        # step 1: generate similar entites
        neg2_left = self.get_nearest_neighbor(sr_sim_mat, math.ceil(self.kg1_ent_num * (1-e)))
        # step 2: sample from nearest neighbors
        tmp = []
        for item in neg2_left:
            tmp.append(random.sample(list(item), neg_samples_size))
        neg2_left = np.array(tmp)
        neg2_left = neg2_left.reshape((t*neg_samples_size,))
        del sr_sim_mat

        R = mapped_train_pair[:,1]
        tg_embedding = tg_embedding[R]
        tg_sim_mat = cal_distance(tg_embedding, tg_embedding, eval_metric, csls_k)
        R = R.reshape((t,1))
        R = np.ones((t,neg_samples_size)) * R
        neg2_right = R.reshape((t*neg_samples_size,))
        # step 1: generate similar entites
        neg1_right = self.get_nearest_neighbor(tg_sim_mat, math.ceil(self.kg1_ent_num * (1-e)))
        # step 2: sample from nearest neighbors
        tmp = []
        for item in neg1_right:
            tmp.append(random.sample(list(item), neg_samples_size))
        neg1_right = np.array(tmp)
        neg1_right = neg1_right.reshape((t*neg_samples_size,))
        del tg_sim_mat
        del mapped_train_pair

        return neg1_left, neg1_right, neg2_left, neg2_right

    def generate_aug_graph(self, triples, ent_size, rel_num, ent_ids, ent2node, d_v, pr=0.02):
        print(f"[generate augmented knowledge graph with pr={round(pr,3)}...]\n")
        aug_d_v = copy.deepcopy(d_v)
        aug_triples = copy.deepcopy(triples)
        aug_ent_ids = copy.deepcopy(ent_ids)
        del_num = math.ceil(len(triples) * pr)
        del_triples = random.sample(triples, del_num) # 随机采样三元组进行剔除

        for t in del_triples:
            if aug_d_v[ent2node[t[0]]][0] > 2 and aug_d_v[ent2node[t[2]]][0] > 2: # Try not to delete the edges of nodes with less degrees
                
                aug_triples.remove(t)
                aug_d_v[ent2node[t[0]]][0] = aug_d_v[ent2node[t[0]]][0] - 1
                aug_d_v[ent2node[t[2]]][0] = aug_d_v[ent2node[t[2]]][0] - 1
        
        aug_adj, _, aug_ent2node, _ = self.get_adj(ent_ids=aug_ent_ids, triples=aug_triples)
        rel_adj = self.get_rel_adj(aug_triples, aug_ent2node, ent_size, rel_num, self.direct)
        del aug_ent2node
        if self.direct:
            rel_adj = [sparse_mx_to_torch_sparse_tensor(rel_adj[0]).to(self.device), sparse_mx_to_torch_sparse_tensor(rel_adj[1]).to(self.device)]
        else:
            rel_adj = sparse_mx_to_torch_sparse_tensor(rel_adj).to(self.device)
        
        return sparse_mx_to_torch_sparse_tensor(normalize_adj(aug_adj)).to(self.device), rel_adj

    def generate_augment_negsamples(self):
        aug_neg_1 = np.random.randint(low=0, high=self.kg1_ent_num, size=(self.kg1_ent_num))
        aug_neg_2 = np.random.randint(low=0, high=self.kg2_ent_num, size=(self.kg2_ent_num))
        return aug_neg_1, aug_neg_2
        # train_pair_1 = self.train_pair[:,0]
        # train_pair_1 = train_pair_1.reshape(-1,1)
        # train_pair_2 = self.train_pair[:,1]
        # train_pair_2 = train_pair_2.reshape(-1,1)
        # aug_train_pair_1 = np.concatenate((train_pair_1, aug_neg_1), axis=1)
        # aug_train_pair_2 = np.concatenate((train_pair_2, aug_neg_2), axis=1)
        # return aug_train_pair_1, aug_train_pair_2
    
    def get_nearest_neighbor(self, sim_mat, samples_size):
        '''calculate nearest samples for each entity'''
        ranks = np.argsort(sim_mat, axis=1)
        candidates = ranks[:, 1:samples_size + 1]
        return candidates

    def find_potential_alignment(self, sr_embedding, tg_embedding, threshold=0):
        '''find potential alignment and add them into training data'''
        print(f"[find potential alignment for iterative learning...]")
        new_link = []
        sim_mat_left2right = scipy.spatial.distance.cdist(sr_embedding, tg_embedding, metric="cosine")
        left2right = np.argmin(sim_mat_left2right, axis=1) # form like entid_from_kg1, entid_from,_kg2 
        del sim_mat_left2right
        sim_mat_right2left = scipy.spatial.distance.cdist(tg_embedding, sr_embedding, metric="cosine")
        right2left = np.argmin(sim_mat_right2left, axis=1)
        del sim_mat_right2left
        for leftid, rightid in enumerate(left2right):
            if right2left[rightid] == leftid and leftid not in self.mapped_train_pair[:,0] and rightid not in self.mapped_train_pair[:,1]: # 双向匹配
                new_link.append((leftid, rightid))
        self.new_train_pair = np.array(new_link)
        if len(new_link) == 0:
            return None
        else:
            return np.array(new_link)

    def get_attribute(self, file_name, ent_num, ent2id, ent2node):
        '''
            get attribute matrix in GCN-Align: M_attr shape as [ent_num * attr_num]
            the attribute file is formed like: entity, property1, preperty2, ...
            each row mean one entity has xxx attributes
        '''
        count = {} # for counting the most frequent attributes.
        for line in open(file_name, 'r', encoding="utf-8"):
            items = line[:-1].split('\t')
            if items[0] not in ent2id:
                continue;
            for i in range(1, len(items)):
                if items[i] not in count:
                    count[items[i]] = 1
                else:
                    count[items[i]] += 1
        fre = [(k, count[k]) for k in sorted(count, key=count.get, reverse=True)]
        num_attrs = min(len(fre), 2000)

        attr2id = {}
        for i in range(num_attrs):
            attr2id[fre[i][0]] = i
        M_attr = {}
        for line in open(file_name, 'r', encoding="utf-8"):
            items = line[:-1].split('\t')
            if items[0] in ent2id:
                item_mapped_index = ent2node[ent2id[items[0]]]
                for i in range(1, len(items)):
                    if items[i] in attr2id:
                        M_attr[(item_mapped_index, attr2id[items[i]])] = 1.0
        
        # generate coo_matrix
        row = []
        col = []
        data = []
        for key in M_attr:
            row.append(key[0])
            col.append(key[1])
            data.append(M_attr[key])
        return sp.coo_matrix((data, (row, col)), shape=(ent_num, num_attrs)),num_attrs

    @property
    def tensor_adj1(self):
        '''convert sp.adj to tensor and normalized adj'''
        return sparse_mx_to_torch_sparse_tensor(normalize_adj(self.adj1)).to(self.device)

    @property
    def tensor_adj2(self):
        '''convert sp.adj to tensor and normalized adj'''
        return sparse_mx_to_torch_sparse_tensor(normalize_adj(self.adj2)).to(self.device)

    @property
    def tensor_attr_adj1(self):
        return sparse_mx_to_torch_sparse_tensor(self.attr_adj1).to(self.device)
    
    @property
    def tensor_attr_adj2(self):
        return sparse_mx_to_torch_sparse_tensor(self.attr_adj2).to(self.device)

    @property
    def tensor_rel_adj1(self):
        if self.direct:
            return [sparse_mx_to_torch_sparse_tensor(self.rel_adj1[0]).to(self.device), sparse_mx_to_torch_sparse_tensor(self.rel_adj1[1]).to(self.device)]
        else:
            return sparse_mx_to_torch_sparse_tensor(self.rel_adj1).to(self.device)
    
    @property
    def tensor_rel_adj2(self):
        if self.direct:
            return [sparse_mx_to_torch_sparse_tensor(self.rel_adj2[0]).to(self.device), sparse_mx_to_torch_sparse_tensor(self.rel_adj2[1]).to(self.device)]
        else:
            return sparse_mx_to_torch_sparse_tensor(self.rel_adj2).to(self.device)

    @property
    def mapped_train_pair(self):
        '''map the index in kg to the adjacent matrix'''
        mapped_result = list()
        for item in self.train_pair:
            mapped_result.append([self.ent2node1[item[0]], self.ent2node2[item[1]]])
        return np.array(mapped_result)
    
    @property
    def mapped_val_pair(self):
        '''map the index in kg to the adjacent matrix'''
        mapped_result = list()
        for item in self.val_pair:
            mapped_result.append([self.ent2node1[item[0]], self.ent2node2[item[1]]])
        return np.array(mapped_result)
    
    @property
    def mapped_test_pair(self):
        '''map the index in kg to the adjacent matrix'''
        mapped_result = list()
        for item in self.test_pair:
            mapped_result.append([self.ent2node1[item[0]], self.ent2node2[item[1]]])
        return np.array(mapped_result)

    @property
    def mapped_triples_1(self):
        mapped_result = list()
        for item in self.triples1:
            mapped_result.append([self.ent2node1[item[0]], item[1], self.ent2node1[item[2]]])
        return np.array(mapped_result)

    @property
    def mapped_triples_2(self):
        mapped_result = list()
        for item in self.triples2:
            mapped_result.append([self.ent2node2[item[0]], item[1], self.ent2node2[item[2]]])
        return np.array(mapped_result)

class Dataset(object):
    def __init__(self, anchor):
        self.anchor = torch.from_numpy(anchor)
        self.a1anchor = self.anchor[:,0].numpy()
        self.a2anchor = self.anchor[:,1].numpy()
    def __getitem__(self, index):
        return self.a1anchor[index], self.a2anchor[index]
    def __len__(self):
        return len(self.a1anchor)