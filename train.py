import os
import sys
import math
import time
import torch
import numpy as np
from torch.utils.data import DataLoader
import argparse
import random
import datetime
np.set_printoptions(suppress=True)
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

from src.loss import *
from src.dataset import KGData, Dataset
from src.utils import *
from src.baselines import *
from src.model import GAEA

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\n- current device is {device}\n")

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=42, help="random seed")
parser.add_argument("--epoch", type=int, default=1000, help="epoch to run")
parser.add_argument("--model", type=str, default='gaea', help="the model's name", choices=['gaea', 'gat', 'gcn'])
parser.add_argument("--task", type=str, default='en_fr_15k', help="the alignment task name", choices=['zh_en', 'fr_en', 'ja_en', 'en_fr_15k', 'en_de_15k', 'd_w_15k', 'd_y_15k', 'en_fr_100k', 'en_de_100k', 'd_w_100k', 'd_y_100k'])
parser.add_argument("--fold", type=int, default=1, help="the fold cross-validation")
parser.add_argument("--train_ratio", type=int, default=3, help="training set ratio")
parser.add_argument("--val", action="store_true", default=True, help="need validation?")
parser.add_argument("--val_start", type=int, default=50, help="when to start validation")
parser.add_argument('--val_iter', type=int, default=10, help='If val, whats the validation step?')
parser.add_argument("--il", action="store_true", default=False, help="Iterative learning?")
parser.add_argument("--il_start", type=int, default=200, help="If Il, when to start?")
parser.add_argument("--il_iter", type=int, default=100, help="If IL, what's the update step?")
parser.add_argument('--lr', type=float, default=0.001, help="learning rate")
parser.add_argument('--k', type=float, default=5, help="hit@k")
parser.add_argument("--eval_metric", type=str, default="euclidean", help="the distance metric of entity pairs", choices=["cityblock", "euclidean", "csls", "inner"])
parser.add_argument('--neg_samples_size', type=int, default=5, help="number of negative samples")
parser.add_argument('--neg_iter', type=int, default=10, help="re-calculate epoch of negative samples")
parser.add_argument("--neg_metric", type=str, default="inner", choices=["cityblock", "euclidean", "csls", "inner"]) # same as BootEA
parser.add_argument('--weight_decay', type=float, default=1e-5, help="weight decay coefficient")
parser.add_argument("--csls", action="store_true", default=False, help="use CSLS for inference?")
parser.add_argument("--csls_k", type=int, default=10, help="top k for csls")
parser.add_argument("--train_batch_size", type=int, default=-1, help="train batch_size (-1 means all in)")
parser.add_argument('--patience', type=int, default=5, help='patience default 5')
parser.add_argument('--optim', type=str, default='adam', help='the optimizer')
parser.add_argument('--truncated_epsilon', type=float, default=0.9, help='the epsilon of truncated negative sampling')
parser.add_argument('--loss_fn', type=str, default="margin_based", choices=["limit_based", "margin_based"], help="the selected loss function")
parser.add_argument('--loss_norm', type=str, default='l2', help='the distance metric of loss function')
parser.add_argument('--pos_margin', type=float, default=0.01, help="positive margin in limit-based loss function")
parser.add_argument('--neg_margin', type=float, default=1.0, help='negative margin for loss computation')
parser.add_argument('--neg_param', type=float, default=0.2, help="the neg_margin_balance")
parser.add_argument('--save', action="store_true", default=False)
parser.add_argument('--result', action="store_true", default=True)

parser.add_argument('--init_type', default="xavier", choices=["xavier", "normal"], type=str)
parser.add_argument('--direct', action="store_true", default=True, help="use the diretions of relations?")
parser.add_argument('--res', action="store_true", default=False, help="use residual link?")
parser.add_argument('--ent_dim', type=int, default=256, help="hidden dimension of entity embeddings")
parser.add_argument('--rel_dim', type=int, default=128, help="hidden dimension of relation embeddings")
parser.add_argument('--dropout', type=float, default=0.2, help="dropout rate")
parser.add_argument('--layer', type=int, default=2, help="layer number of GNN-based encoder")
parser.add_argument('--n_head', type=int, default=1, help="number of multi-head attention")
parser.add_argument('--pr', type=float, default=0.1, help='the edge drop rate')
parser.add_argument('--aug_balance', type=float, default=100, help='the hyper-parameter of consistency loss')
parser.add_argument('--aug_iter', type=int, default=10)

args = parser.parse_args()
set_random_seed(args.seed)
if args.save:
    save_args(args=args)

#STEP: load data
kgdata = KGData(model=args.model, task=args.task, device=device, neg_samples_size=args.neg_samples_size, fold=args.fold, train_ratio=args.train_ratio*0.1, val=args.val, direct=args.direct)
kgdata.data_summary()
if args.train_batch_size == -1:
    batchsize = kgdata.train_pair_size
else:
    batchsize = args.train_batch_size
train_dataset = Dataset(np.array(kgdata.mapped_train_pair))
train_loader = DataLoader(dataset=train_dataset, batch_size=batchsize, shuffle=False)

#STEP: model initialization
print('[model initializing...]\n')
'''model selection'''
if args.model == "gcn":
    model = GCN(num_sr=kgdata.kg1_ent_num, num_tg=kgdata.kg2_ent_num, adj_sr=kgdata.tensor_adj1, adj_tg=kgdata.tensor_adj2, embedding_dim=args.ent_dim, dropout=args.dropout, layer=args.layer)
elif args.model == "gat":
    model = GAT(num_sr=kgdata.kg1_ent_num, num_tg=kgdata.kg2_ent_num, adj_sr=kgdata.tensor_adj1, adj_tg=kgdata.tensor_adj2, embedding_dim=args.ent_dim, dropout=args.dropout, layer=args.layer)
elif args.model == "gaea":
    model = GAEA(num_sr=kgdata.kg1_ent_num, num_tg=kgdata.kg2_ent_num, adj_sr=kgdata.tensor_adj1, adj_tg=kgdata.tensor_adj2, rel_num=kgdata.rel_num, rel_adj_sr=kgdata.tensor_rel_adj1, rel_adj_tg=kgdata.tensor_rel_adj2, args=args)
model = model.to(device=device)
pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

'''optimizer selection'''
if args.optim == "adam":
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.weight_decay)
elif args.optim == "sgd":
    optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.weight_decay)
else:
    optimizer = torch.optim.Adagrad(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.weight_decay)

'''loss function selection'''
if args.loss_fn == "limit_based":
    loss_fn = limited_based_loss
elif args.loss_fn == "margin_based":
    loss_fn = margin_based_loss

print("---------------model summary---------------\n")
print(f'#batch size: {batchsize}\n')
print(f'#total params: {pytorch_total_params}\n')
print("#number of parameter: %.2fM \n" % (pytorch_total_params/1e6))
print(f"model architecture:\n {model}")
print("-------------------------------------------\n")

def evaluate(test_pair, k, eval_metric, phase="test", model_path=None):
    if phase == "test" and args.val and args.save:
        with open(model_path, "rb") as f:
            model.load_state_dict(torch.load(f))
    
    model.eval()

    sr_embedding, tg_embedding = model(phase="eval")
    sr_embedding = sr_embedding.detach().cpu().numpy() # Returns a new Tensor, detached from the current graph. The result will never require gradient. Before transform to numpy need transfer to cpu firstly.
    tg_embedding = tg_embedding.detach().cpu().numpy()

    Lvec = np.array([sr_embedding[e1] for e1, _ in test_pair])
    Rvec = np.array([tg_embedding[e2] for _, e2 in test_pair])
    del sr_embedding, tg_embedding

    if eval_metric == "euclidean":
        hit_1_score, hit_k_score, mrr, mean = cal_metrics_faiss(Lvec, Rvec, test_pair, k=k)
        del Lvec, Rvec;
    else:
        '''step 1: generate sim mat'''
        similarity_matrix = cal_distance(Lvec, Rvec, eval_metric, csls_k=args.csls_k) # Note that the ground truth alignment relation is on the diagonal of similarity matrix
        '''step 2: calculate the hit@1, hit@k, and MRR'''
        hit_1_score, hit_k_score, mrr, mean = cal_metrics(similarity_matrix, Lvec, Rvec, test_pair, k=k)
        del similarity_matrix, Lvec, Rvec;
    
    return hit_1_score, hit_k_score, mrr, mean

#STEP: begin training
try:
    print("[start training...]\n")
    best_val = 0.0
    bad_count = 0
    t_start = time.time()
    t_total_start = time.time()
    model_path = f"save/{args.model}_{round(t_total_start, 3)}"
    sr_embedding, tg_embedding = None, None
    '''generate negative samples'''
    neg1_left, neg1_right, neg2_left, neg2_right = kgdata.generate_neg_sample(neg_samples_size=args.neg_samples_size)
    '''generate augmented KG graphs'''
    if args.pr != 0:
        pr1, pr2 = random.uniform(0, args.pr), random.uniform(0, args.pr)
        aug_adj1, aug_rel_adj1 = kgdata.generate_aug_graph(kgdata.triples1, kgdata.kg1_ent_num, kgdata.rel_num, kgdata.kg1_ent_ids, kgdata.ent2node1, kgdata.d_v1, pr=pr1)
        aug_adj2, aug_rel_adj2 = kgdata.generate_aug_graph(kgdata.triples2, kgdata.kg2_ent_num, kgdata.rel_num, kgdata.kg2_ent_ids, kgdata.ent2node2, kgdata.d_v2, pr=pr2)

    for e in range(args.epoch):
        model.train()
        '''model training'''
        for _, data in enumerate(train_loader):
            optimizer.zero_grad()
            a1_align, a2_align = data
            if args.model == "gaea":
                if args.pr != 0:
                    sr_embedding, tg_embedding = model(phase="norm")
                    aug_sr_embedding, aug_tg_embedding = model(aug_adj1, aug_rel_adj1, aug_adj2, aug_rel_adj2, phase="augment")
                    '''alignment loss'''
                    loss = loss_fn(aug_sr_embedding, aug_tg_embedding, a1_align, a2_align, neg1_left, neg1_right, neg2_left, neg2_right, neg_samples_size=args.neg_samples_size, loss_norm=args.loss_norm, pos_margin=args.pos_margin, neg_margin=args.neg_margin, neg_param=args.neg_param)
                else:
                    sr_embedding, tg_embedding = model(phase="norm")
                    loss = loss_fn(sr_embedding, tg_embedding, a1_align, a2_align, neg1_left, neg1_right, neg2_left, neg2_right, neg_samples_size=args.neg_samples_size, loss_norm=args.loss_norm, pos_margin=args.pos_margin, neg_margin=args.neg_margin, neg_param=args.neg_param)
                '''contrastive loss'''
                if args.pr != 0 and args.aug_balance != 0:
                    aug_loss1 = model.contrastive_loss(sr_embedding, aug_sr_embedding, kgdata.kg1_ent_num)
                    aug_loss2 = model.contrastive_loss(tg_embedding, aug_tg_embedding, kgdata.kg2_ent_num)
                    loss = loss + args.aug_balance * (aug_loss1 + aug_loss2)
            elif args.model in ["gcn", "gat"]:
                sr_embedding, tg_embedding = model(input1=kgdata.tensor_adj1, input2=kgdata.tensor_adj2)
                loss = loss_fn(sr_embedding, tg_embedding, a1_align, a2_align, neg1_left, neg1_right, neg2_left, neg2_right, neg_samples_size=args.neg_samples_size, loss_norm=args.loss_norm, pos_margin=args.pos_margin, neg_margin=args.neg_margin, neg_param=args.neg_param)
            
            '''multi-loss learning'''
            loss.backward() # print([x.grad for x in optimizer.param_groups[0]['params']])
            optimizer.step()
        
        print(f"epoch: {e+1}, loss: {round(loss.item(), 3)}, time: {round((time.time()-t_start), 2)}\n")
        
        '''validation phase'''
        if args.val and (e+1) % args.val_iter == 0 and e > 0 and (e+1) >= args.val_start:
            hit_1_score, hit_k_score, mrr, mean = evaluate(test_pair=kgdata.mapped_val_pair, k=args.k, eval_metric=args.eval_metric, phase="val")
            if best_val < mrr:
                bad_count = 0 # 重置patience计数器
                best_val = mrr # 使用MRR来进行early stopping (RoadEA)
                if args.save:
                    with open(model_path, "wb") as f:
                        torch.save(model.state_dict(), f)
                print(f"[validation: epoch: {e+1}, hit@1: {round(hit_1_score, 3)}, hit@{args.k}: {round(hit_k_score, 3)}, mrr is {round(mrr, 3)}]\n")
            else:
                bad_count += 1
            if bad_count == args.patience:
                break;

        '''iterative learning'''
        if args.il and (e+1) >= args.il_start and (e+1) % args.il_iter == 0 and e > 0 and e+1 != args.epoch:
            new_train_pair = kgdata.find_potential_alignment(sr_embedding.detach().cpu().numpy(), tg_embedding.detach().cpu().numpy())
            if len(new_train_pair) != 0:
                print(f'new added aligned pair: {len(new_train_pair)}')
                new_train_pair = np.concatenate((kgdata.mapped_train_pair, new_train_pair), axis=0)
                train_dataset = Dataset(new_train_pair)
                batchsize = len(new_train_pair)
                train_loader = DataLoader(dataset=train_dataset, batch_size=batchsize, shuffle=False)
        
        '''update negative sampling'''
        if (e+1) % args.neg_iter == 0 and e > 0 and e+1 != args.epoch:
            neg1_left, neg1_right, neg2_left, neg2_right = kgdata.update_neg_sample(sr_embedding.detach().cpu().numpy(), tg_embedding.detach().cpu().numpy(), neg_samples_size=args.neg_samples_size, eval_metric=args.neg_metric, csls_k=args.csls_k, e=args.truncated_epsilon)
        
        '''update augmented knowledge graph'''
        if (e+1) % args.aug_iter == 0 and e > 0 and args.pr != 0 and e+1 != args.epoch:
            pr1, pr2 = random.uniform(0, args.pr), random.uniform(0, args.pr)
            aug_adj1, aug_rel_adj1 = kgdata.generate_aug_graph(kgdata.triples1, kgdata.kg1_ent_num, kgdata.rel_num, kgdata.kg1_ent_ids, kgdata.ent2node1, kgdata.d_v1, pr=pr1)
            aug_adj2, aug_rel_adj2 = kgdata.generate_aug_graph(kgdata.triples2, kgdata.kg2_ent_num, kgdata.rel_num, kgdata.kg2_ent_ids, kgdata.ent2node2, kgdata.d_v2, pr=pr2)
        
        t_start = time.time()
        del sr_embedding, tg_embedding

except KeyboardInterrupt:
    print('-' * 40)
    print(f'Exiting from training early, epoch {e+1}')

#STEP: begin testing
print("[evaluating...]\n")
del neg1_left, neg1_right, neg2_left, neg2_right
if args.pr != 0:
    del aug_adj1, aug_rel_adj1, aug_adj2, aug_rel_adj2
try:
    hit_1_score, hit_k_score, mrr, mean = evaluate(test_pair=kgdata.mapped_test_pair, k=args.k, eval_metric=args.eval_metric, phase="test", model_path=model_path)
    print("----------------final score----------------\n")
    print(f'+ total time consume: {datetime.timedelta(seconds=int(time.time()-t_total_start))}\n')
    print(f"+ Hit@1: {round(hit_1_score, 3)}\n")
    print(f"+ Hit@{args.k}: {round(hit_k_score, 3)}\n")
    print(f"+ MRR: {round(mrr, 3)}\n")
    print(f"+ mean rank: {round(mean, 3)}\n")
    print("-------------------------------------------\n")
    # record experimental results
    if args.result:
        print("---------------save result-----------------\n")
        with open(f'result/{args.model}_{args.task}.csv', 'a', encoding='utf-8') as file:
            file.write('\n')
            file.write(f"{args.model}, {args.fold}, {round(hit_1_score, 3)}, {round(hit_k_score, 3)}, {round(mrr, 3)}, {round(mean, 3)}, {args.ent_dim}, {args.rel_dim}, {args.lr}, {args.pr}, {args.aug_balance}, {args.eval_metric}, {args.neg_margin}")
        print("-------------------------------------------\n")
except KeyboardInterrupt:
    sys.exit()