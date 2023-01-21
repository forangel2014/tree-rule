import numpy as np
from scipy.sparse import csr_matrix
from tqdm import tqdm
import os
import json
import copy
import torch
import random
from dataset import RuleDataset
from torch.utils.data.dataloader import DataLoader

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def get_rule_loader_split(rules, kg, max_rule_len, batchsize, valid_ratio):
    rules_copy = copy.deepcopy(rules)
    random.shuffle(rules_copy)
    n = int(valid_ratio*len(rules_copy)) + 1
    rules_train, rules_valid = rules_copy[n:], rules_copy[:n]
    train_loader = get_rule_loader(rules_train, kg, max_rule_len, batchsize)
    valid_loader = get_rule_loader(rules_valid, kg, max_rule_len, batchsize)
    return train_loader, valid_loader

def get_rule_loader(rules, kg, max_rule_len, batchsize):
    rule_dataset = RuleDataset(rules, kg, max_rule_len=max_rule_len)
    rule_loader = DataLoader(rule_dataset, batch_size=batchsize, shuffle=True)
    return rule_loader

def episilon_uniform_sampling(probs, episilon, num_samples):
    probs_fuse = (1-episilon) * probs + episilon * torch.ones_like(probs)/probs.shape[-1]
    categorical = torch.distributions.Categorical(probs_fuse)
    samples = categorical.sample([num_samples])
    return samples

def construct_adj(triplets, num_entity, num_relation):
    indices = torch.tensor(triplets)
    indices_copy = copy.deepcopy(indices)
    indices_copy[:,0], indices_copy[:,1] = indices[:,1], indices[:,0]
    values = torch.ones(indices.shape[0])
    size = (num_relation, num_entity, num_entity)
    adj = torch.sparse.FloatTensor(indices_copy.T, values, size)
    return adj

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape).cuda()

def eval(ts, ranks):
    n_eval = len(ts)
    rank = []
    for i in tqdm(range(n_eval)):
        rank.append(np.argwhere(ranks[i] == ts[i]))
    rank = np.array(rank) + 1
    mrr = np.mean(1/rank)
    hit1 = np.sum(rank == 1)/n_eval
    hit3 = np.sum(rank <= 3)/n_eval
    hit10 = np.sum(rank <= 10)/n_eval
    print('MRR hit@1 hit@3 hit@10')
    print(mrr, hit1, hit3, hit10)
    return mrr

def calc_entropy(arr):
    n = len(arr)
    if not n:
        return 1e10
    n_pos = np.sum(arr)
    p = max(n_pos/n, 1e-5)
    entropy = -p*np.log(p) - (1-p)*np.log(1-p)
    return entropy

def encode(id_ls, n):
    vec = np.zeros(n)
    for id in id_ls:
        vec[id] += 1
    return vec

def onehot(id_ls, n):
    vec = np.zeros(n)
    for id in id_ls:
        vec[id] = 1
    return vec

def noisy_or(ls):
    res = 1
    for sc in ls:
        res *= 1-sc
    return 1-res

def test_chunk(params):
    chunk, relation2rule, r2triple, hr2t, num_entity = params
    ts = []
    ranks = []
    for h, r, t in tqdm(chunk):
        score = [[0] for e in range(num_entity)]
        rules = relation2rule[r]
        for rule in rules:
            path = rule.apply(r2triple, hr2t, h)#self.apply_rule(rule, h)
            if path:
                for p in path:
                    t = rule.result(p)
                    score[t].append(rule.sc)
        score = list(map(noisy_or, score)) # aggre func
        rank = np.argsort(-np.array(score))
        ts.append(t)
        ranks.append(rank)
    return ts, ranks

def parse_sparql_res(res):
    all_path = []
    variables = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
    res = json.loads(res)
    for dic in res["results"]["bindings"]:
        path = []
        for v in variables:
            if v in dic.keys():
                path.append(int(dic[v]["value"]))
            else:
                break
        all_path.append(path)
    return all_path

def dict_append(dict, key, value):
    if key in dict.keys():
        dict[key].append(value)
    else:
        dict[key] = [value]