import os
import numpy as np
import pandas as pd
import json
import copy
from tqdm import tqdm
from scipy.sparse import csr_matrix
import torch
import torch.distributions as dist
from rule import *
from utils import *


class KG():

    def __init__(self, args):
        '''
            argument:
                filepath: ./data/FB15k-237
            
            return:
                entity2id, relation2id, train_triplets, valid_triplets, test_triplets
        '''
        self.args = args
        self.device = args.device
        self.filepath = args.data_dir + args.dataset
        self.exp_path = args.exp_dir + args.exp_name
        self.rule_path = args.rule_dir + args.dataset
        mkdir(self.rule_path)
        mkdir(self.exp_path)

        print("load data from {}".format(self.filepath))

        with open(os.path.join(self.filepath, 'entities.dict')) as f:
            self.entity2id = dict()
            self.entities = dict()

            for line in f:
                eid, entity = line.strip().split('\t')
                self.entity2id[entity] = int(eid)
                self.entities[int(eid)] = entity

        with open(os.path.join(self.filepath, 'relations.dict')) as f:
            self.relation2id = dict()
            self.relations = dict()

            for line in f:
                rid, relation = line.strip().split('\t')
                self.relation2id[relation] = int(rid)
                self.relations[int(rid)] = relation

        self.num_entity = len(self.entity2id)
        self.num_relation = len(self.relation2id)
        self.relation_id2inv_id = lambda id: (id + self.num_relation) % (2*self.num_relation)
            #id+self.num_relation if id < self.num_relation else id-self.num_relation
        
        relation2id_inv = dict()
        for relation in self.relation2id:
            inv_relation = relation + '(inv)'
            relation2id_inv[inv_relation] = self.relation_id2inv_id(self.relation2id[relation])
        self.relation2id.update(relation2id_inv)
        
        self.train_triplets = self.read_triplets(os.path.join(self.filepath, 'train.txt'), add_inv=False)
        self.valid_triplets = self.read_triplets(os.path.join(self.filepath, 'valid.txt'))
        self.test_triplets = self.read_triplets(os.path.join(self.filepath, 'test.txt'))

        print("indexing knowledge graph......")
        # self.e2triple, self.r2triple, self.r2t, self.r2ht, self.hr2t, self.hr2ht, self.rt2ht, self.hr2triple\
        #     = self.build_mapping(self.train_triplets)
        
        # 1*E  E*E
        self.adj = construct_adj(self.train_triplets, self.num_entity, self.num_relation).cuda(self.device)
        self.r2t_mat = torch.cat((torch.sparse.sum(self.adj, dim=1).to_dense().T > 0, 
                                    torch.sparse.sum(self.adj, dim=2).to_dense().T > 0), dim=1).type(torch.float)

        if self.args.type_info:
            self.type_mat = self.read_type_info(os.path.join(self.filepath, 'entity_type.txt')).to_dense().cuda(self.device)

        print("finish indexing")

        print('num_entity: {}'.format(len(self.entity2id)))
        print('num_relation: {}'.format(len(self.relation2id)))
        print('num_train_triples: {}'.format(len(self.train_triplets)))
        print('num_valid_triples: {}'.format(len(self.valid_triplets)))
        print('num_test_triples: {}'.format(len(self.test_triplets)))


    def sparse_encode(self, entity_ls):
        indices = torch.tensor(entity_ls)
        values = torch.ones_like(indices)*1.0
        indices = indices.view(1,-1)
        zeros = torch.zeros_like(indices)
        indices = torch.cat([zeros, indices], dim=0)
        vec = torch.sparse.FloatTensor(indices, values, (1, self.num_entity)).cuda(self.device)
        return vec

    def get_adj(self, r):
        if r < self.num_relation:
            #return self.adj[r].to_dense()
            return self.adj[r]
        else:
            #return self.adj[self.relation_id2inv_id(r)].to_dense().T
            return self.adj[self.relation_id2inv_id(r)].transpose(0,1)

    def read_triplets(self, filepath, add_inv=False):
        triplets = []

        with open(filepath) as f:
            for line in f:
                h, r, t = line.strip().split('\t')
                if h in self.entity2id.keys() and r in self.relation2id.keys() and t in self.entity2id.keys():
                    triplets.append([self.entity2id[h], self.relation2id[r], self.entity2id[t]])
                    if add_inv:
                        triplets.append([self.entity2id[t], self.relation_id2inv_id(self.relation2id[r]), self.entity2id[h]])

        return triplets

    def read_type_info(self, filepath):
        with open(filepath) as f:
            lines = f.readlines()
            type_ls = []
            entity_ls = []
            for line in lines:
                type_idx, entities = line.strip().split('\t')
                type_idx = int(type_idx)
                entities = list(map(int, entities.split(' ')))
                type_ls.extend([type_idx]*len(entities))
                entity_ls.extend(entities)
            
            indices = torch.tensor([type_ls, entity_ls])
            values = torch.ones(len(type_ls))
            type_mat = torch.sparse.FloatTensor(indices, values, [len(lines), self.num_entity]).transpose(0,1)
        
        return type_mat

    def build_mapping(self, dataset):
        inv_data = [[t,self.relation_id2inv_id(r),h] for h,r,t in dataset]
        dataset = dataset + inv_data
        
        e2triple = dict([[e,[]] for e in self.entity2id.values()])
        r2triple = dict([[r,[]] for r in self.relation2id.values()])
        r2t = dict([[r,[]] for r in self.relation2id.values()])
        r2ht = dict([[r,[]] for r in self.relation2id.values()])
        hr2t = {}
        hr2ht = {}
        rt2ht = {}
        hr2triple = {}


        for t in tqdm(dataset):
            e2triple[t[0]].append(t)
            r2triple[t[1]].append(t)
            r2t[t[1]].append(t[2])
            r2ht[t[1]].append([t[0], t[2]])
            
            dict_append(hr2t, (t[0], t[1]), t[2])
            dict_append(hr2ht, (t[0], t[1]), [t[0], t[2]])
            dict_append(rt2ht, (t[1], t[2]), [t[0], t[2]])
            dict_append(hr2triple, (t[0], t[1]), t)

        for r in r2triple:
            #self.r2triple[r] = list(set(self.r2triple[r]))
            r2t[r] = list(set(r2t[r]))
        for hr in hr2t:
            hr2t[hr] = list(set(hr2t[hr]))

        return e2triple, r2triple, r2t, r2ht, hr2t, hr2ht, rt2ht, hr2triple

    def get_subgraph(self, r):
        all_r_triples = self.r2triple[r]
        ht_dic = dict([[t[0], []] for t in all_r_triples])
        for t in all_r_triples:
            ht_dic[t[0]].append(t[2])
        return all_r_triples, ht_dic

    def mine_cp_rule_BFS(self, r, n):
        all_r_triples = self.r2triple[r]
        ht_dic = dict([[t[0], []] for t in all_r_triples])
        for t in all_r_triples:
            ht_dic[t[0]].append(t[2])
        
        path = self.n_hop_path([[t[0]] for t in all_r_triples], n)
        
        cp_rule = []
        for p in path:
            if p[-1] in ht_dic[p[0]]:
                cp_rule.append(p[1::2])
        uni_cp_rule = []
        [uni_cp_rule.append(rule) for rule in cp_rule if rule not in uni_cp_rule]

        #hc = [cp_rule.count(rule)/len(all_r_triples) for rule in uni_cp_rule] # use dict
        #rule_path = [self.apply_rule(rule) for rule in uni_cp_rule]
        #sc = [sum([(p[-1] in ht_dic[p[0]]) if p[0] in ht_dic else 0 for p in path])/len(path) for path in rule_path]

        return uni_cp_rule

    def mine_cp_rule_BBFS(self, r, n):
        all_r_triples, ht_dic = self.get_subgraph(r)
        h2t_n = n//2
        t2h_n = n - h2t_n

        #print("--------------mine path------------------")
        path = []
        for t in tqdm(all_r_triples):
            path_h2t = self.n_hop_path([[t[0]]], h2t_n)
            path_t2h = self.n_hop_path([[t[2]]], t2h_n, reverse=True)
            p = self.merge_path(path_h2t, path_t2h)
            path.extend(p)
        
        #print("--------------extract rules------------------")
        cp_rule = []
        for p in path:
            if p[-1] in ht_dic[p[0]]:
                cp_rule.append(p[1::2])
        uni_cp_rule = []
        [uni_cp_rule.append(rule) for rule in cp_rule if rule not in uni_cp_rule]

        #print("--------------calc head coverage------------------")
        #hc = [cp_rule.count(rule)/len(all_r_triples) for rule in tqdm(uni_cp_rule)] # use dict

        #print("--------------apply rules------------------")
        #rule_path = [rule.apply(self) for rule in tqdm(uni_cp_rule)]
        
        #print("--------------calc standard confidence------------------")
        #sc = [sum([(p[-1] in ht_dic[p[0]]) if p[0] in ht_dic else 0 for p in path])/len(path) for path in tqdm(rule_path)]

        return uni_cp_rule

    def n_hop_path(self, h, n, reverse=False):
        path = h
        for i in range(n):
            new_path = []
            for p in path:
                h = p[-1]
                ts = self.e2triple[h]
                for t in ts:
                    r = self.relation_id2inv_id(t[1]) if reverse else t[1]
                    new_p = p + [r, t[2]]
                    new_path.append(new_p)
            path = new_path
        if reverse:
            for p in path:
                p.reverse()
        return path

    def merge_path(self, h2t, t2h):
        if not t2h:
            return []
        h2t_n = len(h2t[0])
        t2h_n = len(t2h[0])
        h2t_table = pd.DataFrame(h2t, columns=range(h2t_n))
        t2h_table = pd.DataFrame(t2h, columns=range(h2t_n-1, h2t_n+t2h_n-1))
        path_table = h2t_table.merge(t2h_table, on=h2t_n-1)
        return path_table.values.tolist()

    def calc_sc(self, rule):
        all_r_triples, ht_dic = self.get_subgraph(rule.head.predicate)
        all_path = rule.apply(self)
        sc = sum([(path[-1] in ht_dic[path[0]]) if path[0] in ht_dic else 0 for path in all_path])/len(all_path) if len(all_path) else 0
        rule.sc = sc
        return sc

    def mine_rules(self, max_rule_len=3):
        mined_rules = []
        for r in tqdm(range(self.num_relation)):
            for l in range(1, max_rule_len+1):
                rules = self.mine_cp_rule_BBFS(r, l)
                for rule in rules:
                    cp_rule = CPRule().from_list(r, rule)
                    try:
                        cp_rule.reasoning_matmul(self)
                    except:
                        print("cuda out of memory occured")
                        continue
                    mined_rules.append(cp_rule)
        return mined_rules

    def load_rules(self, filename):
        rules = []
        with open(os.path.join(self.rule_path, filename)) as f:
            rule_dict_ls = json.loads(f.read())
            rule_type = list(rule_dict_ls.keys())[0]
            '''
            if rule_type == 'cp':
                for rule_dict in rule_dict_ls[rule_type]:
                    rule = CPRule().from_dict(rule_dict)
                    rule.standardize(self)
                    rules.append(rule) 
            if rule_type == 'tree':
                for rule_dict in rule_dict_ls[rule_type]:
                    rule = TreeRule().from_dict(rule_dict)
                    rule.standardize(self)
                    rules.append(rule)
            '''
            for rule_dict in rule_dict_ls[rule_type]:
                try:
                    rule = TreeRule().from_dict(rule_dict)
                except:
                    rule = CPRule().from_dict(rule_dict)                    
                flag = rule.standardize(self)
                if flag:
                    rules.append(rule)
        return rules

    def write_rules(self, rules, filename, rule_type):
        rules_dict_ls = []
        #with open(os.path.join(self.rule_path, filename), 'w') as f:
        with open(os.path.join(self.exp_path, filename), 'w') as f:
            for rule in rules:
                rules_dict_ls.append(rule.to_dict())
            f.write(json.dumps({rule_type:rules_dict_ls}))

    def write_rules_anyburl(self, rules, filename):
        #with open(os.path.join(self.rule_path, filename), 'w') as f:
        with open(os.path.join(self.exp_path, filename), 'w') as f:
            for rule in tqdm(rules):
                f.writelines(rule.to_anyburl(self) + '\n')

    def load_rules_anyburl(self, filename):
        rules = []
        parser = RuleParser(self)
        with open(os.path.join(self.rule_path, filename)) as f:
            for line in f:
                rule = parser.parse(line, format='anyburl')
                flag = rule.standardize(self)
                if flag:
                    rules.append(rule)
        return rules
    
    def get_average_confidence(self, filename):
        conf = []
        with open(os.path.join(self.rule_path, filename)) as f:
            for line in f:
                _, _, sc, _ = line.strip().split("\t")
                conf.append(float(sc))
        return np.mean(conf)    
    
    def load_rules_neuralLP(self, filename):
        rules = []
        parser = RuleParser(self)
        with open(os.path.join(self.rule_path, filename)) as f:
            for line in f:
                rule = parser.parse(line, format='neuralLP')
                flag = rule.standardize(self)
                if flag:
                    rules.append(rule)
        return rules

    def load_rules_amie(self, filename):
        rules = []
        parser = RuleParser(self)
        with open(os.path.join(self.rule_path, filename)) as f:
            for line in f:
                if '=>' in line:
                    rule = parser.parse(line, format='amie')
                    flag = rule.standardize(self)
                    if flag:
                        rules.append(rule)
        return rules

    def test_rules_matmul_all(self, rules, valid=False):
        ts = []
        ranks = []
        triples = self.valid_triplets if valid else self.test_triplets

        relation2rule = dict([[r, []] for r in range(self.num_relation)])
        for rule in rules:
            relation2rule[rule.head.predicate].append(rule)

        test_r2ht = dict([[r, []] for r in range(self.num_relation)])
        for h, r, t in triples:
            test_r2ht[r].append((h,t))
        for r in tqdm(range(self.num_relation)):
            pred_adj_r = torch.ones([self.num_entity, self.num_entity]).cuda(self.device)
            for rule in relation2rule[r]:
                paths, answers, masks = rule.reasoning_matmul(self, learning=False)
                answers = answers.to_dense()
                pred_adj_r[masks] *= (1-rule.sc)**answers
            pred_adj_r = 1 - pred_adj_r
            rank_tensor = torch.argsort(pred_adj_r, descending=True, dim=1)
            for h, t in test_r2ht[r]:
                ts.append(t)
                ranks.append(rank_tensor[h].detach().cpu().numpy())
        eval(ts, ranks)

    def test_rules_matmul_all_max(self, rules, valid=False):
        ts = []
        ranks = []
        triples = self.valid_triplets if valid else self.test_triplets

        relation2rule = dict([[r, []] for r in range(self.num_relation)])
        for rule in rules:
            relation2rule[rule.head.predicate].append(rule)

        test_r2ht = dict([[r, []] for r in range(self.num_relation)])
        for h, r, t in triples:
            test_r2ht[r].append((h,t))
        for r in tqdm(range(self.num_relation)):
            pred_adj_r = torch.zeros([self.num_entity, self.num_entity]).cuda(self.device)
            for rule in relation2rule[r]:
                paths, answers, masks = rule.reasoning_matmul(self, learning=False)
                answers = answers.to_dense()
                pred_adj_r[masks] = torch.max(pred_adj_r[masks], rule.sc*answers)
            rank_tensor = torch.argsort(pred_adj_r, descending=True, dim=1)
            for h, t in test_r2ht[r]:
                ts.append(t)
                ranks.append(rank_tensor[h].detach().cpu().numpy())
        eval(ts, ranks)

    def test_rules_matmul(self, rules, valid=False):
        ts = []
        ranks = []
        relation2rule = dict([[r, []] for r in range(self.num_relation)])
        for rule in rules:
            relation2rule[rule.head.predicate].append(rule)

        triples = self.valid_triplets if valid else self.test_triplets
        for h, r, t in tqdm(triples):
            scores = np.ones(self.num_entity)
            rules = relation2rule[r]
            for rule in rules:
                paths, answers = rule.reasoning_matmul(self, h, learning=False)
                answers = answers.to_dense().detach().cpu().numpy()[0]
                entity_scores = (1-rule.sc)**answers
                scores *= np.array(entity_scores)
            scores = 1 - scores
            rank = np.argsort(-scores)
            ts.append(t)
            ranks.append(rank)
        eval(ts, ranks)

    def test_rules(self, rules):
        ts = []
        ranks = []
        relation2rule = dict([[r, []] for r in range(self.num_relation)])
        for rule in rules:
            relation2rule[rule.head.predicate].append(rule)
        for h, r, t in tqdm(self.test_triplets):
            score = [[0] for e in range(self.num_entity)]
            
            rules = relation2rule[r]
            for rule in rules:
                paths, answers = rule.reasoning(self, h)
                if answers:
                    for answer in answers:
                        t_pred = answer[-1]
                        score[t_pred].append(rule.sc)
            score = list(map(noisy_or, score)) # aggre func
            rank = np.argsort(-np.array(score))
            ts.append(t)
            ranks.append(rank)
        eval(ts, ranks)

    def refine_rule_matmul(self, rule:CPRule):
        refined_rule = TreeRule().from_rule(rule)
        #all_r_triples, ht_dic = self.get_subgraph(rule.head.predicate)
        paths, answers, labels_pos, labels_neg = rule.reasoning_matmul(self, learning=True)
        n = len(rule.body)
        sc = rule.sc
        for idx in range(n+1):
            #sc = 1e-10
            #idx = np.random.randint(len(rule.body)+1)
            entity_label = (1-sc)*labels_pos[idx] - sc*labels_neg[idx]
            sim = torch.matmul(entity_label, self.r2t_mat).view(1,-1).cpu().detach().numpy()
            score = torch.sum(entity_label)
            sim[0, rule.head.predicate] = -1e5
            sim[0, self.relation_id2inv_id(rule.head.predicate)] = -1e5
            if np.max(sim) > score:
                r = int(np.argmax(sim))
                var = rule.get_variables()[idx]
                aux_var = refined_rule.aux_variables.pop(0)
                branch = CPRule(head=Atom(-1,[-1,var]), body=[Atom(r,[aux_var,var])])
                refined_rule.add_branch(branch)

                mask = self.r2t_mat[:,r].T
                for j in range(idx, n+1):
                    labels_pos[j] = labels_pos[j].multiply(mask)
                    labels_neg[j] = labels_neg[j].multiply(mask)
                    if j < n:
                        mask = torch.matmul(mask, self.get_adj(rule.body[j].predicate)) > 0
                        mask = mask.type(torch.float)
        
        n_pos = int(torch.sum(labels_pos[-1]).detach().cpu().numpy())
        n_neg = int(torch.sum(labels_neg[-1]).detach().cpu().numpy())
        refined_rule.num_cover = n_pos + n_neg
        refined_rule.sc = n_pos / (n_pos+n_neg) if refined_rule.num_cover else 0
        
        return refined_rule

    def refine_rule_matmul_stoch(self, rule:CPRule, num_samples=5, num_branches=1):
        refined_rules = [TreeRule().from_rule(rule)]
        #all_r_triples, ht_dic = self.get_subgraph(rule.head.predicate)
        try:
            paths, answers, labels_pos, labels_neg = rule.reasoning(self, learning=True)
        except:
            print("error occured")
            return refined_rules
        n = len(rule.body)
        sc = rule.sc
        masked_predicates = [atom.predicate for atom in [rule.head] + rule.body]
        masked_predicates += [self.relation_id2inv_id(p) for p in masked_predicates]
        masked_predicates = list(set(masked_predicates))
        for idx in range(n+1):
            var = rule.get_variables()[idx]
            if type(var) != int:
                # heuristic 1
                entity_label = (1-sc)*labels_pos[idx] - sc*labels_neg[idx]
                score = torch.sum(entity_label)
                sim = (torch.matmul(entity_label, self.r2t_mat).view(1,-1) - score)/1#answers.shape[0]
                
                # heuristic 2
                #sim = (torch.matmul(labels_pos[idx], self.r2t_mat)/torch.matmul(labels_pos[idx]+labels_neg[idx], self.r2t_mat)).view(1,-1) - sc
                #sim[torch.isnan(sim)] = -1e5
                
                sim[0, masked_predicates] = -1e5
                sim[sim <= 0] = 0
                if torch.max(sim) > 0:
                    #probs = sim/torch.sum(sim)#torch.softmax(sim, dim=1)
                    #categorical = dist.Categorical(probs)
                    #samples = categorical.sample([num_samples])
                    #unique_samples = torch.unique(samples).tolist()
                    #unique_samples = [torch.argmax(sim).tolist()]
                    unique_samples = torch.argsort(sim, descending=True)[0].tolist()[:self.args.num_refine_sample]
                    for r in unique_samples:
                        refined_rule = TreeRule().from_rule(rule)
                        aux_var = refined_rule.aux_variables.pop(0)
                        branch = CPRule(head=Atom(-1,[-1,var]), body=[Atom(r,[aux_var,var])])
                        refined_rule.add_branch(branch)
                        
                        mask = self.r2t_mat[:,r].T
                        labels_pos_copy = copy.deepcopy(labels_pos)
                        labels_neg_copy = copy.deepcopy(labels_neg)
                        for j in range(idx, n+1):
                            labels_pos_copy[j] = labels_pos_copy[j].multiply(mask)
                            labels_neg_copy[j] = labels_neg_copy[j].multiply(mask)
                            if j < n:
                                mask = (torch.matmul(mask, self.get_adj(rule.body[j].predicate).to_dense()) > 0).type(torch.float)
                        n_pos = int(torch.sum(labels_pos_copy[-1]).detach().cpu().numpy())
                        n_neg = int(torch.sum(labels_neg_copy[-1]).detach().cpu().numpy())
                        refined_rule.num_cover = n_pos + n_neg
                        refined_rule.sc = n_pos / (n_pos+n_neg) if refined_rule.num_cover else 0
                        refined_rules.append(refined_rule)
            
        return refined_rules

    def refine_rule_matmul_addtype(self, rule:CPRule, num_samples=5, num_branches=1):
        refined_rules = [TreeRule().from_rule(rule)]
        try:
            paths, answers, labels_pos, labels_neg = rule.reasoning(self, learning=True)
        except:
            print("error occured")
            return refined_rules
        n = len(rule.body)
        sc = rule.sc
        unary_mat = torch.cat([self.r2t_mat, self.type_mat], dim=1) if self.args.type_info else self.r2t_mat
        masked_predicates = [atom.predicate for atom in [rule.head] + rule.body]
        masked_predicates += [self.relation_id2inv_id(p) for p in masked_predicates]
        masked_predicates = list(set(masked_predicates))
        for idx in range(n+1):
            var = rule.get_variables()[idx]
            if type(var) != int:
                # heuristic 1
                entity_label = (1-sc)*labels_pos[idx] - sc*labels_neg[idx]
                score = torch.sum(entity_label)
                sim = torch.matmul(entity_label, unary_mat).view(1,-1) - score
                #sim = torch.matmul(entity_label, self.type_mat).view(1,-1) - score
                #sim[0, :2*self.num_relation] = 0
                
                # heuristic 2
                #sim = (torch.matmul(labels_pos[idx], unary_mat)/torch.matmul(labels_pos[idx]+labels_neg[idx], unary_mat)).view(1,-1) - sc
                #sim[torch.isnan(sim)] = -1e5
                
                sim[0, masked_predicates] = -1e5
                sim[sim <= 0] = 0
                if torch.max(sim) > 0:
                    unique_samples = torch.argsort(sim, descending=True)[0].tolist()[:self.args.num_refine_sample]
                    for r in unique_samples:
                        refined_rule = TreeRule().from_rule(rule)
                        aux_var = refined_rule.aux_variables.pop(0) if r < 2*self.num_relation else var
                        branch = CPRule(head=Atom(-1,[-1,var]), body=[Atom(r,[aux_var,var])])
                        refined_rule.add_branch(branch)
                        
                        mask = unary_mat[:,r].T
                        labels_pos_copy = copy.deepcopy(labels_pos)
                        labels_neg_copy = copy.deepcopy(labels_neg)
                        for j in range(idx, n+1):
                            labels_pos_copy[j] = labels_pos_copy[j].multiply(mask)
                            labels_neg_copy[j] = labels_neg_copy[j].multiply(mask)
                            if j < n:
                                mask = (torch.matmul(mask, self.get_adj(rule.body[j].predicate).to_dense()) > 0).type(torch.float)
                        n_pos = int(torch.sum(labels_pos_copy[-1]).detach().cpu().numpy())
                        n_neg = int(torch.sum(labels_neg_copy[-1]).detach().cpu().numpy())
                        refined_rule.num_cover = n_pos + n_neg
                        refined_rule.sc = n_pos / (n_pos+n_neg) if refined_rule.num_cover else 0
                        refined_rules.append(refined_rule)
            
        return refined_rules

    def refine_rule_matmul_add_semi_grounded(self, rule:CPRule, num_samples=5, num_branches=1):
        refined_rules = [TreeRule().from_rule(rule)]
        try:
            paths, answers, labels_pos, labels_neg = rule.reasoning(self, learning=True)
        except:
            print("error occured")
            return refined_rules
        n = len(rule.body)
        sc = rule.sc
        unary_mat = self.r2t_mat
        if self.args.entity:
            unary_mat = torch.cat([unary_mat, torch.eye(self.num_entity).to(self.device)], dim=1)
        if self.args.type_info:
            unary_mat = torch.cat([unary_mat, self.type_mat], dim=1)
        masked_predicates = [atom.predicate for atom in [rule.head] + rule.body]
        masked_predicates += [self.relation_id2inv_id(p) for p in masked_predicates]
        masked_predicates = list(set(masked_predicates))
        for idx in range(n+1):
            var = rule.get_variables()[idx]
            if type(var) != int:
                # heuristic 1
                entity_label = (1-sc)*labels_pos[idx] - sc*labels_neg[idx]
                score = torch.sum(entity_label)
                sim = torch.matmul(entity_label, unary_mat).view(1,-1) - score
                #sim = torch.matmul(entity_label, self.type_mat).view(1,-1) - score
                #sim[0, :2*self.num_relation] = 0
                
                # heuristic 2
                #sim = (torch.matmul(labels_pos[idx], unary_mat)/torch.matmul(labels_pos[idx]+labels_neg[idx], unary_mat)).view(1,-1) - sc
                #sim[torch.isnan(sim)] = -1e5
                
                sim[0, masked_predicates] = -1e5
                sim[sim <= 0] = 0
                if torch.max(sim) > 0:
                    unique_samples = torch.argsort(sim, descending=True)[0].tolist()[:self.args.num_refine_sample]
                    for r in unique_samples:
                        refined_rule = TreeRule().from_rule(rule)
                        if r < 2*self.num_relation:
                            aux_var = refined_rule.aux_variables.pop(0) 
                        elif r < 2*self.num_relation + self.num_entity:
                            if self.args.entity:
                                aux_var = self.entities[r - 2*self.num_relation]
                            else:
                                aux_var = var
                        else:
                            aux_var = var
                        branch = CPRule(head=Atom(-1,[-1,var]), body=[Atom(r,[aux_var,var])])
                        refined_rule.add_branch(branch)
                        
                        mask = unary_mat[:,r].T
                        labels_pos_copy = copy.deepcopy(labels_pos)
                        labels_neg_copy = copy.deepcopy(labels_neg)
                        for j in range(idx, n+1):
                            labels_pos_copy[j] = labels_pos_copy[j].multiply(mask)
                            labels_neg_copy[j] = labels_neg_copy[j].multiply(mask)
                            if j < n:
                                mask = (torch.matmul(mask, self.get_adj(rule.body[j].predicate).to_dense()) > 0).type(torch.float)
                        n_pos = int(torch.sum(labels_pos_copy[-1]).detach().cpu().numpy())
                        n_neg = int(torch.sum(labels_neg_copy[-1]).detach().cpu().numpy())
                        refined_rule.num_cover = n_pos + n_neg
                        refined_rule.sc = n_pos / (n_pos+n_neg) if refined_rule.num_cover else 0
                        if refined_rule.sc > 0:
                            refined_rules.append(refined_rule)
            
        return refined_rules

    def refine_rule_matmul_sparse(self, rule:CPRule, num_samples=5, num_branches=1):
        refined_rules = [TreeRule().from_rule(rule)]
        try:
            paths, answers, labels_pos, labels_neg = rule.reasoning(self, learning=True)
        except:
            print("error occured")
            return refined_rules
        n = len(rule.body)
        sc = rule.sc
        unary_mat = self.r2t_mat.to_sparse()
        if self.args.entity:
            indices = torch.tensor(range(self.num_entity)).view(1,-1)
            values = torch.ones(indices.shape[1])
            indices = torch.cat([indices, indices], dim=0)
            size = (self.num_entity, self.num_entity)
            eye = torch.sparse.FloatTensor(indices, values, size).cuda(self.device)
            unary_mat = torch.cat([unary_mat, eye], dim=1)
        if self.args.type_info:
            unary_mat = torch.cat([unary_mat, self.type_mat], dim=1)
        masked_predicates = [atom.predicate for atom in [rule.head] + rule.body]
        masked_predicates += [self.relation_id2inv_id(p) for p in masked_predicates]
        masked_predicates = list(set(masked_predicates))
        for idx in range(n+1):
            var = rule.get_variables()[idx]
            if type(var) != int:
                # heuristic 1
                entity_label = ((1-sc)*labels_pos[idx] - sc*labels_neg[idx]).view(1,-1).to_sparse()
                score = torch.sparse.sum(entity_label)
                sim = torch.sparse.mm(entity_label, unary_mat).to_dense().view(1,-1) - score
                #sim = torch.matmul(entity_label, self.type_mat).view(1,-1) - score
                sim[0, :2*self.num_relation] = 0
                
                # heuristic 2
                #sim = (torch.matmul(labels_pos[idx], unary_mat)/torch.matmul(labels_pos[idx]+labels_neg[idx], unary_mat)).view(1,-1) - sc
                #sim[torch.isnan(sim)] = -1e5
                
                sim[0, masked_predicates] = -1e5
                sim[sim <= 0] = 0
                if torch.max(sim) > 0:
                    unique_samples = torch.argsort(sim, descending=True)[0].tolist()[:self.args.num_refine_sample]
                    for r in unique_samples:
                        refined_rule = TreeRule().from_rule(rule)
                        if r < 2*self.num_relation:
                            aux_var = refined_rule.aux_variables.pop(0) 
                        elif r < 2*self.num_relation + self.num_entity:
                            if self.args.entity:
                                aux_var = self.entities[r - 2*self.num_relation]
                            else:
                                aux_var = var
                        else:
                            aux_var = var
                        branch = CPRule(head=Atom(-1,[-1,var]), body=[Atom(r,[aux_var,var])])
                        refined_rule.add_branch(branch)
                        
                        mask = unary_mat.transpose(0,1)[r]
                        labels_pos_copy = copy.deepcopy(labels_pos).to_sparse()
                        labels_neg_copy = copy.deepcopy(labels_neg).to_sparse()
                        for j in range(idx, n+1):
                            pos = labels_pos_copy[j].mul(mask)
                            neg = labels_neg_copy[j].mul(mask)
                            if j < n:
                                mask = torch.sparse.mm(mask.to_dense().view(1,-1).to_sparse(), self.get_adj(rule.body[j].predicate))[0].coalesce()
                                mask.values().fill_(1)
                        n_pos = int(torch.sparse.sum(pos))
                        n_neg = int(torch.sparse.sum(neg))
                        refined_rule.num_cover = n_pos + n_neg
                        refined_rule.sc = n_pos / (n_pos+n_neg) if refined_rule.num_cover else 0
                        if refined_rule.sc > 0:
                            refined_rules.append(refined_rule)
            
        return refined_rules

    def refine_rules(self, rules):
        refined_rules = []
        for rule in tqdm(rules):
            #if rule.is_semi_grounded():
            refined_rule = self.refine_rule_matmul_sparse(rule)
            #else:
            #    refined_rule = [rule]
            refined_rules.extend(refined_rule)
        return refined_rules
