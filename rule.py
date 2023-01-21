import numpy as np
import pandas as pd
import json
from tqdm import tqdm
from scipy.sparse import csr_matrix
from utils import *



class Atom():

    def __init__(self, predicate=None, arguments=None):

        self.predicate = predicate
        self.arguments = arguments

    def to_dict(self):

        return {"predicate": self.predicate, "arguments": self.arguments}

    def from_dict(self, dict):

        self.predicate = dict["predicate"]
        self.arguments = dict["arguments"]
        return self

    def to_str(self, kg):
        atom = copy.deepcopy(self)
    
        if atom.predicate < 2*kg.num_relation:
            if atom.predicate >= kg.num_relation:
                atom.reverse(kg)
            arg0 = atom.arguments[0] if type(atom.arguments[0]) == str else kg.entities[atom.arguments[0]]
            arg1 = atom.arguments[1] if type(atom.arguments[1]) == str else kg.entities[atom.arguments[1]]
            return kg.relations[atom.predicate] + '({},{})'.format(arg0, arg1)

        elif atom.predicate < 2*kg.num_relation + kg.num_entity:
            if kg.args.entity:
                return 'is' + '({},{})'.format(atom.arguments[0], atom.arguments[1])
            else:
                return 'type_{}'.format(atom.predicate-2*kg.num_relation) + '({},{})'.format(atom.arguments[0], atom.arguments[1])
        else:
            return 'type_{}'.format(atom.predicate-2*kg.num_relation-kg.num_entity) + '({},{})'.format(atom.arguments[0], atom.arguments[1])


    def reverse(self, kg):

        relation_id, arguments = self.predicate, self.arguments
        self.arguments = [arguments[1], arguments[0]]
        self.predicate = kg.relation_id2inv_id(relation_id)
        return self

    def grounding(self, kg, paths, constraints=None):
        if paths:
            new_paths = []
            for path in paths:
                try:
                    ts = kg.hr2t[(path[-1], self.predicate)]
                    for t in ts:
                        if type(self.arguments[1]) == str or t == self.arguments[1]:
                            new_path = path + [t]
                            new_paths.append(new_path)
                except:
                    pass
            res = new_paths
        else:
            if type(self.arguments[0]) == int:
                res = kg.hr2ht[(self.arguments[0], self.predicate)]
            elif type(self.arguments[1]) == int:
                res = kg.rt2ht[(self.predicate, self.arguments[1])]
            else:
                res = kg.r2ht[self.predicate]
    
        if constraints:
            if self.arguments[0] in constraints.keys():
                constraint_entities = constraints[self.arguments[0]]
                res = [path for path in res if path[-2] in constraint_entities]
            if self.arguments[1] in constraints.keys():
                constraint_entities = constraints[self.arguments[1]]
                res = [path for path in res if path[-1] in constraint_entities]

        return res

class AtomParser():

    def __init__(self, kg):

        self.kg = kg

    def parse(self, atom, format='amie'):
        
        if format == 'anyburl':
            atom = atom.strip()
            predicate, arguments = atom.split('(', 1)
            predicate_id = self.kg.relation2id[predicate]
            arguments = arguments[:-1].split(',')
            arguments_id = [self.kg.entity2id[a.strip()] if a.strip() in self.kg.entity2id else a for a in arguments]
            return Atom(predicate_id, arguments_id)
        
        if format == 'amie':
            arg0, predicate_str, arg1 = atom.split('  ')
            arg0 = arg0[1:].upper().replace('A', 'X').replace('B', 'Y')
            arg1 = arg1[1:].upper().replace('A', 'X').replace('B', 'Y')
            predicate_id = self.kg.relation2id[predicate_str]
            return Atom(predicate_id, [arg0, arg1])
        
        if format == 'neuralLP':
            atom = atom.strip()
            predicate, arguments = atom.split('(')
            if predicate[:4] == 'inv_':
                predicate_id = self.kg.relation_id2inv_id(self.kg.relation2id[predicate[4:]])
            else:
                predicate_id = self.kg.relation2id[predicate]
            arguments = arguments.strip(')').split(', ')
            #arguments.reverse()
            return Atom(predicate_id, arguments)

class RuleParser():

    def __init__(self, kg):

        self.kg = kg
        self.atom_parser = AtomParser(kg)

    def parse(self, line, format='amie'):

        if format == 'amie':
            #Rule	Head Coverage	Std Confidence	PCA Confidence	Positive Examples	Body size	PCA Body size	Functional variable
            #?a  /film/film/executive_produced_by  ?b   => ?a  /film/film/edited_by  ?b	0.044692737	0.012084592	0.101265823	8	662	79	?a
            #?a  /music/genre/artists  ?h  ?h  /music/group_member/membership./music/group_membership/role  ?b   => ?a  /music/instrument/family  ?b	0.01	0.00053135	1	1	1882	1	?a
            rule_str, hc, sc, pca_conf, num_true, num_cover = line.strip().split('\t')[:6]
            sc = float(sc)
            hc = float(hc)
            body_str, head_str = rule_str.split('   => ')
            head_atom = self.atom_parser.parse(head_str, format)
            body_atom_str = body_str.split('  ')
            body_atoms = []
            for i in range(len(body_atom_str)//3):
                atom_str = '  '.join(body_atom_str[3*i:3*(i+1)])
                body_atoms.append(self.atom_parser.parse(atom_str, format))
            rule = CPRule(head_atom, body_atoms, hc, sc)
            rule.num_cover = int(num_cover)
            return rule
        
        if format == 'anyburl':
            num_cover, num_true, sc, rule_str = line.strip().split('\t')
            sc = float(sc)
            num_cover = int(num_cover)
            head_str, body_str = rule_str.split(' <= ')
            head_atom = self.atom_parser.parse(head_str, format)
            body_atom_str = body_str.split(', ')
            body_atoms = []
            for atom_str in body_atom_str:
                body_atoms.append(self.atom_parser.parse(atom_str, format))
            rule = CPRule(head_atom, body_atoms, sc=sc)
            rule.num_cover = num_cover
            return rule
        
        if format == 'neuralLP':
            schc, rule_str = line.strip().split('\t')
            sc, hc = schc.split(' ')
            sc = float(sc)
            hc = float(hc[1:-1])
            sc, hc = hc, sc
            head_str, body_str = rule_str.split(' <-- ')
            head_atom = self.atom_parser.parse(head_str, format)
            body_atom_str = body_str.split('), ')
            body_atoms = []
            for atom_str in body_atom_str:
                body_atoms.append(self.atom_parser.parse(atom_str, format))
            rule = CPRule(head_atom, body_atoms, sc=sc, hc=hc)
            if head_atom.predicate >= self.kg.num_relation:
                variable_map = {rule.head.arguments[0]: 'Y', rule.head.arguments[1]: 'X'}
            else:
                variable_map = {rule.head.arguments[0]: 'X', rule.head.arguments[1]: 'Y'}
            for atom in [rule.head] + rule.body:
                if atom.arguments[0] in variable_map.keys():
                    atom.arguments[0] = variable_map[atom.arguments[0]]
                if atom.arguments[1] in variable_map.keys():
                    atom.arguments[1] = variable_map[atom.arguments[1]]
            return rule

class Rule():

    def __init__(self, head=None, body=None, hc=None, sc=None):
        
        self.head = head
        self.body = body
        self.hc = hc
        self.sc = sc
        self.num_cover = None
        self.num_head = None
        
    def all_predicates(self, kg):
        predicates = [atom.predicate for atom in [self.head] + self.body]
        predicates += [kg.relation_id2inv_id(p) for p in predicates]
        predicates = list(set(predicates))
        predicates += [predicates[-1]]*(10-len(predicates))
        return predicates
        
    def to_dict(self):
        pass

    def from_dict(self, dict):
        pass

    def to_anyburl(self, kg):
        pass

    def apply(self, kg, h=None):
        pass

    def apply_sparql(self, gc, h=None):
        pass
    
    def is_semi_grounded(self):
        atoms = [self.head] + self.body
        for atom in atoms:
            if type(atom.arguments[0]) == int or type(atom.arguments[1]) == int:
                return True
            if len(atom.arguments[0]) > 1 or len(atom.arguments[1]) > 1:
                return True
        return False

    def eval(self, kg):
        paths, answers = self.reasoning(kg)
        num_true = sum([answer[1] in kg.hr2t[(answer[0], self.head.predicate)] for answer in answers])
        self.num_cover = len(answers)
        self.hc = num_true/len(kg.r2triple[self.head.predicate])
        self.sc = num_true/self.num_cover

class CPRule(Rule):

    def __init__(self, head=None, body=None, hc=None, sc=None):
        
        super(CPRule, self).__init__(head, body, hc, sc)

    def from_list(self, head, body):
        body_variables = ['A', 'B', 'C', 'D']
        self.head = Atom(head, ['X', 'Y'])
        body_atoms = []
        current_variable = 'X'
        for b in body:
            next_variable = body_variables.pop(0)
            body_atoms.append(Atom(b, [current_variable, next_variable]))
            current_variable = next_variable
        body_atoms[-1].arguments[1] = 'Y'
        self.body = body_atoms
        return self

    def to_dict(self):

        return {'head': self.head.to_dict(), 'body':[atom.to_dict() for atom in self.body], 'hc':self.hc, 'sc':self.sc, 'num_cover': self.num_cover, 'num_head': self.num_head}

    def from_dict(self, dict):

        self.head = Atom().from_dict(dict['head'])
        self.body = [Atom().from_dict(atom) for atom in dict['body']]
        self.hc = dict['hc']
        self.sc = dict['sc']
        self.num_cover = dict['num_cover'] if 'num_cover' in dict.keys() else None
        self.num_head = dict['num_head'] if 'num_head' in dict.keys() else None
        #self.formalize()
        return self

    def get_variables(self):
        
        ls = [self.body[0].arguments[0]]
        for atom in self.body:
            ls.append(atom.arguments[1])
        return ls

    def check(self):
        variable_cnt = {}
        for atom in [self.head] + self.body:
            for variable in atom.arguments:
                if variable in variable_cnt:
                    variable_cnt[variable] += 1
                else:
                    variable_cnt[variable] = 1
        for variable, appear_times in variable_cnt.items():
            if not appear_times == 2 and not variable in self.head.arguments and type(variable) == str:
                return False
        return True

    def standardize(self, kg):
        body_atom_ls = []
        atoms = self.body
        if type(self.head.arguments[0]) == str:
            current_variable = self.head.arguments[0]
            for _ in range(len(self.body)):
                for i in range(len(atoms)):
                    if atoms[i].arguments[0] == current_variable:
                        atom = atoms.pop(i)
                        body_atom_ls.append(atom)
                        current_variable = atom.arguments[1]
                        break
                    elif atoms[i].arguments[1] == current_variable:
                        atom = atoms.pop(i)
                        atom.reverse(kg)
                        body_atom_ls.append(atom)
                        current_variable = atom.arguments[1]
                        break
        else:
            current_variable = self.head.arguments[1]
            for _ in range(len(self.body)):
                for i in range(len(atoms)):
                    if atoms[i].arguments[1] == current_variable:
                        atom = atoms.pop(i)
                        body_atom_ls.append(atom)
                        current_variable = atom.arguments[0]
                        break
                    elif atoms[i].arguments[0] == current_variable:
                        atom = atoms.pop(i)
                        atom.reverse(kg)
                        body_atom_ls.append(atom)
                        current_variable = atom.arguments[0]
                        break
            body_atom_ls.reverse()
        
        self.body = body_atom_ls
        
        return len(self.body) > 0

    def to_anyburl(self, kg):
        head_str = self.head.to_str(kg)
        body_str_ls = [atom.to_str(kg) for atom in self.body]
        body_str = ', '.join(body_str_ls)
        rule_str = head_str + ' <= ' + body_str
        num_cover = self.num_cover
        num_correct = round(num_cover*self.sc)
        return '\t'.join([str(num_cover), str(num_correct), str(self.sc), rule_str])

    def apply(self, kg, h=None):
        if h:
            if type(self.head.arguments[0]) == int:
                if self.head.arguments[0] == h:
                    paths = self.grounding_body(kg, [[h]])
                else:
                    return {}
            else:
                paths = self.grounding_body(kg, [[h]])
        else:
            paths = self.grounding_body(kg)
        return paths

    def grounding_body(self, kg, paths=None, constraints=None):
        for atom in self.body:
            paths = atom.grounding(kg, paths, constraints)
            if not paths:
                return paths
        return paths

    def reasoning_list(self, kg, h=None):
        
        paths = self.apply(kg, h)
        if type(self.head.arguments[1]) == int:
            answers = [(path[0], self.head.arguments[1]) for path in paths]            
        elif type(self.head.arguments[0]) == int:
            answers = [(self.head.arguments[0], path[-1]) for path in paths]
        else:
            answers = [(path[0], path[-1]) for path in paths]
        return paths, answers

    def forward(self, kg, vec, constraints=None):
        paths = [vec]
        for atom in self.body:
            if constraints:
                if atom.arguments[0] in constraints.keys():
                    constraint_vec = constraints[atom.arguments[0]]
                    vec = (vec.to_dense().multiply(constraint_vec.to_dense())).to_sparse()

            adj = kg.get_adj(atom.predicate)
            #vec = torch.matmul(vec, adj)
            vec = torch.sparse.mm(vec, adj)

            #vec = vec.coalesce()
            #vec.values().fill_(1)

            paths.append(vec)
        answers = paths[-1]
        return paths, answers

    def reasoning(self, kg, h=None, learning=True, constraints=None):
        if self.is_semi_grounded():
            return self.reasoning_matmul_semi_grounded(kg, h, learning, constraints)
        else:
            return self.reasoning_matmul(kg, h, learning, constraints)

    def reasoning_matmul(self, kg, h=None, learning=True, constraints=None):
        
        if h is not None:
            vec = kg.sparse_encode([h])
            return self.forward(kg, vec)
        else:
            vec = torch.sparse.sum(kg.get_adj(self.body[0].predicate), dim=1)
            indices = vec.indices()
            n = indices.shape[1]
            values = torch.ones_like(indices)[0]*1.0
            indices_expand = torch.cat([torch.arange(n).cuda(kg.device).view_as(indices), indices], dim=0)
            vec = torch.sparse.FloatTensor(indices_expand, values, torch.Size([n, kg.num_entity]))
            paths, answers = self.forward(kg, vec, constraints)

            if not learning:
                return paths, answers, indices[0]

            else:
                ground_truth = torch.sparse.mm(vec, kg.get_adj(self.head.predicate))#.to_dense()
                
                #ground_truth_pos = (ground_truth > 0).type(torch.float)#.to_sparse()
                #ground_truth_neg = (ground_truth == 0).type(torch.float)#.to_sparse()
                ground_truth_pos = ground_truth.coalesce()
                ground_truth_pos.values().fill_(1)
                
                #answers = (answers.to_dense() > 0).type(torch.float).to_sparse()
                vec_pos = ground_truth_pos.mul(answers)
                if vec_pos._nnz() == 0:
                    label_pos = torch.zeros(kg.num_entity).to_sparse().cuda(kg.device)
                else:
                    label_pos = torch.sparse.sum(vec_pos, dim=0) # true-pos
                label_neg = (torch.sparse.sum(answers, dim=0) - label_pos).coalesce() # false-pos
                
                #label_true_neg = torch.sum(ground_truth_pos.multiply(answers), dim=0) # true-neg
                #label_false_neg = torch.sum(ground_truth_neg.multiply(answers), dim=0) # false-neg
                n_pos = int(torch.sparse.sum(label_pos))
                n_neg = int(torch.sparse.sum(label_neg))
                self.num_cover = n_pos + n_neg
                self.num_head = int(torch.sparse.sum(ground_truth_pos))
                self.sc = n_pos / self.num_cover if self.num_cover else 0
                self.hc = n_pos / self.num_head if self.num_head else 0
                
                label_pos = label_pos.to_dense().view(1,-1).to_sparse()
                label_neg = label_neg.to_dense().view(1,-1).to_sparse()
                labels_pos = [label_pos]
                labels_neg = [label_neg]

                for i in range(len(self.body)-1, -1, -1):
                    atom = self.body[i]
                    path = paths[i]

                    vec = torch.sparse.sum(path, dim=0).to_dense().view(1,-1).to_sparse()
                    #vec = torch.sum((path > 0).type(torch.float), dim=0)
                    next_label_pos = torch.sparse.mm(label_pos, kg.get_adj(atom.predicate).transpose(0,1))
                    next_label_pos = (next_label_pos.to_dense() > 0).to_sparse()
                    label_pos = next_label_pos.mul(vec)
                    next_label_neg = torch.sparse.mm(label_neg, kg.get_adj(atom.predicate).transpose(0,1))
                    next_label_neg = (next_label_neg.to_dense() > 0).to_sparse()
                    label_neg = next_label_neg.mul(vec)

                    labels_pos.append(label_pos)
                    labels_neg.append(label_neg)
                    
                    
                labels_pos.reverse()
                labels_neg.reverse()
                labels_pos = torch.cat(labels_pos, dim=0)
                labels_neg = torch.cat(labels_neg, dim=0)
                return paths, answers, labels_pos.to_dense(), labels_neg.to_dense()
    
    def reasoning_matmul_semi_grounded(self, kg, h=None, learning=True, constraints=None):
        
        if h is not None:
            vec = kg.sparse_encode([h])
            return self.forward(kg, vec)
        else:
            if type(self.head.arguments[1]) == int:
                self.head.reverse(kg)
                for atom in self.body:
                    atom.reverse(kg)
                self.body.reverse()
                
            vec_head = kg.sparse_encode([self.head.arguments[0]])
            if type(self.body[0].arguments[0]) == int:
                vec = kg.sparse_encode([self.body[0].arguments[0]])
            else:
                assert self.body[0].arguments[0] != self.head.arguments[0]
                vec = torch.ones([1, kg.num_entity]).to_sparse().cuda(kg.device)
            paths, answers = self.forward(kg, vec, constraints)

            if not learning:
                return paths, answers, self.head.arguments[0]

            else:
                ground_truth = torch.sparse.mm(vec_head, kg.get_adj(self.head.predicate))#.to_dense()
                
                #ground_truth_pos = (ground_truth > 0).type(torch.float)#.to_sparse()
                #ground_truth_neg = (ground_truth == 0).type(torch.float)#.to_sparse()
                ground_truth_pos = ground_truth.coalesce()
                ground_truth_pos.values().fill_(1)
                
                #answers = (answers.to_dense() > 0).type(torch.float).to_sparse()
                vec_pos = ground_truth_pos.mul(answers)
                if vec_pos._nnz() == 0:
                    label_pos = torch.zeros(kg.num_entity).to_sparse().cuda(kg.device)
                else:
                    label_pos = torch.sparse.sum(vec_pos, dim=0) # true-pos
                label_neg = (torch.sparse.sum(answers, dim=0) - label_pos).coalesce() # false-pos
                
                #label_true_neg = torch.sum(ground_truth_pos.multiply(answers), dim=0) # true-neg
                #label_false_neg = torch.sum(ground_truth_neg.multiply(answers), dim=0) # false-neg
                n_pos = int(torch.sparse.sum(label_pos))
                n_neg = int(torch.sparse.sum(label_neg))
                self.num_cover = n_pos + n_neg
                self.num_head = int(torch.sparse.sum(ground_truth_pos))
                self.sc = n_pos / self.num_cover if self.num_cover else 0
                self.hc = n_pos / self.num_head if self.num_head else 0
                
                label_pos = label_pos.to_dense().view(1,-1).to_sparse()
                label_neg = label_neg.to_dense().view(1,-1).to_sparse()
                labels_pos = [label_pos]
                labels_neg = [label_neg]

                for i in range(len(self.body)-1, -1, -1):
                    atom = self.body[i]
                    path = paths[i]

                    vec = torch.sparse.sum(path, dim=0).to_dense().view(1,-1).to_sparse()
                    #vec = torch.sum((path > 0).type(torch.float), dim=0)
                    next_label_pos = torch.sparse.mm(label_pos, kg.get_adj(atom.predicate).transpose(0,1))
                    next_label_pos = (next_label_pos.to_dense() > 0).to_sparse()
                    label_pos = next_label_pos.mul(vec)
                    next_label_neg = torch.sparse.mm(label_neg, kg.get_adj(atom.predicate).transpose(0,1))
                    next_label_neg = (next_label_neg.to_dense() > 0).to_sparse()
                    label_neg = next_label_neg.mul(vec)

                    labels_pos.append(label_pos)
                    labels_neg.append(label_neg)
                    
                labels_pos.reverse()
                labels_neg.reverse()
                labels_pos = torch.cat(labels_pos, dim=0)
                labels_neg = torch.cat(labels_neg, dim=0)
                return paths, answers, labels_pos.to_dense(), labels_neg.to_dense()
            
class TreeRule(Rule):

    def __init__(self, head=None, body=None, hc=None, sc=None):
        
        super(TreeRule, self).__init__(head, body, hc, sc)
        self.aux_variables = ['M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V']
        #body: {"stem" : CPRule, "branches" : List[CPRule]}
        #branch: head.arg: [0, 'Y(branch variable)']

    def from_rule(self, cp_rule:CPRule):
        rule = copy.deepcopy(cp_rule)
        self.head = rule.head
        self.body = {'stem': rule, 'branches': []}
        self.hc = rule.hc
        self.sc = rule.sc
        self.num_cover = rule.num_cover
        self.num_head = rule.num_head
        return self

    def to_dict(self):

        return {'head': self.head.to_dict(), 'body':{'stem': self.body['stem'].to_dict(), 'branches':[branch.to_dict() for branch in self.body['branches']]},
                'hc':self.hc, 'sc':self.sc, 'num_cover': self.num_cover, 'num_head': self.num_head}

    def from_dict(self, dict):

        self.head = Atom().from_dict(dict['head'])
        self.body = {}
        self.body['stem'] = CPRule().from_dict(dict['body']['stem'])
        self.body['branches'] = [CPRule().from_dict(branch) for branch in dict['body']['branches']]
        self.hc = dict['hc']
        self.sc = dict['sc']
        self.num_cover = dict['num_cover'] if 'num_cover' in dict.keys() else None
        self.num_head = dict['num_head'] if 'num_head' in dict.keys() else None
        #self.formalize()
        return self

    def standardize(self, kg):
        self.body['stem'].standardize(kg)
        for branch in self.body['branches']:
            branch.standardize(kg)

    def to_anyburl(self, kg):

        head_str = self.head.to_str(kg)

        body_str_ls = [atom.to_str(kg) for atom in self.body['stem'].body]

        for branch in self.body['branches']:
            body_str_ls += [atom.to_str(kg) for atom in branch.body]
        
        body_str = ', '.join(body_str_ls)

        rule_str = head_str + ' <= ' + body_str

        num_cover = self.num_cover
        num_correct = round(num_cover*self.sc)
        
        return '\t'.join([str(num_cover), str(num_correct), str(self.sc), rule_str])

    def add_branch(self, branch:CPRule):
        self.body['branches'].append(branch)

    def apply(self, kg, h=None):
        constraints = {}
        for branch in self.body['branches']:
            paths = branch.apply(kg)
            constraints[branch.head.arguments[1]] = [path[-1] for path in paths]
        
        if h:
            if type(self.head.arguments[0]) == int:
                if self.head.arguments[0] == h:
                    paths = self.body['stem'].grounding_body(kg, [[h]], constraints=constraints)
                else:
                    return {}
            else:
                paths = self.body['stem'].grounding_body(kg, [[h]], constraints=constraints)
        else:
            paths = self.body['stem'].grounding_body(kg, constraints=constraints)
        return paths

    def reasoning(self, kg, h=None):

        paths = self.apply(kg, h)
        if type(self.head.arguments[1]) == int:
            answers = [(path[0], self.head.arguments[1]) for path in paths]            
        elif type(self.head.arguments[0]) == int:
            answers = [(self.head.arguments[0], path[-1]) for path in paths]
        else:
            answers = [(path[0], path[-1]) for path in paths]
        return paths, answers

    def reasoning_matmul(self, kg, h=None, learning=False):
        constraints = {}
        for branch in self.body['branches']:
            paths, answers, _ = branch.reasoning_matmul(kg, learning=False)
            answers = torch.sparse.sum(answers, dim=0).coalesce().to_dense().view(1,-1).to_sparse()
            answers.values().fill_(1)
            constraints[branch.head.arguments[1]] = answers

        res = self.body['stem'].reasoning_matmul(kg, learning=learning, constraints=constraints)
        self.sc = self.body['stem'].sc
        self.hc = self.body['stem'].hc
        
        return res