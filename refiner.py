from kg import *
import torch
import logging

class RepBuffer():
    
    def __init__(self):
        self.memory = []
        
    def push(self, state, action_id, reward):
        self.memory.append([state, action_id, reward])

    def shuffle(self):
        random.shuffle(self.memory)
    
    def get_mean_reward(self):
        reward = [t[2] for t in self.memory]
        mean_reward = np.mean(reward)
        return mean_reward
    
    def zero_mean(self):
        mean_reward = self.get_mean_reward()
        for t in self.memory:
            t[2] -= mean_reward
    
    def clear(self):
        self.memory = []

class MLP(torch.nn.Module):
 
    def __init__(self, num_i, num_h, num_o):
        super(MLP,self).__init__()
        
        self.linear1=torch.nn.Linear(num_i,num_h)
        self.act1=torch.nn.Tanh()
        self.linear2=torch.nn.Linear(num_h,num_h)
        self.act2=torch.nn.Tanh()
        self.linear3=torch.nn.Linear(num_h,num_o)
  
    def forward(self, x):
        x = self.linear1(x)
        x = self.act1(x)
        x = self.linear2(x)
        x = self.act2(x)
        x = self.linear3(x)
        return x

class Refiner(torch.nn.Module):

    def __init__(self, kg:KG, args):
        super(Refiner, self).__init__()
        self.kg = kg
        self.args = args
        self.device = args.device
        self.max_rule_len = args.max_rule_len
        self.exp_path = args.exp_dir + args.exp_name + '/'
        mkdir(self.exp_path)

        self.n_action = (self.args.max_rule_len+1)*self.kg.num_relation*2 + 1
        self.mlp = MLP((self.max_rule_len+1)*self.kg.num_entity*2, self.args.hidden_dim, self.n_action).cuda(self.device)
        self.lstm = torch.nn.LSTM(input_size=2*self.kg.num_entity, hidden_size=self.args.hidden_dim, bidirectional=True).cuda(self.device)
        self.linear = torch.nn.Linear(2*self.args.hidden_dim*(self.max_rule_len+1), self.n_action).cuda(self.device)
    
    def from_ckpt(self, ckpt):
        self.load_state_dict(torch.load(self.exp_path+ckpt))

    def encode_state_lstm(self, rule):
        paths, answers, labels_pos, labels_neg = rule.reasoning_matmul(self.kg, learning=True)
        labels_pos = [tensor.view(1,-1) for tensor in labels_pos]
        labels_neg = [tensor.view(1,-1) for tensor in labels_neg]
        pos_input = torch.cat(labels_pos, dim=0).unsqueeze(0)
        neg_input = torch.cat(labels_neg, dim=0).unsqueeze(0)
        pos_state = torch.mean(self.lstm(pos_input)[0][0], dim=0)
        neg_state = torch.mean(self.lstm(neg_input)[0][0], dim=0)
        state = torch.cat([pos_state, neg_state], dim=-1)
        return state, labels_pos, labels_neg
    
    def encode_state(self, rule):
        paths, answers, labels_pos, labels_neg = rule.reasoning_matmul(self.kg, learning=True)
        n = labels_pos.shape[0]
        labels_pos = torch.cat([labels_pos, torch.zeros([self.max_rule_len+1-n, self.kg.num_entity]).cuda(self.device)], dim=0)
        labels_neg = torch.cat([labels_neg, torch.zeros([self.max_rule_len+1-n, self.kg.num_entity]).cuda(self.device)], dim=0)
        state = torch.cat([labels_pos, labels_neg], dim=1).unsqueeze(0)
        return state, labels_pos, labels_neg
    
    def get_policy(self, state):
        batchsize = state.shape[0]
        policy = self.linear(self.lstm(state)[0].view(batchsize, -1))
        policy = torch.softmax(policy, dim=1)
        return policy
        
    def sample_trajectory(self, rules):
        rep = RepBuffer()
        for rule in tqdm(rules):
            n = len(rule.body)
            state, labels_pos, labels_neg = self.encode_state(rule)
            policy = self.get_policy(state)
            samples = episilon_uniform_sampling(policy, episilon=0.1, num_samples=self.args.num_reinforce_sample).view(-1)
            for action_id in samples.tolist():
                if action_id == self.n_action-1:
                    reward = 0
                else:
                    idx = action_id // (2*self.kg.num_relation)
                    r = action_id - idx*2*self.kg.num_relation
                    if idx > n:
                        reward = - (rule.sc + rule.hc)
                    else:
                        mask = self.kg.r2t_mat[:,r].T
                        labels_pos_copy = copy.deepcopy(labels_pos)[:n+1]
                        labels_neg_copy = copy.deepcopy(labels_neg)[:n+1]
                        for j in range(idx, n+1):
                            labels_pos_copy[j] = labels_pos_copy[j].multiply(mask)
                            labels_neg_copy[j] = labels_neg_copy[j].multiply(mask)
                            if j < n:
                                mask = (torch.matmul(mask, self.kg.get_adj(rule.body[j].predicate)) > 0).type(torch.float)
                        n_pos = int(torch.sum(labels_pos_copy[-1]))
                        n_neg = int(torch.sum(labels_neg_copy[-1]))
                        num_cover = n_pos + n_neg
                        sc = n_pos / (n_pos+n_neg) if num_cover else 0
                        hc = n_pos / rule.num_head
                        #reward = sc + hc - (rule.sc + rule.hc)
                        reward = sc - rule.sc
                rep.push(state, action_id, reward)
        return rep
    
    def get_loss(self, state, labels_pos, labels_neg, sc, masked_predicates, valid=False):
        batchsize = state.shape[0]
        policy = self.get_policy(state)
        # heuristic 1
        heuristic = torch.matmul(labels_pos, self.kg.r2t_mat)/torch.matmul(labels_pos+labels_neg, self.kg.r2t_mat)
        heuristic[torch.isnan(heuristic)] = -1e5
        # heuristic 2
        #sc = sc.view(batchsize, 1, 1)
        #labels = (1-sc)*labels_pos - sc*labels_neg
        #heuristic = torch.matmul(labels, self.kg.r2t_mat)
        
        ## -score
        #score = torch.sum(labels, dim=-1, keepdim=True)
        #heuristic -= score
        
        for i in range(batchsize):
            heuristic[i,:,masked_predicates[i]] = -1e5
            
        heuristic = heuristic.view(batchsize, -1) - sc
        heuristic[heuristic <= 0] = 0
        
        heuristic = torch.cat([heuristic, torch.sum(heuristic, dim=1, keepdim=True)==0], dim=1)
        heuristic = heuristic / torch.sum(heuristic, dim=1, keepdim=True)
        
        loss = torch.sum(heuristic*torch.log(torch.clamp(heuristic, 1e-10)) - heuristic*torch.log(policy), dim=1)
        if valid:
            loss = torch.sum(loss)
        else:
            loss = torch.mean(loss)
        
        return loss

    def pretrain_policy_net(self, train_loader, valid_loader):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.args.lr_pretrain)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=5)
        logging.basicConfig(filename=self.exp_path+'pretrain.log', level=logging.INFO)
        for e in range(self.args.epochs_pretrain):
            logging.info("validating now")
            valid_loss = 0
            for state, labels_pos, labels_neg, sc, masked_predicates in tqdm(iter(valid_loader)):
                valid_loss += self.get_loss(state, labels_pos, labels_neg, sc, masked_predicates, valid=True)
            scheduler.step(valid_loss)
            logging.info("valid loss = {}".format(float(valid_loss)))
            
            logging.info("start training epoch {}".format(e))
            for state, labels_pos, labels_neg, sc, masked_predicates in tqdm(iter(train_loader)):
                loss = self.get_loss(state, labels_pos, labels_neg, sc, masked_predicates, valid=False)
                optimizer.zero_grad()       
                loss.backward()
                optimizer.step()
                logging.info(loss)
            torch.save(self.state_dict(), self.exp_path+"pretrain.pkl")

    def train_reinforce(self, rules):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.args.lr_reinforce)
        #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=5)
        logging.basicConfig(filename=self.exp_path+'reinforce.log', level=logging.INFO)
        for e in range(self.args.epochs_reinforce):
            logging.info("start sampling epoch {}".format(e))
            rep = self.sample_trajectory(rules)
            logging.info("mean reward = {}".format(rep.get_mean_reward()))
            logging.info("start training epoch {}".format(e))
            for e_train in range(1):
                rep.shuffle()
                rep.zero_mean()
                loss, cnt = 0, 0
                for state, action_id, reward in rep.memory:
                    policy = self.get_policy(state)
                    prob = policy[0][action_id]
                    #logging.info("logprob {} reward {}".format(torch.log(prob), reward))
                    loss += -torch.clamp(torch.log(prob), min=-10) * reward
                    cnt += 1
                    if cnt == self.args.batch_size:
                        loss /= self.args.batch_size
                        optimizer.zero_grad()       
                        loss.backward()
                        optimizer.step()
                        logging.info(loss)
                        loss, cnt = 0, 0
            torch.save(self.state_dict(), self.exp_path+"reinforce.pkl")
            
    def refine_rules(self, rules):
        #logging.basicConfig(filename=self.exp_path+'refine.log', level=logging.INFO)
        #logging.info("start refining rules")
        refined_rules = []
        for rule in tqdm(rules):
            n = len(rule.body)
            refined_rules.append(rule)
            state, labels_pos, labels_neg = self.encode_state(rule)
            policy = self.get_policy(state)
            categorical = dist.Categorical(policy)
            samples = categorical.sample([self.args.num_refine_sample])
            unique_samples = torch.unique(samples).tolist()
            for action_id in unique_samples:
                if action_id == self.n_action-1:
                    continue
                idx = action_id // (2*self.kg.num_relation)
                r = action_id - idx*2*self.kg.num_relation
                if idx <= n:
                    refined_rule = TreeRule().from_rule(rule)
                    var = rule.get_variables()[idx]
                    aux_var = refined_rule.aux_variables.pop(0)
                    branch = CPRule(head=Atom(-1,[-1,var]), body=[Atom(r,[aux_var,var])])
                    refined_rule.add_branch(branch)
                    mask = self.kg.r2t_mat[:,r].T
                    labels_pos_copy = copy.deepcopy(labels_pos)[:n+1]
                    labels_neg_copy = copy.deepcopy(labels_neg)[:n+1]
                    for j in range(idx, n+1):
                        labels_pos_copy[j] = labels_pos_copy[j].multiply(mask)
                        labels_neg_copy[j] = labels_neg_copy[j].multiply(mask)
                        if j < n:
                            mask = (torch.matmul(mask, self.kg.get_adj(rule.body[j].predicate)) > 0).type(torch.float)
                    n_pos = int(torch.sum(labels_pos_copy[-1]).detach().cpu().numpy())
                    n_neg = int(torch.sum(labels_neg_copy[-1]).detach().cpu().numpy())
                    refined_rule.num_cover = n_pos + n_neg
                    refined_rule.sc = n_pos / (n_pos+n_neg) if refined_rule.num_cover else 0
                    refined_rules.append(refined_rule)
        return refined_rules