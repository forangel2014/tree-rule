import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader

class RuleDataset(Dataset):
    def __init__(self, rules, kg, max_rule_len):
        self.rules = rules
        self.kg = kg
        self.max_rule_len = max_rule_len

    # need to overload
    def __len__(self):
        return len(self.rules)

    # need to overload
    def __getitem__(self, idx):
        rule = self.rules[idx]
        paths, answers, labels_pos, labels_neg = rule.reasoning_matmul(self.kg, learning=True)
        n = labels_pos.shape[0]
        labels_pos = torch.cat([labels_pos, torch.zeros([self.max_rule_len+1-n, self.kg.num_entity]).cuda(self.kg.device)], dim=0)
        labels_neg = torch.cat([labels_neg, torch.zeros([self.max_rule_len+1-n, self.kg.num_entity]).cuda(self.kg.device)], dim=0)
        state = torch.cat([labels_pos, labels_neg], dim=1)
        return state, labels_pos, labels_neg, torch.tensor([rule.sc]).cuda(self.kg.device), torch.tensor(rule.all_predicates(self.kg)).cuda(self.kg.device)
