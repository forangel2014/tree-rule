import argparse
from ast import arg
from kg import KG
import torch
from refiner_truncate import Refiner
from utils import get_rule_loader, get_rule_loader_split
torch.manual_seed(37)
from tqdm import tqdm

def main(args):

    kg = KG(args)
    #mined_rules = kg.mine_rules(max_rule_len=2)
    #kg.write_rules_anyburl(mined_rules, 'bbfs-{}.ab'.format(args.dataset))

    rules = kg.load_rules_anyburl('{}.ab'.format(args.rule_name))
    #print(kg.get_average_confidence('test'))
    refined_rules = kg.refine_rules(rules)
    info = "aux"
    if args.entity:
        info += "+entity"
    if args.type_info:
        info += "+type"
    kg.write_rules_anyburl(refined_rules, '{}+{}+{}+{}.ab'.format(args.rule_name, args.dataset, info, args.num_refine_sample))

    #refiner = Refiner(kg, args)
    #train_loader, valid_loader = get_rule_loader_split(rules, kg, args.max_rule_len, args.batch_size, args.valid_ratio)
    #refiner.pretrain_policy_net(train_loader, valid_loader)
    #refined_rules = refiner.refine_rules(rules)
    #kg.write_rules_anyburl(refined_rules, 'amie-pretrain-{}.ab'.format(args.num_refine_sample))

    #refiner.from_ckpt('pretrain.pkl')
    #refiner.from_ckpt('reinforce.pkl')
    #refiner.train_reinforce(rules)
    #refined_rules = refiner.refine_rules(rules)
    #kg.write_rules_anyburl(refined_rules, 'amie-reinforce-{}.ab'.format(args.num_refine_sample))


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='TreeRule')
    
    parser.add_argument("--type_info", type=lambda x: x.lower() == 'true', required=False)
    parser.add_argument("--entity", type=lambda x: x.lower() == 'true', required=False)
    
    parser.add_argument("--hidden_dim", type=int, default=100)    
    parser.add_argument("--max_rule_len", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=10)
    parser.add_argument("--lr_pretrain", type=float, default=1e-3)
    parser.add_argument("--lr_reinforce", type=float, default=1e-3)
    parser.add_argument("--valid_ratio", type=float, default=0.01)
    parser.add_argument("--epochs_pretrain", type=int, default=10)
    parser.add_argument("--epochs_reinforce", type=int, default=10)
    parser.add_argument("--num_reinforce_sample", type=int, default=10)
    parser.add_argument("--num_refine_sample", type=int, default=1)
    parser.add_argument("--device", type=int, default=0)

    parser.add_argument("--dataset", type=str, default='FB15k-237')
    parser.add_argument("--rule_dir", type=str, default='./rules/')
    parser.add_argument("--data_dir", type=str, default='./data/')
    parser.add_argument("--exp_dir", type=str, default='./exp/')
    parser.add_argument("--exp_name", type=str, default='test')
    parser.add_argument("--rule_name", type=str, default='bbfs')

    args = parser.parse_args()
    print(args)

    main(args)