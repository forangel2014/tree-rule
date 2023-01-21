type_rule_id2entity = {}
with open('./WN18RR/entity2id.txt') as f:
    lines = f.readlines()[1:]
    for line in lines:
        entity, id = line.strip().split('\t')
        type_rule_id2entity[id] = entity

entity2tree_rule_id = {}
with open('../../tree-rule/data/wn18/entities.dict') as f:
    lines = f.readlines()
    for line in lines:
        id, entity = line.strip().split('\t')
        entity2tree_rule_id[entity] = id

type_rule_id2tree_rule_id = {}
for id in type_rule_id2entity.keys():
    type_rule_id2tree_rule_id[id] = entity2tree_rule_id[type_rule_id2entity[id]]

entity_type = {}
with open('./WN18RR/type_entity.txt') as f:
    lines = f.readlines()[1:]
    for line in lines:
        ls = line.strip().split('\t')
        type, entities = ls[0], ls[1:]
        entities = [type_rule_id2tree_rule_id[id] for id in entities]
        entity_type[type] = entities

with open('../../tree-rule/data/wn18/entity_type.txt', 'w') as f:
    for id, entities in entity_type.items():
        f.writelines(id + '\t' + ' '.join(entities) + '\n')