entities = []

with open('entities.dict') as f:
    lines = f.readlines()
    for line in lines:
        entity_str = line.strip().split('\t')[1]
        entities.append(entity_str)
        
with open('entity_type.txt') as f, open('entity_type_str.txt', 'w') as g:
    lines = f.readlines()
    for line in lines:
        type_id, entity_ids = line.strip().split('\t')
        entity_ids = entity_ids.split(' ')
        type_str = "type_" + type_id
        entity_str = [entities[int(id)] for id in entity_ids]
        g.writelines(type_str + '\t' + ' '.join(entity_str) + '\n')
        