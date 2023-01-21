entities = []
relations = []
with open('train.txt') as f:
    lines = f.readlines()
    for line in lines:
        h, r, t = line.strip().split('\t')
        #if h not in entities:
        entities.append(h)
        #if r not in relations:
        relations.append(r)
        #if t not in entities:
        entities.append(t)

entities = list(set(entities))
relations = list(set(relations))

with open('entities.dict', 'w') as f:
    for i in range(len(entities)):
        f.writelines(str(i) + '\t' + entities[i] + '\n')
        
with open('relations.dict', 'w') as f:
    for i in range(len(relations)):
        f.writelines(str(i) + '\t' + relations[i] + '\n')