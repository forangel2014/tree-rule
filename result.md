### Results on FB15k-237
| Rule        | MRR             | hit@1 | hit@3 | hit@10 | mean conf |
| :--:                          | :--:  | :--:   |  :--:  | ----   |----   |
|BBFS-CP       |24.32           | 18.38  | 25.97  |  36.65  | 13.02    |
|BBFS-CP+TREE(aux)  |26.03       | 19.46 | 28.64  |  38.78  | 15.61     |
|BBFS-CP+TREE(aux+entity)  |26.50       | 20.03 | 29.05  |  39.09  | 19.23   |
|BBFS-CP+TREE(aux+entity+type)  |26.44       | 19.93 | 29.05  |  39.06  | 17.65   |
|AMIE-CP    |22.60           | 17.25  | 24.27  |  33.78  | 30.71|
|AMIE-CP+TREE(aux)  |24.53       | 18.80 | 26.98  |  35.72  | 33.43 |
|AMIE-CP+TREE(aux+entity)  |24.81       | 19.20 | 27.21  |  35.77  | 35.72   |
|AMIE-CP+TREE(aux+entity+type)  |24.80       | 19.16 | 27.27  |  35.75  | 33.21|
|Anyburl-CP  |32.74           | 23.94  | 35.75  |  50.98  |  27.12  |
|Anyburl-CP+TREE(aux)  |32.12           | 24.16  | 33.99  |  49.19  | 30.59  |
|Anyburl-CP+TREE(aux+entity)  |33.81           | 25.40  | 36.96  |  50.90  | 32.99   |
|Anyburl-CP+TREE(aux+entity+type)  |33.44           | 25.13  | 36.61  |  50.19  |  30.39  |

### Results on WN18RR
| Rule        | MRR             | hit@1 | hit@3 | hit@10 |
| :--:                          | :--:  | :--:   |  :--:  | ----   |
|BBFS-CP       |39.29           | 37.94  | 40.08  |  42.02  |
|BBFS-CP+TREE(aux)       |39.41           | 37.97  | 40.43  |  42.25  |
|BBFS-CP+TREE(aux+entity)       |39.81           | 38.55  | 40.58  |  42.37  |
|AMIE-CP    |36.21           | 36.06  | 36.31  |  36.47  |
|AMIE-CP+TREE(aux)    |36.18           | 36.06  | 36.25  |  36.44  |
|AMIE-CP+TREE(aux+entity)    |36.19           | 36.06  | 36.27  |  36.44  |
|Anyburl-CP    |48.42           | 44.22  | 50.99  |  56.03  |
|Anyburl-CP+TREE(aux)  |32.12           | 24.16  | 33.99  |  49.19  |
|Anyburl-CP+TREE(aux+entity)  |33.81           | 25.40  | 36.96  |  50.90  |

### Results on YAGO3-10
| Rule        | MRR             | hit@1 | hit@3 | hit@10 |
| :--:                          | :--:  | :--:   |  :--:  | ----   |
|BBFS-CP       |53.47           | 47.56  | 58.34  |  63.32  |
|BBFS-CP+TREE(aux)    |54.43           | 48.90  | 59.02  |  63.74  |
|BBFS-CP+TREE(aux+entity)    |54.40           | 48.86  | 58.98  |  63.76  |
|AMIE-CP    |52.07           | 46.68  | 57.08  |  60.74  |
|AMIE-CP+TREE(aux)    |52.69           | 47.60  | 57.54  |  60.74  |
|AMIE-CP+TREE(aux+entity)    |52.70           | 47.60  | 57.54  |  60.80  |
|Anyburl-CP    |63.07           | 57.34  | 67.30  |  72.10  |
|Anyburl-CP+TREE(aux)   |61.29          | 55.08  | 65.98  |  71.34  | 
|Anyburl-CP+TREE(aux+entity)  |62.65     | 57.02  | 66.98  |  71.38  |
