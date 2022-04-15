import random
import collections
from datasets import DatasetDict

def split(dataset, validation_ratio=0.2) :
    id_map = collections.defaultdict(list)

    id_list = dataset['id']
    for i, id in enumerate(id_list) :
        doc_id = id.split('-')[0]
        id_map[doc_id].append(i)

    ids = list(id_map.keys())
    id_size = len(ids)
    validation_size = int(id_size * validation_ratio)

    validation_ids = random.sample(ids, validation_size)
    validaion_index_list = []
    for id in validation_ids :
        validaion_index_list.extend(id_map[id])

    train_index_list = list(set(range(len(dataset))) - set(validaion_index_list))

    train_dataset = dataset.select(train_index_list)
    validation_dataset = dataset.select(validaion_index_list)

    dset = DatasetDict({'train' : train_dataset, 'validation' : validation_dataset})
    return dset