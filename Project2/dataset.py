import numpy as np
#from torch.utils.data import Dataset
from collections import defaultdict

'''class NICODataset(Dataset):
    def __init__(self, raw_dataset):
        self.input, self.label = raw_dataset[:, :-2], raw_dataset[:, -1]

    def shuffle(self):
        shuffle_index = np.arange(self.__len__())
        np.random.shuffle(shuffle_index)
        self.input = self.input[shuffle_index]
        self.label = self.label[shuffle_index]

    def __getitem__(self, index):
        return self.input[index], self.label[index]

    def __len__(self):
        return len(self.input)'''

def split_random(raw_dataset, local_config):
    train_size = 2956
    dataset = raw_dataset
    np.random.seed(19260817)
    np.random.shuffle(dataset)
    return dataset[:train_size], dataset[train_size:]

def split_by_context(raw_dataset, local_config):
    train_ratio, test_ratio = local_config["split_ratio"]
    context_label = raw_dataset[:, -2:].astype(int)
    train_context, valid_context = defaultdict(set), defaultdict(set)
    for context, label in context_label:
        if len(train_context[label]) < train_ratio:
            train_context[label].add(context)
        elif context not in train_context[label]:
            valid_context[label].add(context)
    train_set, valid_set = [], []
    for i in range(len(context_label)):
        context, label = context_label[i]
        if context in train_context[label]:
            train_set.append(raw_dataset[i])
        else:
            assert context in valid_context[label]
            valid_set.append(raw_dataset[i])
    train_set = np.stack(train_set, axis=0)
    valid_set = np.stack(valid_set, axis=0)
    print(train_set.shape, valid_set.shape, raw_dataset.shape)
    return train_set, valid_set
