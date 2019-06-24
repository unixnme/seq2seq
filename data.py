import numpy as np
import torch.utils.data as data

class Dataset(data.Dataset):
    def __init__(self, data:np.ndarray, target:np.ndarray):
        assert len(data) == len(target), "data & target must be of equal size"
        self.data = data
        self.target = target

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item:int):
        return self.data[item], self.target[item]


class DataGenerator(object):
    def __init__(self,
                 max_vocab:int=1000, # without BOS & EOS
                 max_length:int=20,
                 num_examples:int=None,
                 ):
        self.max_vocab = max_vocab
        self.max_length = max_length
        if num_examples is None:
            self.num_examples = np.max(max_vocab**2)
        else:
            self.num_examples = num_examples

        self.BOS = 0
        self.EOS = max_vocab + 1
        self.data = np.ones([self.num_examples, max_length], dtype=np.int64) * self.EOS
        self.target = np.ones([self.num_examples, max_length + 2], dtype=np.int64) * self.EOS
        self.seq_lengths = np.random.randint(max_length // 2, max_length + 1, [len(self.data)])
        for idx in range(len(self.data)):
            seq_length = self.seq_lengths[idx]
            seq = np.random.randint(1, max_vocab + 1, [seq_length])
            self.data[idx][:seq_length] = seq
            self.target[:,0] = self.BOS
            self.target[idx][1:seq_length+1] = seq[::-1]

    def get_tarin_test_datasets(self, ratio:list=None) -> [Dataset, Dataset]:
        if ratio is None:
            ratio = [.8, .2]
        assert np.abs(np.sum(ratio) - 1) < 1e-6, "ratio must sum to 1"
        train_idx = int(self.num_examples * ratio[0])
        return Dataset(self.data[:train_idx], self.target[:train_idx]), \
                Dataset(self.data[train_idx:], self.target[train_idx:])


if __name__ == '__main__':
    generator = DataGenerator(10, 5, 7)
    train_set, test_set = generator.get_tarin_test_datasets()
    print("Train set:")
    for x,y in train_set:
        print(x, y)
    print("Test set:")
    for x,y in test_set:
        print(x,y)
