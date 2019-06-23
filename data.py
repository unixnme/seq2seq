import numpy as np

class DataGenerator(object):
    def __init__(self,
                 max_vocab:int=1000,
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
        self.data = np.ones([self.num_examples, max_length]) * self.EOS
        self.target = np.ones([self.num_examples, max_length + 1]) * self.EOS
        self.seq_lengths = np.random.randint(1, max_length + 1, [len(self.data)])
        for idx in range(len(self.data)):
            seq_length = self.seq_lengths[idx]
            seq = np.random.randint(1, max_vocab + 1, [seq_length])
            self.data[idx][:seq_length] = seq
            self.target[:,0] = self.BOS
            self.target[idx][1:seq_length+1] = seq[::-1]


if __name__ == '__main__':
    generator = DataGenerator(10, 5, 7)
    print(generator.data)
    print(generator.target)