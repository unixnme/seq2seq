import argparse
import torch
import sys


def main():
    parser = argparse.ArgumentParser('Test Seq2Seq')
    parser.add_argument('--load', required=True, type=str)
    args = parser.parse_args()

    network = torch.load(args.load, map_location='cpu')
    max_num = network.encoder.embedding.num_embeddings - 2
    max_length = network.max_length
    print('input your sequence:')
    for line in sys.stdin:
        seq_in = [int(token) for token in line.split()]
        if len(seq_in) > max_length:
            raise Exception("input seq exceeds model's max length %s" % max_length)
        for tok in seq_in:
            if tok == 0: print('0 cannot be in the sequence')
            elif tok > max_num: print('%d is the max number allowed in the sequence' % max_num)
            else: continue
            raise Exception()
        out = network.generate(seq_in, max_length)
        print(out)
        print('input your sequence:')


if __name__ == '__main__':
    main()