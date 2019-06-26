import argparse
import torch
import sys


def main():
    parser = argparse.ArgumentParser('Test Seq2Seq')
    parser.add_argument('--load', required=True, type=str)
    parser.add_argument('--max_length', type=int, default=100)
    args = parser.parse_args()

    network = torch.load(args.load, map_location='cpu')
    print('input your sequence:')
    for line in sys.stdin:
        seq_in = [int(token) for token in line.split()]
        if 0 in seq_in:
            print('0 cannot be in the input sequence')
            continue
        out = network.generate(seq_in, args.max_length)
        print(out)
        print('input your sequence:')


if __name__ == '__main__':
    main()