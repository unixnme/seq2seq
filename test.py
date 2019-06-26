import argparse
import torch


def main():
    parser = argparse.ArgumentParser('Test Seq2Seq')
    parser.add_argument('--load', required=True, type=str)
    parser.add_argument('--max_length', type=int, default=100)
    parser.add_argument('--input', type=int, nargs='+', required=True)
    args = parser.parse_args()

    if 0 in args.input:
        print('0 cannot be allowed in the input')
        exit(-1)

    network = torch.load(args.load, map_location='cpu')
    out = network.generate(args.input, args.max_length)
    print(out)


if __name__ == '__main__':
    main()