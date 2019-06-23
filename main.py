import argparse
from data import DataGenerator
from torch.utils.data import DataLoader


def parse_args():
    parser = argparse.ArgumentParser('Training script')
    parser.add_argument('--num_vocab', type=int, default=100)
    parser.add_argument('--max_length', type=int, default=20)
    parser.add_argument('--num_examples', type=int)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--device', type=str, default='cpu')

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    gen = DataGenerator(args.num_vocab, args.max_length, args.num_examples)
    train_set, test_set = gen.get_tarin_test_datasets()
    train_loader = DataLoader(train_set, args.batch_size, True, pin_memory=True)
    test_loader = DataLoader(test_set, args.batch_size, True, pin_memory=True)


if __name__ == "__main__":
    main()