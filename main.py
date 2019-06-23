import argparse
from data import DataGenerator
import torch.optim as optim
from torch.utils.data import DataLoader
from network import Network


def parse_args():
    parser = argparse.ArgumentParser('Training script')
    parser.add_argument('--num_vocab', type=int, default=100)
    parser.add_argument('--max_length', type=int, default=20)
    parser.add_argument('--num_examples', type=int)
    parser.add_argument('--emb_dim', type=int, default=32)
    parser.add_argument('--hidden_dim', type=int, default=16)
    parser.add_argument('--rnn_type', type=str, default='GRU')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--device', type=str, default='cpu')

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    gen = DataGenerator(args.num_vocab, args.max_length, args.num_examples)
    train_set, test_set = gen.get_tarin_test_datasets()
    train_loader = DataLoader(train_set, args.batch_size, True, pin_memory=True)
    test_loader = DataLoader(test_set, args.batch_size, True, pin_memory=True)
    network = Network(args.num_vocab + 2,
                      args.num_vocab + 2,
                      args.emb_dim,
                      args.hidden_dim,
                      args.rnn_type,
                      args.device)
    optimizer = optim.SGD(network.parameters(), args.lr, 0.9)

    total_loss = 0
    for epoch in range(args.epochs):
        for x,trg in train_loader:
            optimizer.zero_grad()
            loss = network(x, trg)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(total_loss)

if __name__ == "__main__":
    main()