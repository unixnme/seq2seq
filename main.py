import argparse
from data import DataGenerator
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from network import Network
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser('Training script')
    parser.add_argument('--num_vocab', type=int, default=10)
    parser.add_argument('--max_length', type=int, default=10)
    parser.add_argument('--num_examples', type=int)
    parser.add_argument('--emb_dim', type=int, default=32)
    parser.add_argument('--hidden_dim', type=int, default=16)
    parser.add_argument('--rnn_type', type=str, default='GRU')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--device', type=str, default='cpu')

    args = parser.parse_args()
    return args


def single_step(network: Network, loader: DataLoader, optimizer=None):
    train = True if optimizer is not None else False
    total_loss = 0
    total_correct = 0

    network.train(train)
    for param in network.parameters():
        param.requires_grad = train

    for x, trg in loader:
        if train:
            optimizer.zero_grad()
        loss, out = network(x, trg)
        if train:
            loss.backward()
            optimizer.step()
        total_loss += loss.item()
        correct = torch.sum(out == trg)
        total_correct += correct.item()

    acc = total_correct / np.prod(loader.dataset.target.shape) * 100
    return acc, total_loss


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
    #optimizer = optim.SGD(network.parameters(), args.lr, 0.99)
    optimizer = optim.RMSprop(network.parameters(), args.lr)

    for epoch in range(args.epochs):
        acc, loss = single_step(network, train_loader, optimizer)
        print("#%d: \tTRAIN  loss:%.03f \t acc:%.03f" % (epoch, loss, acc))
        acc, loss = single_step(network, test_loader)
        print("#%d: \tTEST  loss:%.03f \t acc:%.03f" % (epoch, loss, acc))



if __name__ == "__main__":
    main()