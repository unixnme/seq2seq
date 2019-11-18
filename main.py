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
    parser.add_argument('--max_length', type=int, default=20)
    parser.add_argument('--num_examples', type=int)
    parser.add_argument('--emb_dim', type=int, default=16)
    parser.add_argument('--hidden_dim', type=int, default=16)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--tied', action='store_true')
    parser.add_argument('--rnn_type', type=str, default='GRU')
    parser.add_argument('--drop', type=float, default=0.5)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--save', type=str, default='model.pt')
    parser.add_argument('--lr_decay', type=float, default=.999)

    args = parser.parse_args()
    if args.tied: assert args.emb_dim == args.hidden_dim
    return args


def single_step(network: Network, loader: DataLoader, optimizer=None, force_prob:float=0.5):
    train = True if optimizer is not None else False
    total_loss = 0
    total_correct = 0

    network.train(train)
    for param in network.parameters():
        param.requires_grad = train

    for x, trg in loader:
        if train:
            optimizer.zero_grad()
            loss, out = network(x, trg, force_prob)
        else:
            loss, out = network(x, trg, 0)
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
                      args.num_layers,
                      args.drop,
                      args.rnn_type,
                      args.tied,
                      args.device)
    #optimizer = optim.SGD(network.parameters(), args.lr, 0.99)
    optimizer = optim.RMSprop(network.parameters(), args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, 1, args.lr_decay)

    best_loss = float('inf')
    for epoch in range(args.epochs):
        for param_group in optimizer.param_groups:
            print('#%d: \tLR %f' % (epoch, param_group['lr']))
        acc, loss = single_step(network, train_loader, optimizer, .5 - epoch / args.epochs)
        print("#%d: \tTRAIN  loss:%.03f \t acc:%.03f" % (epoch, loss, acc))
        acc, loss = single_step(network, test_loader)
        print("#%d: \tTEST  loss:%.03f \t acc:%.03f" % (epoch, loss, acc))
        scheduler.step()

        if best_loss > loss:
            best_loss = loss
            print('Saving model to %s' % args.save)
            torch.save(network, args.save)


if __name__ == "__main__":
    main()
