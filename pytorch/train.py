#!/usr/bin/env python
# coding: utf-8
# @Author: lapis-hong
# @Date  : 2018/5/15
import os
import sys

import argparse

import torch
import torch.autograd as autograd
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import Dictionary, MyDataset
from siamese_network import EmbeddingCNN, EmbeddingRNN, SiameseNet, ContrastiveLoss


def get_args():
    parser = argparse.ArgumentParser(description='Siamese text classifier')
    parser.add_argument('--dictionary', type=str, default='',
                        help='path to save the dictionary, for faster corpus loading')
    parser.add_argument('--word_vector', type=str, default='',
                        help='path for pre-trained word vectors (e.g. GloVe)')
    parser.add_argument('--word_embedding_type', type=str, default='rand',
                        help='word embedding type {`rand`, `static`, `non-static`}')
    parser.add_argument('--train_data', type=str, default='../data/train.csv',
                        help='training data path')
    parser.add_argument('--val_data', type=str, default='../data/test.csv',
                        help='validation data path')
    parser.add_argument('--test_data', type=str, default='',
                        help='test data path')
    parser.add_argument('--char_model', type=bool, default=True,
                        help='whether to use character level model')
    # RNN
    parser.add_argument('--sequence_length', type=bool, default=20,
                        help='max sequence length')
    parser.add_argument('--embedding_dim', type=int, default=64,
                        help='size of word embeddings')
    parser.add_argument('--hidden_units', type=int, default=200,
                        help='number of hidden units per layer')
    parser.add_argument('--num_layers', type=int, default=2,
                        help='number of layers in BiLSTM')
    # CNN
    parser.add_argument('--kernel_sizes', type=list, default=[2,3,4,5],
                        help='kernel sizes in CNN')
    parser.add_argument('--num_kernels', type=int, default=100,
                        help='number of kernels in CNN')
    # parser.add_argument('--attention-unit', type=int, default=350,
    #                     help='number of attention unit')
    # parser.add_argument('--attention-hops', type=int, default=1,
    #                     help='number of attention hops, for multi-hop attention model')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='dropout applied to layers (0 = no dropout)')
    parser.add_argument('--clip', type=float, default=0.5,
                        help='clip to prevent the too large grad in LSTM')
    # parser.add_argument('--nfc', type=int, default=512,
    #                     help='hidden (fully connected) layer size for classifier MLP')
    # train
    parser.add_argument('--lr', type=float, default=.001,
                        help='initial learning rate')
    parser.add_argument('--epochs', type=int, default=50,
                        help='upper epoch limit')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='batch size for training')
    parser.add_argument('--cuda', action='store_true',
                        help='use CUDA')
    parser.add_argument('--log_interval', type=int, default=100, metavar='N',
                        help='train log interval')
    parser.add_argument('--test_interval', type=int, default=100, metavar='N',
                        help='eval interval')
    parser.add_argument('--save_interval', type=int, default=1000, metavar='N',
                        help='save interval')
    parser.add_argument('--save_dir', type=str, default='model_torch',
                        help='path to save the final model')
    parser.add_argument('--optimizer', type=str, default='Adam',
                        help='type of optimizer')
    parser.add_argument('--seed', type=int, default=123,
                        help='random seed')
    # parser.add_argument('--penalization-coeff', type=float, default=1,
    #                     help='the penalization coefficient')
    return parser.parse_args()


def metrics(y, y_pred):
    # 8-bit integer (unsigned)
    # y, y_pred = torch.ByteTensor(y), torch.ByteTensor(y_pred)
    TP = ((y_pred == 1) & (y == 1)).sum().float()
    TN = ((y_pred == 0) & (y == 0)).sum().float()
    FN = ((y_pred == 0) & (y == 1)).sum().float()
    FP = ((y_pred == 1) & (y == 0)).sum().float()
    p = TP / (TP + FP).clamp(min=1e-8)
    r = TP / (TP + FN).clamp(min=1e-8)
    F1 = 2 * r * p / (r + p).clamp(min=1e-8)
    acc = (TP + TN) / (TP + TN + FP + FN).clamp(min=1e-8)
    return acc, p, r, F1


def train(train_iter, dev_iter, model, args):
    if args.cuda:
        model.cuda()
    # for param in model.parameters():
    #     print(param)

    def adjust_learning_rate(optimizer, learning_rate, epoch):
        lr = learning_rate * (0.1 ** (epoch // 10))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        return optimizer

    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=[0.9, 0.999], eps=1e-8, weight_decay=0)
    elif args.optimizer == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=0.01)
    else:
        raise Exception('For other optimizers, please add it yourself. supported ones are: SGD and Adam.')

    F1_best = 0.0
    last_improved_step = 0
    model.train()
    steps = 0
    for epoch in range(1, args.epochs+1):
        for batch in train_iter:
            optimizer = adjust_learning_rate(optimizer, args.lr, epoch)
            x1, x2, y = batch
            y = torch.squeeze(y, 1).float()   # [[1], [1], [0]...] to [1, 1, 0, ...]

            # if args.cuda:
            #     x1, x2, y = Variable(x1).cuda(), Variable(x2).cuda(), Variable(y).cuda()
            # else:
            #     x1, x2, y = Variable(x1), Variable(x2), Variable(y)
            optimizer.zero_grad()
            _, _, score = model(x1, x2)

            # print('out1', out1.dtype)
            # print('target vector', y.dtype)

            # loss_function = nn.CrossEntropyLoss()
            # loss = loss_function(output, Variable(train_labels))
            # criterion = nn.CosineEmbeddingLoss(margin=0, size_average=True, reduce=False)
            # loss = criterion(out1, out2, (2 * y - 1))  # cast y to {1, -1} and float type
            # criterion = ContrastiveLoss()
            # loss = criterion(y, sim)

            # loss = F.cross_entropy(sim, y)
            loss = F.binary_cross_entropy_with_logits(score, y)
            loss.backward()
            optimizer.step()
            steps += 1

            if steps % args.log_interval == 0:
                # _, pred = torch.max(sim.data, 1)
                print('model sim and label tuples:')
                for i, j in zip(score, y):
                    print(i.item(), j.item())

                pred = score.data >= 0.5
                acc, p, r, f1 = metrics(y, pred)
                print('TRAIN[steps={}] loss={:.6f} acc={:.3f} P={:.3f} R={:.3f} F1={:.6f}'.format(
                    steps, loss.item(), acc, p, r, f1))
            if steps % args.test_interval == 0:
                loss, acc, p, r, f1 = eval(dev_iter, model)

                if f1 > F1_best:
                    F1_best = f1
                    last_improved_step = steps
                    if F1_best > 0.5:
                        save_prefix = os.path.join(args.save_dir, 'snapshot')
                        save_path = '{}_steps{}.pt'.format(save_prefix, steps)
                        torch.save(model, save_path)
                    improved_token = '*'
                else:
                    improved_token = ''
                print('DEV[steps={}] loss={:.6f} acc={:.3f} P={:.3f} R={:.3f} F1={:.6f} {}'.format(
                    steps, loss, acc, p, r, f1, improved_token))

            if steps % args.save_interval == 0:
                if not os.path.isdir(args.save_dir):
                    os.makedirs(args.save_dir)
                save_prefix = os.path.join(args.save_dir, 'snapshot')
                save_path = '{}_steps{}.pt'.format(save_prefix, steps)
                torch.save(model, save_path)

            if steps - last_improved_step > 2000:  # 2000 steps
                print("No improvement for a long time, early-stopping at best F1={}".format(F1_best))
                break


def eval(data_iter, model):
    loss_tot, y_list, y_pred_list = 0, [], []
    model.eval()
    for x1, x2, y in data_iter:
        # if args.cuda:
        #     x1, x2, y = Variable(x1).cuda(), Variable(x2).cuda(), Variable(y).cuda()
        # else:
        #     x1, x2, y = Variable(x1), Variable(x2), Variable(y)
        out1, out2, sim = model(x1, x2)
        # loss = F.cross_entropy(output, y, size_average=False)
        criterion = nn.CosineEmbeddingLoss()
        loss = criterion(out1, out2, (2*y-1).float())
        loss_tot += loss.item()  # 0-dim scaler
        y_pred = sim.data >= 0.5
        y_pred_list.append(y_pred)
        y_list.append(y)
    y_pred = torch.cat(y_pred_list, 0)
    y = torch.cat(y_list, 0)
    acc, p, r, f1 = metrics(y, y_pred)
    size = len(data_iter.dataset)
    loss_avg = loss_tot / float(size)
    model.train()
    return loss_avg, acc, p, r, f1


def predict(text, model, text_field, label_feild, cuda_flag):
    assert isinstance(text, str)
    model.eval()
    # text = text_field.tokenize(text)
    text = text_field.preprocess(text)
    text = [[text_field.vocab.stoi[x] for x in text]]
    x = text_field.tensor_type(text)
    x = autograd.Variable(x, volatile=True)
    if cuda_flag:
        x = x.cuda()
    print(x)
    output = model(x)
    _, predicted = torch.max(output, 1)
    return label_feild.vocab.itos[predicted.data[0][0]+1]


if __name__ == '__main__':
    # parse the arguments
    args = get_args()
    print(args)

    # Set the random seed manually for reproducibility.
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        if not args.cuda:
            print("WARNING: You have a CUDA device, so you should probably run with --cuda")
        else:
            torch.cuda.manual_seed(args.seed)

    # Load Dictionary
    assert os.path.exists(args.train_data)
    assert os.path.exists(args.val_data)
    print('Begin to load the dictionary.')
    dictionary = Dictionary('../data/atec_nlp_sim_train.csv')

    args.vocab_size = len(dictionary)

    best_val_loss = None
    best_f1 = None
    n_token = len(dictionary)

    embedding_net = EmbeddingCNN(args)
    print("embedding_net: {}".format(embedding_net))
    model = SiameseNet(embedding_net)
    print(model)

    print('Begin to load data.')
    train_data = MyDataset(args.train_data, args.sequence_length, dictionary.word2idx, args.char_model)
    val_data = MyDataset(args.val_data, args.sequence_length, dictionary.word2idx, args.char_model)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=16)
    val_loader = DataLoader(val_data, batch_size=1, shuffle=False)
    try:
        for epoch in range(args.epochs):
            train(train_loader, val_loader, model, args)
    except KeyboardInterrupt:
        print('-' * 89)
        print('Exit from training early.')

