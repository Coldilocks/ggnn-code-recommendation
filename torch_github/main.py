import argparse

import torch
from model import CodeRecModel
from dataset import JavaCodeDataset
from dataloader import JavaCodeDataloader
from train import train

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

parser = argparse.ArgumentParser()
parser.add_argument('--workers', type=int, help='number of data loading workers', default=0)
parser.add_argument('--batch_size', type=int, default=128, help='input batch size')

# 1.Code Rec Model
parser.add_argument('--hidden_dim', type=int, default=300, help='GGNN hidden state size')
parser.add_argument('--dropout_rate', type=float, default=0.25, help='probability of dropout')
parser.add_argument('--n_node_types', type=int, default=39070, help='node types(API Vocabulary)')
parser.add_argument('--vocab_size', type=int, default=12776, help='vocab size')
parser.add_argument('--var_embedding_dim', type=int, default=300, help='variable embedding size')
parser.add_argument('--softmax_size', type=int, default=800, help='softmax layer size')

# 2.GGNN Model
# opt.n_nodes
parser.add_argument('--n_edges', type=int, default=4, help='edge number')
parser.add_argument('--n_steps', type=int, default=5, help='propogation steps number of GGNN')
# hidden_dim
parser.add_argument('--annotation_dim', type=int, default=300, help='node annotation size')
parser.add_argument('--ggnn_dropout_rate', type=float, default=0.0, help='probability of ggnn dropout')

# 3.Propagator
# n_nodes
# n_edges
# hidden_dim

# 4.Variable Name Model
# opt.n_vars
# vocab_size
# var_embedding_dim
parser.add_argument('--var_drop_out_prob', type=float, default=0.0, help='probability of variable model dropout')


parser.add_argument('--niter', type=int, default=50, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.005, help='learning rate')
parser.add_argument('--cuda', action='store_true', default=False, help='enables cuda')
# parser.add_argument('--use_bias', action='store_true', help='enables bias for edges', default=True)
parser.add_argument('--verbal', action='store_true', help='print training info or not', default=True)
parser.add_argument('--manualSeed', type=int, help='manual seed', default=983)

opt = parser.parse_args()

opt.dataroot = 'data/small_input.json'


def main(opt):
    train_dataset = JavaCodeDataset(opt.dataroot, opt.n_edges, True)
    train_dataloader = JavaCodeDataloader(train_dataset, batch_size=opt.num_graphs,
                                          shuffle=True, num_workers=opt.workers)

    # test_dataset = JavaCodeDataset(opt.dataroot, opt.n_edges, False)
    # test_dataloader = JavaCodeDataloader(test_dataset, batch_size=opt.batch_size,
    #                                       shuffle=False, num_workers=opt.workers)

    opt.n_nodes = train_dataset.n_nodes
    opt.n_vars = train_dataset.n_vars

    print('data load finish!')

    model = CodeRecModel(opt)
    model.double()
    criterion = torch.nn.CrossEntropyLoss()
    # criterion = torch.nn.NLLLoss()

    if opt.cuda:
        model.cuda()
        criterion.cuda()
        print('using cuda')

    optimizer = torch.optim.SGD(model.parameters(), lr=opt.lr)

    best_acc = 0.0      # best accuracy has been achieved
    num_of_dec = 0      # number of epochs have a decline of accuracy, used for early stop
    acc_last_iter = 0.0     # accuracy of the last iteration
    for epoch in range(0, opt.niter):
        # if num_of_dec >= 15:
        #     print("Early stop! The accuracy has been dropped for 15 iterations!")
        #     break
        train(epoch, train_dataloader, model, criterion, optimizer, opt)
        # correct = test(test_dataloader, model, criterion, opt)
        # acc = float(correct) / float(len(test_dataset))
        # if acc > best_acc:
        #     best_acc = acc
        #     print("Best accuracy by far: ", best_acc)
        #     torch.save(model, 'save/model.pth')
        # if acc >= best_acc:
        #     num_of_dec = 0
        # else:
        #     num_of_dec += 1
        # print("The best accuracy achieved by far: ", best_acc)


if __name__ == '__main__':
    main(opt)
