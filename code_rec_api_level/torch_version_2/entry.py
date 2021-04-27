import time

import numpy as np
import torch
from code_rec_api_level.torch_version_2.model import DenseGGNNModel
from code_rec_api_level.torch_version_2.utils import ThreadedIterator
from code_rec_api_level.torch_version_2.dataset import JavaCodeDataset
import os
import gc

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

param = dict()
param['num_edge_types'] = 8
param['hidden_size'] = 30  # 300
param['graph_state_keep_prob'] = 0.75
param['edge_weight_dropout_keep_prob'] = 0.75
param['out_layer_dropout_keep_prob'] = 0.75
param['num_time_steps'] = 4
param['embedding_size'] = 30  # 300
param['api_vocab_size'] = 39070  # 21505
param['var_vocab_size'] = 12776  # 7142
param['whole_vocab_size'] = 39070  # 22821
param['softmax_size'] = 80  # 800
param['max_var_len'] = 10

opt = dict()
opt['num_epochs'] = 50
opt['learning_rate'] = 0.005
opt['momentum'] = 0.9
opt['decay_rate'] = 1.0  # 0.96
opt['clipping_value'] = 1.0
opt['batch_size'] = 256  # 256
opt['tie_fwd_bkwd'] = False
opt['patience'] = 5
opt['cuda'] = False
opt['manual_seed'] = 983

opt['train_file_folder'] = '../../data2'
opt['validation_file_folder'] = '../../data2'

if opt['cuda']:
    torch.cuda.manual_seed_all(opt['manual_seed'])

def prepare_file_list(file_folder):
    json_file_list = []
    for root, dirs, files in os.walk(file_folder):
        for file in files:
            if file.endswith(".json"):
                p = os.path.join(root, file)
                json_file_list.append(p)
    return json_file_list


def evaluate(model, epoch_idx, dataset_path_list, criterion):
    epoch_loss = 0
    epoch_accuracies = []
    epoch_graph_sum = 0

    start_time = time.time()

    model.eval()
    for dataset_path in dataset_path_list:
        data_load_start_time = time.time()
        java_code_dataset = JavaCodeDataset(file_name=dataset_path, is_training=True, param=param, opt=opt)
        data_read_time = time.time() - data_load_start_time
        batch_iterator = ThreadedIterator(java_code_dataset.make_minibatch_iterator(), max_queue_size=5)

        # for i in range(len(num_vertices)):
        for step, batch_data in enumerate(batch_iterator):
            input_orders = batch_data['input_orders']  # [b, v]
            node_mask = batch_data['node_mask']  # [b, v, h]
            variable_orders = batch_data['variable_orders']  # [b, max_var_len]
            variable_mask = batch_data['variable_masks']  # [b, max_var_len, e_dim]
            adjacency_matrix = batch_data['adj_mat']  # [b, max_var_len, e_dim]
            ground_truth = batch_data['target_values']  # [b, e, v, v]
            num_vertices = batch_data['num_vertices']  # v
            num_graphs = batch_data['num_graphs']  # b
            epoch_graph_sum += num_graphs

            if opt['cuda']:
                input_orders = input_orders.cuda()
                node_mask = node_mask.cuda()
                variable_orders = variable_orders.cuda()
                variable_mask = variable_mask.cuda()
                adjacency_matrix = adjacency_matrix.cuda()
                ground_truth = ground_truth.cuda()

            # [b, whole_vocab_size]
            computed_values = model(input_orders, node_mask, variable_orders, variable_mask, adjacency_matrix,
                                    num_vertices)

            loss = criterion(computed_values, ground_truth)

            predict = torch.argmax(computed_values, dim=1)

            correct_prediction = torch.eq(predict, ground_truth)

            # torch.int64 to torch.float32
            correct_count = torch.sum(correct_prediction.type(torch.LongTensor))
            accuracy = torch.mean(correct_prediction.type(torch.FloatTensor))

            epoch_accuracies.append(np.array(accuracy.data) * num_graphs)
            x_batch_loss = loss.data
            epoch_loss += x_batch_loss * num_graphs

        del java_code_dataset
        del batch_iterator
        gc.collect()

    epoch_end_time = time.time() - start_time
    m, s = divmod(epoch_end_time, 60)
    h, m = divmod(m, 60)
    time_str = "%02d:%02d:%02d" % (h, m, s)

    if epoch_graph_sum > 0:
        epoch_accuracies = np.sum(epoch_accuracies, axis=0) / epoch_graph_sum
        epoch_loss = epoch_loss / epoch_graph_sum
        print('[EVALUATING] epoch [%d/%d] | acc %.5f | loss: %.5f | time cost: %s | train total graph: %d\n' % (
            epoch_idx + 1, opt['num_epochs'], epoch_accuracies, epoch_loss, time_str, epoch_graph_sum))

    # return epoch_loss, epoch_accuracies, time_str
    return epoch_accuracies


def train(model, epoch_idx, dataset_path_list, criterion, optimizer, scheduler):
    epoch_loss = 0
    epoch_accuracies = []
    epoch_graph_sum = 0

    start_time = time.time()

    model.train()

    batch_num = 0
    batch_acc_sum = 0
    batch_loss_sum = 0

    for dataset_path in dataset_path_list:

        data_load_start_time = time.time()
        java_code_dataset = JavaCodeDataset(file_name=dataset_path, is_training=True, param=param, opt=opt)
        data_read_time = time.time() - data_load_start_time
        batch_iterator = ThreadedIterator(java_code_dataset.make_minibatch_iterator(), max_queue_size=5)

        # for i in range(len(num_vertices)):
        for step, batch_data in enumerate(batch_iterator):
            input_orders = batch_data['input_orders']  # [b, v]
            node_mask = batch_data['node_mask']  # [b, v, h]
            variable_orders = batch_data['variable_orders']  # [b, max_var_len]
            variable_mask = batch_data['variable_masks']  # [b, max_var_len, e_dim]
            adjacency_matrix = batch_data['adj_mat']  # [b, max_var_len, e_dim]
            ground_truth = batch_data['target_values']  # [b, e, v, v]
            num_vertices = batch_data['num_vertices']  # v
            num_graphs = batch_data['num_graphs']  # b
            epoch_graph_sum += num_graphs

            model.zero_grad()

            if opt['cuda']:
                input_orders = input_orders.cuda()
                node_mask = node_mask.cuda()
                variable_orders = variable_orders.cuda()
                variable_mask = variable_mask.cuda()
                adjacency_matrix = adjacency_matrix.cuda()
                ground_truth = ground_truth.cuda()

            # [b, whole_vocab_size]
            computed_values = model(input_orders, node_mask, variable_orders, variable_mask, adjacency_matrix,
                                    num_vertices)

            loss = criterion(computed_values, ground_truth)

            predict = torch.argmax(computed_values, dim=1)

            correct_prediction = torch.eq(predict, ground_truth)

            # torch.int64 to torch.float32
            correct_count = torch.sum(correct_prediction.type(torch.LongTensor))
            accuracy = torch.mean(correct_prediction.type(torch.FloatTensor))

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=opt['clipping_value'])
            optimizer.step()
            scheduler.step()

            batch_num += 1
            batch_acc_sum += accuracy.data
            batch_loss_sum += loss.data

            epoch_accuracies.append(np.array(accuracy.data) * num_graphs)
            x_batch_loss = loss.data
            epoch_loss += x_batch_loss * num_graphs

            # if step % 20 == 0:
            print('Epoch [%d/%d] | batch: %d | acc: %.4f | loss: %.4f | correct count: [%d/%d]' % (
                epoch_idx + 1, opt['num_epochs'], batch_num, batch_acc_sum / batch_num, batch_loss_sum / batch_num,
                correct_count.data, num_graphs))

        del java_code_dataset
        del batch_iterator
        gc.collect()

    epoch_end_time = time.time() - start_time
    m, s = divmod(epoch_end_time, 60)
    h, m = divmod(m, 60)
    time_str = "%02d:%02d:%02d" % (h, m, s)

    if epoch_graph_sum > 0:
        epoch_accuracies = np.sum(epoch_accuracies, axis=0) / epoch_graph_sum
        epoch_loss = epoch_loss / epoch_graph_sum
        print('\n[TRAINING] epoch [%d/%d] | acc %.5f | loss: %.5f | time cost: %s | train total graph: %d\n' % (
            epoch_idx + 1, opt['num_epochs'], epoch_accuracies, epoch_loss, time_str, epoch_graph_sum))


def run(train_file_list, valid_file_list):
    print('Building Model...')
    model = DenseGGNNModel(param)
    # print(list(model.parameters()))

    # reduction='sum' or 'mean', Default: 'mean'
    criterion = torch.nn.CrossEntropyLoss(reduction='sum')
    optimizer = torch.optim.SGD(model.parameters(), lr=opt['learning_rate'], momentum=opt['momentum'])
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=opt['decay_rate'], last_epoch=-1)

    if opt['cuda']:
        model.cuda()
        criterion.cuda()

    best_acc = 0.0  # best accuracy has been achieved
    best_eval_acc_epoch_idx = 0

    print('Start Training...')
    for epoch_idx in range(opt['num_epochs']):

        train(model, epoch_idx, train_file_list, criterion, optimizer, scheduler)
        eval_acc = evaluate(model, epoch_idx, valid_file_list, criterion)

        if eval_acc > best_acc:
            best_acc = eval_acc
            best_eval_acc_epoch_idx = epoch_idx
            torch.save(model.state_dict(), '../../save/api_rec_model.pth')
        elif epoch_idx - best_eval_acc_epoch_idx >= opt['patience']:
            print('Stop training after %d epochs without improvement on validation accuracy.' % opt['patience'])


if __name__ == '__main__':

    train_file_list = prepare_file_list(opt['train_file_folder'])
    valid_file_list = prepare_file_list(opt['validation_file_folder'])

    run(train_file_list, valid_file_list)
