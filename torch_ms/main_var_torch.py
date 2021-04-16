import torch
from torch_ms.model_var_torch import DenseGGNNModel
from torch_ms.utils import ThreadedIterator
from torch_ms.dataset import JavaCodeDataset
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

if __name__ == '__main__':
    param = dict()
    param['num_edge_types'] = 8
    param['hidden_size'] = 300  # 300
    param['graph_state_keep_prob'] = 0.75
    param['edge_weight_dropout_keep_prob'] = 0.75
    param['out_layer_dropout_keep_prob'] = 0.75
    param['num_time_steps'] = 4
    param['embedding_size'] = 300   # 300
    param['api_vocab_size'] = 39070     # 21505
    param['var_vocab_size'] = 12776  # 7142
    param['whole_vocab_size'] = 39070   # 22821
    param['softmax_size'] = 800  # 800
    param['max_var_len'] = 10

    opt = dict()
    opt['num_epochs'] = 100
    opt['learning_rate'] = 0.001
    opt['momentum'] = 0.9
    opt['decay_rate'] = 1.0     # 0.96
    opt['clipping_value'] = 1.0
    opt['batch_size'] = 256   # 256
    opt['tie_fwd_bkwd'] = False

    SMALL_NUMBER = 1e-7


    dataset_path = '../data/small_input.json'

    model = DenseGGNNModel(param)
    # print(model)

    num_vertices = [5, 6, 7, 8, 10]

    # todo: reduction='sum' or 'mean', Default: 'mean'
    criterion = torch.nn.CrossEntropyLoss(reduction='mean')
    optimizer = torch.optim.SGD(model.parameters(), lr=opt['learning_rate'], momentum=opt['momentum'])
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=opt['decay_rate'], last_epoch=-1)

    for epoch in range(opt['num_epochs']):

        java_code_dataset = JavaCodeDataset(file_name=dataset_path, is_training=True, param=param, opt=opt)
        # batch_iterator = ThreadedIterator(java_code_dataset.make_minibatch_iterator(), max_queue_size=5)
        batch_iterator = java_code_dataset.make_minibatch_iterator()

        # for i in range(len(num_vertices)):
        for step, batch_data in enumerate(batch_iterator):

            input_orders = batch_data['input_orders']   # [b, v]
            node_mask = batch_data['node_mask']     # [b, v, h]
            variable_orders = batch_data['variable_orders']     # [b, max_var_len]
            variable_mask = batch_data['variable_masks']        # [b, max_var_len, e_dim]
            adjacency_matrix = batch_data['adj_mat']        # [b, max_var_len, e_dim]
            ground_truth = batch_data['target_values']         # [b, e, v, v]
            num_vertices = batch_data['num_vertices']
            batch_size = batch_data['num_graphs']

            model.train()

            model.zero_grad()

            # [b, whole_vocab_size]
            computed_values = model(input_orders, node_mask, variable_orders, variable_mask, adjacency_matrix, num_vertices)

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

            print('epoch [%d / %d] batch %d: acc %.4f, loss %.4f, correct %d / %d' % (epoch + 1, opt['num_epochs'], step + 1, accuracy.data, loss.item(), correct_count.data, batch_size))





