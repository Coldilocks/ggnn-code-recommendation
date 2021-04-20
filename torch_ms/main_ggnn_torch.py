import torch
from torch_ms.model_ggnn_torch import DenseGGNNModel
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

if __name__ == '__main__':
    param = dict()
    param['num_edge_types'] = 4
    param['hidden_size'] = 30 # 300
    param['graph_state_keep_prob'] = 0.75
    param['edge_weight_dropout_keep_prob'] = 0.75
    param['out_layer_dropout_keep_prob'] = 0.75
    param['num_time_steps'] = 4

    opt = dict()
    opt['num_epochs'] = 10
    opt['learning_rate'] = 0.001
    opt['clamp_gradient_norm'] = 1.0
    opt['batch_size'] = 8 # 256

    SMALL_NUMBER = 1e-7

    model = DenseGGNNModel(param)
    # print(model)

    num_vertices = [5, 6]

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=opt['learning_rate'])

    for epoch in range(opt['num_epochs']):
        for i in range(2):
            initial_node_representation = torch.rand([opt['batch_size'], num_vertices[i], param['hidden_size']])   # [b, v, h]
            adjacency_matrix = torch.ones([opt['batch_size'], param['num_edge_types'], num_vertices[i], num_vertices[i]]) # [b, e, v, v]
            node_mask = torch.ones([opt['batch_size'], num_vertices[i]])       # [b, v]
            target_values = torch.ones([opt['batch_size']])   # [b]
            target_mask = torch.ones([opt['batch_size']])     # [b]

            model.train()

            model.zero_grad()

            computed_values = model(initial_node_representation, adjacency_matrix, node_mask, num_vertices[i])

            diff = computed_values - target_values
            task_target_mask = target_mask
            task_target_num = torch.sum(task_target_mask) + SMALL_NUMBER
            diff = diff * task_target_mask
            acc = torch.sum(torch.abs(diff)) / task_target_num

            loss = criterion(computed_values, target_values)

            loss.backward()
            optimizer.step()

            print('epoch [%d / %d] batch [%d / %d]: acc %.4f, loss %.4f' % (epoch + 1, opt['num_epochs'], i + 1, 2, loss.data, acc))





