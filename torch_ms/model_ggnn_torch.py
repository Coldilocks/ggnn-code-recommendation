import torch
import torch.nn as nn


class AttrProxy(object):
    def __init__(self, module, prefix):
        self.module = module
        self.prefix = prefix

    def __getitem__(self, i):
        return getattr(self.module, self.prefix + str(i))


class DenseGGNNModel(nn.Module):
    def __init__(self, param):
        super(DenseGGNNModel, self).__init__()

        # e
        self.num_edge_types = param['num_edge_types']
        # h
        self.hidden_size = param['hidden_size']
        self.graph_state_keep_prob = param['graph_state_keep_prob']
        self.edge_weight_dropout_keep_prob = param['edge_weight_dropout_keep_prob']
        self.out_layer_dropout_keep_prob = param['out_layer_dropout_keep_prob']
        self.num_time_steps = param['num_time_steps']

        for i in range(self.num_edge_types):
            edge_fc = nn.Sequential(
                nn.Linear(self.hidden_size, self.hidden_size),
                nn.Dropout(1 - self.edge_weight_dropout_keep_prob)
            )
            nn.init.xavier_uniform_(edge_fc[0].weight)
            nn.init.zeros_(edge_fc[0].bias)
            self.add_module("edge_{}".format(i), edge_fc)

        self.edge_fcs = AttrProxy(self, "edge_")

        self.gru_cell = nn.GRUCell(input_size=self.hidden_size, hidden_size=self.hidden_size, bias=True)

        self.regression_gate = nn.Sequential(
            nn.Linear(2 * self.hidden_size, 1),
            nn.Dropout(1 - self.out_layer_dropout_keep_prob),
            nn.ReLU(),
            nn.Sigmoid()
        )

        self.regression_transform = nn.Sequential(
            nn.Linear(self.hidden_size, 1),
            nn.Dropout(1 - self.out_layer_dropout_keep_prob),
            nn.ReLU()
        )

    def forward(self, initial_node_representation, adjacency_matrix, node_mask, num_vertices):
        # initial_node_representation: [b, v, h]
        # adjacency_matrix: [b, e, v, v] -> [e, b, v, v]
        adj_matrix = torch.transpose(adjacency_matrix, 0, 1)
        # node_mask: [b, v]

        # target_values
        # target_mask
        v = num_vertices
        h_dim = self.hidden_size

        # [b, v, h] -> [b*v, h]
        h = initial_node_representation
        h = h.view([-1, h_dim])

        # 1.compute final node representations
        for step in range(self.num_time_steps):
            for edge_type in range(self.num_edge_types):
                m = self.edge_fcs[edge_type](h)     # [b*v, h]
                m = m.view([-1, v, h_dim])  # [b, v, h]
                # acts: [b, v, v] x [b, v, h] -> [b, v, h]
                if edge_type == 0:
                    acts = torch.matmul(adj_matrix[edge_type], m)
                else:
                    acts += torch.matmul(adj_matrix[edge_type], m)
            acts = acts.view([-1, h_dim])   # [b*v, h]
            h = self.gru_cell(acts, h)   # [b*v, h]

        # final node representations
        # last_h = torch.reshape(h, [-1, v, h_dim])   # [b, v, h]
        last_h = h.view([-1, v, h_dim])     # [b, v, h]

        # 2. gated regression
        gate_input = torch.cat((last_h, initial_node_representation), dim=2)    # [b, v, 2*h]
        gate_input = gate_input.view([-1, 2 * h_dim])   # [b, v, 2*h] -> [b*v, 2h]
        last_h = last_h.view([-1, h_dim])     # [b, v, h] -> [b*v, h]
        gated_outputs = self.regression_gate(gate_input) * self.regression_transform(last_h)    # [b*v, 1]
        gated_outputs = gated_outputs.view([-1, v])   # [b*v, 1] -> [b, v]
        masked_gated_outputs = gated_outputs * node_mask    # [b, v] * [b, v] -> [b, v]
        output = torch.sum(masked_gated_outputs, dim=1)     # [b]

        return output







