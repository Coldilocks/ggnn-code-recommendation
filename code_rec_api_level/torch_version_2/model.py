import torch
import torch.nn as nn


class AttrProxy(object):
    def __init__(self, module, prefix):
        self.module = module
        self.prefix = prefix

    def __getitem__(self, i):
        return getattr(self.module, self.prefix + str(i))


class VarModel(nn.Module):
    def __init__(self, param):
        super(VarModel, self).__init__()
        self.var_vocab_size = param['var_vocab_size']
        # e_dim
        self.embedding_size = param['embedding_size']
        self.out_layer_dropout_keep_prob = param['out_layer_dropout_keep_prob']

        self.var_embedding = nn.Embedding(self.var_vocab_size, self.embedding_size, sparse=False)

        self.var_fcs = nn.Sequential(
            nn.Linear(self.embedding_size, self.embedding_size),
            nn.Tanh(),
            nn.Dropout(1 - self.out_layer_dropout_keep_prob),
            nn.Linear(self.embedding_size, self.embedding_size),
            nn.Tanh(),
            nn.Dropout(1 - self.out_layer_dropout_keep_prob),
            nn.Linear(self.embedding_size, self.embedding_size),
            nn.Tanh(),
            nn.Dropout(1 - self.out_layer_dropout_keep_prob)
        )

    def forward(self, variable_orders, variable_mask):
        """
        :param variable_orders: [b, max_var_len]
        :param variable_mask: [b, max_var_len, e_dim], max_var_len = 10
        :return:
        """
        vars = self.var_embedding(variable_orders)  # [b, max_vae_len] -> [b, max_var_len, e_dim]
        vars = self.var_fcs(vars)
        vars = vars * variable_mask
        return vars


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
        # e_dim
        self.embedding_size = param['embedding_size']
        self.api_vocab_size = param['api_vocab_size']
        self.softmax_size = param['softmax_size']   # 800
        self.whole_vocab_size = param['whole_vocab_size']

        self.api_embedding = nn.Embedding(self.api_vocab_size, self.embedding_size, sparse=False)

        self.var_model = VarModel(param)

        for i in range(self.num_edge_types):
            edge_fc = nn.Sequential(
                nn.Linear(self.hidden_size, self.hidden_size),
                nn.Dropout(1 - self.edge_weight_dropout_keep_prob)
            )
            nn.init.xavier_uniform_(edge_fc[0].weight)
            nn.init.zeros_(edge_fc[0].bias)
            self.add_module("edge_{}".format(i), edge_fc)

        self.edge_fcs = AttrProxy(self, "edge_")

        # gru_cell->dropoutwrapper(graph_state_keep_prob): h_dim
        # input_size: [b, h]
        self.gru_cell = nn.GRUCell(input_size=self.hidden_size, hidden_size=self.hidden_size, bias=True)

        self.regression_gate = nn.Sequential(
            nn.Linear(2 * self.hidden_size, self.hidden_size),
            nn.Dropout(1 - self.out_layer_dropout_keep_prob),
            nn.ReLU(),
            nn.Sigmoid()
        )
        nn.init.xavier_uniform_(self.regression_gate[0].weight)
        nn.init.zeros_(self.regression_gate[0].bias)

        self.regression_transform = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.Dropout(1 - self.out_layer_dropout_keep_prob),
            nn.ReLU()
        )
        nn.init.xavier_uniform_(self.regression_transform[0].weight)
        nn.init.zeros_(self.regression_transform[0].bias)

        self.softmax_layer = nn.Sequential(
            nn.Linear(2 * self.hidden_size, self.softmax_size),
            nn.Tanh(),
            nn.Dropout(1 - self.out_layer_dropout_keep_prob),
            nn.Linear(self.softmax_size, self.whole_vocab_size)
        )
        nn.init.xavier_uniform_(self.softmax_layer[3].weight)
        nn.init.zeros_(self.softmax_layer[3].bias)

    def forward(self, input_orders, node_mask, variable_orders, variable_mask, adjacency_matrix, num_vertices):
        """
        :param input_orders: [b, v]
        :param node_mask: [b, v, h]
        :param variable_orders: [b, max_var_len]
        :param variable_mask: [b, max_var_len, e_dim]
        :param adjacency_matrix: [b, e, v, v]
        :param num_vertices: v
        :return:
        """

        # [b, e, v, v] -> [e, b, v, v]
        adj_matrix = torch.transpose(adjacency_matrix, 0, 1)

        v = num_vertices
        h_dim = self.hidden_size

        # [b, v] -> [b, v, h]
        order_embed = self.api_embedding(input_orders)
        # [b, v, h] -> [b*v, h]
        h = order_embed.view([-1, h_dim])

        # 1.compute final node representations
        for step in range(self.num_time_steps):
            for edge_type in range(self.num_edge_types):
                m = self.edge_fcs[edge_type](h)  # [b*v, h]
                m = m.view([-1, v, h_dim])  # [b, v, h]
                # acts: [b, v, v] x [b, v, h] -> [b, v, h]
                if edge_type == 0:
                    acts = torch.matmul(adj_matrix[edge_type], m)
                else:
                    acts += torch.matmul(adj_matrix[edge_type], m)
            acts = acts.view([-1, h_dim])  # [b*v, h]
            h = self.gru_cell(acts, h)  # [b*v, h]

        # final node representations
        last_h = h.view([-1, v, h_dim])  # [b, v, h]

        # 2. gated regression
        gate_input = torch.cat((last_h, order_embed), dim=2)  # [b, v, 2*h]
        gate_input = gate_input.view([-1, 2 * h_dim])  # [b, v, 2*h] -> [b*v, 2h]
        last_h = last_h.view([-1, h_dim])  # [b, v, h] -> [b*v, h]
        gated_outputs = self.regression_gate(gate_input) * self.regression_transform(last_h)  # [b*v, h]
        gated_outputs = gated_outputs.view([-1, v, h_dim])  # [b*v, h] -> [b, v, h]
        masked_gated_outputs = gated_outputs * node_mask  # [b, v, h] * [b, v, h] -> [b, v, h]

        graph_vector = torch.sum(masked_gated_outputs, dim=1)    # [b, v, h] -> [b, h]
        # [b, max_var_len] -> [b, max_var_len, e_dim]
        variable_vector = self.var_model(variable_orders, variable_mask)
        variable_vector = torch.sum(variable_vector, dim=1)     # [b, max_var_len, e_dim] -> [b, e_dim]
        # concat the graph vector and the variable vector
        inte_vectors = torch.cat((graph_vector, variable_vector), dim=1)    # [b, 2*h]

        # [b, 2*h] -> [b, whole_vocab_size]
        output = self.softmax_layer(inte_vectors)

        return output