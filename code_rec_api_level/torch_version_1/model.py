import torch
import torch.nn as nn


class AttrProxy(object):
    def __init__(self, module, prefix):
        self.module = module
        self.prefix = prefix

    def __getitem__(self, i):
        return getattr(self.module, self.prefix + str(i))


class MessagePassing(nn.Module):
    def __init__(self, hidden_dim, n_nodes, n_edges):
        super(MessagePassing, self).__init__()

        self.n_nodes = n_nodes
        self.n_edges = n_edges

        self.reset_gate = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.Sigmoid()
        )

        self.update_gate = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.Sigmoid()
        )

        self.transform = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.Tanh()
        )

    def forward(self, incoming_state, outgoing_state, last_state, A):
        """
        :param incoming_state: [b, n * e, h]
        :param outgoing_state: [b, n * e, h]
        :param last_state: [b, n, h]
        :param A: [b, n, e * n * 2]
        :return: h: [b, n, h]
        """
        # A_in & A_out: [b, n, e * n]
        A_in = A[:, :, :self.n_nodes * self.n_edges]
        A_out = A[:, :, self.n_nodes * self.n_edges:]

        # a_in & a_out: [b, n, h]
        a_in = torch.bmm(A_in, incoming_state)
        a_out = torch.bmm(A_out, outgoing_state)

        # a: [b, n, h * 3]
        a = torch.cat((a_in, a_out, last_state), dim=2)

        # update geate, z: [b, n, h]
        z = self.update_gate(a)
        # reset gate, r: [b, n, h]
        r = self.reset_gate(a)

        # r * last_state: [b, n, h]
        # a_new: [b, n, h * 3]
        a_new = torch.cat((a_in, a_out, r * last_state), dim=2)

        # h_: [b, n, h]
        h_ = self.transform(a_new)
        # h: [b, n, h]
        h = (1 - z) * last_state + z * h_

        return h


class GGNN(nn.Module):
    def __init__(self, opt):
        super(GGNN, self).__init__()
        # n
        self.n_nodes = opt.n_nodes
        # e
        self.n_edges = opt.n_edges
        self.n_steps = opt.n_steps
        # h
        self.hidden_dim = opt.hidden_dim
        # a
        self.annotation_dim = opt.annotation_dim
        self.ggnn_dropout_rate = opt.ggnn_dropout_rate

        for i in range(self.n_edges * 2):
            in_fc = nn.Linear(self.hidden_dim, self.hidden_dim)
            out_fc = nn.Linear(self.hidden_dim, self.hidden_dim)
            self.add_module("in_{}".format(i), in_fc)
            self.add_module("out_{}".format(i), out_fc)

        self.in_fcs = AttrProxy(self, "in_")
        self.out_fcs = AttrProxy(self, "out_")

        self.propagator = MessagePassing(self.hidden_dim, self.n_nodes, self.n_edges)

        # readout
        self.attention = nn.Sequential(
            nn.Linear(self.hidden_dim + self.annotation_dim, self.hidden_dim),
            nn.Sigmoid()
        )
        self.state_transform = nn.Sequential(
            nn.Linear(self.hidden_dim + self.annotation_dim, self.hidden_dim),
            nn.Tanh()
        )
        self.readout = nn.Tanh()

        self._initialization()

    def _initialization(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0.0, 0.02)
                m.bias.data.fill_(0)

    def forward(self, hidden_state, annotation, A):
        """
        :param hidden_state: [b, n, h]
        :param annotation: [b, n, h]
        :param A: [b, n, e * n * 2]    [batch_size, n_node, n_edge * n_node * 2]
        :return: [b, h]
        """
        for step in range(self.n_steps):
            incoming_states = []
            outgoing_states = []
            for i in range(self.n_edges):
                incoming_states.append(self.in_fcs[i](hidden_state))
                outgoing_states.append(self.out_fcs[i](hidden_state))
            # incoming_states: e * [b, n, h]
            # after stack: [e, b, n, h]
            # after transpose: [b, e, n, h]
            # after view: [b, n * e, h]
            incoming_states = torch.stack(incoming_states).transpose(0, 1).contiguous()
            incoming_states = incoming_states.view(-1, self.n_nodes * self.n_edges, self.hidden_dim)
            outgoing_states = torch.stack(outgoing_states).transpose(0, 1).contiguous()
            outgoing_states = outgoing_states.view(-1, self.n_nodes * self.n_edges, self.hidden_dim)

            # hidden_state: [b, n, h]
            hidden_state = self.propagator(incoming_states, outgoing_states, hidden_state, A)

        # join_state: [b, n, h]
        join_state = torch.cat((hidden_state, annotation), dim=2)
        # attention_score: [b, n, h]
        attention_score = self.attention(join_state)
        # tanh_state: [b, n, h]
        tanh_state = self.state_transform(join_state)
        # mul_state: [b, n, h]
        mul_state = attention_score * tanh_state
        # wighted_sum: [b, h]
        wighted_sum = torch.sum(mul_state, dim=1)
        # final_graph_state: [b, h]
        final_graph_state = self.readout(wighted_sum)

        return final_graph_state


class VariableNameModel(nn.Module):
    def __init__(self, opt):
        """
        :param opt:
        """
        super(VariableNameModel, self).__init__()
        # v: max number of variables in a graph
        self.n_vars = opt.n_vars
        self.vocab_size = opt.vocab_size
        # v_h
        self.var_embedding_dim = opt.var_embedding_dim
        self.var_drop_out_prob = opt.var_drop_out_prob

        self.vocab_embedding = nn.Embedding(self.vocab_size, self.var_embedding_dim, sparse=False)

        self.fc1 = nn.Sequential(
            nn.Linear(self.var_embedding_dim, self.var_embedding_dim),
            nn.Tanh(),
            nn.Dropout(self.var_drop_out_prob)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(self.var_embedding_dim, self.var_embedding_dim),
            nn.Tanh(),
            nn.Dropout(self.var_drop_out_prob)
        )
        self.fc3 = nn.Sequential(
            nn.Linear(self.var_embedding_dim, self.var_embedding_dim),
            nn.Tanh(),
            nn.Dropout(self.var_drop_out_prob)
        )

    def forward(self, var_orders):
        """
        :param var_orders: [b, v]
        :return: [b, v, v_h]
        """
        # vars_embedding: [b, v, v_h]
        vars_embedding = self.vocab_embedding(var_orders)
        vars_embedding = self.fc1(vars_embedding)
        vars_embedding = self.fc2(vars_embedding)
        vars_embedding = self.fc3(vars_embedding)
        return vars_embedding

class CodeRecModel(nn.Module):
    def __init__(self, opt):
        super(CodeRecModel, self).__init__()
        # n
        self.n_nodes = opt.n_nodes
        # e
        # self.n_edges = opt.n_edges
        # self.n_steps = opt.n_steps
        # h
        self.hidden_dim = opt.hidden_dim
        # a
        self.annotation_dim = opt.annotation_dim
        self.dropout_rate = opt.dropout_rate

        # v
        # self.n_vars = opt.n_vars
        self.vocab_size = opt.vocab_size
        # v_h
        self.var_embedding_dim = opt.var_embedding_dim

        # s: softmax dimension of the final layer
        self.softmax_size = opt.softmax_size

        # number of node types
        self.n_node_types = opt.n_node_types

        self.use_cuda =  opt.cuda

        self.ggnn_model = GGNN(opt)
        self.variable_model = VariableNameModel(opt)

        self.node_embedding = nn.Embedding(self.n_node_types, self.annotation_dim, sparse=False)

        self.output_layer = nn.Sequential(
            nn.Linear(self.hidden_dim + self.var_embedding_dim, self.n_node_types)
        )

    def forward(self, node_annotation, A, var_orders):
        """
        :param node_hidden_state: [b, n, h]
        :param node_annotation: [b, n] -> [b, n, h]
        :param A: [b, n, e * n * 2]
        :param var_orders: [b, v]
        :return: [b, vocab_size]
        """

        new_node_annotation = []
        for i in range(node_annotation.shape[0]):
            annotation_i = []
            for id in node_annotation[i]:
                if id.long() != 0:
                    type_idx = torch.LongTensor([int(id)])
                    if self.use_cuda:
                        type_idx = type_idx.cuda()
                    if self.use_cuda:
                        annotation_i.append(self.node_embedding(type_idx).view(self.annotation_dim).double().cuda())
                    else:
                        annotation_i.append(self.node_embedding(type_idx).view(self.annotation_dim).double())
                else:
                    if self.use_cuda:
                        annotation_i.append(torch.zeros(self.annotation_dim).double().cuda())
                    else:
                        annotation_i.append(torch.zeros(self.annotation_dim).double())
            annotation_i = torch.stack(annotation_i)  # [n_node, annotation_dim]
            new_node_annotation.append(annotation_i)
        node_annotation = torch.stack(new_node_annotation)

        if self.use_cuda:
            node_annotation = node_annotation.cuda()

        assert self.hidden_dim >= self.annotation_dim
        if self.hidden_dim > self.annotation_dim:
            node_hidden_state_padding = torch.zeros(len(node_annotation), self.n_nodes, self.hidden_dim - self.annotation_dim).double()
            node_hidden_state = torch.cat((node_annotation.double(), node_hidden_state_padding), 2)
        else:
            node_hidden_state = node_annotation.clone()

        # graph_vector: [b, h]
        graph_vector = self.ggnn_model(node_hidden_state, node_annotation, A)
        # var_vector: [b, v, v_h]
        var_vector = self.variable_model(var_orders)
        # var_vector: [b, v_h]
        var_vector = torch.sum(var_vector, dim=1)
        # join_vector: [b, h + v_h]
        join_vector = torch.cat((graph_vector, var_vector), dim=1)
        # output: [b, vocab_size]
        output = self.output_layer(join_vector)

        return output
