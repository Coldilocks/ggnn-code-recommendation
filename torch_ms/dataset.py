import json
import numpy as np
from collections import defaultdict

import torch


class JavaCodeDataset():
    def __init__(self, file_name, is_training, param, opt):
        self.max_num_vertices = 0
        self.num_edge_types = 8
        self.unk_id = 7141
        self.sample_ratios = None
        self.is_training = is_training

        self.tie_fwd_bkwd = opt['tie_fwd_bkwd']
        self.max_var_len = param['max_var_len']
        self.hidden_size = param['hidden_size']
        self.embedding_size = param['embedding_size']
        self.batch_size = opt['batch_size']

        # node_num_2_graph_dict, node_sizes, node_num_at_batch
        # self.bucketed, self.bucket_sizes, self.bucket_at_step
        self.node_num_2_graph_dict, self.node_sizes, self.node_num_at_batch = self.load_graphs_from_file(file_name)


    # def __getitem__(self, index):
    #     return
    # 
    # def __len__(self):
    #     return

    def make_minibatch_iterator(self):
        # node_num_2_graph_dict: node_num -> [graph1, graph2, graph3]
        # node_sizes: [1,3,5]
        # node_num_at_batch: [1,1,1,1,3,3,3,5,5,5,5,5]

        # shuffle training data
        if self.is_training:
            # 打乱需要执行的顺序
            # [1,1,1,1,3,3,3,5,5,5,5,5] -> [3, 3, 5, 5, 3, 5, 1, 5, 5, 1, 1, 1]
            np.random.shuffle(self.node_num_at_batch)
            for _, graph_data_of_node_num in self.node_num_2_graph_dict.items():
                # 打乱指定bucketed中同长度的数据
                # {1->[graph1, graph2, graph3], 3->[graph1, graph2, graph3]}
                #       -> {1->[graph3, graph1, graph2], 3->[graph3, graph2, graph1]}
                np.random.shuffle(graph_data_of_node_num)

        bucket_counters = defaultdict(int)

        for batch_step in range(len(self.node_num_at_batch)):
            # 当前batch的图的结点数
            node_size_index = self.node_num_at_batch[batch_step]
            node_num = self.node_sizes[node_size_index]
            start_idx = bucket_counters[node_size_index] * self.batch_size
            end_idx = (bucket_counters[node_size_index] + 1) * self.batch_size
            elements = self.node_num_2_graph_dict[node_size_index][start_idx:end_idx]
            batch_data = self.make_batch(elements)
            num_graphs = len(batch_data['orders'])

            # todo: unknown?
            batch_data['orders'] = np.squeeze(batch_data['orders'], axis=1)
            batch_data['labels'] = np.squeeze(batch_data['labels'], axis=1)
            batch_data['task_masks'] = np.squeeze(batch_data['task_masks'], axis=1)
            # if len(batch_data['orders'].shape) == 1:
            #     batch_data['orders'] = np.expand_dims(batch_data['orders'], axis=0)
            #
            if len(batch_data['labels']) == 0:
                continue

            bucket_counters[node_size_index] += 1

            final_batch_data = {
                'adj_mat': torch.FloatTensor(batch_data['adj_mat']),
                'input_orders': torch.LongTensor(batch_data['orders']),
                'target_values': torch.LongTensor(batch_data['labels']),
                'node_mask': torch.FloatTensor(batch_data['node_mask']),
                'target_mask': torch.FloatTensor(batch_data['task_masks']),
                'variable_orders': torch.LongTensor(batch_data['variables']),
                'variable_masks': torch.FloatTensor(batch_data['variable_masks']),
                'num_vertices': node_num,
                'num_graphs': num_graphs
            }

            yield final_batch_data


    def make_batch(self, elements):
        batch_data = {
            'adj_mat': [],
            'orders': [],
            'labels': [],
            'node_mask': [],
            'task_masks': [],
            'variables': [],
            'variable_masks': []
        }

        for d in elements:
            variable_length = len(d['variable'][0])
            batch_data['adj_mat'].append(d['adj_mat'])
            batch_data['orders'].append(d['orders'])
            batch_data['node_mask'].append(d['mask'])
            variables = [idx for idx in d['variable'][0][:self.max_var_len]] + [self.unk_id for _ in
                                                                  range(self.max_var_len - variable_length)]
            batch_data['variables'].append(variables)

            target_task_values = []
            target_task_mask = []

            for target_val in d['labels']:
                if target_val is None:
                    target_task_values.append(0.)
                    target_task_mask.append(0.)
                else:
                    target_task_values.append(target_val)
                    target_task_mask.append(1.)

            variable_mask = []
            for i in range(self.max_var_len):
                if i < variable_length:
                    variable_mask.append([1 for _ in range(self.embedding_size)])
                else:
                    variable_mask.append([0 for _ in range(self.embedding_size)])

            batch_data['labels'].append(target_task_values)
            batch_data['task_masks'].append(target_task_mask)
            batch_data['variable_masks'].append(variable_mask)

        return batch_data

    def load_graphs_from_file(self, file_name):
        with open(file_name, 'r') as f:
            data = json.load(f)
        num_fwd_edge_types = 0
        for g in data:
            # 找出最大顶点序号
            self.max_num_vertices = max(self.max_num_vertices, max([v for e in g['graph'] for v in [e[0], e[2]]]))
            # 找出最大边序号
            num_fwd_edge_types = max(num_fwd_edge_types, max([e[1] for e in g['graph']]))

        self.num_edge_types = max(self.num_edge_types, num_fwd_edge_types * (1 if self.tie_fwd_bkwd else 2))

        return self.process_raw_graphs(data)

    def process_raw_graphs(self, raw_data, node_sizes=None):
        if node_sizes is None:
            node_sizes = np.array(list(range(1, 300, 2)))
        node_num_2_graph_dict = defaultdict(list)
        for d in raw_data:
            chosen_node_idx = np.argmax(node_sizes > max([v for e in d['graph']
                                                            for v in [e[0], e[2]]]))
            chosen_node_num = node_sizes[chosen_node_idx]
            n_actual_nodes = len(d['orders'][0])

            # print('actual: %d, choose: %d' % (n_actual_nodes, chosen_node_num))

            # 添加变量名mask,之前数据集合中有-1，现在转换成unk
            cur_variable_indexes = []
            variables = d['variable']
            if len(variables) > self.max_var_len:
                variables = variables[:self.max_var_len]

            for idx, i in enumerate(variables):
                if i == -1:
                    cur_variable_indexes.append(self.unk_id)
                else:
                    cur_variable_indexes.append(i)

            node_num_2_graph_dict[chosen_node_idx].append({
                'adj_mat': self.graph_to_adj_mat(d['graph'], chosen_node_num, self.num_edge_types,
                                                 self.tie_fwd_bkwd),
                'orders': [d["orders"][0] + [0 for _ in range(chosen_node_num - n_actual_nodes)]],
                'labels': [d["targets"][0][0]],
                'mask': [[1.0 for _ in range(self.hidden_size)] for _ in range(n_actual_nodes)] +
                        [[0. for _ in range(self.hidden_size)] for _ in
                         range(chosen_node_num - n_actual_nodes)],
                'variable': [cur_variable_indexes]
            })

        # # todo: 抽样（将一部分数据的labels变成None）
        # if self.is_training:
        #     for (node_num_id, bucket) in node_num_2_graph_dict.items():
        #         # 打乱每个bucket（包含了相同节点）的数据
        #         np.random.shuffle(bucket)
        #         if self.sample_ratios is not None:
        #             ex_to_sample = int(len(bucket) * self.sample_ratios)
        #             for ex_id in range(ex_to_sample, len(bucket)):
        #                 bucket[ex_id]['labels'][0] = None

        # [[1,1,1,1],[3,3,3],[5,5,5,5,5]]
        node_num_at_batch = [[node_num_id for _ in range(len(graph_data_of_node_num) // self.batch_size + 1)]
                          for node_num_id, graph_data_of_node_num in node_num_2_graph_dict.items()]

        # [1,1,1,1,3,3,3,5,5,5,5,5], 每个数字代表每个batch中每个图的结点数
        node_num_at_batch = [x for y in node_num_at_batch for x in y]

        return node_num_2_graph_dict, node_sizes, node_num_at_batch

    def graph_to_adj_mat(self, graph, max_n_vertices, num_edge_types, tie_fwd_bkwd=False):
        bwd_edge_offeset = 0 if tie_fwd_bkwd else (num_edge_types // 2)
        # [e, v, v]
        amat = np.zeros((num_edge_types, max_n_vertices, max_n_vertices))
        for src, e, dest in graph:
            if e == 0 and src == 0 and dest == 0:
                continue
            amat[e - 1, dest - 1, src - 1] = 1
            amat[e - 1 + bwd_edge_offeset, src - 1, dest - 1] = 1
        return amat
