from typing import Any, Sequence
import os

import tensorflow as tf
import numpy as np

import json
from collections import defaultdict

from code_rec_api_level.tf_version_2.prog_model import ProgModel
from code_rec_api_level.tf_version_2.util import glorot_init, graph_to_adj_mat

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class DenseGGNNProgModel(ProgModel):
    def __init__(self, args, training_file_count, valid_file_count):
        super().__init__(args, training_file_count, valid_file_count)
        # 取得embedding层预训练参数
        # self.embedding_weight = self.load_embedding_weight()
        # with open('/Users/coldilock/Documents/Code/Github/FDSEDeepLearning/Code-Recommendation/android_vocabulary/APIVocabulary.txt', 'r') as f:
        #     self.api2idx = {}
        #     self.idx2api = {}
        #     for idx, word in enumerate(f.readlines()):
        #         api = word.strip()
        #         self.api2idx[api] = idx
        #         self.idx2api[idx] = api
        # with open('/Users/coldilock/Documents/Code/Github/FDSEDeepLearning/Code-Recommendation/android_vocabulary/WholeVocabulary.txt', 'r') as f:
        #     self.word2idx = {}
        #     self.idx2word = {}
        #     for idx, token in enumerate(f.readlines()):
        #         word = token.strip()
        #         self.word2idx[word] = idx
        #         self.idx2word[idx] = word
        # self.variable2index = {}
        # self.index2variable = {}
        # with open('/Users/coldilock/Documents/Code/Github/FDSEDeepLearning/Code-Recommendation/android_vocabulary/VariableNameVocabulary.txt', 'r') as f:
        #     lines = f.readlines()
        #     for (index, word) in enumerate(lines):
        #         word = word.strip()
        #         self.variable2index[word] = index
        #         self.index2variable[index] = word
        # print("On Serving...")

    # 读取treelstm模型中的embedding层
    def load_embedding_weight(self):
        load = np.load('data/treelstm.npy', encoding="latin1").item()
        return load['word_embedding/weights']

    @classmethod
    def default_params(cls):
        params = dict(super().default_params())
        params.update({
            'batch_size': 256,
            'graph_state_dropout_keep_prob': 0.75,
            'task_sample_ratios': {},
            'use_edge_bias': True,
        })
        return params

    # 实现抽象方法
    def prepare_specific_graph_model(self) -> None:
        h_dim = self.params['hidden_size']
        e_dim = self.params['embed_size']
        v_dim = self.params['variable_size']

        # input
        self.placeholders['graph_state_keep_prob'] = tf.placeholder(tf.float32, None, name='graph_state_keep_prob')
        self.placeholders['edge_weight_dropout_keep_prob'] = tf.placeholder(tf.float32, None,
                                                                            name='edge_weight_dropout_keep_prob')

        self.placeholders['input_orders'] = tf.placeholder(tf.int32, [None, None, ], name='input_orders')
        self.weights['index2vector'] = tf.Variable(dtype=tf.float32,
                                                   initial_value=np.random.uniform(-0.5, 0.5, [39070, e_dim]))

        # 这里加入变量名模型
        self.placeholders['variable_orders'] = tf.placeholder(tf.int32, [None, 10, ], name='variable_order')
        self.weights['variable_index2vector'] = tf.Variable(dtype=tf.float32,
                                                            initial_value=np.random.uniform(-0.5, 0.5, [12776, v_dim]))

        # 对变量名和api进行embeddding,变量名维度也是[b, v, v_dim]
        self.placeholders['variable_orders_embed_first'] = tf.nn.embedding_lookup(self.weights['variable_index2vector'],
                                                                                  self.placeholders['variable_orders'])
        self.placeholders['variable_orders_embed'] = tf.layers.dense(self.placeholders['variable_orders_embed_first'],
                                                                     self.params['variable_embed_size'], tf.tanh)
        self.placeholders['variable_orders_embed'] = tf.nn.dropout(self.placeholders['variable_orders_embed'],
                                                                   self.placeholders['out_layer_dropout_keep_prob'])
        self.placeholders['variable_orders_embed'] = tf.layers.dense(self.placeholders['variable_orders_embed'],
                                                                     self.params['variable_embed_size'], tf.tanh)
        self.placeholders['variable_orders_embed'] = tf.nn.dropout(self.placeholders['variable_orders_embed'],
                                                                   self.placeholders['out_layer_dropout_keep_prob'])
        self.placeholders['variable_orders_embed'] = tf.layers.dense(self.placeholders['variable_orders_embed'],
                                                                     self.params['variable_embed_size'], tf.tanh)
        self.placeholders['variable_orders_embed'] = tf.nn.dropout(self.placeholders['variable_orders_embed'],
                                                                   self.placeholders['out_layer_dropout_keep_prob'])

        # attention
        # self.variable_vectors = tf.contrib.seq2seq.LuongAttention(
        #    self.params['variable_embed_size'], self.placeholders['variable_orders_embed'],
        #    memory_sequence_length=[10], scale=True
        # )

        self.placeholders['orders_embed'] = tf.nn.embedding_lookup(self.weights['index2vector'],
                                                                   self.placeholders['input_orders'])
        # self.placeholders['orders_embed_first'] = tf.nn.embedding_lookup(self.weights['index2vector'],self.placeholders['input_orders'])
        # self.placeholders['orders_embed'] = tf.layers.dense(self.placeholders['orders_embed_first'], h_dim,tf.tanh)

        # 利用max_pooling选出适合的特征
        # shape = self.placeholders['variable_orders_embed'].get_shape().as_list()

        # 使用mask过滤掉0的部分
        self.placeholders['variable_mask'] = tf.placeholder(tf.float32, [None, 10, self.params['variable_embed_size']],
                                                            name='variable_mask')
        self.placeholders['variable_orders_embed'] = self.placeholders['variable_orders_embed'] * self.placeholders[
            'variable_mask']

        # self.placeholders['variable_orders_embed'] = self.Position_Embedding_Attention(self.placeholders['variable_orders_embed'],self.params['variable_embed_size'])
        self.variable_embeddings = self.placeholders['variable_orders_embed']
        # self.variable_embeddings = tf.expand_dims(self.placeholders['variable_orders_embed'],axis=len(shape))
        print('self.variable_embedding ', self.variable_embeddings.get_shape().as_list())
        # print('variable_embeddings shape: ',self.variable_embeddings.get_shape().as_list())
        # self.variable_vectors = tf.nn.max_pool(self.variable_embeddings,[1,shape[1],1,1],[1,1,1,1],'VALID')
        # print('variable after max_pooling: ',self.variable_vectors.get_shape().as_list())
        # self.variable_vectors = tf.squeeze(self.variable_vectors,[1,3]) # [b, v_dim]

        # print('final vairable vector: ',self.variable_vectors)

        # [b, v, h_dim]
        self.placeholders['initial_node_representation'] = tf.placeholder(tf.float32,
                                                                          [None, None, self.params['hidden_size']],
                                                                          name='node_features')
        # [b, v]
        self.placeholders['node_mask'] = tf.placeholder(tf.float32, [None, None, self.params['hidden_size']],
                                                        name='node_mask')
        self.placeholders['num_vertices'] = tf.placeholder(tf.int32, ())
        # [b,e,v,v]
        self.placeholders['adjacency_matrix'] = tf.placeholder(tf.float32,
                                                               [None, self.num_edge_types, None, None])
        # [b,e,v,v] -> [e,b,v,v]
        self.__adjacency_matrix = tf.transpose(self.placeholders['adjacency_matrix'], [1, 0, 2, 3])

        # weights
        # 边的weight和biase
        # weight -> [num_edge_types,dim,dim],初始化graph的weight
        # biase -> [num_edge_types,1,dim]，初始化graph的biase
        # 现在修改weights成[edge_type, h_dim, h_dim]
        self.weights['edge_weights'] = tf.Variable(glorot_init([self.num_edge_types, h_dim, h_dim]),
                                                   name='edge_weights')
        if self.params['use_edge_bias']:
            self.weights['edge_biases'] = tf.Variable(np.zeros([self.num_edge_types, 1, h_dim]).astype(np.float32),
                                                      name='edge_biases')
        with tf.variable_scope("gru_scope"):
            cell = tf.contrib.rnn.GRUCell(h_dim)
            cell = tf.nn.rnn_cell.DropoutWrapper(cell, state_keep_prob=self.placeholders['graph_state_keep_prob'])
            self.weights['node_gru'] = cell

    # 实现抽象方法
    # 这里定义了每次信息传递过程中需要的计算
    # 包括了边的传播和GRU单元的计算
    def compute_final_node_representations(self) -> tf.Tensor:
        v = self.placeholders['num_vertices']
        h_dim = self.params['hidden_size']
        e_dim = self.params['embed_size']
        # [b, v, h] -> [b*v, h]
        # 输入[batchsize, words] -> [batchsize, words, vec]
        orders = self.placeholders['orders_embed']
        # self.orders_embed = tf.nn.embedding_lookup(self.weights['index2vector'],orders)

        # orders_embed: [b, v, h_dim] -> [b*v, h_dim]
        orders_embed = tf.reshape(orders, [-1, h_dim])
        # h = self.placeholders['initial_node_representation']
        # h = tf.reshape(h, [-1, h_dim])

        # precompute edge biases
        # 计算出每种edge的biase
        if self.params['use_edge_bias']:
            biases = []
            # unstack成(e; t) -> 最终变成[b*v, h] , t:[b, v, v]
            for edge_type, a in enumerate(tf.unstack(self.__adjacency_matrix, axis=0)):
                # 变形成[b*v, 1]
                summed_a = tf.reshape(tf.reduce_sum(a, axis=-1), [-1, 1])
                # 相乘后[b*v, h]
                # self.weights['edge_biases'][edge_type] --- [1, h_dim]
                biases.append(tf.matmul(summed_a, self.weights['edge_biases'][edge_type]))

        # GRU单元信息传递
        with tf.variable_scope("gru_scope") as scope:
            # 每个节点进行一次信息传递（和每种边进行矩阵乘法），然后放入GRU中利用门结构进行信息提取
            for i in range(self.params['num_timesteps']):
                # 共享每一步t的gru参数
                if i > 0:
                    tf.get_variable_scope().reuse_variables()
                # 遍历每一种edge_type
                for edge_type in range(self.num_edge_types):
                    # [b*v, e] * [e_dim, h_dim] -> [b*v, h_dim]
                    m = tf.matmul(orders_embed, tf.nn.dropout(self.weights['edge_weights'][edge_type],
                                                              self.placeholders['edge_weight_dropout_keep_prob']))
                    if self.params['use_edge_bias']:
                        m += biases[edge_type]
                    # [b*v, h_dim] -> [b, v, h_dim]
                    m = tf.reshape(m, [-1, v, h_dim])
                    # 这里存储act内容，因为是遍历每个edgetype，因此把每个信息都加到一起
                    if edge_type == 0:
                        # __adjacency_matrix[edge_type] -> [b, v, v]
                        # [b, v, v] * [b, v, dim] -> [b, v, dim]
                        acts = tf.matmul(self.__adjacency_matrix[edge_type], m)
                    else:
                        acts += tf.matmul(self.__adjacency_matrix[edge_type], m)
                acts = tf.reshape(acts, [-1, h_dim])

                # 向gru单元里输入取出hidden,[b*v, h_dim]
                orders_embed = self.weights['node_gru'](acts, orders_embed)[1]
            # [b, v, h_dim]
            last_h = tf.reshape(orders_embed, [-1, v, h_dim])
        return last_h

    # 计算模型的输出部分
    def gated_regression(self, last_h, regression_gate, regression_transform, softmax_weights, softmax_biases):
        # last_h: [b, v, h]
        # gate_input: [b, v, 2*h] -> [b*v, 2*h]
        # 把最后一次time step的GRU结果拿来和initial_node_representation合并
        # gate_input = tf.concat([last_h, self.placeholders['initial_node_representation']], axis=2)
        gate_input = tf.concat([last_h, self.placeholders['orders_embed']], axis=2)
        gate_input = tf.reshape(gate_input, [-1, 2 * self.params["hidden_size"]])

        # [b*v, h]
        last_h = tf.reshape(last_h, [-1, self.params["hidden_size"]])

        # regression_gate: [b*v, 2*h] -> [b*v, 1] (NEW! 现在修改成[b*v, 2*h]->[b*v, h])
        # regression_transform: [b*v, h] -> [b*v, 1] (NEW! 现在修改成[b*v, h]->[b*v, h])
        # gated_outputs: [b*v, 1] (NEW! 现在修改成[b*v, h])
        # 由regreesion_gate充当了一个门结构，决定last_h中有多少传递进来（有点像lstm的遗忘门）
        gated_outputs = tf.nn.sigmoid(regression_gate(gate_input)) * regression_transform(last_h)

        # [b*v, h] -> [b, v, h]
        gated_outputs = tf.reshape(gated_outputs, [-1, self.placeholders['num_vertices'], self.params["hidden_size"]])

        # 用node_mask确定哪些点需要mask掉
        masked_gated_outputs = gated_outputs * self.placeholders['node_mask']
        # self.variable_embeddings = tf.squeeze(self.variable_embeddings,axis=-1)
        # self.variable_vectors = tf.reduce_sum(self.variable_embeddings, axis = 1)
        # 尝试性操作，把所有的节点加到一起，建立[b,h]维度
        try_sum = tf.reduce_sum(masked_gated_outputs, axis=1)
        print('graph vector: ', try_sum)
        self.variable_vectors = tf.reduce_sum(self.variable_embeddings, axis=1)
        print('variable vector: ', self.variable_vectors)
        inte_vectors = tf.concat([try_sum, self.variable_vectors], axis=-1)
        print('inte_vectors: ', inte_vectors)
        inte_vectors = tf.layers.dense(inte_vectors, self.params['softmax_size'], tf.tanh)
        inte_vectors = tf.nn.dropout(inte_vectors, self.placeholders['out_layer_dropout_keep_prob'])
        print('after concat: ', inte_vectors.get_shape().as_list())
        # 计算和: [b, v] -> [b]
        # output = tf.reduce_sum(masked_gated_outputs, axis = 1)

        # 对抽取的特征做softmax分类
        output = tf.matmul(inte_vectors, self.weights['softmax_weights']) + self.weights['softmax_biases']

        self.output = output
        return output

    # 数据预处理以及minibatch
    def process_raw_graphs(self, raw_data: Sequence[Any], is_training_data: bool, bucket_sizes=None) -> Any:
        # 生成不同的bucket，装入和它大小相近的数据进行分组，之后填充零
        if bucket_sizes is None:
            bucket_sizes = np.array(list(range(1, 300, 2)))
        bucketed = defaultdict(list)
        # 获取数据中node_features的维度（默认的分子数据中是5）
        # x_dim = len(raw_data[0]["node_features"][0])

        # 遍历每个数据
        # print(len(raw_data))
        for d in raw_data:
            # 返回第一个满足判断的index
            chosen_bucket_idx = np.argmax(bucket_sizes > max([v for e in d['graph']
                                                              for v in [e[0], e[2]]]))
            # print(chosen_bucket_idx)
            chosen_bucket_size = bucket_sizes[chosen_bucket_idx]
            # 活跃节点数（或相当于节点数）
            n_active_nodes = len(d["orders"][0])
            # print(n_active_nodes)
            # print(d["orders"])

            # 添加变量名mask,之前数据集合中有-1，现在转换成unk
            cur_variable_indexes = []
            variables = d['variable']
            if len(variables) > 10:
                variables = variables[0:10]
                # variables = variables[len(variables) - 10:len(variables)]
            for idx, i in enumerate(variables):
                if i == -1:
                    cur_variable_indexes.append(7141)
                else:
                    cur_variable_indexes.append(i)
                # print(len(cur_variable_indexes))
            # 将节点数相近的并在一起（用稍大一点的chosen_bucket_size约束），同时添加mask用作屏蔽
            # 比如：原始数据有9个节点，我们设置的chosen_bucket_size为10，那么会添加一个纯0的节点，
            # 并且将增加的这部分也反映到邻接矩阵上，最后添加到mask中，让它的mask为0

            # 如果这里引入了embedding，需要将加入的init的vector对应到字典的unknown

            # 'init': d["node_features"] + [[0 for _ in range(x_dim)] for _ in range(chosen_bucket_size - n_active_nodes)],
            bucketed[chosen_bucket_idx].append({
                'adj_mat': graph_to_adj_mat(d['graph'], chosen_bucket_size, self.num_edge_types,
                                            self.params['tie_fwd_bkwd']),
                'orders': [d["orders"][0] + [0 for _ in range(chosen_bucket_size - n_active_nodes)]],
                'labels': [d["targets"][task_id][0] for task_id in self.params['task_ids']],
                'mask': [[1.0 for _ in range(self.params['hidden_size'])] for _ in range(n_active_nodes)] +
                        [[0. for _ in range(self.params['hidden_size'])] for _ in
                         range(chosen_bucket_size - n_active_nodes)],
                'variable': [cur_variable_indexes]
            })

        if is_training_data:
            for (bucket_idx, bucket) in bucketed.items():
                # 打乱每个bucket（包含了相同节点）的数据
                np.random.shuffle(bucket)
                for task_id in self.params['task_ids']:
                    # 抽样？（似乎是将一部分数据的labels变成None）
                    task_sample_ratio = self.params['task_sample_ratios'].get(str(task_id))
                    if task_sample_ratio is not None:
                        ex_to_sample = int(len(bucket) * task_sample_ratio)
                        for ex_id in range(ex_to_sample, len(bucket)):
                            bucket[ex_id]['labels'][task_id] = None

        # 用bucket_idx表示每个bucket中执行的次数，比如[[1,1,1,1],[2,2,2],[3,3,3,3,3,3]]
        bucket_at_step = [[bucket_idx for _ in range(len(bucket_data) // self.params['batch_size'] + 1)]
                          for bucket_idx, bucket_data in bucketed.items()]

        # 铺平成[1,1,1,1,2,2,2,3,3,3,3,3,3]
        bucket_at_step = [x for y in bucket_at_step for x in y]

        return (bucketed, bucket_sizes, bucket_at_step)

    def make_batch(self, elements):
        batch_data = {'adj_mat': [], 'orders': [], 'labels': [], 'node_mask': [], 'task_masks': [], 'variables': [],
                      'variable_masks': []}
        # 逐数据将信息填入
        max_variable_size = 10
        # print('cur batch variable length: ',max_variable_size)
        for d in elements:
            variable_length = len(d['variable'][0])
            batch_data['adj_mat'].append(d['adj_mat'])
            batch_data['orders'].append(d['orders'])
            batch_data['node_mask'].append(d['mask'])
            # print(d['variable'])
            variables = [idx for idx in d['variable'][0][:10]] + [7141 for _ in
                                                                  range(max_variable_size - variable_length)]
            # print(np.array(variables).shape)
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
            for i in range(max_variable_size):
                if i < variable_length:
                    variable_mask.append([1 for _ in range(self.params['variable_embed_size'])])
                else:
                    variable_mask.append([0 for _ in range(self.params['variable_embed_size'])])

            batch_data['labels'].append(target_task_values)
            batch_data['task_masks'].append(target_task_mask)
            batch_data['variable_masks'].append(variable_mask)
        return batch_data

    def make_minibatch_iterator(self, data, is_training: bool):
        # print(data[1])
        (bucketed, bucket_sizes, bucket_at_step) = data
        if is_training:
            # 打乱需要执行的顺序
            np.random.shuffle(bucket_at_step)
            for _, bucketed_data in bucketed.items():
                # 打乱指定bucketed中同长度的数据
                np.random.shuffle(bucketed_data)

        bucket_counters = defaultdict(int)
        dropout_keep_prob = self.params['graph_state_dropout_keep_prob'] if is_training else 1.
        # 获得batchsize个数据
        for step in range(len(bucket_at_step)):
            # 选中一个bucket，用于寻找该数量nodes的集合
            bucket = bucket_at_step[step]
            start_idx = bucket_counters[bucket] * self.params['batch_size']
            end_idx = (bucket_counters[bucket] + 1) * self.params['batch_size']
            elements = bucketed[bucket][start_idx:end_idx]
            # 将batchsize长个数据做成batch
            batch_data = self.make_batch(elements)

            num_graphs = len(batch_data['orders'])
            # initial_representations = batch_data['init']

            # 补零，将节点的维度拉伸到hidden_size
            # initial_representations = self.pad_annotations(initial_representations)
            # print(initial_representations)

            batch_data['orders'] = np.squeeze(batch_data['orders'], axis=1)
            if len(batch_data['orders'].shape) == 1:
                batch_data['orders'] = np.expand_dims(batch_data['orders'], axis=0)

            if len(batch_data['labels']) == 0:
                continue
            # 打印order
            # print(batch_data['orders'])
            # self.placeholders['initial_node_representation']: initial_representations,

            # print('################################################################################')
            # print('order: ',batch_data['orders'].shape)
            # print('target: ',np.array(batch_data['labels']).shape)
            # print('target_mask: ',np.array(batch_data['task_masks']).shape)
            # print('num_vertices: ',bucket_sizes[bucket])
            # print('adjacency_matrix: ',np.array(batch_data['adj_mat']).shape)
            # print('variable_orders: ',np.array(batch_data['variables']).shape)
            # print('variable_mask: ',np.array(batch_data['variable_masks']).shape)

            # print('################################################################################')
            batch_feed_dict = {
                self.placeholders['input_orders']: batch_data['orders'],  # 顺序
                self.placeholders['target_values']: np.transpose(batch_data['labels'], axes=[1, 0]),  #
                self.placeholders['target_mask']: np.transpose(batch_data['task_masks'], axes=[1, 0]),
                self.placeholders['num_graphs']: num_graphs,  # 多少个数据
                self.placeholders['num_vertices']: bucket_sizes[bucket],  # 点数量（补0后）
                self.placeholders['adjacency_matrix']: batch_data['adj_mat'],  # 邻接矩阵
                self.placeholders['node_mask']: batch_data['node_mask'],  # 节点mask
                self.placeholders['variable_orders']: batch_data['variables'],  # 变量
                self.placeholders['variable_mask']: batch_data['variable_masks'],  # 变量mask
                self.placeholders['graph_state_keep_prob']: dropout_keep_prob,
                self.placeholders['edge_weight_dropout_keep_prob']: dropout_keep_prob
            }

            bucket_counters[bucket] += 1

            yield batch_feed_dict

    # 补零 [b, v, annotation_size] -> [b, v, h_dim]
    def pad_annotations(self, annotations):
        return np.pad(annotations, pad_width=[[0, 0], [0, 0], [0, self.params['hidden_size'] - self.annotation_size]],
                      mode='constant')

    # def evaluate_one_batch(self, initial_node_representations, orders, adjacency_matrices, node_masks=None):
    #     num_vertices = len(initial_node_representations[0])
    #     if node_masks is None:
    #         node_masks = []
    #         for r in initial_node_representations:
    #             node_masks.append([1. for _ in r] + [0. for _ in range(num_vertices - len(r))])
    #
    #     # self.placeholders['initial_node_representation']: self.pad_annotations(initial_node_representations),
    #     batch_feed_dict = {
    #         self.placeholders['input_orders']: orders,
    #         self.placeholders['num_graphs']: len(initial_node_representations),
    #         self.placeholders['num_vertices']: len(initial_node_representations[0]),
    #         self.placeholders['adjacency_matrix']: adjacency_matrices,
    #         self.placeholders['node_mask']: node_masks,
    #         self.placeholders['graph_state_keep_prob']: 1.0,
    #         self.placeholders['out_layer_dropout_keep_prob']: 1.0,
    #     }
    #
    #     fetch_list = self.output
    #     result = self.sess.run(fetch_list, feed_dict=batch_feed_dict)
    #     return result

    # 验证集测试
    def example_evaluation(self):

        n_example_molecules = 10
        with open('molecules_valid.json', 'r') as valid_file:
            example_molecules = json.load(valid_file)[:n_example_molecules]
        example_molecules, _, _ = self.process_raw_graphs(example_molecules, is_training_data=False,
                                                          bucket_sizes=np.array([29]))
        batch_data = self.make_batch(example_molecules[0])
        print(self.evaluate_one_batch(batch_data['init'], batch_data['orders'], batch_data['adj_mat']))

    def attention(self, inputs, query, attention_size):

        q = tf.expand_dims(query, 1)
        q = tf.tile(q, [1, 10, 1])
        # inputs:[None,10,300]表示变量名
        # query:[None,10,300]表示图
        hidden_size = attention_size

        # Trainable parameters
        # w_omega = tf.Variable(tf.random_normal([hidden_size, attention_size], stddev=0.1))
        # u_omega = tf.Variable(tf.random_normal([hidden_size, attention_size], stddev=0.1))
        # v_omega = tf.Variable(tf.random_normal([hidden_size], stddev=0.1))
        # w_omega = tf.Variable(dtype=tf.float32,initial_value=np.random.uniform(-0.05, 0.05,[hidden_size, attention_size]))
        # u_omega = tf.Variable(dtype=tf.float32,initial_value=np.random.uniform(-0.05, 0.05,[hidden_size, attention_size]))
        # v_omega = tf.Variable(dtype=tf.float32,initial_value=np.random.uniform(-0.05, 0.05,[hidden_size]))
        w_omega = tf.Variable(glorot_init([hidden_size, attention_size]))
        u_omega = tf.Variable(glorot_init([hidden_size, attention_size]))
        v_omega = tf.Variable(dtype=tf.float32, initial_value=np.random.uniform(-0.05, 0.05, [hidden_size]))

        with tf.name_scope('v'):
            # print('inputs: ', inputs)
            # print('w_omega: ',w_omega)
            # print('inputs: ', query)
            # print('w_omega: ',u_omega)
            # v = tf.tanh(tf.tensordot(w_omega,inputs, axes=1) + tf.tensordot(u_omega, query, axes=1))
            v = tf.tanh(tf.tensordot(inputs, w_omega, axes=1) + tf.tensordot(q, u_omega, axes=1))

        # print('inputs: ', tf.tensordot(inputs, w_omega, axes=1))
        vu = tf.tensordot(v, v_omega, axes=1, name='vu')
        alphas = tf.nn.softmax(vu, name='alphas')

        # output = tf.reduce_sum(inputs * tf.expand_dims(alphas, -1),1)
        output = tf.reduce_sum(inputs * tf.expand_dims(alphas, -1), 1)

        return output

    def softmax(self, array):
        now_array = np.array(array)
        exp_sum = 0.0
        new_array = []
        for num in now_array[0]:
            exp_sum += np.exp(num)
        for num in now_array[0]:
            new_array.append(np.exp(num) / exp_sum)
        return np.array(new_array)

    # def predict(self, graphRepresent, graphVocab, variableName, single_targets=[[-1]]):
    #     cur_order = []
    #     for idx, word in eval(graphVocab).items():
    #         real_idx = self.word2idx.get(word, 21504)
    #         cur_order.append(real_idx)
    #     tokens = variableName.strip().split(" ")
    #     cur_indexes = []
    #     for token in tokens:
    #         if token == '':
    #             cur_indexes.append(-1)
    #         else:
    #             index = self.variable2index.get(token, 7141)
    #             if index != 7141 and len(token) != 1:
    #                 cur_indexes.append(index)
    #     json_now = {'graph': json.loads(graphRepresent), 'orders': [cur_order], 'targets': single_targets,
    #                 'variable': cur_indexes}
    #     example = self.process_raw_graphs([json_now], is_training_data=False)
    #     one_batch = self.make_minibatch_iterator(example, False)
    #
    #     # print(one_batch)
    #     # actually just one
    #     output_string = []
    #     for step, batch in enumerate(one_batch):
    #         batch[self.placeholders['out_layer_dropout_keep_prob']] = 1.0
    #         # print(batch)
    #
    #
    #         fetch_list = tf.nn.top_k(self.output, 10)
    #         result = self.sess.run(fetch_list, feed_dict=batch)
    #         softmax_prob = self.softmax(result[0])
    #         for prob, idx in zip(softmax_prob, result[1][0]):
    #             output_string.append((self.idx2api[idx], float('%.2f' % prob)))
    #
    #
    #
    #         # output_string = str(output_string)
    #     # print(output_string)
    #     return output_string