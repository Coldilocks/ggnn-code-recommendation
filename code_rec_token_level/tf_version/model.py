from typing import Tuple, List, Any, Sequence
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import numpy as np
import time
import pickle

import json
import queue
import threading
import random
from collections import namedtuple, defaultdict
import gc


class DefaultConfig(object):
    train_data_root = ''
    test_data_root = ''
    valid_data_root = '../data/token_valid.json'
    load_model_path = ''
    restore_model_path = '../save/checkpoints'
    whole_vocabulary_size = 100  # 整个词表的大小，包含token和节点的type
    batch_size = 256
    graph_state_dropout = 0.75
    token_vocabulary_size = 100  # 预测词表的大小，也就是token词表的大小
    token_max_len = 20


# 定义MLP层
class MLP(object):
    def __init__(self, in_size, out_size, hid_sizes, dropout_keep_prob):
        self.in_size = in_size
        self.out_size = out_size
        self.hid_sizes = hid_sizes
        self.dropout_keep_prob = dropout_keep_prob
        self.params = self.make_network_params()

    # MLP层的参数以及存储方式
    def make_network_params(self):
        dims = [self.in_size] + self.hid_sizes + [self.out_size]
        # in_size -> hid_sizes -> out_size
        weight_sizes = list(zip(dims[:-1], dims[1:]))
        weights = [tf.Variable(self.init_weights(s), name='MLP_W_layer%i' % i)
                   for (i, s) in enumerate(weight_sizes)]
        biases = [tf.Variable(np.zeros(s[-1]).astype(np.float32), name='MLP_b_layer%i' % i)
                  for (i, s) in enumerate(weight_sizes)]
        network_params = {
            "weights": weights,
            "biases": biases
        }
        return network_params

    def init_weights(self, shape):
        return np.sqrt(6.0 / (shape[-2] + shape[-1])) * (2 * np.random.rand(*shape).astype(np.float32) - 1)

    # 这里设计了两层的神经网络，因此直接调用就可以输出last_hidden
    def __call__(self, inputs):
        acts = inputs
        for W, b in zip(self.params['weights'], self.params['biases']):
            hid = tf.matmul(acts, tf.nn.dropout(W, self.dropout_keep_prob)) + b
            acts = tf.nn.relu(hid)
        last_hidden = hid
        return last_hidden


# 多线程计算
class ThreadedIterator:
    """An iterator object that computes its elements in a parallel thread to be ready to be consumed.
    The iterator should *not* return None"""

    def __init__(self, original_iterator, max_queue_size: int = 2):
        self.__queue = queue.Queue(maxsize=max_queue_size)
        self.__thread = threading.Thread(target=lambda: self.worker(original_iterator))
        self.__thread.start()

    def worker(self, original_iterator):
        for element in original_iterator:
            assert element is not None, 'By convention, iterator elements much not be None'
            self.__queue.put(element, block=True)
        self.__queue.put(None, block=True)

    def __iter__(self):
        next_element = self.__queue.get(block=True)
        while next_element is not None:
            yield next_element
            next_element = self.__queue.get(block=True)
        self.__thread.join()


# glorot初始化
def glorot_init(shape):
    initialization_range = np.sqrt(6.0 / (shape[-2] + shape[-1]))
    return np.random.uniform(low=-initialization_range, high=initialization_range, size=shape).astype(np.float32)


# 将graph转换成[num_edge_types, max_n_vertices, max_n_vertices]邻接矩阵
def graph_to_adj_mat(graph, max_n_vertices, num_edge_types, tie_fwd_bkwd=False):
    bwd_edge_offset = 0 if tie_fwd_bkwd else (num_edge_types // 2)
    amat = np.zeros((num_edge_types, max_n_vertices, max_n_vertices))
    # 序号的起始是1，因此都记得要减1
    for src, e, dest in graph:
        # 如果graph是通过补0扩展得到的，那么补充的部分应该忽略
        if (e == 0 and src == 0 and dest == 0):
            continue
        amat[e - 1, dest - 1, src - 1] = 1
        amat[e - 1 + bwd_edge_offset, src - 1, dest - 1] = 1
    return amat


class DenseGGNNProgModel():
    # 默认参数
    # cls表示可以直接用类名调用
    # patience表示容忍度
    @classmethod
    def default_params(cls):
        return {
            'num_epochs': 50,
            'patience': 5,
            'learning_rate': 0.005,
            'clamp_gradient_norm': 1.0,
            'out_layer_dropout_keep_prob': 0.75,
            'momentum': 0.9,
            'embed_size': 300,
            'hidden_size': 600,
            'softmax_size': 600,
            'num_timesteps': 5,
            'use_graph': True,
            'tie_fwd_bkwd': False,
            'task_ids': [0],
            'random_seed': 0,

            'batch_size': DefaultConfig.batch_size,
            'graph_state_dropout_keep_prob': DefaultConfig.graph_state_dropout,
            'task_sample_ratios': {},
            'use_edge_bias': True
        }

    # 0. model initialization
    def __init__(self, args, training_file_count, valid_file_count):
        self.args = args
        self.training_file_count = training_file_count
        self.valid_file_count = valid_file_count

        # collect argument things
        data_dir = ''
        if args.data_dir is not None:
            data_dir = args.data_dir
        self.data_dir = data_dir

        # 设置运行记录文件（保存参数）
        # 命名格式如例子：2018-05-02-21-11-08_21156_log.json
        self.run_id = "_".join([time.strftime("%Y-%m-%d-%H-%M-%S"), str(os.getpid())])
        log_dir = '../../tf_ms'
        # 模型存储路径
        self.log_file = os.path.join(log_dir, "%s_log.pickle" % self.run_id)
        # 最佳模型存储路径
        self.best_model_file = os.path.join(log_dir, "%s_model_best.pickle" % self.run_id)
        self.best_model_checkpoint = "model_best-%s" % self.run_id

        # collect parameters
        # 读取默认配置
        params = self.default_params()
        # 找默认配置文件
        config_file = args.config_file
        if config_file is not None:
            with open(config_file, 'r') as f:
                params.update(json.load(f))

        # 把args赋值给当前类的config，用来更新params
        config = args.config
        if config is not None:
            params.update(json.loads(config))

        self.params = params

        # 把现在的参数存入新的json文件
        print("Run %s starting with following parameters:\n%s" % (self.run_id, json.dumps(self.params)))

        random.seed(params['random_seed'])
        np.random.seed(params['random_seed'])

        # load data
        self.max_num_vertices = 0
        self.num_edge_types = 8
        self.annotation_size = 0
        self.mini_train_data = None
        self.mini_valid_data = None

        # Build the actual model
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.log_device_placement = False

        self.graph = tf.Graph()
        self.sess = tf.Session(graph=self.graph, config=config)
        with self.graph.as_default():
            tf.set_random_seed(params['random_seed'])
            self.placeholders = {}
            self.weights = {}
            self.ops = {}

            # 1. make model
            self.make_model()
            # 2. make train step
            self.make_train_step()

            # Restore/initialize variables
            restore_file = args.restore
            if restore_file is not None:
                # 3.存储模型的路径
                self.restore_model2(DefaultConfig.restore_model_path + restore_file)
            else:
                # 4.
                self.initialize_model()

    # 数据预处理以及minibatch
    def process_raw_graphs(self, raw_data: Sequence[Any], is_training_data: bool, bucket_sizes=None) -> Any:
        # 生成不同的bucket，装入和它大小相近的数据进行分组，之后填充零
        if bucket_sizes is None:
            bucket_sizes = np.array(list(range(1, 1000, 2)))
        bucketed = defaultdict(list)

        # 遍历每个数据
        for d in raw_data:
            # 返回第一个满足判断的index
            chosen_bucket_idx = np.argmax(bucket_sizes > max([v for e in d['graph']
                                                              for v in [e[0], e[2]]]))
            chosen_bucket_size = bucket_sizes[chosen_bucket_idx]
            # 活跃节点数（或相当于节点数）
            n_active_nodes = len(d["label"])
            # 将节点数相近的并在一起（用稍大一点的chosen_bucket_size约束），同时添加mask用作屏蔽
            # 比如：原始数据有9个节点，我们设置的chosen_bucket_size为10，那么会添加一个纯0的节点，
            # 并且将增加的这部分也反映到邻接矩阵上，最后添加到mask中，让它的mask为0

            # 1.padding token orders
            # [actual_node_num, max_token_num]
            # [[1,2,3],[4,5],[7]] -> [[1,2,3],[4,5,0],[7,0,0]]
            token_orders = [token + [0 for _ in range(DefaultConfig.token_max_len - len(token))]
                            if len(token) <= DefaultConfig.token_max_len
                            else token[:DefaultConfig.token_max_len]
                            for token in d["token"]]
            # [actual_node_num, max_token_num] -> [max_node_num, max_token_num]
            # [[1,2,3],[4,5,0],[7,0,0]] -> [[1,2,3],[4,5,0],[7,0,0], [0,0,0]]
            padded_token_orders = token_orders + [[0 for _ in range(DefaultConfig.token_max_len)] for _ in
                                                  range(chosen_bucket_size - n_active_nodes)]

            # 2.padding type orders
            # [actual_node_num, 1] -> [actual_node_num]
            type_orders = [t for v in d["label"] for t in v]
            # [actual_node_num] -> [max_node_num]
            padded_type_orders = type_orders + [0 for _ in range(chosen_bucket_size - n_active_nodes)]

            # 3.padding token length
            # [actual_node_num]
            token_length = [len(token) if len(token) < DefaultConfig.token_max_len else DefaultConfig.token_max_len for
                            token in d["token"]]
            # [actual_node_num] -> [max_node_num]
            padded_token_length = token_length + [0 for _ in range(chosen_bucket_size - n_active_nodes)]

            # 如果这里引入了embedding，需要将加入的init的vector对应到字典的unknown
            bucketed[chosen_bucket_idx].append({
                'adj_mat': graph_to_adj_mat(d['graph'], chosen_bucket_size, self.num_edge_types,
                                            self.params['tie_fwd_bkwd']),
                # 'orders': [d["orders"][0] + [0 for _ in range(chosen_bucket_size - n_active_nodes)]],
                # token label
                'token_orders': padded_token_orders,
                # type label
                'type_orders': padded_type_orders,
                'labels': [d["targets"][task_id][0] for task_id in self.params['task_ids']],
                # mask: [n_active_nodes个[1,1,..,1], padding数量的[0,0,...0]]
                'mask': [[1.0 for _ in range(self.params['hidden_size'])] for _ in range(n_active_nodes)] +
                        [[0. for _ in range(self.params['hidden_size'])] for _ in
                         range(chosen_bucket_size - n_active_nodes)],
                'token_length': padded_token_length
                # 'token_length': [DefaultConfig.token_max_len for _ in d["token"]]

            })

        if is_training_data:
            for (bucket_idx, bucket) in bucketed.items():
                # 打乱每个bucket（包含了相同节点）的数据
                np.random.shuffle(bucket)
                for task_id in self.params['task_ids']:
                    # 抽样（将一部分数据的labels变成None）
                    task_sample_ratio = self.params['task_sample_ratios'].get(str(task_id))
                    if task_sample_ratio is not None:
                        ex_to_sample = int(len(bucket) * task_sample_ratio)
                        for ex_id in range(ex_to_sample, len(bucket)):
                            bucket[ex_id]['labels'][task_id] = None

        # 用bucket_idx表示每个bucket中执行的次数，比如[[4,4,4,4,4],[6,6,6],[8,8,8,8,8,8]]
        bucket_at_step = [[bucket_idx for _ in range(len(bucket_data) // self.params['batch_size'] + 1)]
                          for bucket_idx, bucket_data in bucketed.items()]

        # 铺平成[4,4,4,4,4,6,6,6,8,8,8,8,8,8]
        bucket_at_step = [x for y in bucket_at_step for x in y]

        return (bucketed, bucket_sizes, bucket_at_step)

    # 1.搭建模型
    def make_model(self):
        self.placeholders['target_values'] = tf.placeholder(tf.int64, [len(self.params['task_ids']), None],
                                                            name='target_values')
        self.placeholders['target_mask'] = tf.placeholder(tf.float32, [len(self.params['task_ids']), None],
                                                          name='target_mask')
        self.placeholders['num_graphs'] = tf.placeholder(tf.int64, [], name='num_graphs')
        self.placeholders['out_layer_dropout_keep_prob'] = tf.placeholder(tf.float32, [],
                                                                          name='out_layer_dropout_keep_prob')
        with tf.variable_scope("graph_model"):

            # 1.1 此处搭建具体的模型（虚拟函数）
            self.prepare_specific_graph_model()

            # this does the actual graph work:
            if self.params['use_graph']:
                # 1.2
                self.ops['final_node_representations'] = self.compute_final_node_representations()
            else:
                self.ops['final_node_representations'] = tf.zeros_like(self.placeholders['orders_embed'])

        self.ops['losses'] = []

        # 定义输出层（g层）的神经网络,对我们的数据MLP设置成hidden_size
        for (internal_id, task_id) in enumerate(self.params['task_ids']):
            with tf.variable_scope("out_layer_task%i" % task_id):
                with tf.variable_scope("regression_gate"):
                    self.weights['regression_gate_task%i' % task_id] = MLP(2 * self.params['hidden_size'],
                                                                           self.params['hidden_size'], [],
                                                                           self.placeholders[
                                                                               'out_layer_dropout_keep_prob'])
                with tf.variable_scope("regression"):
                    self.weights['regression_transform_task%i' % task_id] = MLP(self.params['hidden_size'],
                                                                                self.params['hidden_size'], [],
                                                                                self.placeholders[
                                                                                    'out_layer_dropout_keep_prob'])

                # 定义softmax层
                with tf.variable_scope("softmax"):
                    # 22821是预测词表的大小，softmax的最终输出维度
                    # softmax_weights: [softmax_size, token_vocab_size]
                    self.weights['softmax_weights'] = tf.Variable(
                        glorot_init([self.params['softmax_size'], DefaultConfig.token_vocabulary_size]))
                    # softmax_biases: [token_vocab_size]
                    self.weights['softmax_biases'] = tf.Variable(
                        np.zeros([DefaultConfig.token_vocabulary_size]).astype(np.float32))
                    print('softmax weights: ', self.weights['softmax_weights'])
                # 1.3 计算最后的g函数（输出函数）
                # computed_values: [b, token_vocab_size], 这里返回的是一个[b,softmax维度]的结果
                computed_values = self.gated_regression(self.ops['final_node_representations'],
                                                        self.weights['regression_gate_task%i' % task_id],
                                                        self.weights['regression_transform_task%i' % task_id],
                                                        self.weights['softmax_weights'],
                                                        self.weights['softmax_biases'])

                # 加上小值保证非零
                # mask out unused values
                # groundtruth: [b]
                groundtruth = self.placeholders['target_values'][internal_id, :]
                loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=computed_values, labels=groundtruth)
                task_loss = tf.reduce_sum(loss)

                # computed_values: [b, token_vocab_size] -> argmax -> [b]
                correct_prediction = tf.equal(tf.argmax(computed_values, 1), groundtruth)
                accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

                self.ops['accuracy_task%i' % task_id] = accuracy
                # normalise loss to account for fewer task-specific example in batch,如果没有就是1.0
                task_loss = task_loss * (1.0 / (self.params['task_sample_ratios'].get(task_id) or 1.0))
                self.ops['losses'].append(task_loss)
        self.ops['loss'] = tf.reduce_sum(self.ops['losses'])

        self.saver = tf.train.Saver()

    # 1.1
    def prepare_specific_graph_model(self) -> None:
        h_dim = self.params['hidden_size']
        e_dim = self.params['embed_size']

        # input
        self.placeholders['graph_state_keep_prob'] = tf.placeholder(tf.float32, None, name='graph_state_keep_prob')
        self.placeholders['edge_weight_dropout_keep_prob'] = tf.placeholder(tf.float32, None,
                                                                            name='edge_weight_dropout_keep_prob')

        # [b, v, h]
        self.placeholders['node_mask'] = tf.placeholder(tf.float32, [None, None, self.params['hidden_size']],
                                                        name='node_mask')
        self.placeholders['num_vertices'] = tf.placeholder(tf.int32, ())
        # [b, e, v, v]
        self.placeholders['adjacency_matrix'] = tf.placeholder(tf.float32,
                                                               [None, self.num_edge_types, None, None])
        # [b, e, v, v] -> [e, b, v, v]
        self.__adjacency_matrix = tf.transpose(self.placeholders['adjacency_matrix'], [1, 0, 2, 3])

        # 边的weight和bias
        # edge_weights -> [e, h, h],初始化graph的weight

        self.weights['edge_weights'] = tf.Variable(glorot_init([self.num_edge_types, h_dim, h_dim]),
                                                   name='edge_weights')
        # edge_biases -> [e,1,h]，初始化graph的bias
        if self.params['use_edge_bias']:
            # edge_biases: [e, 1, h]
            self.weights['edge_biases'] = tf.Variable(np.zeros([self.num_edge_types, 1, h_dim]).astype(np.float32),
                                                      name='edge_biases')
        with tf.variable_scope("gru_scope"):
            cell = tf.contrib.rnn.GRUCell(h_dim)
            cell = tf.nn.rnn_cell.DropoutWrapper(cell, state_keep_prob=self.placeholders['graph_state_keep_prob'])
            self.weights['node_gru'] = cell

            # 1. type embedding
            self.placeholders['type_input_orders'] = tf.placeholder(tf.int32,
                                                                    [None, None],
                                                                    name='type_input_orders')
            self.weights['type_index2vector'] = tf.Variable(dtype=tf.float32,
                                                            initial_value=np.random.uniform(-0.5, 0.5, [
                                                                DefaultConfig.whole_vocabulary_size, e_dim]))
            # type orders embedding: [b, v, e_dim]
            self.placeholders['type_orders_embed'] = tf.nn.embedding_lookup(self.weights['type_index2vector'],
                                                                            self.placeholders['type_input_orders'])

            # 2. token embedding
            # [b, v, max_token_num]
            self.placeholders['token_input_orders'] = tf.placeholder(tf.int32, [None, None, None],
                                                                     name='token_input_orders')
            self.weights['token_index2vector'] = tf.Variable(dtype=tf.float32,
                                                             initial_value=np.random.uniform(-0.5, 0.5, [
                                                                 DefaultConfig.token_vocabulary_size, e_dim]))
            # token orders embedding: [b, v, max_token_num, e_dim]
            self.placeholders['token_orders_embed'] = tf.nn.embedding_lookup(self.weights['token_index2vector'],
                                                                             self.placeholders['token_input_orders'])

            self.placeholders['token_length'] = tf.placeholder(shape=(None,), dtype=tf.int32, name='token_length')

    # 1.2 这里定义了每次信息传递过程中需要的计算, 包括了边的传播和GRU单元的计算
    def compute_final_node_representations(self) -> tf.Tensor:

        v = self.placeholders['num_vertices']
        h_dim = self.params['hidden_size']
        # e_dim * 2 == h_dim
        e_dim = self.params['embed_size']

        # token orders embed: [b, v, max_token_num, e_dim]
        token_orders_embed = self.placeholders['token_orders_embed']
        # [b, v, max_token_num, e_dim] -> [b*v, max_token_num, e_dim]
        token_orders_embed = tf.reshape(token_orders_embed, [-1, DefaultConfig.token_max_len, e_dim])
        token_length = self.placeholders['token_length']

        # 将token embedding输入到gru
        with tf.variable_scope("token_gru_scope"):
            cell = tf.contrib.rnn.GRUCell(e_dim)
            cell = tf.nn.rnn_cell.DropoutWrapper(cell, state_keep_prob=self.placeholders['graph_state_keep_prob'])
            # [b, v, e_dim]
            self.placeholders['token_orders_gru_embed'] = \
                tf.nn.dynamic_rnn(cell, token_orders_embed,
                                  sequence_length=token_length, dtype=tf.float32)[1]

        # token_orders_gru_embed: [b*v, e_dim]
        token_orders_gru_embed = self.placeholders['token_orders_gru_embed']
        # [b*v, e_dim] -> [b, v, e_dim]
        token_orders_gru_embed = tf.reshape(token_orders_gru_embed, [-1, self.placeholders['num_vertices'], e_dim])
        # type_orders_embed: [b, v, e_dim]
        type_orders_embed = self.placeholders['type_orders_embed']
        # type_orders_embed = tf.reduce_sum(type_orders_embed, axis=2)
        type_orders_embed = tf.reshape(type_orders_embed, [-1, self.placeholders['num_vertices'], e_dim])

        # 3. 节点最终的表示，包含了类型和token
        # [b, v, 2 * e_dim]
        self.placeholders['orders_embed'] = tf.concat([token_orders_gru_embed, type_orders_embed], axis=2)

        # orders: [b, v, 2*e_dim]
        orders = self.placeholders['orders_embed']
        # [b, v, 2*e_dim] -> [b * v, h_dim]
        orders_embed = tf.reshape(orders, [-1, h_dim])
        # 计算出每种edge的bias
        if self.params['use_edge_bias']:
            biases = []
            # __adjacency_matrix: [e, b, v, v]
            # unstack(axis=0) -> e * [b, v, v]
            # edge_type取值为[0,e-1], a的shape为[b,v,v]
            for edge_type, a in enumerate(tf.unstack(self.__adjacency_matrix, axis=0)):
                # a: [b,v,v] -> reduce_sum(最后一个维度求和) -> [b,v]
                # a: [b,v] -> reshape -> summed_a: [b * v, 1]
                summed_a = tf.reshape(tf.reduce_sum(a, axis=-1), [-1, 1])
                # edge_biases: [e, 1, h], self.weights['edge_biases'][edge_type]: [1, h]
                # [b*v, 1] * [1, h] -> [b*v, h]
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
                    # orders_embed: [b*v, h]  (h == 2*e_dim)
                    # edge_weights: [e, h, h], self.weights['edge_weights'][edge_type]: [h, h]
                    # [b*v, h] * [h, h] -> [b*v, h]
                    m = tf.matmul(orders_embed, tf.nn.dropout(self.weights['edge_weights'][edge_type],
                                                              self.placeholders['edge_weight_dropout_keep_prob']))
                    if self.params['use_edge_bias']:
                        m += biases[edge_type]
                    # [b*v, h] -> [b, v, h]
                    m = tf.reshape(m, [-1, v, h_dim])
                    # 这里存储act内容，因为是遍历每个edgetype，因此把每个信息都加到一起
                    if edge_type == 0:
                        # __adjacency_matrix[edge_type]: [b, v, v]
                        # [b, v, v] * [b, v, h] -> [b, v, h]
                        acts = tf.matmul(self.__adjacency_matrix[edge_type], m)
                    else:
                        acts += tf.matmul(self.__adjacency_matrix[edge_type], m)
                acts = tf.reshape(acts, [-1, h_dim])

                # 向gru单元里输入取出hidden,[b*v, h]
                orders_embed = self.weights['node_gru'](acts, orders_embed)[1]
            # [b, v, h]
            last_h = tf.reshape(orders_embed, [-1, v, h_dim])
        return last_h

    # 1.3 计算模型的输出部分
    def gated_regression(self, last_h, regression_gate, regression_transform, softmax_weights, softmax_biases):
        # last_h: [b, v, h]
        # gate_input: [b, v, 2*h] -> [b*v, 2*h]
        # 把最后一次time step的GRU结果拿来和orders_embed合并
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
        # [b, v, h] * [b, v, h] -> [b, v, h]
        masked_gated_outputs = gated_outputs * self.placeholders['node_mask']
        # 尝试性操作，把所有的节点加到一起，建立[b,h]维度
        # [b, v, h] -> [b, h]
        try_sum = tf.reduce_sum(masked_gated_outputs, axis=1)
        print('graph vector: ', try_sum)

        # 对抽取的特征做softmax分类
        # softmax_weightmomentums: [softmax_size, token_vocab_size], softmax_size大小和h相同
        # softmax_biases: [token_vocab_size]
        # output: [b, h] * [softmax_size, token_vocab_size] + [token_vocab_size] -> [b, token_vocab_size]
        output = tf.matmul(try_sum, self.weights['softmax_weights']) + self.weights['softmax_biases']

        self.output = output
        # [b, token_vocab_size]
        return output

    # 2
    def make_train_step(self):
        trainable_vars = self.sess.graph.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        if self.args.freeze_graph_model:
            graph_vars = set(self.sess.graph.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="graph_model"))
            filtered_vars = []
            for var in trainable_vars:
                if var not in graph_vars:
                    filtered_vars.append(var)
                else:
                    print("Freezing weights of variable %s." % var.name)
            trainable_vars = filtered_vars

        # 学习率自动衰减
        batch = tf.Variable(0)
        learning_rate = tf.train.exponential_decay(self.params['learning_rate'], batch, 2000, 1.0, staircase=False)

        optimizer = tf.train.MomentumOptimizer(learning_rate, momentum=self.params['momentum'], use_nesterov=True)

        # 找出全部的梯度

        grads_and_vars = optimizer.compute_gradients(self.ops['loss'], var_list=trainable_vars)
        clipped_grads = []

        # 进行梯度削减
        for grad, var in grads_and_vars:
            if grad is not None:
                clipped_grads.append((tf.clip_by_norm(grad, self.params['clamp_gradient_norm']), var))
            else:
                clipped_grads.append((grad, var))
        self.ops['train_step'] = optimizer.apply_gradients(clipped_grads, global_step=batch)
        # Initialize newly-introduced variables
        self.sess.run(tf.local_variables_initializer())

    # 3.读取模型，将模型的变量和参数取出，将每个变量赋上存储的值
    # 如果出现没有对上号的变量，则随机初始化
    def restore_model(self, path: str) -> None:
        print("Restoring weights from file %s." % path)
        with open(path, 'rb') as in_file:
            data_to_load = pickle.load(in_file)

        # Assert that we got the same model configuration
        assert len(self.params) == len(data_to_load['params'])
        for (par, par_value) in self.params.items():
            # Fine to have different task_ids
            if par not in ['task_ids', 'num_epochs']:
                assert par_value == data_to_load['params'][par]

        variables_to_initialize = []
        with tf.name_scope("restore"):
            restore_ops = []
            used_vars = set()
            for variable in self.sess.graph.get_collection(tf.GraphKeys.GLOBAL_VARIABLES):
                used_vars.add(variable.name)
                if variable.name in data_to_load['weights']:
                    # 将读取到的weights赋值给对应的variable
                    restore_ops.append(variable.assign(data_to_load['weights'][variable.name]))
                else:
                    print("Freshly initializing %s since no saved value was found." % variable.name)
                    variables_to_initialize.append(variable)
            for var_name in data_to_load['weights']:
                if var_name not in used_vars:
                    print("Saved weights for %s not used by model." % var_name)

            # 将所有要初始化的变量合并到一起
            restore_ops.append(tf.variables_initializer(variables_to_initialize))
            self.sess.run(restore_ops)

    # 3.
    def restore_model2(self, path: str) -> None:
        print("Restore...")
        # self.sess = tf.Session()
        self.saver.restore(self.sess, path)
        print("Restore done!")

    # 4.初始化模型参数
    def initialize_model(self) -> None:
        init_op = tf.group(tf.global_variables_initializer(),
                           tf.local_variables_initializer())
        self.sess.run(init_op)

    # train0
    def train(self):
        total_time_start = time.time()
        with self.graph.as_default():
            if self.args.restore is not None:
                # train1
                valid_loss, valid_accs, valid_speed, valid_batch, valid_total_graph, valid_time, valid_read_data_time = self.run_epoch(
                    "Resumed (validation)", self.mini_valid_data, False)
                best_val_acc = np.sum(valid_accs)
                best_val_acc_epoch = 0
                print("\r\x1b[KResumed operation, initial cum. val. acc: %.5f" % best_val_acc)
            else:
                (best_val_acc, best_val_acc_epoch) = (float("-inf"), 0)
            for epoch in range(1, self.params['num_epochs'] + 1):
                print("== Epoch %i" % epoch)
                train_loss, train_accs, train_speed, train_batch, train_total_graph, train_time, train_read_data_time = self.run_epoch(
                    "epoch %i (training)" % epoch,
                    self.mini_train_data, True)
                accs_str = " ".join(["%i:%.5f" % (id, acc) for (id, acc) in zip(self.params['task_ids'], train_accs)])
                print(
                    "\r\x1b[K Train: loss: %.5f | acc: %s | instances/sec: %.2f | train_batch: %i | train_total_graph: %i | train_time: %s | train_read_data_time: %s" % (
                        train_loss,
                        accs_str,
                        train_speed, train_batch, train_total_graph, train_time, train_read_data_time))
                valid_loss, valid_accs, valid_speed, valid_batch, valid_total_graph, valid_time, valid_read_data_time = self.run_epoch(
                    "epoch %i (validation)" % epoch,
                    self.mini_valid_data, False)
                accs_str = " ".join(["%i:%.5f" % (id, acc) for (id, acc) in zip(self.params['task_ids'], valid_accs)])
                print(
                    "\r\x1b[K Valid: loss: %.5f | acc: %s | instances/sec: %.2f | valid_batch: %i | valid_total_graph: %i | valid_time: %s | valid_read_data_time: %s" % (
                        valid_loss,
                        accs_str,
                        valid_speed, valid_batch, valid_total_graph, valid_time, valid_read_data_time))

                epoch_time = time.time() - total_time_start

                val_acc = np.sum(valid_accs)  # type: float
                if val_acc > best_val_acc:
                    # train2
                    self.save_model2(DefaultConfig.restore_model_path)
                    print("  (Best epoch so far, cum. val. acc increased to %.5f from %.5f. Saving to '%s')" % (
                        val_acc, best_val_acc, self.best_model_file))
                    best_val_acc = val_acc
                    best_val_acc_epoch = epoch
                elif epoch - best_val_acc_epoch >= self.params['patience']:
                    print("Stopping training after %i epochs without improvement on validation accuracy." % self.params[
                        'patience'])
                    break

    # train1
    def run_epoch(self, epoch_name: str, data, is_training: bool):
        loss = 0
        accuracies = []
        accuracy_ops = [self.ops['accuracy_task%i' % task_id] for task_id in self.params['task_ids']]
        start_time = time.time()
        read_data_time = 0
        total = 0
        processed_graphs = 0
        count = 0
        index = 0
        if is_training:
            file_count = self.training_file_count
            # 训练数据的路径
            prefix_path = DefaultConfig.train_data_root
        else:
            # 验证集的路径
            file_count = self.valid_file_count
            prefix_path = DefaultConfig.valid_data_root

        while count < file_count:
            tempGraph = 0
            tempAcc = []
            full_path = prefix_path + str(count + index) + ".json"
            if is_training:
                filestr = "training"
            else:
                filestr = "valid"
            t = time.time()
            # train1-1
            data = self.load_minidata(full_path, is_training_data=is_training)
            read_data_time = time.time() - t + read_data_time
            count = count + 1
            # train1-2
            batch_iterator = ThreadedIterator(self.make_minibatch_iterator(data, is_training), max_queue_size=5)
            for step, batch_data in enumerate(batch_iterator):
                total = total + 1
                num_graphs = batch_data[self.placeholders['num_graphs']]
                # 记录已处理graphs数
                processed_graphs += num_graphs
                tempGraph += num_graphs
                if is_training:
                    batch_data[self.placeholders['out_layer_dropout_keep_prob']] = self.params[
                        'out_layer_dropout_keep_prob']
                    fetch_list = [self.ops['loss'], accuracy_ops, self.ops['train_step']]
                else:
                    batch_data[self.placeholders['out_layer_dropout_keep_prob']] = 1.0
                    fetch_list = [self.ops['loss'], accuracy_ops]
                result = self.sess.run(fetch_list, feed_dict=batch_data)
                (batch_loss, batch_accuracies) = (result[0], result[1])
                loss += batch_loss * num_graphs
                accuracies.append(np.array(batch_accuracies) * num_graphs)
                tempAcc.append(np.array(batch_accuracies) * num_graphs)

                print("Running %s, %s file %i, batch %i (has %i graphs). Loss so far: %.4f" % (
                    epoch_name, filestr, count, total,
                    num_graphs, loss / processed_graphs), end='\r')
            del data
            del batch_iterator
            gc.collect()
        accuracies = np.sum(accuracies, axis=0) / processed_graphs
        loss = loss / processed_graphs
        end_time = time.time() - start_time
        m, s = divmod(end_time, 60)
        h, m = divmod(m, 60)
        time_str = "%02d:%02d:%02d" % (h, m, s)
        m, s = divmod(read_data_time, 60)
        h, m = divmod(m, 60)
        read_data_time_str = "%02d:%02d:%02d" % (h, m, s)
        instance_per_sec = processed_graphs / (time.time() - start_time - read_data_time)
        return loss, accuracies, instance_per_sec, total, processed_graphs, time_str, read_data_time_str

    # train1-1
    def load_minidata(self, filename, is_training_data: bool):
        # 读取图数据信息
        with open(filename, 'r') as f:
            data = json.load(f)
            # Get some common data out
        num_fwd_edge_types = 0
        for g in data:
            # 找出数据集中节点总数
            # 从点边点组合中的两部分点找
            self.max_num_vertices = max(self.max_num_vertices, max([v for e in g['graph'] for v in [e[0], e[2]]]))
            # 找出前向边的种类
            num_fwd_edge_types = max(num_fwd_edge_types, max([e[1] for e in g['graph']]))
        # 找出边的种类，如果是有向图需要x2，而无向图x1
        self.num_edge_types = max(self.num_edge_types, num_fwd_edge_types * (1 if self.params['tie_fwd_bkwd'] else 2))

        return self.process_raw_graphs(data, is_training_data)

    # train1-2
    def make_minibatch_iterator(self, data, is_training: bool):
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
            num_graphs = len(batch_data['labels'])

            if len(batch_data['labels']) == 0:
                continue
            batch_feed_dict = {
                self.placeholders['token_input_orders']: batch_data['token_orders'],
                self.placeholders['type_input_orders']: batch_data['type_orders'],
                self.placeholders['token_length']: batch_data['token_length'],
                self.placeholders['target_values']: np.transpose(batch_data['labels'], axes=[1, 0]),
                self.placeholders['target_mask']: np.transpose(batch_data['task_masks'], axes=[1, 0]),
                self.placeholders['num_graphs']: num_graphs,  # 多少个数据
                self.placeholders['num_vertices']: bucket_sizes[bucket],  # 点数量（补0后）
                self.placeholders['adjacency_matrix']: batch_data['adj_mat'],  # 邻接矩阵
                self.placeholders['node_mask']: batch_data['node_mask'],  # 节点mask
                self.placeholders['graph_state_keep_prob']: dropout_keep_prob,
                self.placeholders['edge_weight_dropout_keep_prob']: dropout_keep_prob
            }
            bucket_counters[bucket] += 1
            yield batch_feed_dict

    # train2
    def save_model2(self, path: str) -> None:
        self.saver.save(self.sess, path + self.best_model_checkpoint)

    # evaluation-0 验证集测试
    def example_evaluation(self):
        n_example_molecules = 10
        with open(DefaultConfig.valid_data_root, 'r') as valid_file:
            example_molecules = json.load(valid_file)[:n_example_molecules]
        example_molecules, _, _ = self.process_raw_graphs(example_molecules, is_training_data=False,
                                                          bucket_sizes=np.array([29]))
        # evaluation-1
        batch_data = self.make_batch(example_molecules[0])
        print(self.evaluate_one_batch(batch_data['init'], batch_data['orders'], batch_data['adj_mat']))

    # evaluation-1
    def make_batch(self, elements):
        batch_data = {'adj_mat': [], 'token_orders': [], 'type_orders': [], 'labels': [], 'node_mask': [],
                      'task_masks': [], 'token_length': [], 'variable_masks': []}

        # 逐数据将信息填入
        for d in elements:
            batch_data['adj_mat'].append(d['adj_mat'])
            # batch_data['orders'].append(d['orders'])
            batch_data['token_orders'].append(d['token_orders'])
            batch_data['type_orders'].append(d['type_orders'])
            batch_data['token_length'].extend(d['token_length'])
            batch_data['node_mask'].append(d['mask'])
            target_task_values = []
            target_task_mask = []

            for target_val in d['labels']:
                if target_val is None:
                    target_task_values.append(0.)
                    target_task_mask.append(0.)
                else:
                    target_task_values.append(target_val)
                    target_task_mask.append(1.)

            batch_data['labels'].append(target_task_values)
            batch_data['task_masks'].append(target_task_mask)
        return batch_data

    # 读取数据集
    def load_data(self, file_name, is_training_data: bool):
        full_path = os.path.join(self.data_dir, file_name)

        print("Loading data from %s" % full_path)
        with open(full_path, 'r') as f:
            data = json.load(f)

        # 判断是否数据需要缩减
        restrict = self.args.restrict
        if restrict is not None and restrict > 0:
            data = data[:restrict]

        # 从原始数据中抽取信息
        num_fwd_edge_types = 0
        for g in data:
            # 找出数据集中节点总数
            # 从点边点组合中的两部分点找
            self.max_num_vertices = max(self.max_num_vertices, max([v for e in g['graph'] for v in [e[0], e[2]]]))
            # 找出前向边的种类
            num_fwd_edge_types = max(num_fwd_edge_types, max([e[1] for e in g['graph']]))
        # 找出边的种类，如果是有向图需要x2，而无向图x1
        self.num_edge_types = max(self.num_edge_types, num_fwd_edge_types * (1 if self.params['tie_fwd_bkwd'] else 2))

        # 把原始数据进行了处理
        return self.process_raw_graphs(data, is_training_data)
        # return data

    @staticmethod
    def graph_string_to_array(graph_string: str) -> List[List[int]]:
        return [[int(v) for v in s.split(' ')]
                for s in graph_string.split('\n')]

    # 补零 [b, v, annotation_size] -> [b, v, h_dim]
    def pad_annotations(self, annotations):
        return np.pad(annotations, pad_width=[[0, 0], [0, 0], [0, self.params['hidden_size'] - self.annotation_size]],
                      mode='constant')


Parser = namedtuple('parser', ['data_dir', 'config_file', 'config', 'restore', 'restrict', 'freeze_graph_model'])
parser = Parser(data_dir='', config_file=None, config=None, restore=None,
                restrict=None, freeze_graph_model=False)

model = DenseGGNNProgModel(parser, training_file_count=1, valid_file_count=1)
evaluation = False
if evaluation:
    model.example_evaluation()
else:
    model.train()
