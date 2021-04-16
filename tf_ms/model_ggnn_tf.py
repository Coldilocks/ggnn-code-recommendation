#!/usr/bin/env/python
"""
Usage:
    chem_tensorflow_dense.py [options]

Options:
    -h --help                Show this screen.
    --config-file FILE       Hyperparameter configuration file path (in JSON format)
    --config CONFIG          Hyperparameter configuration dictionary (in JSON format)
    --log_dir NAME           log dir name
    --data_dir NAME          data dir name
    --restore FILE           File to restore weights from.
    --freeze-graph-model     Freeze weights of graph model components.
    --evaluate               example evaluation mode using a restored model
"""


from collections import defaultdict
import numpy as np
import tensorflow as tf


import json

from tf_version.utils import glorot_init, MLP, ThreadedIterator, SMALL_NUMBER

import os
import pickle
import random
import time
from typing import Any, Sequence

# 1.1.1
def graph_to_adj_mat(graph, max_n_vertices, num_edge_types, tie_fwd_bkwd=True):
    bwd_edge_offset = 0 if tie_fwd_bkwd else (num_edge_types // 2)
    amat = np.zeros((num_edge_types, max_n_vertices, max_n_vertices))
    for src, e, dest in graph:
        amat[e - 1, dest, src] = 1
        amat[e - 1 + bwd_edge_offset, src, dest] = 1
    return amat


'''
Comments provide the expected tensor shapes where helpful.

Key to symbols in comments:
---------------------------
[...]:  a tensor
; ; :   a list
b:      batch size
e:      number of edge types (4)
v:      number of vertices per graph in this batch
h:      GNN hidden size
'''


class DenseGGNNModel():
    @classmethod
    def default_params(cls):
        return {
            'num_epochs': 3000,
            'patience': 25,
            'learning_rate': 0.001,
            'clamp_gradient_norm': 1.0,
            'out_layer_dropout_keep_prob': 1.0,

            'hidden_size': 100,
            'num_timesteps': 4,
            'use_graph': True,

            'tie_fwd_bkwd': True,
            'task_ids': [0],

            'random_seed': 0,

            'train_file': 'molecules_train.json',
            'valid_file': 'molecules_valid.json',

            'batch_size': 256,
            'graph_state_dropout_keep_prob': 1.,
            'task_sample_ratios': {},
            'use_edge_bias': True,
            'edge_weight_dropout_keep_prob': 1
        }

    # 0. initialize
    def __init__(self, args):
        self.args = args

        # Collect argument things:
        data_dir = ''
        if '--data_dir' in args and args['--data_dir'] is not None:
            data_dir = args['--data_dir']
        self.data_dir = data_dir

        self.run_id = "_".join([time.strftime("%Y-%m-%d-%H-%M-%S"), str(os.getpid())])
        log_dir = args.get('--log_dir') or '.'
        os.makedirs(log_dir, exist_ok=True)
        self.log_file = os.path.join(log_dir, "%s_log.json" % self.run_id)
        self.best_model_file = os.path.join(log_dir, "%s_model_best.pickle" % self.run_id)
        tb_log_dir = os.path.join(log_dir, "tb", self.run_id)
        os.makedirs(tb_log_dir, exist_ok=True)

        # Collect parameters:
        params = self.default_params()
        config_file = args.get('--config-file')
        if config_file is not None:
            with open(config_file, 'r') as f:
                params.update(json.load(f))
        config = args.get('--config')
        if config is not None:
            params.update(json.loads(config))
        self.params = params
        with open(os.path.join(log_dir, "%s_params.json" % self.run_id), "w") as f:
            json.dump(params, f)
        print("Run %s starting with following parameters:\n%s" % (self.run_id, json.dumps(self.params)))
        random.seed(params['random_seed'])
        np.random.seed(params['random_seed'])

        # Load data:
        self.max_num_vertices = 0
        self.num_edge_types = 0
        self.annotation_size = 0
        # 1
        self.train_data = self.load_data(params['train_file'], is_training_data=True)
        self.valid_data = self.load_data(params['valid_file'], is_training_data=False)

        # Build the actual model
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.graph = tf.Graph()
        self.sess = tf.Session(graph=self.graph, config=config)
        with self.graph.as_default():
            tf.set_random_seed(params['random_seed'])
            self.placeholders = {}
            self.weights = {}
            self.ops = {}
            # 2
            self.make_model()
            # 3
            self.make_train_step()
            # 4
            self.make_summaries()

            # Restore/initialize variables:
            restore_file = args.get('--restore')
            if restore_file is not None:
                # 5
                self.train_step_id, self.valid_step_id = self.restore_progress(restore_file)
            else:
                # 6
                self.initialize_model()
                self.train_step_id = 0
                self.valid_step_id = 0
            self.train_writer = tf.summary.FileWriter(os.path.join(tb_log_dir, 'train'), graph=self.graph)
            self.valid_writer = tf.summary.FileWriter(os.path.join(tb_log_dir, 'validation'), graph=self.graph)

    # 1 load data
    def load_data(self, file_name, is_training_data: bool):
        full_path = os.path.join(self.data_dir, file_name)

        print("Loading data from %s" % full_path)
        with open(full_path, 'r') as f:
            data = json.load(f)

        restrict = self.args.get("--restrict_data")
        if restrict is not None and restrict > 0:
            data = data[:restrict]

        # Get some common data out:
        num_fwd_edge_types = 0
        for g in data:
            # 统计三元组(点边点)中最大的顶点id
            self.max_num_vertices = max(self.max_num_vertices, max([v for e in g['graph'] for v in [e[0], e[2]]]))
            # 边的数量
            num_fwd_edge_types = max(num_fwd_edge_types, max([e[1] for e in g['graph']]))
        self.num_edge_types = max(self.num_edge_types, num_fwd_edge_types * (1 if self.params['tie_fwd_bkwd'] else 2))
        self.annotation_size = max(self.annotation_size, len(data[0]["node_features"][0]))

        # 1.1
        return self.process_raw_graphs(data, is_training_data)

    # 1.1 Data preprocessing and chunking into minibatches:
    def process_raw_graphs(self, raw_data: Sequence[Any], is_training_data: bool, bucket_sizes=None) -> Any:
        # 生成不同的bucket, 将node数量相近的数据分到一组，之后填充零
        if bucket_sizes is None:
            bucket_sizes = np.array(list(range(4, 28, 2)) + [29])
        bucketed = defaultdict(list)
        # node_feature的维度
        x_dim = len(raw_data[0]["node_features"][0])
        for d in raw_data:
            chosen_bucket_idx = np.argmax(bucket_sizes > max([v for e in d['graph']
                                                              for v in [e[0], e[2]]]))
            chosen_bucket_size = bucket_sizes[chosen_bucket_idx]
            n_active_nodes = len(d["node_features"])
            # 1.1.1
            bucketed[chosen_bucket_idx].append({
                'adj_mat': graph_to_adj_mat(d['graph'], chosen_bucket_size, self.num_edge_types,
                                            self.params['tie_fwd_bkwd']),
                'init': d["node_features"] + [[0 for _ in range(x_dim)] for __ in
                                              range(chosen_bucket_size - n_active_nodes)],
                'labels': [d["targets"][task_id][0] for task_id in self.params['task_ids']],
                'mask': [1. for _ in range(n_active_nodes)] + [0. for _ in range(chosen_bucket_size - n_active_nodes)]
            })

        if is_training_data:
            for (bucket_idx, bucket) in bucketed.items():
                np.random.shuffle(bucket)
                for task_id in self.params['task_ids']:
                    task_sample_ratio = self.params['task_sample_ratios'].get(str(task_id))
                    if task_sample_ratio is not None:
                        ex_to_sample = int(len(bucket) * task_sample_ratio)
                        for ex_id in range(ex_to_sample, len(bucket)):
                            bucket[ex_id]['labels'][task_id] = None

        bucket_at_step = [[bucket_idx for _ in range(len(bucket_data) // self.params['batch_size'])]
                          for bucket_idx, bucket_data in bucketed.items()]
        bucket_at_step = [x for y in bucket_at_step for x in y]

        # bucketed: bucket_size_key -> [data1, data2, ...], 将node数量相近的graph分到一组，之后填充零;
        #   data中包含了adj_mat、init、labels、mask
        # bucket_sizes: [bucket_size_key1, bucket_size_key2, ...], bucket_size_key代表这个bucket的最大node数
        # bucket_at_step
        return (bucketed, bucket_sizes, bucket_at_step)

    # 2
    def make_model(self):
        # target_values: shape=(1, None)
        self.placeholders['target_values'] = tf.placeholder(tf.float32, [len(self.params['task_ids']), None],
                                                            name='target_values')
        # target_mask: shape=(1, None)
        self.placeholders['target_mask'] = tf.placeholder(tf.float32, [len(self.params['task_ids']), None],
                                                          name='target_mask')
        # num_graphs: shape=()
        self.placeholders['num_graphs'] = tf.placeholder(tf.int32, [], name='num_graphs')
        # out_layer_dropout_keep_prob: shape=()
        self.placeholders['out_layer_dropout_keep_prob'] = tf.placeholder(tf.float32, [], name='out_layer_dropout_keep_prob')

        with tf.variable_scope("graph_model"):
            # 2.1
            self.prepare_specific_graph_model()
            # This does the actual graph work:
            if self.params['use_graph']:
                # 2.2
                # final_node_representations: [b, v, h]
                self.ops['final_node_representations'] = self.compute_final_node_representations()
            else:
                self.ops['final_node_representations'] = tf.zeros_like(self.placeholders['initial_node_representation'])

        self.ops['losses'] = []
        for (internal_id, task_id) in enumerate(self.params['task_ids']):
            with tf.variable_scope("out_layer_task%i" % task_id):
                with tf.variable_scope("regression_gate"):
                    self.weights['regression_gate_task%i' % task_id] = MLP(2 * self.params['hidden_size'], 1, [],
                                                                           self.placeholders['out_layer_dropout_keep_prob'])
                with tf.variable_scope("regression"):
                    self.weights['regression_transform_task%i' % task_id] = MLP(self.params['hidden_size'], 1, [],
                                                                                self.placeholders['out_layer_dropout_keep_prob'])
                # 2.3
                # computed_values: [b]
                computed_values = self.gated_regression(self.ops['final_node_representations'],
                                                        self.weights['regression_gate_task%i' % task_id],
                                                        self.weights['regression_transform_task%i' % task_id])
                # diff: [b]
                diff = computed_values - self.placeholders['target_values'][internal_id, :]
                # task_target_mask: [b]
                task_target_mask = self.placeholders['target_mask'][internal_id, :]
                task_target_num = tf.reduce_sum(task_target_mask) + SMALL_NUMBER
                # diff: [b] * [b] -> [b]
                diff = diff * task_target_mask  # Mask out unused values
                self.ops['accuracy_task%i' % task_id] = tf.reduce_sum(tf.abs(diff)) / task_target_num
                task_loss = tf.reduce_sum(0.5 * tf.square(diff)) / task_target_num
                # Normalise loss to account for fewer task-specific examples in batch:
                task_loss = task_loss * (1.0 / (self.params['task_sample_ratios'].get(task_id) or 1.0))
                self.ops['losses'].append(task_loss)
        self.ops['loss'] = tf.reduce_sum(self.ops['losses'])

    # 2.1
    def prepare_specific_graph_model(self) -> None:
        h_dim = self.params['hidden_size']
        # inputs
        self.placeholders['graph_state_keep_prob'] = tf.placeholder(tf.float32, None, name='graph_state_keep_prob')
        self.placeholders['edge_weight_dropout_keep_prob'] = tf.placeholder(tf.float32, None,
                                                                            name='edge_weight_dropout_keep_prob')
        # initial_node_representation: shape=(None, None, hidden_size), [b, v, h]
        self.placeholders['initial_node_representation'] = tf.placeholder(tf.float32,
                                                                          [None, None, self.params['hidden_size']],
                                                                          name='node_features')
        # node_mask, shape=(None, None), [b, v]
        self.placeholders['node_mask'] = tf.placeholder(tf.float32, [None, None], name='node_mask')
        self.placeholders['num_vertices'] = tf.placeholder(tf.int32, ())
        # adjacency_matrix: shape=(None, num_edge_types, None, None), [b, e, v, v]
        self.placeholders['adjacency_matrix'] = tf.placeholder(tf.float32,
                                                               [None, self.num_edge_types, None, None])
        # [b, e, v, v] -> [e, b, v, v]
        self.__adjacency_matrix = tf.transpose(self.placeholders['adjacency_matrix'], [1, 0, 2, 3])

        # weights
        # edge_weights: [e, h, h]
        self.weights['edge_weights'] = tf.Variable(glorot_init([self.num_edge_types, h_dim, h_dim]))
        if self.params['use_edge_bias']:
            # edge_biases: [e, 1, h]
            self.weights['edge_biases'] = tf.Variable(np.zeros([self.num_edge_types, 1, h_dim]).astype(np.float32))
        with tf.variable_scope("gru_scope"):
            cell = tf.contrib.rnn.GRUCell(h_dim)
            cell = tf.nn.rnn_cell.DropoutWrapper(cell,
                                                 state_keep_prob=self.placeholders['graph_state_keep_prob'])
            self.weights['node_gru'] = cell

    # 2.2
    def compute_final_node_representations(self) -> tf.Tensor:
        v = self.placeholders['num_vertices']
        h_dim = self.params['hidden_size']
        # initial_node_representation, [b, v, h] (等号左边的h和维度h不一样，维度h是h_dim or hidden_size)
        h = self.placeholders['initial_node_representation']
        # [b * v, h]
        h = tf.reshape(h, [-1, h_dim])

        with tf.variable_scope("gru_scope") as scope:
            for i in range(self.params['num_timesteps']):
                if i > 0:
                    tf.get_variable_scope().reuse_variables()
                for edge_type in range(self.num_edge_types):
                    # [b * v, h] * [h, h] -> [b*v, h]
                    m = tf.matmul(h, tf.nn.dropout(self.weights['edge_weights'][edge_type],
                                                   keep_prob=self.placeholders[
                                                       'edge_weight_dropout_keep_prob']))  # [b*v, h]
                    m = tf.reshape(m, [-1, v, h_dim])  # [b, v, h]
                    if self.params['use_edge_bias']:
                        m += self.weights['edge_biases'][edge_type]  # [b, v, h]
                    if edge_type == 0:
                        # [b, v, v] * [b, v, h] -> [b, v, h]
                        acts = tf.matmul(self.__adjacency_matrix[edge_type], m)
                    else:
                        acts += tf.matmul(self.__adjacency_matrix[edge_type], m)
                acts = tf.reshape(acts, [-1, h_dim])  # [b, v, h] -> [b*v, h]
                # output,state=GRUCell(inputs,previous_state)
                # acts: inputs, h: previous state
                h = self.weights['node_gru'](acts, h)[1]  # [b*v, h]
            last_h = tf.reshape(h, [-1, v, h_dim])  # [b, v, h]
        return last_h

    # 2.3
    def gated_regression(self, last_h, regression_gate, regression_transform):
        # last_h: [b, v, h]
        # initial_node_representation: [b, v, h]
        # gate_input: [b, v, 2*h]
        gate_input = tf.concat([last_h, self.placeholders['initial_node_representation']], axis=2)  # [b, v, 2h]
        # [b, v, 2*h] -> [b*v, 2*h]
        gate_input = tf.reshape(gate_input, [-1, 2 * self.params["hidden_size"]])  # [b*v, 2h]
        # [b, v, h] -> [b*v, h]
        last_h = tf.reshape(last_h, [-1, self.params["hidden_size"]])  # [b*v, h]
        # regression_gate(gate_input): [b*v, 1]
        # regression_transform(last_h): [b*v, 1]
        # gated_outputs: [b*v, 1]
        gated_outputs = tf.nn.sigmoid(regression_gate(gate_input)) * regression_transform(last_h)  # [b*v, 1]
        # [b*v, 1] -> [b, v]
        gated_outputs = tf.reshape(gated_outputs, [-1, self.placeholders['num_vertices']])  # [b, v]
        # node_mask: [b, v]
        # masked_gated_outputs: [b, v] * [b, v] -> [b, v]
        masked_gated_outputs = gated_outputs * self.placeholders['node_mask']  # [b x v]
        # [b, v] -> [b]
        output = tf.reduce_sum(masked_gated_outputs, axis=1)  # [b]
        self.output = output
        return output

    # 3
    def make_train_step(self):
        trainable_vars = self.sess.graph.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        if self.args.get('--freeze-graph-model'):
            graph_vars = set(self.sess.graph.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="graph_model"))
            filtered_vars = []
            for var in trainable_vars:
                if var not in graph_vars:
                    filtered_vars.append(var)
                else:
                    print("Freezing weights of variable %s." % var.name)
            trainable_vars = filtered_vars
        optimizer = tf.train.AdamOptimizer(self.params['learning_rate'])
        grads_and_vars = optimizer.compute_gradients(self.ops['loss'], var_list=trainable_vars)
        clipped_grads = []
        for grad, var in grads_and_vars:
            if grad is not None:
                clipped_grads.append((tf.clip_by_norm(grad, self.params['clamp_gradient_norm']), var))
            else:
                clipped_grads.append((grad, var))
        self.ops['train_step'] = optimizer.apply_gradients(clipped_grads)
        # Initialize newly-introduced variables:
        self.sess.run(tf.local_variables_initializer())

    # 4
    def make_summaries(self):
        with tf.name_scope('summary'):
            tf.summary.scalar('loss', self.ops['loss'])
            for task_id in self.params['task_ids']:
                tf.summary.scalar('accuracy%i' % task_id, self.ops['accuracy_task%i' % task_id])
        self.ops['summary'] = tf.summary.merge_all()

    # 5
    def restore_progress(self, model_path: str) -> (int, int):
        print("Restoring weights from file %s." % model_path)
        with open(model_path, 'rb') as in_file:
            data_to_load = pickle.load(in_file)

        # Assert that we got the same model configuration
        assert len(self.params) == len(data_to_load['params'])
        for (par, par_value) in self.params.items():
            # Fine to have different task_ids:
            if par not in ['task_ids', 'num_epochs']:
                assert par_value == data_to_load['params'][par]

        variables_to_initialize = []
        with tf.name_scope("restore"):
            restore_ops = []
            used_vars = set()
            for variable in self.sess.graph.get_collection(tf.GraphKeys.GLOBAL_VARIABLES):
                used_vars.add(variable.name)
                if variable.name in data_to_load['weights']:
                    restore_ops.append(variable.assign(data_to_load['weights'][variable.name]))
                else:
                    print('Freshly initializing %s since no saved value was found.' % variable.name)
                    variables_to_initialize.append(variable)
            for var_name in data_to_load['weights']:
                if var_name not in used_vars:
                    print('Saved weights for %s not used by model.' % var_name)
            restore_ops.append(tf.variables_initializer(variables_to_initialize))
            self.sess.run(restore_ops)

        return data_to_load['train_step'], data_to_load['valid_step']

    # 6
    def initialize_model(self) -> None:
        init_op = tf.group(tf.global_variables_initializer(),
                           tf.local_variables_initializer())
        self.sess.run(init_op)

    # train
    def train(self):
        log_to_save = []
        total_time_start = time.time()
        with self.graph.as_default():
            if self.args.get('--restore') is not None:
                # train-1
                _, valid_accs, _, _, steps = self.run_epoch("Resumed (validation)", self.valid_data, False)
                best_val_acc = np.sum(valid_accs)
                best_val_acc_epoch = 0
                print("\r\x1b[KResumed operation, initial cum. val. acc: %.5f" % best_val_acc)
            else:
                (best_val_acc, best_val_acc_epoch) = (float("+inf"), 0)
            for epoch in range(1, self.params['num_epochs'] + 1):
                print("== Epoch %i" % epoch)
                train_loss, train_accs, train_errs, train_speed, train_steps = self.run_epoch("epoch %i (training)" % epoch,
                                                                                              self.train_data, True, self.train_step_id)
                self.train_step_id += train_steps
                accs_str = " ".join(["%i:%.5f" % (id, acc) for (id, acc) in zip(self.params['task_ids'], train_accs)])
                errs_str = " ".join(["%i:%.5f" % (id, err) for (id, err) in zip(self.params['task_ids'], train_errs)])
                print("\r\x1b[K Train: loss: %.5f | acc: %s | error_ratio: %s | instances/sec: %.2f" % (train_loss,
                                                                                                        accs_str,
                                                                                                        errs_str,
                                                                                                        train_speed))
                valid_loss, valid_accs, valid_errs, valid_speed, valid_steps = self.run_epoch("epoch %i (validation)" % epoch,
                                                                                              self.valid_data, False, self.valid_step_id)
                self.valid_step_id += valid_steps
                accs_str = " ".join(["%i:%.5f" % (id, acc) for (id, acc) in zip(self.params['task_ids'], valid_accs)])
                errs_str = " ".join(["%i:%.5f" % (id, err) for (id, err) in zip(self.params['task_ids'], valid_errs)])
                print("\r\x1b[K Valid: loss: %.5f | acc: %s | error_ratio: %s | instances/sec: %.2f" % (valid_loss,
                                                                                                        accs_str,
                                                                                                        errs_str,
                                                                                                        valid_speed))

                epoch_time = time.time() - total_time_start
                log_entry = {
                    'epoch': epoch,
                    'time': epoch_time,
                    'train_results': (train_loss, train_accs.tolist(), train_errs.tolist(), train_speed),
                    'valid_results': (valid_loss, valid_accs.tolist(), valid_errs.tolist(), valid_speed),
                }
                log_to_save.append(log_entry)
                with open(self.log_file, 'w') as f:
                    json.dump(log_to_save, f, indent=4)

                val_acc = np.sum(valid_accs)  # type: float
                if val_acc < best_val_acc:
                    # train-1.2
                    self.save_progress(self.best_model_file, self.train_step_id, self.valid_step_id)
                    print("  (Best epoch so far, cum. val. acc decreased to %.5f from %.5f. Saving to '%s')" % (
                        val_acc, best_val_acc, self.best_model_file))
                    best_val_acc = val_acc
                    best_val_acc_epoch = epoch
                elif epoch - best_val_acc_epoch >= self.params['patience']:
                    print("Stopping training after %i epochs without improvement on validation accuracy." % self.params['patience'])
                    break

    # train-1
    def run_epoch(self, epoch_name: str, data, is_training: bool, start_step: int = 0):
        chemical_accuracies = np.array([0.066513725, 0.012235489, 0.071939046, 0.033730778, 0.033486113, 0.004278493,
                                        0.001330901, 0.004165489, 0.004128926, 0.00409976, 0.004527465, 0.012292586,
                                        0.037467458])

        loss = 0
        accuracies = []
        accuracy_ops = [self.ops['accuracy_task%i' % task_id] for task_id in self.params['task_ids']]
        start_time = time.time()
        processed_graphs = 0
        steps = 0
        # train-1.1
        batch_iterator = ThreadedIterator(self.make_minibatch_iterator(data, is_training), max_queue_size=5)
        for step, batch_data in enumerate(batch_iterator):
            num_graphs = batch_data[self.placeholders['num_graphs']]
            processed_graphs += num_graphs
            if is_training:
                batch_data[self.placeholders['out_layer_dropout_keep_prob']] = self.params['out_layer_dropout_keep_prob']
                fetch_list = [self.ops['loss'], accuracy_ops, self.ops['summary'], self.ops['train_step']]
            else:
                batch_data[self.placeholders['out_layer_dropout_keep_prob']] = 1.0
                fetch_list = [self.ops['loss'], accuracy_ops, self.ops['summary']]
            result = self.sess.run(fetch_list, feed_dict=batch_data)
            (batch_loss, batch_accuracies, batch_summary) = (result[0], result[1], result[2])
            writer = self.train_writer if is_training else self.valid_writer
            writer.add_summary(batch_summary, start_step + step)
            loss += batch_loss * num_graphs
            accuracies.append(np.array(batch_accuracies) * num_graphs)

            print("Running %s, batch %i (has %i graphs). Loss so far: %.4f" % (epoch_name,
                                                                               step,
                                                                               num_graphs,
                                                                               loss / processed_graphs),
                  end='\r')
            steps += 1

        accuracies = np.sum(accuracies, axis=0) / processed_graphs
        loss = loss / processed_graphs
        error_ratios = accuracies / chemical_accuracies[self.params["task_ids"]]
        instance_per_sec = processed_graphs / (time.time() - start_time)
        return loss, accuracies, error_ratios, instance_per_sec, steps

    # train-1.1
    def make_minibatch_iterator(self, data, is_training: bool):
        (bucketed, bucket_sizes, bucket_at_step) = data
        if is_training:
            np.random.shuffle(bucket_at_step)
            for _, bucketed_data in bucketed.items():
                np.random.shuffle(bucketed_data)

        bucket_counters = defaultdict(int)
        dropout_keep_prob = self.params['graph_state_dropout_keep_prob'] if is_training else 1.
        for step in range(len(bucket_at_step)):
            bucket = bucket_at_step[step]
            start_idx = bucket_counters[bucket] * self.params['batch_size']
            end_idx = (bucket_counters[bucket] + 1) * self.params['batch_size']
            elements = bucketed[bucket][start_idx:end_idx]
            # train-1.1.1
            batch_data = self.make_batch(elements)

            num_graphs = len(batch_data['init'])
            initial_representations = batch_data['init']
            # train-1.1.2
            initial_representations = self.pad_annotations(initial_representations)

            batch_feed_dict = {
                self.placeholders['initial_node_representation']: initial_representations,
                self.placeholders['target_values']: np.transpose(batch_data['labels'], axes=[1, 0]),
                self.placeholders['target_mask']: np.transpose(batch_data['task_masks'], axes=[1, 0]),
                self.placeholders['num_graphs']: num_graphs,
                self.placeholders['num_vertices']: bucket_sizes[bucket],
                self.placeholders['adjacency_matrix']: batch_data['adj_mat'],
                self.placeholders['node_mask']: batch_data['node_mask'],
                self.placeholders['graph_state_keep_prob']: dropout_keep_prob,
                self.placeholders['edge_weight_dropout_keep_prob']: dropout_keep_prob
            }

            bucket_counters[bucket] += 1

            yield batch_feed_dict

    # train-1.1.1
    def make_batch(self, elements):
        batch_data = {'adj_mat': [], 'init': [], 'labels': [], 'node_mask': [], 'task_masks': []}
        for d in elements:
            batch_data['adj_mat'].append(d['adj_mat'])
            batch_data['init'].append(d['init'])
            batch_data['node_mask'].append(d['mask'])

            target_task_values = []
            target_task_mask = []
            for target_val in d['labels']:
                if target_val is None:  # This is one of the examples we didn't sample...
                    target_task_values.append(0.)
                    target_task_mask.append(0.)
                else:
                    target_task_values.append(target_val)
                    target_task_mask.append(1.)
            batch_data['labels'].append(target_task_values)
            batch_data['task_masks'].append(target_task_mask)

        return batch_data

    # train-1.1.2
    def pad_annotations(self, annotations):
        return np.pad(annotations,
                      pad_width=[[0, 0], [0, 0], [0, self.params['hidden_size'] - self.annotation_size]],
                      mode='constant')

    # train-2
    def save_progress(self, model_path: str, train_step: int, valid_step: int) -> None:
        weights_to_save = {}
        for variable in self.sess.graph.get_collection(tf.GraphKeys.GLOBAL_VARIABLES):
            assert variable.name not in weights_to_save
            weights_to_save[variable.name] = self.sess.run(variable)

        data_to_save = {
            "params": self.params,
            "weights": weights_to_save,
            "train_step": train_step,
            "valid_step": valid_step,
        }

        with open(model_path, 'wb') as out_file:
            pickle.dump(data_to_save, out_file, pickle.HIGHEST_PROTOCOL)

    # eval
    def example_evaluation(self):
        ''' Demonstration of what test-time code would look like
        we query the model with the first n_example_molecules from the validation file
        '''
        n_example_molecules = 10
        with open('molecules_valid.json', 'r') as valid_file:
            example_molecules = json.load(valid_file)[:n_example_molecules]

        for mol in example_molecules:
            print(mol['targets'])

        example_molecules, _, _ = self.process_raw_graphs(example_molecules,
                                                          is_training_data=False, bucket_sizes=np.array([29]))
        batch_data = self.make_batch(example_molecules[0])
        # eval-1
        print(self.evaluate_one_batch(batch_data['init'], batch_data['adj_mat']))

    # eval-1
    def evaluate_one_batch(self, initial_node_representations, adjacency_matrices, node_masks=None):
        num_vertices = len(initial_node_representations[0])
        if node_masks is None:
            node_masks = []
            for r in initial_node_representations:
                node_masks.append([1. for _ in r] + [0. for _ in range(num_vertices - len(r))])
        batch_feed_dict = {
            self.placeholders['initial_node_representation']: self.pad_annotations(initial_node_representations),
            self.placeholders['num_graphs']: len(initial_node_representations),
            self.placeholders['num_vertices']: len(initial_node_representations[0]),
            self.placeholders['adjacency_matrix']: adjacency_matrices,
            self.placeholders['node_mask']: node_masks,
            self.placeholders['graph_state_keep_prob']: 1.0,
            self.placeholders['out_layer_dropout_keep_prob']: 1.0,
            self.placeholders['edge_weight_dropout_keep_prob']: 1.0
        }

        fetch_list = self.output
        result = self.sess.run(fetch_list, feed_dict=batch_feed_dict)
        return result



