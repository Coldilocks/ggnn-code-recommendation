from typing import List, Any, Sequence
import os
import tensorflow as tf
import numpy as np
import time
import pickle

import json
import random
import gc
from code_rec_api_level.tf_version_2.mlp import MLP
from code_rec_api_level.tf_version_2.util import glorot_init, ThreadedIterator

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class ProgModel(object):

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
            # 'l2_regulation':0.01,
            'momentum': 0.9,

            'embed_size': 300,
            'hidden_size': 300,
            'variable_size': 300,
            'variable_embed_size': 300,
            'softmax_size': 800,
            'num_timesteps': 5,
            'use_graph': True,

            'tie_fwd_bkwd': False,
            'task_ids': [0],
            'random_seed': 0,

            'train_file': 'chendata_CodeWithVariable-1000.json',
            'valid_file': 'chendata_CodeWithVariable-1000.json',
        }

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
        log_dir = '../../tf_api_rec'
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
        # with open(os.path.join(log_dir, "%s_params.json" % self.run_id), "w") as f:
        #    json.dump(params, f)
        print("Run %s starting with following parameters:\n%s" % (self.run_id, json.dumps(self.params)))

        random.seed(params['random_seed'])
        np.random.seed(params['random_seed'])

        # load data
        self.max_num_vertices = 0
        self.num_edge_types = 8
        self.annotation_size = 0
        self.mini_train_data = None
        self.mini_valid_data = None

        # 读取指定文件路径的训练集和验证集
        # self.train_data = self.load_data(params['train_file'], is_training_data=True)
        # self.valid_data = self.load_data(params['valid_file'], is_training_data=False)
        # print('start reading train data...')
        # self.mini_train_data = self.load_minidata(params['train_file'], is_training_data=True)
        # print('train_data : ',len(self.mini_train_data))
        # print('start reading valid data...')
        # self.mini_valid_data = self.load_minidata(params['valid_file'], is_training_data=False)
        # print('valid_data : ',len(self.mini_valid_data))

        # sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

        # Build the actual model
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.log_device_placement = True

        self.graph = tf.Graph()
        self.sess = tf.Session(graph=self.graph, config=config)
        with self.graph.as_default():
            tf.set_random_seed(params['random_seed'])
            self.placeholders = {}
            self.weights = {}
            self.ops = {}

            # make model
            self.make_model()
            # make train step
            self.make_train_step()

            # # Restore/initialize variables
            # restore_file = args.restore
            # if restore_file is not None:
            #     # self.initialize_model()
            #     self.restore_model2('/Users/coldilock/Documents/Code/Github/Code-Recommendation/android_model_server/save/' + restore_file)
            #     # self.restore_model(restore_file)
            # else:
            #     self.initialize_model()
            self.initialize_model()

    # 搭建模型
    def make_model(self):
        self.placeholders['target_values'] = tf.placeholder(tf.int64, [len(self.params['task_ids']), None],
                                                            name='target_values')
        self.placeholders['target_mask'] = tf.placeholder(tf.float32, [len(self.params['task_ids']), None],
                                                          name='target_mask')
        self.placeholders['num_graphs'] = tf.placeholder(tf.int64, [], name='num_graphs')
        self.placeholders['out_layer_dropout_keep_prob'] = tf.placeholder(tf.float32, [],
                                                                          name='out_layer_dropout_keep_prob')
        with tf.variable_scope("graph_model"):

            # 此处搭建具体的模型（虚拟函数）
            self.prepare_specific_graph_model()

            # this does the actual graph work:
            if self.params['use_graph']:
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
                    # self.weights['softmax_weights'] = tf.Variable(glorot_init([self.params['hidden_size']+ self.params['variable_embed_size'],27638]))
                    self.weights['softmax_weights'] = tf.Variable(glorot_init([self.params['softmax_size'], 39070]))
                    self.weights['softmax_biases'] = tf.Variable(np.zeros([39070]).astype(np.float32))
                    print('softmax weights: ', self.weights['softmax_weights'])
                # 计算最后的g函数（输出函数）
                # 这里返回的是一个[b,softmax维度]的结果
                computed_values = self.gated_regression(self.ops['final_node_representations'],
                                                        self.weights['regression_gate_task%i' % task_id],
                                                        self.weights['regression_transform_task%i' % task_id],
                                                        self.weights['softmax_weights'],
                                                        self.weights['softmax_biases'])

                # diff = computed_values - self.placeholders['target_values'][internal_id,:]
                # task_target_mask = self.placeholders['target_mask'][internal_id,:]
                # 加上小值保证非零
                # task_target_num = tf.reduce_sum(task_target_mask) + 1e-7
                # mask out unused values
                # diff = diff * task_target_mask

                groundtruth = self.placeholders['target_values'][internal_id, :]

                # reg = tf.contrib.layers.apply_regularization(tf.contrib.layers.l2_regularizer(self.params['l2_regulation']), tf.trainable_variables())

                loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=computed_values, labels=groundtruth)
                task_loss = tf.reduce_sum(loss)

                correct_prediction = tf.equal(tf.argmax(computed_values, 1), groundtruth)
                correct_count = tf.reduce_sum(tf.cast(correct_prediction, tf.float32))
                accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

                self.ops['accuracy_task%i' % task_id] = accuracy

                self.ops['correct_count'] = correct_count

                # 计算损失函数，平方根和
                # task_loss = tf.reduce_sum(0.5 * tf.square(diff)) / task_target_num

                # normalise loss to account for fewer task-specific example in batch,如果没有就是1.0
                task_loss = task_loss * (1.0 / (self.params['task_sample_ratios'].get(task_id) or 1.0))
                self.ops['losses'].append(task_loss)
        # self.ops['loss'] = tf.reduce_sum(self.ops['losses'])
        self.ops['loss'] = tf.reduce_mean(self.ops['losses'])

        self.saver = tf.train.Saver()

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

        # optimizer = tf.train.AdamOptimizer(self.params['learning_rate'])
        # optimizer = tf.train.AdagradOptimizer(self.params['learning_rate'])

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
        # self.ops['train_step'] = optimizer.apply_gradients(grads_and_vars)
        # Initialize newly-introduced variables
        self.sess.run(tf.local_variables_initializer())

    # 门回归
    def gated_regression(self, last_h, regreesion_gate, regression_transform, softmax_weights, softmax_biases):
        raise Exception("Models have to implement gated_regression!")

    # 具体模型
    def prepare_specific_graph_model(self) -> None:
        raise Exception("Models have to implement prepare_specific_graph_model!")

    def compute_final_node_representations(self) -> tf.Tensor:
        raise Exception("Models have to implement compute_final_node_representations!")

    def make_minibatch_iterator(self, data: Any, is_training: bool):
        raise Exception("Models have to implement make_minibatch_iterator!")

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

        # 标注的维度
        # self.annotation_size = max(self.annotation_size, len(data[0]['node_features'][0]))

        # if is_training_data:
        #    self.raw_train_data = data
        # else:
        #    self.raw_valid_data = data

        # 把原始数据进行了处理
        return self.process_raw_graphs(data, is_training_data)
        # return data

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
        # self.annotation_size = max(self.annotation_size, len(data[0]['node_features'][0]))
        # print('数据边种类',num_fwd_edge_types)
        # if is_training_data:
        #    self.raw_mini_train_data = data
        # else:
        #    self.raw_mini_valid_data = data
        # print(str(filename) + ' : ' + str(len(data)))
        # print('num_edge_types: ',self.num_edge_types)

        return self.process_raw_graphs(data, is_training_data)

    @staticmethod
    def graph_string_to_array(graph_string: str) -> List[List[int]]:
        return [[int(v) for v in s.split(' ')]
                for s in graph_string.split('\n')]

        # 处理原始数据

    def process_raw_graphs(self, raw_data: Sequence[Any], is_training_data: bool) -> Any:
        raise Exception("Models have to implement process_raw_graphs!")

    def run_epoch(self, epoch_name: str, data, is_training: bool):
        # chemical_accuracies = np.array([0.066513725, 0.012235489, 0.071939046, 0.033730778, 0.033486113, 0.004278493,
        #                                0.001330901, 0.004165489, 0.004128926, 0.00409976, 0.004527465, 0.012292586,
        #                                0.037467458])

        loss = 0
        accuracies = []
        accuracy_ops = [self.ops['accuracy_task%i' % task_id] for task_id in self.params['task_ids']]
        start_time = time.time()
        read_data_time = 0
        total = 0
        processed_graphs = 0
        count = 0
        file_count = 0
        index = 0
        prefix_path = ""
        if is_training:
            file_count = self.training_file_count
        #     prefix_path = "/home/x/mydisk/HuaWeiProject/AndroidData/AndroidUsefulTrainingJson/json"
        # else:
        #     file_count = self.valid_file_count
        #     prefix_path = "/home/x/mydisk/HuaWeiProject/AndroidData/AndroidUsefulValidJson/json"

        while count < file_count:
            tempGraph = 0
            tempAcc = []
            # full_path = prefix_path + str(count + index) + ".json"
            full_path = '../../data/small_input.json'
            # t = time.time()
            filestr = None
            if is_training:
                filestr = "training"
            else:
                filestr = "valid"
            t = time.time()
            data = self.load_minidata(full_path, is_training_data=is_training)
            read_data_time = time.time() - t + read_data_time
            # print(time.time()-t)
            count = count + 1
            # t = time.time()
            batch_iterator = ThreadedIterator(self.make_minibatch_iterator(data, is_training), max_queue_size=5)
            # print(time.time()-t)
            for step, batch_data in enumerate(batch_iterator):
                total = total + 1
                num_graphs = batch_data[self.placeholders['num_graphs']]
                # 记录已处理graphs数
                processed_graphs += num_graphs
                tempGraph += num_graphs
                if is_training:
                    batch_data[self.placeholders['out_layer_dropout_keep_prob']] = self.params[
                        'out_layer_dropout_keep_prob']
                    fetch_list = [self.ops['loss'], accuracy_ops, self.ops['correct_count'], self.ops['train_step']]
                else:
                    batch_data[self.placeholders['out_layer_dropout_keep_prob']] = 1.0
                    fetch_list = [self.ops['loss'], accuracy_ops, self.ops['correct_count']]

                # print(batch_data)

                result = self.sess.run(fetch_list, feed_dict=batch_data)
                (batch_loss, batch_accuracies, batch_correct_result) = (result[0], result[1], result[2])
                loss += batch_loss * num_graphs
                accuracies.append(np.array(batch_accuracies) * num_graphs)
                tempAcc.append(np.array(batch_accuracies) * num_graphs)

                print("[Batch %d Acc %.4f Loss %.4f CORR [%d / %d]" % (step+1, batch_accuracies[0], batch_loss, batch_correct_result, num_graphs))

                print("Running %s, %s file %i, batch %i (has %i graphs). Loss so far: %.4f" % (
                    epoch_name, filestr, count, total,
                    num_graphs, loss / processed_graphs), end='\r')
            # data = None
            # batch_iterator = None
            del data
            del batch_iterator
            gc.collect()
            # print(tempGraph, " " ,np.sum(tempAcc, axis=0) / tempGraph)
        if processed_graphs != 0:
            accuracies = np.sum(accuracies, axis=0) / processed_graphs
            loss = loss / processed_graphs
        # error_ratios = accuracies / chemical_accuracies[self.params["task_ids"]]
        end_time = time.time() - start_time
        m, s = divmod(end_time, 60)
        h, m = divmod(m, 60)
        time_str = "%02d:%02d:%02d" % (h, m, s)
        m, s = divmod(read_data_time, 60)
        h, m = divmod(m, 60)
        read_data_time_str = "%02d:%02d:%02d" % (h, m, s)
        instance_per_sec = processed_graphs / (time.time() - start_time - read_data_time)
        return loss, accuracies, instance_per_sec, total, processed_graphs, time_str, read_data_time_str

    def train(self):
        log_to_save = []
        total_time_start = time.time()
        with self.graph.as_default():
            # if self.args.restore is not None:
            #     valid_loss, valid_accs, valid_speed, valid_batch, valid_total_graph, valid_time, valid_read_data_time = self.run_epoch(
            #         "Resumed (validation)", self.mini_valid_data, False)
            #     best_val_acc = np.sum(valid_accs)
            #     best_val_acc_epoch = 0
            #     print("\r\x1b[KResumed operation, initial cum. val. acc: %.5f" % best_val_acc)
            # else:
            #     (best_val_acc, best_val_acc_epoch) = (float("-inf"), 0)
            (best_val_acc, best_val_acc_epoch) = (float("-inf"), 0)
            for epoch in range(1, self.params['num_epochs'] + 1):
                print("== Epoch %i" % epoch)
                train_loss, train_accs, train_speed, train_batch, train_total_graph, train_time, train_read_data_time = self.run_epoch(
                    "epoch %i (training)" % epoch,
                    self.mini_train_data, True)
                accs_str = " ".join(["%i:%.5f" % (id, acc) for (id, acc) in zip(self.params['task_ids'], train_accs)])
                # errs_str = " ".join(["%i:%.5f" % (id, err) for (id, err) in zip(self.params['task_ids'], train_errs)])
                print(
                    "\r\x1b[K Train: loss: %.5f | acc: %s | instances/sec: %.2f | train_batch: %i | train_total_graph: %i | train_time: %s | train_read_data_time: %s" % (
                        train_loss,
                        accs_str,
                        train_speed, train_batch, train_total_graph, train_time, train_read_data_time))
                valid_loss, valid_accs, valid_speed, valid_batch, valid_total_graph, valid_time, valid_read_data_time = self.run_epoch(
                    "epoch %i (validation)" % epoch,
                    self.mini_valid_data, False)
                accs_str = " ".join(["%i:%.5f" % (id, acc) for (id, acc) in zip(self.params['task_ids'], valid_accs)])
                # errs_str = " ".join(["%i:%.5f" % (id, err) for (id, err) in zip(self.params['task_ids'], valid_errs)])
                print(
                    "\r\x1b[K Valid: loss: %.5f | acc: %s | instances/sec: %.2f | valid_batch: %i | valid_total_graph: %i | valid_time: %s | valid_read_data_time: %s" % (
                        valid_loss,
                        accs_str,
                        valid_speed, valid_batch, valid_total_graph, valid_time, valid_read_data_time))

                epoch_time = time.time() - total_time_start
                #                 log_entry = {
                #                     'epoch': epoch,
                #                     'time': epoch_time,
                #                     'train_results': (train_loss, train_accs.tolist(), train_errs.tolist(), train_speed),
                #                     'valid_results': (valid_loss, valid_accs.tolist(), valid_errs.tolist(), valid_speed),
                #                 }
                #                 log_to_save.append(log_entry)
                #                 with open(self.log_file, 'w') as f:
                #                     json.dump(log_to_save, f, indent=4)

                val_acc = np.sum(valid_accs)  # type: float
                if val_acc > best_val_acc:
                    # self.save_model(self.best_model_file)
                    self.save_model2('/Users/coldilock/Documents/Code/Github/Code-Recommendation/android_model_server/save/models/')
                    print("  (Best epoch so far, cum. val. acc increased to %.5f from %.5f. Saving to '%s')" % (
                        val_acc, best_val_acc, self.best_model_file))
                    best_val_acc = val_acc
                    best_val_acc_epoch = epoch
                elif epoch - best_val_acc_epoch >= self.params['patience']:
                    print("Stopping training after %i epochs without improvement on validation accuracy." % self.params[
                        'patience'])
                    break

    # 初始化模型参数
    def initialize_model(self) -> None:
        init_op = tf.group(tf.global_variables_initializer(),
                           tf.local_variables_initializer())
        self.sess.run(init_op)

    # 保存模型，将模型中的变量和参数存入pickle文件
    # 这里倒蛮牛逼的
    def save_model(self, path: str) -> None:
        weights_to_save = {}
        # 利用get_collection获得graph中的变量
        for variable in self.sess.graph.get_collection(tf.GraphKeys.GLOBAL_VARIABLES):
            assert variable.name not in weights_to_save
            weights_to_save[variable.name] = self.sess.run(variable)

        data_to_save = {
            "params": self.params,
            "weights": weights_to_save
        }

        with open(path, 'wb') as out_file:
            pickle.dump(data_to_save, out_file, pickle.HIGHEST_PROTOCOL)

    def save_model2(self, path: str) -> None:
        self.saver.save(self.sess, path + self.best_model_checkpoint)

    # 读取模型，将模型的变量和参数取出，将每个变量赋上存储的值
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

    def restore_model2(self, path: str) -> None:
        print("Restore...")
        # self.sess = tf.Session()
        self.saver.restore(self.sess, path)
        print("Restore done!")
