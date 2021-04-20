import json
import random
import torch
import numpy as np


class JavaCodeDataset():
    def __init__(self, file_name, n_edges, is_train):
        self.n_edges = n_edges

        self.all_data = self.load_graphs_from_file(file_name)
        # all_train_data, all_val_data = self.split_set(self.all_data, 0.5)
        all_train_data = self.all_data
        all_val_data = self.all_data
        if is_train:
            self.data = all_train_data
        else:
            self.data = all_val_data

    def __getitem__(self, index):
        annotation = self.data[0][index]
        target = self.data[1][index]
        A = self.data[2][index]
        variable = self.data[3][index]
        return annotation, target, A, variable

    def __len__(self):
        return self.data[0].shape[0]

    def load_graphs_from_file(self, file_name):
        with open(file_name, 'r') as f:
            data = json.load(f)
        data = self.clean_data(data)

        self.n_nodes = self.find_max_node_num(data)
        print('max nodes number: ', self.n_nodes)
        self.n_vars = self.find_max_var_num(data)
        print('max variable number: ', self.n_vars)

        target_list = []
        annotation_id_list = []
        variable_id_list = []
        A_list = []

        for i in range(len(data)):
            annotation_id = torch.LongTensor(data[i]['orders'][0])
            annotation_padding = torch.zeros(self.n_nodes - len(data[i]['orders'][0]), dtype=torch.long)
            annotation_id = torch.cat([annotation_id, annotation_padding])
            annotation_id_list.append(annotation_id)

            target_list.append(data[i]['targets'][0][0])

            A = self.create_adjacency_matrix(data[i]['graph'], self.n_nodes, self.n_edges)
            A_list.append(A)

            variable_id = torch.LongTensor(data[i]['variable'])
            variable_padding = torch.zeros(self.n_vars - len(data[i]['variable']), dtype=torch.long)
            variable_id = torch.cat([variable_id, variable_padding])
            variable_id_list.append(variable_id)

        annotation_id_list = torch.stack(annotation_id_list)
        target_list = torch.LongTensor(target_list)
        A_list = torch.stack(A_list)
        variable_id_list = torch.stack(variable_id_list)

        all_data = []
        all_data.append(annotation_id_list)
        all_data.append(target_list)
        all_data.append(A_list)
        all_data.append(variable_id_list)

        return all_data

    @staticmethod
    def clean_data(data):
        cleanned_data = []
        for d in data:
            if len(d['graph']) != 0:
                cleanned_data.append(d)
        return cleanned_data

    @staticmethod
    def find_max_node_num(data):
        max_node_num = 0
        for i in range(len(data)):
            if len(data[i]['orders'][0]) > max_node_num:
                max_node_num = len(data[i]['orders'][0])
        return max_node_num

    @staticmethod
    def find_max_var_num(data):
        max_var_num = 0
        for i in range(len(data)):
            if len(data[i]['variable']) > max_var_num:
                max_var_num = len(data[i]['variable'])
        return max_var_num

    @staticmethod
    def create_adjacency_matrix(edge_list, n_nodes, n_edges):
        a = np.zeros([n_nodes, n_nodes * n_edges * 2])
        for edge in edge_list:
            src_node_idx = edge[0]
            edge_type = edge[1]
            target_node_idx = edge[2]

            in_row_index = target_node_idx - 1
            in_col_index = (edge_type - 1) * n_nodes + (src_node_idx - 1)
            a[in_row_index][in_col_index] = 1

            out_row_index = src_node_idx - 1
            out_col_index = (n_edges + edge_type - 1) * n_nodes + (target_node_idx - 1)
            a[out_row_index][out_col_index] = 1
        return torch.from_numpy(a)

    def split_set(self, data_list, train_size=0.5):
        mod = int(1 / train_size)
        train = []
        val = []
        for i in range(len(data_list)):
            train_i = []
            val_i = []
            for j in range(len(data_list[i])):
                if j % mod == 0:
                    train_i.append(data_list[i][j])
                else:
                    val_i.append(data_list[i][j])
            if i == 0 or i == 2:
                train_i = torch.stack(train_i)
                val_i = torch.stack(val_i)
            train.append(train_i)
            val.append(val_i)

        return train, val


if __name__ == '__main__':
    dataset = JavaCodeDataset('../../data/input.json', 4, True)
    print(dataset.__getitem__(1))
