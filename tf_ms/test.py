# from collections import defaultdict
import numpy as np
import torch
import torch.nn as nn
#
# # bucketed = defaultdict(list)
# bucket_sizes = np.array(list(range(1, 62, 2)))
# d = dict()
# d['graph'] = [[1,2,3], [2,3,4], [3,4,5]]
#
# print(bucket_sizes)
# print([v for e in d['graph']for v in [e[0], e[2]]])
# print(max([1, 3, 2, 4, 3, 5]))
#
# chosen_bucket_idx = np.argmax(bucket_sizes > 5)
# print(chosen_bucket_idx)
#
# test = [[1.0 for _ in range(5)] for _ in range(5)] + [[0. for _ in range(5)] for _ in range(5 - 2)]
#
# print(test)
#
# print(np.array(list(range(4, 28, 2)) + [29]))

# import tensorflow as tf
# x = tf.placeholder(tf.int32, [3,2,])
# print(x.shape)

# z = [2*2] + [] + [1]
#
# print(z[:-1])
# print(z[1:])
# weight_sizes = list(zip(z[:-1], z[1:]))
# print(weight_sizes)

# from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
# import torch
#
# a = torch.ones(2)
# b = torch.ones(1)
# c = torch.ones(3)
#
# abc = pad_sequence([a,b,c], batch_first=True)		# shape(20, 3)
# # abc = torch.transpose(abc, dim0=1, dim1=0)
# print(abc)
#
# abc2 = pack_padded_sequence(abc, [3,2,1], batch_first=True)
# print(abc2)

# import numpy as np
# import tensorflow as tf
# # [b, v, token_num]
# # [b*v, token_num]
# # train_x = np.array()
#
# train_x = np.array([[[1,1,1],[1],[1,1,1]], [[1,1,1],[1],[1,1,1]]])
# train_x = tf.unstack(train_x)
# # train_x = np.reshape(train_x, )
# import tensorflow as tf
# pad_sequences = tf.contrib.keras.preprocessing.sequence.pad_sequences
#
#
# max_len=4
# train_x = pad_sequences(train_x, maxlen=max_len, padding='post')
# print(train_x)

# max_len = 5
# tokens = [[1998, 2021, 278, 666, 7, 90], [123, 456], [66, 77, 2], [99]]
# tokens = [token + [0 for _ in range(max_len - len(token))] if len(token) <= max_len else token[:max_len] for token in tokens]
# print(tokens)
#
# label = [[1]]
# print([label[0][0]])
#
# hidden_size = 10
# tokens = [[1998, 2021, 278, 666, 7, 90], [123, 456], [66, 77, 2], [99]]
# tokens_mask = [[[1.0 for _ in range(hidden_size)] for _ in range(len(token))] +
#                [[0. for _ in range(hidden_size)] for _ in range(max_len - len(token))] if len(token) <= max_len
#                else [[1.0 for _ in range(hidden_size)] for _ in range(len(token))] for token in tokens]
# print(tokens_mask)
#
# token_length = [len(token) if len(token) < max_len else max_len for token in tokens]
# print(token_length)
#
# chosen_bucket_size = 6
# n_active_nodes = 4
#
# padded_token_length = token_length + [0 for _ in range(chosen_bucket_size - n_active_nodes)]
# print(padded_token_length)

# import numpy as np
# bucket_sizes = np.array(list(range(1, 64, 2)))
#
# d = [[0, 1, 1], [1, 1, 2], [2, 1, 3], [2, 2, 7], [2, 2, 8], [2, 2, 13], [2, 2, 15], [2, 2, 17], [2, 2, 18], [2, 2, 19], [2, 2, 23], [2, 2, 24], [2, 2, 25], [2, 2, 30], [2, 2, 33], [2, 2, 38], [2, 2, 39], [2, 2, 40], [2, 2, 44], [2, 2, 49], [2, 2, 52], [2, 2, 57], [2, 2, 61], [3, 1, 4], [7, 1, 8], [8, 1, 9], [17, 1, 18], [18, 1, 19], [19, 1, 20], [23, 1, 24], [24, 1, 25], [25, 1, 26], [33, 1, 34], [38, 1, 39], [39, 1, 40], [40, 1, 41], [44, 1, 45], [57, 1, 58], [4, 1, 5], [4, 2, 6], [4, 2, 7], [9, 1, 10], [9, 2, 13], [20, 1, 21], [20, 1, 22], [20, 1, 32], [26, 1, 27], [34, 1, 35], [34, 1, 36], [34, 1, 53], [41, 1, 42], [41, 1, 43], [41, 1, 51], [45, 1, 46], [58, 1, 59], [58, 1, 60], [5, 1, 6], [6, 1, 7], [10, 1, 11], [10, 1, 12], [10, 1, 14], [10, 1, 16], [22, 1, 23], [32, 1, 33], [27, 1, 28], [27, 1, 29], [27, 1, 31], [36, 1, 37], [53, 1, 54], [43, 1, 44], [51, 1, 52], [46, 1, 47], [46, 1, 48], [46, 1, 50], [60, 1, 61], [12, 1, 13], [14, 1, 15], [16, 1, 17], [29, 1, 30], [37, 1, 38], [54, 1, 55], [54, 1, 56], [48, 1, 49], [56, 1, 57]]
#
# d_max = max([v for e in d for v in [e[0], e[2]]])
#
# # print(d_max)
#
# chosen_bucket_idx = np.argmax(bucket_sizes > d_max)
# chosen_bucket_size = bucket_sizes[chosen_bucket_idx]
#
# # print(bucket_sizes)
# # print(chosen_bucket_idx)
# # print(chosen_bucket_size)
#
# # x = np.array([  [[1,2,3]]  , [[4,5,6]]    ])
# x = np.array([[1]])
# print(x)
# x = np.squeeze(x, axis=1)
# print(x)
# print(len(x.shape))
# if len(x.shape) == 1:
#     x = np.expand_dims(x, axis=0)
#     print(x)
#
#

# import numpy as np
# annotations = [[1,2,3],[4,5,6]]
# annotation_size = 3
# hidden_size = 10
# annotations = np.pad(annotations, pad_width=[[0, 0], [0, 0], [0, hidden_size - annotation_size]],
#                       mode='constant')
# print(annotations)

# token_max_len = 3
# chosen_bucket_size = 4
# n_active_nodes = 3
# token_orders = [[1,2,3],[4,5,0],[7,0,0]]
# token_orders = token_orders + [[0 for _ in range(token_max_len)] for _ in range(chosen_bucket_size - n_active_nodes)]
#
# print(token_orders)
#
# label = [[1548], [1548], [99]]
# # type_orders = [t for v in label for t in v]
# type_orders = label
# padded_type_orders = type_orders + [[0] for _ in range(chosen_bucket_size - n_active_nodes)]
# print(padded_type_orders)

# adjacency_matrix = torch.empty([2,3])
# print(adjacency_matrix)
# nn.init.xavier_uniform_(adjacency_matrix)
# print(adjacency_matrix)
# # adjacency_matrix = torch.transpose(adjacency_matrix, 0, 1)
# # print(adjacency_matrix.shape)
#
# def glorot_init(shape):
#     initialization_range = np.sqrt(6.0 / (shape[-2] + shape[-1]))
#     return np.random.uniform(low=-initialization_range, high=initialization_range, size=shape).astype(np.float32)
#
# print(glorot_init([2,3]))

# x = torch.rand([2,3])
# print(x)
#
# for i in range(2):
#     print(1)

# from collections import defaultdict
#
# y = [1,1,1,1,3,3,3,5,5,5,5,5]
# np.random.shuffle(y)
# print(y)
#
# bucketed = defaultdict(list)
#
# bucketed[0].append({'a':[666]})
#
# bucketed[1].append({'a':[1,2,3,4,5]})
# bucketed[1].append({'a':[6,7,8]})
# bucketed[1].append({'a':[9,10,11,12]})
#
# bucketed[2].append({'a':[1,2,3]})
# bucketed[2].append({'a':[4]})
# bucketed[2].append({'a':[5,6,7]})
# bucketed[2].append({'a':[8,8,8]})
# bucketed[2].append({'a':[9]})
#
# bucketed[3].append({'a':[1]})
# bucketed[3].append({'a':[2,2]})
# print(bucketed)
#
# for _, bucketed_data in bucketed.items():
#     # 打乱指定bucketed中同长度的数据
#     np.random.shuffle(bucketed_data)
# print(bucketed)
#
# bucket_counters = defaultdict(int)
# print(bucket_counters)
#
# print(bucket_counters[20])

# x = torch.ones([1,2])
# print(x.dtype)

batch_data = [  [ [1] ],   [ [1] ],   [ [1] ]   ]
# print(batch_data.shape)
batch_data = np.squeeze(batch_data, axis=1)
print(batch_data)
