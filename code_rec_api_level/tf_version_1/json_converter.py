import numpy as np
import json

# 词表的路径以及每个词表的大小
whole_vocab = 'vocab/WholeVocabulary.txt'
whole_vocab_size = 2294
api_vocab = 'vocab/APIVocabulary.txt'
api_vocab_size = 2376
variable_name_vocab = 'vocab/VariableNameVocabulary.txt'
variable_name_vocab_size = 593

# 输入的txt格式的数据集
raw_graph_represent = 'rawdata/graph_reprensent.txt'
raw_graph_vocab = 'rawdata/graph_vocab.txt'
raw_prediction = 'rawdata/prediction.txt'
raw_variable_names = 'rawdata/variable_names.txt'

# 输出的json格式数据集所在的文件夹
output_json_folder = 'jsondata/'

with open(raw_graph_represent, 'r') as f:
    graph_data = []
    for line in f.readlines():
        temp_data = json.loads(line)
        graph_data.append(temp_data)

print(len(graph_data))

with open(raw_graph_vocab, 'r') as f:
    graph_vocab = []
    for num,line in enumerate(f.readlines()):
        temp_data = eval(line)
        graph_vocab.append(temp_data)

print(len(graph_vocab))

# word的映射表
with open(whole_vocab, 'r') as f:
    word2idx = {}
    idx2word = {}
    for idx, token in enumerate(f.readlines()):
        word = token.strip()
        print(word,idx)
        word2idx[word] = idx
        idx2word[idx] = word

# api的映射表
with open(api_vocab, 'r') as f:
    api2idx = {}
    idx2api = {}
    for idx, word in enumerate(f.readlines()):
        api = word.strip()
        print(api,idx)
        api2idx[api] = idx
        idx2api[idx] = api

graph_order = []
# 对每一个graph都要找到它的order
for cur_dict in graph_vocab:
    cur_order = []
    for idx, word in cur_dict.items():
        # 找到真实的idx，对应于order
        real_idx = word2idx.get(word, whole_vocab_size - 1)
        cur_order.append(real_idx)
    graph_order.append([cur_order])

print(len(graph_order))

# 预测结果（生成）的映射表
with open(raw_prediction, 'r') as f:
    graph_labels = []
    for idx, word in enumerate(f.readlines()):
        pre = word.strip()
        pre_idx = api2idx.get(pre, api_vocab_size - 1)
        graph_labels.append([[pre_idx]])

print(len(graph_labels))

# 预处理添加的变量名模型，统计变量名并创建词表
variable2index = {}
index2variable = {}
with open(variable_name_vocab, 'r') as f:
    lines = f.readlines()
    for (index,word) in enumerate(lines):
        word = word.strip()
        print(word,index)
        variable2index[word] = index
        index2variable[index] = word

# 预处理所有的变量名文本
variable_data = []
with open(raw_variable_names, 'r') as f:
    for line in f.readlines():
        tokens = line.strip().split(" ")
        cur_indexes = []
        for token in tokens:
            if token == '':
                continue
                #cur_indexes.append(-1)
            else:
                index = variable2index.get(token, variable_name_vocab_size - 1)
                if index != (variable_name_vocab_size - 1) and len(token) != 1:
                    cur_indexes.append(index)
        variable_data.append(cur_indexes)

print(len(variable_data))

data = []
for graph,order,label,variable in zip(graph_data,graph_order,graph_labels,variable_data):

    if graph is None or len(graph) == 0:
        continue

    if len(order[0]) > 60:
        continue

    for e in graph:
        if e[0] is None:
            continue

    cur_json = {}
    cur_json["graph"] = graph
    cur_json["targets"] = label
    cur_json["orders"] = order
    cur_json["variable"] = variable
    data.append(cur_json)

#打乱数据
np.random.shuffle(data)
# 训练数据写入json文件
count = 0
while count < 1:
    batch = None
    if (count * 330000) < len(data):
        batch = data[count * 330000: (count + 1) * 330000]
    else:
        batch = data[count * 330000:len(data)]
    with open(output_json_folder + str(count) + '.json','w') as f:
        json.dump(batch, f)
    count = count + 1

