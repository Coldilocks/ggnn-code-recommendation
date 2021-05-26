import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

from collections import namedtuple
from code_rec_api_level.tf_version_2.dense_ggnn_prog_model import DenseGGNNProgModel

# Parser = namedtuple('parser', ['data_dir', 'config_file', 'config', 'restore', 'restrict', 'freeze_graph_model'])
Parser = namedtuple('parser', ['data_dir', 'config_file', 'config', 'restore', 'restrict', 'freeze_graph_model'])
parser = Parser(data_dir='../../data/small_input.json', config_file=None, config=None, restore="model_best-2021",
                restrict=None, freeze_graph_model=False)

model = DenseGGNNProgModel(parser,training_file_count=1,valid_file_count=1)
evaluation = False
if evaluation:
   model.example_evaluation()
else:
   model.train()

# s = zerorpc.Server(DenseGGNNProgModel(parser, training_file_count=11, valid_file_count=6))
# s.bind("tcp://0.0.0.0:4242")
# s.run()