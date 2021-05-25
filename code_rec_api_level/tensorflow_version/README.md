
## 1.运行CodeRecPro项目，生成txt格式的数据集


## 2.生成词表
- 运行create_vocab.py
    - 构造 WholeVocabulary 时，输入文件的路径为 graph_vocab.txt 
    - 构造 APIVocabulary，输入文件的路径为 prediction.txt 
    - 构造 VariableNameVocab，输入文件的路径为 variable_names.txt

- 控制台会输出三个词表的大小


## 3.将txt格式的数据集转化为JSON格式

- 根据控制台输出的三个词表的大小，修改raw_data_2_json_converter.py文件中
  whole_vocab_size、api_vocab_size、variable_name_vocab_size的值
  
- 设置好三个词表的路径，以及4个txt文件的路径（graph_represent.txt, graph_vocab.txt, prediction.txt, variable_names.txt）
  
- 运行raw_data_2_json_converter.py