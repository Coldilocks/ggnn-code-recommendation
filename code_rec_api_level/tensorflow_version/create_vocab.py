

def create_whole_vocab(intput_raw_file, output_vocab_file):
    graph_vocab = []

    with open(intput_raw_file) as f:
        for num, line in enumerate(f.readlines()):
            temp_dict = eval(line)
            for value in temp_dict.values():
                graph_vocab.append(value)

    graph_vocab = list(set(graph_vocab))

    # if 'Hole' in graph_vocab:
    #     graph_vocab.remove('Hole')
    if '' in graph_vocab:
        graph_vocab.remove('')
    graph_vocab.append('<unk>')

    print('Whole Vocabulary 大小: %d' % len(graph_vocab))

    with open(output_vocab_file, 'w') as file:
        for graph in graph_vocab:
            file.write(graph)
            file.write('\n')


def create_api_vocab(intput_raw_file, output_vocab_file):
    api_vocab = []
    with open(intput_raw_file) as f:
        for line in f.readlines():
            api_vocab.append(line.strip())

    api_vocab = list(set(api_vocab))

    if '' in api_vocab:
        api_vocab.remove('')

    print('API Vocabulary 大小: %d' % len(api_vocab))

    with open(output_vocab_file, 'w') as file:
        for api in api_vocab:
            file.write(api)
            file.write('\n')


def create_variable_vocab(intput_raw_file, output_vocab_file):
    variable_vocab = []
    with open(intput_raw_file) as f:
        for line in f.readlines():
            variable_vocab.extend(line.strip().split(' '))

    variable_vocab = list(set(variable_vocab))

    if '' in variable_vocab:
        variable_vocab.remove('')
    variable_vocab.append('<unk>')

    with open(output_vocab_file, 'w') as file:
        for variable in variable_vocab:
            file.write(variable)
            file.write('\n')

    print('Variable Name Vocabulary 大小: %d' % len(variable_vocab))





if __name__ == '__main__':
    # 构造 whole vocab
    # 输入文件为 graph_vocab.txt 的路径
    # 输出文件为 WholeVocabulary.txt 的路径
    create_whole_vocab(
        intput_raw_file='rawdata/graph_vocab.txt',
        output_vocab_file='vocab/WholeVocabulary.txt'
    )
    # 构造 API vocab
    # 输入文件为 prediction.txt 的路径
    # 输出文件为 APIVocabulary.txt 的路径
    create_api_vocab(
        intput_raw_file='rawdata/prediction.txt',
        output_vocab_file='vocab/APIVocabulary.txt'
    )
    # 构造 variable vocab
    # 输入文件为 variable_names.txt 的路径
    # 输出文件为 VariableNameVocabulary.txt 的路径
    create_variable_vocab(
        intput_raw_file='rawdata/variable_names.txt',
        output_vocab_file='vocab/VariableNameVocabulary.txt'
    )

