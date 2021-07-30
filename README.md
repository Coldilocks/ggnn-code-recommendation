## GGNN Code Recommendation

### 1. 简单介绍
这个项目包含了GGNN(Gated Graph Neural Network[<sup>1</sup>](#refer-anchor1)，门控图神经网络)的代码，以及基于GGNN的API代码推荐和token推荐模型的源代码。

### 2. 项目结构
本项目有三个模块:

####2.1 basic_ggnn_model
   
> GGNN模型的代码，tf版本和pytorch版本

(1) **tf_version**
  

- 代码来源：微软开源的 tensorflow1.0+ 版本的GGNN源代码，项目路径：**microsoft/gated-graph-neural-network-samples**[<sup>2</sup>](#refer-anchor2)。

  
- 注意：为了方便阅读和调试，我们对`microsoft/gated-graph-neural-network-samples`的代码进行了简单的修改，将原来的DenseGGNNChemModel类和ChemModel类合并为DenseGGNNModel类。

(2) **torch_version**

  
- 说明：参考Tensorflow版本复现的PyTorch版本 GGNN 模型。


- 注意：只包含了模型的代码，不包含数据集加载部分。目前 main.py 中所有的数据都是随机生成的。

####2.2 code_rec_api_level

>基于GGNN模型的API推荐，包含一个tf版本和两个pytorch版本。
> 
>Step1. 将代码的 API上下文图 输入到 GGNN 中得到此图的向量表示 
> 
>Step2. 将代码的 token序列 输入到 LSTM 中的到隐藏状态
> 
>Step3. 将两个网络输出的向量拼接，在输入到Softmax得到输出


(1) **tf_version_1**

- 说明：tensorflow 1.0+ 版本的 API推荐模型


- 代码来源：论文 **Holistic Combination of Structural and Textual Code Information for Context based API Recommendation**[<sup>3</sup>](#refer-anchor3) 的源代码。


- 注意：此API推荐模型中的GGNN模型的代码是在`microsoft/gated-graph-neural-network-samples`的基础上改的。
  

(2) **tf_version_2**

- 说明：依然是tensorflow 1.0+ 版本的 API推荐模型。这里的代码和 `code_rec_api_level/tf_version_1` 几乎是一样，只是将 tf_version_1/model_train.py 中的类拆分到了不同文件。


- 代码来源：依然是论文 **Holistic Combination of Structural and Textual Code Information for Context based API Recommendation**[<sup>3</sup>](#refer-anchor3) 的源代码。



(3) **torch_version_1**

- 说明：Pytorch版本的 API推荐。
  

- 代码来源：GGNN模型的代码参考了另外两个项目来实现：**ggnn.pytorch**[<sup>4</sup>](#refer-anchor4) 和 **GGNN_Reasoning**[<sup>5</sup>](#refer-anchor5)。



(4) **torch_version_2**

- 说明：Pytorch版本的 API推荐，和（3）的区别是GGNN模型的代码不同。

- 代码来源：GGNN模型的代码使用的是`basic_ggnn_model/torch_version`中的。



####2.3 code_rec_token_level

>基于GGNN模型的代码token推荐，目前只包含一个tf版本。

(1) **tf_version**

- 说明：tensorflow版本的 token推荐，参考`code_rec_api_level/tf_version_1`来实现。


### 3. Reference

<div id="refer-anchor1"></div>

[1] [Gated Graph Sequence Neural Networks.](https://arxiv.org/abs/1511.05493)

<div id="refer-anchor2"></div>

[2] [GitHub - microsoft/gated-graph-neural-network-samples: Sample Code for Gated Graph Neural Networks](https://github.com/microsoft/gated-graph-neural-network-samples)

<div id="refer-anchor3"></div>

[3] [Holistic Combination of Structural and Textual Code Information for Context based API Recommendation.](https://arxiv.org/abs/2010.07514)

<div id="refer-anchor4"></div>

[4] [GitHub - chingyaoc/ggnn.pytorch: A PyTorch Implementation of Gated Graph Sequence Neural Networks (GGNN)](https://github.com/chingyaoc/ggnn.pytorch)

<div id="refer-anchor5"></div>

[5] [GitHub - entslscheia/GGNN_Reasoning: PyTorch implementation for Graph Gated Neural Network (for Knowledge Graphs)](https://github.com/entslscheia/GGNN_Reasoning)

   

    


