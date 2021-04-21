## GGNN Token Recommendation Pro

### 1. Brief Introduction
This repository contains GGNN(Gated Graph Neural Network[<sup>1</sup>](#refer-anchor1)) based code recommendation models implemented in PyTorch and Tensorflow 1.0, which can generate API or token predictions for incomplete code snippets.

### 2. Project Structure
There are three packages inside the project:

1. basic_ggnn_model 
- **tf_version**
      
  The model under this package is a duplicate implementation of GGNN model in Tensorflow in another repository: **gated-graph-neural-network-samples**[<sup>2</sup>](#refer-anchor2).
  A small modification is to merge the DenseGGNNChemModel and ChemModel in the original version into DenseGGNNModel for easy reading.
 
- **torch_version**

  A re-implemented PyTorch GGNN model based on the Tensorflow version(**gated-graph-neural-network-samples**[<sup>2</sup>](#refer-anchor2)).
  Currently there is no dataset loading method️ for this implementation, and all the input data is randomly generated.

2. code_rec_api_level
- **torch_version_1**
  
  An unofficial implementation of Java API recommendation model for the paper **Holistic Combination of Structural and Textual Code Information for Context based API Recommendation**[<sup>3</sup>](#refer-anchor3). This model is inspired by another two repository: **ggnn.pytorch**[<sup>4</sup>](#refer-anchor4) and **GGNN_Reasoning**[<sup>5</sup>](#refer-anchor5).
  
- **torch_version_2**
  
  This is also an unofficial implementation of the above paper. Please note that it is based on the model under the `basic_ggnn_model/torch_version` package.

3. code_rec_token_level
- **tf_version**

  Token-level Code recommentation model inspired by the paper **Holistic Combination of Structural and Textual Code Information for Context based API Recommendation**.
  




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

   

    


