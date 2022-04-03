# Sequential Recommendation System

> **Lighter and Better: Low-Rank Decomposed Self-Attention Networks for Next-Item Recommendation** [[pdf](https://www.microsoft.com/en-us/research/uploads/prod/2021/05/LighterandBetter_Low-RankDecomposedSelf-AttentionNetworksforNext-ItemRecommendation.pdf)]
> [Xinyan Fan](1,2)\*, [Zheng Liu](3)\*, [Jianxun Lian](3)\*, [Wayne Xin Zhao](1,2)\*, [Xing Xie](3)\*, [Ji-Rong Wen](1,2)\*
> Gaoling School of Artificial Intelligence, Renmin University of China (1)
> Beijing Key Laboratory of Big Data Management and Analysis Methods (2)
> Microsoft Research Asia (3)
> Accepted to SIGIR 2021.

### Overview
This project is tensorflow implementation for the low-rank decomposed self-attention networks LightSANs.
Particularly, it projects user's historical items into a small constant number of latent interests, and leverages item-to-interest interaction to generate the user history representation.
Besides, the decoupled position encoding is introduced, which expresses the items’ sequential relationships much more precisely.

### License
* Apache License 2.0