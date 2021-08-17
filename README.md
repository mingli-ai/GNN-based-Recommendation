# GNN-based Recommendation

:memo:  Matrix Completion/ Collaborative Filtering/ link prediction 


:high_brightness: [Datasets](#datasets)

:high_brightness: [Surveys](#Surveys)

:high_brightness: [Papers](#Papers)

:high_brightness: [Tutorials](#Tutorials)
***

## Datasets
- [movielens](https://grouplens.org/datasets/movielens/)
- [amazon-book](https://jmcauley.ucsd.edu/data/amazon/)
- [gowalla](https://snap.stanford.edu/data/loc-gowalla.html)
- [yelp 2018](https://www.yelp.com/dataset)
- [Tiktok](http://ai-lab-challenge.bytedance.com/tce/vc/)
- [Flixster](https://figshare.com/articles/dataset/Flixster-dataset_zip/5677741)
- [Douban](https://www.heywhale.com/mw/dataset/58acf6f1d2445916845b4033)
- [Taobao](https://tianchi.aliyun.com/dataset/dataDetail?dataId=649) for recommendation problem with implicit feedback



##  Surveys
- (2021) **Graph Neural Networks in Recommender Systems: A Survey** [[paper](https://arxiv.org/pdf/2011.02260.pdf)]
- (2021) **A Survey on Neural Recommendation: From Collaborative Filtering to Content and Context Enriched Recommendation** [[paper](https://www.zhuanzhi.ai/paper/cbf33028b44f85138520717fd1d72792)]
- (2019) Deep learning based recommender system: A survey and new perspectives. [[paper](https://arxiv.org/pdf/1707.07435.pdf)]


## Papers
##### :small_orange_diamond:2021
- (SIGIR 2021) **Self-supervised Graph Learning for Recommendation.**  [[paper](https://arxiv.org/pdf/2010.10783.pdf)] [[code](https://github.com/wujcan/SGL)]
- (SIGIR 2021) **Neural Graph Matching based Collaborative Filtering** [[paper](https://arxiv.org/abs/2105.04067)] [[code](https://github.com/ruizhang-ai/GMCF_Neural_Graph_Matching_based_Collaborative_Filtering)]
- (SIGIR 2021) **Structured Graph Convolutional Networks with Stochastic Masks for Recommender Systems**[[paper](http://yusanlin.com/files/papers/sigir21_structure.pdf)]
- (SIGIR 2021) **Enhanced Graph Learning for Collaborative Filtering via Mutual Information Maximization** [[paper](http://le-wu.com/files/Publications/CONFERENCES/SIGIR2021-yang.pdf)]
- (SIGIR 2021) **Sequential Recommendation with Graph Neural Networks** [[paper](https://arxiv.org/abs/2106.14226)]
- (SIGIR 2021) Contrastive Learning for Sequential Recommendation. [[paper](https://arxiv.org/abs/2010.14395)]
- (WWW 2021) **RetaGNN: Relational Temporal Attentive Graph Neural Networks for Holistic Sequential Recommendation** [[paper](https://arxiv.org/abs/2101.12457)] [[code](https://github.com/retagnn/RetaGNN)]
- (WWW 2021) **Interest-aware Message-Passing GCN for Recommendation** [[paper](https://arxiv.org/abs/2102.10044)] [[code](https://github.com/liufancs/IMP_GCN)]
- (WWW 2021) Adversarial and Contrastive Variational Autoencoder for Sequential Recommendation.[[paper](https://arxiv.org/pdf/2103.10693.pdf)] [[code]
- (WWW 2021) Large-scale Comb-K Recommendation [[paper](http://shichuan.org/doc/106.pdf)]
- (KDD 2021) **MixGCF: An Improved Training Method for Graph Neural Network-based Recommender Systems** [[paper](https://keg.cs.tsinghua.edu.cn/jietang/publications/KDD21-Huang-et-al-MixGCF.pdf)] [[code](https://github.com/huangtinglin/MixGCF)]
- (KDD 2021) Contrastive Learning for Debiased Candidate Generation in Large-Scale Recommender Systems. [[paper](https://arxiv.org/abs/2005.12964)]
(https://github.com/ACVAE/ACVAE-PyTorch)]
- (AAAI 2021) Detecting Beneficial Feature Interactions for Recommender Systems[[paper](https://www.aaai.org/AAAI21Papers/AAAI-279.SuY.pdf)]
- (2021) **基于增强图卷积神经网络的协同推荐模型**[[paper](https://kns.cnki.net/kcms/detail/11.1777.TP.20210203.1157.004.html)]
- (2021) **Localized Graph Collaborative Filtering** [[paper](https://arxiv.org/pdf/2108.04475.pdf)]





##### :small_orange_diamond:2020


- (SIGIR 2020) **LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation** [[paper](https://arxiv.org/abs/2002.02126)][[code](https://github.com/gusye1234/pytorch-light-gcn)]
- (IEEE 2020) **Co-embedding of Nodes and Edges with Graph Neural Networks** [[paper](https://arxiv.org/abs/2010.13242)]
- (IEEE 2020) A Graph Neural Network Framework for Social Recommendations. [[paper](https://ieeexplore.ieee.org/abstract/document/9139346)]
- (AAAI 2020) **Revisiting Graph based Collaborative Filtering: A Linear Residual Graph Convolutional Network Approach.** [[paper](https://arxiv.org/abs/2001.10167)] [[code]( https://github.com/newlei/LR-GCCF.)]
- (AAAI 2020) Memory Augmented Graph Neural Networks for Sequential Recommendation. [[paper](https://arxiv.org/abs/1912.11730)]
- (AAAI 2020) Who You Would Like to Share With? A Study of Share Recommendation in Social E-commerce [[paper](https://www.aaai.org/AAAI21Papers/AAAI-1214.JiH.pdf)]
- (WWW 2020) Disentangling User Interest and Conformity for Recommendation with Causal Embeddings.[[paper](https://arxiv.org/abs/2006.11011)]
- (ICLR 2020) Inductive Matrix Completion Based on Graph Neural Networks. [[paper](https://openreview.net/pdf?id=ByxxgCEYDS)]
- (Elsevier 2020) **MGAT: Multimodal Graph Attention Network for Recommendation** [[paper](https://www.sciencedirect.com/science/article/abs/pii/S0306457320300182)] [[code]( https://github.com/zltao/MGAT)]
- (WSDM 2020) **Denoising Implicit Feedback for Recommendation.** [[paper](https://arxiv.org/abs/2006.04153)]
- (2020) **Graph-Refined Convolutional Network for Multimedia Recommendation with Implicit Feedback.** [[paper](http://data-science.ustc.edu.cn/_upload/article/files/c4/4f/10f4da284171a6275429698edccf/c3aada42-ddfd-48e3-ae59-943ba9bb6edb.pdf)]


##### :small_orange_diamond:2019
- (2019) **NGCF:Neural Graph Collaborative Filtering** [[paper](https://arxiv.org/abs/1905.08108)][[code]( https://github.com/xiangwang1223/neural_graph_collaborative_filtering)]
- (2019) **MMGCN: Multi-modal Graph Convolution Network for Personalized Recommendation of Micro-video** [[paper](http://staff.ustc.edu.cn/~hexn/papers/mm19-MMGCN.pdf)] [[code](https://github.com/weiyinwei/MMGCN)]
- (2019) **Simplifying Graph Convolutional Networks** [[paper](http://proceedings.mlr.press/v97/wu19e/wu19e.pdf)][[code]( https://github.com/Tiiiger/SGC)]
- (2019) Graph neural networks for social recommendation. [[paper](https://arxiv.org/pdf/1902.07243.pdf)]
- (2019) A neural influence diffusion model for social recommendation. [[paper](https://arxiv.org/pdf/1904.10322.pdf)]
- (2019) Inductive Matrix Completion Based on Graph Neural Networks. [[paper](https://arxiv.org/abs/1904.12058)]
- (2019) STAR-GCN: Stacked and Reconstructed Graph Convolutional Networks for Recommender Systems. [[paper](https://arxiv.org/pdf/1905.13129.pdf)]
- (2019) Binarized Collaborative Filtering with Distilling Graph Convolutional Networks. [[paper](https://arxiv.org/pdf/1906.01829.pdf)]
- (2019) Graph Contextualized Self-Attention Network for Session-based Recommendation. [[paper](https://www.ijcai.org/proceedings/2019/0547.pdf)]
- (2019) Session-based Recommendation with Graph Neural Networks.[[paper](https://arxiv.org/pdf/1811.00855.pdf)]
- (2019) Knowledge-aware Graph Neural Networks with Label Smoothness Regularization for Recommender Systems. [[paper](https://arxiv.org/pdf/1905.04413)]
- (2019) Exact-K Recommendation via Maximal Clique Optimization. [[paper](https://arxiv.org/pdf/1905.07089)]
- (2019) KGAT: Knowledge Graph Attention Network for Recommendation. [[paper](https://arxiv.org/pdf/1905.07854)]  
- (2019) Knowledge Graph Convolutional Networks for Recommender Systems. [[paper](https://arxiv.org/pdf/1904.12575.pdf)]  
- (2019) Dual Graph Attention Networks for Deep Latent Representation of Multifaceted Social Effects in Recommender Systems. [[paper](https://arxiv.org/pdf/1903.10433.pdf)]  
- (2019) Graph Neural Networks for Social Recommendation. [[paper](https://arxiv.org/pdf/1902.07243.pdf)]
- (KDD 2019) Exact-K Recommendation via Maximal Clique Optimization[[paper](https://arxiv.org/pdf/1905.07089.pdf)]

##### :small_orange_diamond:2018
- (2018) **Spectral Collaborative Filtering** [[paper](https://arxiv.org/abs/1808.10523)] [[code](https://github.com/lzheng21/SpectralCF)]
- (2018) **Outer product-based neural collaborative filtering** [[paper](https://arxiv.org/pdf/1808.03912.pdf3)]
- (2018) Graph Convolutional Neural Networks for Web-Scale Recommender Systems. [[paper](https://arxiv.org/abs/1806.01973)]

##### :small_orange_diamond:2017
- (2017) **Attentive collaborative filtering: Multimedia recommendation with item-and component-level attention.** [[paper](https://ai.tencent.com/ailab/media/publications/Wei_Liu-Attentive_Collaborative_Filtering_Multimedia_Recommendation-SIGIR17.pdf)] 
- (2017) **GCMC: Graph Convolutional Matrix Completion.** [[paper](https://arxiv.org/abs/1706.02263)] [[code](https://github.com/hengruizhang98/GCMC-Pytorch-dgl)]
- (2017) **Neural Collaborative Filtering** [[paper](https://arxiv.org/pdf/1708.05031.pdf?source=post_page---------------------------)]
- (2017)Geometric Matrix Completion with Recurrent Multi-Graph Neural Networks. [[paper](https://arxiv.org/abs/1704.06803)]
- (2017) DeepFM: a factorization-machine based neural network for CTR prediction. [[paper](https://arxiv.org/pdf/1703.04247.pdf)]

	
	
## Tutorials
- (2021) **Deep Recommender System: Fundamentals and Advances.** [[website](https://deeprs-tutorial.github.io)]
- (2020) Learning and Reasoning on Graph for Recommendation  [[website](https://next-nus.github.io/)]
