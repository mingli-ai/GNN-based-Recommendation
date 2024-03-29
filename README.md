# GNN-based Recommendation

:memo:  Matrix Completion/ Collaborative Filtering/ link prediction 


:high_brightness: [Datasets](#datasets)

:high_brightness: [Surveys](#Surveys)

:high_brightness: [Papers](#Papers)

:high_brightness: [Resources](#Useful resources)

:high_brightness: [Tutorials](#Tutorials)

:high_brightness: [Blogs](#Blogs)


***

## Datasets
- [movielens](https://grouplens.org/datasets/movielens/)
- [amazon-book](https://jmcauley.ucsd.edu/data/amazon/)
- [gowalla](https://snap.stanford.edu/data/loc-gowalla.html)
- [yelp 2018](https://www.yelp.com/dataset)
- [Tiktok](http://ai-lab-challenge.bytedance.com/tce/vc/)
- [Flixster](https://figshare.com/articles/dataset/Flixster-dataset_zip/5677741)
- [Douban](https://www.heywhale.com/mw/dataset/58acf6f1d2445916845b4033)
- [Taobao](https://tianchi.aliyun.com/dataset/dataDetail?dataId=649) 



##  Surveys
- (2022) Recommender Systems Based on Graph Embedding Techniques: A Review [[paper](https://www.zhuanzhi.ai/paper/3c66703f93cc63d358e1bec24211ebf3)]
- (2022) Deep Meta-learning in Recommendation Systems: A Survey [[paper](https://arxiv.org/pdf/2206.04415.pdf)]
- (SIGIR 2022) **Self-Supervised Learning for Recommender Systems: A Survey** [[paper](https://arxiv.org/pdf/2203.15876.pdf)]
- (IEEE 2021) **A Survey on Neural Recommendation: From Collaborative Filtering to Content and Context Enriched Recommendation** [[paper](https://arxiv.org/pdf/2104.13030.pdf)]
- (ACM 2021) **Graph Neural Networks in Recommender Systems: A Survey** [[paper](https://arxiv.org/pdf/2011.02260.pdf)]
- (ACM 2019) Deep learning based recommender system: A survey and new perspectives. [[paper](https://arxiv.org/pdf/1707.07435.pdf)]
- (2021) Recommender systems based on graph embedding techniques:A comprehensive review [[paper](https://www.zhuanzhi.ai/paper/3c66703f93cc63d358e1bec24211ebf3)]
- (ACM 2021) Graph Neural Networks for Recommender Systems: Challenges, Methods, and Directions[[paper](https://arxiv.org/pdf/2109.12843.pdf)] [[code](https://github.com/tsinghua-fib-lab/GNN-Recommender-Systems)]
- 协同过滤推荐系统综述 [[paper](https://kns.cnki.net/kcms/detail/detail.aspxdbcode=CJFD&dbname=CJFDAUTO&filename=XAXB202105003&uniplatform=NZKPT&v=AAQtkV0Zi8XowncS1kPR56vpuNUAhhcOEyri0-rc5e6w7D3u0aKaBuiwkI0IhKRY)]


## Papers
### :small_orange_diamond:user-item collaborative filtering
- (ACM Web Conference 2022)**Improving Graph Collaborative Filtering with Neighborhood-enriched Contrastive Learning** [[paper](https://arxiv.org/pdf/2202.06200.pdf)][[code](https://github.com/RUCAIBox/NCL)]
- (SIGIR 2022) Hypergraph Contrastive Collaborative Filtering [[paper](https://arxiv.org/pdf/2204.12200.pdf)] [[code](https://github.com/akaxlh/HCCF)]
- (ACM 2022)Are Graph Augmentations Necessary? Simple Graph Contrastive Learning for Recommendation [[paper](https://www.researchgate.net/profile/Junliang-Yu/publication/359788233_Are_Graph_Augmentations_Necessary_Simple_Graph_Contrastive_Learning_for_Recommendation/links/624e802ad726197cfd426f81/Are-Graph-Augmentations-Necessary-Simple-Graph-Contrastive-Learning-for-Recommendation.pdf?ref=https://githubhelp.com)][[code](https://github.com/Coder-Yu/QRec)]
- (2022)Enhancing Sequential Recommendation with Graph Contrastive Learning [[paper](https://arxiv.org/pdf/2205.14837.pdf)]
- (2022)**Self-supervised Graph Neural Networks for Multi-behavior Recommendation** [[paper](http://www.shichuan.org/doc/134.pdf)]
- (SIGIR 2021) **Neural Graph Matching based Collaborative Filtering** [[paper](https://arxiv.org/abs/2105.04067)] [[code](https://github.com/ruizhang-ai/GMCF_Neural_Graph_Matching_based_Collaborative_Filtering)]
- (SIGIR 2021) **Structured Graph Convolutional Networks with Stochastic Masks for Recommender Systems**[[paper](http://yusanlin.com/files/papers/sigir21_structure.pdf)]
- (SIGIR 2021) **Enhanced Graph Learning for Collaborative Filtering via Mutual Information Maximization** [[paper](http://le-wu.com/files/Publications/CONFERENCES/SIGIR2021-yang.pdf)]
- (WWW 2021) **Interest-aware Message-Passing GCN for Recommendation** [[paper](https://arxiv.org/abs/2102.10044)] [[code](https://github.com/liufancs/IMP_GCN)]
- (KDD 2021) **MixGCF: An Improved Training Method for Graph Neural Network-based Recommender Systems** [[paper](https://keg.cs.tsinghua.edu.cn/jietang/publications/KDD21-Huang-et-al-MixGCF.pdf)] [[code](https://github.com/huangtinglin/MixGCF)]
- (KDD 2021) Deep Graph Convolutional Networks with Hybrid Normalization for Accurate and Diverse Recommendation [[paper](https://dlp-kdd.github.io/assets/pdf/DLP-KDD_2021_paper_1.pdf)]
- (AAAI 2021) Detecting Beneficial Feature Interactions for Recommender Systems[[paper](https://www.aaai.org/AAAI21Papers/AAAI-279.SuY.pdf)]
- (IJCAI 2021)User-as-Graph: User Modeling with Heterogeneous Graph Pooling for News Recommendation [[paper](https://www.ijcai.org/proceedings/2021/0224.pdf)]
- (2021) **基于增强图卷积神经网络的协同推荐模型**[[paper](https://kns.cnki.net/kcms/detail/11.1777.TP.20210203.1157.004.html)]
- (2021) **Localized Graph Collaborative Filtering** [[paper](https://arxiv.org/pdf/2108.04475.pdf)]
- (2021) 面向推荐系统的图卷积网络 [[paper](http://www.jos.org.cn/html/2020/4/5928.htm)]
- (2021) **How Powerful is Graph Convolution for Recommendation?** [[paper](https://arxiv.org/pdf/2108.07567.pdf)]
- (CIKM 2021)UltraGCN: Ultra Simplification of Graph Convolutional Networks for Recommendation [[paper](https://arxiv.org/pdf/2110.15114.pdf)]
- (2021) SimpleX: A Simple and Strong Baseline for Collaborative Filtering [[paper](https://arxiv.org/abs/2109.12613)]
- (SIGIR 2020) **LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation** [[paper](https://arxiv.org/abs/2002.02126)][[code](https://github.com/gusye1234/pytorch-light-gcn)]
- (IEEE 2020) **Co-embedding of Nodes and Edges with Graph Neural Networks** [[paper](https://arxiv.org/abs/2010.13242)]
- (IEEE 2020) A Graph Neural Network Framework for Social Recommendations. [[paper](https://ieeexplore.ieee.org/abstract/document/9139346)]
- (AAAI 2020) **Revisiting Graph based Collaborative Filtering: A Linear Residual Graph Convolutional Network Approach.** [[paper](https://arxiv.org/abs/2001.10167)][[code]( https://github.com/newlei/LR-GCCF.)]
- (WWW 2020) Disentangling User Interest and Conformity for Recommendation with Causal Embeddings.[[paper](https://arxiv.org/abs/2006.11011)]
- (ICLR 2020) Inductive Matrix Completion Based on Graph Neural Networks. [[paper](https://openreview.net/pdf?id=ByxxgCEYDS)]
- (Elsevier 2020) **MGAT: Multimodal Graph Attention Network for Recommendation** [[paper](https://www.sciencedirect.com/science/article/abs/pii/S0306457320300182)] [[code]( https://github.com/zltao/MGAT)]
- (WSDM 2020) **Denoising Implicit Feedback for Recommendation.** [[paper](https://arxiv.org/abs/2006.04153)]
- (2020) **Graph-Refined Convolutional Network for Multimedia Recommendation with Implicit Feedback.** [[paper](http://data-science.ustc.edu.cn/_upload/article/files/c4/4f/10f4da284171a6275429698edccf/c3aada42-ddfd-48e3-ae59-943ba9bb6edb.pdf)]
- (SIGIR 2019) **NGCF:Neural Graph Collaborative Filtering** [[paper](https://arxiv.org/abs/1905.08108)][[code]( https://github.com/xiangwang1223/neural_graph_collaborative_filtering)]
- (2019) **MMGCN: Multi-modal Graph Convolution Network for Personalized Recommendation of Micro-video** [[paper](http://staff.ustc.edu.cn/~hexn/papers/mm19-MMGCN.pdf)] [[code](https://github.com/weiyinwei/MMGCN)]
- (2019) **Simplifying Graph Convolutional Networks** [[paper](http://proceedings.mlr.press/v97/wu19e/wu19e.pdf)][[code]( https://github.com/Tiiiger/SGC)]
- (2019) STAR-GCN: Stacked and Reconstructed Graph Convolutional Networks for Recommender Systems. [[paper](https://arxiv.org/pdf/1905.13129.pdf)]
- (2019) Binarized Collaborative Filtering with Distilling Graph Convolutional Networks. [[paper](https://arxiv.org/pdf/1906.01829.pdf)]
- (ICLR 2019) Inductive Matrix Completion Based on Graph Neural Networks. [[paper](https://arxiv.org/abs/1904.12058)]
- (KDD 2019) Exact-K Recommendation via Maximal Clique Optimization[[paper](https://arxiv.org/pdf/1905.07089.pdf)]
- (WWW 2019) Dual Graph Attention Networks for Deep Latent Representation of Multifaceted Social Effects in Recommender Systems. [[paper](https://arxiv.org/pdf/1903.10433.pdf)]
- (RecSys 2018) **Spectral Collaborative Filtering** [[paper](https://arxiv.org/abs/1808.10523)] [[code](https://github.com/lzheng21/SpectralCF)]
- (IJCAI 2018) **Outer product-based neural collaborative filtering** [[paper](https://arxiv.org/pdf/1808.03912.pdf)]
- (KDD 2018) Graph Convolutional Neural Networks for Web-Scale Recommender Systems. [[paper](https://arxiv.org/abs/1806.01973)]
- (SIGIR 2017) **Attentive collaborative filtering: Multimedia recommendation with item-and component-level attention.** [[paper](https://ai.tencent.com/ailab/media/publications/Wei_Liu-Attentive_Collaborative_Filtering_Multimedia_Recommendation-SIGIR17.pdf)] 
- (2017) **GCMC: Graph Convolutional Matrix Completion.** [[paper](https://arxiv.org/abs/1706.02263)] [[code](https://github.com/hengruizhang98/GCMC-Pytorch-dgl)]
- (WWW 2017) **Neural Collaborative Filtering** [[paper](https://arxiv.org/pdf/1708.05031.pdf?source=post_page---------------------------)]
- (2017)Geometric Matrix Completion with Recurrent Multi-Graph Neural Networks. [[paper](https://arxiv.org/abs/1704.06803)]
- (2017) DeepFM: a factorization-machine based neural network for CTR prediction. [[paper](https://arxiv.org/pdf/1703.04247.pdf)]
- (SIGIR 2021) Graph Meta Network for Multi-Behavior Recommendation [[paper](https://dl.acm.org/doi/pdf/10.1145/3404835.3462972)] [[code](https://github.com/akaxlh/MB-GMN)]
- (WWW 2021) Large-scale Comb-K Recommendation [[paper](http://shichuan.org/doc/106.pdf)]




### :small_orange_diamond:sequential recommendation
- (2022)Enhancing Sequential Recommendation with Graph Contrastive Learnin [[paper](https://arxiv.org/pdf/2205.14837.pdf)]
- (WWW 2021) Adversarial and Contrastive Variational Autoencoder for Sequential Recommendation.[[paper](https://arxiv.org/pdf/2103.10693.pdf)]
- (arxiv 2021)Dynamic Graph Neural Networks for Sequential Recommendation[[paper](https://arxiv.org/pdf/2104.07368)]
- (TKDE 2021)Graph-based Embedding Smoothing for Sequential Recommendation[[paper](https://ieeexplore.ieee.org/abstract/document/9405450/)][[code](https://github.com/zhuty16/GES)]
- (SIGIR 2021)**Sequential Recommendation with Graph Neural Networks**[[paper](https://arxiv.org/pdf/2106.14226)]
- (WWW 2021) **RetaGNN: Relational Temporal Attentive Graph Neural Networks for Holistic Sequential Recommendation** [[paper](https://arxiv.org/abs/2101.12457)] [[code](https://github.com/retagnn/RetaGNN)]
- (AAAI 2020) Memory Augmented Graph Neural Networks for Sequential Recommendation. [[paper](https://arxiv.org/abs/1912.11730)]


### :small_orange_diamond:social recommendation
- (SIGIR 2021) Social Recommendation with Implicit Social Influence[[paper](https://dl.acm.org/doi/abs/10.1145/3404835.3463043)]
- (KDD 2021)Socially-Aware Self-Supervised Tri-Training for Recommendation [[paper](https://arxiv.org/pdf/2106.03569)]
- (ICDE 2021)Group-Buying Recommendation for Social E-Commerce[[paper](https://github.com/Sweetnow/group-buying-recommendation)]
- (WWW 2021)Self-Supervised Multi-Channel Hypergraph Convolutional Network for Social Recommendation[[code](https://github.com/Coder-Yu/QRec)]
- (AAAI 2021)Knowledge-aware coupled graph neural network for social recommendation[[paper](https://www.aaai.org/AAAI21Papers/AAAI-9069.HuangC.pdf)]
- (AAAI 2020) Who You Would Like to Share With? A Study of Share Recommendation in Social E-commerce [[paper](https://www.aaai.org/AAAI21Papers/AAAI-1214.JiH.pdf)]
- (SIGIR 2019) A neural influence diffusion model for social recommendation. [[paper](https://arxiv.org/pdf/1904.10322.pdf)]
- (WWW 2019) Graph Neural Networks for Social Recommendation. [[paper](https://arxiv.org/pdf/1902.07243.pdf)]



### :small_orange_diamond:knowledge graph-based recommendation
- (SIGIR 2022) **Multi-level Cross-view Contrastive Learning for Knowledge-aware Recommender System** [[paper](https://arxiv.org/pdf/2204.08807.pdf)][[code](https://github.com/CCIIPLab/MCCLK)]
- (SIGIR 2022)**Knowledge Graph Contrastive Learning for Recommendation** [[paper](https://arxiv.org/pdf/2205.00976.pdf)][[code](https://github.com/yuh-yang/KGCL-SIGIR22)]
- (KDD 2019) Knowledge-aware Graph Neural Networks with Label Smoothness Regularization for Recommender Systems. [[paper](https://arxiv.org/pdf/1905.04413)]
- (KDD 2019) KGAT: Knowledge Graph Attention Network for Recommendation. [[paper](https://arxiv.org/pdf/1905.07854)]  
- (WWW 2019) Knowledge Graph Convolutional Networks for Recommender Systems. [[paper](https://arxiv.org/pdf/1904.12575.pdf)]  


### :small_orange_diamond:Session Recommendation
- (SDM 2021)Session-based Recommendation with Hypergraph Attention Networks[[paper](https://epubs.siam.org/doi/pdf/10.1137/1.9781611976700.10)]
- (CIKM 2021)Self-Supervised Graph Co-Training for Session-based Recommendation[[paper](https://arxiv.org/pdf/2108.10560)][[code](https://github.com/xiaxin1998/COTREC)]
- (SIGIR 2021)Temporal Augmented Graph Neural Networks for Session-Based Recommendations[[paper](https://www4.comp.polyu.edu.hk/~xiaohuang/docs/Huachi_sigir2021.pdf)]
- (WSDM 2021)An Efficient and Effective Framework for Session-based Social Recommendation[[paper](http://www.cse.ust.hk/~raywong/paper/wsdm21-SEFrame.pdf)][[code](https://github.com/twchen/SEFrame)]
- (IJCAI 2019) Graph Contextualized Self-Attention Network for Session-based Recommendation. [[paper](https://www.ijcai.org/proceedings/2019/0547.pdf)]
- (AAAI 2019) Session-based Recommendation with Graph Neural Networks.[[paper](https://arxiv.org/pdf/1811.00855.pdf)]

### :small_orange_diamond:**Contrastive Learning**
- (SIGIR 2021) Contrastive Learning for Sequential Recommendation. [[paper](https://arxiv.org/abs/2010.14395)]
- (SIGIR 2021) **Self-supervised Graph Learning for Recommendation.**  [[paper](https://arxiv.org/pdf/2010.10783.pdf)] [[code](https://github.com/wujcan/SGL)]
- (KDD 2021) Contrastive Learning for Debiased Candidate Generation in Large-Scale Recommender Systems. [[paper](https://arxiv.org/abs/2005.12964)]
- (AAAI 2021) Self-Supervised Hypergraph Convolutional Networks for Session-based Recommendation [[paper](https://arxiv.org/abs/2012.06852)] [[code](https://github.com/xiaxin1998/DHCN)]
- (WWW 2021)Self-Supervised Multi-Channel Hypergraph Convolutional Network for Social Recommendation [[paper](https://arxiv.org/abs/2101.06448)] [[code](https://github.com/Coder-Yu/QRec)]
- (IJCAI 2021)Multi-Scale Contrastive Siamese Networks for Self-Supervised Graph Representation Learning  [[paper](https://arxiv.org/pdf/2105.05682.pdf)] 
- (2021)Self-supervised Learning for Large-scale Item Recommendations[[paper](https://arxiv.org/pdf/2007.12865.pdf)]
- (2021)Contrastive Learning for Recommender System[[paper](https://arxiv.org/pdf/2101.01317.pdf)]
- (2021)Pre-training Graph Neural Network for Cross Domain Recommendation [[paper](https://arxiv.org/pdf/2111.08268.pdf)]
- (2021)Socially-Aware Self-Supervised Tri-Training for Recommendation [[paper](https://arxiv.org/pdf/2106.03569.pdf)]
- (2021)Self-supervised Recommendation with Cross-channel Matching Representation and Hierarchical Contrastive Learning [[paper](https://arxiv.org/pdf/2109.00676.pdf)]



##### :small_orange_diamond:**other recommendation**
- (2022)CrossCBR: Cross-view Contrastive Learning for Bundle Recommendation [[paper](https://arxiv.org/pdf/2206.00242.pdf)]




## Ueeful resoucres
- Graph-RSs-Reproducibility[[website](https://github.com/Coder-Yu/QRec).]
- QRec[[website](https://github.com/Coder-Yu/QRec)]
- Rechours[[website](https://github.com/THUwangcy/ReChorus)]
- RecBole[[website](https://github.com/RUCAIBox/RecBole)]
	
## Tutorials
- (2022) WSDM 2022 Tutorial:Graph Neural Network for Recommender System [[website](https://sites.google.com/view/gnn-recsys/home)]
- (2022) Self-Supervised Learning in Recommendation [[website]([https://sites.google.com/view/gnn-recsys/home](https://ssl-recsys.github.io/))]
- (2021) **Deep Recommender System: Fundamentals and Advances.** [[website](https://deeprs-tutorial.github.io)]
- (2021) **Multi-Modal Recommender Systems: Hands-On Exploration** [[website](https://recsys.acm.org/recsys21/tutorials/#content-tab-1-1-tab)] [[pdf](https://github.com/PreferredAI/tutorials/tree/master/multimodal-recsys)]
- (2020) Learning and Reasoning on Graph for Recommendation  [[website](https://next-nus.github.io/)]



## Blogs 
- Explanation and Demo for LightGCN [[website](https://mp.weixin.qq.com/s/G2SEydgOI09FqtpMvWZKvw)]
- Discussion on graph neural network recommendation system in industry [[website](https://mp.weixin.qq.com/s/BVNxbiHgo0T2vqNUi6RrGQ)]
- What is Contrastive Learning?[[website](https://mp.weixin.qq.com/s/VlSoMmAGDblQ2UYhLD96gA)]
