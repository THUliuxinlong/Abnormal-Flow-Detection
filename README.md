# 实验报告—异常流量检测

> [SIGS_Big_Data_ML_2022 | Kaggle](https://www.kaggle.com/competitions/sigs-big-data-ml-2022)

[TOC]

## 1、组员分工

- 刘鑫龙(50%)：EDA、测试各算法性能并对模型和参数做对比实验、最优模型的确定和调参、实验报告撰写

- 杨金舜(50%)：测试各算法性能，对参数做对比实验，尝试数据降纬加快预测速度

## 2、EDA

### 1. 不同种类的数量

<img src="https://github.com/THUliuxinlong/PicGo/raw/main/img/2022-12-21-200911.png" alt="image-20221210175122459" style="zoom:80%;" />

``` python
train_data.shape:(125973, 40), test_data.shape:(22544, 40)
12    67343
10    41214
18     3633
6      3599
16     2931
19     2646
11     1493
1       956
21      892
22      890
15      201
4        53
2        30
23       20
7        18
5        11
17       10
8         9
3         8
9         7
14        4
13        3
20        2
```

### 2. 相关性分析

计算不同特征的spearman系数

<img src="https://github.com/THUliuxinlong/PicGo/raw/main/img/2022-12-21-200919.png" alt="image-20221210175140401" style="zoom:80%;" />

## 3、SMOTE

数据的类别非常不均衡，所以考虑用SMOTE算法进行过采样，但是大约有三分之一的类别只有个位数，为了防止对点数较少的类别过拟合，所以先把量少的数据复制四份份，然后只在几个数量较少的类别做SMOTE均衡。

```python
train_data.shape:(136425, 40), test_data.shape:(22544, 40)
12    67343
10    41214
18     3633
6      3599
16     2931
19     2646
11     1493
15     1005
1       956
21      892
22      890
4       857
2       834
23      824
7       822
5       815
17      814
8       813
3       812
9       811
14      808
13      807
20      806
```

## 4、模型和参数选择

| model              | parameter                                                  | data process          | score   |
| ------------------ | ---------------------------------------------------------- | --------------------- | ------- |
| KNN                | k=3                                                        | raw_data              | 0.36113 |
| KNN                | metric='cosine', k=3                                       | raw_data              | 0.40474 |
| KNN                | metric='cosine', k=50                                      | SMOTE                 | 0.43263 |
| KNN                | metric='cosine', k=55                                      | SMOTE                 | 0.43495 |
| KNN                | metric='cosine', k=60                                      | SMOTE                 | 0.43120 |
| KNN                | weights='distance', metric='cosine', k=50                  | SMOTE                 | 0.43517 |
| KNN+NCA            | weights='distance', metric='cosine', k=50                  | SMOTE                 | 0.36488 |
| RadiusNeighbors    | weights='distance', metric='cosine', radius=0.15           | SMOTE                 | 0.46283 |
| RadiusNeighbors    | weights='distance', metric='cosine', radius=0.142          | SMOTE                 | 0.46813 |
| RadiusNeighbors    | weights='distance', metric='cosine', radius=0.13           | SMOTE                 | 0.45894 |
| MLP                | lr=0.001, epoch=400                                        | SMOTE、undersampling  | 0.42416 |
| LogisticRegression | penalty='l2', solver='sag', C=0.5, max_iter=300            | SMOTE                 | 0.39050 |
| SVM                | default                                                    | SMOTE                 | 0.33999 |
| DecisionTree       | criterion="gini", splitter="best", class_weight='balanced' | SMOTE                 | 0.16771 |
| GradientBoosting   | estimators=10                                              | SMOTE                 | 0.25338 |
| ExtraTrees         | estimators=100                                             | SMOTE                 | 0.31622 |
| RandomForest       | estimators=100, criterion="gini"                           | SMOTE、under sampling | 0.26838 |
| RandomForest       | estimators=100, criterion="gini", class_weight='balanced'  | SMOTE                 | 0.24159 |
| RandomForest       | estimators=500, criterion="gini", class_weight='balanced'  | SMOTE                 | 0.25754 |

AdaBoost和RNN在训练集的效果很差，没有提交。采用PCA降纬，数据处理时间无显著提升，分数有略微下降，最高分有0.46682->0.46383。

综上复杂的模型非常容易过拟合，最终选取**KNN和RadiusNeighbors集成**，KNN距离度量选取余弦相似度，并根据距离计算权重，k取55比较合适，RN的半径选取0.142。


## 5、对类别0分类

### 1. 近邻的平均距离阈值

<img src="https://github.com/THUliuxinlong/PicGo/raw/main/img/2022-12-21-200922.png" alt="image-20221217165048679" style="zoom:80%;" />

<img src="https://github.com/THUliuxinlong/PicGo/raw/main/img/2022-12-21-200923.png" alt="image-20221217165108258" style="zoom:80%;" />

根据上图选取距离阈值应该在0.1到0.2之间，最终选取阈值为0.164，当数据与近邻的平均距离大于等于0.164，则判别为类别0。

| model | parameter                                 | data process | post process         | score   |
| ----- | ----------------------------------------- | ------------ | -------------------- | ------- |
| KNN   | weights='distance', metric='cosine', k=55 | SMOTE        | dist_threshold=0.16  | 0.47277 |
| KNN   | weights='distance', metric='cosine', k=55 | SMOTE        | dist_threshold=0.164 | 0.47509 |
| KNN   | weights='distance', metric='cosine', k=55 | SMOTE        | dist_threshold=0.17  | 0.47451 |

### 2.概率阈值

<img src="https://github.com/THUliuxinlong/PicGo/raw/main/img/2023-01-07-122838.png" alt="image-20230107122837087" style="zoom:80%;" />

<img src="https://github.com/THUliuxinlong/PicGo/raw/main/img/2023-01-07-122850.png" alt="image-20230107122849287" style="zoom:80%;" />



概率大于0.9的数量过多，为了更好的分析概率分布，只绘制分类概率小于0.9的数据。根据上图最终选取概率阈值0.4，分类概率小于阈值则判别为新类0。

| MODEL           | DATA PROCESS | POST PROCESS                                    | SCORE  |
| --------------- | ------------ | ----------------------------------------------- | ------ |
| KNN RN Ensemble | SMOTE        | dist_threshold=0.164  probability_threshold=0.4 | 0.4794 |

## 6、最终分数

![image-20230110102813473](https://github.com/THUliuxinlong/PicGo/raw/main/img/2023-01-10-102815.png)
