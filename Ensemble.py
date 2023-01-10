import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import RadiusNeighborsClassifier
from sklearn.neighbors import KNeighborsClassifier
import datetime
from collections import Counter
from sklearn.linear_model import LogisticRegression

# 读取数据
raw_dataset = pd.read_csv("./data/processed_data.csv", header=None)
train_label = raw_dataset[40]
train_data = raw_dataset.drop(labels=40, axis=1)
test_data = pd.read_csv("./data/test.csv", header=None)

# KNN [50-60)之间
n_neighbors = 55
KNN = KNeighborsClassifier(n_neighbors=n_neighbors, weights='distance', metric='cosine')
KNN.fit(train_data, train_label)
probability_knn = KNN.predict_proba(test_data)
print("model1 ok")

# RN
RN = RadiusNeighborsClassifier(radius=0.142, weight='distance', metric='cosine', outlier_label=0)
RN.fit(train_data, train_label)
probability_rn = RN.predict_proba(test_data)
print("model2 ok")

################################ ensemble ################################
probability = probability_knn*0.7 + probability_rn*0.3
y_pred = np.argmax(probability, axis=1) + 1  # 类别
y_pred_value = np.max(probability, axis=1)  # 类别的概率，画图找概率阈值
# 概率为1的数量太多了，不绘制
idx = np.where(y_pred_value < 0.9)
y_pred_value = y_pred_value[idx]
# # 绘图
# sns.histplot(data=y_pred_value)
# plt.xlabel('Classification probability')
# plt.show()
# sns.boxplot(data=y_pred_value, width=0.5)
# plt.ylabel('Classification probability')
# plt.show()

# 概率阈值
probability_threshold = 0.4
idx_0 = np.where(y_pred_value < probability_threshold)
y_pred[idx_0] = 0

# 测试集和近邻的平均距离
# test_neigh_dist, _ = KNN.kneighbors(test_data, n_neighbors=n_neighbors, return_distance=True)
# np.save('test_neigh_dist', test_neigh_dist)
test_neigh_dist = np.load('test_neigh_dist.npy')  # 可以加载预先算好的，减小计算量
test_dist_avg = np.mean(test_neigh_dist, axis=1)  # 计算每一行的均值
print(f'test_neigh_dist.shape:{test_neigh_dist.shape}')

# 近邻的平均距离阈值
dist_threshold = 0.164
idx_0 = np.where(test_dist_avg >= dist_threshold)
y_pred[idx_0] = 0

print(f'The number of new class:{len(np.where(y_pred == 0)[0])}')
print(sorted(Counter(y_pred).items()))

# 生成submission
submission = pd.DataFrame(data=y_pred, columns=['Expected'])
submission.insert(loc=0, column='Id', value=np.arange(0, submission.shape[0], 1))
time = datetime.datetime.now().strftime('%H_%M_%S')  # ('%Y-%m-%d %H:%M:%S')加个时间好区分
submission.to_csv(f'./data/submission{time}.csv', index=False)
print('finished submission')
