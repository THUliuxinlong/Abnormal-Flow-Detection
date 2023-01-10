# SMOTE过采样：只对较少的几个类别之间平衡，防止过拟合
# knn：余弦相似度，距离权重
# post process：与近邻距离大于阈值(0.164)设为0
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier

# 绘图参数
sns.set(style='whitegrid', palette='muted', font_scale=0.7)

# 读取数据
raw_dataset = pd.read_csv("./data/processed_data.csv", header=None)
test_data = pd.read_csv("./data/test.csv", header=None)
train_label = raw_dataset[40]
train_data = raw_dataset.drop(labels=40, axis=1)
print(f'train_data.shape:{train_data.shape}, train_label.shape:{train_label.shape}, test_data.shape:{test_data.shape}')

# 看不同种类的数量
label_counts = train_label.value_counts()
print(label_counts)
sns.countplot(x=train_label)
plt.show()

# 相关性分析
sns.heatmap(train_data.corr('spearman'), annot=True, fmt='.2f', cmap='YlGnBu', square=True)
fig = plt.gcf()
fig.set_size_inches(25, 25)
plt.show()

# KNN [50-60)之间
n_neighbors = 55
KNN = KNeighborsClassifier(n_neighbors=n_neighbors, weights='distance', metric='cosine')
KNN.fit(train_data, train_label)
y_pred = KNN.predict(test_data)
print(f'y_pred.shape:{y_pred.shape}')

# 测试集和近邻的平均距离
test_neigh_dist, _ = KNN.kneighbors(test_data.values, n_neighbors=n_neighbors, return_distance=True)
np.save('test_neigh_dist', test_neigh_dist)
test_neigh_dist = np.load('test_neigh_dist.npy')  # 可以加载预先算好的，减小计算量
test_dist_avg = np.mean(test_neigh_dist, axis=1)  # 计算每一行的均值
print(f'test_neigh_dist.shape:{test_neigh_dist.shape}')

# 绘制测试集与近邻平均距离的直方图和小提琴图
sns.histplot(data=test_dist_avg)
plt.xlabel('test average cosine distance')
plt.show()
sns.boxplot(data=test_dist_avg, width=0.5)
plt.ylabel('test average cosine distance')
plt.show()

# 后处理找到新类 0
dist_threshold = 0.164
idx_0 = np.where(test_dist_avg >= dist_threshold)
print(f'The number of new class:{np.array(idx_0).shape[1]}')
y_pred[idx_0] = 0

# 生成submission
submission = pd.DataFrame(data=y_pred, columns=['Expected'])
submission.insert(loc=0, column='Id', value=np.arange(0, submission.shape[0], 1))
import datetime
time = datetime.datetime.now().strftime('%H_%M_%S')  # ('%Y-%m-%d %H:%M:%S')加个时间好区分
submission.to_csv(f'./data/submission{time}.csv', index=False)
print('finished submission')
