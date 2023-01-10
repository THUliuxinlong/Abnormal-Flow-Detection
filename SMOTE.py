import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE, BorderlineSMOTE, SVMSMOTE
from collections import Counter

train_data = pd.read_csv("./data/train.csv", header=None)
print(f'train_data.shape:{train_data.shape}')
print(sorted(Counter(train_data[40]).items()))

# 先将数量少的复制一点，然后通过SMOTE均衡一下
few_point_classes = [15, 4, 2, 23, 7, 5, 17, 8, 3, 9, 14, 13, 20]
few_points = train_data[train_data[40].isin(few_point_classes)]
few_points=pd.concat([few_points, few_points, few_points, few_points])
few_points.reset_index()
print(f'few_points.shape:{few_points.shape}')
print(sorted(Counter(few_points[40]).items()))

few_points_label = few_points[40]
few_points_data = few_points.drop(labels=40, axis=1)
# label_counts = train_label.value_counts()

# SMOTE
x_oversampled, y_oversampled = SMOTE(random_state=42).fit_resample(few_points_data, few_points_label)
print(f'x_oversampled.shape:{x_oversampled.shape}')
print(sorted(Counter(y_oversampled).items()))

# 数据拼接
few_points_oversampled = pd.concat([x_oversampled, y_oversampled], axis=1)
processed_data = pd.concat([train_data, few_points_oversampled])
print(f'processed_data.shape:{processed_data.shape}')
print(sorted(Counter(processed_data[40]).items()))

processed_data.to_csv('./data/processed_data.csv', index=False, header=False)

# # 降采样
# from imblearn.under_sampling import ClusterCentroids, EditedNearestNeighbours, RepeatedEditedNearestNeighbours
# from imblearn.under_sampling import InstanceHardnessThreshold, NearMiss, NeighbourhoodCleaningRule, OneSidedSelection
# from imblearn.under_sampling import TomekLinks
# much_point_classes = [12, 10]
# much_points = train_data[train_data[40].isin(much_point_classes)]
# much_points.reset_index()
# print(f'much_points.shape:{much_points.shape}')
# print(sorted(Counter(much_points[40]).items()))
#
# much_points_label = much_points[40]
# much_points_data = much_points.drop(labels=40, axis=1)
#
# # undersampled
# x_undersampled, y_undersampled = NearMiss().fit_resample(much_points_data, much_points_label)
# # x_undersampled, y_undersampled = ClusterCentroids(random_state=42).fit_resample(much_points_data, much_points_label)
# print(f'x_undersampled.shape:{x_undersampled.shape}')
# print(sorted(Counter(y_undersampled).items()))
#
# # 数量适中的数据
# normal_point_classes = [18, 6, 16, 19, 11, 1, 21, 22]
# normal_points = train_data[train_data[40].isin(normal_point_classes)]
# normal_points.reset_index()
#
# # 数据拼接
# few_points_oversampled = pd.concat([x_oversampled, y_oversampled], axis=1)
# much_points_undersampled = pd.concat([x_undersampled, y_undersampled], axis=1)
# processed_data = pd.concat([much_points_undersampled, few_points_oversampled, normal_points])
# print(processed_data)
#
# print(f'processed_data.shape:{processed_data.shape}')
# print(sorted(Counter(processed_data[40]).items()))
#
# processed_data.to_csv('./data/smote_under.csv', index=False, header=False)
