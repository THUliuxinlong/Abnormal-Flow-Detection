import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import RadiusNeighborsClassifier
from collections import Counter

# 读取数据
raw_dataset = pd.read_csv("./data/processed_data.csv", header=None)
test_data = pd.read_csv("./data/test.csv", header=None)
train_label = raw_dataset[40]
train_data = raw_dataset.drop(labels=40, axis=1)
print(f'train_data.shape:{train_data.shape}, train_label.shape:{train_label.shape}, test_data.shape:{test_data.shape}')

RN = RadiusNeighborsClassifier(radius=0.142, weight='distance', metric='cosine', outlier_label=0)
RN.fit(train_data, train_label)
y_pred = RN.predict(test_data)
print(f'y_pred.shape:{y_pred.shape}')
# predict_proba(X)
print(sorted(Counter(y_pred).items()))

# 生成submission
submission = pd.DataFrame(data=y_pred, columns=['Expected'])
submission.insert(loc=0, column='Id', value=np.arange(0, submission.shape[0], 1))
import datetime
time = datetime.datetime.now().strftime('%H_%M_%S')  # ('%Y-%m-%d %H:%M:%S')加个时间好区分
submission.to_csv(f'./data/submission{time}.csv', index=False)
print('finished submission')
