import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn import preprocessing
from scipy.spatial.distance import cdist

# 读取原始数据
X = []  # 数据
feature = open('wine.data')
for line in feature:
    y = []
    for index, item in enumerate(line.split(",")):
        y.append(float(item))
    X.append(y)

# 转化为numpy array
X = np.array(X)  # 数据

# 数据归一化处理
min_max_scaler = preprocessing.MinMaxScaler()
X = min_max_scaler.fit_transform(X)

# 肘方法确定K值
K = range(1, 11)
distortions = []
for k in K:
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(X)
    distortions.append(sum(np.min(cdist(X, kmeans.cluster_centers_, 'euclidean'), axis=1)) / X.shape[0])
plt.plot(K, distortions, 'bx-')
plt.xlabel('k')
plt.ylabel('distortions')
plt.title('best K of the model')
plt.savefig('肘方法.png')
