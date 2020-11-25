import numpy as np
import matplotlib.pyplot as plt

# 读取原始数据
X = []  # 数据
y_true = []
fname = []
feature = open('wine.data')
name = open('wine.names')
for line in feature:
    y = []
    for index, item in enumerate(line.split(",")):
        if index == 0:
            y_true.append(int(item))
            continue
        y.append(float(item))
    X.append(y)
i = 1
for line in name:
    if str(i) + ')' in line and '(' + str(i) + ')' not in line:
        line = line[line.index(')') + 1:].strip()
        fname.append(line)
        i += 1

# 转化为numpy array
X = np.array(X)  # 数据
y_true = np.array(y_true)  # 标签

# 数据归一化处理
x = X[:]
X = x / x.max(axis=0)

# 肘方法确定的K值
K = 3

# 随机选择K个聚类中心
centers = np.zeros([K, X.shape[1]])
for i in range(K):
    # 把数据分成K份，每一份中随机选一个最为初始聚类中心
    index = np.random.randint(i * (X.shape[0] // K), (i + 1) * (X.shape[0] // K))
    centers[i] = X[index][:]

# 开始迭代
iteration = 0
while True:
    # 计算每个点到聚类中心的距离，将其划分到最近的一类
    y_pre = [0] * X.shape[0]  # 预测结果，表示每个点属于哪一类
    for i in range(X.shape[0]):
        distance = list()
        for j in range(K):
            # 计算数据点i到聚类中心j的欧式距离
            d = np.sqrt(np.sum(np.square(centers[j] - X[i])))
            distance.append(d)
        # 数据点i离哪个中心最近，就把i归为哪一类
        y_pre[i] = distance.index(min(distance)) + 1
    # 转化为numpy array
    y_pre = np.array(y_pre)

    # 重新计算类别中新的聚类中心
    new_centers = np.zeros(centers.shape)
    for i in range(K):
        # 找到同一聚类的下标
        index_sets = np.where(y_pre == i + 1)
        # 找出同一聚类的数据
        cluster = X[index_sets]
        for j in range(cluster.shape[1]):
            # 计算这个聚类的均值
            mean = np.mean(cluster[:, j])
            # 用均值作为新的聚类中心
            new_centers[i][j] = mean

    # 计算新旧聚类中心的欧式距离
    d = np.sqrt(np.sum(np.square(new_centers - centers)))

    # 达到稳定，结束迭代
    if d == 0:
        break
    # 未达到稳定，更新聚类中心，继续迭代
    centers = new_centers

    print('==========')
    # SSE误差平方和
    SSE = 0
    for i in range(X.shape[0]):
        SSE += np.sum(np.square(X[i] - centers[y_pre[i] - 1]))
    print('第', iteration, '次迭代 SSE误差平方和：', SSE)

    # 兰德指数
    a, b, c, d = 0, 0, 0, 0
    for i in range(X.shape[0]):
        for j in range(i + 1, X.shape[0]):
            if y_true[i] == y_true[j] and y_pre[i] == y_true[j]:
                a += 1
            if y_true[i] == y_true[j] and y_pre[i] != y_true[j]:
                b += 1
            if y_true[i] != y_true[j] and y_pre[i] == y_true[j]:
                c += 1
            if y_true[i] != y_true[j] and y_pre[i] != y_true[j]:
                d += 1
    ART = (a + d) / (a + b + c + d)
    print('第', iteration, '次迭代 兰德指数：', ART)
    iteration += 1

# 选择你要展示的两个特征，（wine数据有13个特征，要画图展示只能展示其中两个特征的聚类关系，因为平面直角坐标系只有xy两个轴）
f1, f2 = 0, 1
# 设置不同类别的颜色
colors = ['#ff0000', '#00ff00', '#0000ff']
plt.figure()
for i in range(K):  # 循环读取类别
    # 找到同一聚类的下标
    index_sets = np.where(y_pre == i + 1)
    # 找出同一聚类的数据
    cluster = X[index_sets]
    # 展示聚类子集内的样本点
    plt.scatter(cluster[:, f1], cluster[:, f2], c=colors[i], marker='.')
    # 展示聚类中心
    plt.plot(centers[i][f1], centers[i][f2], '*', markerfacecolor=colors[i], markeredgecolor='k', markersize=6)

plt.xlabel(fname[f1])
plt.ylabel(fname[f2])
plt.savefig('wine_result.png')
plt.show()
