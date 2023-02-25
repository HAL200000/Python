import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn import metrics
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rc("font", family='YouYuan')
matplotlib.rc('axes', unicode_minus=False)  # 用来正常显示负号

# pex11_2,3,4
data = pd.read_excel("./data/Pdata11_2.xlsx", header=None, names=['序号', 'x1', 'x2', 'x3', 'x4', 'x5', '类型'],
                     index_col=0)

example = data.loc[0:20, 'x1':'x5'].astype(float)
patient = data.loc[21:22, 'x1':'x5'].astype(float)
g = data.loc[0:20, '类型'].astype(int)

v = np.cov(example.T)
knn = KNeighborsClassifier(3, metric='mahalanobis', metric_params={'V': v})
knn.fit(example, g)
pre = knn.predict(patient)
print("马氏距离判别法分类结果", pre)
print("误判率", 1 - knn.score(example, g))

clf = LDA()
clf.fit(example, g)
print("Fisher方差判别法分类结果", clf.predict(patient))
print("误判率", 1 - clf.score(example, g))

clf = GaussianNB()
clf.fit(example, g)
print("贝叶斯判别法分类结果", clf.predict(patient))
print("误判率", 1 - clf.score(example, g))

# pex11_6 交叉判别法研究分组判别准确率
model = LDA()
print("用LDA方法,交叉验证样本准确率", cross_val_score(model, example, g, cv=2))
model = GaussianNB()
print("用GaussianNB方法,交叉验证样本准确率", cross_val_score(model, example, g, cv=2))
print("用knn方法,交叉验证样本准确率", cross_val_score(knn, example, g, cv=2))

# pex11_8 城市主要经济指标
data = np.loadtxt("./data/Pdata11_8.txt")
# 注意这里研究x1-5之间的相关关系,要把x1-x5对应的每一列数据转成每一行,做到每一行对应一个变量
print("相关系数矩阵:\n", np.corrcoef(data.T))
data_1 = np.delete(data, 0, axis=1)  # 相关系数矩阵中r12 = r21 = 1,说明x1, x2强相关,可以舍弃一个
model = PCA().fit(data_1)
print("特征值为:", np.around(model.explained_variance_, 3))
print("各主成分贡献率为:", np.around(model.explained_variance_ratio_, 3))
print("奇异值为:", np.around(model.singular_values_, 3))
print("各主成分的系数为:\n", np.around(model.components_, 3))

factor = np.around(model.components_, 3)[0, :]
score = data_1.dot(factor)
print("得分从低到高排序的城市序号为:", score.argsort() + 1)


# pex11_2 轮廓系数确定K值
X = np.load("./data/Pzdata11_1.npy")
s = []
k_max = 10
for k in range(2, k_max + 1):
    md = KMeans(k)
    md.fit(X)
    labels = md.labels_
    s.append(metrics.silhouette_score(X, labels, metric='euclidean'))

plt.plot(range(2, k_max + 1), s, 'b*-')
plt.xlabel('簇的个数')
plt.ylabel('轮廓系数')
plt.show()



# pex11_14 鸢尾花分类
data = pd.read_csv("./data/iris.csv")
data = data.iloc[:, :-1]

model = KMeans(3)
model.fit(data)
labels = model.labels_
centers = model.cluster_centers_
data['cluster'] = labels
print(data)

counts = data.cluster.value_counts()
shape = ['^', '.', '*']
color = ['r', 'b', 'y']
species = ['猫猫1', '猫猫2', '猫猫3']
for i in range(len(counts)):
    plt.plot(data['Petal_Length'][labels == i],
             data['Petal_Width'][labels == i],
             color[i] + shape[i],
             markersize=5,
             label=species[i]
             )
plt.xlabel('花瓣长度')
plt.ylabel('花瓣宽度')
plt.title("猫尾花聚类结果")
plt.legend()
plt.show()


# hw11_3 乳房肿瘤判别
data = pd.read_excel("./data/乳房肿瘤数据.xlsx")
data = data.values
example = data[:-3, 1:-1].astype(float)
example_types = data[:-3, -1]
x = data[-3:, 1:-1].astype(float)

v = np.cov(example.T)
knn = KNeighborsClassifier(2, metric='mahalanobis', metric_params={'V': v})
knn.fit(example, example_types)
print("距离判别分类结果及准确率:", knn.predict(x), 1 - knn.score(example, example_types))

model = GaussianNB()
model.fit(example, example_types)
print("距离判别分类结果及准确率:", model.predict(x), 1 - model.score(example, example_types))
