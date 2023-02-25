import numpy as np
import pandas as pd
from sklearn import datasets, svm
from sklearn.model_selection import GridSearchCV
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVR
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rc("font", family='YouYuan')
matplotlib.rc('axes', unicode_minus=False)  # 用来正常显示负号

data1 = pd.read_excel("./data/hw19.xlsx", sheet_name=0)
data2 = pd.read_excel("./data/hw19.xlsx", sheet_name=1)

salt_info = data1.loc[:9, 'x1':'x4']
salt_kind = data1.loc[:9, '种类']
unknown_salt_info = data1.loc[10:, 'x1':'x4']
print(salt_kind)

# 在测试数据中寻找最佳参数
'''您使用了 GridSearchCV 对训练集进行了交叉验证，
并基于交叉验证的平均得分来选择了最佳模型参数，
而 clf.score 输出的是模型在测试集上的准确率。

由于训练集和测试集的数据可能会略有不同，
因此模型在训练集上的得分可能会与在测试集上的得分有所不同。
特别地，在某些情况下，模型在训练集上的得分可能会高于在测试集上的得分，
例如当模型在训练集上过度拟合时。
'''
model = svm.SVC()
parameters = {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf'], 'gamma': [0.1, 1, 10]}
clf_test = GridSearchCV(model, parameters)
clf_test.fit(salt_info, salt_kind)
print('Best parameters:', clf_test.best_params_)
print('Best score:', clf_test.best_score_)
print('score:', clf_test.score(salt_info, salt_kind))
print(clf_test.predict(unknown_salt_info))

'''
对于线性SVM模型，参数C的值越小，模型对噪声的容忍度越高，
模型的复杂度越小，容易出现欠拟合的情况。
如果使用GridSearchCV得到的最佳参数C=0.1，可能会导致模型过于简单，
欠拟合的情况出现。而如果你手动设置了C=1，可能会使模型更复杂，从而更好地拟合数据。
'''
clf = svm.LinearSVC(C=1, max_iter=50000)
clf.fit(salt_info, salt_kind)
print(clf.predict(unknown_salt_info))
print('score:', clf.score(salt_info, salt_kind))

'''距离判别法验证'''
v = np.cov(salt_info.T)
knn = KNeighborsClassifier(2)
knn.fit(salt_info, salt_kind)
print('knn:', knn.predict(unknown_salt_info))
print('score:', knn.score(salt_info, salt_kind))


'''Fisher判别法验证'''
clf_lda = LDA()
clf_lda.fit(salt_info, salt_kind)
print('Fisher:', clf_lda.predict(unknown_salt_info))
print('误判率:', 1 - clf_lda.score(salt_info, salt_kind))

'''贝叶斯判别法验证'''
clf_bys = GaussianNB()
clf_bys.fit(salt_info, salt_kind)
print('贝叶斯:', clf_bys.predict(unknown_salt_info))
print('误判率:', 1 - clf_bys.score(salt_info, salt_kind))


T = np.array(data2['T']).reshape(-1, 1)
A = data2['A']
# 定义要搜索的超参数空间
parameters = {'kernel': ['linear', 'rbf'], 'C': [0.1, 1, 10], 'gamma': [0.1, 1, 10], 'epsilon': [0.1, 1, 10]}

# 定义要使用的回归器
svr = svm.SVR()

# 创建一个 GridSearchCV 对象.
# GridSearchCV的目的不是选择一个对给定数据集预测效果最好的模型，
# 而是选出一个泛化能力比较好的模型，能对不同测试集做出较准确的预测
clf = GridSearchCV(svr, parameters)

# 训练模型并得到最佳参数
clf.fit(T, A)
print("Best parameters: ", clf.best_params_)
print("Best score: ", clf.best_score_)

plt.scatter(T, A, label='original data')
model_2 = SVR(kernel='linear', C=10, epsilon=0.1, gamma=0.1)
model_2.fit(T, A)
predicted_a = model_2.predict(T)
plt.plot(T, predicted_a, '-y*', label='SVR')

x_new = np.linspace(180, 300, 24).reshape(-1, 1)

plt.plot(x_new, model_2.predict(x_new), '-r*', label='SVR预测值')

plt.title(f'score:{model_2.score(T, A)}')

plt.legend()
plt.show()