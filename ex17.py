from sklearn.linear_model import Perceptron

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rc("font", family='YouYuan')
matplotlib.rc('axes', unicode_minus=False)  # 用来正常显示负号
from sklearn.neural_network import MLPRegressor

'''
# pex17_3
x0 = np.array([[-0.5, -0.5, 0.3, 0],
               [-0.5, 0.5, -0.5, 1]]).T
y0 = np.array([1, 1, 0, 0])
model = Perceptron(tol=1e-3)
model.fit(x0, y0)
print(model.coef_, model.intercept_)
print(model.score(x0, y0))
print(model.predict(np.array([[-0.5, 0.2]])))
'''

# hw17_4
# 处理5.5k这样的数字
# data = pd.read_excel("./data/hw17_4_1.xlsx", index_col=0, dtype=str)
# data = data.applymap(lambda x: float(x[:-1]) * 1000 if x[-1] == 'k' else (float(x[:-1]) * 1000000 if x[-1] == 'm' else float(x)))
data = pd.read_excel("./data/hw17_4_1.xlsx", index_col=0)
print(data)
model = MLPRegressor(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=10)
# loc可以直接通过行、列标签定位，但iloc只能接受数字
x = data.loc[:'1957', :'x3']
y = data.loc[:'1957', 'y']

print(x)
print(y)
index = np.array(data.index)
model.fit(x, y)
predict_x = data.loc['1958':, :'x3']
print(model.score(x, y))
ans = model.predict(predict_x)
print(ans)

plt.plot(index[:-2], y, 'o')
plt.plot(index, model.predict(data.loc[:, :'x3']), '-*')

plt.show()
# 27.6, 26.3

