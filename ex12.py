import numpy as np
import statsmodels.api as sm
import pandas as pd
'''
# hw12_1
data = pd.read_excel("./data/hw12_1.xlsx")
print(data)
x = data.loc[:, 'x1':]
X = sm.add_constant(x)
y = data['y']
model = sm.OLS(y, X).fit()
print(model.summary2())
print(np.corrcoef(x.T))
print(model.predict([1, 10, 9600]))  # 注意预测的变量也要写成增广矩阵形式
'''

# hw12_4
data = pd.read_excel("./data/hw12_4.xlsx")

# 计算发芽率p和逻辑变换yi
for i in range(20):
    if data.loc[i, 'x2'] == 0: data.loc[i, 'p'] = (100 - data.loc[i, '频数']) / 100
    else: data.loc[i, 'p'] = data.loc[i, '频数'] / 100
    pi = data.loc[i, 'p']
    data.loc[i, 'yi'] = np.log(pi / (1 - pi))

x = sm.add_constant(data.loc[:, 'x1':'x2'])
y = data['p']
model = sm.OLS(y, x).fit()
print(model.summary2())



