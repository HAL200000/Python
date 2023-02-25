import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.graphics.tsaplots import plot_predict
import matplotlib
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from pmdarima.arima import auto_arima
matplotlib.rc("font", family='YouYuan')
matplotlib.rc('axes', unicode_minus=False)  # 用来正常显示负号


data = pd.read_excel("./data/hw18_7.xlsx", header=None)
data = data.values.flatten()
N = len(data)
x = np.arange(N)

df = pd.DataFrame(data={'day': x, 'price': data})
print(df)

plt.subplot(331)
plt.plot(df['day'], df['price'], 'b-')
plt.title('原始数据折线图')

plt.subplot(332)
plt.plot(df['price'].diff(), 'r-')
plt.title('原始数据一次差分后折线图')
# 差分1次后稳定波动->ARIMA模型 d=1


ax3 = plt.subplot(333)
plot_acf(df['price'], ax=ax3, title='自相关图(ACF)')

ax4 = plt.subplot(334)
plot_pacf(df['price'], ax=ax4, title='偏自相关图(PACF)')


model = auto_arima(df['price'])
print(model.order)
# p=2, d=0, q=0 相信auto_arima的预测！

md = ARIMA(df['price'], order=(2, 0, 0))
mdf = md.fit()
print(mdf.summary())

residuals = pd.DataFrame(mdf.resid)

ax5 = plt.subplot(335)
residuals.plot(title='残差', ax=ax5)
ax5.legend_.remove()

ax6 = plt.subplot(336)
residuals.plot(kind='kde', title='密度', ax=ax6)
ax6.legend_.remove()

predict_day = 10  # 预测10天的
ax7 = plt.subplot(313)
plot_predict(mdf, ax=ax7, start=0, end=N + predict_day, alpha=0.05)  # 95%置信区间(默认值,更新后似乎锁定了这一默认值)
result = mdf.forecast(steps=predict_day)
print(result)
plt.plot(df['day'], df['price'], 'r*-', label='Original data')
plt.title('原始数据与预测值对比图')
plt.legend()
plt.show()

