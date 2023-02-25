import sympy as spy
import pandas as pd
import numpy as np
import numpy.linalg as la
import matplotlib
import matplotlib.pyplot as plt
import statsmodels.api as sm

matplotlib.rc("font", family='YouYuan')
matplotlib.rc('axes', unicode_minus=False)  # 用来正常显示负号
import scipy.stats as stats
from statsmodels.stats.weightstats import ztest
'''
# 身高和体重处理
df_ori = pd.read_excel("./data/PData4_6_1.xlsx", header=None)
df = df_ori.values
height1 = df[:, ::2]  # 奇数列身高
weight1 = df[:, 1::2]  # 偶数列体重
height = np.reshape(height1, (-1, 1))
weight = np.reshape(weight1, (-1, 1))
height_weight = np.hstack([height, weight])
print(f"身高均值：{np.mean(height)}\n中位数：{np.median(height)}\n极差：{np.ptp(height)}\n"
      f"方差：{np.var(height)}\n标准差：{np.std(height)}\n"
      f"\n身高和体重的协方差：{np.cov(height_weight.T)[0, 1]}\n相关系数：{np.corrcoef(height_weight.T)[0, 1]}\n")

# 求描述统计量，偏度，峰度，分位数
df_1 = pd.DataFrame(height_weight, columns=['height', 'weight'])
print(f"描述统计量如下：\n{df_1.describe()}\n")
print(f"偏度：\n{df_1.skew()}\n")
print(f"峰度：\n{df_1.kurt()}\n")
print(f"分位数：\n{df_1.quantile(0.9)}\n")

# 画条形统计图并返回频数表
plt.subplot(121)
plt.xlabel('height')
height_data = plt.hist(height, 10)
print(f"\n{height_data[0]}\n{height_data[1]}")

plt.subplot(122)
plt.xlabel('weight')
weight_data = plt.hist(weight, 10)
print(f"\n{weight_data[0]}\n{weight_data[1]}")


# 画箱线图
plt.boxplot(height, labels=['height'])


# 画经验分布函数图
h = plt.hist(weight, 20, density=True, histtype='stepfilled', cumulative=True)
print(h)
plt.grid()  # 绘制网格


# 假设身高正态分布，画Q - Q图判断拟合效果
# 通过统计数据，x拔 = u = 170.25， s = σ = 5.3747

height2 = height.flatten()
mu = np.mean(height2)
s = np.std(height2)
sorted_height = np.sort(height2)

n = len(sorted_height)
x_i = (np.arange(1, n + 1) - 1/2) / n
y_i = stats.norm.ppf(x_i, mu, s)

plt.plot(y_i, sorted_height, 'o', label='Q-Q图')
plt.plot([150, 190], [150, 190], 'r-', label='参照直线')  # 保证是一条45°直线即可


plt.show()


# 求参数假设检验 ex4_16(待深入探究)
a = np.array([3.25, 3.27, 3.24, 3.26, 3.24])
t_stat, p_value = ztest(a, value=3.25)
print(t_stat, p_value)



# 重复观测处理
data4_26 = pd.read_excel("./data/Pdata4_26_1.xlsx")
print(f"是否存在重复观测?{any(data4_26.duplicated())}")
if any(data4_26.duplicated()):
    data4_26.drop_duplicates(inplace=True)
    f = pd.ExcelWriter('Pdata4_26_2.xlsx')
    data4_26.to_excel(f)
    f.save()


# 太阳黑子个数 异常值处理
data_sunspots = pd.read_csv("./data/sunspots.csv")
plt.style.use('ggplot')
# 使用DataFrame的plot方法绘制图像会按照数据的每一列绘制一条曲线，
# 默认按照列columns的名称在适当的位置展示图例，比matplotlib绘制节省时间，
# 且DataFrame格式的数据更规范，方便向量化及计算。

data_sunspots.counts.plot(kind='hist', bins=30, density=True)
data_sunspots.counts.plot(kind='kde')
plt.ylabel('核密度')
plt.show()
'''

# 单因素方差分析
y = np.array([1620, 1670, 1700, 1750, 1800, 1580, 1600, 1640, 1720, 1460, 1540, 1620, 1680, 1500, 1550, 1610])
x = np.hstack([np.ones(5), np.full(4, 2), np.full(4, 3), np.full(3, 4)])
d = {'x': x, 'y': y}
model = sm.formula.ols("y~C(x)", d).fit()
anovat = sm.stats.anova_lm(model)
print(anovat)