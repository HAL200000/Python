import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rc("font", family='YouYuan')
matplotlib.rc('axes', unicode_minus=False)  # 用来正常显示负号

# ex2_4
'''
# 生成3*5的二维数组，每个数组元素在1-10之间随机取值
a = np.random.randint(1, 11, (3, 5))
# print(a)

# 广播机制
b = np.arange(0, 20, 10)
# print(b)
# print(b.reshape(-1, 1))
b_1 = np.arange(0, 5, 1)
# print(b_1 + b.reshape(-1, 1))

# 分隔符替换
y = '27.0 26.8 26.5 26.3 26.1 25.7 25.3 24.8'
# 数据是粘贴过来的, eval() 函数用来执行一个字符串表达式，并返回表达式的值。
y = ",".join(y.split())  # 把空格替换成逗号
y = np.array(eval(y))
print(y)

# ex2_38
c_x = np.linspace(0, 2 * np.pi, 500)
c_y1 = np.sin(c_x)
c_y2 = np.cos(pow(c_x, 2))

plt.plot(c_x, c_y1, 'r', label="$sin(x)$")
plt.plot(c_x, c_y2, 'b', label="$cos(x^2)$")
plt.legend()
# plt.show()

# ex2_39 一幅一幅的绘制！
d_x = np.linspace(0, 2 * np.pi, 500)
d_y1 = np.sin(d_x)
d_y2 = np.cos(d_x)
d_y3 = np.sin(pow(d_x, 2))

ax1 = plt.subplot(2, 2, 1)
ax1.plot(d_x, d_y1, 'r', label='$sin(x)$')
plt.legend()

ax2 = plt.subplot(2, 2, 2)
ax2.plot(d_x, d_y2, 'b', label='$cos(x)$')
plt.legend()

ax3 = plt.subplot(2, 1, 2)
ax3.plot(d_x, d_y3, 'g', label='$sin(x^2)$')
plt.legend()

# ex2_41

e_x = np.linspace(-5, 5, 100)
e_y = np.linspace(-5, 5, 100)
e_X, e_Y = np.meshgrid(e_x, e_y)
e_Z = np.sin(np.sqrt(e_X ** 2 + e_Y ** 2))

ax1 = plt.subplot(1, 2, 1, projection='3d')
ax1.plot_surface(e_X, e_Y, e_Z, cmap='viridis')
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_zlabel('z')

ax2 = plt.subplot(1, 2, 2, projection='3d')
ax2.plot_wireframe(e_X, e_Y, e_Z, color='c')
ax2.set_xlabel('x')
ax2.set_ylabel('y')
ax2.set_zlabel('z')

# ex2_44 组合图

matplotlib.rc('font', size = 7)

f_x = np.linspace(0, 2 * np.pi, 200)
f_y1 = np.sin(f_x)
f_y2 = np.cos(f_x)
f_y3 = np.sin(pow(f_x, 2))
f_y4 = f_x * np.sin(f_x)

ax1 = plt.subplot(2, 3, 1)
ax1.plot(f_x,f_y1, 'r', label='$sin(x)$')
plt.legend()

ax2 = plt.subplot(2, 3, 2)
ax2.plot(f_x, f_y2, 'b', label='$cos(x)$')
plt.legend()

ax3 = plt.subplot(2, 3, (3, 6))
ax3.plot(f_x, f_y3, 'g', label='$sin(x^2)$')
plt.legend()

ax4 = plt.subplot(2, 3, (4, 5))
ax4.plot(f_x, f_y4, 'y', label='$xsin(x)$')
plt.legend()

# 数据特征可视化综合
data1 = pd.read_excel("./data/Trade.xlsx")
# 下面用pd.Series.dt.year提取年份，注意表格里如果不是date格式，需要有pd.to_datetime()转化
data1['year'] = data1.Date.dt.year
data1['month'] = data1.Date.dt.month

ax1 = plt.subplot(2, 3, 1)
# value_counts计算不同'Order_Class'列种类的个数
counts_2012 = data1.Order_Class[data1.year == 2012].value_counts()
counts_percent_2012 = counts_2012 / counts_2012.sum()
ax1.pie(counts_percent_2012, labels=counts_percent_2012.index, autopct="%.1f%%")
ax1.set_title("2012年各等级订单比例")

ax2 = plt.subplot(2, 3, 2)
# 分组操作
Month_sales_group = data1[data1.year == 2012].groupby(by='month')
# 聚合操作
Month_sales_group_sum = Month_sales_group.aggregate({'Sales': np.sum})
Month_sales_group_sum.plot(title="2012年各月销售趋势", ax=ax2, legend=False)

ax3 = plt.subplot(2, 3, (3, 6))
cost = data1['Trans_Cost'].groupby(data1['Transport'])
d1 = cost.get_group('大卡')
d2 = cost.get_group('火车')
d3 = cost.get_group('空运')
dd = np.array([d1, d2, d3])
# 绘制箱型图
plt.boxplot(dd)
# 更改坐标轴
plt.gca().set_xticklabels(['大卡', '火车', '空运'])

ax4 = plt.subplot(2, 3, (4, 5))
# 直方图，density=True时，显示频率而不是频数， bins是柱子数量
plt.hist(data1.Sales[data1.year == 2012], bins=400, density=True)
ax4.set_title("2012年销售额分布图")
ax4.set_xlabel('销售额')
'''

# hw2_13
'''
注意：附件1提供了所指区域内部分点处的高程值（单位米），
区域用等距50米网格进行划分，数据共有874行，1165列。
本表中，第1行第1列表示坐标（0,0）点处的高程，
第m行第n列的数据代表坐标((50(m-1）,50(n-1))处的高程值，坐标单位均为米。
（0,0）点位于附件2示意图的左下角，（50X873,50X1164）点位于示意图右上角。
'''

data2 = pd.read_excel("./data/区域高程数据.xlsx", header=None)

fig = plt.figure(figsize=(10, 5))

ax1 = fig.add_subplot(1, 2, 1)
x = np.arange(0, 1165 * 50, 50)
y = np.arange(873 * 50, -50, -50)
contr = plt.contour(x, y, data2, cmap='summer')
plt.clabel(contr)
ax1.set_xlabel('x')
ax1.set_ylabel('y', rotation=0)
ax1.set_title('区域等高线图')
plt.scatter(30000, 0, s=100, color='r', label='一号基地')
plt.scatter(43000, 30000, s=100, color='b', label='二号基地')
'''
标签位置 loc=
|-------------------------------------------------------|
|   2(upper left)   9(upper center)   1(upper right)    |
|   6(center left)  10(center)        5/7(center right) |
|   3(lower left)   8(lower center)   4(lower right)    |
|-------------------------------------------------------|
'''
plt.legend(loc='lower right')

ax2 = fig.add_subplot(1, 2, 2, projection='3d')
X, Y = np.meshgrid(x, y)
ax2.plot_surface(X, Y, data2, cmap='winter')
ax2.set_xlabel('x')
ax2.set_ylabel('y')
ax2.set_zlabel('z')
ax2.set_title('区域三维网格图')
plt.show()