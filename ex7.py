import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

matplotlib.rc("font", family='YouYuan')
matplotlib.rc('axes', unicode_minus=False)  # 用来正常显示负号
'''
# hw7_1 车流量统计
min_second = 60 * 60
car_recorded = np.array([2, 2, 0, 2, 5, 8, 25, 12, 5, 10,
                         12, 7, 9, 28, 22, 10, 9, 11, 8, 9, 3])
time = np.array([0, 2 * min_second, 4 * min_second, 5 * min_second,
                 6 * min_second, 7 * min_second, 8 * min_second, 9 * min_second,
                 10.5 * min_second, 11.5 * min_second, 12.5 * min_second,
                 14 * min_second, 16 * min_second, 17 * min_second, 18 * min_second,
                 19 * min_second, 20 * min_second, 21 * min_second, 22 * min_second,
                 23 * min_second, 24 * min_second])
seconds_of_day = np.linspace(0, 24*min_second, 24*min_second)
print(seconds_of_day)
f1 = interp1d(time, car_recorded)
y1 = f1(seconds_of_day)
print(f1)
f2 = interp1d(time, car_recorded, kind='quadratic')
y2 = f2(seconds_of_day)

fig = plt.figure(figsize=(10, 5))
plt.subplot(121)
plt.xlabel('秒数')
plt.ylabel('车辆通过数')
plt.plot(seconds_of_day, y1, label='线性插值拟合')
plt.scatter(time, car_recorded, s=20, c='r', label='观测值')
plt.title(f"预测值为:{np.sum(y1)}")
plt.legend()

plt.subplot(122)
plt.xlabel('秒数')
plt.ylabel('车辆通过数')
plt.plot(seconds_of_day, y2, label='二次样条插值拟合(更光滑)')
plt.scatter(time, car_recorded, s=20, c='r', label='观测值')
plt.title(f"预测值为:{np.sum(y2)}")

plt.legend()
plt.show()


# hw7_6 轿车价格
year_used = np.linspace(1, 10, 10)
avg_price = np.array([2615, 1943, 1494, 1087, 765, 538, 484, 290, 226, 204])
p = np.polyfit(year_used, avg_price, 3)
print("拟合二次多项式的从高次幂到低次幂系数分别为：", p)
y_hat = np.polyval(p, np.linspace(1, 10, 500))
y_predicted = np.polyval(p, 4.5)
plt.xlabel('轿车使用年数')
plt.ylabel('平均价格')

plt.plot(np.linspace(1, 10, 500), y_hat, label='三次多项式拟合')
plt.scatter(4.5, y_predicted, s=50, c='r')
plt.scatter(year_used, avg_price, s=20, c='b', label='观测值')
plt.title(f"4,5年后价格预测值为:{y_predicted}")
plt.legend()
plt.show()
'''

# hw7_7 国土数据计算
scale = 40 / 18  # 比例尺, 单位km:mm
data = pd.read_excel("./data/hw7_7.xlsx")
x = data['x']
y1 = data['y1']
y2 = data['y2']
f1 = interp1d(x, y1, kind='cubic')
f2 = interp1d(x, y2, kind='cubic')
accuracy = 500
x_new = np.linspace(np.min(x), np.max(x), accuracy)
y1_predicted = f1(x_new)
y2_predicted = f2(x_new)
plt.fill_between(x_new, y1_predicted, y2_predicted, alpha=0.5, label='拟合国土')
plt.scatter(x, y1, label='y1观测值')
plt.scatter(x, y2, label='y2观测值')


# 计算一个面积元:用两个小三角形表示.
# dist = np.sqrt(np.sum(np.square(vec1 - vec2)))
def dist(x_1, y_1, x_2, y_2):
    return np.sqrt((x_1 - x_2) ** 2 + (y_1 - y_2) ** 2)


def helen(d1, d2, d3):
    p = (d1 + d2 + d3) / 2
    s = (p * (p - d1) * (p - d2) * (p - d3)) ** 0.5
    return s


size = len(x_new)
i = 0
area = 0
length = 0
for x_i in x_new:
    if i + 1 < len(x_new):
        d1 = dist(x_new[i], y1_predicted[i], x_new[i], y2_predicted[i])
        d2 = dist(x_new[i], y1_predicted[i], x_new[i + 1], y1_predicted[i + 1])
        d3 = dist(x_new[i], y2_predicted[i], x_new[i + 1], y2_predicted[i + 1])
        d4 = dist(x_new[i], y1_predicted[i], x_new[i + 1], y2_predicted[i + 1])
        d5 = dist(x_new[i], y2_predicted[i], x_new[i + 1], y1_predicted[i + 1])
        length = length + (d2 + d3) * scale
        area = area + (helen(d1, d2, d5) + helen(d1, d3, d4)) * (scale ** 2)
        i += 1
plt.title(f"国土面积近似值为${area}km^2$\n国土边界长度近似值为${length}km$")
plt.legend()
plt.show()
