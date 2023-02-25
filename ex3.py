import sympy as spy
import numpy as np
import numpy.linalg as la
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rc("font", family='YouYuan')
matplotlib.rc('axes', unicode_minus=False)  # 用来正常显示负号
from scipy.optimize import curve_fit
from scipy.optimize import minimize

'''
x, y = spy.symbols('x, y')
print(spy.solve((x - 3) ** 2 * (x + 5) ** 3, x))
print(spy.solve([x ** 2 + y ** 2 - 1, x - y], [x, y]))

# ex3_34


a = np.arange(1, 17).reshape(4, 4)
b = np.eye(4)
print('a为：\n', a)
print('b为：\n', b)
print('a的行列式为：\n', la.det(a))
print('a的秩为：\n', la.matrix_rank(a))
print('a的转置矩阵为：\n', a.T)
print('a的平方为：\n', a.dot(a))
print('a*b为：\n', a.dot(b))
print('横联矩阵为：\n', np.c_[a, b])
print('纵联矩阵为：\n', np.r_[a, b])
print('a左上角2*2子块为：\n', a[0:2, 0:2])
print('a为：\n', a)
print('a的特征值和特征向量为：\n', la.eig(a))

# ex3_41

x = np.array([0, 1, 2, 3])
y = np.array([-1, 0.2, 0.9, 2.1])
A = np.c_[x, np.ones_like(x)]
k, b = la.lstsq(A, y, rcond=None)[0]
plt.plot(x, y, 'o', label='原始数据')
plt.plot(x, k * x + b, 'r', label='拟合直线')  # 注意是一条直线的形式
plt.legend(loc='upper left')
plt.show()

# 使用curve_fit()函数进行任意类型的函数拟合（以a*x*x + b*x + c为例）
# 或者使用np.polyfit、np.poly1d进行拟合
x = np.array([0, 1, 2, 3])
y = np.array([1, 4, 10, 18.5])
# 拟合优度R^2的计算
def __sst(y_no_fitting):
    """
    计算SST(total sum of squares) 总平方和
    :param y_no_predicted: List[int] or array[int] 待拟合的y
    :return: 总平方和SST
    """
    y_mean = sum(y_no_fitting) / len(y_no_fitting)
    s_list = [(y - y_mean) ** 2 for y in y_no_fitting]
    sst = sum(s_list)
    return sst


def __ssr(y_fitting, y_no_fitting):
    """
    计算SSR(regression sum of squares) 回归平方和
    :param y_fitting: List[int] or array[int]  拟合好的y值
    :param y_no_fitting: List[int] or array[int] 待拟合y值
    :return: 回归平方和SSR
    """
    y_mean = sum(y_no_fitting) / len(y_no_fitting)
    s_list = [(y - y_mean) ** 2 for y in y_fitting]
    ssr = sum(s_list)
    return ssr


def goodness_of_fit(y_fitting, y_no_fitting):
    """
    计算拟合优度R^2
    :param y_fitting: List[int] or array[int] 拟合好的y值
    :param y_no_fitting: List[int] or array[int] 待拟合y值
    :return: 拟合优度R^2
    """
    SSR = __ssr(y_fitting, y_no_fitting)
    SST = __sst(y_no_fitting)
    rr = SSR / SST
    return rr


def fun(x, a, b, c):  # 注意x一定要放在最前面！！！！必须将自变量作为第一个参数，其余你需要求的参数都放后面
    return a * x * x + b * x + c


popt1 = curve_fit(fun, x, y)[0]
print(popt1)
a1, b1, c1 = popt1[0: 3]
x1 = np.linspace(x.min(), x.max(), 100)
plt.plot(x1, fun(x1, a1, b1, c1), 'r', label='拟合曲线')
plt.scatter(x, y, color='b', label='原始数据')
plt.title(f"拟合优度为：{goodness_of_fit(fun(x, a1, b1, c1), y)}")
plt.legend()
plt.show()
'''

# 求函数f(x1, x2)的一个局部极小点

f = lambda x: 100 * (x[1] - x[0] ** 2) ** 2 + (1 - np.sin(x[0])) ** 2 * np.cos(x[1])

x0 = minimize(f, [0, 0])
print(f"极小点为{x0.x},极小值为{x0.fun}")