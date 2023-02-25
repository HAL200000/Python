import sympy as spy
import pandas as pd
import numpy as np
import numpy.linalg as la
import matplotlib
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy import optimize
import cvxpy as cp
matplotlib.rc("font", family='YouYuan')
matplotlib.rc('axes', unicode_minus=False)  # 用来正常显示负号

# hw6_1 投资计划
c = np.array([150, 210, 60, 80, 180])
beq = np.array([210, 300, 100, 130, 260])
x = cp.Variable(5, integer=True)
obj = cp.Maximize(cp.sum(cp.multiply(c, x)))
cons = [
    x[0] + x[1] + x[2] == 1,
    x[2] + x[3] == 1,
    x[4] <= x[0],
    cp.sum(cp.multiply(beq, x)) <= 600,
    x <= 1,
    x >= 0,
]
ans = cp.Problem(obj, cons)
ans.solve()
print("最大投资收益为:", ans.value)
print("最优解为:", x.value)

# hw6_2 货机载重
c = np.array([3, 5, 2, 4, 2, 3])
bound = np.array([8, 13, 6, 9, 5, 7])
x = cp.Variable(6, integer=True)
obj = cp.Maximize(cp.sum(cp.multiply(c, x)))
cons = [
    cp.sum(cp.multiply(bound, x)) <= 24,
    x <= 1,
    x >= 0,
]
ans = cp.Problem(obj, cons)
ans.solve()
print("最大运费收入为:", ans.value)
print("最优解为:", x.value)

# hw6_6 银行营业
p = np.array([[20, 12, 10],
              [12, 15, 9],
              [6, 5, 10]])
q = np.array([[6, 8, 10],
              [6, 5, 9],
              [9, 10, 8]])

c = 0.5 * p + 0.5 * q  # 线性加权

x = cp.Variable((3, 3), integer=True)
obj = cp.Maximize(cp.sum(cp.multiply(c, x)))
cons = [
    x >= 0,
    x <= 1,
    cp.sum(x, axis=0, keepdims=True) == 1,
    cp.sum(x, axis=1, keepdims=True) == 1,
]
ans = cp.Problem(obj, cons)
ans.solve()
print("最优加权服务量化为:", ans.value)
print("最优解为:", x.value)