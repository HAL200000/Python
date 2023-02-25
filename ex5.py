import sympy as spy
import pandas as pd
import numpy as np
import numpy.linalg as la
import matplotlib
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy import optimize
matplotlib.rc("font", family='YouYuan')
matplotlib.rc('axes', unicode_minus=False)  # 用来正常显示负号
'''
# 求解线性规划问题 ex5_2
c = [-1, 2, 3]
A = [[-2, 1, 1], [3, -1, -2]]
b = [[9], [-4]]
Aeq = [[4, -2, -1]]
beq = [-6]
lower_bound = [-10, 0, None]
upper_bound = [None, None, None]
bound = tuple(zip(lower_bound, upper_bound))
ans = optimize.linprog(c, A, b, Aeq, beq, bound)
print(f"目标函数最小值:{ans.fun},\n最优解:{ans.x}")

# 生产牛奶 ex5_7
c = np.array([3*24, 4*16])
A = [[12, 8], [3, 0]]
b = [[480], [100]]
Aeq = [[1, 1]]
beq = [50]
lower_bound = [0, 0]
upper_bound = [50, 50]
bound = tuple(zip(lower_bound, upper_bound))
ans = optimize.linprog(-c, A, b, Aeq, beq, bound)
print(f"目标函数最大值:{-ans.fun},\n最优解:{ans.x}")

# 投资模型 ex5_4
data = pd.read_excel("./data/投资.xlsx")
ri = data['r_i/%'].values / 100
qi = data['q_i/%'].values / 100
pi = data['p_i/%'].values / 100

c = np.array(ri - pi)
A = np.c_[np.zeros(4), np.diag(qi[1: len(qi)])]

Aeq = [np.array(1 + pi)]

beq = [1]
lower_bound = [0] * 5
upper_bound = [None] * 5
bound = tuple(zip(lower_bound, upper_bound))

print(c)
print(A)
print(Aeq)


a_max = 0.05
step = 0.001
a = 0
a_best = []
Q_best = []

while a <= a_max:
    b = np.ones(4) * a
    print(b)
    ans = optimize.linprog(-c, A, b, Aeq, beq, bound)
    a_best.append(a)
    Q_best.append(-ans.fun)
    a = a + step

plt.plot(a_best, Q_best, 'r*')
plt.xlabel('风险度$a$')
plt.ylabel('总体收益$Q$', rotation=90)
plt.show()
'''
# 糖果生产 hw5_4
c = np.array([0.9, 0.45, -0.05, 1.4, 0.95, 0.45, 1.9, 1.45, 0.95])
A = [[-0.4, 0, 0, 0.6, 0, 0, 0.6, 0, 0],
     [0, -0.7, 0, 0, 0.3, 0, 0, 0.3, 0],
     [-0.2, 0, 0, -0.2, 0, 0, 0.8, 0, 0],
     [0, -0.5, 0, 0, -0.5, 0, 0, 0.5, 0],
     [0, 0, -0.6, 0, 0, -0.6, 0, 0, 0.4],

     [1, 1, 1, 0, 0, 0, 0, 0, 0],
     [0, 0, 0, 1, 1, 1, 0, 0, 0],
     [0, 0, 0, 0, 0, 0, 1, 1, 1]]

b = [0, 0, 0, 0, 0, 2000, 2500, 1200]

lower_bound = np.zeros(12)
upper_bound = [2000, 2000, 2000, 2500, 2500, 2500, 1200, 1200, 1200]
bound = tuple(zip(lower_bound, upper_bound))
ans = optimize.linprog(-c, A, b, None, None, bound)
print(f"目标函数最大值:{-ans.fun}")
tot = 0
for x in ans.x:
    tot = tot + 1
    print(f"x{tot}: {x}")

print(np.reshape(ans.x, (3, 3)))

