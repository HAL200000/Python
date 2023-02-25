import numpy as np
import pandas as pd
from scipy.stats import rankdata
data = pd.read_excel("./data/hw9_1.xlsx", header=0, usecols=np.arange(1, 7))

print(data)
c_max = data.max(axis=0)
c_min = data.min(axis=0)
print(f"正理想解 = \n {c_max} \n负理想解 = \n {c_min}")
d1 = np.linalg.norm(data - c_max, axis=1)
d2 = np.linalg.norm(data - c_min, axis=1)
print(d1, d2)
f1 = d2 / (d1 + d2)
print("TOPSIS的评价值为：", f1)

t = c_max - data
mmin = t.min()
mmax = t.max()
rho = 0.5
xs = (mmin + rho * mmax) / (t + rho * mmax)
f2 = xs.mean(axis=1)
print("\n关联系数=", xs, '\n关联度=', f2)  # 显示灰色关联系数和灰色关联度

[n, m] = data.shape
cs = data.sum(axis=0)
P = 1 / cs * data
e = -(P * np.log(P)).sum(axis=0) / np.log(n)
g = 1 - e
w = g /sum(g)
F = P @ w
print("\nP={}\n,e={}\n,g={}\n,w={}\nF={}".format(P,e,g,w,F))
