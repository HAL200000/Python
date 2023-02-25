import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib
matplotlib.rc("font", family='YouYuan')
matplotlib.rc('axes', unicode_minus=False)  # 用来正常显示负号
'''
List = [(1, 2, 9), (1, 3, 2), (1, 4, 4), (1, 5, 7),
        (2, 3, 3), (2, 4, 4), (3, 4, 8), (3, 5, 4), (4, 5, 6)]
G = nx.Graph()
G.add_nodes_from(range(1, 6))
G.add_weighted_edges_from(List)
pos = nx.shell_layout(G)
w = nx.get_edge_attributes(G, 'weight')
nx.draw(G, pos, with_labels=True)
nx.draw_networkx_edge_labels(G, pos, edge_labels=w)
pl.show()
'''

# hw10_5
data = pd.read_excel("./data/Pex10_5.xlsx")
# 获取坐标
x = data['x坐标'].tolist()
y = data['y坐标'].tolist()
n = len(x)
nodes = data['顶点'].tolist()
adjacent_matrix = pd.DataFrame(np.zeros((n, n)), index=nodes, columns=nodes)

# 计算几何距离的函数
distance = lambda X, Y: np.around(np.sqrt((X[0] - X[1]) ** 2 + (Y[0] - Y[1]) ** 2))

data_1 = data.set_index('顶点')
print(data_1)
for i in nodes:
    to_node1 = data_1['相邻的顶点1'][i]
    to_node2 = data_1['相邻的顶点2'][i]
    x1 = data_1['x坐标'][i]
    y1 = data_1['y坐标'][i]

    if not pd.isnull(to_node1):  # 不能直接比较来判断!!!
        x2 = data_1['x坐标'][to_node1]
        y2 = data_1['y坐标'][to_node1]
        adjacent_matrix[i][to_node1] = distance([x1, x2], [y1, y2])

    if not pd.isnull(to_node2):
        x2 = data_1['x坐标'][to_node2]
        y2 = data_1['y坐标'][to_node2]
        adjacent_matrix[i][to_node2] = distance([x1, x2], [y1, y2])

G = nx.from_pandas_adjacency(adjacent_matrix)
pos = nx.shell_layout(G)
w = nx.get_edge_attributes(G, 'weight')

# 求L到M3的最短距离和最短路径
min_dis = nx.shortest_path_length(G, 'L', 'M3', weight='weight')
print('L到M3的最短距离为：', min_dis)
path = nx.shortest_path(G, 'L', 'M3')
print('L到M3的路径为：', path)

# 构造最短路径矢量数组
path_edges = list(zip(path, path[1:]))
print(path_edges)

# 求最小生成树
min_Tree = nx.minimum_spanning_tree(G)


# 画图
plt.figure(figsize=(19.2, 10.8))
pos = dict(zip(nodes, zip(x, y)))
print(pos)
nx.draw(G, pos, with_labels=True, node_color='skyblue', alpha=0.7, node_size=200, linewidths=0.2, font_size=10, font_color='blue')
nx.draw_networkx_edge_labels(G, pos, edge_labels=w, font_size=5)

# 绘制最短路径和最小生成树
nx.draw_networkx_edges(G, pos, edgelist=path_edges, edge_color='red', width=5, alpha=0.6, label='从L到M3的最短路径')
nx.draw_networkx_edges(min_Tree, pos, edge_color='green', width=8, alpha=0.3, label='最小生成树')

# 更改标记点形状
data_2 = data.set_index('顶点类别')
print(data_2)
point1 = data_2['顶点'][1].tolist()  # 变为索引之后可以直接调用[1]
point2 = data_2['顶点'][2].tolist()
nx.draw_networkx_nodes(G, pos, nodelist=point1, node_color='tomato', node_shape='*', label='一级目标点')
nx.draw_networkx_nodes(G, pos, nodelist=point2, node_color='gold', node_shape='^', label='二级目标点')
plt.legend()
plt.savefig("road.png", dpi=300)
plt.show()
