import copy
from _collections import deque

MAX_POINT = 100  # 表示最多顶点个数
INF = 0x3f3f3f3f  # 表示∞


class ArcNode:  # 边结点
    def __init__(self, adjacent, w):  # 构造方法
        self.adjacent = adjacent  # 邻接点
        self.weight = w  # 边的权值


class AdjGraph:  # 图邻接表类
    def __init__(self, vertex_num=0, edge_num=0):  # 构造方法
        self.adj_list = []  # 邻接表数组
        self.vertex_info = []  # 存放顶点信息，暂时未用
        self.vertex_num = vertex_num  # 顶点数
        self.edge_num = edge_num  # 边数

    def create_adj_graph(self, a, n, e):  # 通过数组a、n和e建立图的邻接表
        self.vertex_num = n  # 置顶点数和边数
        self.edge_num = e
        for i in range(n):  # 检查边数组a中每个元素
            adi = []  # 存放顶点i的邻接点
            for j in range(n):
                if a[i][j] != 0 and a[i][j] != INF:
                    # 存在一条边
                    # 自己到自己的权重为0，未链路权重为无穷
                    p = ArcNode(j, a[i][j])  # 创建<j,a[i][j]>出边的结点p
                    adi.append(p)  # 将结点p添加到adi中
            self.adj_list.append(adi)

    def display_adj_graph(self):  # 输出图的邻接表
        for temp in range(self.vertex_num):  # 遍历每一个顶点i
            print("  [%d]" % temp, end='')
            for p in self.adj_list[temp]:
                print("->(%d,%d)" % (p.adjacent, p.weight), end='')
            print("->∧")


class MatGraph:  # 图邻接矩阵类
    def __init__(self, vertex_num=0, edge_num=0):  # 构造方法
        self.edges = []  # 邻接矩阵数组
        self.vertex_info = []  # 存放顶点信息，暂时未用
        self.vertex_num = vertex_num  # 顶点数
        self.edge_num = edge_num  # 边数

    def create_mat_graph(self, array, vertex_num, edge_num):  # 通过数组a、n和e建立图的邻接矩阵
        self.vertex_num = vertex_num  # 置顶点数和边数
        self.edge_num = edge_num
        self.edges = copy.deepcopy(array)  # 深拷贝

    def display_mat_graph(self):  # 输出图
        for i in range(self.vertex_num):
            for j in range(self.vertex_num):
                if self.edges[i][j] == INF:
                    print("%4s" % "∞", end=' ')
                else:
                    print("%5d" % (self.edges[i][j]), end=' ')
            print()


def mat_to_adj(g):  # 由图的邻接矩阵转换为邻接表
    temp = MatGraph()
    assert type(g) == type(temp)  # 断言，输入类型错误则报错
    G = AdjGraph(g.vertex_num, g.edge_num)
    for i in range(g.vertex_num):  # 检查数组g.edges中每个元素
        adi = []  # 存放顶点i的邻接点
        for j in range(g.vertex_num):
            if g.edges[i][j] != 0 and g.edges[i][j] != INF:  # 存在一条边
                p = ArcNode(j, g.edges[i][j])  # 创建<j,g.edges[i][j]>出边的结点p
                adi.append(p)  # 将结点p添加到adi中
        G.adj_list.append(adi)
    return G


def adj_to_mat(G):  # 由图的邻接表转换为邻接矩阵
    temp = AdjGraph()
    assert type(G) == type(temp)
    g = MatGraph(G.vertex_num, G.edge_num)
    g.edges = [[INF] * g.vertex_num for i in range(g.vertex_num)]
    for i in range(g.vertex_num):  # 对角线置为0
        g.edges[i][i] = 0
        # 自己到自己的权重是0
    for i in range(g.vertex_num):
        for p in G.adj_list[i]:
            g.edges[i][p.adjacent] = p.weight
    return g


def depth_traversal_tree(G, v):  # 邻接表G中从顶点v出发遍历全图的深度优先遍历
    global Visited_0
    global T1
    Visited_0[v] = 1  # 置已访问标记
    for j in range(len(G.adj_list[v])):  # 处理顶点v的所有出边顶点j
        w = G.adj_list[v][j].adjacent  # 取顶点v的一个相邻点w
        if Visited_0[w] == 0:
            T1.append([v, w])  # 产生深度优先生成树的一条边
            depth_traversal_tree(G, w)  # 若w顶点未访问,递归访问它


def breadth_traversal_tree(G, v):  # 邻接表G中从顶点v出发遍历全图的广度优先遍历
    global T2
    qu = deque()  # 将双端队列作为普通队列qu
    Visited_0[v] = 1  # 置已访问标记
    qu.append(v)  # v进队
    while len(qu) > 0:  # 队不空循环
        v = qu.popleft()  # 出队顶点v
        for j in range(len(G.adj_list[v])):  # 处理顶点v的第j个相邻点
            w = G.adj_list[v][j].adjacent  # 取第j个相邻顶点w
            if Visited_0[w] == 0:  # 若w未访问
                T2.append([v, w])  # 产生广度优先生成树的一条边
                Visited_0[w] = 1  # 置已访问标记
                qu.append(w)  # w进队
                

if __name__ == '__main__':
    G = AdjGraph()
    N_0, E_0 = 5, 5
    A_0 = [[0, 8, INF, 5, INF],
           [INF, 0, 3, INF, INF],
           [INF, INF, 0, INF, 6],
           [INF, INF, 9, 0, INF],
           [INF, INF, INF, INF, 0]]
    for i in range(N_0):
        print(A_0[i])
    print(" (1)由a创建邻接表G")
    G.create_adj_graph(A_0, N_0, E_0)
    print("  G:")