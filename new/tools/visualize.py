# Install required packages.
import os
import torch
os.environ['TORCH'] = torch.__version__
print(torch.__version__)
# !pip install -q torch-scatter -f https://data.pyg.org/whl/torch-${TORCH}.html
# !pip install -q torch-sparse -f https://data.pyg.org/whl/torch-${TORCH}.html
# !pip install -q git+https://github.com/pyg-team/pytorch_geometric.git
 
# # Helper function for visualization.
# %matplotlib inline

import networkx as nx
import matplotlib.pyplot as plt
 
 
def visualize_graph(G, color):
    plt.figure(figsize=(7,7))
    plt.xticks([])
    plt.yticks([])
    nx.draw_networkx(G, pos=nx.spring_layout(G, seed=42), with_labels=False,
                     node_color=color, cmap="Set2")
    plt.show()
 
 
def visualize_embedding(h, color, epoch=None, loss=None):
    plt.figure(figsize=(7,7))
    plt.xticks([])
    plt.yticks([])
    h = h.detach().cpu().numpy()
    plt.scatter(h[:, 0], h[:, 1], s=140, c=color, cmap="Set2")
    if epoch is not None and loss is not None:
        plt.xlabel(f'Epoch: {epoch}, Loss: {loss.item():.4f}', fontsize=16)
    plt.show()


    import networkx as nx
    import matplotlib.pyplot as plt
    from torch_geometric.datasets import Planetoid

    # 可视化
    num_nodes = 200 # 需要可视化的节点数
    graph = nx.Graph() # 创建一个图

    # 将Cora的边信息添加到nx图中
    for i in range(num_nodes):
        graph.add_edge(data["Lecture", "pageRank", "entity"].edge_index[i][0].item(), data["Lecture", "pageRank", "entity"].edge_index[i][1].item())
        
    # 计算每个节点的位置信息，采用kamada_kawai_layout布局方式
    pos = nx.kamada_kawai_layout(graph)

    # # Cora有7个类别，对应7个颜色
    # color = ['red', 'orange', 'blue', 'green', 'yellow', 'pink', 'darkviolet']

    # # 每个节点对应的颜色
    # node_color = [color[data.data.y[i]] for i in range(len(graph.nodes))]

    # 绘制图
    nx.draw(graph, pos, node_size=50)

    # 保存图
    plt.savefig('a.png', dpi=1280)
    plt.show()

