import torch
from torch_geometric.nn import SAGEConv, to_hetero

class GNN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = SAGEConv((-1, -1), 32)
        self.conv2 = SAGEConv((32, 32), 32)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index).relu()
        return x

model = GNN()

node_types = ['paper', 'author']
edge_types = [
    ('paper', 'cites', 'paper'),
    ('paper', 'written_by', 'author'),
    ('author', 'writes', 'paper'),
]
metadata = (node_types, edge_types)

model = to_hetero(model, metadata)
model(metadata.x_dict, metadata.edge_index_dict)