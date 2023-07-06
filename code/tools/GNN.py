import torch
from torch_geometric.nn import SAGEConv, to_hetero
import torch.nn.functional as F
class GNN(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        self.conv1 = SAGEConv(hidden_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
    # def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return x
# Our final classifier applies the dot-product between source and destination
# node embeddings to derive edge-level predictions:
class Classifier(torch.nn.Module):
    # def forward(self, x_Lecture: Tensor, x_entity: Tensor, edge_label_index: Tensor) -> Tensor:
    def forward(self, x_Lecture, x_entity, edge_label_index):

        # Convert node embeddings to edge-level representations:
        edge_feat_Lecture = x_Lecture[edge_label_index[0]]
        edge_feat_entity = x_entity[edge_label_index[1]]
        # Apply dot-product to get a prediction per supervision edge:
        return (edge_feat_Lecture * edge_feat_entity).sum(dim=-1)

class Model(torch.nn.Module):
    def __init__(self, data, hidden_channels):
        super().__init__()
        # Since the dataset does not come with rich features, we also learn two
        # embedding matrices for users and movies:
        self.entity_lin = torch.nn.Linear(20, hidden_channels)
        self.Lecture_emb = torch.nn.Embedding(data["Lecture"].num_nodes, hidden_channels)
        self.entity_emb = torch.nn.Embedding(data["entity"].num_nodes, hidden_channels)
        # Instantiate homogeneous GNN:
        self.gnn = GNN(hidden_channels)
        # Convert GNN model into a heterogeneous variant:
        self.gnn = to_hetero(self.gnn, metadata=data.metadata())
        self.classifier = Classifier()
    # def forward(self, data: HeteroData) -> Tensor:
    def forward(self, data):
        x_dict = {
          "Lecture": self.Lecture_emb(data["Lecture"].node_id),
          "entity": self.entity_lin(data["entity"].x) + self.entity_emb(data["entity"].node_id),
        } 
        # `x_dict` holds feature matrices of all node types
        # `edge_index_dict` holds all edge indices of all edge types
        x_dict = self.gnn(x_dict, data.edge_index_dict)
        pred = self.classifier(
            x_dict["Lecture"],
            x_dict["entity"],
            data["Lecture", "pageRank", "entity"].edge_label_index,
        )
        return pred
        
# model = Model(hidden_channels=64)