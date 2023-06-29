from tools import *
import os.path as osp

import pandas as pd
import torch
import argparse
import tqdm
import torch.nn.functional as F

from torch_geometric.data import HeteroData
from torch_geometric.transforms import RandomLinkSplit, ToUndirected
from torch_geometric.nn import SAGEConv, to_hetero, HeteroConv
import torch_geometric.transforms as T
from torch_geometric.loader import LinkNeighborLoader
from sklearn.metrics import roc_auc_score
if __name__ == '__main__':
    file_path="data/all_file1.csv"
    # Load the entire entity data frame into memory:
    entity_df = pd.read_csv(file_path, index_col='entityID').head(1000)

    # Split genres and convert into indicator variables:
    # genres = entity_df['genres'].str.get_dummies('|')
    # print(genres[["Action", "Adventure", "Drama", "Horror"]].head())
    # Use genres as entity input features:
    # entity_feat = torch.from_numpy(genres.values).to(torch.float)
    # assert entity_feat.size() == (9742, 20)  # 20 genres in total.

    # Load the entire pageRank data frame into memory:
    pageRank_df = pd.read_csv(file_path).head(1000)

    # Create a mapping from unique Lecture indices to range [0, num_Lecture_nodes):
    unique_Lecture_id = pageRank_df['LectureId'].unique()
    unique_Lecture_id = pd.DataFrame(data={
        'LectureId': unique_Lecture_id,
        'mappedID': pd.RangeIndex(len(unique_Lecture_id)),
    })
    print("Mapping of Lecture IDs to consecutive values:")
    print("==========================================")
    print(unique_Lecture_id.head())
    print()
    # Create a mapping from unique entity indices to range [0, num_entity_nodes):
    unique_entity_id = pageRank_df['entityID'].unique()
    unique_entity_id = pd.DataFrame(data={
        'entityID': unique_entity_id,
        'mappedID': pd.RangeIndex(len(unique_entity_id)),
    })
    print("Mapping of entity IDs to consecutive values:")
    print("===========================================")
    print(unique_entity_id.head())
    # Perform merge to obtain the edges from Lectures and entitys:
    pageRank_Lecture_id = pd.merge(pageRank_df['LectureId'], unique_Lecture_id,
                                left_on='LectureId', right_on='LectureId', how='left')
    pageRank_Lecture_id = torch.from_numpy(pageRank_Lecture_id['mappedID'].values)
    pageRank_entity_id = pd.merge(pageRank_df['entityID'], unique_entity_id,
                                left_on='entityID', right_on='entityID', how='left')
    pageRank_entity_id = torch.from_numpy(pageRank_entity_id['mappedID'].values)
    # With this, we are ready to construct our `edge_index` in COO format
    # following PyG semantics:
    edge_index_Lecture_to_entity = torch.stack([pageRank_Lecture_id, pageRank_entity_id], dim=0)
    edge_index_entity_to_Lecture = torch.stack([pageRank_entity_id, pageRank_Lecture_id], dim=0)
    df1 = pd.DataFrame(edge_index_Lecture_to_entity.numpy().T, columns=['LectureId', 'entityID'])
    df2 = pd.DataFrame(edge_index_entity_to_Lecture.numpy().T, columns=['entityID', 'Lectureid'])
    # df=pd.concat([df1,df2],axis=0)
    df=pd.merge(df1,df2,how='inner')
    
    df.drop('entityID',axis=1,inplace=True)
    df.drop_duplicates(inplace=True)
    # print("df",df.head(50))
    edge_index_Lecture_to_Lecture = torch.tensor([df['LectureId'].values, df['Lectureid'].values])

    # assert edge_index_Lecture_to_entity.size() == (2, 100836)
    print()
    print("Final edge indices pointing from Lectures to entitys:")
    print("=================================================")
    print(edge_index_Lecture_to_entity)
    print("Final edge indices pointing from entitys to Lectures:")
    print("=================================================")
    print(edge_index_entity_to_Lecture)
    print("Final edge indices pointing from Lectures to Lectures:")
    print("=================================================")
    print(edge_index_Lecture_to_Lecture)

    data = HeteroData()
    # Save node indices:
    data["Lecture"].node_id = torch.arange(len(unique_Lecture_id))
    data["entity"].node_id = torch.arange(len(entity_df))

    # Add the node features and edge indices:
    entityEncoders={'entity': SequenceEncoder()}
    entityxs = [encoder(pageRank_df[col]) for col, encoder in entityEncoders.items()]
    entity_feat = torch.cat(entityxs, dim=-1)
    data["entity"].x = entity_feat
    # LectureEncoders={'Lecture': SequenceEncoder()}
    # Lecturexs = [encoder(pageRank_df[col]) for col, encoder in LectureEncoders.items()]
    # Lecture_feat = torch.cat(Lecturexs, dim=-1)
    # data["Lecture"].x = Lecture_feat
    data["Lecture", "pageRank", "entity"].edge_index = edge_index_Lecture_to_entity
    data["Lecture", "pageRank", "Lecture"].edge_index = edge_index_Lecture_to_Lecture
    print("data[Lecture,pageRank,entity].edge_index",data["Lecture", "pageRank", "entity"].edge_index)
    # data["Lecture", "pageRank", "entity"].edge_weight = torch.from_numpy(pageRank_df['pageRank'].values)
    print("data",data)

    print("data[Lecture].node_id",data["Lecture"].node_id.shape[0])
    print("data[Lecture].num_nodes",data["Lecture"].num_nodes)
    print("data[entity].num_nodes",data["entity"].num_nodes)

    # print("edge_weight",data["Lecture", "pageRank", "entity"].edge_weight)
    # We also need to make sure to add the reverse edges from entitys to Lectures
    # in order to let a GNN be able to pass messages in both directions.
    # We can leverage the `T.ToUndirected()` transform for this from PyG:
    data = T.ToUndirected()(data)
    # For this, we first split the set of edges into
    # training (80%), validation (10%), and testing edges (10%).
    # Across the training edges, we use 70% of edges for message passing,
    # and 30% of edges for supervision.
    # We further want to generate fixed negative edges for evaluation with a ratio of 2:1.
    # Negative edges during training will be generated on-the-fly.
    # We can leverage the `RandomLinkSplit()` transform for this from PyG:
    transform = T.RandomLinkSplit(
        num_val=0.1,
        num_test=0.1,
        disjoint_train_ratio=0.3,
        neg_sampling_ratio=2.0,
        add_negative_train_samples=False,
        # key="edge_weight",  # Negative edges will inherit this attribute. 
        edge_types=[("Lecture", "pageRank", "entity"),("Lecture", "pageRank", "Lecture")],
        rev_edge_types=[("entity", "rev_pageRank", "Lecture"), ("Lecture", "pageRank", "Lecture")],
    )
    train_data, val_data, test_data = transform(data)

    def weighted_mse_loss(pred, target, weight=None):
        weight = 1. if weight is None else weight[target].to(pred.dtype)
        return (weight * (pred - target.to(pred.dtype)).pow(2)).mean()

    # Define seed edges:
    L_e_edge_label_index = train_data["Lecture", "pageRank", "entity"].edge_label_index
    L_l_edge_label_index = train_data["Lecture", "pageRank", "Lecture"].edge_label_index

    print("train_data: ", train_data)
    print("edge_label_index: ", L_e_edge_label_index)
    print("edge_label_index: ", L_l_edge_label_index)

    L_e_edge_label = train_data["Lecture", "pageRank", "entity"].edge_label
    L_l_edge_label = train_data["Lecture", "pageRank", "Lecture"].edge_label


    # print("edge_label: ", edge_label)
    # train_loader = LinkNeighborLoader(
    #     data=train_data,
    #     num_neighbors=[20, 10],
    #     neg_sampling_ratio=2.0,
    #     edge_label_index=([("Lecture", "pageRank", "entity"), L_e_edge_label_index],[("Lecture", "pageRank", "Lecture"), L_l_edge_label_index]),
    #     edge_label=L_l_edge_label,
    #     batch_size=128,
    #     shuffle=True
    # )

    # model = Model(hidden_channels=64)
    class GNN(torch.nn.Module):
        def __init__(self, hidden_channels,out_channels,num_layers):
            super().__init__()
            self.convs=torch.nn.ModuleList()
            for _ in range(num_layers):
                conv = HeteroConv({
                    ("Lecture","pageRank","entity"):SAGEConv(hidden_channels,hidden_channels),
                    ("Lecture","pageRank","Lecture"):SAGEConv(hidden_channels,hidden_channels)
                },aggr='sum')
                self.convs.append(conv)
            self.lin = torch.nn.Linear(hidden_channels, out_channels)
            # self.conv1 = SAGEConv(hidden_channels, hidden_channels)
            # self.conv2 = SAGEConv(hidden_channels, hidden_channels)
            # self.conv3 = SAGEConv(hidden_channels, hidden_channels)
        def forward(self, x:torch.Tensor, edge_index:torch.Tensor)-> torch.Tensor:
            # x = F.relu(self.conv1(x, edge_index))
            # x = F.dropout(x)
            # x = self.conv2(x, edge_index)
            # x = self.conv3(x, edge_index)
            for conv in self.convs:
                x = conv(x, edge_index)
                x = {key: F.relu(x) for key, x in x.items()}
                x = {key: F.dropout(x, p=0.5, training=self.training) for key, x in x.items()}
            return x

    class EdgeDecoder(torch.nn.Module):
        def __init__(self, hidden_channels):
            super().__init__()
            self.lin1 = torch.nn.Linear(2 * hidden_channels, hidden_channels)
            self.lin2 = torch.nn.Linear(hidden_channels, 1)

        def forward(self, z_dict, edge_label_index):
            row, col = edge_label_index
            z = torch.cat([z_dict["Lecture"][row], z_dict["entity"][col]], dim=-1)

            z = self.lin1(z).relu()
            z = self.lin2(z)
            return z.view(-1)


    class Model(torch.nn.Module):
        def __init__(self, hidden_channels):
            super().__init__()
            self.entity_lin = torch.nn.Linear(384, hidden_channels)
            self.Lecture_emb = torch.nn.Embedding(data["Lecture"].num_nodes, hidden_channels)
            # self.Lecture_lin = torch.nn.Linear(384, hidden_channels)
            self.entity_emb = torch.nn.Embedding(data["entity"].num_nodes, hidden_channels)
            self.encoder = GNN(hidden_channels=64, out_channels=64, num_layers=2)
            # self.encoder = to_hetero(self.encoder, data.metadata(), aggr='sum')
            self.decoder = EdgeDecoder(hidden_channels)

        def forward(self, data: HeteroData):
            x_dict = {
            "Lecture": self.Lecture_emb(data["Lecture"].node_id),
            "entity": self.entity_lin(data["entity"].x) + self.entity_emb(data["entity"].node_id),
            }
            z_dict = self.encoder(x_dict, data.edge_index_dict)
            return self.decoder(z_dict, data.edge_label_index)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # model = Model(hidden_channels=64).to(device)
    model = GNN(hidden_channels=64, out_channels=64, num_layers=2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    def train():
        # model.train()
        optimizer.zero_grad()
        print("train_data: ", train_data.x_dict)
        print(train_data["Lecture", "pageRank", "entity"].edge_label)
        pred = model(train_data)
        target = train_data["Lecture", "pageRank", "entity"].edge_label
        loss = F.binary_cross_entropy_with_logits(pred, target)
        # loss = weighted_mse_loss(pred, target, weight)
        loss.backward()
        optimizer.step()
        return float(loss)


    @torch.no_grad()
    def test(data):
        # model.eval()
        pred = model(data)
        pred = pred.clamp(min=0, max=5)
        target = data["Lecture", "pageRank", "entity"].edge_label.float()
        rmse = F.mse_loss(pred, target).sqrt()
        return float(rmse)


    for epoch in range(1, 301):
        loss = train()
        train_rmse = test(train_data)
        val_rmse = test(val_data)
        test_rmse = test(test_data)
        print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Train: {train_rmse:.4f}, '
            f'Val: {val_rmse:.4f}, Test: {test_rmse:.4f}')
    
