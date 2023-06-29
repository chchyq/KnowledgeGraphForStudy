from tools import *
import os.path as osp

import pandas as pd
import torch
import argparse
import tqdm
import torch.nn.functional as F
import numpy as np

from torch_geometric.data import HeteroData
from torch_geometric.transforms import RandomLinkSplit, ToUndirected
from torch_geometric.nn import SAGEConv, to_hetero
import torch_geometric.transforms as T
from torch_geometric.loader import LinkNeighborLoader
from sklearn.metrics import roc_auc_score

from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv
from torch_geometric.utils import negative_sampling
if __name__ == '__main__':
    file_path="data/all_file1.csv"

    # Load the entire pageRank data frame into memory:
    pageRank_df = pd.read_csv(file_path)#.head(1000)
    
    unique_Lecture_name = pageRank_df['Lecture'].unique()
    unique_Lecture_name = pd.DataFrame(data={
        'Lecture': unique_Lecture_name,
        'mappedID': pd.RangeIndex(len(unique_Lecture_name)),
    })

        #embeddings
    file="data/Transcriptions.csv"
    df_Transcriptions = pd.read_csv(file).dropna()
    merged_df = pd.merge(df_Transcriptions, unique_Lecture_name, left_on='Lecture', right_on='Lecture', how='left').dropna()
    print(merged_df.head())
    print(merged_df.shape)
    
    pageRank_df = pd.merge(merged_df, pageRank_df, left_on='Lecture', right_on='Lecture', how='inner').dropna()
    print(pageRank_df.head())

    unique_Lecture_name = pageRank_df['Lecture'].unique()
    unique_Lecture_name = pd.DataFrame(data={
        'Lecture': unique_Lecture_name,
        'mappedID': pd.RangeIndex(len(unique_Lecture_name)),
    })

    # Create a mapping from unique Lecture indices to range [0, num_Lecture_nodes):
    unique_Lecture_id = pageRank_df['LectureId'].unique()
    unique_Lecture_id = pd.DataFrame(data={
        'LectureId': unique_Lecture_id,
        'mappedID': pd.RangeIndex(len(unique_Lecture_id)),
    })
    print("uqniue_Lecture_id.shape:",unique_Lecture_id.shape)
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
    array = np.array(df)
    edge_index_Lecture_to_Lecture = torch.stack([torch.from_numpy(array[:,0]), torch.from_numpy(array[:,1])], dim=0)

    data = HeteroData()
    # Save node indices:
    data["Lecture"].node_id = torch.arange(len(unique_Lecture_id))

    unique_Transcription_id = df_Transcriptions['Lecture'].unique()
    
    unique_Transcription_id = pd.DataFrame(data={
        'Lecture': unique_Transcription_id,
        'mappedID': pd.RangeIndex(len(unique_Transcription_id)),
    })
    mer = pd.merge(df_Transcriptions, unique_Transcription_id, left_on='Lecture', right_on='Lecture', how='left').dropna()
    print('mer',mer.shape)
    print('mer',mer.head())


    LectureEncoders={'Transcription': SequenceEncoder()}
    Lecturexs = [encoder(mer[col]) for col, encoder in LectureEncoders.items()]
    Lecture_feat = torch.cat(Lecturexs, dim=-1)
    data["Lecture"].x = Lecture_feat
    print(data)

    data["Lecture", "pageRank", "Lecture"].edge_index = edge_index_Lecture_to_Lecture

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    data = data.to_homogeneous()
    print(data)
    transform = T.Compose([
        T.NormalizeFeatures(),
        T.ToDevice(device),
        T.RandomLinkSplit(num_val=0.1, num_test=0.1, is_undirected=True,
                        add_negative_train_samples=False),
    ])
    # dataset = Planetoid('/data/pyg_data/Planetoid', name='Cora', transform=transform)
    train_data, val_data, test_data = transform(data)
    print("train_data",train_data)
    print("val_data",val_data)
    print("test_data",test_data)

    class Net(torch.nn.Module):
        def __init__(self, in_channels, hidden_channels, out_channels):
            super().__init__()
            self.conv1 = GCNConv(in_channels, hidden_channels)
            self.conv2 = GCNConv(hidden_channels, out_channels)
        def encode(self, x, edge_index):
            x = self.conv1(x, edge_index).relu()
            return self.conv2(x, edge_index)
        def decode(self, z, edge_label_index):
            return (z[edge_label_index[0]] * z[edge_label_index[1]]).sum(dim=-1)
        def decode_all(self, z):
            prob_adj = z @ z.t()
            return (prob_adj > 0).nonzero(as_tuple=False).t()
    model = Net(data.num_features, 128, 64).to(device)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=0.01)
    criterion = torch.nn.BCEWithLogitsLoss()
    def train():
        model.train()
        optimizer.zero_grad()
        z = model.encode(train_data.x, train_data.edge_index)
        # We perform a new round of negative sampling for every training epoch:
        neg_edge_index = negative_sampling(
            edge_index=train_data.edge_index, num_nodes=train_data.num_nodes,
            num_neg_samples=train_data.edge_label_index.size(1), method='sparse')
        edge_label_index = torch.cat(
            [train_data.edge_label_index, neg_edge_index],
            dim=-1,
        )
        edge_label = torch.cat([
            train_data.edge_label,
            train_data.edge_label.new_zeros(neg_edge_index.size(1))
        ], dim=0)
        out = model.decode(z, edge_label_index).view(-1)
        loss = criterion(out, edge_label)
        loss.backward()
        optimizer.step()
        return loss
    @torch.no_grad()
    def test(data):
        model.eval()
        z = model.encode(data.x, data.edge_index)
        out = model.decode(z, data.edge_label_index).view(-1).sigmoid()
        return roc_auc_score(data.edge_label.cpu().numpy(), out.cpu().numpy())
    best_val_auc = final_test_auc = 0
    for epoch in range(1, 101):
        loss = train()
        val_auc = test(val_data)
        test_auc = test(test_data)
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            final_test_auc = test_auc
        print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Val: {val_auc:.4f}, '
            f'Test: {test_auc:.4f}')
    print(f'Final Test: {final_test_auc:.4f}')
    z = model.encode(test_data.x, test_data.edge_index)
    final_edge_index = model.decode_all(z)
    print(final_edge_index.size())