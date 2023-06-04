from tools import *
import os.path as osp

import pandas as pd
import torch
import argparse
import tqdm
import torch.nn.functional as F

from torch_geometric.data import HeteroData
from torch_geometric.transforms import RandomLinkSplit, ToUndirected
from torch_geometric.nn import SAGEConv, to_hetero
import torch_geometric.transforms as T
from torch_geometric.loader import LinkNeighborLoader
from sklearn.metrics import roc_auc_score


if __name__ == '__main__':
    file_path="data/all_file1.csv"
    df = pd.read_csv(file_path)
    Lecture_x, Lecture_mapping = load_node_csv(file_path, index_col='LectureId',encoders={
        'Lecture': SequenceEncoder()
    })
    # print('Lecture_x',Lecture_x)
    # print('Lecture_mapping',Lecture_mapping)

    entity_x, entity_mapping = load_node_csv(
        file_path, index_col='entityID',encoders={
        'entity': SequenceEncoder()
    })
    # print('entity_x',entity_x)
    # print('entity_mapping',entity_mapping)

    edge_index, edge_label = load_edge_csv(
        file_path,
        src_index_col='LectureId',
        src_mapping=Lecture_mapping,
        dst_index_col='entityID',
        dst_mapping=entity_mapping,
        encoders={'pageRank': load_csv.IdentityEncoder(dtype=torch.bool)},
    )

    data = HeteroData()
    # data['Lecture'].num_nodes = len(Lecture_mapping)  # Lectures do not have any features.
    data['Lecture'].x = Lecture_x
    data['entity'].x = entity_x
    data['Lecture', 'pageRank', 'entity'].edge_index = edge_index
    data['Lecture', 'pageRank', 'entity'].edge_label = edge_label
    
    data['Lecture'].node_id = torch.arange(len(Lecture_x))
    data['entity'].node_id = torch.arange(len(entity_x))
    print('entity_x.shape',data['entity'].x.shape)
    print('entity.node_id',data['entity'].node_id.shape)

    # node classification
    # splitt = T.RandomNodeSplit(split="train_rest", key="entity")
    # graph = splitt(data)
    # print('------------------')
    # for store in graph.node_stores:
    #     print('node store',store)
    #     print(store.train_mask)
    # print("train mask ", graph.train_mask)
    # print("graph",graph)
    # print("data",data)

    # We can now convert `data` into an appropriate format for training a
    # graph-based machine learning model:

    # 1. Add a reverse ('movie', 'rev_rates', 'user') relation for message passing.
    data = ToUndirected()(data)
    del data['entity', 'rev_pageRank', 'Lecture'].edge_label  # Remove "reverse" label.

    # 2. Perform a link-level split into training, validation, and test edges.
    transform = RandomLinkSplit(
        num_val=0.05,
        num_test=0.1,
        neg_sampling_ratio=0.0,
        edge_types=[('Lecture', 'pageRank', 'entity')],
        rev_edge_types=[('entity', 'rev_pageRank', 'Lecture')],
    )
    # Da=transform(data)
    # print("Da",Da)
    train_data, val_data, test_data = transform(data)

    # In the first hop, we sample at most 20 neighbors.
    # In the second hop, we sample at most 10 neighbors.
    # In addition, during training, we want to sample negative edges on-the-fly with
    # a ratio of 2:1.
    # We can make use of the `loader.LinkNeighborLoader` from PyG:

    # Define seed edges:
    edge_label_index = train_data["Lecture", "pageRank", "entity"].edge_index
    edge_label_label = train_data["Lecture", "pageRank", "entity"].edge_label

    train_loader = LinkNeighborLoader(
        data=train_data,
        num_neighbors=[2, 1],
        neg_sampling_ratio=2.0,
        edge_label_index=(("Lecture", "pageRank", "entity"), edge_label_index),
        edge_label=edge_label_label,
        batch_size=128,
        shuffle=True,
    )

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
        def __init__(self, hidden_channels):
            super().__init__()
            # Since the dataset does not come with rich features, we also learn two
            # embedding matrices for users and movies:
            self.entity_lin = torch.nn.Linear(384, hidden_channels)
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

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: '{device}'")
    model = Model(hidden_channels=64)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    for epoch in range(1, 6):
        total_loss = total_examples = 0
        for sampled_data in tqdm.tqdm(train_loader):
            optimizer.zero_grad()
            sampled_data.to(device)
            pred = model(sampled_data)
            # print('pred',pred)
            ground_truth = sampled_data["Lecture", "pageRank", "entity"].edge_label
            # print('ground_truth',ground_truth)
            loss = F.binary_cross_entropy_with_logits(pred.unsqueeze(1), ground_truth)
            loss.backward()
            optimizer.step()
            total_loss += float(loss) * pred.numel()
            total_examples += pred.numel()
        print(f"Epoch: {epoch:03d}, Loss: {total_loss / total_examples:.4f}")

    # Define the validation seed edges:
    edge_label_index = val_data["Lecture", "pageRank", "entity"].edge_label_index
    edge_label = val_data["Lecture", "pageRank", "entity"].edge_label
    val_loader = LinkNeighborLoader(
        data=val_data,
        num_neighbors=[20, 10],
        edge_label_index=(("Lecture", "pageRank", "entity"), edge_label_index),
        edge_label=edge_label,
        batch_size=3 * 128,
        shuffle=False,
    )
    sampled_data = next(iter(val_loader))

    preds = []
    ground_truths = []
    for sampled_data in tqdm.tqdm(val_loader):
        with torch.no_grad():
            sampled_data.to(device)
            preds.append(model(sampled_data).unsqueeze(1))
            ground_truths.append(sampled_data["Lecture", "pageRank", "entity"].edge_label)
    pred = torch.cat(preds, dim=0).cpu().numpy()
    ground_truth = torch.cat(ground_truths, dim=0).cpu().numpy()
    # print('pred',pred)
    # print('ground_truth',ground_truth)
    print('pred.shape',pred.size)
    print('ground_truth',ground_truth.size)
    from sklearn.utils.multiclass import type_of_target
    # print(type_of_target(pred))
    # print(type_of_target(ground_truth))
    # auc = roc_auc_score(ground_truth, pred)
    # print()
    # print(f"Validation AUC: {auc:.4f}")

    # 导包
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve, auc

    # # 计算
    # fpr, tpr, thread = roc_curve(ground_truth, pred)
    # roc_auc = auc(fpr, tpr)
    # print(f"Validation AUC: {roc_auc:.4f}")
    # # 绘图
    # plt.figure()
    # lw = 2
    # plt.plot(fpr, tpr, color='darkorange',
    #         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    # plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    # plt.xlim([0.0, 1.0])
    # plt.ylim([0.0, 1.05])
    # plt.xlabel('False Positive Rate')
    # plt.ylabel('True Positive Rate')
    # plt.title('Receiver operating characteristic example')
    # plt.legend(loc="lower right")
    # plt.savefig('roc.png',)
    # plt.show()

    from sklearn.metrics import mean_squared_error
    from sklearn.metrics import mean_absolute_error
    print("MAE:",mean_absolute_error(ground_truth, pred))
    print("MSE:",mean_squared_error(ground_truth, pred, squared=False))

    from sklearn import metrics
    print("RMSE:",np.sqrt(metrics.mean_squared_error(ground_truth, pred)))


