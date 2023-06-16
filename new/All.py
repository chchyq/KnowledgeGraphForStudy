from tools import *
import os.path as osp

import pandas as pd
import torch
import argparse
import tqdm
import torch.nn.functional as F

from torch_geometric.data import HeteroData
from torch_geometric.transforms import RandomLinkSplit, ToUndirected
from torch_geometric.nn import SAGEConv, to_hetero, HGTConv, Linear, GATConv, SimpleConv
import torch_geometric.transforms as T
from torch_geometric.loader import LinkNeighborLoader
from sklearn.metrics import roc_auc_score
if __name__ == '__main__':
    file_path="data/all_file1.csv"
    # Load the entire entity data frame into memory:
    entity_df = pd.read_csv(file_path, index_col='entityID')#.head(1000)

    # Split genres and convert into indicator variables:
    # genres = entity_df['genres'].str.get_dummies('|')
    # print(genres[["Action", "Adventure", "Drama", "Horror"]].head())
    # Use genres as entity input features:
    # entity_feat = torch.from_numpy(genres.values).to(torch.float)
    # assert entity_feat.size() == (9742, 20)  # 20 genres in total.

    # Load the entire pageRank data frame into memory:
    pageRank_df = pd.read_csv(file_path)#.head(1000)

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
    # assert edge_index_Lecture_to_entity.size() == (2, 100836)
    print()
    print("Final edge indices pointing from Lectures to entitys:")
    print("=================================================")
    print(edge_index_Lecture_to_entity)

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
    data["Lecture", "pageRank", "entity"].edge_weight = torch.from_numpy(pageRank_df['pageRank'].values)
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
        edge_types=("Lecture", "pageRank", "entity"),
        rev_edge_types=("entity", "rev_pageRank", "Lecture"), 
    )
    train_data, val_data, test_data = transform(data)

    # Define seed edges:
    edge_label_index = train_data["Lecture", "pageRank", "entity"].edge_label_index
    print("train_data: ", train_data)
    print("edge_label_index: ", edge_label_index)

    edge_label = train_data["Lecture", "pageRank", "entity"].edge_label
    # print("edge_label: ", edge_label)
    train_loader = LinkNeighborLoader(
        data=train_data,
        num_neighbors=[20, 10],
        neg_sampling_ratio=2.0,
        edge_label_index=(("Lecture", "pageRank", "entity"), edge_label_index),
        edge_label=edge_label,
        batch_size=128,
        shuffle=True
    )
    class Conv(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = SimpleConv()
            self.conv2 = SimpleConv()
            self.conv3 = SimpleConv()
        def forward(self, x:torch.Tensor, edge_index:torch.Tensor)-> torch.Tensor:
            x = F.relu(self.conv1(x, edge_index))
            x = F.dropout(x)
            x = self.conv2(x, edge_index)
            x = self.conv3(x, edge_index)
            return x
        
    class GNN(torch.nn.Module):
        def __init__(self, hidden_channels):
            super().__init__()
            self.conv1 = SAGEConv(hidden_channels, hidden_channels)
            self.conv2 = SAGEConv(hidden_channels, hidden_channels)
            self.conv3 = SAGEConv(hidden_channels, hidden_channels)
        def forward(self, x:torch.Tensor, edge_index:torch.Tensor)-> torch.Tensor:
            x = F.relu(self.conv1(x, edge_index))
            x = F.dropout(x)
            x = self.conv2(x, edge_index)
            x = self.conv3(x, edge_index)
            return x
    
    class HGT(torch.nn.Module):
        def __init__(self, hidden_channels, out_channels, num_heads, num_layers):
            super().__init__()

            self.lin_dict = torch.nn.ModuleDict()
            for node_type in data.node_types:
                self.lin_dict[node_type] = Linear(-1, hidden_channels)

            self.convs = torch.nn.ModuleList()
            for _ in range(num_layers):
                conv = HGTConv(hidden_channels, hidden_channels, data.metadata(),
                            num_heads, group='sum')
                self.convs.append(conv)

            self.lin = Linear(hidden_channels, out_channels)

        def forward(self, x_dict, edge_index_dict):
            for node_type, x in x_dict.items():
                x_dict[node_type] = self.lin_dict[node_type](x).relu_()

            for conv in self.convs:
                x_dict = conv(x_dict, edge_index_dict)

            return x_dict

    class GAT(torch.nn.Module):
        def __init__(self, hidden_channels):
            super(GAT, self).__init__()
            self.conv1 = GATConv(hidden_channels, hidden_channels,add_self_loops=False, heads=4)
            self.conv2 = GATConv(hidden_channels*4, hidden_channels,add_self_loops=False)
            # self.conv3 = GATConv(hidden_channels*8, hidden_channels,add_self_loops=False)
            
        def forward(self, x:torch.Tensor, edge_index:torch.Tensor)-> torch.Tensor:
            x = self.conv1(x, edge_index)
            x = F.relu(x)
            # x = F.dropout(x, training=self.training)
            x = self.conv2(x, edge_index)
            # x = self.conv3(x, edge_index)
            return x

    # Our final classifier applies the dot-product between source and destination
    # node embeddings to derive edge-level predictions:
    class Classifier(torch.nn.Module):
        def forward(self, x_Lecture:torch.Tensor, x_entity:torch.Tensor, edge_label_index:torch.Tensor):
            # Convert node embeddings to edge-level representations:
            edge_feat_Lecture = x_Lecture[edge_label_index[0]]
            edge_feat_entity = x_entity[edge_label_index[1]]
            # Apply dot-product to get a prediction per supervision edge:
            return (edge_feat_Lecture * edge_feat_entity).sum(dim=-1)

    class Model(torch.nn.Module):
        def __init__(self, hidden_channels,model):
            super().__init__()
            # Since the dataset does not come with rich features, we also learn two
            # embedding matrices for Lectures and entitys:
            self.entity_lin = torch.nn.Linear(384, hidden_channels)
            self.Lecture_emb = torch.nn.Embedding(data["Lecture"].num_nodes, hidden_channels)
            # self.Lecture_lin = torch.nn.Linear(384, hidden_channels)
            self.entity_emb = torch.nn.Embedding(data["entity"].num_nodes, hidden_channels)
            # Instantiate homogeneous GNN:
            self.model = model
            self.classifier = Classifier()
        def forward(self, data: HeteroData) -> torch.Tensor:
            # print("data",self.Lecture_lin(data["lecture"].x).shape)   
            x_dict = {
            # "Lecture": self.Lecture_lin(data["Lecture"].x)+self.Lecture_emb(edge_index_Lecture_to_entity[0]),
            "Lecture": self.Lecture_emb(data["Lecture"].node_id),
            "entity": self.entity_lin(data["entity"].x) + self.entity_emb(data["entity"].node_id),
            } 
            # `x_dict` holds feature matrices of all node types
            # `edge_index_dict` holds all edge indices of all edge types
            x_dict = self.model(x_dict, data.edge_index_dict)
            pred = self.classifier(
                x_dict["Lecture"],
                x_dict["entity"],
                data["Lecture", "pageRank", "entity"].edge_label_index,
            )
            return pred

    model0 = Conv()
    model0 = to_hetero(model0,metadata=data.metadata())
    ConvModel= Model(hidden_channels=64,model=model0)        
    
    model1 = GNN(hidden_channels=64)
    model1 = to_hetero(model1,metadata=data.metadata())
    GnnModel= Model(hidden_channels=64,model=model1)

    model2 = HGT(hidden_channels=64, out_channels=64,
                num_heads=2, num_layers=4)
    HgtModel= Model(hidden_channels=64,model=model2)

    model3 = GAT(hidden_channels=64)
    model3 = to_hetero(model3, metadata=data.metadata())#aggr='sum'
    GatModel= Model(hidden_channels=64,model=model3)

    import tqdm
    import torch.nn.functional as F
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ConvModel = ConvModel.to(device)
    GnnModel = GnnModel.to(device)
    HgtModel = HgtModel.to(device)
    GatModel = GatModel.to(device)

    res=[]

    print("=================================================")
    print("Conv")
    optimizer = torch.optim.Adam(ConvModel.parameters(), lr=0.001)
    for epoch in range(1, 6):
        total_loss = total_examples = 0
        for sampled_data in tqdm.tqdm(train_loader):
            optimizer.zero_grad()
            sampled_data.to(device)
            pred = ConvModel(sampled_data)
            ground_truth = sampled_data["Lecture", "pageRank", "entity"].edge_label
            loss = F.binary_cross_entropy_with_logits(pred, ground_truth)
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
            preds.append(ConvModel(sampled_data))
            ground_truths.append(sampled_data["Lecture", "pageRank", "entity"].edge_label)
    pred = torch.cat(preds, dim=0).cpu().numpy()
    ground_truth = torch.cat(ground_truths, dim=0).cpu().numpy()
    auc = roc_auc_score(ground_truth, pred)
    print()
    print(f"Validation AUC: {auc:.4f}")

    test_data.to(device)
    predict=ConvModel(test_data)
    truth=test_data["Lecture", "pageRank", "entity"].edge_label
    auc=roc_auc_score(truth.cpu().detach().numpy(),predict.cpu().detach().numpy())
    print(f"Test AUC: {auc:.4f}")
    res.append(auc)

    print("=================================================")
    print("GNN")
    optimizer = torch.optim.Adam(GnnModel.parameters(), lr=0.001)
    for epoch in range(1, 6):
        total_loss = total_examples = 0
        for sampled_data in tqdm.tqdm(train_loader):
            optimizer.zero_grad()
            sampled_data.to(device)
            pred = GnnModel(sampled_data)
            ground_truth = sampled_data["Lecture", "pageRank", "entity"].edge_label
            loss = F.binary_cross_entropy_with_logits(pred, ground_truth)
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

    from sklearn.metrics import roc_auc_score
    preds = []
    ground_truths = []
    for sampled_data in tqdm.tqdm(val_loader):
        with torch.no_grad():
            sampled_data.to(device)
            preds.append(GnnModel(sampled_data))
            ground_truths.append(sampled_data["Lecture", "pageRank", "entity"].edge_label)
    pred = torch.cat(preds, dim=0).cpu().numpy()
    ground_truth = torch.cat(ground_truths, dim=0).cpu().numpy()
    auc = roc_auc_score(ground_truth, pred)
    print()
    print(f"Validation AUC: {auc:.4f}")

    test_data.to(device)
    predict=GnnModel(test_data)
    truth=test_data["Lecture", "pageRank", "entity"].edge_label
    auc=roc_auc_score(truth.cpu().detach().numpy(),predict.cpu().detach().numpy())
    print(f"Test AUC: {auc:.4f}")
    res.append(auc)

    print("=================================================")
    print("HGT")
    optimizer = torch.optim.Adam(HgtModel.parameters(), lr=0.001)
    for epoch in range(1, 6):
        total_loss = total_examples = 0
        for sampled_data in tqdm.tqdm(train_loader):
            optimizer.zero_grad()
            sampled_data.to(device)
            pred = HgtModel(sampled_data)
            ground_truth = sampled_data["Lecture", "pageRank", "entity"].edge_label
            loss = F.binary_cross_entropy_with_logits(pred, ground_truth)
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
            preds.append(HgtModel(sampled_data))
            ground_truths.append(sampled_data["Lecture", "pageRank", "entity"].edge_label)
    pred = torch.cat(preds, dim=0).cpu().numpy()
    ground_truth = torch.cat(ground_truths, dim=0).cpu().numpy()
    auc = roc_auc_score(ground_truth, pred)
    print()
    print(f"Validation AUC: {auc:.4f}")

    test_data.to(device)
    predict=HgtModel(test_data)
    truth=test_data["Lecture", "pageRank", "entity"].edge_label
    auc=roc_auc_score(truth.cpu().detach().numpy(),predict.cpu().detach().numpy())
    print(f"Test AUC: {auc:.4f}")
    res.append(auc)

    print("=================================================")
    print("GAT")
    optimizer = torch.optim.Adam(GatModel.parameters(), lr=0.001)
    for epoch in range(1, 6):
        total_loss = total_examples = 0
        for sampled_data in tqdm.tqdm(train_loader):
            optimizer.zero_grad()
            sampled_data.to(device)
            pred = GatModel(sampled_data)
            ground_truth = sampled_data["Lecture", "pageRank", "entity"].edge_label
            loss = F.binary_cross_entropy_with_logits(pred, ground_truth)
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
            preds.append(GatModel(sampled_data))
            ground_truths.append(sampled_data["Lecture", "pageRank", "entity"].edge_label)
    pred = torch.cat(preds, dim=0).cpu().numpy()
    ground_truth = torch.cat(ground_truths, dim=0).cpu().numpy()
    auc = roc_auc_score(ground_truth, pred)
    print()
    print(f"Validation AUC: {auc:.4f}")

    test_data.to(device)
    predict=GatModel(test_data)
    truth=test_data["Lecture", "pageRank", "entity"].edge_label
    auc=roc_auc_score(truth.cpu().detach().numpy(),predict.cpu().detach().numpy())
    print(f"Test AUC: {auc:.4f}")
    res.append(auc)

    import matplotlib.pyplot as plt
    import seaborn as sns
    
    models = ['Connv','GNNConv','HGTConv','GATConv']
    plt.rcParams['figure.figsize']=10,6 
    plt.style.use('fivethirtyeight')
    ax = sns.barplot(x=models,y=res,palette="deep")
    plt.xlabel("Models",fontsize=13)
    plt.ylabel("% of Accuracy",fontsize=13)
    plt.title("Accuracy of different Models",fontsize=20,color='black')
    plt.xticks(fontsize=10,horizontalalignment='center',rotation=0)
    plt.yticks(fontsize=10)
    for p in ax.patches:
        width,height = p.get_width(),p.get_height()
        x, y = p.get_xy() 
        ax.annotate(f'{height:.6%}',(x+width/2,y+height*1.02),ha='center',fontsize=10,color='black')
    plt.show()
    plt.savefig('Accuracy.png')