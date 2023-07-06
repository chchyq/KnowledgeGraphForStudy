import pandas as pd
import torch
import tqdm
import torch_geometric as tg
from torch_geometric import nn
from torch_geometric import transforms
from sklearn import metrics
import tqdm
import torch.nn.functional as F
from tools import SequenceEncoder
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns

class Conv(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.SimpleConv()
            self.conv2 = nn.SimpleConv()
            self.conv3 = nn.SimpleConv()
        def forward(self, x:torch.Tensor, edge_index:torch.Tensor)-> torch.Tensor:
            x = F.relu(self.conv1(x, edge_index))
            x = F.dropout(x)
            x = self.conv2(x, edge_index)
            x = self.conv3(x, edge_index)
            return x
        

class GNN(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        self.conv1 = nn.SAGEConv(hidden_channels, hidden_channels)
        self.conv2 = nn.SAGEConv(hidden_channels, hidden_channels)
        self.conv3 = nn.SAGEConv(hidden_channels, hidden_channels)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x)
        x = self.conv2(x, edge_index)
        x = self.conv3(x, edge_index)
        return x
class HGT(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels, num_heads, num_layers, data):
        super().__init__()

        self.lin_dict = torch.nn.ModuleDict()
        for node_type in data.node_types:
            self.lin_dict[node_type] = nn.Linear(-1, hidden_channels)

        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            conv = nn.HGTConv(hidden_channels, hidden_channels, data.metadata(),
                        num_heads, group='sum')
            self.convs.append(conv)

        self.lin = nn.Linear(hidden_channels, out_channels)

    def forward(self, x_dict, edge_index_dict):
        for node_type, x in x_dict.items():
            x_dict[node_type] = self.lin_dict[node_type](x).relu_()

        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)

        return x_dict

class GAT(torch.nn.Module):
    def __init__(self, hidden_channels):
        super(GAT, self).__init__()
        self.conv1 = nn.GATConv(hidden_channels, hidden_channels,add_self_loops=False, heads=4)
        self.conv2 = nn.GATConv(hidden_channels*4, hidden_channels,add_self_loops=False)
        
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
    def forward(self, of_embeddings, to_embeddings, edge_label_index):
        edge_of = of_embeddings[edge_label_index[0]]
        edge_to = to_embeddings[edge_label_index[1]]
        return (edge_of * edge_to).sum(dim=-1)


class Model(torch.nn.Module):
    def __init__(self, hidden_channels, model, data, of, to, ref):
        super().__init__()
        self.of, self.too, self.ref = of, to, ref
        # Since the dataset does not come with rich features, we also learn two
        # embedding matrices for Lectures and entitys:
        self.entity_lin = torch.nn.Linear(data[to].num_features, hidden_channels)
        self.of_emb = torch.nn.Embedding(
            data[of].num_nodes, hidden_channels)
        self.to_emb = torch.nn.Embedding(
            data[to].num_nodes, hidden_channels)
        # Instantiate homogeneous GNN:
        self.model = model
        self.classifier = Classifier()

    def forward(self, data) -> torch.Tensor:
        x_dict = {
            # of: self.Lecture_lin(data[of].x)+self.Lecture_emb(edge_index_Lecture_to_entity[0]),
            self.of: self.of_emb(data[self.of].node_id),
            self.too: self.entity_lin(data[self.too].x) + self.to_emb(data[self.too].node_id),
        }
        # `x_dict` holds feature matrices of all node types
        # `edge_index_dict` holds all edge indices of all edge types
        x_dict = self.model(x_dict, data.edge_index_dict)
        pred = self.classifier(
            x_dict[self.of],
            x_dict[self.too],
            data[self.of, self.ref, self.too].edge_label_index,
        )
        return pred


def create_pairs(data, col, id_col):
    unique = data[id_col].unique()
    mapped = pd.DataFrame(
        data={id_col: unique, 'mappedID': pd.RangeIndex(len(unique))})
    edges = pd.merge(data[id_col], mapped,
                     left_on=id_col, right_on=id_col, how='left')
    edges = torch.from_numpy(edges['mappedID'].values)
    return data[col].unique(), unique, edges


def read_data(file_path, of, of_id, to, to_id, ref, take=None):
    pageRank_df = pd.read_csv(file_path) if take is None else pd.read_csv(
        file_path).head(take)

    of_mapping, of_unique, of_edges = create_pairs(pageRank_df, of, of_id)
    _, _, to_edges = create_pairs(pageRank_df, to, to_id)

    of_to_edges = torch.stack([of_edges, to_edges], dim=0)

    data = tg.data.HeteroData()
    data[of].node_id = torch.arange(len(of_unique))
    data[to].node_id = torch.arange(len(pageRank_df))

    encoder = SequenceEncoder()
    to_encoding = encoder(pageRank_df[to])
    data[to].x = to_encoding

    data[of, ref, to].edge_index = of_to_edges
    data[of, ref, to].edge_weight = torch.from_numpy(
        pageRank_df[ref].values)

    return transforms.ToUndirected()(data), of_mapping


def get_data_iterator(data, of, to, ref, **kwargs):
    edge_label_index = data[of,
                            ref, to].edge_label_index
    edge_label = data[of, ref, to].edge_label
    return tqdm.tqdm(tg.loader.LinkNeighborLoader(
        data=data,
        num_neighbors=[20, 10],
        edge_label_index=((of, ref, to), edge_label_index),
        edge_label=edge_label,
        **kwargs
    ))


def sample_model(model, data, of, to, ref):
    return model(data), data[of, ref, to].edge_label


def train(model, of, to, ref, iterator):
    optimizer = torch.optim.Adam(model.parameters())
    for epoch in range(12):
        total_loss = total_examples = 0
        for sampled_data in iterator:
            optimizer.zero_grad()
            pred, ground_truth = sample_model(model, sampled_data, of, to, ref)
            loss = F.binary_cross_entropy_with_logits(pred, ground_truth)
            loss.backward()
            optimizer.step()
            total_loss += float(loss) * pred.numel()
            total_examples += pred.numel()
        print(f'Epoch: {epoch:03d}, Loss: {total_loss / total_examples:.4f}')


def evaluate(model, of, to, ref, iterator):
    with torch.no_grad():
        results = [sample_model(model, batch, of, to, ref)
                   for batch in iterator]
    preds, ground_truths = (torch.cat(dataset, dim=0).cpu().numpy()
                            for dataset in zip(*results))
    auc = metrics.roc_auc_score(ground_truths, preds)
    return auc


def compare_nodes(tensors, results, top=5):
    of_dict = defaultdict(dict)
    for i, of in enumerate(tensors.edge_label_index[0]):
        to = tensors.edge_label_index[1][i].item()
        of_dict[of.item()][to] = results[i].item()
    similarities = defaultdict(int)
    for i in of_dict.keys():
        for j in range(i):
            for common in (set(of_dict[i]) & set(of_dict[j])):
                similarities[(i, j)] += of_dict[i][common]
    return sorted(similarities.items(), key=lambda e: e[1], reverse=True)[:top]


def ids_to_name(data, mapping):
    return [mapping[d] for d in data]


def plot_accuracy(title, res):   
    plt.rcParams['figure.figsize']=10,6 
    plt.style.use('fivethirtyeight')
    ax = sns.barplot(x=title,y=res,palette="deep")
    plt.xlabel("Models",fontsize=13)
    plt.ylabel("% of AUC",fontsize=13)
    plt.title("Auc of different Models",fontsize=20,color='black')
    plt.xticks(fontsize=10,horizontalalignment='center',rotation=0)
    plt.yticks(fontsize=10)
    for p in ax.patches:
        width,height = p.get_width(),p.get_height()
        x, y = p.get_xy() 
        ax.annotate(f'{height:.6%}',(x+width/2,y+height*1.02),ha='center',fontsize=10,color='black')
    plt.show()
    plt.savefig('AUC of models.png')
import networkx as nx    
def draw(edge_index, name=None):
    G = nx.Graph(node_size=15, font_size=8)
    src = edge_index[0].cpu().numpy()
    dst = edge_index[1].cpu().numpy()
    edgelist = zip(src, dst)
    for i, j in edgelist:
        print(i, j)
        G.add_edge(i, j)
    plt.figure(figsize=(20, 14)) # 设置画布的大小
    nx.draw_networkx(G)
    plt.savefig('{}.png'.format(name if name else 'path'))


def main():
    file_path = 'data/all_file1.csv'
    of, of_id, to, to_id, ref = 'Lecture', 'LectureId', 'entity', 'entityID', 'pageRank'
    data, of_mapping = read_data(
        file_path, of, of_id, to, to_id, ref)#, take=10000)
    
    # import networkx as nx
    # G = nx.Graph()
    # G.add_nodes_from(data['Lecture'].node_id.numpy())
    # G.add_edges_from(data['lecture', 'pageRank', 'entity'].edge_index.numpy().T)

    draw(data[of,ref, to].edge_index, name='Lecture_to_entity')
    


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print('data.num_features', data.num_features)
    model1 = nn.to_hetero(GNN(hidden_channels=64), metadata=data.metadata())
    GnnModel = Model(hidden_channels=64, model=model1,
                     data=data, of=of, to=to, ref=ref).to(device)
    
    model0 = nn.to_hetero(Conv(),metadata=data.metadata())
    ConvModel= Model(hidden_channels=64,model=model0,
                     data=data, of=of, to=to, ref=ref).to(device)
    
    model2 = HGT(hidden_channels=64, out_channels=64,
                num_heads=2, num_layers=4,data=data)
    HgtModel= Model(hidden_channels=64,model=model2,
                     data=data, of=of, to=to, ref=ref).to(device)

    model3 = nn.to_hetero(GAT(hidden_channels=64), metadata=data.metadata())#aggr='sum'
    GatModel= Model(hidden_channels=64,model=model3,
                     data=data, of=of, to=to, ref=ref).to(device)

    models = [ConvModel,GnnModel,HgtModel,GatModel]     

    # Across the training edges, we use 70% of edges for message passing,
    # and 30% of edges for supervision.
    # We further want to generate fixed negative edges for evaluation with a ratio of 2:1.
    # Negative edges during training will be generated on-the-fly.
    transform = transforms.RandomLinkSplit(
        num_val=0.1,
        num_test=0.1,
        disjoint_train_ratio=0.3,
        neg_sampling_ratio=2.0,
        add_negative_train_samples=False,
        edge_types=(of, ref, to),
        rev_edge_types=(to, 'rev_pageRank', of),
    )
    train_data, val_data, test_data = (
        dataset.to(device) for dataset in transform(data))

    train_iterator = get_data_iterator(
        train_data, of, to, ref, neg_sampling_ratio=2.0, batch_size=128, shuffle=True)
    
    res=[]
    for model in models:
        train(model, of, to, ref, train_iterator)

        val_iterator = get_data_iterator(
            val_data, of, to, ref, batch_size=128*3, shuffle=False)
        auc = evaluate(model, of, to, ref, val_iterator)
        print(f'Validation AUC: {auc:.4f}')

        predict = GnnModel(test_data)
        truth = test_data[of, ref, to].edge_label
        auc = metrics.roc_auc_score(truth.cpu().detach().numpy(),
                                    predict.cpu().detach().numpy())
        res.append(auc)
        print(f'Test AUC: {auc:.4f}')

        print('(pred) TOP 5:')
        for e, v in compare_nodes(test_data[of, ref, to], predict):
            print(ids_to_name(e, of_mapping), v)

        print('(corr) TOP 5:')
        for e, v in compare_nodes(test_data[of, ref, to], truth):
            print(ids_to_name(e, of_mapping), v)

    title = ['Connv','SAGEConv','HGTConv','GATConv']
    plot_accuracy(title,res)



if __name__ == '__main__':
    main()
