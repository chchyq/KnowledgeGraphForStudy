import os.path as osp

import pandas as pd
import torch
from sentence_transformers import SentenceTransformer

from torch_geometric.data import HeteroData
from torch_geometric.transforms import RandomLinkSplit, ToUndirected

# url = 'https://files.grouplens.org/datasets/movielens/ml-latest-small.zip'
# root = osp.join(osp.dirname(osp.realpath(__file__)), '../../data/MovieLens')
# extract_zip(download_url(url, root), root)
# movie_path = osp.join(root, 'ml-latest-small', 'movies.csv')
# rating_path = osp.join(root, 'ml-latest-small', 'ratings.csv')


def load_node_csv(path, index_col, encoders=None, **kwargs):
    df = pd.read_csv(path, index_col=index_col, **kwargs).head(1000)
    # df['title']=df['title'].astype("string")
    # df['Course']=df['Course'].astype("string")
    # df['Lecture']=df['Lecture'].astype("string")
    df['pageRank']=df['pageRank'].astype("float64")
    # print(df.dtypes)
    mapping = {index: i for i, index in enumerate(df.index.unique())}
    # print("mapping",mapping)

    x = None
    if encoders is not None:
        xs = [encoder(df[col]) for col, encoder in encoders.items()]
        x = torch.cat(xs, dim=-1)

    return x, mapping


def load_edge_csv(path, src_index_col, src_mapping, dst_index_col, dst_mapping,
                  encoders=None, **kwargs):
    df = pd.read_csv(path, **kwargs).head(1000)
    # df['title']=df['title'].astype("string")
    # df['Course']=df['Course'].astype("string")
    # df['Lecture']=df['Lecture'].astype("string")
    df['pageRank']=df['pageRank'].astype("float64")
    # print(df.dtypes)

    src = [src_mapping[index] for index in df[src_index_col]]
    dst = [dst_mapping[index] for index in df[dst_index_col]]
    # print("src",src)
    # print("dst",dst)
    edge_index = torch.tensor([src, dst])

    print("edge_index",edge_index)
    print("edge_index.shape",edge_index.size())
    # print("encoders",encoders)
    for col, encoder in encoders.items():
        print("col",col)
        print("encoder",encoder)

    edge_attr = None
    if encoders is not None:
        edge_attrs = [encoder(df[col]) for col, encoder in encoders.items()]
        edge_attr = torch.cat(edge_attrs, dim=-1)
    # print("edge_attr",edge_attr)
    return edge_index, edge_attr


class SequenceEncoder(object):
    # The 'SequenceEncoder' encodes raw column strings into embeddings.
    def __init__(self, model_name='all-MiniLM-L6-v2', device=None):
        self.device = device
        self.model = SentenceTransformer(model_name, device=device)

    @torch.no_grad()
    def __call__(self, df):
        x = self.model.encode(df.values, show_progress_bar=True,
                              convert_to_tensor=True, device=self.device)
        return x.cpu()


class GenresEncoder(object):
    # The 'GenreEncoder' splits the raw column strings by 'sep' and converts
    # individual elements to categorical labels.
    def __init__(self, sep=','):
        self.sep = sep

    def __call__(self, df):
        genres = set(g for col in df.values for g in col.split(self.sep))
        mapping = {genre: i for i, genre in enumerate(genres)}

        x = torch.zeros(len(df), len(mapping))
        for i, col in enumerate(df.values):
            for genre in col.split(self.sep):
                x[i, mapping[genre]] = 1
        return x


class IdentityEncoder(object):
    # The 'IdentityEncoder' takes the raw column values and converts them to
    # PyTorch tensors.
    def __init__(self, dtype=None):
        self.dtype = dtype

    def __call__(self, df):
        # print("df.values",df.values)  #df.values <class 'numpy.ndarray'>
        return torch.from_numpy(df.values).view(-1, 1).to(self.dtype)

if __name__ == '__main__':
    file_path="data/all_file.csv"
    df = pd.read_csv(file_path)
    Lecture_x, Lecture_mapping = load_node_csv(file_path, index_col='Lecture')


    entity_x, entity_mapping = load_node_csv(
        file_path, index_col='entity',encoders={
        'entity': SequenceEncoder()  
    })

    edge_index, edge_label = load_edge_csv(
        file_path,
        src_index_col='Lecture',
        src_mapping=Lecture_mapping,
        dst_index_col='entity',
        dst_mapping=entity_mapping,
        encoders={'pageRank': IdentityEncoder(dtype=torch.long)},
    )

    data = HeteroData()
    data['Lecture'].num_nodes = len(Lecture_mapping)  # Users do not have any features.
    data['entity'].x = entity_x
    data['Lecture', 'pageRank', 'entity'].edge_index = edge_index
    data['Lecture', 'pageRank', 'entity'].edge_label = edge_label
    print(data)

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
    train_data, val_data, test_data = transform(data)
    
 