import numpy as np
import pickle as pkl
import scipy.sparse as sp
import torch
import torch.nn as nn
import random
from sklearn import metrics
import pandas as pd


def shuffle_nodes(items, masks):
    node_num = masks.sum(axis=1)
    batch_size = masks.shape[0]
    node_max = masks.shape[1]
    shuffle_item = items.copy()
    for i in range(batch_size):
        shuf_idx = np.append(np.random.permutation(node_num[i]), np.ones(node_max - node_num[i]) * (node_max - 1))
        idx = shuf_idx.astype(np.int32)
        shuffle_item[i] = shuffle_item[i, idx]
    return shuffle_item

###############################################
# This section of code adapted from tkipf/gcn #
###############################################

def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)


def macro_f1(preds, labels):
    labels = labels.to(torch.device("cpu")).numpy()
    preds = preds.to(torch.device("cpu")).numpy()
    macro = metrics.f1_score(labels, preds, average='macro')
    return macro


def sparse_to_tuple(sparse_mx, insert_batch=False):
    """Convert sparse matrix to tuple representation."""
    """Set insert_batch=True if you want to insert a batch dimension."""

    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        if insert_batch:
            coords = np.vstack((np.zeros(mx.row.shape[0]), mx.row, mx.col)).transpose()
            values = mx.data
            shape = (1,) + mx.shape
        else:
            coords = np.vstack((mx.row, mx.col)).transpose()
            values = mx.data
            shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx


def standardize_data(f, train_mask):
    """Standardize feature matrix and convert to tuple representation"""
    # standardize data
    f = f.todense()
    mu = f[train_mask == True, :].mean(axis=0)
    sigma = f[train_mask == True, :].std(axis=0)
    f = f[:, np.squeeze(np.array(sigma > 0))]
    mu = f[train_mask == True, :].mean(axis=0)
    sigma = f[train_mask == True, :].std(axis=0)
    f = (f - mu) / sigma
    return f


def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return features.todense(), sparse_to_tuple(features)


def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def preprocess_adj(adj):
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
    adj_normalized = normalize_adj(adj + sp.eye(adj.shape[0]))
    return sparse_to_tuple(adj_normalized)


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

###############################################
# This section of code adapted from DGRec #
###############################################


def load_latest_session(data_path):
    ret = []
    for line in open(data_path + '/latest_sessions.txt'):
        chunks = line.strip().split(',')
        ret.append(chunks)
    return ret


def load_map(data_path):
    id_map = {}
    for line in open(data_path):
        k, v = line.strip().split(',')
        id_map[k] = str(v)

    map_num = len(id_map)
    return map_num


def load_adj(data_path, dataset="Yelp"):
    if dataset == "Yelp":
        adj_social = sp.load_npz(data_path + "/meta_path/adj_user.npz")
        mp_city = sp.load_npz(data_path + "/meta_path/mp_city.npz")
        mp_category = sp.load_npz(data_path + "/meta_path/mp_category.npz")
        # return [mp_iui, mp_social_dict, mp_category, mp_city]
        adj_social = adj_social.tolil()
        return [adj_social, mp_category.tolil(), mp_city.tolil()]
    elif dataset == "Tmall":
        mp_brand = sp.load_npz(data_path + "/meta_path/mp_brand.npz")
        mp_seller = sp.load_npz(data_path + "/meta_path/mp_seller.npz")
        mp_cate = sp.load_npz(data_path + "/meta_path/mp_category.npz")
        return [mp_seller.tolil(), mp_brand.tolil(), mp_cate.tolil()]
    elif dataset == "Nowplaying":
        mp_artist = sp.load_npz(data_path + "/meta_path/mp_artist.npz")
        mp_hashtag = sp.load_npz(data_path + "/meta_path/mp_hashtag.npz")
        mp_context = sp.load_npz(data_path + "/meta_path/mp_context.npz")
        return [mp_artist.tolil(), mp_hashtag.tolil(), mp_context.tolil()]


def load_data(path="./Yelp/processed/", dataset="Yelp"):
    latest_sessions = load_latest_session(path)
    mp_adj_list = load_adj(path, dataset)
    mp_test_adj = load_adj(path + '/test', dataset)
    # mp_adj_list = mp_test_adj
    if dataset == "Yelp":
        business_file = path + '/business_id_map.csv'
        user_file = path + '/user_id_map.csv'
        city_file = path + '/city_id_map.csv'
        category_file = path + '/category_id_map.csv'

        business_num = load_map(data_path=business_file)
        user_num = load_map(data_path=user_file)
        city_num = load_map(data_path=city_file)
        category_num = load_map(data_path=category_file)

        num_list = [business_num, user_num, city_num, category_num]

        train = pd.read_csv(path + '/train.csv', sep=',', dtype={0: str, 1: str, 2: str, 3: str, 4: str, 5: str, 6: str,
                                                                 7: str})
        valid = pd.read_csv(path + '/valid.csv', sep=',', dtype={0: str, 1: str, 2: str, 3: str, 4: str, 5: str,
                                                                 6: str, 7: str})
        test = pd.read_csv(path + '/test.csv', sep=',', dtype={0: str, 1: str, 2: str, 3: str, 4: str, 5: str, 6: str,
                                                               7: str})

    # return adjs, features, labels, idx_train, idx_val, idx_test
    elif dataset == "Tmall":

        train = pd.read_csv(path + '/train.csv', sep=',', dtype={0: str, 1: int, 2: int, 3: int})
        valid = pd.read_csv(path + '/valid.csv', sep=',', dtype={0: str, 1: int, 2: int, 3: int})
        test = pd.read_csv(path + '/test.csv', sep=',', dtype={0: str, 1: int, 2: int, 3: int})
        df_concat = pd.concat([train, valid, test])
        brand_df = pd.read_csv(path + 'item_brand.csv', sep=',', dtype={0: int, 1: int})
        seller_df = pd.read_csv(path + 'item_seller.csv', sep=',', dtype={0: int, 1: int})
        cate_df = pd.read_csv(path + 'item_category.csv', sep=',', dtype={0: int, 1: int})
        business_num = df_concat['item_id'].nunique()
        seller_num = seller_df['seller_id'].nunique()
        brand_num = brand_df['brand_id'].nunique()
        category_num = cate_df['cat_id'].nunique()
        num_list = [business_num, seller_num, brand_num, category_num]

    elif dataset == 'Nowplaying':
        train = pd.read_csv(path + '/train.csv', sep=',', dtype={0: int, 1: int, 2: int, 3: str})
        valid = pd.read_csv(path + '/valid.csv', sep=',', dtype={0: int, 1: int, 2: int, 3: str})
        test = pd.read_csv(path + '/test.csv', sep=',', dtype={0: int, 1: int, 2: int, 3: str})
        df_concat = pd.concat([train, valid, test])
        artist_df = pd.read_csv(path + '/artist.csv', sep=',', dtype={0: int, 1: int})
        hashtag_df = pd.read_csv(path + '/hashtag.csv', sep=',', dtype={0: int, 1: int})
        context_df = pd.read_csv(path + '/context.csv', sep=',', dtype={0: int, 1: int})
        business_num = df_concat['track_id'].nunique()
        artist_num = artist_df['artist_id'].nunique()
        hashtag_num = hashtag_df['hashtag'].nunique()
        # context_num = context_df['class_id'].nunique()
        context_num = context_df['track_id'].nunique()
        num_list = [business_num, artist_num, hashtag_num, context_num]
    else:
        raise ValueError('dataset not find error')
    return mp_adj_list, mp_test_adj, latest_sessions, num_list, train, valid, test
