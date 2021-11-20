import numpy as np
import pandas as pd
import os
import scipy.sparse as sp
import pickle as pkl


def build_adj(data_frame, shape, name_reverse=False):
    adj = np.zeros(shape=shape, dtype=np.int)
    header = data_frame.columns.tolist()
    if name_reverse:
        r_id = list(map(int, data_frame[header[1]].tolist()))
        c_id = list(map(int, data_frame[header[0]].tolist()))
    else:
        r_id = list(map(int, data_frame[header[0]].tolist()))
        c_id = list(map(int, data_frame[header[1]].tolist()))
    adj[r_id, c_id] = 1
    return adj


def meta_path_build(adj1, adj2=None):
    adj_left = sp.csr_matrix(adj1)
    adj_right = sp.csr_matrix(adj1.T)
    if adj2 is not None:
        adj_m = sp.csr_matrix(adj2)
        # mp_adj = np.matmul(adj1, np.matmul(adj2, adj1.T))
        mp_adj = adj_left.dot(adj_m.dot(adj_right))
    else:
        # mp_adj = np.matmul(adj1, adj1.T)
        mp_adj = adj_left.dot(adj_right)

    mp_adj[mp_adj > 1] = 1    # 暂且对大于1的元素做一个替换，替换为1，表示相连接
    return mp_adj


if __name__ == '__main__':
    cwd = os.getcwd()
    print("the working path: "+cwd)
    path = "../Nowplaying/processed/"
    mode = 'train'
    # df_user = pd.read_csv(path+'/adj.csv', sep=',', dtype={0: str, 1: str})
    df_review = pd.read_csv(path+'/train.csv', sep=',', dtype={0: int, 1: int, 2: int, 3: int, 4: str})
    # df_hashtag = pd.read_csv(path + '/hashtag.csv', sep=',', dtype={0: int, 1: int})
    df_hashtag = pd.read_csv(path + '/hashtag.csv', sep=',', dtype={0: int, 1: int})
    # df_hashtag['hashtag'] = df_hashtag['hashtag'].astype("int")
    df_context = pd.read_csv(path + '/context.csv', sep=',', dtype={0: int, 1: int})
    # df_context['class_id'] = df_context['class_id'].astype("int")
    df_artist = pd.read_csv(path + '/artist.csv', sep=',', dtype={0: int, 1: int})
    # df_artist['artist_id'] = df_artist['artist_id'].astype("int")
    if mode == 'test':
        df_valid = pd.read_csv(path+'/valid.csv', sep=',', dtype={0: int, 1: int, 2: int, 3: int, 4: str})
        df_test = pd.read_csv(path+'/test.csv', sep=',', dtype={0: int, 1: int, 2: int, 3: int, 4: str})
        df_review = pd.concat([df_review, df_valid, df_test])
        path = "../Nowplaying/processed/test"
    elif mode == 'train':       # processing_tmall里面的数据集处理得到的item，category等CSV是训练集测试集共有的，这里把测试集的一些去掉
        df_context = df_context.loc[df_context['track_id'].isin(df_review.track_id.unique())]
        df_hashtag = df_hashtag.loc[df_hashtag['track_id'].isin(df_review.track_id.unique())]
        df_artist = df_artist.loc[df_artist['track_id'].isin(df_review.track_id.unique())]
    # num_user = df_review.user_id.nunique()
    num_business = df_review.track_id.nunique() + 1  # 这里+1要考虑在之后mask的时候把0算进去
    num_artist = df_artist.artist_id.nunique()
    num_hashtag = df_hashtag.hashtag.nunique()
    # num_context = df_context.class_id.nunique()

    adj_item_artist = build_adj(df_artist, (num_business, num_artist), True)
    adj_item_hashtag = build_adj(df_hashtag, (num_business, num_hashtag))
    # adj_item_context = build_adj(df_context, (num_business, num_context), True)
    adj_item_context = build_adj(df_context, (num_business, num_business), True)

    mp_hashtag = meta_path_build(adj_item_hashtag)
    # mp_context = meta_path_build(adj_item_context)
    mp_artist = meta_path_build(adj_item_artist)
    mp_context = sp.csr_matrix(adj_item_context)

    sp.save_npz(path + "/meta_path/mp_artist", mp_artist)
    sp.save_npz(path + "/meta_path/mp_hashtag", mp_hashtag)
    sp.save_npz(path + "/meta_path/mp_context", mp_context)
