import numpy as np
import pandas as pd
import os
import scipy.sparse as sp
import pickle as pkl


def build_adj(data_frame, shape):
    adj = np.zeros(shape=shape, dtype=np.int)
    header = data_frame.columns.tolist()
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
    path = "../Tmall/processed/"
    mode = 'test'
    # df_user = pd.read_csv(path+'/adj.csv', sep=',', dtype={0: str, 1: str})
    df_review = pd.read_csv(path+'/train.csv', sep=',', dtype={0: str, 1: int, 2: int, 3: int})
    df_cate = pd.read_csv(path+'/item_category.csv', sep=',', dtype={0: int, 1: int})
    df_seller = pd.read_csv(path+'/item_seller.csv', sep=',', dtype={0: int, 1: int})
    df_brand = pd.read_csv(path+'/item_brand.csv', sep=',', dtype={0: int, 1: int})
    if mode == 'test':
        df_valid = pd.read_csv(path+'/valid.csv', sep=',', dtype={0: str, 1: int, 2: int, 3: int})
        df_test = pd.read_csv(path+'/test.csv', sep=',', dtype={0: str, 1: int, 2: int, 3: int})
        df_review = pd.concat([df_review, df_valid, df_test])
        path = "../Tmall/processed/test"
    elif mode == 'train':       # processing_tmall里面的数据集处理得到的item，category等CSV是训练集测试集共有的，这里把测试集的一些去掉
        df_seller = df_seller.loc[df_seller['item_id'].isin(df_review.item_id.unique())]
        df_cate = df_cate.loc[df_cate['item_id'].isin(df_review.item_id.unique())]
        df_brand = df_brand.loc[df_brand['item_id'].isin(df_review.item_id.unique())]
    # num_user = df_review.user_id.nunique()
    num_business = df_review.item_id.nunique() + 1  # 这里+1要考虑在之后mask的时候把0算进去
    num_brand = df_brand.brand_id.nunique()
    num_cate = df_cate.cat_id.nunique()
    num_seller = df_seller.seller_id.nunique()

    adj_item_cate = build_adj(df_cate, (num_business, num_cate))
    adj_item_brand = build_adj(df_brand, (num_business, num_brand))
    adj_item_seller = build_adj(df_seller, (num_business, num_seller))

    mp_brand = meta_path_build(adj_item_brand)
    mp_seller = meta_path_build(adj_item_seller)
    mp_category = meta_path_build(adj_item_cate)

    sp.save_npz(path + "/meta_path/mp_brand", mp_brand)
    sp.save_npz(path + "/meta_path/mp_seller", mp_seller)
    sp.save_npz(path + "/meta_path/mp_category", mp_category)
    """
    np.save(path+"/meta_path/mp_social.npy", mp_social)
    np.save(path + "/meta_path/mp_city.npy", mp_city)
    np.save(path + "/meta_path/mp_category.npy", mp_category)
    np.save(path + "/meta_path/mp_item_user", mp_item_user)
    """
