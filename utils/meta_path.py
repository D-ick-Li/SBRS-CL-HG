import numpy as np
import pandas as pd
import os
import scipy.sparse as sp
import pickle as pkl

"""
In this package we process the meta-path of different relationships
"""


def build_adj(data_frame, shape, adj_type=None):
    """
    We build the adjacent matrix for different relations like item-item or user-user
    for the next step of matrix multiplication.
    :param data_frame: The dataframe contains the list of relations.
    :param shape: The shape of the adjacent matrix
    :param adj_type: A flag for search the column index.
    """
    adj = np.zeros(shape=shape, dtype=np.int)

    if adj_type == 'item_user':     # np数组的行表示item，列表示user
        r_id = list(map(int, data_frame.business_id.tolist()))
        c_id = list(map(int, data_frame.user_id.tolist()))
        adj[r_id, c_id] = 1
    elif adj_type == 'user_user':
        r_id = list(map(int, data_frame.follower.tolist()))
        c_id = list(map(int, data_frame.followee.tolist()))
        adj[r_id, c_id] = 1
        adj[c_id, r_id] = 1     # 这里考虑社交网络为无向图
        adj = adj + np.eye(shape[0])
    else:
        header = data_frame.columns.tolist()
        if len(header) != 2:
            raise AssertionError    # 表头元素个数不为2的时候触发异常
        r_id = list(map(int, data_frame[header[0]].tolist()))
        c_id = list(map(int, data_frame[header[1]].tolist()))
        adj[r_id, c_id] = 1
    return adj


def meta_path_build(adj1, adj2=None):
    """
    We build the meta=path matrix via matrix multiplication
    :param adj1: The matrix of different relationships like item-user, item-category
    :param adj2: additional matrix like user-user when we build social influences
    :return: meta-path matrix
    """
    adj_left = sp.csr_matrix(adj1)
    adj_right = sp.csr_matrix(adj1.T)
    if adj2 is not None:
        # we have to make sure the length of the meta-path.
        adj_m = sp.csr_matrix(adj2)
        mp_adj = adj_left.dot(adj_m.dot(adj_right))
    else:
        mp_adj = adj_left.dot(adj_right)

    mp_adj[mp_adj > 1] = 1    # 暂且对大于1的元素做一个替换，替换为1，表示相连接
    # We replace the element larger than 1 with 1 indicating the existence of connection
    return mp_adj


def process_influence(review_df, social_adj, user_num, business_num):
    # time_list = list(review_df['time_id'].unique())
    review_data = review_df.groupby('time_id')
    influence_mp_adj = {}
    df_temp = None
    for name, group in review_data:
        if df_temp is not None:
            mp_df = pd.concat([df_temp, group])
        else:
            mp_df = group
        df_temp = group
        adj_temp = build_adj(mp_df, (business_num, user_num), 'item_user')
        mp_social_temp = meta_path_build(adj_temp, social_adj)
        influence_mp_adj[name] = mp_social_temp
    return influence_mp_adj


if __name__ == '__main__':
    cwd = os.getcwd()
    print("the working path: "+cwd)
    path = "../Yelp/processed/"
    mode = 'train'
    df_user = pd.read_csv(path+'/adj.csv', sep=',', dtype={0: str, 1: str})
    df_review = pd.read_csv(path+'/train.csv', sep=',', dtype={0: str, 1: str, 2: str, 3: str, 4: str, 5: str, 6: str,
                                                               7: str})
    df_cate = pd.read_csv(path+'/category.csv', sep=',', dtype={0: str, 1: str})
    df_city = pd.read_csv(path+'/city.csv', sep=',', dtype={0: str, 1: str})
    if mode == 'test':
        df_valid = pd.read_csv(path+'/valid.csv', sep=',', dtype={0: str, 1: str, 2: str, 3: str, 4: str, 5: str,
                                                                  6: str, 7: str})
        df_test = pd.read_csv(path+'/test.csv', sep=',', dtype={0: str, 1: str, 2: str, 3: str, 4: str, 5: str, 6: str,
                                                                7: str})
        df_review = pd.concat([df_review, df_valid, df_test])
        path = "../Yelp/processed/test"
    elif mode == 'train':       # processing_Yelp里面的数据集处理得到的user，city等CSV是训练集测试集共有的，这里把测试集的一些去掉
        df_user = df_user.loc[df_user['follower'].isin(df_review.user_id.unique())]
        df_user = df_user.loc[df_user['followee'].isin(df_review.user_id.unique())]
        df_cate = df_cate.loc[df_cate['business_id'].isin(df_review.business_id.unique())]
        df_city = df_city.loc[df_city['business_id'].isin(df_review.business_id.unique())]
    num_user = df_review.user_id.nunique()
    num_business = df_review.business_id.nunique() + 1  # 这里+1要考虑在之后mask的时候把0算进去
    num_city = df_city.city.nunique()
    num_category = df_cate.category.nunique()

    adj_item_user = build_adj(df_review, (num_business, num_user), 'item_user')
    adj_social = build_adj(df_user, (num_user, num_user), 'user_user')
    adj_cate = build_adj(df_cate, (num_business, num_category))
    adj_city = build_adj(df_city, (num_business, num_city))

    mp_item_user = meta_path_build(adj_item_user)
    mp_city = meta_path_build(adj_city)
    mp_category = meta_path_build(adj_cate)
    mp_item_social = meta_path_build(adj_item_user, adj_social)
    influence_map = process_influence(df_review, adj_social, num_user, num_business)
    adj_social = sp.csr_matrix(adj_social)

    sp.save_npz(path + "/meta_path/adj_user", adj_social)
    sp.save_npz(path + "/meta_path/mp_item_user", mp_item_user)
    sp.save_npz(path + "/meta_path/mp_city", mp_city)
    sp.save_npz(path + "/meta_path/mp_category", mp_category)
    sp.save_npz(path + "/meta_path/mp_item_social", mp_item_social)
    with open(path + '/meta_path/influence.pickle', 'wb') as f:
        pkl.dump(influence_map, f)
    """
    np.save(path+"/meta_path/mp_social.npy", mp_social)
    np.save(path + "/meta_path/mp_city.npy", mp_city)
    np.save(path + "/meta_path/mp_category.npy", mp_category)
    np.save(path + "/meta_path/mp_item_user", mp_item_user)
    """
