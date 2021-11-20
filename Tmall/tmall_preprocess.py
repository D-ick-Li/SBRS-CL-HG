import pandas as pd
import numpy as np
max_length = 30
date_length = 15
Filepath = '../Tmall/user_log_format1.csv'


def reset_id(data, id_map, column_name='user_id'):
    mapped_id = data[column_name].map(id_map)  # map函数是把对应的列里面的元素换成map里面的参数，里面传入的是一个字典
    data.loc[:, column_name] = mapped_id
    if column_name == 'user_id':
        sid = [str(uid) + '_' + str(tid) for uid, tid in zip(data['user_id'], data['time_stamp'])]
        data.loc[:, 'session_id'] = sid      # 如果用户改变需要把session的id也改变
    return data


def latest_sessions(review, path):
    user_num = review['user_id'].nunique()
    tmax = review.time_stamp.max()
    user2sessions = review.groupby('user_id')['session_id'].apply(set).to_dict()
    user_last_session = []
    for idx in range(user_num):
        sessions = user2sessions[idx]
        latest = []
        for t in range(tmax + 1):
            if t == 0:
                latest.append('NULL')
            else:
                sess_id_tmp = str(idx) + '_' + str(t - 1)       # 考虑在不同的time_id时，存储用户的历史记录情况，用于比较不同用户之间的相互影响
                if sess_id_tmp in sessions:
                    latest.append(sess_id_tmp)
                else:
                    latest.append(latest[t - 1])
        user_last_session.append(latest)

    with open(path + 'latest_sessions1.txt', 'w') as fout:
        for idx in range(user_num):
            fout.write(','.join(user_last_session[idx]) + '\n')


def padding_sessions(raw_data):
    """
    transform the dataframe to the session list
    """
    raw_data = raw_data.sort_values(by=['time_stamp']).groupby('session_id')['item_id'].apply(
                                list).to_dict()  # 同一个session内按照时间排序,，最后得到一个根据session_id的dict
    new_data = {}
    total_len = 0
    for k, v in raw_data.items():
        sess = list(map(int, v))    # 这里要把session节点的str类型变为int
        # out_seqs = []
        # labs = []
        for i in range(1, len(sess)):
            tar = sess[-i]
            # labs += [tar]
            # out_seqs += [sess[:-i]]
            key_new = k + '_' + str(i)
            new_data[key_new] = [sess[:-i], [tar]]
            total_len += len(new_data[key_new][0]) - 1

    return new_data, total_len


def train_test_validate_split(dataframe, train_percent=0.7, test_percent=0.5):
    """
    m = dataframe['session_id'].nunique()
    uni = dataframe['session_id'].unique()
    np.random.shuffle(uni)
    train_end = int(train_percent * m)
    test_end = int(test_percent * m) + train_end
    train_l = uni[: train_end]
    test_l = uni[train_end: test_end]
    valid_l = uni[test_end:]
    train = dataframe[dataframe['session_id'].isin(train_l)]
    test = dataframe[dataframe['session_id'].isin(test_l)]
    validate = dataframe[dataframe['session_id'].isin(valid_l)]
    """

    max_timestamp = dataframe['time_stamp'].max()
    df_holdout = dataframe[dataframe['time_stamp'] >= max_timestamp-1]
    train = dataframe[dataframe['time_stamp'] < max_timestamp-1]
    train = train[train['item_id'].groupby(train['item_id']).transform('size') >= 5]
    train = train[train['session_id'].groupby(train['session_id']).transform('size') > 1]
    df_holdout = df_holdout[df_holdout['item_id'].isin(train['item_id'].unique())]
    df_holdout = df_holdout[df_holdout['session_id'].groupby(df_holdout['session_id']).transform('size') > 1]

    m = df_holdout['session_id'].nunique()
    uni = df_holdout['session_id'].unique()
    np.random.shuffle(uni)
    test_range = int(test_percent * m)
    test_l = uni[: test_range]
    valid_l = uni[test_range:]
    test = df_holdout[df_holdout['session_id'].isin(test_l)]
    validate = df_holdout[df_holdout['session_id'].isin(valid_l)]

    # 保证测试集、验证机的item在训练集都出现过
    """
    test = test[test['item_id'].isin(train['item_id'].unique())]
    validate = validate[validate['item_id'].isin(train['item_id'].unique())]

    total_df = pd.concat([train, test, validate])
    # total_df = total_df[total_df['item_id'].groupby(total_df['item_id']).transform('size') >= 5]
    total_df = total_df[total_df['session_id'].groupby(total_df['session_id']).transform('size') > 1]
    total_df = total_df[total_df['session_id'].groupby(total_df['session_id']).transform('size') <= max_length]

    train = total_df[total_df['session_id'].isin(train_l)]
    validate = total_df[total_df['session_id'].isin(valid_l)]  # 验证集
    test = total_df[total_df['session_id'].isin(test_l)]
    """
    return train, test, validate


def process_data(filepath):
    df = pd.read_csv(filepath)
    # print(df.shape[0])
    df = df.dropna(axis=0, how='any')  # 过滤空值
    # print(df.shape[0])
    # max_timestamp = df['time_stamp'].max()
    # print(max_timestamp)
    # df = df[(df['time_stamp'] <= max_timestamp) & (df['time_stamp'] > (max_timestamp - 4))]  # 筛选最近7 day的数据
    df = df[(df['time_stamp'] <= 1111) & (df['time_stamp'] > 1106)]
    df = df[~(df['time_stamp'] == 1111)]  # 去除11.11
    df = df[~(df['time_stamp'] == 1110)]
    df = df[(df['action_type'] == 0) | (df['action_type'] == 2)]

    df = df[df['item_id'].groupby(df['item_id']).transform('size') >= 5]

    sessionsid = [str(uid) + '_' + str(tid) for uid, tid in zip(df['user_id'], df['time_stamp'])]   # 生成session id
    df['session_id'] = sessionsid
    df = df[df['session_id'].groupby(df['session_id']).transform('size') > 1]
    df = df[df['session_id'].groupby(df['session_id']).transform('size') < max_length]

    train_df, test_df, validate_df = train_test_validate_split(df)

    total_df = pd.concat([train_df, test_df, validate_df])

    user_map = dict(zip(total_df.user_id.unique(), range(total_df.user_id.nunique())))
    item_map = dict(zip(total_df.item_id.unique(), range(1, 1 + total_df.item_id.nunique())))
    seller_map = dict(zip(total_df.seller_id.unique(), range(total_df.seller_id.nunique())))
    brand_map = dict(zip(total_df.brand_id.unique(), range(total_df.brand_id.nunique())))
    cat_map = dict(zip(total_df.cat_id.unique(), range(total_df.cat_id.nunique())))

    new_seq, total_len = padding_sessions(total_df)
    print('The session_num: {}'.format(len(new_seq)))
    print('The average length: {}'.format(total_len/len(new_seq)))

    reset_id(total_df, user_map)
    reset_id(total_df, item_map, 'item_id')
    reset_id(total_df, cat_map, 'cat_id')
    reset_id(total_df, seller_map, 'seller_id')
    reset_id(total_df, brand_map, 'brand_id')
    reset_id(train_df, user_map)
    reset_id(train_df, item_map, 'item_id')
    reset_id(train_df, cat_map, 'cat_id')
    reset_id(train_df, seller_map, 'seller_id')
    reset_id(train_df, brand_map, 'brand_id')

    reset_id(test_df, user_map)
    reset_id(test_df, item_map, 'item_id')
    reset_id(test_df, cat_map, 'cat_id')
    reset_id(test_df, seller_map, 'seller_id')
    reset_id(test_df, brand_map, 'brand_id')
    reset_id(validate_df, user_map)
    reset_id(validate_df, item_map, 'item_id')
    reset_id(validate_df, cat_map, 'cat_id')
    reset_id(validate_df, seller_map, 'seller_id')
    reset_id(validate_df, brand_map, 'brand_id')

    latest_sessions(total_df, 'processed/')
    train_df.to_csv('processed/train.csv', sep=',', columns=['session_id', 'user_id', 'item_id', 'time_stamp'], index=False)
    validate_df.to_csv('processed/valid.csv', sep=',', columns=['session_id', 'user_id', 'item_id', 'time_stamp'], index=False)
    test_df.to_csv('processed/test.csv', sep=',', columns=['session_id', 'user_id', 'item_id', 'time_stamp'], index=False)
    total_df.to_csv('processed/item_category.csv', sep=',', columns=['item_id', 'cat_id'], index=False)
    total_df.to_csv('processed/item_seller.csv', sep=',', columns=['item_id', 'seller_id'], index=False)
    total_df.to_csv('processed/item_brand.csv', sep=',', columns=['item_id', 'brand_id'], index=False)

    print('Train set\n\tSessions: {}\n\tItems: {}\n\tCategory: {}\n\tSeller: {}\n\tBrand: {}\n\tAvg length: {}'.format(
                                                                                          train_df.session_id.nunique(),
                                                                                          train_df.item_id.nunique(),
                                                                                          train_df.cat_id.nunique(),
                                                                                          train_df.seller_id.nunique(),
                                                                                          train_df.brand_id.nunique(),
                                                                                          train_df.groupby(
                                                                                           'session_id').size().mean()))
    print('Test set\n\tSessions: {}\n\tItems: {}\n\tCategory: {}\n\tSeller: {}\n\tBrand: {}\n\tAvg length: {}'.format(
                                                                                          test_df.session_id.nunique(),
                                                                                          test_df.item_id.nunique(),
                                                                                          test_df.cat_id.nunique(),
                                                                                          test_df.seller_id.nunique(),
                                                                                          test_df.brand_id.nunique(),
                                                                                          test_df.groupby(
                                                                                           'session_id').size().mean()))
    print('Validate set\n\tSessions: {}\n\tItems: {}\n\tCategory: {}\n\tSeller: {}\n\tBrand: {}\n\tAvg length: {}'.format(
                                                                                          validate_df.session_id.nunique(),
                                                                                          validate_df.item_id.nunique(),
                                                                                          validate_df.cat_id.nunique(),
                                                                                          validate_df.seller_id.nunique(),
                                                                                          validate_df.brand_id.nunique(),
                                                                                          validate_df.groupby(
                                                                                           'session_id').size().mean()))
    print('Total set\n\tSessions: {}\n\tItems: {}\n\tCategory: {}\n\tSeller: {}\n\tBrand: {}\n\tAvg length: {}'.format(
                                                                                          total_df.session_id.nunique(),
                                                                                          total_df.item_id.nunique(),
                                                                                          total_df.cat_id.nunique(),
                                                                                          total_df.seller_id.nunique(),
                                                                                          total_df.brand_id.nunique(),
                                                                                          total_df.groupby(
                                                                                           'session_id').size().mean()))
    print("the largest item_id:{}".format(total_df.item_id.max()))
    print("the largest category_id:{}".format(total_df.cat_id.max()))
    print("the largest brand_id:{}".format(total_df.brand_id.max()))
    print("the largest seller_id:{}".format(total_df.seller_id.max()))

process_data(Filepath)
