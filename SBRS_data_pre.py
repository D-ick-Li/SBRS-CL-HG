import pandas as pd
import numpy as np
max_length = 30
Filepath = 'D:/Python/pycharm/AIEx/user_log_format1.csv'


def reset_id(data, id_map, column_name='user_id'):
    mapped_id = data[column_name].map(id_map)  # map函数是把对应的列里面的元素换成map里面的参数，里面传入的是一个字典
    data.loc[:, column_name] = mapped_id
    if column_name == 'user_id':
        sid = [str(uid) + '_' + str(tid) for uid, tid in zip(data['user_id'], data['time_stamp'])]
        data.loc[:, 'session_id'] = sid      # 如果用户改变需要把session的id也改变
    return data


def train_test_validate_split(dataframe, train_percent=.7, test_percent=.2, seed=None):
    np.random.seed(seed)
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

    return train, test, validate


def process_data(filepath):
    df = pd.read_csv(filepath)
    # print(df.shape[0])
    df = df.dropna(axis=0, how='any')  # 过滤空值
    # print(df.shape[0])
    max_timestamp = df['time_stamp'].max()
    # print(max_timestamp)
    df = df[(df['time_stamp'] < max_timestamp) & (df['time_stamp'] > max_timestamp - 30)]  # 筛选最近一个月的数据
    # print(df.shape[0])
    byitem = df.groupby('item_id').aggregate(np.count_nonzero)
    nitems = byitem[byitem.user_id > 4].index
    df = df[df['item_id'].isin(nitems)]  # 删除出现次数小于5的item
    # print(df['item_id'].value_counts())
    sessionsid = [str(uid) + '_' + str(tid) for uid, tid in zip(df['user_id'], df['time_stamp'])]   # 生成session id
    df['session_id'] = sessionsid
    df = df[df['session_id'].groupby(df['session_id']).transform('size') > 1]
    df = df[df['session_id'].groupby(df['session_id']).transform('size') < max_length]

    df = df.reset_index(drop=True)

    train_df, test_df, validate_df = train_test_validate_split(df)

    # 保证测试集、验证机的item在训练集都出现过

    test_df = test_df[test_df['item_id'].isin(train_df['item_id'].unique())]
    validate_df = validate_df[validate_df['item_id'].isin(train_df['item_id'].unique())]

    total_df = pd.concat([train_df, test_df, validate_df])

    user_map = dict(zip(total_df.user_id.unique(), range(total_df.user_id.nunique())))
    item_map = dict(zip(total_df.item_id.unique(), range(1, 1 + total_df.item_id.nunique())))
    seller_map = dict(zip(total_df.seller_id.unique(), range(total_df.seller_id.nunique())))
    brand_map = dict(zip(total_df.brand_id.unique(), range(total_df.brand_id.nunique())))
    cat_map = dict(zip(total_df.cat_id.unique(), range(total_df.cat_id.nunique())))

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

    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)
    validate_df = validate_df.reset_index(drop=True)

    train_df.to_csv('train.csv', sep='\t', columns=['session_id', 'user_id', 'item_id'], index=False)
    validate_df.to_csv('valid.csv', sep='\t', columns=['session_id', 'user_id', 'item_id'], index=False)
    test_df.to_csv('test.csv', sep='\t', columns=['session_id', 'user_id', 'item_id'], index=False)
    total_df.to_csv('item_category.csv', sep='\t', columns=['item_id', 'cat_id'], index=False)
    total_df.to_csv('item_seller.csv', sep='\t', columns=['item_id', 'seller_id'], index=False)
    total_df.to_csv('item_brand.csv', sep='\t', columns=['item_id', 'brand_id'], index=False)

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


process_data(Filepath)
