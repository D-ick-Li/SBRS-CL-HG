import pandas as pd
import numpy as np
import datetime as dt
import time
import math
import re


def time2stamp(cmnttime):   # Transform the string into timestamp, applied directly dataframe
    cmnttime = time.strptime(cmnttime, '%Y-%m-%d %H:%M:%S')
    stamp = int(time.mktime(cmnttime))
    return stamp


def is_none(n):
    return n != 'None'


def friends2list(friends_str):
    # 将用户CSV文件读取好友信息的str转化为list求好友
    # Return a list of the social information for each customer
    friend_list = list(filter(is_none, re.split(',', friends_str)))
    return friend_list


def friend_count(friends_str):
    # 将用户CSV文件读取好友信息的str转化为list求好友数量
    # Return the number of friends for each customer
    friend_list = re.split(',', friends_str)
    return len(friend_list)


def category2list(cate_str):
    # Return a list of categories of a business, since a business may by associated with multiple categories
    cate_list = re.split(',', cate_str)
    return cate_list


def build_df(value):
    # Build the dataframe of social information, the map function below may accelerate this procedure
    # Each unique friend of a customer will be listed separately in one row
    followee = value['followee']
    follower = [value['follower']] * len(followee)
    temp_list = list(map(lambda x, y: [x, y], follower, followee))
    df_temp = pd.DataFrame(temp_list, columns=['follower', 'followee'])
    return df_temp


def process_user(data_path):
    print("-- Start processing users @ %ss" % dt.datetime.now())
    df_user = pd.read_csv(data_path, sep=',', dtype={0: str, 1: str, 2: np.int32, 3: str, 4: str, 5: np.int32})
    friends_num = [t for t in df_user['friends'].apply(friend_count)]
    df_user['friends_num'] = friends_num
    print("-- finish reading users @ %ss" % dt.datetime.now())

    df_user['friends'] = df_user['friends'].apply(friends2list)      # 直接将用户的好友转化为list
    df_net = df_user[['user_id', 'friends']]
    df_net = df_net.rename(columns={'user_id': 'follower', 'friends': 'followee'})

    # df_net.drop_duplicates(subset=['follower', 'followee'], inplace=True)  # 去掉重复行
    print("-- finish processing social networks @ %ss" % dt.datetime.now())
    print('\tTotal user in social network:{}.\n\tTotal edges(links) in social network:{}.'.format(
        df_user.user_id.nunique(), df_user['friends_num'].sum()))
    print('\tAverage edges(links) in social network:{}.'.format(df_user['friends_num'].sum()/df_user.user_id.nunique()))
    # print('\tTotal edges(links) in social network after processing:{}.'.format(len(df_net)))
    print("-- finish processing social networks @ %ss" % dt.datetime.now())

    return df_net


def process_review(path, mode='small', duration=30):
    print("-- Start processing reviews @ %ss" % dt.datetime.now())
    df = pd.read_csv(path, sep=',', dtype={0: str, 1: str, 2: str, 3: str, 4: str})
    df['time_stamp'] = df['date'].apply(time2stamp)     # 这里将session的日期转换为时间戳处理
    time_stamp_min = time2stamp("2009-01-01 00:00:00")
    if mode == 'small':
        # The small mode is same with the parameters in DGRec model. So it can be compared with baseline models directly
        time_stamp_max = time2stamp("2010-10-15 00:00:00")
    elif mode == 'middle':
        time_stamp_max = time2stamp("2015-01-01 00:00:00")
    else:                   # large mode: the latest time record is December 13 2019
        time_stamp_max = time2stamp("2019-12-13 00:00:00")
    df = df[df['time_stamp'].between(time_stamp_min, time_stamp_max, inclusive=True)]
    time_stamp_min = df['time_stamp'].min()
    time_id = [int(math.floor((t - time_stamp_min) / (86400 * duration))) for t in df['time_stamp']]
    # 这里划分session,30天为一个session
    # Here we split the session each 30 days
    df['time_id'] = time_id
    session_id_series = [str(uid) + '_' + str(tid) for uid, tid in zip(df['user_id'], df['time_id'])]
    df['session_id'] = session_id_series
    df = df[df['business_id'].groupby(df['business_id']).transform('size') >= 5]
    # 这里去掉item出现次数小于5的 Here we filter out the business occurred less than 5 times
    df = df[df['session_id'].groupby(df['session_id']).transform('size') > 1]
    # 这里保证session的长度至少为2 Here we make sure the session length is no less than 2
    print('Statistics of user reviews:')
    print('\tNumber of total ratings: {}'.format(len(df)))
    print('\tNumber of users: {}'.format(df.user_id.nunique()))
    print('\tNumber of items: {}'.format(df.business_id.nunique()))
    print("-- finish reading reviews @ %ss" % dt.datetime.now())
    return df


def process_business(data_path):
    print("-- Start processing business @ %ss" % dt.datetime.now())
    df_business = pd.read_csv(data_path, sep=',', dtype={0: str, 1: str, 2: str, 3: str, 4: np.float32, 5: np.float32,
                                                         6: np.float32, 7: np.int32, 8: str})
    df_business = df_business[df_business.review_count > 5]

    '''
    for index, content in df_business.iterrows():
        content_cate = category2list(content['categories'])
        # content_city = content['city']
        for cate_item in content_cate:
            if cate_item not in category_list:
                category_list.append(cate_item)
    '''
    # df_category = df_business[['businesss_id', 'categories']]
    df_city_cate = df_business[['business_id', 'city', 'categories']]
    df_city_cate['categories'] = df_city_cate['categories'].apply(category2list)

    print('\tNumber of cites: {}'.format(df_city_cate.city.nunique()))
    print("-- finish reading business @ %ss" % dt.datetime.now())
    return df_city_cate


def reset_id(data, id_map, column_name):
    # We reset the ids into consecutive sequences for the simplicity of later work
    mapped_id = data[column_name].map(id_map)       # map函数是把对应的列里面的元素换成map里面的参数，里面传入的是一个字典
    data[column_name] = mapped_id
    if column_name == 'user_id':
        # If the user_id changes, we also need to change the corresponding session_id
        session_id = [str(uid) + '_' + str(tid) for uid, tid in zip(data['user_id'], data['time_id'])]
        data['session_id'] = session_id      # 如果用户改变需要把session的id也改变
    return data


"""
Inspired by DGRec https://github.com/DeepGraphLearning/RecommenderSystems/tree/master/socialRec/dgrec
We followed their work to record the historical influences by user himself as well as friends.
"""


def latest_sessions(review, path):
    user_num = review['user_id'].nunique()
    tmax = review.time_id.max()
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

    with open(path + 'latest_sessions.txt', 'w') as fout:
        for idx in range(user_num):
            fout.write(','.join(user_last_session[idx]) + '\n')


"""
This function is designed to secure the data integrity after filtering different dataframes 
"""


def filter_data(df_review, df_net, df_business, max_length=30):
    print("-- Start filtering data @ %ss" % dt.datetime.now())
    # split the dataset
    tmax = df_review.time_id.max()
    session_max_times = df_review.groupby('session_id').time_id.max()       # 得到一个序列，代表着每个session的time_id,其索引为session_id
    session_train = session_max_times[session_max_times < tmax - 4].index      # 假设取最后的4个月为测试集
    session_holdout = session_max_times[session_max_times >= tmax - 4].index
    train_data = df_review[df_review['session_id'].isin(session_train)]
    holdout_data = df_review[df_review['session_id'].isin(session_holdout)]
    holdout_data = holdout_data.loc[holdout_data['business_id'].isin(train_data['business_id'].unique())]
    # 过滤保证测试集的user在训练集出现过， 如果不过滤的话再考虑冷启动
    # Make sure the business in valid/test set exists in the training set

    # Split the valid and test set equally
    holdout_cn = holdout_data.session_id.nunique()
    holdout_ids = holdout_data.session_id.unique()
    np.random.shuffle(holdout_ids)
    valid_cn = int(holdout_cn * 0.5)
    session_valid = holdout_ids[0: valid_cn]
    session_test = holdout_ids[valid_cn:]
    valid = holdout_data[holdout_data['session_id'].isin(session_valid)]       # 验证集
    test = holdout_data[holdout_data['session_id'].isin(session_test)]          # 测试集

    df_cat = pd.concat([train_data, valid, test])    # 重新拼接
    # We concatenate the train/valid/test again for further filtering

    """
    The following part for filtering could be a little sophisticated
    You may draw a Venn diagram and searching the intersection of different parts for your better understanding
    """
    df_cat = df_cat[df_cat['session_id'].groupby(df_cat['session_id']).transform('size') > 1]
    df_cat = df_cat[df_cat['session_id'].groupby(df_cat['session_id']).transform('size') <= max_length]
    # Filter again to secure the session length

    df_net = df_net.loc[df_net['follower'].isin(df_cat['user_id'].unique())]
    df_cat = df_cat.loc[df_cat['user_id'].isin(df_net.follower.unique())]
    # 保证社交网络中的用户有session
    # Keep the consistency between reviewing dataframe and social dataframe

    df_business = df_business.loc[df_business['business_id'].isin(df_cat['business_id'].unique())]
    df_cat = df_cat.loc[df_cat['business_id'].isin(df_business.business_id.unique())]
    # 保证review和商家信息一致
    # Keep the consistency between reviewing dataframe and business dataframe

    df_cat = df_cat[df_cat['session_id'].groupby(df_cat['session_id']).transform('size') > 1]

    train_data = df_cat[df_cat['session_id'].isin(session_train)]
    valid = df_cat[df_cat['session_id'].isin(session_valid)]  # 验证集
    test = df_cat[df_cat['session_id'].isin(session_test)]
    valid = valid.loc[valid['business_id'].isin(train_data['business_id'].unique())]
    test = test.loc[test['business_id'].isin(train_data['business_id'].unique())]
    # We have to make sure the item existence in both training set and test/valid set again

    df_cat = pd.concat([train_data, valid, test])
    df_cat = df_cat[df_cat['session_id'].groupby(df_cat['session_id']).transform('size') > 1]

    train_data = df_cat[df_cat['session_id'].isin(session_train)]
    valid = df_cat[df_cat['session_id'].isin(session_valid)]  # 验证集
    test = df_cat[df_cat['session_id'].isin(session_test)]

    df_net = df_net.loc[df_net['follower'].isin(df_cat['user_id'].unique())]
    df_business = df_business[df_business['business_id'].isin(df_cat['business_id'].unique())]
    df_net.reset_index(drop=True, inplace=True)

    print('Number of users after the first filtering in df_net: {}'.format(df_net.follower.nunique()))
    print('Number of users after the first filtering in df_review: {}'.format(df_cat.user_id.nunique()))

    df_net_filter = pd.DataFrame(columns=['follower', 'followee'])
    df_category = pd.DataFrame(columns=['business_id', 'category'])

    for index, net_row in df_net.iterrows():
        print(index)
        df_net_filter = df_net_filter.append(build_df(net_row))
    # Build a new dataframe to process the friend information separately

    for index, busi_row in df_business.iterrows():
        cate_list = busi_row['categories']
        business_id = busi_row['business_id']
        # print(index)
        for a_category in cate_list:
            df_category = df_category.append([{'business_id': business_id, 'category': a_category}], ignore_index=True)
    # Build a new dataframe to process the category information separately

    df_net_filter = df_net_filter.loc[df_net_filter['followee'].isin(df_cat['user_id'].unique())]  # 过滤好友网络中的好友
    # df_net_filter = df_net_filter.loc[df_net_filter['followee'].isin(df_net_filter['follower'].unique())]
    df_category = df_category[df_category['business_id'].isin(df_cat['business_id'].unique())]
    df_business = df_business[['business_id', 'city']]
    # df1.reset_index(drop=True, inplace=True)
    df_business.reset_index(drop=True, inplace=True)
    df_category.reset_index(drop=True, inplace=True)

    user_map = dict(zip(df_cat.user_id.unique(), range(df_cat.user_id.nunique())))
    # item_map = dict(zip(df_review_cat.business_id.unique(), range(1, 1 + df_review_cat.business_id.nunique())))
    item_map = dict(zip(df_cat.business_id.unique(), range(1, df_cat.business_id.nunique() + 1)))
    # The business_id starts from 1, while 0 could be used for padding during training
    city_map = dict(zip(df_business.city.unique(), range(df_business.city.nunique())))
    category_map = dict(zip(df_category.category.unique(), range(df_category.category.nunique())))
    print('Number of users after filtering in df_net: {}'.format(df_net_filter.follower.nunique()))
    print('Number of users after filtering in df_review: {}'.format(df_cat.user_id.nunique()))
    print('Number of train/test: {}/{}'.format(len(train_data), len(holdout_data)))
    print('Number of business after filtering in df_review: {}'.format(df_cat.business_id.nunique()))
    print('Number of business after filtering in df_business: {}'.format(df_business.business_id.nunique()))
    print('Number of business after filtering in df_category: {}'.format(df_category.business_id.nunique()))
    with open('Yelp/processed/user_id_map.csv', 'w') as fout:  # 将原有的id和新的id映射写入文件
        for k, v in user_map.items():
            fout.write(str(k) + ',' + str(v) + '\n')
    with open('Yelp/processed/business_id_map.csv', 'w') as fout:
        for k, v in item_map.items():
            fout.write(str(k) + ',' + str(v) + '\n')
    with open('Yelp/processed/city_id_map.csv', 'w') as fout:
        for k, v in city_map.items():
            fout.write(str(k) + ',' + str(v) + '\n')
    with open('Yelp/processed/category_id_map.csv', 'w') as fout:
        for k, v in category_map.items():
            fout.write(str(k) + ',' + str(v) + '\n')

    reset_id(df_cat, user_map, 'user_id')
    reset_id(train_data, user_map, 'user_id')
    reset_id(valid, user_map, 'user_id')
    reset_id(test, user_map, 'user_id')
    reset_id(df_net_filter, user_map, 'follower')
    reset_id(df_net_filter, user_map, 'followee')
    reset_id(df_cat, item_map, 'business_id')
    reset_id(train_data, item_map, 'business_id')
    reset_id(valid, item_map, 'business_id')
    reset_id(test, item_map, 'business_id')
    reset_id(df_business, city_map, 'city')
    reset_id(df_business, item_map, 'business_id')
    reset_id(df_category, item_map, 'business_id')
    reset_id(df_category, category_map, 'category')

    latest_sessions(df_cat, 'Yelp/processed/')

    train_data.to_csv('Yelp/processed/train.csv', sep=',', index=False)
    valid.to_csv('Yelp/processed/valid.csv', sep=',', index=False)
    test.to_csv('Yelp/processed/test.csv', sep=',', index=False)
    df_net_filter.to_csv('Yelp/processed/adj.csv', sep=',', index=False)
    df_business.to_csv('Yelp/processed/city.csv', sep=',', index=False)
    df_category.to_csv('Yelp/processed/category.csv', sep=',', index=False)

    print('Train set\nEvents: {}\nSessions: {}\nItems: {}\nAvg length: {}'.format(len(train_data),
                                                                                  train_data.session_id.nunique(),
                                                                                  train_data.business_id.nunique(),
                                                                                  train_data.groupby(
                                                                                  'session_id').size().mean()))
    print('Valid set\nEvents: {}\nSessions: {}\nItems: {}\nAvg length: {}'.format(len(valid),
                                                                                  valid.session_id.nunique(),
                                                                                  valid.business_id.nunique(),
                                                                                  valid.groupby(
                                                                                  'session_id').size().mean()))
    print('Test set\nEvents: {}\nSessions: {}\nItems: {}\nAvg length: {}'.format(len(test),
                                                                                 test.session_id.nunique(),
                                                                                 test.business_id.nunique(),
                                                                                 test.groupby(
                                                                                 'session_id').size().mean()))
    print("The number of user-user:{}".format(df_net_filter.shape))
    print("The number of item-city:{}".format(df_business.shape))
    print("The number of item-category:{}".format(df_category.shape))


if __name__ == '__main__':
    review_path = 'Yelp/review.csv'
    user_path = 'Yelp/user.csv'
    business_path = 'Yelp/business.csv'

    review_df = process_review(review_path, mode='small')
    net_df = process_user(user_path)
    business_df = process_business(business_path)
    filter_data(review_df, net_df, business_df)
