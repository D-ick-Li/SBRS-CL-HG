# coding=utf-8
from __future__ import division
from __future__ import print_function

import time
import numpy as np
from utils import process
from utils.sampling import RandomWalkRestart, UniformNeighborSampler, construct_social_influence, self_sample


"""
Inspired by DGRec https://github.com/DeepGraphLearning/RecommenderSystems/tree/master/socialRec/dgrec
We generate the batches of heterogeneous information iteratively
"""


class MinibatchIterator(object):

    def __init__(self,
                 adjs,  # adj[0] in lil matrix, adj[1], adj[2] in csr sparse matrix
                 adjs_test,
                 latest_sessions,
                 data_list,  # DataFrame list, either [train, valid] or [train, valid, test].
                 batch_size,
                 max_degree,
                 num_nodes,
                 max_length=150,
                 samples_max=5,
                 state_flag='train'):
        self.adjs = adjs
        self.adjs_test = adjs_test
        self.latest_sessions = latest_sessions
        self.state_flag = state_flag
        self.train_df, self.valid_df, self.test_df = data_list
        self.batch_size = batch_size
        self.max_degree = max_degree
        self.num_nodes = num_nodes
        self.max_length = max_length
        self.samples_max = samples_max
        self.visible_time = self.user_visible_time()
        self.train_sampler = UniformNeighborSampler(self.adjs[0], self.visible_time, self.samples_max)
        self.test_sampler = UniformNeighborSampler(self.adjs_test[0], self.visible_time, self.samples_max)
        self.adj_list = None
        self.seq_train = self.padding_sessions(self.train_df)
        self.seq_val = self.padding_sessions(self.valid_df)
        self.seq_test = self.padding_sessions(self.test_df)
        self.seq_all = self.seq_train.copy()
        self.seq_all.update(self.seq_val)
        self.seq_all.update(self.seq_test)

        self.train_keys = list(self.seq_train.keys())
        self.valid_keys = list(self.seq_val.keys())
        self.test_keys = list(self.seq_test.keys())
        self.key_pretrain = self.pretrain_keys()

        self.batch_num = 0
        self.batch_num_val = 0
        self.batch_num_test = 0
        self.sampling = RandomWalkRestart(self.adjs)
        self.sampling_test = RandomWalkRestart(self.adjs_test)

    def user_visible_time(self):
        """
            Find out when each user is 'visible' to her friends, i.e., every user's first click/watching time.
            确认相互影响的最早时间
        """
        visible_time = []
        for l in self.latest_sessions:  # 第一层循环是每个用户
            timeid = max(loc for loc, val in enumerate(l) if val == 'NULL') + 1
            # 出现记录的最早时间， 整个列表长度为用户数量
            visible_time.append(timeid)
            assert 0 < timeid <= len(l), 'Wrong when create visible time {}'.format(timeid)
        return visible_time

    @staticmethod
    def padding_sessions(raw_data):
        """
        transform the dataframe to the session list
        """
        raw_data = raw_data.sort_values(by=['time_stamp']).groupby('session_id')['business_id'].apply(
                                list).to_dict()
        # 同一个session内按照时间排序,，最后得到一个根据session_id的dict
        new_data = {}
        for k, v in raw_data.items():
            sess = list(map(int, v))    # 这里要把session节点的str类型变为int

            for i in range(1, len(sess)):
                tar = sess[-i]
                key_new = k + '_' + str(i)
                new_data[key_new] = [sess[:-i], [tar]]
        return new_data

    def next_val_minibatch_feed_dict(self, val_or_test='val'):
        """
        Construct evaluation or test inputs.
        :returns [0] masked_input: A list of binary vector of max(node_num) of the mask of nodes in heterogeneous graph across different channels.
                 [1] session_adjs: A tensor of adjacent matrix of heterogeneous graph
                 [2] edge_mask: A list of binary vector of max(edge_num) for edge masking. [0] and [2] can be used as a flag in augmentation process
                 [3] items: A list of vector of max(node_num) which contains the item_id
                 [4] targets: The labels for evaluating results
        """
        if val_or_test == 'val':
            start = self.batch_num_val * self.batch_size
            self.batch_num_val += 1
            seq_data = self.seq_val
            seq_keys = self.valid_keys
        elif val_or_test == 'test':
            start = self.batch_num_test * self.batch_size
            self.batch_num_test += 1
            seq_data = self.seq_test
            seq_keys = self.test_keys
        else:
            raise NotImplementedError
        if start+self.batch_size < len(seq_keys):
            current_keys = seq_keys[start: start + self.batch_size]
        else:
            current_keys = seq_keys[start: len(seq_keys)]

        total_graph_list = []
        users = [int(session_id.split('_')[0]) for session_id in current_keys]
        time_ids = [int(session_id.split('_')[1]) for session_id in current_keys]
        # obtain the user_id and time_id through session_id

        self.adj_list = self.test_sampler([users, time_ids])

        for i in range(len(current_keys)):
            # total_graph_list = [mp_self_slice, mp_social_slice]
            # time_start = time.time()
            current_key = current_keys[i]
            user = users[i]
            time_id = time_ids[i]
            if len(self.adj_list[i]) > 0:
                neighbors = np.unique(self.adj_list[i])
            else:
                neighbors = []
            support_items = []
            self_sessions = self_sample([user, time_id, self.latest_sessions[user]])

            # Construct the historical sessions for the current user

            for support_user in neighbors:
                support_session_id = str(self.latest_sessions[support_user][time_id]) + '_' + str(1)
                # Here we add string "1" because the padded session name is not exactly same with the preprocessing process
                # And index "1" means the longest sequence in padding function.
                support_items += self.seq_all[support_session_id][1]
                support_items += self.seq_all[support_session_id][0]
                # For self.seq_all, index[1] refers to the session label, index [0] refers to the rest of the session (input part)

            support_items = np.unique(support_items)
            self_items = []
            for sess_id in self_sessions:
                # self_session_id = str(self.latest_sessions[users[i]][time_id])
                sess_id = sess_id + '_' + str(1)
                self_items += self.seq_all[sess_id][1] + self.seq_all[sess_id][0]
                # index [1] means the session target (label), index [0] mean the remaining sequence(input part)
            if len(self_items) > 0:
                self_items = list(np.unique(self_items))

            seq = seq_data[current_key][0]
            node_sample = min(int(self.max_length - len(seq) / (4 * len(seq))), self.samples_max)
            mp_cate_node, mp_cate_edge = self.sampling_test.random_walk_restart((seq, node_sample, 'cate'))
            mp_city_node, mp_city_edge = self.sampling_test.random_walk_restart((seq, node_sample, 'city'))
            mp_social_node, mp_social_edge = construct_social_influence((seq, support_items, node_sample))
            mp_self_node, mp_self_edge = construct_social_influence((seq, self_items, node_sample))
            total_graph_list.append([[mp_self_node, mp_self_edge], [mp_social_node, mp_social_edge],
                                     [mp_cate_node, mp_cate_edge], [mp_city_node, mp_city_edge]])

        masked_input, session_adjs, edge_mask, items, targets = self.process_slice(total_graph_list, current_keys, mode=val_or_test)
        return masked_input, session_adjs, edge_mask, items, targets

    def next_train_minibatch_feed_dict(self, pretrain=None):
        """
        Generate next training batch data.
        :returns [0] masked_input: A list of binary vector of max(node_num) of the mask of nodes in heterogeneous graph across different channels.
                 [1] session_adjs: A tensor of adjacent matrix of heterogeneous graph
                 [2] edge_mask: A list of binary vector of max(edge_num) for edge masking. [0] and [2] can be used as a flag in augmentation process
                 [3] items: A list of vector of max(node_num) which contains the item_id
                 [4] targets: The labels for training

        """
        start = self.batch_num * self.batch_size
        self.batch_num += 1
        total_graph_list = []
        if pretrain:
            keys = self.key_pretrain
        else:
            keys = self.train_keys
        if start+self.batch_size < len(keys):
            current_keys = keys[start: start + self.batch_size]
        else:
            current_keys = keys[start: len(keys)]
        users = [int(session_id.split('_')[0]) for session_id in current_keys]
        time_ids = [int(session_id.split('_')[1]) for session_id in current_keys]
        self.adj_list = self.train_sampler([users, time_ids])

        for i in range(len(current_keys)):           
            # total_graph_list = [mp_self_slice, mp_social_slice]
            # time_start = time.time()
            current_key = current_keys[i]
            user = users[i]
            time_id = time_ids[i]
            if len(self.adj_list[i]) > 0:
                neighbors = np.unique(self.adj_list[i])
            else:
                neighbors = []
            support_items = []
            self_sessions = self_sample([user, time_id, self.latest_sessions[user]])
            for support_user in neighbors:
                support_session_id = str(self.latest_sessions[support_user][time_id]) + '_' + str(1)
                # support session with index 1 has most items
                support_items += self.seq_train[support_session_id][1]
                support_items += self.seq_train[support_session_id][0]
                # 0表示的是session的feature部分，-1是在padding_session的时候session第一个item单独放置的位置
            support_items = np.unique(support_items)
            self_items = []
            for sess_id in self_sessions:
                sess_id = sess_id + '_' + str(1)
                # self_session_id = str(self.latest_sessions[users[i]][time_id])
                self_items += self.seq_train[sess_id][1] + self.seq_train[sess_id][0]
            if len(self_items) > 0:
                self_items = list(np.unique(self_items))
            # for key in self.seq_train[current_key][0]:
            seq = self.seq_train[current_key][0]
            node_sample = min(int((self.max_length - len(seq)) / (4 * len(seq))), self.samples_max)
            mp_cate_node, mp_cate_edge = self.sampling.random_walk_restart((seq, node_sample, 'cate'))
            # mp_cate_slice = map(self.sampling.random_walk_sampling_with_fly_back, seq,
            # [self.samples_max for _ in range(len(seq))], ['cate' for _ in range(len(seq))])
            mp_city_node, mp_city_edge = self.sampling.random_walk_restart((seq, node_sample, 'city'))
            #  mp_city_slice = map(self.sampling.random_walk_sampling_with_fly_back, seq,
            # [self.samples_max for _ in range(len(seq))], ['city' for _ in range(len(seq))])
            mp_social_node, mp_social_edge = construct_social_influence((seq, support_items, node_sample))
            mp_self_node, mp_self_edge = construct_social_influence((seq, self_items, node_sample))
            total_graph_list.append([[mp_self_node, mp_self_edge], [mp_social_node, mp_social_edge],
                                     [mp_cate_node, mp_cate_edge], [mp_city_node, mp_city_edge]])

        # total_graph_list = [mp_self_slice, mp_social_slice, mp_cate_slice, mp_city_slice]

        # total_graph_list = [mp_self, mp_social, mp_cate, mp_city]
        masked_input, session_adjs, edge_mask, items, targets = self.process_slice(total_graph_list, current_keys)
        return masked_input, session_adjs, edge_mask, items, targets

    def process_slice(self, total_graph_list, key_list, mode='train'):
        """
        We calculate the adjacent matrix/mask matrix of heterogeneous graph for feeding the data to the next training step
        :param total_graph_list: The list of nodes and edges of all different channels of heterogeneity
        :param key_list: The list of session_id going to fed into current batch
        :param mode: train/valid/test to decide
        """

        items, edge_ind, mask_inputs, edge_mask, len_list, edge_num_list, targets = [], [], [], [], [], [], []
        if mode == 'train':
            session_list = self.seq_train
        elif mode == 'val':
            session_list = self.seq_val
        else:
            session_list = self.seq_test
        for g_slice in total_graph_list:
            node = list(np.unique(g_slice[0][0] + g_slice[1][0] + g_slice[2][0] + g_slice[3][0]))
            edge_num = np.max([len(g_slice[0][1]), len(g_slice[1][1]), len(g_slice[2][1]), len(g_slice[3][1])])
            len_list.append(len(node))
            edge_num_list.append(edge_num)
        # print(len(len_list))
        max_n_node = np.max(len_list)
        max_n_edge = np.max(edge_num_list)
        graph_ind = 0
        for key in key_list:
            # for _ in session_list[key][0]:
            g_slice = total_graph_list[graph_ind]
            node = np.unique(g_slice[0][0] + g_slice[1][0] + g_slice[2][0] + g_slice[3][0])
            # 这里需要把每个小图的所有类型的meta_path的节点加起来
            # We add every node across meta-path channels to calculate the total node number
            s_item = np.array(node.tolist() + (max_n_node - len(node)) * [0])
            items.append(s_item)
            edge_ind_self = np.zeros((2, max_n_edge))
            edge_ind_social = np.zeros((2, max_n_edge))
            edge_ind_city = np.zeros((2, max_n_edge))
            edge_ind_cate = np.zeros((2, max_n_edge))
            # we define different types of edges containing the node index

            edge_self_mask = np.zeros(max_n_edge)
            edge_social_mask = np.zeros(max_n_edge)
            edge_city_mask = np.zeros(max_n_edge)
            edge_cate_mask = np.zeros(max_n_edge)
            # The mask of different types of edges

            self_edge = g_slice[0][1]
            social_edge = g_slice[1][1]
            cate_edge = g_slice[2][1]
            city_edge = g_slice[3][1]

            for i, edge in enumerate(self_edge):
                u = np.where(node == edge[0])[0][0]
                v = np.where(node == edge[1])[0][0]
                edge_ind_self[0][i] = u
                edge_ind_self[1][i] = v
                edge_self_mask[i] = 1
            for i, edge in enumerate(social_edge):
                u = np.where(node == edge[0])[0][0]
                v = np.where(node == edge[1])[0][0]
                edge_ind_social[0][i] = u
                edge_ind_social[1][i] = v
                edge_social_mask[i] = 1
            for i, edge in enumerate(cate_edge):
                u = np.where(node == edge[0])[0][0]
                v = np.where(node == edge[1])[0][0]
                edge_ind_cate[0][i] = u
                edge_ind_cate[1][i] = v
                edge_cate_mask[i] = 1
            for i, edge in enumerate(city_edge):
                u = np.where(node == edge[0])[0][0]
                v = np.where(node == edge[1])[0][0]
                edge_ind_city[0][i] = u
                edge_ind_city[1][i] = v
                edge_city_mask[i] = 1

            edge_ind.append([edge_ind_self, edge_ind_social, edge_ind_cate, edge_ind_city])
            edge_mask.append(([edge_self_mask, edge_social_mask, edge_cate_mask, edge_city_mask]))

            mask_self = np.in1d(s_item, g_slice[0][0]).astype(int)
            mask_social = np.in1d(s_item, g_slice[1][0]).astype(int)
            mask_cate = np.in1d(s_item, g_slice[2][0]).astype(int)
            mask_city = np.in1d(s_item, g_slice[3][0]).astype(int)
            mask_inputs.append([mask_self, mask_social, mask_cate, mask_city])
            graph_ind += 1
            # Alias_input表示的是在一个session中的节点在node_list（这里返回的是items）里面的位置
            targets += session_list[key][1]

        return np.array(mask_inputs), edge_ind, edge_mask, np.array(items), np.array(targets)

    def end(self):
        """
        Indicate whether we finish a pass over all training samples.
        """
        # return self.batch_num * self.batch_size > len(self.train_keys) - self.batch_size
        end = self.batch_num * self.batch_size < len(self.train_keys)
        # print('batch:', self.batch_num)
        if not end:
            self.batch_num = 0
            self.shuffle()
        return end

    def end_pretrain(self):
        # return self.batch_num * self.batch_size > len(self.train_keys) - self.batch_size
        end = self.batch_num * self.batch_size < len(self.key_pretrain)
        # print('batch:', self.batch_num)
        if not end:
            self.batch_num = 0
            self.shuffle()
        return end

    def end_val(self, val_or_test='val'):
        """
        Indicate whether we finish a pass over all testing or evaluation samples.
        """
        batch_num = self.batch_num_val if val_or_test == 'val' else self.batch_num_test
        data_len = len(self.valid_keys) if val_or_test == 'val' else len(self.test_keys)
        # end = batch_num * self.batch_size > data_len - self.batch_size
        end = batch_num * self.batch_size < data_len
        if not end:
            # print('wanna finish')
            if val_or_test == 'val':
                self.batch_num_val = 0
            elif val_or_test == 'test':
                self.batch_num_test = 0
            else:
                raise NotImplementedError
        return end

    def shuffle(self):
        """
        Shuffle training data.
        """
        self.train_keys = np.random.permutation(self.train_keys)  # 随机打乱
        self.key_pretrain = np.random.permutation(self.key_pretrain)

    def pretrain_keys(self):
        pr_key_list = []
        for key in self.train_keys:
            if key.split('_')[2] == '1':
                pr_key_list.append(key)
        return pr_key_list


if __name__ == '__main__':
    data = process.load_data('Yelp/processed')
    adj_info = data[0]
    adj_test = data[1]
    latest_per_user_by_time = data[2]
    num_list = data[3]
    train_df = data[4]
    valid_df = data[5]
    test_df = data[6]
    minibatch = MinibatchIterator(adj_info,
                                  adjs_test=adj_test,
                                  latest_sessions=latest_per_user_by_time,
                                  data_list=[train_df, valid_df, test_df],
                                  batch_size=32,
                                  max_degree=50,
                                  num_nodes=num_list,
                                  max_length=150)
    time_start = time.time()
    while minibatch.end():
        sess_alias, sess_adjs, sess_edge_mask, sess_item, sess_targets = minibatch.next_train_minibatch_feed_dict()
    time_end = time.time()
    print("finish the time:{}".format(time_end - time_start))
