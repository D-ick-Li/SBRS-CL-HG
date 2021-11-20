# coding=utf-8
from __future__ import division
from __future__ import print_function

import time
import numpy as np
from utils import process
from utils.sampling import RandomWalkRestart, construct_social_influence, self_sample


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
        # self.visible_time = self.user_visible_time()
        # self.test_adj, self.test_deg = self.construct_test_adj(adjs_test)
        # if self.state_flag == 'train':
        # self.train_sampler = UniformNeighborSampler(self.adjs[0], self.visible_time, self.samples_max)
        # self.test_sampler = UniformNeighborSampler(self.adjs_test[0], self.visible_time, self.samples_max)
        # self.test_session_ids = self._remove_infoless(self.test_df, self.test_adj, self.test_deg)
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


    @staticmethod
    def padding_sessions(raw_data):
        """
        transform the dataframe to the session list
        """
        raw_data = raw_data.sort_values(by=['time_stamp']).groupby('session_id')['item_id'].apply(
                                list).to_dict()  # 同一个session内按照时间排序,，最后得到一个根据session_id的dict
        new_data = {}
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
        return new_data

    def next_val_minibatch_feed_dict(self, val_or_test='val'):
        """
        Construct evaluation or test inputs.
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
        for i in range(len(current_keys)):
            # total_graph_list = [mp_self_slice, mp_social_slice]
            # time_start = time.time()
            current_key = current_keys[i]
            seq = seq_data[current_key][0]
            node_sample = min(int(self.max_length - len(seq) / (3 * len(seq))), self.samples_max)
            mp_seller_node, mp_seller_edge = self.sampling_test.random_walk_restart((seq, node_sample, 0))
            mp_brand_node, mp_brand_edge = self.sampling_test.random_walk_restart((seq, node_sample, 1))
            mp_cate_node, mp_cate_edge = self.sampling_test.random_walk_restart((seq, node_sample, 2))
            total_graph_list.append([[mp_seller_node, mp_seller_edge], [mp_brand_node, mp_brand_edge],
                                     [mp_cate_node, mp_cate_edge]])

        # total_graph_list = [mp_self_slice, mp_social_slice, mp_cate_slice, mp_city_slice]

        # total_graph_list = [mp_self, mp_social, mp_cate, mp_city]
        alias_input, session_adjs, edge_mask, items, targets = self.process_slice(total_graph_list, current_keys, mode=val_or_test)
        return alias_input, session_adjs, edge_mask, items, targets
        # return self._batch_feed_dict([current_batch_sessions, samples, support_sizes])

    def next_train_minibatch_feed_dict(self, pretrain=None):
        """
        Generate next training batch data.
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

        for i in range(len(current_keys)):
            current_key = current_keys[i]
            seq = self.seq_train[current_key][0]
            node_sample = min(int(self.max_length - len(seq) / (3 * len(seq))), self.samples_max)
            mp_seller_node, mp_seller_edge = self.sampling_test.random_walk_restart((seq, node_sample, 0))
            mp_brand_node, mp_brand_edge = self.sampling_test.random_walk_restart((seq, node_sample, 1))
            mp_cate_node, mp_cate_edge = self.sampling_test.random_walk_restart((seq, node_sample, 2))
            total_graph_list.append([[mp_seller_node, mp_seller_edge], [mp_brand_node, mp_brand_edge],
                                     [mp_cate_node, mp_cate_edge]])

        # total_graph_list = [mp_self_slice, mp_social_slice, mp_cate_slice, mp_city_slice]

        # total_graph_list = [mp_self, mp_social, mp_cate, mp_city]
        alias_input, session_adjs, edge_mask, items, targets = self.process_slice(total_graph_list, current_keys)
        return alias_input, session_adjs, edge_mask, items, targets

    def process_slice(self, total_graph_list, key_list, mode='train'):
        items, edge_ind, mask_inputs, edge_mask, len_list, edge_num_list, targets = [], [], [], [], [], [], []
        if mode == 'train':
            session_list = self.seq_train
        elif mode == 'val':
            session_list = self.seq_val
        else:
            session_list = self.seq_test
        for g_slice in total_graph_list:
            node = list(np.unique(g_slice[0][0] + g_slice[1][0] + g_slice[2][0]))
            edge_num = np.max([len(g_slice[0][1]), len(g_slice[1][1]), len(g_slice[2][1])])
            len_list.append(len(node))
            edge_num_list.append(edge_num)
        # print(len(len_list))
        max_n_node = np.max(len_list)
        max_n_edge = np.max(edge_num_list)
        graph_ind = 0
        for key in key_list:
            # for _ in session_list[key][0]:
            g_slice = total_graph_list[graph_ind]
            node = np.unique(g_slice[0][0] + g_slice[1][0] + g_slice[2][0])
            # 这里需要把每个小图的所有类型的meta_path的节点加起来
            s_item = np.array(node.tolist() + (max_n_node - len(node)) * [0])
            items.append(s_item)
            edge_ind_seller = np.zeros((2, max_n_edge))
            edge_ind_brand = np.zeros((2, max_n_edge))
            edge_ind_cate = np.zeros((2, max_n_edge))
            edge_seller_mask = np.zeros(max_n_edge)
            edge_brand_mask = np.zeros(max_n_edge)
            edge_cate_mask = np.zeros(max_n_edge)
            seller_edge = g_slice[0][1]
            brand_edge = g_slice[1][1]
            cate_edge = g_slice[2][1]
            for i, edge in enumerate(seller_edge):
                u = np.where(node == edge[0])[0][0]
                v = np.where(node == edge[1])[0][0]
                edge_ind_seller[0][i] = u
                edge_ind_seller[1][i] = v
                edge_seller_mask[i] = 1
            for i, edge in enumerate(brand_edge):
                u = np.where(node == edge[0])[0][0]
                v = np.where(node == edge[1])[0][0]
                edge_ind_brand[0][i] = u
                edge_ind_brand[1][i] = v
                edge_brand_mask[i] = 1
            for i, edge in enumerate(cate_edge):
                u = np.where(node == edge[0])[0][0]
                v = np.where(node == edge[1])[0][0]
                edge_ind_cate[0][i] = u
                edge_ind_cate[1][i] = v
                edge_cate_mask[i] = 1

            edge_ind.append([edge_ind_seller, edge_ind_brand, edge_ind_cate])
            edge_mask.append(([edge_seller_mask, edge_brand_mask, edge_cate_mask]))
            # alias_inputs.append(sess)
            # alias_inputs.append([np.where(node == i)[0][0] for i in sess])
            mask_self = np.in1d(s_item, g_slice[0][0]).astype(int)
            mask_brand = np.in1d(s_item, g_slice[1][0]).astype(int)
            mask_cate = np.in1d(s_item, g_slice[2][0]).astype(int)
            mask_inputs.append([mask_self, mask_brand, mask_cate])
            graph_ind += 1
            # Alias_input表示的是在一个session中的节点在node_list（这里返回的是items）里面的位置
            targets += session_list[key][1]

        return np.array(mask_inputs), edge_ind, edge_mask, np.array(items), np.array(targets)

    def get_slice(self, total_graph_list, key_list, mode='train'):
        # inputs, mask, targets = self.inputs[i], self.mask[i], self.targets[i]
        items, adj, mask_inputs, len_list, targets = [], [], [], [], []
        if mode == 'train':
            session_list = self.seq_train
        elif mode == 'val':
            session_list = self.seq_val
        else:
            session_list = self.seq_test

        for g_slice in total_graph_list:
            node = list(np.unique(g_slice[0][0] + g_slice[1][0] + g_slice[2][0]))
            len_list.append(len(node))
        # print(len(len_list))
        max_n_node = np.max(len_list)
        graph_ind = 0
        for key in key_list:
            # for _ in session_list[key][0]:
            g_slice = total_graph_list[graph_ind]
            node = np.unique(g_slice[0][0] + g_slice[1][0] + g_slice[2][0])
            # 这里需要把每个小图的所有类型的meta_path的节点加起来
            s_item = np.array(node.tolist() + (max_n_node - len(node)) * [0])
            items.append(s_item)
            adj_seller = np.zeros((max_n_node, max_n_node))
            adj_brand = np.zeros((max_n_node, max_n_node))
            adj_cate = np.zeros((max_n_node, max_n_node))
            adj_eye = np.eye(max_n_node)
            seller_edge = g_slice[0][1]
            brand_edge = g_slice[1][1]
            cate_edge = g_slice[2][1]
            for edge in seller_edge:
                u = np.where(node == edge[0])[0][0]
                v = np.where(node == edge[1])[0][0]
                adj_seller[u][v] = 1
            for edge in brand_edge:
                u = np.where(node == edge[0])[0][0]
                v = np.where(node == edge[1])[0][0]
                adj_brand[u][v] = 1
            for edge in cate_edge:
                u = np.where(node == edge[0])[0][0]
                v = np.where(node == edge[1])[0][0]
                adj_cate[u][v] = 1

            adj.append([adj_seller + adj_eye, adj_brand + adj_eye, adj_cate + adj_eye])
            # alias_inputs.append(sess)
            # alias_inputs.append([np.where(node == i)[0][0] for i in sess])
            mask_self = np.in1d(s_item, g_slice[0][0]).astype(int)
            mask_brand = np.in1d(s_item, g_slice[1][0]).astype(int)
            mask_cate = np.in1d(s_item, g_slice[2][0]).astype(int)
            mask_inputs.append([mask_self, mask_brand, mask_cate])
            graph_ind += 1
            # Alias_input表示的是在一个session中的节点在node_list（这里返回的是items）里面的位置
            targets += session_list[key][1]

        return np.array(mask_inputs), adj, np.array(items), np.array(targets)

    def end(self):
        """
        Indicate whether we finish a pass over all training samples.
        """
        # return self.batch_num * self.batch_size > len(self.train_keys) - self.batch_size
        end = self.batch_num * self.batch_size < len(self.train_keys)
        print('batch:', self.batch_num)
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
    data = process.load_data('Tmall/processed/', 'Tmall')
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
                                  batch_size=16,
                                  max_degree=50,
                                  num_nodes=num_list,
                                  max_length=120)
    time_start = time.time()
    while minibatch.end():
        sess_alias, sess_adjs, edge_mask, sess_item, sess_targets = minibatch.next_train_minibatch_feed_dict()
    time_end = time.time()
    print("finish the time:{}".format(time_end - time_start))
