import numpy as np

"""This section is inspired by https://github.com/Ashish7129/Graph_Sampling"""


class RandomWalkRestart(object):

    def __init__(self, adj_mat):
        self.growth_size = 2
        self.T = 10    # number of iterations
        self.adj_mat = adj_mat
        # with a probability (1-fly_back_prob) select a neighbor node
        # with a probability fly_back_prob go back to the initial vertex
        self.fly_back_prob = 0.35

        for i in range(len(self.adj_mat)):
            self.adj_mat[i] = self.adj_mat[i].tolil()
        # self.pool = Pool(1)

    def random_walk_sampling_with_fly_back(self, param_tuple):
        node = param_tuple[0]
        nodes_to_sample = param_tuple[1]
        mp_type = param_tuple[2]
        if mp_type == 'cate':
            adj_mat = self.adj_mat[1]
        else:
            adj_mat = self.adj_mat[2]
        curr_node = node
        sampled_node_list = [curr_node]
        iteration = 0
        node_before_t_iter = 0
        sampled_edge_list = []
        while iteration < nodes_to_sample:
            candidate_list = adj_mat.rows[curr_node]
            # sample_ind = random.randint(0, len(candidate_list) - 1)
            chosen_node = int(np.random.choice(candidate_list, 1, replace=False))
            # chosen_node = candidate_list[sample_ind]
            sampled_node_list.append(chosen_node)
            sampled_edge_list.append((curr_node, chosen_node))
            choice = np.random.choice(['prev', 'neigh'], 1, p=[self.fly_back_prob, 1 - self.fly_back_prob])
            if choice == 'neigh':
                curr_node = chosen_node
            iteration = iteration + 1

            if iteration % self.T == 0:
                if (len(sampled_node_list) - node_before_t_iter) >= self.growth_size:
                    break
                else:
                    iteration = 0
        sampled_node_list = list(np.unique(sampled_node_list))
        sampled_edge_list = list(set(sampled_edge_list))

        return sampled_node_list, sampled_edge_list

    # def random_walk_restart(self, adj_mat, node_list, node_to_sample):
    def random_walk_restart(self, param_tuple):
        """
        Due to the tremendous complexity of sampling recursively, we just find that randomly sampling
        the one-hop meta-path neighbors also achieves a satisfying result
        """

        # node_list = param_tuple[0]
        node_list = param_tuple[0]
        node_to_sample = param_tuple[1]
        mp_type = param_tuple[2]
        if isinstance(mp_type, int):
            adj_mat = self.adj_mat[mp_type]
        elif mp_type == 'cate':
            adj_mat = self.adj_mat[1]
        else:
            adj_mat = self.adj_mat[2]
        sampled_node_list = []
        sampled_edge_list = []
        count = 0
        for node in node_list:
            # The following code in comments is designed to sample nodes recursively
            """
            curr_node = node
            sampled_node = [curr_node]
            iteration = 0
            node_before_t_iter = 0
            while iteration < node_to_sample:
                candidate_list = adj_mat.rows[curr_node]
                # sample_ind = random.randint(0, len(candidate_list) - 1)
                chosen_node = int(np.random.choice(candidate_list, 1, replace=False))
                # chosen_node = candidate_list[sample_ind]
                sampled_node.append(chosen_node)
                sampled_edge_list.append((curr_node, chosen_node))
                choice = np.random.choice(['prev', 'neigh'], 1, p=[self.fly_back_prob, 1 - self.fly_back_prob])
                if choice == 'neigh':
                    curr_node = chosen_node
                iteration = iteration + 1

                if iteration % self.T == 0:
                    if (len(sampled_node) - node_before_t_iter) >= self.growth_size:
                        break
                    else:
                        iteration = 0
                    node_before_t_iter = len(sampled_node)
            sampled_node_list += list(np.unique(sampled_node))
            """
            sampled_node = [node]
            if adj_mat.rows[node]:
                chosen_node = list(np.random.choice(adj_mat.rows[node], node_to_sample))
            else:
                chosen_node = []
            # chosen_node = list(np.random.choice(adj_mat[node], node_to_sample))
            sampled_node += chosen_node
            sample_len = len(sampled_node)
            for i in range(1, sample_len):
                sampled_edge_list.append((sampled_node[0], sampled_node[i]))
                sampled_edge_list.append((sampled_node[i], sampled_node[0]))
            if count > 0:
                sampled_edge_list.append((node_list[count-1], node_list[count]))
            count += 1
            sampled_node_list += sampled_node

        sampled_node_list = list(np.unique(sampled_node_list))
        sampled_edge_list = list(set(sampled_edge_list))

        return sampled_node_list, sampled_edge_list


class UniformNeighborSampler(object):
    """
    Uniformly samples neighbors.
    Assumes that adj lists are padded with random re-sampling
    """
    def __init__(self, adj_info, visible_time, num_samples):
        self.adj_info = adj_info        # data structure type of adj_info: lil matrix
        self.visible_time = visible_time
        self.deg = np.array(adj_info.sum(axis=1)).squeeze()
        self.num_samples = num_samples

    def __call__(self, inputs):
        user_ids, time_ids = inputs

        adj_lists = []
        for idx in range(len(user_ids)):
            node = user_ids[idx]
            time_id = time_ids[idx]
            adj = list(self.adj_info.rows[node])
            neighbors = []
            for neighbor in adj:
                if self.visible_time[neighbor] <= time_id:
                    neighbors.append(neighbor)
            # assert len(neighbors) > 0
            if len(neighbors) > 0:
                if len(neighbors) < self.num_samples:
                    neighbors = list(np.unique(neighbors))
                elif len(neighbors) >= self.num_samples:
                    neighbors = list(np.random.choice(neighbors, self.num_samples, replace=False))
            adj_lists.append(neighbors)
        return adj_lists


def self_sample(inputs):
    """
    This function is designed for sampling the historical items of the same user only on onymous scenarios
    :param inputs: [0] Current user id [1] Current time id [2] The whole data of historical session (of each user)
    :return: A list of historical session_id
    """
    user_id, time_id, last_sessions = inputs
    sess_candi = last_sessions[0:time_id + 1]
    sess_candi = list(np.unique(sess_candi))
    sess_candi.remove('NULL')
    return sess_candi


def construct_social_influence(param_list):
    """
    This function is designed for constructing the social influence via heterogeneous graph
    and the meta-path Item-User-User-Item only on onymous scenarios.
    :param param_list: [0] The user's historical sequence, [1] The neighbors of meta-path Item-User-User-Item
    [2] The defined sample size (scale)
    :return: The node list and edge list of the social channel
    """

    seq = param_list[0]
    support_items = param_list[1]
    sample_size = param_list[2]
    mp_social_node = []
    mp_social_edge = []
    count = 0
    for seq_node in seq:
        # Here we traverse every item in a sequence, so the different item nodes may connect with different support items via random sampling
        if len(support_items) >= sample_size:
            social_node_temp = list(np.random.choice(support_items, sample_size, replace=False))
            # social_node_temp += [seq_node]
        else:
            social_node_temp = list(support_items)
        for support_node in social_node_temp:
            if not seq_node == support_node:
                mp_social_edge.append((seq_node, support_node))
                mp_social_edge.append((support_node, seq_node))
        if count > 0:
            mp_social_edge.append((seq[count-1], seq[count]))
        mp_social_node += social_node_temp
        count += 1
    mp_social_node += seq
    mp_social_edge = list(set(mp_social_edge))
    mp_social_node = list(np.unique(mp_social_node))
    return mp_social_node, mp_social_edge
