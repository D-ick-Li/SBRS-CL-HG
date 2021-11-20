import torch
import copy
import random
import scipy.sparse as sp
import numpy as np
import copy

"""
This section is inspired by GraphCL in https://github.com/Shen-Lab/GraphCL
"""


def aug_random_mask(input_mask, input_items, drop_percent=0.2):
    """
    This function randomly masks the node attribute
    :param input_mask: the original mask of items in generating batches
    :param input_items: the original input (item_id)
    :param drop_percent: mask rate
    :return: augmented node information
    """
    mask_graph = input_mask.sum(axis=1)
    mask_graph[mask_graph > 1] = 1
    node_num = mask_graph.sum(axis=1)
    mask_num = np.array(node_num * drop_percent, dtype=np.int32)
    aug_items = []
    for idx in range(len(node_num)):
        node_idx = [i for i in range(node_num[idx])]
        mask_idx = random.sample(node_idx, mask_num[idx])
        node_aug = copy.deepcopy(input_items[idx])
        for j in mask_idx:
            node_aug[j] = 0
        aug_items.append(node_aug)
    return np.array(aug_items)


def aug_random_edge(input_adj, input_mask, drop_percent=0.2):
    """
    The erasive version of edge perturbation
    """
    percent = drop_percent / 2
    row_idx, col_idx = input_adj.nonzero()  # 有几个维度就返回几个array,比如一个array表示行，另一个表示列

    index_list = []
    for i in range(len(row_idx)):
        index_list.append((row_idx[i], col_idx[i]))

    edge_num = len(row_idx)
    add_drop_num = int(edge_num * percent / 2)  # 这里除以,一半用来加边，一半用来减边
    # aug_adj = copy.deepcopy(input_adj.tolist())  # 这里deepcopy完全将之前的邻接矩阵复制
    aug_adj = input_adj.copy().tolist()
    edge_idx = [i for i in range(edge_num)]
    drop_idx = random.sample(edge_idx, add_drop_num)  # random.sample可以进行随机的取样

    for i in drop_idx:
        aug_adj[index_list[i][0]][index_list[i][1]] = 0
        # aug_adj[single_index_list[i][1]][single_index_list[i][0]] = 0

    '''
    above finish drop edges
    '''
    node_num = input_mask.sum()
    edge_list = [(i, j) for i in range(node_num) for j in range(i)]
    add_list = random.sample(edge_list, add_drop_num)

    for i in add_list:
        aug_adj[i[0]][i[1]] = 1

    aug_adj = np.array(aug_adj)
    # aug_adj = sp.csr_matrix(aug_adj)
    return aug_adj  # 返回一个numpy稀疏矩阵


def aug_drop_node(input_items, input_masks, input_adj, drop_percent=0.2):
    """
    The node drop augmentation.
    :param input_items:
    :param input_masks:
    :param input_adj:
    :param drop_percent:
    :return:
    """
    batch_num = input_masks.shape[0]
    node_num = input_masks.sum(axis=1)
    drop_num = np.array(node_num * drop_percent, dtype=np.int32)  # number of drop nodes
    aug_items = input_items.copy()
    aug_masks = input_masks.copy()
    aug_adj = copy.deepcopy(input_adj)
    mp_num = len(aug_adj[0])
    for i in range(batch_num):
        drop_candidate = np.squeeze(np.array(np.where(aug_masks[i, :] == 1)))
        drop_idx = np.random.choice(drop_candidate, drop_num[i])
        for j in drop_idx:
            aug_items[i][j] = 0
            aug_masks[i][j] = 0
            for k in range(mp_num):
                aug_adj[i][k][k, :] = 0
                aug_adj[i][k][:, k] = 0

    return aug_masks, aug_items, aug_adj


def aug_subgraph(input_fea, input_adj, drop_percent=0.2):
    """
    Subgraph Sampling in GraphCL
    """
    input_adj = torch.tensor(input_adj.todense().tolist())
    input_fea = input_fea.squeeze(0)
    node_num = input_fea.shape[0]

    all_node_list = [i for i in range(node_num)]
    s_node_num = int(node_num * (1 - drop_percent))  # 抽取子图，是看保留多少个节点
    center_node_id = random.randint(0, node_num - 1)  # 中央的节点，要围绕这个节点来抽取
    sub_node_id_list = [center_node_id]
    all_neighbor_list = []

    for i in range(s_node_num - 1):

        all_neighbor_list += torch.nonzero(input_adj[sub_node_id_list[i]], as_tuple=False).squeeze(1).tolist()
        # 更新所有邻居节点的list,第i个循环就把对应节点的邻接矩阵的行找到
        all_neighbor_list = list(set(all_neighbor_list))  # set可以删除重复的元素。
        new_neighbor_list = [n for n in all_neighbor_list if n not in sub_node_id_list]  # 这里把加入子图节点的元素划去
        if len(new_neighbor_list) != 0:
            new_node = random.sample(new_neighbor_list, 1)[0]
            sub_node_id_list.append(new_node)
        else:
            break

    drop_node_list = sorted([i for i in all_node_list if i not in sub_node_id_list])  # 把不再子图里面的节点给去掉

    aug_input_fea = delete_row_col(input_fea, drop_node_list, only_row=True)
    aug_input_adj = delete_row_col(input_adj, drop_node_list)

    aug_input_fea = aug_input_fea.unsqueeze(0)
    aug_input_adj = sp.csr_matrix(np.matrix(aug_input_adj))

    return aug_input_fea, aug_input_adj  # 返回邻接矩阵是稀疏矩阵


def delete_row_col(input_matrix, drop_list, only_row=False):
    remain_list = [i for i in range(input_matrix.shape[0]) if i not in drop_list]
    out = input_matrix[remain_list, :]  # 这里把剩下的行进行保留,only_row是处理特征的时候使用
    if only_row:
        return out
    out = out[:, remain_list]  # 这里把剩下的列进行保留

    return out


def execute_random_edge(sess_adj, sess_masks, aug_idx=1):
    aug_adj = copy.deepcopy(sess_adj)
    for i in range(len(sess_adj)):
        aug_slice = aug_random_edge(sess_adj[i][aug_idx], sess_masks[i])
        aug_adj[i][aug_idx] = aug_slice
    return aug_adj


def get_degree(adjs, masks):
    """
    Get the node degree
    :param adjs: adjacent matrix
    :param masks: mask of edges
    :return: The array of the degree of each edge.
    """
    node_max = 0
    for adj in adjs:
        if adj.max() > node_max:
            node_max = adj.max()
    degree_array = np.zeros((len(adjs), int(node_max)+1))
    for i in range(len(adjs)):
        for j in range(len(masks[0])):
            if masks[i][j] != 0:
                tar = int(adjs[i][1][j])
                degree_array[i][tar] += 1
    return degree_array


def degree_weights(sess_adjs, sess_edge_mask, degree_list, threshold=0.6, drop_rate=0.6):
    """
    Calculate the edge centrality on target node degrees
    Please refer to Eq.(2) and (3) for more details
    """
    # weight_list= []
    for i in range(len(sess_edge_mask)):
        weight_temp = np.zeros(int(sess_edge_mask[i].sum()))
        for j in range(len(sess_edge_mask[0])):
             if sess_edge_mask[i][j] != 0:
                tar = int(sess_adjs[i][1][j])
                weight_temp[j] = degree_list[i][tar]
        weight_temp = np.log(weight_temp + 1)
        if np.all(weight_temp==1):
            return sess_adjs
        weight_temp = (weight_temp.max() - weight_temp) / (weight_temp.max() - weight_temp.mean()) / weight_temp.mean() * drop_rate
        weight_temp = np.where(weight_temp < threshold, weight_temp, threshold)
        mask_perturbe = torch.bernoulli(1 - torch.FloatTensor(weight_temp))
        mask_perturbe = mask_perturbe.detach().numpy()
        node_weight = degree_list[i] / degree_list[i].sum()
        for k in range(weight_temp.shape[0]):
            if mask_perturbe[k] == 0:
                new_target = np.random.choice(degree_list[i].shape[0], 1, p=np.array(node_weight))
                sess_adjs[i][1][k] = new_target
    return sess_adjs


def adaptive_drop_edge(sess_adjs, sess_edge_mask):
    """
    Our proposed adaptive edge perturbation
    :param sess_adjs: adjacent matrix
    :param sess_edge_mask: mask of edges
    :return: the augmented adjacent matrix
    """
    sess_adj_new = []
    sess_adjs_temp = copy.deepcopy(sess_adjs)
    for i in range(len(sess_adjs)):
        deg_mat = get_degree(sess_adjs_temp[i], sess_edge_mask[i])
        sess_adj_new.append(degree_weights(sess_adjs_temp[i], sess_edge_mask[i], deg_mat))

    return sess_adj_new

