import torch
import torch.nn as nn
import torch.nn.functional as F


class NodeAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, in_features, out_features, alpha, device, concat=True):
        super(NodeAttentionLayer, self).__init__()
        self.device = device
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.zeros(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def sp_softmax(self, source, values, node_num):
        # source, _ = indices
        v_max = values.max()
        exp_v = torch.exp(values - v_max)
        exp_sum = torch.zeros(node_num, 1, device=self.device)
        exp_sum.scatter_add_(0, source.unsqueeze(1), exp_v)
        # 将exp_v 中的值按照sorce的索引加到exp_sum中，这样就把0矩阵变成根据索引来
        exp_sum += 1e-10
        softmax_v = exp_v / exp_sum[source]
        return softmax_v

    def sp_matmul(self, indices, values, mat):
        source, target = indices
        out = torch.zeros_like(mat, device=self.device)
        out.scatter_add_(0, source.expand(mat.size(1), -1).t(), values * mat[target])
        return out

    def forward(self, input_hid, adj):
        h = torch.matmul(input_hid, self.W)
        item_num = h.size()[1]
        sess_num = h.size()[0]

        e_1 = torch.cat([h.repeat(1, 1, item_num).view(sess_num, item_num * item_num, -1),
                         h.repeat(1, item_num, 1)], dim=2).view(sess_num, item_num, -1, 2 * self.out_features)
        # tensor.repeat指的是在对应发维度恩据对应的参数改变形状，view相当于reshape， 这里的操作是将两个特征按照不同的顺序拼接。
        # 拼接的第一项进行view之后是feature倒数第二个维度顺序的排序是111222333这种，第二项拼接后是123123123这种，因此最后可以直接乘以a就得到注意力系数
        e_1 = self.leakyrelu(torch.matmul(e_1, self.a).squeeze(3))
        # a_input形状为[batch,batch,2*out_feature],最后e为[N,N]
        """
        h_list = []
        # e_list = []
        for i, h_slice in enumerate(h):
            
            a_input = torch.cat([h_slice.repeat(1, item_num).view(item_num * item_num, -1), h_slice.repeat(item_num, 1)],
                                dim=1).view(item_num, -1, 2 * self.out_features)
            e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))
            e_list.append(e)
            
            adj_slice = adj[i].to_sparse()
            source = adj_slice.indices()[0]
            target = adj_slice.indices()[1]
            a_input = torch.cat([h_slice[source], h_slice[target]], dim=1)
            e = self.leakyrelu(torch.matmul(a_input, self.a))
            attention = self.sp_softmax(source, e, item_num)
            h_prime = self.sp_matmul((source, target), attention, h_slice)
            h_list.append(h_prime)
       
        e = torch.stack(e_list, dim=0)
        # matmul一般有高维矩阵进行乘法
        """
        zero_vec = -9e15 * torch.ones_like(e_1, device=self.device)
        attention_1 = torch.where(adj > 0, e_1, zero_vec)
        # 相当于if else进行替换，第一个参数不满足的元素换成第二个
        attention_1 = F.softmax(attention_1, dim=2)
        # attention = F.dropout(attention, self.nd_dropout, training=self.training)
        h_prime = torch.bmm(attention_1, h)

        # h_prime = torch.stack(h_list, dim=0)
        if self.concat:
            # return F.elu(h_prime)
            return F.leaky_relu(h_prime)
        else:
            return h_prime

    def __repr__(self):  # 显示属性
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class SemanticAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, in_features, out_features):
        super(SemanticAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.b = nn.Parameter(torch.zeros(size=(1, out_features)))
        nn.init.xavier_uniform_(self.b.data, gain=1.414)
        self.q = nn.Parameter(torch.zeros(size=(1, out_features)))
        nn.init.xavier_uniform_(self.q.data, gain=1.414)
        self.Tanh = nn.Tanh()
        self.leakyrelu = nn.LeakyReLU()

    #   input (PN)*F
    def forward(self, input_item, mp_num):
        batch_size = input_item.size()[0]
        h = torch.matmul(input_item, self.W)
        # h=(PN)*F'
        # h_prime = self.Tanh(h + self.b.repeat(h.size()[1], 1))
        h_prime = self.Tanh(h + self.b)
        # h_prime=(PN)*F'
        semantic_attentions = torch.matmul(h_prime, torch.t(self.q)).view(batch_size, mp_num, -1)
        # semantic_attentions = batch*P*N, torch.t是转置,最后的维度N是节点数量
        num_head = semantic_attentions.size()[2]
        semantic_attentions = semantic_attentions.mean(dim=2, keepdim=True)
        # semantic_attentions = batch*P*1
        semantic_attentions = F.softmax(semantic_attentions, dim=1)
        # print(semantic_attentions)
        semantic_attentions = semantic_attentions.view(batch_size, mp_num, 1, 1)
        semantic_attentions = semantic_attentions.repeat(1, 1, num_head, self.in_features)
        # print(semantic_attentions)

        # input_embedding = batch*P*N*F
        input_embedding = input_item.view(batch_size, mp_num, num_head, self.in_features)

        # h_embedding = batch*N*F
        h_embedding = torch.mul(input_embedding, semantic_attentions)
        h_embedding = torch.sum(h_embedding, dim=1).squeeze()

        return h_embedding

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class GATLayer(torch.nn.Module):
    """
    Base class for all implementations as there is much code that would otherwise be copy/pasted.
    """

    head_dim = 1

    def __init__(self, num_in_features, num_out_features, num_of_heads, concat=True, activation=nn.ELU(),
                 dropout_prob=0.6, add_skip_connection=False, bias=True, log_attention_weights=False):

        super().__init__()

        # Saving these as we'll need them in forward propagation in children layers (imp1/2/3)
        self.num_of_heads = num_of_heads
        self.num_out_features = num_out_features
        self.concat = concat  # whether we should concatenate or average the attention heads
        self.add_skip_connection = add_skip_connection

        #
        # Trainable weights: linear projection matrix (denoted as "W" in the paper), attention target/source
        # (denoted as "a" in the paper) and bias (not mentioned in the paper but present in the official GAT repo)
        #
        # You can treat this one matrix as num_of_heads independent W matrices
        self.linear_proj = nn.Linear(num_in_features, num_of_heads * num_out_features, bias=False)

        # After we concatenate target node (node i) and source node (node j) we apply the additive scoring function
        # which gives us un-normalized score "e". Here we split the "a" vector - but the semantics remain the same.

        # Basically instead of doing [x, y] (concatenation, x/y are node feature vectors) and dot product with "a"
        # we instead do a dot product between x and "a_left" and y and "a_right" and we sum them up
        self.scoring_fn_target = nn.Parameter(torch.Tensor(1, num_of_heads, num_out_features))  # 相当于a的上半部分
        self.scoring_fn_source = nn.Parameter(torch.Tensor(1, num_of_heads, num_out_features))  # 相当于a的下半部分

        # Bias is definitely not crucial to GAT - feel free to experiment (I pinged the main author, Petar, on this one)
        if bias and concat:
            self.bias = nn.Parameter(torch.Tensor(num_of_heads * num_out_features))
        elif bias and not concat:
            self.bias = nn.Parameter(torch.Tensor(num_out_features))
        else:
            self.register_parameter('bias', None)

        if add_skip_connection:
            self.skip_proj = nn.Linear(num_in_features, num_of_heads * num_out_features, bias=False)
        else:
            self.register_parameter('skip_proj', None)

        #
        # End of trainable weights
        #

        self.leakyReLU = nn.LeakyReLU(0.2)  # using 0.2 as in the paper, no need to expose every setting
        self.softmax = nn.Softmax(dim=-1)  # -1 stands for apply the log-softmax along the last dimension
        self.activation = activation
        # Probably not the nicest design but I use the same module in 3 locations, before/after features projection
        # and for attention coefficients. Functionality-wise it's the same as using independent modules.
        self.dropout = nn.Dropout(p=dropout_prob)

        self.log_attention_weights = log_attention_weights  # whether we should log the attention weights
        self.attention_weights = None  # for later visualization purposes, I cache the weights here

        self.init_params()

    def init_params(self):
        """
        The reason we're using Glorot (aka Xavier uniform) initialization is because it's a default TF initialization:
            https://stackoverflow.com/questions/37350131/what-is-the-default-variable-initializer-in-tensorflow
        The original repo was developed in TensorFlow (TF) and they used the default initialization.
        Feel free to experiment - there may be better initializations depending on your problem.
        """
        nn.init.xavier_uniform_(self.linear_proj.weight)
        nn.init.xavier_uniform_(self.scoring_fn_target)
        nn.init.xavier_uniform_(self.scoring_fn_source)

        if self.bias is not None:
            torch.nn.init.zeros_(self.bias)

    def skip_concat_bias(self, attention_coefficients, in_nodes_features, out_nodes_features):
        if self.log_attention_weights:  # potentially log for later visualization in playground.py
            self.attention_weights = attention_coefficients

        # if the tensor is not contiguously stored in memory we'll get an error after we try to do certain ops like view
        # only imp1 will enter this one
        if not out_nodes_features.is_contiguous():
            out_nodes_features = out_nodes_features.contiguous()

        if self.add_skip_connection:  # add skip or residual connection
            if out_nodes_features.shape[-1] == in_nodes_features.shape[-1]:  # if FIN == FOUT
                # unsqueeze does this: (N, FIN) -> (N, 1, FIN), out features are (N, NH, FOUT) so 1 gets broadcast to NH
                # thus we're basically copying input vectors NH times and adding to processed vectors
                out_nodes_features += in_nodes_features.unsqueeze(1)
            else:
                # FIN != FOUT so we need to project input feature vectors into dimension that can be added to output
                # feature vectors. skip_proj adds lots of additional capacity which may cause overfitting.
                out_nodes_features += self.skip_proj(in_nodes_features).view(-1, self.num_of_heads,
                                                                             self.num_out_features)

        num_batches = out_nodes_features.shape[0]
        if self.concat:
            # shape = (B, N, NH, FOUT) -> (B, N, NH*FOUT)
            out_nodes_features = out_nodes_features.view(num_batches, -1, self.num_of_heads * self.num_out_features)
        else:
            # shape = (B, N, NH, FOUT) -> (B, N, FOUT)
            out_nodes_features = out_nodes_features.mean(dim=self.head_dim)

        if self.bias is not None:
            out_nodes_features += self.bias

        return out_nodes_features if self.activation is None else self.activation(out_nodes_features)


class GATLayerImp3(GATLayer):
    """
    Implementation #3 was inspired by PyTorch Geometric: https://github.com/rusty1s/pytorch_geometric
    But, it's hopefully much more readable! (and of similar performance)
    It's suitable for both transductive and inductive settings. In the inductive setting we just merge the graphs
    into a single graph with multiple components and this layer is agnostic to that fact! <3
    """

    src_nodes_dim = 0  # position of source nodes in edge index
    trg_nodes_dim = 1  # position of target nodes in edge index

    nodes_dim = 1  # node dimension/axis
    head_dim = 2  # attention head dimension/axis

    def __init__(self, num_in_features, num_out_features, num_of_heads, concat=True, activation=nn.ELU(),
                 dropout_prob=0.1, add_skip_connection=False, bias=True, log_attention_weights=False):
        # Delegate initialization to the base class
        super().__init__(num_in_features, num_out_features, num_of_heads, concat, activation, dropout_prob,
                         add_skip_connection, bias, log_attention_weights)

    def forward(self, data):
        #
        # Step 1: Linear Projection + regularization
        #

        in_nodes_features, edge_index, edge_mask = data  # unpack data
        num_of_nodes = in_nodes_features.shape[self.nodes_dim]
        num_of_batches = in_nodes_features.shape[0]
        assert edge_index.shape[1] == 2, f'Expected edge index with shape=(batch,2,E) got {edge_index.shape}'

        in_nodes_features = self.dropout(in_nodes_features)
        edge_offset = torch.tensor(range(num_of_batches), dtype=edge_index.dtype, device=edge_index.device). \
                                   view(-1, 1) * num_of_nodes
        edge_offset = self.explicit_broadcast(edge_offset, edge_index)
        edge_index += edge_offset
        # shape = (B, N, FIN) where N - number of nodes in the graph, FIN - number of input features per node
        # We apply the dropout to all of the input node features (as mentioned in the paper)
        # Note: for Cora features are already super sparse so it's questionable how much this actually helps

        # shape = (B, N, FIN) * (FIN, NH*FOUT) -> (B, N, NH, FOUT) where NH - number of heads, FOUT - num of output features
        # We project the input node features into NH independent output features (one for each attention head)
        nodes_features_proj = self.linear_proj(in_nodes_features).view(num_of_batches, -1, self.num_of_heads,
                                                                       self.num_out_features)

        nodes_features_proj = self.dropout(nodes_features_proj)  # in the official GAT imp they did dropout here as well

        #
        # Step 2: Edge attention calculation
        #

        # Apply the scoring function (* represents element-wise (a.k.a. Hadamard) product)
        # shape = (B, N, NH, FOUT) * (1, NH, FOUT) -> (B, N, NH, 1) -> (B, N, NH) because sum squeezes the last dimension
        # and reshape the score to (B*N, NH) for next calculation
        # Optimization note: torch.sum() is as performant as .sum() in my experiments

        scores_source = (nodes_features_proj * self.scoring_fn_source).sum(dim=-1).view(num_of_batches * num_of_nodes,
                                                                                        -1)
        scores_target = (nodes_features_proj * self.scoring_fn_target).sum(dim=-1).view(num_of_batches * num_of_nodes,
                                                                                        -1)

        nodes_features_proj = nodes_features_proj.view(num_of_batches * num_of_nodes, self.num_of_heads, -1)
        # We simply copy (lift) the scores for source/target nodes based on the edge index. Instead of preparing all
        # the possible combinations of scores we just prepare those that will actually be used and those are defined
        # by the edge index.
        # scores shape = (B*E, NH), nodes_features_proj_lifted shape = (B*E, NH, FOUT), E - number of edges in the graph
        # edge_mask shape=(B*E, NH)
        edge_index = torch.chunk(edge_index, num_of_batches, 0)
        edge_index = torch.cat(edge_index, dim=-1).squeeze()
        edge_mask = torch.chunk(edge_mask, num_of_batches, 0)
        edge_mask = torch.cat(edge_mask, dim=-1).squeeze().t()

        scores_source_lifted, scores_target_lifted, nodes_features_proj_lifted = self.lift(scores_source, scores_target,
                                                                                           nodes_features_proj,
                                                                                           edge_index)
        scores_per_edge = self.leakyReLU(scores_source_lifted + scores_target_lifted)
        # 直接相加是因为原本a * (Whi||Whj)相当于内积，就是逐个元素的相加,而且最后是对应i,j的相加
        # shape = (B*E, NH, 1)
        attentions_per_edge = self.neighborhood_aware_softmax(scores_per_edge, edge_index[self.trg_nodes_dim], edge_mask,
                                                              num_of_nodes, num_of_batches)
        # Add stochasticity to neighborhood aggregation
        attentions_per_edge = self.dropout(attentions_per_edge)

        #
        # Step 3: Neighborhood aggregation
        #

        # Element-wise (aka Hadamard) product. Operator * does the same thing as torch.mul
        # shape = (B*E, NH, FOUT) * (B*E, NH, 1) -> (B*E, NH, FOUT), 1 gets broadcast into FOUT
        nodes_features_proj_lifted_weighted = nodes_features_proj_lifted * attentions_per_edge
        # 这是相当于 α*HW

        # This part sums up weighted and projected neighborhood feature vectors for every target node
        # shape = (B*N, NH, FOUT)
        out_nodes_features = self.aggregate_neighbors(nodes_features_proj_lifted_weighted, edge_index,
                                                      in_nodes_features, num_of_nodes, num_of_batches)

        out_nodes_features = out_nodes_features.view(num_of_batches, num_of_nodes, -1, self.num_out_features)
        #
        # Step 4: Residual/skip connections, concat and bias
        #

        out_nodes_features = self.skip_concat_bias(attentions_per_edge, in_nodes_features, out_nodes_features)
        return out_nodes_features

    #
    # Helper functions (without comments there is very little code so don't be scared!)
    #

    def neighborhood_aware_softmax(self, scores_per_edge, trg_index, edge_mask, num_of_nodes, num_of_batches):
        """
        As the fn name suggest it does softmax over the neighborhoods. Example: say we have 5 nodes in a graph.
        Two of them 1, 2 are connected to node 3. If we want to calculate the representation for node 3 we should take
        into account feature vectors of 1, 2 and 3 itself. Since we have scores for edges 1-3, 2-3 and 3-3
        in scores_per_edge variable, this function will calculate attention scores like this: 1-3/(1-3+2-3+3-3)
        (where 1-3 is overloaded notation it represents the edge 1-3 and it's (exp) score) and similarly for 2-3 and 3-3
         i.e. for this neighborhood we don't care about other edge scores that include nodes 4 and 5.
        Note:
        Subtracting the max value from logits doesn't change the end result but it improves the numerical stability
        and it's a fairly common "trick" used in pretty much every deep learning framework.
        Check out this link for more details:
        https://stats.stackexchange.com/questions/338285/how-does-the-subtraction-of-the-logit-maximum-improve-learning
        """
        # Calculate the numerator. Make logits <= 0 so that e^logit <= 1 (this will improve the numerical stability)
        # scores_per_edge = scores_per_edge - scores_per_edge.max()
        edge_mask = 9e15 * (edge_mask - 1)
        edge_mask = self.explicit_broadcast(edge_mask, scores_per_edge)
        scores_per_edge = torch.where(edge_mask < 0, edge_mask, scores_per_edge)
        exp_scores_per_edge = scores_per_edge.exp()  # softmax
        # exp+scores_per_edge.shape = (B*E, NH)


        # Calculate the denominator. shape = (B*E, NH)
        neigborhood_aware_denominator = self.sum_edge_scores_neighborhood_aware(exp_scores_per_edge, trg_index,
                                                                                num_of_nodes, num_of_batches)

        # 1e-16 is theoretically not needed but is only there for numerical stability (avoid div by 0) - due to the
        # possibility of the computer rounding a very small number all the way to 0.
        attentions_per_edge = exp_scores_per_edge / (neigborhood_aware_denominator + 1e-16)

        # shape = (B*E, NH) -> (B*E, NH, 1) so that we can do element-wise multiplication with projected node features
        return attentions_per_edge.unsqueeze(-1)

    def sum_edge_scores_neighborhood_aware(self, exp_scores_per_edge, trg_index, num_of_nodes, num_of_batches):
        # The shape must be the same as in exp_scores_per_edge (required by scatter_add_) i.e. from E -> (E, NH)
        trg_index_broadcasted = self.explicit_broadcast(trg_index, exp_scores_per_edge)
        # 把trg_index原本是(B*E, )扩展到(B*E, NH)
        # shape = (B*N, NH), where N is the number of nodes and NH the number of attention heads
        size = list(exp_scores_per_edge.shape)  # convert to list otherwise assignment is not possible
        size[0] = num_of_batches * num_of_nodes
        # size原本是(B*E, NH),现在改成(B*N,NH)
        neighborhood_sums = torch.zeros(size, dtype=exp_scores_per_edge.dtype, device=exp_scores_per_edge.device)

        # position i will contain a sum of exp scores of all the nodes that point to the node i (as dictated by the
        # target index)

        neighborhood_sums.scatter_add_(self.nodes_dim - 1, trg_index_broadcasted, exp_scores_per_edge)
        # 将exp_scores_per_edge的数据按照trg_index_broadcasted加入neighborhood_sums矩阵中，其中
        # self.nodes_dim规定了加到目标矩阵的第几个维度,这里就是在第0个维度对应的是节点索引上面相加
        # self[index[i][j][k]][j][k] += other[i][j][k] # 如果 dim == 0

        # Expand again so that we can use it as a softmax denominator. e.g. node i's sum will be copied to
        # all the locations where the source nodes pointed to i (as dictated by the target index)
        # shape = (B*N, NH) -> (B*E, NH)
        return neighborhood_sums.index_select(self.nodes_dim - 1, trg_index)

    def aggregate_neighbors(self, nodes_features_proj_lifted_weighted, edge_index, in_nodes_features, num_of_nodes, num_of_batch):
        size = list(nodes_features_proj_lifted_weighted.shape)  # convert to list otherwise assignment is not possible
        size[0] = num_of_batch * num_of_nodes  # shape = (B*N, NH, FOUT)
        out_nodes_features = torch.zeros(size, dtype=in_nodes_features.dtype, device=in_nodes_features.device)

        # shape = (B*E) -> (B*E, NH, FOUT)
        trg_index_broadcasted = self.explicit_broadcast(edge_index[self.trg_nodes_dim],
                                                        nodes_features_proj_lifted_weighted)
        # aggregation step - we accumulate projected, weighted node features for all the attention heads
        # shape = (B*E, NH, FOUT) -> (B*N, NH, FOUT)
        out_nodes_features.scatter_add_(self.nodes_dim - 1, trg_index_broadcasted, nodes_features_proj_lifted_weighted)

        return out_nodes_features

    def lift(self, scores_source, scores_target, nodes_features_matrix_proj, edge_index):
        """
        Lifts i.e. duplicates certain vectors depending on the edge index.
        One of the tensor dims goes from N -> E (that's where the "lift" comes from).
        """
        src_nodes_index = edge_index[self.src_nodes_dim]
        trg_nodes_index = edge_index[self.trg_nodes_dim]

        # Using index_select is faster than "normal" indexing (scores_source[src_nodes_index]) in PyTorch!
        scores_source = scores_source.index_select(self.nodes_dim - 1, src_nodes_index)  # index_select是指定的维度选择切片
        scores_target = scores_target.index_select(self.nodes_dim - 1, trg_nodes_index)  # 选择了对应节点的分数
        nodes_features_matrix_proj_lifted = nodes_features_matrix_proj.index_select(self.nodes_dim - 1, src_nodes_index)
        # 选择源节点的特征信息
        return scores_source, scores_target, nodes_features_matrix_proj_lifted

    def explicit_broadcast(self, this, other):
        # Append singleton dimensions until this.dim() == other.dim()
        for _ in range(this.dim(), other.dim()):
            this = this.unsqueeze(-1)

        # Explicitly expand so that shapes are the same
        return this.expand_as(other)  # 把this重复到跟other一样
