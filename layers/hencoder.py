import torch
import torch.nn as nn
from layers import NodeAttentionLayer, SemanticAttentionLayer, GRUSet2Set, AvgReadout, GATLayerImp3


class HEncoder(nn.Module):
    def __init__(self, nfeat, nhid, shid, alpha, nheads, mp_num, device):
        """Dense version of GAT and semantic level aggregation(soft attention)"""
        super(HEncoder, self).__init__()
        self.device = device
        self.node_level_attentions = []
        self.P = mp_num  # the number of meta-path
        self.read = AvgReadout()

        for _ in range(mp_num):
            self.node_level_attentions.append(GATLayerImp3(nfeat, nhid, nheads).to(device))

        self.semantic_level_attention = SemanticAttentionLayer(nhid * nheads, shid)

    def forward(self, x, adjs, msk, edge_msk):
        # print(x.size())
        meta_path_x = []
        for i, adj in enumerate(adjs):
            adj = torch.squeeze(adj, 0)
            m_x = self.node_level_attentions[i]((x, adj, edge_msk[i]))
            meta_path_x.append(m_x)

        meta_path_x = torch.stack(meta_path_x, dim=0)
        meta_path_x = meta_path_x.permute(1, 0, 2, 3)

        return meta_path_x


class HAggregate(nn.Module):
    def __init__(self, nfeat, nhid, shid, alpha, nheads, mp_num, device):
        super(HAggregate, self).__init__()
        self.device = device
        self.node_level_attentions = []
        self.P = mp_num  # the number of meta-path
        self.add_pool = AvgReadout()    # graph pooling
        self.read = GRUSet2Set(nheads * nhid, 10, device)   # item order reconstruction in fine-tuning
        self.hencoder = HEncoder(nfeat, nhid, shid, alpha, nheads, mp_num, device)
        self.semantic_level_attention = SemanticAttentionLayer(nhid * nheads, shid)
        self.fc = nn.Linear(2 * nhid * nheads, nhid * nheads)
        # fully connected layer for graph embedding in fine-tuning, similar to projection head but with different parameters

    def forward(self, x, adjs, msk, edge_msk):
        """
        The pre-training forward procedure
        :param x: input embeddings(node embeddings)
        :param adjs: meta-path adjacent matrix
        :param msk: mask on node information
        :param edge_msk: mask on edge information
        :return: aggregated graph embedding
        """
        adjs = adjs.permute(1, 0, 2, 3)
        msk = msk.permute(1, 0, 2)
        edge_msk = edge_msk.permute(1, 0, 2)
        x = torch.squeeze(x, 0)
        meta_path_x = self.hencoder(x, adjs, msk, edge_msk)
        graph_embed = []
        meta_path_x = meta_path_x.permute(1, 0, 2, 3)
        for i, adj in enumerate(adjs):
            m_x = self.add_pool(meta_path_x[i], msk[i])
            graph_embed.append(m_x)
        graph_embed = torch.stack(graph_embed, dim=0)
        graph_embed = graph_embed.permute(1, 0, 2)

        # x = torch.cat([m_x for m_x in meta_path_x], dim=0)
        # print(x.size())
        x = self.semantic_level_attention(graph_embed, self.P)  # 最后语义层面聚合输入的维度有[P*input*head,embed_dim]

        # x = torch.unsqueeze(x, 0)
        return x

    def ft_forward(self, x, adjs, msk, edge_msk):
        """
        The fine-tuning procedure. The parameters refers to forward function
        """
        # with torch.no_grad():
        adjs = adjs.permute(1, 0, 2, 3)
        msk = msk.permute(1, 0, 2)
        edge_msk = edge_msk.permute(1, 0, 2)
        x = torch.squeeze(x, 0)
        meta_path_x = self.hencoder(x, adjs, msk, edge_msk)
        graph_embed = []
        meta_path_x = meta_path_x.permute(1, 0, 2, 3)
        for i, adj in enumerate(adjs):
            m_x = self.read(meta_path_x[i], msk[i])
            graph_embed.append(m_x)
        graph_embed = torch.stack(graph_embed, dim=0)
        graph_embed = graph_embed.permute(1, 0, 2)
        graph_embed = self.fc(graph_embed)
        graph_embed = self.semantic_level_attention(graph_embed, self.P)
        return graph_embed


