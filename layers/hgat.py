import torch
import torch.nn as nn
from layers import NodeAttentionLayer, SemanticAttentionLayer


class HGAT(nn.Module):
    def __init__(self, nfeat, nhid, shid, alpha, nheads, mp_num, device):
        """Dense version of GAT."""
        super(HGAT, self).__init__()
        self.device = device
        self.node_level_attentions = []
        self.P = mp_num  # the number of meta-path
        for _ in range(mp_num):
            self.node_level_attentions.append([NodeAttentionLayer(nfeat, nhid, alpha=alpha, concat=True, device=device) for _ in range(nheads)])
            # 每种meta-path的节点聚合都有nhead个attention

        for i, node_attentions_path in enumerate(self.node_level_attentions):
            for j, node_attention in enumerate(node_attentions_path):
                self.add_module('attention_path_{}_head_{}'.format(i, j), node_attention)

        self.semantic_level_attention = SemanticAttentionLayer(nhid*nheads, shid)
        
    def forward(self, x, adjs):
        adjs = adjs.permute(1, 0, 2, 3)
        x = torch.squeeze(x, 0)
        # print(x.size())
        meta_path_x = []
        for i, adj in enumerate(adjs):
            adj = torch.squeeze(adj, 0)
            m_x = torch.cat([att(x, adj) for att in self.node_level_attentions[i]], dim=2)
            # print(m_x.size())
            meta_path_x.append(m_x)

        meta_path_x = torch.stack(meta_path_x, dim=0)
        meta_path_x = meta_path_x.permute(1, 0, 2, 3)
        x_per_sess = []
        for m_x in meta_path_x:
            sess_x = torch.cat([h_x for h_x in m_x], dim=0)
            x_per_sess.append(sess_x)
        x = torch.stack(x_per_sess, dim=0)
        # x = torch.cat([m_x for m_x in meta_path_x], dim=0)
        # print(x.size())
        x = self.semantic_level_attention(x, self.P)    # 最后语义层面聚合输入的维度有[P*input*head,embed_dim]
        
        # x = torch.unsqueeze(x, 0)
        return x
