import torch
import torch.nn as nn
from layers import GRUSet2Set, HGAT, AvgReadout, Discriminator


class DGI(nn.Module):
    def __init__(self, nfeat, nhid, shid, alpha, nheads, mp_num, nb_node, device):
        super(DGI, self).__init__()
        self.T = 0.2
        self.embedding = nn.Embedding(nb_node, nfeat)
        self.hgat = HGAT(nfeat, nhid, shid, alpha, nheads, mp_num, device)
        # self.read = AvgReadout()
        self.read = GRUSet2Set(nheads * nhid, 10, device)
        self.sum_pool = AvgReadout()
        self.sigm = nn.Sigmoid()
        self.relu = nn.ReLU()
        # self.disc = Discriminator(nhid*nheads)
        self.fc = nn.Linear(2*nhid*nheads, nfeat)
        self.proj_head = nn.Sequential(nn.Linear(nhid * nheads, nhid * nheads), nn.ReLU(inplace=True),
                                       nn.Linear(nhid * nheads, nhid * nheads))
        # self.b_xent = nn.BCEWithLogitsLoss(reduce=False)

    def forward(self, items, items_aug, adjs, aug_adjs, msk, msk_aug, aug_type="subgraph"):
    # def forward(self, items, items_ne, items_aug, adjs, aug_adjs, msk, msk_aug, aug_type="subgraph"):
        seq1 = self.embedding(items)
        # seq2 = self.embedding(items_ne)
        seq3 = self.embedding(items_aug)

        if aug_type == "subgraph" or aug_type == "drop_node" or aug_type == "subgraph_nodemask":

            h_1 = self.hgat(seq3, aug_adjs)
            c = self.sum_pool(h_1, msk_aug)
            # c = self.sigm(c)

        elif aug_type == "random_edge":
            h_1 = self.hgat(seq1, aug_adjs)
            c = self.sum_pool(h_1, msk)
            # c = self.sigm(c)

        elif aug_type == "node_mask":
            h_1 = self.hgat(seq3, adjs)
            c = self.sum_pool(h_1, msk_aug)
            # c = self.sigm(c)

        else:
            assert False

        h_0 = self.hgat(seq1, adjs)
        # h_2 = self.hgat(seq2, adjs)

        # ret = self.disc(c, h_0, h_2, None, None)
        c_0 = self.sum_pool(h_0, msk)
        c_0 = self.sigm(c_0)

        c = self.proj_head(c)
        c_0 = self.proj_head(c_0)

        return c, c_0
        # return ret

    # freeze the encoder variables
    def embed(self, items, adjs, msk):
        with torch.no_grad():
            seq = self.embedding(items)
            h_1 = self.hgat(seq, adjs)
        c = self.read(h_1, msk)
        # c = self.proj_head(c)
        return h_1, c

    def loss_cal(self, c, c_aug):
        batch_size = c.size()[0]
        x_abs = c.norm(dim=1)
        x_aug_abs = c_aug.norm(dim=1)
        sim_matrix = torch.einsum('ik,jk->ij', c, c_aug) / torch.einsum('i,j->ij', x_abs, x_aug_abs)
        # 这里矩阵先做乘法再进行正则化
        sim_matrix = torch.exp(sim_matrix / self.T)
        pos_sim = sim_matrix[range(batch_size), range(batch_size)]
        loss = pos_sim / (sim_matrix.sum(dim=1))
        loss = - torch.log(loss).mean()
        return loss

    def ft_forward(self, items, adjs, msk):
        _, c = self.embed(items, adjs, msk)
        s = self.fc(c)
        scores = torch.matmul(s, self.embedding.weight.transpose(1, 0))
        return scores

    def bce_loss(self, logits, labels, msk):
        loss = self.b_xent(logits, labels)
        loss = loss * msk.repeat(1, 2)
        return loss.mean()
