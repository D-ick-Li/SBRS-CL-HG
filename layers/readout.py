import torch
import torch.nn as nn

# Applies an average on seq, of shape (batch, nodes, features)
# While taking into account the masking of msk


class AvgReadout(nn.Module):
    def __init__(self):
        super(AvgReadout, self).__init__()

    @staticmethod
    def forward(seq, msk):
        if msk is None:
            return torch.mean(seq, 1)
#            return torch.sum(seq, 1)
#             return torch.ones(1,seq.size()[2]).cuda() 
        else:
            msk = torch.unsqueeze(msk, -1)
            seq_h = torch.sum(seq * msk, 1)
            return seq_h / torch.sum(msk, 1)     # 点对点相乘
#           return torch.ones(1,seq.size()[2]).cuda()
            # return seq_h


class GRUSet2Set(torch.nn.Module):
    r"""The global pooling operator based on iterative content-based attention
    from the `"Order Matters: Sequence to sequence for sets"
    <https://arxiv.org/abs/1511.06391>`_ paper

    .. math::
        \mathbf{q}_t &= \mathrm{LSTM}(\mathbf{q}^{*}_{t-1})

        \alpha_{i,t} &= \mathrm{softmax}(\mathbf{x}_i \cdot \mathbf{q}_t)

        \mathbf{r}_t &= \sum_{i=1}^N \alpha_{i,t} \mathbf{x}_i

        \mathbf{q}^{*}_t &= \mathbf{q}_t \, \Vert \, \mathbf{r}_t,

    where :math:`\mathbf{q}^{*}_T` defines the output of the layer with twice
    the dimensionality as the input.

    Args:
        in_channels (int): Size of each input sample.
        processing_steps (int): Number of iterations :math:`T`.
        num_layers (int, optional): Number of recurrent layers, *.e.g*, setting
            :obj:`num_layers=2` would mean stacking two LSTMs together to form
            a stacked LSTM, with the second LSTM taking in outputs of the first
            LSTM and computing the final results. (default: :obj:`1`)
    """

    def __init__(self, in_channels, processing_steps, device, num_layers=1):
        super(GRUSet2Set, self).__init__()

        self.device = device
        self.in_channels = in_channels
        self.out_channels = 2 * in_channels
        self.processing_steps = processing_steps
        self.num_layers = num_layers

        self.rnn = nn.GRU(self.out_channels, self.in_channels,
                          num_layers)
        # self.linear = nn.Linear(in_channels * 3, in_channels)
        self.reset_parameters()

    def reset_parameters(self):
        for weight in self.rnn.parameters():
            if len(weight.size()) > 1:
                torch.nn.init.orthogonal_(weight.data)

    def forward(self, seq, msk):
        """"""
        seq_t = seq.permute(1, 0, 2)
        batch_size = msk.size()[0]
        h = (seq.new_zeros((self.num_layers, batch_size, self.in_channels)),
             seq.new_zeros((self.num_layers, batch_size, self.in_channels)))
        q_star = seq.new_zeros(batch_size, self.out_channels)
        msk_vet = -9e15 * torch.ones_like(msk, device=self.device)

        """
        sections = torch.bincount(msk)
        v_i = torch.split(seq, tuple(sections.cpu().numpy()))  # split whole x back into graphs G_i
        v_n_repeat = tuple(nodes[-1].view(1, -1).repeat(nodes.shape[0], 1) for nodes in
                           v_i)  # repeat |V|_i times for the last node embedding
        """
        # x = x * v_n_repeat

        for i in range(self.processing_steps):
            if i == 0:
                q, h = self.rnn(q_star.unsqueeze(0))
            else:
                q, h = self.rnn(q_star.unsqueeze(0), h)
            q = q.view(batch_size, self.in_channels)

            # e = self.linear(torch.cat((x, q[batch], torch.cat(v_n_repeat, dim=0)), dim=-1)).sum(dim=-1, keepdim=True)
            # e = (seq * q[msk]).sum(dim=-1, keepdim=True)
            e = torch.diagonal(torch.matmul(seq_t, torch.t(q)), 0, 1, 2).squeeze()
            # e = torch.diagonal(torch.matmul(seq, torch.t(q)), 0).squeeze()
            e = torch.where(msk > 0, torch.t(e), msk_vet)
            a = torch.nn.functional.softmax(e, dim=1)
            # msk = torch.unsqueeze(msk, -1)
            a = torch.unsqueeze(a, -1)
            r = torch.sum(seq * a, 1)

            # a = softmax(e, msk, num_nodes=batch_size)
            q_star = torch.cat([q, r], dim=-1)

        return q_star

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)
