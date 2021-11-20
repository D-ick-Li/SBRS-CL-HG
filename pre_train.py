import numpy as np
import torch
import torch.nn as nn
import argparse
import time
import random
from models import DGI, HGCL
from utils import process
from tmall_minibatch import MinibatchIterator
import augmentation


parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False, help='Validate during training pass.')
parser.add_argument('--sparse', action='store_true', default=False, help='GAT with sparse version or not.')
parser.add_argument('--seed', type=int, default=72, help='Random seed.')
parser.add_argument('--epochs', type=int, default=50, help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.001, help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=128, help='Number of hidden units.')
parser.add_argument('--shid', type=int, default=32, help='Number of semantic level hidden units.')
parser.add_argument('--out', type=int, default=32, help='Number of output feature dimension.')
parser.add_argument('--nb_heads', type=int, default=4, help='Number of head attentions.')
parser.add_argument('--dropout', type=float, default=0.6, help='Dropout rate (1 - keep probability).')
parser.add_argument('--alpha', type=float, default=0.2, help='Alpha for the leaky_relu.')
parser.add_argument('--patience', type=int, default=30, help='Patience')
parser.add_argument('--embed_size', type=int, default=128, help='The initial embedding size')
parser.add_argument('--drop_percent',     type=float, default=0.15, help='drop percent')
parser.add_argument('--aug_type',     type=str,         default="subgraph_nodemask", help='the data augmentation type like subgraph/edge perturbation')

args = parser.parse_args()

args.cuda = not args.no_cuda and torch.cuda.is_available()

if torch.cuda.is_available():
    device = torch.device("cuda")
    torch.cuda.set_device(0)
else:
    device = torch.device("cpu")

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)

if args.cuda:
    torch.cuda.manual_seed(args.seed)

# training params
batch_size = 128
nb_epochs = args.epochs
patience = args.patience
lr = args.lr
l2_coef = args.weight_decay
drop_prob = args.dropout
hid_units = args.hidden
shid = args.shid
sparse = args.sparse
nout = hid_units * args.nb_heads
aug_type = args.aug_type
exp = 'tmall_hcgl_pre_50'       # This variable consists the name of the result file.
drop_percent = args.drop_percent

adjs, adjs_test, last_sessions, num_list, train_df, val_df, test_df = process.load_data(path="Tmall/processed/", dataset="Tmall")

nb_users = num_list[1]
ft_size = args.embed_size
nb_business = num_list[0] + 1

P = int(3)
# The number of the meta-path type

minibatch = MinibatchIterator(adjs,
                              adjs_test=adjs_test,
                              latest_sessions=last_sessions,
                              data_list=[train_df, val_df, test_df],
                              batch_size=batch_size,
                              max_degree=50,
                              num_nodes=num_list,
                              max_length=180)

#
# if sparse:
#    sp_adj = process.sparse_mx_to_torch_sparse_tensor(adj)
# else:
#    adj = (adj + sp.eye(adj.shape[0])).todense()

# features = torch.FloatTensor(features[np.newaxis])
# features = torch.FloatTensor(features)
# if not sparse:
# adj = torch.FloatTensor(adj)
# labels = torch.FloatTensor(labels[np.newaxis])

model = HGCL(ft_size, hid_units, shid, args.alpha, args.nb_heads, P, nb_business, device)


optimiser = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=l2_coef)
b_xent = nn.BCEWithLogitsLoss()     # be used for contrastive loss
xent = nn.CrossEntropyLoss()        # crossEntropyLoss输入的y可以为1维的序列,而且函数里面集成了softmax
cnt_wait = 0
best = 1e9
best_t = 0
pretrain_loss = []
time_start = time.time()
if args.cuda:
    model.cuda()


for epoch in range(nb_epochs):
    loss_batch = 0
    while minibatch.end():
        sess_mask, sess_adjs, sess_edge_mask,  sess_item, sess_targets = minibatch.next_train_minibatch_feed_dict()
        # print('node_num:', sess_mask.shape[1])
        if aug_type == "subgraph":
            # The proposed adaptive subgraph sampling
            minibatch.batch_num -= 1
            masks_aug1, adjs_aug1, edge_mask_aug1, items_aug1, _ = minibatch.next_train_minibatch_feed_dict()
        elif aug_type == "node_mask":
            items_aug1 = augmentation.aug_random_mask(sess_mask, sess_item, drop_percent)
            masks_aug1 = sess_mask
            adjs_aug1 = sess_adjs
            edge_mask_aug1 = sess_edge_mask

        elif aug_type == "drop_edge":
            # The adaptive edge dropping
            adjs_aug1 = augmentation.adaptive_drop_edge(sess_adjs, sess_edge_mask)
            edge_mask_aug1 = sess_edge_mask
            items_aug1 = sess_item
            masks_aug1 = sess_mask
        elif aug_type == "subgraph_drop_edge":
            # The combination of subgraoh sampling and edge dropping
            minibatch.batch_num -= 1
            masks_aug1, adjs_aug1, edge_mask_aug1, items_aug1, _ = minibatch.next_train_minibatch_feed_dict()
            adjs_aug1 = augmentation.adaptive_drop_edge(adjs_aug1, edge_mask_aug1)
        elif aug_type == "subgraph_nodemask":
            # The combination of subgraph sampling and node masking
            minibatch.batch_num -= 1
            masks_aug1, adjs_aug1, edge_mask_aug1, items_aug1, _ = minibatch.next_train_minibatch_feed_dict()
            items_aug1 = augmentation.aug_random_mask(masks_aug1, items_aug1, drop_percent)
        else:
            assert False
        """
        shuffle_items = process.shuffle_nodes(sess_item, sess_mask)     # 这里打乱节点创造负样本
        
        nb_items = sess_mask.shape[1]
        cur_size = sess_mask.shape[0]
        lbl_1 = torch.ones(cur_size, nb_items)
        lbl_2 = torch.zeros(cur_size, nb_items)
        lbl = torch.cat((lbl_1, lbl_2), 1)
        """
        if args.cuda:
            sess_mask = torch.cuda.LongTensor(sess_mask)
            sess_item = torch.cuda.LongTensor(sess_item)
            sess_adjs = torch.cuda.LongTensor(sess_adjs)
            sess_edge_mask = torch.cuda.LongTensor(sess_edge_mask)
            masks_aug1 = torch.cuda.LongTensor(masks_aug1)
            items_aug1 = torch.cuda.LongTensor(items_aug1)
            adjs_aug1 = torch.cuda.LongTensor(adjs_aug1)
            edge_mask_aug1 = torch.cuda.LongTensor(edge_mask_aug1)

        else:
            sess_mask = torch.LongTensor(sess_mask)
            sess_item = torch.LongTensor(sess_item)
            sess_adjs = torch.FloatTensor(sess_adjs)
            sess_edge_mask = torch.LongTensor(sess_edge_mask)
            masks_aug1 = torch.LongTensor(masks_aug1)
            items_aug1 = torch.LongTensor(items_aug1)
            adjs_aug1 = torch.FloatTensor(adjs_aug1)
            edge_mask_aug1 = torch.LongTensor(edge_mask_aug1)

        model.train()
        optimiser.zero_grad()

        logits_aug, logits = model(sess_item, items_aug1, sess_adjs, sess_edge_mask, adjs_aug1, edge_mask_aug1,
                                   sess_mask, masks_aug1, aug_type)

        loss = model.loss_cal(logits, logits_aug)

        loss.backward()
        optimiser.step()
        loss_batch += loss.cpu().detach().numpy()
    if args.cuda:
        torch.cuda.empty_cache()
    pretrain_loss.append(loss_batch)
    print('Epoch:', epoch, 'Loss:', loss_batch)
    """
    In the pre-training stage, we write the parameters of lowest NCE loss into files
    """
    if loss_batch < best:
        best = loss_batch
        best_t = epoch
        cnt_wait = 0
        torch.save(model.state_dict(), str(
            'best_dgi_head_' + str(args.nb_heads) + '_nhidden_' + str(args.hidden) + '_exp_' + str(exp) + '.pkl'))
    else:
        cnt_wait += 1
        print("wait: " + str(cnt_wait))
    if cnt_wait == patience:
        print('Early stopping!')
        break

time_end = time.time()
print("finish the time:{}".format(time_end - time_start))
