import numpy as np
import torch
import torch.nn as nn
import argparse
import random
from models import DGI, HGCL
from utils import process
from tmall_minibatch import MinibatchIterator
import time


def in_top_k(targets, preds, k):
    """
    The metric on Precision@k
    :param targets: ground truth labels
    :param preds: predicted labels
    :param k: parameter of top k selection
    :return: the precision ratio for the current batch
    """
    topk = preds.topk(k)[1]
    # result = torch.BoolTensor(targets.unsqueeze(1) == topk).any(dim=1)
    result = (targets.unsqueeze(1) == topk).any(dim=1)
    result = result.cpu().detach().numpy()
    result = result.astype(np.int)
    return result.sum()


def _ndcg(targets, preds):
    """
    The metric of NDCG
    """
    _, pred_ind = torch.sort(preds, dim=1, descending=True)
    rank_pose = torch.where(targets.unsqueeze(dim=1) == pred_ind)[1] + 2
    rank_pose = rank_pose.float()
    ndcg_val = 1 / torch.log2(rank_pose)
    return ndcg_val.sum().cpu().detach().numpy()


def _mrr(targets, preds, k):
    """
    The metric of MRR@k
    """
    topk = preds.topk(k)[1]
    # result = torch.BoolTensor(targets.unsqueeze(1) == topk).any(dim=1)
    result = (targets.unsqueeze(1) == topk).any(dim=1)
    _, pred_ind = torch.sort(preds, dim=1, descending=True)
    # rank_pose = torch.where(torch.BoolTensor(targets.unsqueeze(dim=1) == pred_ind, device=targets.device))[1] + 1
    rank_pose = torch.where(targets.unsqueeze(dim=1) == pred_ind)[1] + 1
    rank_pose = rank_pose.float()
    ndcg_val = 1 / rank_pose
    zero_vec = torch.zeros(ndcg_val.shape, device=targets.device)
    ndcg_val = torch.where(result, ndcg_val, zero_vec)
    # be caution of inf
    return ndcg_val.sum().cpu().detach().numpy()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training.')
    parser.add_argument('--fastmode', action='store_true', default=False, help='Validate during training pass.')
    parser.add_argument('--sparse', action='store_true', default=False, help='GAT with sparse version or not.')
    parser.add_argument('--seed', type=int, default=72, help='Random seed.')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train.')
    parser.add_argument('--lr', type=float, default=0.0005, help='Initial learning rate.')
    parser.add_argument('--weight_decay', type=float, default=0.0, help='Weight decay (L2 loss on parameters).')
    parser.add_argument('--hidden', type=int, default=128, help='Number of hidden units.')
    parser.add_argument('--shid', type=int, default=32, help='Number of semantic level hidden units.')
    parser.add_argument('--out', type=int, default=32, help='Number of output feature dimension.')
    parser.add_argument('--nb_heads', type=int, default=4, help='Number of head attentions.')
    parser.add_argument('--dropout', type=float, default=0.6, help='Dropout rate (1 - keep probability).')
    parser.add_argument('--alpha', type=float, default=0.2, help='Alpha for the leaky_relu.')
    parser.add_argument('--patience', type=int, default=15, help='Patience')
    parser.add_argument('--embed_size', type=int, default=128, help='The initial embedding size')
    parser.add_argument('--drop_percent', type=float, default=0.1, help='drop percent')
    parser.add_argument('--aug_type', type=str, default="subgraph")
    parser.add_argument('--top_k', type=int, default=20)
    args = parser.parse_args()
    adjs, adjs_test, last_sessions, num_list, train_df, val_df, test_df = process.load_data(path="./Tmall/processed/", dataset='Tmall')

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

    # finetuning params
    patience = args.patience
    batch_size = 128
    top_k = args.top_k
    lr = args.lr
    hid_units = args.hidden
    shid = args.shid
    nout = hid_units * args.nb_heads
    exp = 'tmall_hcgl_pre_50'
    drop_percent = args.drop_percent
    nb_users = num_list[1]
    ft_size = args.embed_size
    nb_business = num_list[0] + 1
    xent = nn.CrossEntropyLoss()

    P = int(3)  # number of meta-path type

    minibatch = MinibatchIterator(adjs,
                                  adjs_test=adjs_test,
                                  latest_sessions=last_sessions,
                                  data_list=[train_df, val_df, test_df],
                                  batch_size=batch_size,
                                  max_degree=50,
                                  num_nodes=num_list,
                                  max_length=180)

    recall_all = []
    ndcg_all = []
    test_loss_all = []

    time_start = time.time()
    for _ in range(1):
        bad_counter = 0
        best = 1e9
        loss_tr = []
        loss_val = []
        loss_test = []
        recalls = []
        ndcg_list = []
        mrr_list = []
        best_epoch = 0
        count = 0
        model = HGCL(ft_size, hid_units, shid, args.alpha, args.nb_heads, P, nb_business, device)

        model.load_state_dict(torch.load(
            str('best_dgi_head_' + str(args.nb_heads) + '_nhidden_' + str(args.hidden) + '_exp_' + str(exp) + '.pkl')))

        opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
        if args.cuda:
            model.cuda()

        minibatch.shuffle()
        for epoch in range(10):
            loss_temp = 0
            loss_temp_val = 0
            model.train()
            for i in range(P):
                model.hencoder.hencoder.node_level_attentions[i].train()
            while minibatch.end():
                sess_mask, sess_adjs, sess_edge_mask, sess_item, sess_targets = minibatch.next_train_minibatch_feed_dict()
                sess_mask = torch.LongTensor(sess_mask)
                sess_item = torch.LongTensor(sess_item)
                sess_adjs = torch.LongTensor(sess_adjs)
                sess_edge_mask = torch.LongTensor(sess_edge_mask)
                sess_targets = torch.LongTensor(sess_targets)
                if args.cuda:
                    sess_mask = sess_mask.cuda()
                    sess_item = sess_item.cuda()
                    sess_adjs = sess_adjs.cuda()
                    sess_edge_mask = sess_edge_mask.cuda()
                    sess_targets = sess_targets.cuda()
                model.train()
                opt.zero_grad()

                scores = model.ft_forward(sess_item, sess_adjs, sess_mask, sess_edge_mask)
                loss = xent(scores, sess_targets)

            #        print("train_loss: "+ str(loss) +"  "+"val_loss: "+ str(loss_val) )
                loss.backward()
                opt.step()
                loss_temp += loss.cpu().detach().numpy()
            loss_tr.append(loss_temp)
            print('The train loss:', loss_temp)
            if args.cuda:
                torch.cuda.empty_cache()
            model.eval()
            for i in range(P):
                model.hencoder.hencoder.node_level_attentions[i].eval()
            while minibatch.end_val('val'):
                with torch.no_grad():
                    sess_mask, sess_adjs, sess_edge_mask, sess_item, sess_targets = minibatch.next_val_minibatch_feed_dict('val')
                    sess_mask = torch.LongTensor(sess_mask)
                    sess_item = torch.LongTensor(sess_item)
                    sess_adjs = torch.LongTensor(sess_adjs)
                    sess_edge_mask = torch.LongTensor(sess_edge_mask)
                    sess_targets = torch.LongTensor(sess_targets)
                    if args.cuda:
                        sess_mask = sess_mask.cuda()
                        sess_item = sess_item.cuda()
                        sess_adjs = sess_adjs.cuda()
                        sess_edge_mask = sess_edge_mask.cuda()
                        sess_targets = sess_targets.cuda()

                    scores = model.ft_forward(sess_item, sess_adjs, sess_mask, sess_edge_mask)
                    loss = xent(scores, sess_targets)
                    loss_temp_val += loss.cpu().detach().numpy()
            loss_val.append(loss_temp_val)
            print('The validation loss of epoch {}: {}'.format(count, loss_temp_val))

            if loss_val[-1] < best:
                best = loss_val[-1]
                best_epoch = epoch
                bad_counter = 0
                torch.save(model.state_dict(), 'current_best_mlp_exp{}.pkl'.format(exp))
            else:
                bad_counter += 1

            if bad_counter == patience:
                # patience for early stopping
                break
            count += 1
        time_end = time.time()
        print("finish the time:{}".format(time_end - time_start))
        # Restore best model

        print('Loading {}th epoch'.format(best_epoch))
        model.load_state_dict(torch.load('current_best_mlp_exp{}.pkl'.format(exp)))
        loss_temp_test = 0
        if args.cuda:
            torch.cuda.empty_cache()
        model.eval()
        for i in range(P):
            model.hencoder.hencoder.node_level_attentions[i].eval()
        while minibatch.end_val('test'):

            sess_mask, sess_adjs, sess_edge_mask, sess_item, sess_targets = minibatch.next_val_minibatch_feed_dict('test')
            sess_mask = torch.LongTensor(sess_mask)
            sess_item = torch.LongTensor(sess_item)
            sess_adjs = torch.LongTensor(sess_adjs)
            sess_edge_mask = torch.LongTensor(sess_edge_mask)
            sess_targets = torch.LongTensor(sess_targets)
            with torch.no_grad():
                if args.cuda:
                    sess_mask = sess_mask.cuda()
                    sess_item = sess_item.cuda()
                    sess_adjs = sess_adjs.cuda()
                    sess_edge_mask = sess_edge_mask.cuda()
                    sess_targets = sess_targets.cuda()

                scores = model.ft_forward(sess_item, sess_adjs, sess_mask, sess_edge_mask)
                loss = xent(scores, sess_targets)
                # sess_targets = sess_targets.cpu()
                # scores = scores.cpu()
                ndcg_list.append(_ndcg(sess_targets, scores))
                mrr_list.append(_mrr(sess_targets, scores, k=top_k))
                loss_temp_test += loss.cpu().detach().numpy()
                recalls.append(in_top_k(sess_targets, scores, top_k))

            # scores = scores.cpu().detach.numpy()

        loss_test.append(loss_temp_test)
        print('The recall@20:', np.array(recalls).sum()/len(minibatch.test_keys))
        print('The MRR:', np.array(mrr_list).sum() / len(minibatch.test_keys))
        print('The NDCG:', np.array(ndcg_list).sum()/len(minibatch.test_keys))
        print('The test loss:{}'.format(loss_temp_test))
        recall_all.append(np.array(recalls).sum()/len(minibatch.test_keys))
        ndcg_all.append(np.array(ndcg_list).sum()/len(minibatch.test_keys))
        test_loss_all.append(loss_temp_test)

    recall_all = np.array(recall_all)
    ndcg_all = np.array(ndcg_all)
    test_loss_all = np.array(test_loss_all)
    np.save('result/recall.npy', recall_all)
    np.save('result/ndcg.npy', ndcg_all)
    np.save('result/test_loss.npy', test_loss_all)

