import pandas as pd
import torch


class LoadSeq(object):
    def __init__(self, path, dataset='Yelp'):
        self.dataset = dataset
        self.path = path
        self.train, self.valid, self.test = self.load_data()

    def load_data(self):
        if self.dataset == "Yelp":
            train = pd.read_csv(self.path + '/train.csv', sep=',',
                                dtype={0: str, 1: str, 2: str, 3: str, 4: str, 5: str, 6: str,
                                       7: str})
            valid = pd.read_csv(self.path + '/valid.csv', sep=',', dtype={0: str, 1: str, 2: str, 3: str, 4: str, 5: str,
                                                                     6: str, 7: str})
            test = pd.read_csv(self.path + '/test.csv', sep=',',
                               dtype={0: str, 1: str, 2: str, 3: str, 4: str, 5: str, 6: str,
                                      7: str})

        elif self.dataset == "Tmall":

            train = pd.read_csv(self.path + '/train.csv', sep=',', dtype={0: str, 1: int, 2: int, 3: int})
            valid = pd.read_csv(self.path + '/valid.csv', sep=',', dtype={0: str, 1: int, 2: int, 3: int})
            test = pd.read_csv(self.path + '/test.csv', sep=',', dtype={0: str, 1: int, 2: int, 3: int})

        else:
            raise ValueError('dataset not find error')

        return train, valid, test

    def padding_sessions(self, raw_data):
        """
        transform the dataframe to the session list
        """
        if self.dataset == 'Yelp':
            raw_data = raw_data.sort_values(by=['time_stamp']).groupby('session_id')['business_id'].apply(
                list).to_dict()  # 同一个session内按照时间排序,，最后得到一个根据session_id的dict
            #new_data = {}
            new_data = []
            for k, v in raw_data.items():
                sess = list(map(int, v))  # 这里要把session节点的str类型变为int
                if len(sess) <= 1:
                    continue
                # out_seqs = []
                # labs = []
                for i in range(1, len(sess)):
                    tar = sess[-i]
                    # labs += [tar]
                    # out_seqs += [sess[:-i]]
                    key_new = k + '_' + str(i)
                    #new_data[key_new] = [sess[:-i], [tar]]
                    new_data.append([sess[:-i], [tar]])
            return new_data

        elif self.dataset == 'Tmall':
            raw_data = raw_data.sort_values(by=['time_stamp']).groupby('session_id')['item_id'].apply(
                list).to_dict()  # 同一个session内按照时间排序,，最后得到一个根据session_id的dict
            #new_data = {}
            new_data = []
            for k, v in raw_data.items():
                sess = list(map(int, v))  # 这里要把session节点的str类型变为int
                if len(sess) <= 1:
                    continue
                # out_seqs = []
                # labs = []
                for i in range(1, len(sess)):
                    tar = sess[-i]
                    # labs += [tar]
                    # out_seqs += [sess[:-i]]
                    key_new = k + '_' + str(i)
                    #new_data[key_new] = [sess[:-i], [tar]]
                    new_data.append([sess[:-i], [tar]])
            return new_data


def in_top_k(targets, preds, k):
    topk = preds.topk(k)[1]
    # result = torch.BoolTensor(targets.unsqueeze(1) == topk).any(dim=1)
    result = (targets.unsqueeze(1) == topk).any(dim=1)
    result = result.cpu().detach().numpy()
    result = result.astype(np.int)
    return result.sum()

def _ndcg(targets, preds):
    # NDCG
    _, pred_ind = torch.sort(preds, dim=1, descending=True)
    rank_pose = torch.where(targets.unsqueeze(dim=1) == pred_ind)[1] + 2
    rank_pose = rank_pose.float()
    ndcg_val = 1 / torch.log2(rank_pose)
    return ndcg_val.sum().cpu().detach().numpy()

def _mrr(targets, preds, k):
    # NDCG
    # _, pred_ind = torch.sort(preds, dim=1, descending=True)
    # rank_pose = torch.where(torch.BoolTensor(targets.unsqueeze(dim=1) == pred_ind))[1] + 2
    # rank_pose = rank_pose.float()
    # ndcg_val = 1 / torch.log2(rank_pose)
    # MRR
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