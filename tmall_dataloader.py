from mini_batch import MinibatchIterator
from torch.utils.data import IterableDataset, DataLoader
from utils import process


class MyIterableDataset(IterableDataset):
    def __init__(self, datapath='Yelp/processed/', dataset_name='Yelp'):
        data = process.load_data(datapath, dataset_name)
        adj_info = data[0]
        adj_test = data[1]
        latest_per_user_by_time = data[2]
        num_list = data[3]
        train_df = data[4]
        valid_df = data[5]
        test_df = data[6]
        self.mini_batch = MinibatchIterator(adj_info,
                                            adjs_test=adj_test,
                                            latest_sessions=latest_per_user_by_time,
                                            data_list=[train_df, valid_df, test_df],
                                            batch_size=16,
                                            max_degree=50,
                                            num_nodes=num_list,
                                            max_length=120)

    def __iter__(self, mode):
        if mode != 'train':
            sess_alias, sess_adjs, edge_mask, sess_item, sess_targets = self.mini_batch.next_val_minibatch_feed_dict(mode)
        else:
            sess_alias, sess_adjs, edge_mask, sess_item, sess_targets = self.mini_batch.next_train_minibatch_feed_dict()
        return sess_alias, sess_adjs, edge_mask, sess_item, sess_targets


train_loader2 = DataLoader(dataset=MyIterableDataset('Yelp/processed/', 'Yelp'),
                          batch_size=32,
                          shuffle=True)


for epoch in range(2):
    for i, data in enumerate(train_loader2):
        # 将数据从 train_loader 中读出来,一次读取的样本数是32个
        sess_alias, sess_adjs, edge_mask, sess_item, sess_targets = data

        # 将这些数据转换成Variable类型
        print('batch {}'.format(i))

