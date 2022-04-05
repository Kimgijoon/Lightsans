import pickle
from typing import Dict, Tuple, List
from collections import OrderedDict

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def load_interaction(filepath: str, max_item_list_len: int) -> Tuple:
    """Load interaction from raw dataset. Create feature dict for creating tfrecord,
    and also get information about dataset for several uses.

    Args:
        filepath (str): Raw dataset path
        max_item_list_len (int): Maximum item sequence length

    Returns:
    	Tuple: Dataset and info dictionary
    """
    # load raw dataset from filepath
    df = pd.read_csv(filepath, sep='\t')

    # load item sequence and timestamp of each user
    seq_dict = {}
    for i, uid in enumerate(df['user_id:token'].values):
        if df['rating:float'].values[i] >= 3:
            if uid in seq_dict.keys():
                seq_dict[uid].append((df['item_id:token'].values[i], df['timestamp:float'].values[i]))
            else:
                seq_dict[uid] = [(df['item_id:token'].values[i], df['timestamp:float'].values[i])]

    # sort item sequence in order of timestamp
    # if there are same timestamp for two or more items, sort them in order of item id
    for seq in seq_dict.values():
        seq.sort(key=lambda x: (x[1], x[0]))

    uid_list = []
    seq_list = []
    seq_len_list = []
    label_list = []

    # to respond to the case that sparse item ids, remap item ids to continuous indices.
    # also, save mapping information between ids and indices for inference
    # index '0' is for padding
    item2idx = {'PAD': 0}
    idx2item = {0: 'PAD'}
    count = 1

    # create sequence-label pairs from original full sequence.
    # for example, if full sequence is [1, 2, 3, 4], this process will get sequence-label pairs like
    # [1]: 2, [1, 2]: 3, [1, 2, 3]: 4
    # if max_item_list_len is 2, the last sequence-label pair will become [2, 3]: 4
    for k, v in seq_dict.items():
        full_seq = [x[0] for x in v]
        start_seq = 0
        for i, j in enumerate(full_seq):
            if j not in item2idx.keys():
                item2idx[j] = count
                idx2item[count] = j
                count += 1
            if i == 0:
                continue
            if i - start_seq > max_item_list_len:
                start_seq += 1
            uid_list.append(k)
            seq = [item2idx[x] for x in full_seq[slice(start_seq, i)]]
            seq_len = i - start_seq
            padded_seq = seq + [0] * (max_item_list_len - seq_len)
            seq_list.append(padded_seq)
            label_list.append(item2idx[j])
            seq_len_list.append(seq_len)

    one_hot_label_list = []

    # transform label into one-hot
    for label in label_list:
        one_hot_label_list.append(_one_hot(label, count))

    info = OrderedDict()
    info['total_user_num'] = len(uid_list)
    info['total_item_num'] = count
    info['item2idx'] = item2idx
    info['idx2item'] = idx2item

    result = [[int(x) for x in uid_list], seq_list, seq_len_list, one_hot_label_list]
    return result, info


def _one_hot(label_idx: int, total_num: int) -> List:
    """Create one-hot vector of label.

    Args:
        label_idx (int): index of label
        total_num (int): total number of indices in label

    Returns:
        List: one-hot vector
    """
    result = [0] * total_num
    result[label_idx] = 1

    return result


def split_data(data: List, split_size: float) -> Tuple:
    """Split arrays or matrices into random train and test subsets

    Args:
        data (List): Process data
        split_size (float): The proportion of the dataset to include in the train split

    Returns:
        List: List containing train-test split of inputs
    """
    trainset, testset = [], []
    for i in data:
        x, y = train_test_split(i, test_size=split_size)
        trainset.append(x)
        testset.append(y)

    return trainset, testset


def chunk_data(data: List, chunk_size: int) -> Tuple:
    """Split an array into multiple sub-arrays

    Args:
        data (int): Process data
        chunk_size (int): Split size by integer N

    Returns:
        Tuple: Chunk data
    """
    uid_list, seq_list, seq_len_list, label_list = data
    uid_list = np.array_split(uid_list, chunk_size)
    seq_list = np.array_split(seq_list, chunk_size)
    seq_len_list = np.array_split(seq_len_list, chunk_size)
    label_list = np.array_split(label_list, chunk_size)

    return uid_list, seq_list, seq_len_list, label_list


if __name__ == "__main__":
    
    filepath = 'data/raw/ml-100k.inter'
    data, dataset_info = load_interaction(filepath, 200)
    
    valid_ratio, test_ratio = 0.2, 0.1
    trainset, testset = split_data(data, test_ratio)
    trainset, validset = split_data(trainset, valid_ratio)
    
    chunk_size = 10
    trainset = chunk_data(trainset, chunk_size)
    validset = chunk_data(validset, chunk_size)