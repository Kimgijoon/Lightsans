from typing import List
from collections import OrderedDict

import ray
import numpy as np
import tensorflow as tf


def _create_int_feature(values: List[int]) -> tf.train.Feature:
    """Convert data to int64 list
    Args:
        values (List[int]): data
    Returns:
        tf.train.Feature: int64list
    """    
    if type(values) is list:
        return tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
    else:
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[values]))


def _to_example(data_arr: np.ndarray) -> tf.train.Example:
    """Creates a tf.Example message ready to be written to a file
    Args:
        data_arr (np.ndarray): features
    Returns:
        tf.train.Example: features message
    """        
    user_ids, sequences, sequence_length, labels = data_arr
    tfr_features = OrderedDict()

    tfr_features['user_id'] = _create_int_feature(user_ids)
    tfr_features['sequence'] = _create_int_feature(sequences.tolist())
    tfr_features['sequence_length'] = _create_int_feature(sequence_length)
    tfr_features['label'] = _create_int_feature(labels.tolist())

    example = tf.train.Example(features=tf.train.Features(feature=tfr_features))
    return example


@ ray.remote
def save_tfrecord(uid_list, seq_list, seq_len_list, label_list, path: str):
    """Create the history data of student to a TFRecord file
    Args:
        chunk (np.ndarray): input dataset per chunk
        path (str): filename of the dataset to be saved
    """    
    with tf.io.TFRecordWriter(path) as writer:
        for row in zip(uid_list, seq_list, seq_len_list, label_list):
            example = _to_example(row)
            writer.write(example.SerializeToString())

     
def save_tfrecords(chunks: List[np.ndarray],
                   chunk_size: int,
                   output_path: str,
                   data_type: str):
    """Create the tfrecord files for a dataset using parallel
    Args:
        chunks (List[np.ndarray]): Full dataset
        chunk_size (int): The number of tfrecord dataset i.e. raw data divice to tfrecord of size N
        output_path (str): Directory path
        data_type (str): Flag that distinguishes whether the data is for train or valid
    """
    uid_list, seq_list, seq_len_list, label_list = chunks
    futures = [save_tfrecord.remote(uid_list[i],
                                    seq_list[i],
                                    seq_len_list[i],
                                    label_list[i],
                                    f'{output_path}/{data_type}_{i}.tfrecords')
                for i in range(chunk_size)]
    ray.get(futures)
