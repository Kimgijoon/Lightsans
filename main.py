import json
import pickle
from datetime import datetime

import numpy as np
import tensorflow as tf

from utils.create_tfrecord import save_tfrecords
from src.recommender import SequentialRecommender
from utils.data_util import load_interaction, split_data, chunk_data, get_features_from_tfrecord


flags = tf.compat.v1.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('op', 'train', '[REQUIRED] Operation code to do')
flags.mark_flag_as_required('op')

flags.DEFINE_string('data_dir', 'data', 'Path to input directory')
flags.DEFINE_string('config_dir', 'configs', 'Directory of config file')
flags.DEFINE_string('checkpoint_dir', 'checkpoints', 'Directory of model checkpoints')
flags.DEFINE_string('logs_dir', 'logs', 'Directory with tensorboard log')
flags.DEFINE_string('dataset', 'ml-100k.inter', 'Raw data filename')
flags.DEFINE_integer('chunk_size', 10, 'The number of tfrecord dataset i.e. raw data divice to tfrecord of size N')
flags.DEFINE_string('split_ratio', '0.2, 0.1', 'Valid, Test ratio in dataset')
flags.DEFINE_integer('batch_size', 512, 'Train batch size')
flags.DEFINE_integer('epochs', 49, 'Train epoch')
flags.DEFINE_integer('save_period', 5, 'Save checkpoint per step')
flags.DEFINE_integer('topk', 1, 'Metric topk')
flags.DEFINE_boolean('use_mp', False, 'Flags that determine whether to use mixed precision')


def main(_):
    
    with open(f'{FLAGS.config_dir}/config.json', 'r') as f:
        configs = json.load(f)
        
    if FLAGS.op == 'preprocessing':
        filepath = f'{FLAGS.data_dir}/raw/{FLAGS.dataset}'
        valid_ratio, test_ratio = [float(x) for x in FLAGS.split_ratio.split(', ')]

        # load and preprocess data
        dataset, dataset_info = load_interaction(filepath, configs['max_seq_length'])

        # split data
        trainset, testset = split_data(dataset, test_ratio)
        trainset, validset = split_data(trainset, valid_ratio)

        configs['train'] = len(trainset[0])
        configs['valid'] = len(validset[0])
        configs['test'] = len(testset[0])

	    # save train/valid set to tfrecord format
        trainset = chunk_data(trainset, FLAGS.chunk_size)
        validset = chunk_data(validset, FLAGS.chunk_size)
        save_tfrecords(trainset, FLAGS.chunk_size, f'{FLAGS.data_dir}/prep', 'train')
        save_tfrecords(validset, FLAGS.chunk_size, f'{FLAGS.data_dir}/prep', 'valid')

        # save testset
        test_dic = {
            'user_id': testset[0],
	        'sequence': testset[1],
	        'sequence_length': testset[2],
	        'label': testset[3]
        }
        with open(f'{FLAGS.data_dir}/prep/test.json', 'w') as f:
            f.write(json.dumps(test_dic, ensure_ascii=False))

        # info add and save
        info_path = f'{FLAGS.data_dir}/prep/info.pickle'
        with open(info_path, 'wb') as f:
            pickle.dump(dataset_info, f)

        with open(f'{FLAGS.config_dir}/config.json', 'w') as f:
            f.write(json.dumps(configs, ensure_ascii=False))
    elif FLAGS.op == 'train':
        feature_description = dict(
            user_id=tf.io.FixedLenFeature([1], tf.int64),
            sequence=tf.io.FixedLenFeature([1, 200], tf.int64),
            sequence_length=tf.io.FixedLenFeature([1], tf.int64),
            label=tf.io.FixedLenFeature(configs['n_items'], tf.int64)
        )
        trainset = get_features_from_tfrecord(feature_description,
                                             f'{FLAGS.data_dir}/prep',
                                             batch_size=FLAGS.batch_size,
                                             shuffle_size=configs.get('train'),
                                             data_type='train')
        validset = get_features_from_tfrecord(feature_description,
                                             f'{FLAGS.data_dir}/prep',
                                             batch_size=FLAGS.batch_size,
                                             shuffle_size=configs.get('valid'),
                                             data_type='valid')

        lightsans = SequentialRecommender(configs,
                                          batch_size=FLAGS.batch_size,
                                          epochs=FLAGS.epochs,
                                          is_training=True,
                                          topk=FLAGS.topk)
        current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
        ckpt_dir = f'{FLAGS.checkpoint_dir}/{current_time}'
        logs_dir = f'{FLAGS.logs_dir}/{current_time}'
        lightsans.train(trainset, validset, ckpt_dir, FLAGS.save_period, logs_dir)
    else:
        """Predict"""
        data_info_path = f'{FLAGS.data_dir}/prep/info.pickle'
        with open(data_info_path, 'rb') as f:
            data_info = pickle.load(f)

        idx2item = data_info['idx2item']
        item2idx = data_info['item2idx']

        # sample test sequence
        test_seq = [242, 302, 377]
        test_seq_length = len(test_seq)

        # remap item ids to indices
        test_seq_convert = [item2idx[x] for x in test_seq]

        # make feature dict
        test_feature = {'sequence': [[test_seq_convert]], 'sequence_length': [[test_seq_length]]}
        lightsans = SequentialRecommender(configs, FLAGS.batch_size, FLAGS.epochs, FLAGS.topk, is_training=False)
        prediction = lightsans.predict(test_feature, FLAGS.checkpoint_dir)

        # create list of (index, score) pair and sort in order of score. Larger score means high probability to buy that item.
        pred_list = []
        for idx, score in enumerate(prediction[0]):
            pred_list.append((idx, score))

        pred_list.sort(key=lambda x: (-x[1], x[0]))

        # remap indices to item ids
        print(idx2item[pred_list[0][0]])
        print(pred_list[0][1])
        print(idx2item[pred_list[1][0]])
        print(pred_list[1][1])


if __name__ == '__main__':
    
    tf.compat.v1.app.run()
