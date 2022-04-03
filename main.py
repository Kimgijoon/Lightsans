import tensorflow as tf


flags = tf.compat.v1.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('op', 'train', '[REQUIRED] Operation code to do')
flags.mark_flag_as_required('op')


def main(_):
    

    if FLAGS.op == 'preprocessing':
        pass
    elif FLAGS.op == 'train':
        pass
    else:
        pass


if __name__ == '__main__':
    
    tf.compat.v1.app.run()
