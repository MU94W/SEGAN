import tensorflow as tf
from config import *

class D(object):
    def __init__(self, name="Discriminator"):
        self.__name = name

    @property
    def name(self):
        return self.__name

    def __call__(self, inp, scope=None, reuse=None):
        """
        Args:
            inp: A tensor, with shape (batch_size, time_steps, feat_map[1])
        """
        with tf.variable_scope(scope or self.name, reuse=reuse):
            assert inp.shape[1:].is_fully_defined(), "[!] Time dim and feat dim must be defined."

            ### like G's encode stage
            with tf.variable_scope("D_network"):
                last_out = inp
                for i in range(1, len(feature_map_lst)):
                    with tf.variable_scope("stage_%d" % i):
                        out_channles = feature_map_lst[i]
                        in_channles = last_out.shape[-1].value
                        this_filter = tf.get_variable(name="filter", \
                                shape=(filter_width, in_channles, out_channles))
                        conv_out = tf.nn.conv1d(last_out, this_filter, stride, padding)
                        norm_out = tf.contrib.layers.batch_norm(conv_out)
                        this_out = tf.contrib.keras.layers.LeakyReLU(0.3)(norm_out)
                        last_out = this_out
            with tf.variable_scope("finnal_conv"):
                in_channles = last_out.shape[-1].value
                this_filter = tf.get_variable(name="filter", \
                        shape=(1, in_channles, 1))
                conv_out = tf.nn.conv1d(last_out, this_filter, stride, padding)
                norm_out = tf.contrib.layers.batch_norm(conv_out)
                this_out = tf.contrib.keras.layers.LeakyReLU(0.3)(norm_out)
                last_out = tf.squeeze(this_out, axis=-1)
            with tf.variable_scope("dense_out"):
                score = tf.layers.dense(last_out, 1, tf.sigmoid)

            return score


