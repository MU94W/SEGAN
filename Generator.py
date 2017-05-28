import tensorflow as tf
from config import *

class G(object):
    def __init__(self, name="Generator"):
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
            assert inp.shape[1:].is_fully_defined(), \
                    "[!] Time dim and feat dim must be defined."
            feat_map = inp.shape.with_rank(3).as_list()[-1]
            assert feat_map == feature_map_lst[0], \
                    "[!] input's last dim must be 1."

            ### encode stage
            with tf.variable_scope("encode"):
                per_layer_out = []
                last_out = inp
                for i in range(1, len(feature_map_lst)):
                    with tf.variable_scope("stage_%d" % i):
                        out_channles = feature_map_lst[i]
                        in_channles = feature_map_lst[i-1]
                        this_filter = tf.get_variable(name="filter", \
                                shape=(filter_width, in_channles, out_channles))
                        this_out = tf.contrib.keras.layers.PReLU()(\
                                tf.nn.conv1d(last_out, this_filter, stride, padding))
                        per_layer_out.append(this_out)
                        last_out = this_out

            ### decode stage
            with tf.variable_scope("decode"):
                latent_z = tf.expand_dims(\
                        tf.random_normal(shape=tf.shape(per_layer_out[-1])), axis=1)
                last_out = latent_z
                out_channel_lst = feature_map_lst[::-1][1:]
                batch_size = tf.shape(latent_z)[0]
                for out_channles, res_out, idx in \
                        zip(out_channel_lst, per_layer_out[::-1], range(1, len(feature_map_lst))):
                    with tf.variable_scope("stage_%d" % idx):
                        res_out = tf.expand_dims(res_out, axis=1)
                        last_out_conc = tf.concat([last_out, res_out], axis=-1)
                        in_channles = last_out_conc.shape[-1].value
                        this_filter = tf.get_variable(name="filter", \
                                shape=(1, filter_width, out_channles, in_channles))
                        output_shape = (batch_size, 1, 2*last_out_conc.shape[2].value, out_channles)
                        this_out = tf.contrib.keras.layers.PReLU()(\
                                tf.reshape(\
                                tf.nn.conv2d_transpose(last_out_conc, this_filter, \
                                output_shape=output_shape, \
                                strides=[1, 1, stride, 1], \
                                padding=padding), \
                                shape=output_shape))
                        last_out = this_out

            enhanced_out = tf.reshape(last_out, shape=tf.shape(inp))

            return enhanced_out



