import tensorflow as tf
from TFCommon.Model import Model
from Generator import G
from Discriminator import D

learning_rate = 0.0002
l1_lambda = 200

class SEGAN(Model):
    """
    """

    def __init__(self, name="SEGAN"):
        self.__name = name
        self.g_nn = G()
        self.d_nn = D()
        self.g_opt = tf.train.RMSPropOptimizer(learning_rate)
        self.d_opt = tf.train.RMSPropOptimizer(learning_rate)
        self.l1_lambda = l1_lambda
        self.forward_done = False
        self.backprop_done = False
        self.build_done = False

    @property
    def name(self):
        return self.__name

    def build(self, clean_gtruth, noisy_signal, scope=None):
        """
        """
        assert not self.build_done, "[!] have built model."
        with tf.variable_scope(scope or "model"):
            self.__build_forward(clean_gtruth, noisy_signal)
            self.__build_backprop()

    def train(self, feed_dict, sess):
        """
        """
        ### update D
        real_d_loss_eval, fake_d_loss_eval, *_ = sess.run([self.real_d_loss, self.fake_d_loss, self.d_upd], feed_dict)
        d_loss_eval = real_d_loss_eval + fake_d_loss_eval
        ### update G
        g_loss_eval, *_ = sess.run([self.g_loss, self.g_upd], feed_dict)
        print("d_loss: %f; g_loss: %f" % (d_loss_eval, g_loss_eval))
        print("real_d_loss: %f; fake_d_loss: %f" % (real_d_loss_eval, fake_d_loss_eval))

    def __build_forward(self, clean_gtruth, noisy_signal, scope=None):
        """
        """
        assert not self.forward_done, "[!] have built forward."
        with tf.variable_scope(scope or "forward"):
            ### 1st. use clean_gtruth and noisy_signal to get gtruth_score.
            gtruth_score = self.d_nn(tf.concat([clean_gtruth, noisy_signal], axis=-1), reuse=False)

            ### 2nd. use noisy_signal to get enhenced_signal, and then cal the fake_score.
            enhenced_signal = self.g_nn(noisy_signal, reuse=False)
            fake_score = self.d_nn(tf.concat([enhenced_signal, noisy_signal], axis=-1), reuse=True)

            ### 3rd. cal the real_d_loss, fake_d_loss, and fake_g_loss.
            self.real_d_loss = tf.reduce_mean(tf.squared_difference(gtruth_score, 1.))
            self.fake_d_loss = tf.reduce_mean(tf.squared_difference(fake_score, 0.))
            self.fake_g_loss = tf.reduce_mean(tf.squared_difference(fake_score, 1.)) + \
                    self.l1_lambda * tf.reduce_mean(tf.abs(tf.subtract(enhenced_signal, clean_gtruth)))
            self.d_loss = self.real_d_loss + self.fake_d_loss
            self.g_loss = self.fake_g_loss
            self.forward_done = True

    def __build_backprop(self, scope=None):
        """
        """
        assert self.forward_done, "[!] must run build_forward method first."
        assert not self.backprop_done, "[!] have built backprop."
        with tf.variable_scope(scope or "backprop"):
            ### get G and D 's trainable vars.
            train_vars = tf.trainable_variables()
            g_vars = [var for var in train_vars if self.g_nn.name in var.name]
            d_vars = [var for var in train_vars if self.d_nn.name in var.name]

            ### get G and D 's grads and apply update.
            d_opt = tf.train.RMSPropOptimizer(learning_rate)
            g_opt = tf.train.RMSPropOptimizer(learning_rate)
            d_grads = d_opt.compute_gradients(self.d_loss, d_vars)
            g_grads = g_opt.compute_gradients(self.g_loss, g_vars)

            self.d_upd = d_opt.apply_gradients(d_grads)
            self.g_upd = g_opt.apply_gradients(g_grads)
            self.backprop_done = True


