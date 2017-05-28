import tensorflow as tf, numpy as np
from TFCommon.Model import Model
from Generator import G
from Discriminator import D

learning_rate = 0.0002
l1_lambda = 100
max_time_steps = 16384

class SEGAN(Model):
    """
    """

    def __init__(self, name="SEGAN"):
        self.__name = name
        self.g_nn = G()
        self.d_nn = D()
        self.__forward_done = False
        self.__backprop_done = False
        self.__build_done = False

    @property
    def name(self):
        return self.__name

    @property
    def clean_gtruth(self):
        return self.__clean_gtruth

    @property
    def noisy_signal(self):
        return self.__noisy_signal

    @property
    def enhenced_signal(self):
        return self.__enhenced_signal

    @property
    def gtruth_score(self):
        return self.__gtruth_score

    @property
    def fake_score(self):
        return self.__fake_score

    def build(self, scope=None):
        """
        """
        assert not self.__build_done, "[!] have built model."
        with tf.variable_scope(scope or "model"):
            with tf.variable_scope("data"):
                self.__clean_gtruth = tf.placeholder(name="clean_or_enhenced", \
                        shape=(None, max_time_steps, 1), dtype=tf.float32)
                self.__noisy_signal = tf.placeholder(name="noisy_signal", \
                        shape=(None, max_time_steps, 1), dtype=tf.float32)
            self.__build_forward(self.clean_gtruth, self.noisy_signal)
            self.__build_backprop()
            self.__build_done = True

    def predict(self, data, sess, get_score=True):
        """
        """
        noisy_signal = data.get("noisy_signal")
        _, time_steps, channels = noisy_signal.shape
        assert channels == 1, "[!] The input channel must be 1."
        assert time_steps == max_time_steps, \
                "[!] The input time_steps must be %d." % max_time_steps
        feed_dict={self.noisy_signal: noisy_signal}
        if get_score:
            enhenced_signal_eval, fake_score_eval = \
                    sess.run([self.enhenced_signal, self.fake_score], feed_dict)
            return enhenced_signal_eval, fake_d_loss_eval
        else:
            enhenced_signal_eval = sess.run(self.enhenced_signal, feed_dict)
            return enhenced_signal_eval

    def fit(self, data, sess, batch_size=10, epochs=10):
        """
        """
        clean_gtruth = data.get("clean_gtruth")
        noisy_signal = data.get("noisy_signal")
        assert clean_gtruth.shape == noisy_signal.shape, \
                "[!] The shape of clean_gtruth and noisy_signal is not compatible."
        samples, time_steps, channels = clean_gtruth.shape
        assert channels == 1, "[!] The input channel must be 1."
        assert time_steps == max_time_steps, \
                "[!] The input time_steps must be %d." % max_time_steps
        for epoch in range(epochs):
            print("Epoch %d / %d" % (epoch, epochs))
            perm = np.random.permutation(samples)
            start = 0
            end = start + batch_size
            while start < samples:
                perm_index = perm[start:end]
                feed_dict = {self.clean_gtruth: clean_gtruth[perm_index], \
                        self.noisy_signal: noisy_signal[perm_index]}
                self.train(min(end, samples), samples, feed_dict, sess)
                start = end
                end = start + batch_size
            print("")

    def train(self, used_samples_cnt, total_samples, feed_dict, sess):
        """
        """
        ### update D.
        real_d_loss_eval, fake_d_loss_eval, *_ = \
                sess.run([self.__real_d_loss, self.__fake_d_loss, self.__d_upd], feed_dict)
        d_loss_eval = real_d_loss_eval + fake_d_loss_eval
        ### update G.
        g_loss_eval, *_ = sess.run([self.__g_loss, self.__g_upd], feed_dict)
        ### show some useful info.
        console_log = "\r[%d / %d]:\td_loss: %f; g_loss: %f; real_d_loss: %f; fake_d_loss: %f" \
                % (used_samples_cnt, total_samples, d_loss_eval, g_loss_eval, real_d_loss_eval, fake_d_loss_eval)
        print(console_log, end="")

    def __build_forward(self, clean_gtruth, noisy_signal, scope=None):
        """
        """
        assert not self.__forward_done, "[!] have built forward."
        with tf.variable_scope(scope or "forward"):
            ### 1st. use clean_gtruth and noisy_signal to get gtruth_score.
            self.__gtruth_score = self.d_nn(tf.concat([clean_gtruth, noisy_signal], axis=-1), reuse=False)

            ### 2nd. use noisy_signal to get enhenced_signal, and then cal the fake_score.
            self.__enhenced_signal = self.g_nn(noisy_signal, reuse=False)
            self.__fake_score = self.d_nn(tf.concat([self.__enhenced_signal, self.__noisy_signal], axis=-1), reuse=True)

            ### 3rd. cal the real_d_loss, fake_d_loss, and fake_g_loss.
            self.__real_d_loss = tf.reduce_mean(tf.squared_difference(self.__gtruth_score, 1.))
            self.__fake_d_loss = tf.reduce_mean(tf.squared_difference(self.__fake_score, 0.))
            self.__fake_g_loss = tf.reduce_mean(tf.squared_difference(self.__fake_score, 1.)) + \
                    l1_lambda * tf.reduce_mean(tf.abs(tf.subtract(self.__enhenced_signal, self.__clean_gtruth)))
            self.__d_loss = self.__real_d_loss + self.__fake_d_loss
            self.__g_loss = self.__fake_g_loss
            self.__forward_done = True

    def __build_backprop(self, scope=None):
        """
        """
        assert self.__forward_done, "[!] must run build_forward method first."
        assert not self.__backprop_done, "[!] have built backprop."
        with tf.variable_scope(scope or "backprop"):
            ### get G and D 's trainable vars.
            train_vars = tf.trainable_variables()
            g_vars = [var for var in train_vars if self.g_nn.name in var.name]
            d_vars = [var for var in train_vars if self.d_nn.name in var.name]

            ### get G and D 's grads and apply update.
            with tf.variable_scope("optimizer"):
                d_opt = tf.train.RMSPropOptimizer(learning_rate)
                g_opt = tf.train.RMSPropOptimizer(learning_rate)
                d_grads = d_opt.compute_gradients(self.__d_loss, d_vars)
                g_grads = g_opt.compute_gradients(self.__g_loss, g_vars)

                self.__d_upd = d_opt.apply_gradients(d_grads)
                self.__g_upd = g_opt.apply_gradients(g_grads)

            self.__backprop_done = True



