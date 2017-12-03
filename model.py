import tensorflow as tf
import numpy as np
from visualisation import show_losses_ex

class Model():
    def __init__(self, image_shape, out_dims):
        self.image_shape = image_shape
        self.out_dims    = out_dims
        self._session    = None
        self._graph      = None
        self.scope       = 'embedding'

    def _make_loss(self, main, same, diff, margin):
        def dist(a, b):
            return tf.reduce_sum((a-b)**2, axis=1)
        pos_dist = dist(main, same)
        neg_dist = dist(main, diff)
        loss = tf.nn.relu(pos_dist - neg_dist + margin)
        loss = tf.reduce_mean(loss)
        return loss

    def build(self, arc):
        tf.reset_default_graph()
        self._graph = tf.Graph()
        img_shape = self.image_shape
        with self._graph.as_default(), tf.variable_scope(self.scope):
            # placeholders
            self.img_main_pl  = tf.placeholder(dtype=tf.float32, shape=[None,]+img_shape, name='main_img')
            self.img_same_pl  = tf.placeholder(dtype=tf.float32, shape=[None,]+img_shape, name='same_img')
            self.img_diff_pl  = tf.placeholder(dtype=tf.float32, shape=[None,]+img_shape, name='diff_img')
            self.margin_pl    = tf.placeholder(dtype=tf.float32, name='margin')
            self.lr_pl        = tf.placeholder(dtype=tf.float32, name='lr')
            self.training_pl  = tf.placeholder(dtype=tf.bool,    name='training')
            # network
            self.nn_main      = arc(self.img_main_pl, self.out_dims, reuse=False, training=self.training_pl)
            self.nn_same      = arc(self.img_same_pl, self.out_dims, reuse=True,  training=self.training_pl)
            self.nn_diff      = arc(self.img_diff_pl, self.out_dims, reuse=True,  training=self.training_pl)
            # operations            
            self.update_ops   = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            self.loss_op      = self._make_loss(self.nn_main, self.nn_same, self.nn_diff, self.margin_pl)
            with tf.control_dependencies(self.update_ops):
                self.train_op = tf.train.AdamOptimizer(self.lr_pl).minimize(self.loss_op)
            self.init_op      = tf.global_variables_initializer()
            
        self._session = tf.Session(graph=self._graph)
        self._session.run(self.init_op)
        
    def train(self, tr_dataset, va_dataset, step_num, batch_size, margin, lr, log_every=10, mean_win=100):
        try:
            tr_losses = []
            va_losses = []
            for step in range(step_num):
                img_main, img_same, img_diff = tr_dataset.get_next_batch(batch_size)
                _, tr_loss = self._session.run([self.train_op, self.loss_op], feed_dict={
                    self.img_main_pl: img_main,
                    self.img_same_pl: img_same,
                    self.img_diff_pl: img_diff,
                    self.margin_pl:   margin,
                    self.lr_pl:       lr,
                    self.training_pl: True,
                })
                tr_losses.append(tr_loss)
                if step % log_every == log_every-1:
                    img_main, img_same, img_diff = va_dataset.get_next_batch(batch_size)
                    va_loss = self._session.run(self.loss_op, feed_dict={
                        self.img_main_pl: img_main,
                        self.img_same_pl: img_same,
                        self.img_diff_pl: img_diff,
                        self.margin_pl:   margin,
                        self.training_pl: False,
                    })
                    va_losses.append([va_loss]*log_every)
                    show_losses_ex(step, tr_losses, va_losses, mean_win)
        except KeyboardInterrupt:
            pass
        show_losses_ex(step, tr_losses, va_losses, mean_win)

    def save(self, path):
        with self._graph.as_default(), tf.variable_scope(self.scope):
            saver = tf.train.Saver()
        saver.save(self._session, path)
        
    def restore(self, path):
        with self._graph.as_default(), tf.variable_scope(self.scope):
            saver = tf.train.Saver()
        saver.restore(self._session, path)        
    
    def forward(self, images, batch_size=100):
        def beg_end(num, bs):
            for i in range(0, num, bs):
                yield (i, min(i + bs, num))
        outputs = np.zeros(shape=[len(images), self.out_dims])
        for b,e in beg_end(len(images), batch_size):
            batch = images[b:e]
            res = self._session.run(self.nn_main, feed_dict = {
                self.img_main_pl: batch,
                self.training_pl: False,
            })
            outputs[b:e] = res
        return outputs