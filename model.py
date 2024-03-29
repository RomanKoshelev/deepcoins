import tensorflow as tf
import numpy as np
import pickle
import os
from visualization import show_train_stats
from utils import make_dir


def euclidean_dist(a, b):
    return tf.sqrt(tf.reduce_sum((a-b)**2, axis=1))


class Model():
    def __init__(self, image_shape, out_dims, acc_batch_size=100):
        self.image_shape    = image_shape
        self.out_dims       = out_dims
        self._session       = None
        self._graph         = None
        self.scope          = 'embedding'
        self.acc_batch_size = acc_batch_size
        # state
        self.tr_step        = 0
        self.tr_losses      = []
        self.va_losses      = []
        self.tr_accs        = []
        self.va_accs        = []        
        self.neg_distances  = []
        self.pos_distances  = []


    def _calc_lr(self, lr, losses, mean_win):
        def losses_to_lr(lr_dict, losses, mean_win):
            def running_mean(x, N):
                cumsum = np.cumsum(np.insert(x, 0, 0)) 
                return (cumsum[N:] - cumsum[:-N]) / N
            first_key = sorted(lr_dict.keys())[-1]
            if(len(losses) > mean_win):        
                loss = running_mean(losses, mean_win)[-1]
            else:
                return lr_dict[first_key]
            for k,v in sorted(lr_dict.items()):
                if loss<k:
                    return lr_dict[k]
            return lr_dict[first_key]

        if type(lr) is dict:
            return losses_to_lr(lr, losses, mean_win)
        elif type(lr) is float:
            return lr
        else:
            raise RuntimeError("wrong lr type")
    
    def _make_mean_distance(self, a, b):
        return tf.reduce_mean(euclidean_dist(a, b))


    def _make_loss(self, main, same, diff, margin):
        pos = euclidean_dist(main, same)
        neg = euclidean_dist(main, diff)
        loss = tf.nn.relu(pos - neg + margin)
        loss = tf.reduce_mean(loss)
        return loss
    

    def _make_acc(self, emb_main, emb_same):
        bs = self.acc_batch_size
        e0 = emb_main
        e1 = emb_same
        e0 = tf.tile(e0, [bs,1])
        e1 = tf.tile(e1, [1,bs])

        e0 = tf.reshape(e0, [-1])
        e1 = tf.reshape(e1, [-1])

        d = tf.sqrt((e1 - e0)**2)
        d = tf.reshape(d, [bs, bs, -1])
        d = tf.reduce_sum(d, axis=2)
        r = tf.argmin(d, 1)
        l = np.linspace(0,bs-1,bs)
        q = tf.cast(tf.equal(r,l), dtype=tf.float32)
        acc = tf.reduce_mean(q)        
        return acc

    
    def build(self, net):
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
            self.emb_main     = net(self.img_main_pl, self.out_dims, reuse=False, training=self.training_pl)
            self.emb_same     = net(self.img_same_pl, self.out_dims, reuse=True,  training=self.training_pl)
            self.emb_diff     = net(self.img_diff_pl, self.out_dims, reuse=True,  training=self.training_pl)
            # operations
            self.update_ops   = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            self.pos_dist_op  = self._make_mean_distance(self.emb_main, self.emb_same)
            self.neg_dist_op  = self._make_mean_distance(self.emb_main, self.emb_diff)
            self.loss_op      = self._make_loss(self.emb_main, self.emb_same, self.emb_diff, self.margin_pl)
            self.acc_op       = self._make_acc(self.emb_main, self.emb_same)
            with tf.control_dependencies(self.update_ops):
                self.train_op = tf.train.AdamOptimizer(self.lr_pl).minimize(self.loss_op)
            self.init_op      = tf.global_variables_initializer()
            
        self._session = tf.Session(graph=self._graph)
        self._session.run(self.init_op)
        
        
    def train(self, tr_dataset, va_dataset, step_num, batch_size, margin, lr, log_every=10, mean_win=100, log_scale=False):
        try:
            data_size = tr_dataset.get_data_size()
            for self.tr_step in range(self.tr_step, step_num-1):
                cur_lr = self._calc_lr(lr, self.tr_losses, mean_win)
                ep = self.tr_step*batch_size/data_size
                # Train
                img_main, img_same, img_diff = tr_dataset.get_next_batch(batch_size)
                _, tr_loss, pos_dist, neg_dist = self._session.run(
                    [self.train_op, self.loss_op, self.pos_dist_op, self.neg_dist_op], 
                    feed_dict={
                        self.img_main_pl: img_main,
                        self.img_same_pl: img_same,
                        self.img_diff_pl: img_diff,
                        self.margin_pl:   margin,
                        self.lr_pl:       cur_lr,
                        self.training_pl: True })
                self.tr_losses.append(tr_loss)
                self.pos_distances.append(pos_dist)
                self.neg_distances.append(neg_dist)
                # Eval
                if self.tr_step % log_every == log_every-1:
                    img_main, img_same, img_diff = va_dataset.get_next_batch(batch_size)
                    # loss
                    va_loss = self._session.run(
                        self.loss_op, 
                        feed_dict={
                            self.img_main_pl: img_main,
                            self.img_same_pl: img_same,
                            self.img_diff_pl: img_diff,
                            self.margin_pl:   margin,
                            self.training_pl: False })
                    self.va_losses.extend([va_loss]*log_every)
                    # acc
                    tr_acc = self.calc_acc(tr_dataset)
                    va_acc = self.calc_acc(va_dataset)
                    self.tr_accs.extend([tr_acc]*log_every)
                    self.va_accs.extend([va_acc]*log_every)
                    # show
                    show_train_stats(
                        ep, cur_lr, 
                        self.tr_losses, self.va_losses, 
                        self.tr_accs, self.va_accs,
                        self.neg_distances, self.pos_distances, 
                        mean_win, log_scale)
        except KeyboardInterrupt:
            show_train_stats(
                ep, cur_lr, 
                self.tr_losses, self.va_losses, 
                self.tr_accs, self.va_accs,
                self.neg_distances, self.pos_distances, 
                mean_win, log_scale)

        
    def save(self, path):
        make_dir(path)
        # state
        pickle.dump(
            [self.tr_step, 
             self.tr_losses, self.va_losses, 
             self.tr_accs, self.va_accs, 
             self.neg_distances, self.pos_distances], 
            open(os.path.join(path, "state.p"), "wb"))
        # weights
        with self._graph.as_default(), tf.variable_scope(self.scope):
            saver = tf.train.Saver()
        saver.save(self._session, path)
        
    def restore(self, path):
        # state
        try:
             [self.tr_step, 
             self.tr_losses, self.va_losses, 
             self.tr_accs, self.va_accs, 
             self.neg_distances, self.pos_distances] = pickle.load(
                open(os.path.join(path, "state.p"), "rb"))
        except: 
            print("State not found at", path)
        # weights
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
            emb = self._session.run(self.emb_main, feed_dict = {
                self.img_main_pl: batch,
                self.training_pl: False,
            })
            outputs[b:e] = emb
        return outputs
    
    def calc_acc(self, dataset):
        img_main, img_same, _ = dataset.get_next_batch(self.acc_batch_size)
        acc = self._session.run(self.acc_op, feed_dict = {
            self.img_main_pl: img_main,
            self.img_same_pl: img_same,
            self.training_pl: False
        })
        return acc