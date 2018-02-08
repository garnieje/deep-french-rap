import tensorflow as tf
import numpy as np


class LstmWords:

    def __init__(self, cell_size, dropout_rate, num_layers, vocab, batch_size, seq_len, embed_dim, learning_rate, sess):

        self.sess = sess
        self.vocab = vocab
        self.vocab_size = len(vocab)
        self.batch_size = batch_size
        self.cell_size = cell_size
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate
        self.embed_dim = embed_dim
        self.learning_rate = learning_rate
        self.seq_len = seq_len

        self.build_graph()

    def build_graph(self):

        # placeholders
        self.inputs = tf.placeholder(tf.int32, shape=(
            None, None), name="inputs")
        self.targets = tf.placeholder(tf.int32, shape=(
            None, None), name="targets")
        input_text_shape = tf.shape(self.inputs)

        # define cells
        lstm_cell = tf.contrib.rnn.BasicLSTMCell(num_units=self.cell_size)
        drop_cell = tf.contrib.rnn.DropoutWrapper(
            lstm_cell, input_keep_prob=self.dropout_rate)
        self.cells = tf.contrib.rnn.MultiRNNCell([drop_cell] * self.num_layers)

        # initial state
        self.initial_state = self.cells.zero_state(
            input_text_shape[0], tf.float32)

        # embeddings
        self.embeddings = tf.contrib.layers.embed_sequence(
            self.inputs, self.vocab_size, self.embed_dim,
            initializer=tf.random_uniform_initializer(-1, 1))

        outputs, self.final_state = tf.nn.dynamic_rnn(
            self.cells, self.embeddings, dtype=tf.float32)

        logits = tf.contrib.layers.fully_connected(
            outputs, self.vocab_size, activation_fn=None)

        self.probs = tf.nn.softmax(logits, name='probs')

        self.cost = tf.contrib.seq2seq.sequence_loss(
            logits,
            self.targets,
            tf.ones([input_text_shape[0], input_text_shape[1]])
        )

        self.optimizer = tf.train.AdamOptimizer(
            self.learning_rate, name="optimizer")

        self.global_step = tf.Variable(0, trainable=False, name='global_step')
        self.train_op = self.optimizer.minimize(self.cost,
                                                global_step=self.global_step,
                                                name='train_op')

    def generate(self, num_out, prime, sample=True):
        state = self.sess.run(self.cells.zero_state(
            self.batch_size, tf.float32))
        prime = prime.decode("utf-8")
        gen_seq = prime
        prime = prime.lower().split()
        input_i = np.array([self.vocab.index(word)
                            for word in prime[-self.seq_len:]]).reshape(self.seq_len, 1)

        for n in range(num_out):
            feed_dict = {self.inputs: input_i, self.initial_state: state}
            probs, state = self.sess.run(
                [self.probs, self.final_state], feed_dict=feed_dict)
            probs = probs[self.seq_len - 1, 0, :]
            if sample:
                gen_word_i = np.random.choice(np.arange(len(probs)), p=probs)
            else:
                gen_word_i = np.argmax(probs)
            gen_word = self.vocab[gen_word_i]
            gen_seq += ' ' + gen_word
            input_i = np.array(
                list(input_i[1:, 0]) + [gen_word_i]).reshape(self.seq_len, 1)

        return gen_seq

    def train(self, input_batch, target_batch, path_save, save_freq=10):

        saver = tf.train.Saver(max_to_keep=None)

        feed_dict = {self.inputs: input_batch, self.targets: target_batch}
        global_step, loss, _ = self.sess.run([self.global_step, self.cost, self.train_op],
                                             feed_dict=feed_dict)
        print("step: {}, loss: {}".format(global_step, loss))

        if global_step % save_freq == 0:
            saver.save(self.sess, path_save, global_step=global_step)
