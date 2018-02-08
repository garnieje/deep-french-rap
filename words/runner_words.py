# coding: utf-8
import tensorflow as tf

from lstm_words import LstmWords
from data_reader import DataReader

PATH_LYRICS = "/Users/jerome/Documents/garnieje/deep-french-rap/data/lyrics_Iam.csv"
PATH_SAVE = "./models/test_model_word.ckpt"

sess = tf.Session()
batch_size = 200
cell_size = 512
num_layers = 2
seq_len = 10
dropout_rate = 0.7
learning_rate = 0.002
embed_dim = 512
nb_epoch = 10000

reader = DataReader(PATH_LYRICS)
vocab = reader.vocab

with tf.Session() as sess:
    network = LstmWords(
        cell_size=cell_size,
        dropout_rate=dropout_rate,
        num_layers=num_layers,
        vocab=vocab,
        batch_size=batch_size,
        seq_len=seq_len,
        embed_dim=embed_dim,
        learning_rate=learning_rate,
        sess=sess)

    sess.run(tf.global_variables_initializer())

    for epoch in range(nb_epoch):
        input_batch, target_batch = reader.get_batch(batch_size, seq_len)
        network.train(input_batch, target_batch, PATH_SAVE)

        if epoch % 100 == 0:
            print(network.generate(50, "ce soir à marseille la nuit tombe noire et dure", sample=False))
            print("##############################################################")


