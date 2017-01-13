import tensorflow as tf
from tensorflow.models.rnn import seq2seq
from models.sequenceNet import NeuralNet
from abc import abstractmethod


class Simple(NeuralNet):
    def __init__(self,review_summary_file, checkpointer, attention = False):
        """

        :param review_summary_file:
        :param checkpointer:
        :param attention:
        """
        NeuralNet.__init__(self, review_summary_file, checkpointer, attention)

    @abstractmethod
    def get_cell(self):
        """

        :return:
        """
        pass

    def __load_model(self):
        """

        :return:
        """
        # Initial memory value for recurrence.
        self.prev_mem = tf.zeros((self.train_batch_size, self.memory_dim))

        # choose RNN/GRU/LSTM cell
        with tf.variable_scope("train_test", reuse=True):
            self.cell = self.get_cell

        # embedding model
        if not self.attention:
            with tf.variable_scope("train_test"):
                self.dec_outputs, self.dec_memory = seq2seq.embedding_rnn_seq2seq(
                                self.enc_inp, self.dec_inp, self.cell,
                                self.vocab_size, self.vocab_size, self.seq_length)
            with tf.variable_scope("train_test", reuse = True):
                self.dec_outputs_tst, _ = seq2seq.embedding_rnn_seq2seq(
                                self.enc_inp, self.dec_inp, self.cell,
                                self.vocab_size, self.vocab_size, self.seq_length, feed_previous=True)

        else:
            with tf.variable_scope("train_test"):
                self.dec_outputs, self.dec_memory = seq2seq.embedding_attention_seq2seq(
                                self.enc_inp, self.dec_inp, self.cell,
                                self.vocab_size, self.vocab_size, self.seq_length)
            with tf.variable_scope("train_test", reuse = True):
                self.dec_outputs_tst, _ = seq2seq.embedding_attention_seq2seq(
                                self.enc_inp, self.dec_inp, self.cell,
                                self.vocab_size, self.vocab_size, self.seq_length, feed_previous=True)
