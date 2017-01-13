import tensorflow as tf
from models.simple import Simple


class LstmSimple(Simple):
    def __init__(self, review_summary_file, checkpointer, attention=False):
        """

        :param review_summary_file:
        :param checkpointer:
        :param attention:
        """
        Simple.__init__(self, review_summary_file, checkpointer, attention)

    def get_cell(self):
        """

        :return:
        """
        return tf.nn.rnn_cell.LSTMCell(self.memory_dim)
