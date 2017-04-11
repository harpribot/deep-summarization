import tensorflow as tf
from models.bidirectional import Bidirectional


class LstmBidirectional(Bidirectional):
    def __init__(self, review_summary_file, checkpointer, attention=False):
        """

        :param review_summary_file:
        :param checkpointer:
        :param attention:
        """
        super(LstmBidirectional, self).__init__(review_summary_file, checkpointer, attention)

    def get_cell(self):
        """
        Return the atomic RNN cell type used for this model
        
        :return: The atomic RNN Cell
        """
        return tf.nn.rnn_cell.LSTMCell(self.memory_dim)
