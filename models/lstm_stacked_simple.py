import tensorflow as tf
from models.stacked_simple import StackedSimple


class LstmStackedSimple(StackedSimple):
    def __init__(self, review_summary_file, checkpointer, num_layers, attention=False):
        """

        :param review_summary_file:
        :param checkpointer:
        :param num_layers:
        :param attention:
        """
        super(LstmStackedSimple, self).__init__(review_summary_file, checkpointer, num_layers, attention)

    def get_cell(self):
        """
        Return the atomic RNN cell type used for this model
        
        :return: The atomic RNN Cell
        """
        return tf.nn.rnn_cell.LSTMCell(self.memory_dim)
