import tensorflow as tf
from models.stacked_simple import StackedSimple


class GruStackedSimple(StackedSimple):
    def __init__(self, review_summary_file, checkpointer, num_layers, attention=False):
        """

        :param review_summary_file:
        :param checkpointer:
        :param attention:
        """
        self.num_layers = num_layers
        StackedSimple.__init__(self, review_summary_file, checkpointer, attention)

    def get_cell(self):
        """

        :return:
        """
        return tf.nn.rnn_cell.GRUCell(self.memory_dim)
