import tensorflow as tf
from models.simple import Simple


class GruSimple(Simple):
    def __init__(self, review_summary_file, checkpointer, attention=False):
        """

        :param review_summary_file:
        :param checkpointer:
        :param attention:
        """
        super(GruSimple, self).__init__(review_summary_file, checkpointer, attention)

    def get_cell(self):
        """
        Return the atomic RNN cell type used for this model
        
        :return: The atomic RNN Cell
        """
        return tf.nn.rnn_cell.GRUCell(self.memory_dim)
