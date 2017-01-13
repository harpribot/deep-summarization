from tensorflow.python.framework import ops
import tensorflow as tf
import numpy as np
from helpers.data2tensor import Mapper
import pandas as pd
from abc import abstractmethod


class NeuralNet:
    def __init__(self, review_summary_file, checkpointer, attention=False):
        """

        :param review_summary_file:
        :param checkpointer:
        :param attention:
        """
        # Set attention flag
        self.attention = attention
        # Store the provided checkpoint (if any)
        self.checkpointer = checkpointer
        # Get the input labels and output review
        self.review_summary_file = review_summary_file
        self.__load_data()

        # Load all the parameters
        self.__load_model_params()
        self.sess = None

        # Testing parameters
        self.test_review = None
        self.predicted_test_summary = None
        self.true_summary = None
        self.test_size = None
        self.train_batch_size = None
        self.test_batch_size = None
        self.memory_dim = None
        self.learning_rate = None

    def set_parameters(self, train_batch_size, test_batch_size, memory_dim, learning_rate):
        """

        :param train_batch_size:
        :param test_batch_size:
        :param memory_dim:
        :param learning_rate:
        :return:
        """
        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size
        self.memory_dim = memory_dim
        self.learning_rate = learning_rate

    def __load_data(self):
        """
        Load data only if the present data is not checkpointed, else, just load the checkpointed data
        :return:
        """
        self.mapper = Mapper()
        self.mapper.generate_vocabulary(self.review_summary_file)
        self.X_fwd, self.X_bwd, self.Y = self.mapper.get_tensor(reverseflag=True)
        # Store all the mapper values in a dict for later recovery
        self.mapper_dict = dict()
        self.mapper_dict['seq_length'] = self.mapper.get_seq_length()
        self.mapper_dict['vocab_size'] = self.mapper.get_vocabulary_size()
        self.mapper_dict['rev_map'] = self.mapper.get_reverse_map()
        # Split into test and train data
        self.__split_train_tst()

    @abstractmethod
    def __split_train_tst(self):
        pass

    def __load_model_params(self):
        """

        :return:
        """
        # parameters
        self.seq_length = self.mapper_dict['seq_length']
        self.vocab_size = self.mapper_dict['vocab_size']
        self.momentum = 0.9

    def begin_session(self):
        """

        :return:
        """
        # start the tensorflow session
        ops.reset_default_graph()
        # assign efficient allocator
        config = tf.ConfigProto()
        config.gpu_options.allocator_type = 'BFC'
        # initialize interactive session
        self.sess = tf.InteractiveSession(config=config)

    def form_model_graph(self):
        """

        :return:
        """
        self.__load_data_graph()
        self.__load_model()
        self.__load_optimizer()
        self.__start_session()

    @abstractmethod
    def __load_data_graph(self):
        pass

    @abstractmethod
    def __load_model(self):
        pass

    @abstractmethod
    def __load_optimizer(self):
        pass

    def __start_session(self):
        """

        :return:
        """
        self.sess.run(tf.global_variables_initializer())
        # initialize the saver node
        self.saver = tf.train.Saver()
        # get the latest checkpoint
        last_checkpoint_path = self.checkpointer.get_last_checkpoint()
        if last_checkpoint_path is not None:
            print 'Previous saved tensorflow objects found... Extracting...'
            # restore the tensorflow variables
            self.saver.restore(self.sess, last_checkpoint_path)
            print 'Extraction Complete. Moving Forward....'

    @abstractmethod
    def fit(self):
        pass

    def __index2sentence(self, list_):
        """

        :param list_:
        :return:
        """
        rev_map = self.mapper_dict['rev_map']
        sentence = ""
        for entry in list_:
            if entry != 0:
                sentence += (rev_map[entry] + " ")

        return sentence

    def store_test_predictions(self, prediction_id='_final'):
        """

        :param prediction_id:
        :return:
        """
        # prediction id is usually the step count
        print 'Storing predictions on Test Data...'
        review = []
        true_summary = []
        generated_summary = []
        for i in range(self.test_size):
            if not self.checkpointer.is_output_file_present():
                review.append(self.__index2sentence(self.test_review[i]))
                true_summary.append(self.__index2sentence(self.true_summary[i]))
            if i < (self.test_batch_size * (self.test_size // self.test_batch_size)):
                generated_summary.append(self.__index2sentence(self.predicted_test_summary[i]))
            else:
                generated_summary.append('')

        prediction_nm = 'generated_summary' + prediction_id
        if self.checkpointer.is_output_file_present():
            df = pd.read_csv(self.checkpointer.get_result_location(), header=0)
            df[prediction_nm] = np.array(generated_summary)
        else:
            df = pd.DataFrame()
            df['review'] = np.array(review)
            df['true_summary'] = np.array(true_summary)
            df[prediction_nm] = np.array(generated_summary)
        df.to_csv(self.checkpointer.get_result_location(), index=False)
        print 'Stored the predictions. Moving Forward'
        if prediction_id == '_final':
            print 'All done. Exiting..'
            print 'Exited'

    def close_session(self):
        """

        :return:
        """
        self.sess.close()
