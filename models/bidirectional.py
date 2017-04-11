import tensorflow as tf
from models.sequenceNet import NeuralNet
from abc import abstractmethod
import cPickle as Pickle
import numpy as np
import random
from helpers.data2tensor import Mapper


class Bidirectional(NeuralNet):
    def __init__(self, review_summary_file, checkpointer, attention=False):
        """
        A Bidirectional ([Forward + Backward direction], One Layer) Seq2Seq Encoder-Decoder model

        :param review_summary_file: The file containing the (food review, target tip summary) pair in CSV format
        :param checkpointer: The checkpoint handling object [Object]
        :param attention: True, if attention mechanism is to be implemented, else False. Default: False
        """
        self.test_review = None
        self.predicted_test_summary = None
        self.true_summary = None
        self.train_size = None
        self.test_size = None
        self.X_fwd = None
        self.X_bwd = None
        self.Y = None
        self.prev_mem = None
        self.cell = None
        self.dec_outputs = None
        self.dec_memory = None
        self.labels = None
        self.loss = None
        self.weights = None
        self.optimizer = None
        self.train_op = None
        self.mapper_dict = None
        self.seq_length = None
        self.vocab_size = None
        self.momentum = None

        self.attention = attention
        self.review_summary_file = review_summary_file
        self.checkpointer = checkpointer

        self.enc_inp = None
        self.dec_inp = None

        self._load_data()
        super(Bidirectional, self).__init__()

    @abstractmethod
    def get_cell(self):
        pass

    def _split_train_tst(self):
        """
        divide the data into training and testing data
        Create the X_trn, X_tst, for both forward and backward, and Y_trn and Y_tst
        Note that only the reviews are changed, and not the summary.

        :return: None
        """
        num_samples = self.Y.shape[0]
        mapper_file = self.checkpointer.get_mapper_file_location()
        if not self.checkpointer.is_mapper_checkpointed():
            print 'No mapper checkpoint found. Fresh loading in progress ...'
            # Now shuffle the data
            sample_id = range(num_samples)
            random.shuffle(sample_id)
            print 'Dumping the mapper shuffle for reuse.'
            Pickle.dump(sample_id, open(mapper_file, 'wb'))
            print 'Dump complete. Moving Forward...'
        else:
            print 'Mapper Checkpoint found... Reading from mapper dump'
            sample_id = Pickle.load(open(mapper_file, 'rb'))
            print 'Mapping unpickling complete.. Moving forward...'

        self.X_fwd = self.X_fwd[sample_id]
        self.X_bwd = self.X_bwd[sample_id]
        self.Y = self.Y[sample_id]
        # Now divide the data into test ans train set
        test_fraction = 0.01
        self.test_size = int(test_fraction * num_samples)
        self.train_size = num_samples - self.test_size
        # Forward review
        self.X_trn_fwd = self.X_fwd[0:self.train_size]
        self.X_tst_fwd = self.X_fwd[self.train_size:num_samples]
        # Backward review
        self.X_trn_bwd = self.X_bwd[0:self.train_size]
        self.X_tst_bwd = self.X_bwd[self.train_size:num_samples]
        # Summary
        self.Y_trn = self.Y[0:self.train_size]
        self.Y_tst = self.Y[self.train_size:num_samples]

    def _load_data(self):
        """
        Load data only if the present data is not checkpointed, else, just load the checkpointed data

        :return: None
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
        self._split_train_tst()

    def _load_data_graph(self):
        """
        Loads the data graph consisting of the encoder and decoder input placeholders, Label (Target tip summary)
        placeholders and the weights of the hidden layer of the Seq2Seq model.

        :return: None
        """
        # input
        with tf.variable_scope("train_test", reuse=True):
            # review input - Both original and reversed
            self.enc_inp_fwd = [tf.placeholder(tf.int32, shape=(None,), name="input%i" % t)
                                for t in range(self.seq_length)]
            self.enc_inp_bwd = [tf.placeholder(tf.int32, shape=(None,), name="input%i" % t)
                                for t in range(self.seq_length)]
            # desired output
            self.labels = [tf.placeholder(tf.int32, shape=(None,), name="labels%i" % t)
                           for t in range(self.seq_length)]
            # weight of the hidden layer
            self.weights = [tf.ones_like(labels_t, dtype=tf.float32)
                            for labels_t in self.labels]

            # Decoder input: prepend some "GO" token and drop the final
            # token of the encoder input
            self.dec_inp = ([tf.zeros_like(self.labels[0], dtype=np.int32, name="GO")] + self.labels[:-1])

    def _load_model(self):
        """
        Creates the encoder decoder model

        :return: None
        """
        # Initial memory value for recurrence.
        self.prev_mem = tf.zeros((self.train_batch_size, self.memory_dim))

        # choose RNN/GRU/LSTM cell
        with tf.variable_scope("forward"):
            self.forward_cell = self.get_cell()
        with tf.variable_scope("backward"):
            self.backward_cell = self.get_cell()

        # embedding model
        if not self.attention:
            with tf.variable_scope("forward"):
                self.dec_outputs_fwd, _ = tf.nn.seq2seq.embedding_rnn_seq2seq(
                                self.enc_inp_fwd, self.dec_inp, self.forward_cell,
                                self.vocab_size, self.vocab_size, self.seq_length)
            with tf.variable_scope("forward", reuse=True):
                self.dec_outputs_fwd_tst, _ = tf.nn.seq2seq.embedding_rnn_seq2seq(
                                self.enc_inp_fwd, self.dec_inp, self.forward_cell,
                                self.vocab_size, self.vocab_size, self.seq_length, feed_previous=True)

            with tf.variable_scope("backward"):
                self.dec_outputs_bwd, _ = tf.nn.seq2seq.embedding_rnn_seq2seq(
                                self.enc_inp_bwd, self.dec_inp, self.backward_cell,
                                self.vocab_size, self.vocab_size, self.seq_length)

            with tf.variable_scope("backward", reuse=True):
                self.dec_outputs_bwd_tst, _ = tf.nn.seq2seq.embedding_rnn_seq2seq(
                                self.enc_inp_bwd, self.dec_inp, self.backward_cell,
                                self.vocab_size, self.vocab_size, self.seq_length, feed_previous=True)
        else:
            with tf.variable_scope("forward"):
                self.dec_outputs_fwd, _ = tf.nn.seq2seq.embedding_attention_seq2seq(
                                self.enc_inp_fwd, self.dec_inp, self.forward_cell,
                                self.vocab_size, self.vocab_size, self.seq_length)
            with tf.variable_scope("forward", reuse=True):
                self.dec_outputs_fwd_tst, _ = tf.nn.seq2seq.embedding_attention_seq2seq(
                                self.enc_inp_fwd, self.dec_inp, self.forward_cell,
                                self.vocab_size, self.vocab_size, self.seq_length, feed_previous=True)

            with tf.variable_scope("backward"):
                self.dec_outputs_bwd, _ = tf.nn.seq2seq.embedding_attention_seq2seq(
                                self.enc_inp_bwd, self.dec_inp, self.backward_cell,
                                self.vocab_size, self.vocab_size, self.seq_length)

            with tf.variable_scope("backward", reuse=True):
                self.dec_outputs_bwd_tst, _ = tf.nn.seq2seq.embedding_attention_seq2seq(
                                self.enc_inp_bwd, self.dec_inp, self.backward_cell,
                                self.vocab_size, self.vocab_size, self.seq_length, feed_previous=True)

    def _load_optimizer(self):
        """
        Load the SGD optimizer

        :return: None
        """
        # loss function
        with tf.variable_scope("forward"):
            self.loss_fwd = tf.nn.seq2seq.sequence_loss(self.dec_outputs_fwd,
                                                        self.labels, self.weights, self.vocab_size)

            # optimizer
            # self.optimizer_fwd = tf.train.MomentumOptimizer(self.learning_rate,
            #                                        self.momentum)
            self.optimizer_fwd = tf.train.GradientDescentOptimizer(self.learning_rate)
            self.train_op_fwd = self.optimizer_fwd.minimize(self.loss_fwd)

        with tf.variable_scope("backward"):
            self.loss_bwd = tf.nn.seq2seq.sequence_loss(self.dec_outputs_bwd,
                                                        self.labels, self.weights, self.vocab_size)

            # optimizer
            # self.optimizer_bwd = tf.train.MomentumOptimizer(self.learning_rate,
            #                                        self.momentum)
            self.optimizer_bwd = tf.train.GradientDescentOptimizer(self.learning_rate)
            self.train_op_bwd = self.optimizer_bwd.minimize(self.loss_bwd)

    def fit(self):
        """
        Train the model with the training data

        :return: None
        """
        # Iterate and train.
        step_file = self.checkpointer.get_step_file()
        start_step = Pickle.load(open(step_file, 'rb'))
        for step in xrange(start_step, self.train_size // self.train_batch_size):
            print 'Step No.:', step
            # Checkpoint tensorflow variables for recovery
            if step % self.checkpointer.get_checkpoint_steps() == 0:
                print 'Checkpointing: Saving Tensorflow variables'
                self.saver.save(self.sess, self.checkpointer.get_save_address())
                Pickle.dump(step + 1, open(step_file, 'wb'))
                print 'Checkpointing Complete. Deleting historical checkpoints....'
                self.checkpointer.delete_previous_checkpoints(num_previous=2)
                print 'Deleted.. Moving forward...'

            offset = (step * self.train_batch_size) % self.train_size
            batch_data_fwd = self.X_trn_fwd[offset:(offset + self.train_batch_size), :].T
            batch_data_bwd = self.X_trn_bwd[offset:(offset + self.train_batch_size), :].T
            batch_labels = self.Y_trn[offset:(offset + self.train_batch_size), :].T

            loss_t_forward, loss_t_backward = self._train_batch(batch_data_fwd, batch_data_bwd, batch_labels)
            print "Present Loss Forward:", loss_t_forward
            print "Present Loss Backward:", loss_t_backward

            # check results on 2 tasks - Visual Validation
            print 'Train Data Validation\n'
            self._visual_validate(self.X_trn_fwd[301, :], self.X_trn_bwd[301, :], self.Y_trn[301, :])
            print
            print
            print 'Test Data Validation\n'
            self._visual_validate(self.X_tst_fwd[56, :], self.X_tst_bwd[56, :], self.Y_tst[56, :])
            print
            print

            # Store prediction after certain number of steps #############
            # This will be useful for the graph construction
            '''
            if(step % self.checkpointer.get_prediction_checkpoint_steps() == 0):
                self.predict()
                self.store_test_predictions('_' + str(step))
            '''

    def _train_batch(self, review_fwd, review_bwd, summary):
        """
        Train a batch of the data

        :param review: The input review data (X) shape[seq_length x batch_length]
        :param summary: The target tip data (Y) shape[seq_length x batch_length]
        :return: None
        """
        # feed in the data for forward model
        feed_dict_fwd = {self.enc_inp_fwd[t]: review_fwd[t] for t in range(self.seq_length)}
        feed_dict_fwd.update({self.labels[t]: summary[t] for t in range(self.seq_length)})

        # feed in the data for the backward model
        feed_dict_bwd = {self.enc_inp_bwd[t]: review_bwd[t] for t in range(self.seq_length)}
        feed_dict_bwd.update({self.labels[t]: summary[t] for t in range(self.seq_length)})

        # train forward model
        print 'Forward Batch Training.......'
        _, loss_t_forward = self.sess.run([self.train_op_fwd, self.loss_fwd], feed_dict_fwd)

        # train backward model
        print 'Backward Batch Training.......'
        _, loss_t_backward = self.sess.run([self.train_op_bwd, self.loss_bwd], feed_dict_bwd)

        return loss_t_forward, loss_t_backward

    def _visual_validate(self, review_fwd, review_bwd, true_summary):
        """
        Validate Result and display them on a sample

        :param review: The input review sentence
        :param true_summary: The true summary (target)
        :return: None
        """
        # review
        print 'Original Review'
        print self._index2sentence(review_fwd)
        print
        # True summary
        print 'True Summary'
        print self._index2sentence(true_summary)
        print
        # Generated Summary
        summary_out = self.generate_one_summary(review_fwd, review_bwd)
        print 'Generated Summary'
        print self._index2sentence(summary_out)
        print

    def generate_one_summary(self, review_fwd, review_bwd):
        """
        Create summary for one review using Encoder Decoder Seq2Seq model

        :param review_fwd: The input review for forward direction model
        :param review_bwd: The input review for backward direction model
        :return: Output Summary of the model
        """
        review_fwd = review_fwd.T
        review_bwd = review_bwd.T
        review_fwd = [np.array([int(x)]) for x in review_fwd]
        review_bwd = [np.array([int(x)]) for x in review_bwd]
        feed_dict_review_fwd = {self.enc_inp_fwd[t]: review_fwd[t] for t in range(self.seq_length)}
        feed_dict_review_fwd.update(
            {self.labels[t]: review_fwd[t] for t in range(self.seq_length)})  # Adds dummy label # Not used

        feed_dict_review_bwd = {self.enc_inp_bwd[t]: review_bwd[t] for t in range(self.seq_length)}
        feed_dict_review_bwd.update(
            {self.labels[t]: review_bwd[t] for t in range(self.seq_length)})  # Adds dummy label # Not used

        summary_prob_fwd = self.sess.run(self.dec_outputs_fwd_tst, feed_dict_review_fwd)
        summary_prob_bwd = self.sess.run(self.dec_outputs_bwd_tst, feed_dict_review_bwd)

        summary_sum_pool = [x + y for x, y in zip(summary_prob_fwd, summary_prob_bwd)]
        summary_out = [logits_t.argmax(axis=1) for logits_t in summary_sum_pool]
        summary_out = [x[0] for x in summary_out]

        return summary_out

    def predict(self):
        """
        Make test time predictions of summary

        :return: None
        """
        self.predicted_test_summary = []
        for step in xrange(0, self.test_size // self.test_batch_size):
            print 'Predicting Batch No.:', step
            offset = (step * self.test_batch_size) % self.test_size
            batch_data_fwd = self.X_tst_fwd[offset:(offset + self.test_batch_size), :].T
            batch_data_bwd = self.X_tst_bwd[offset:(offset + self.test_batch_size), :].T
            summary_test_out = self._predict_batch(batch_data_fwd, batch_data_bwd)
            self.predicted_test_summary.extend(summary_test_out)

        print 'Prediction Complete. Moving Forward..'

        # test answers
        self.test_review = self.X_tst_fwd
        self.predicted_test_summary = self.predicted_test_summary
        self.true_summary = self.Y_tst

    def _predict_batch(self, review_fwd, review_bwd):
        """
        Predict test reviews in batches
        
        :param review_fwd: Input review batch for forward propagation model
        :param review_bwd: Input review batch for backward propagation model
        :return: None
        """
        summary_out = []
        # Forward
        feed_dict_test_fwd = {self.enc_inp_fwd[t]: review_fwd[t] for t in range(self.seq_length)}
        feed_dict_test_fwd.update({self.labels[t]: review_fwd[t] for t in range(self.seq_length)})
        summary_test_prob_fwd = self.sess.run(self.dec_outputs_fwd_tst, feed_dict_test_fwd)
        # Backward
        feed_dict_test_bwd = {self.enc_inp_bwd[t]: review_bwd[t] for t in range(self.seq_length)}
        feed_dict_test_bwd.update({self.labels[t]: review_bwd[t] for t in range(self.seq_length)})
        summary_test_prob_bwd = self.sess.run(self.dec_outputs_bwd_tst, feed_dict_test_bwd)

        summary_sum_pool = [x + y for x, y in zip(summary_test_prob_fwd, summary_test_prob_bwd)]
        # Do a softmax layer to get the final result
        summary_test_out = [logits_t.argmax(axis=1) for logits_t in summary_sum_pool]

        for i in range(self.test_batch_size):
            summary_out.append([x[i] for x in summary_test_out])

        return summary_out
