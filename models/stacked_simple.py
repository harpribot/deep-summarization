import tensorflow as tf
from models.sequenceNet import NeuralNet
from abc import abstractmethod
import cPickle as Pickle
import numpy as np
import random
from helpers.data2tensor import Mapper


class StackedSimple(NeuralNet):
    def __init__(self, review_summary_file, checkpointer, num_layers, attention=False):
        """
        A Simple Stacked (Unidirectional, Many Layers) Seq2Seq Encoder-Decoder model

        :param review_summary_file: The file containing the (food review, target tip summary) pair in CSV format
        :param checkpointer: The checkpoint handling object [Object]
        :param num_layers: Number of layer stacks
        :param attention:True, if attention mechanism is to be implemented, else False. Default: False
        """
        self.test_review = None
        self.predicted_test_summary = None
        self.true_summary = None
        self.train_size = None
        self.test_size = None
        self.X = None
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
        self.num_layers = num_layers

        self.enc_inp = None
        self.dec_inp = None

        self._load_data()
        super(StackedSimple, self).__init__()

    @abstractmethod
    def get_cell(self):
        pass

    def _split_train_tst(self):
        """
        divide the data into training and testing data
        Create the X_trn, X_tst, and Y_trn and Y_tst
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

        self.X = self.X[sample_id]
        self.Y = self.Y[sample_id]
        # Now divide the data into test ans train set
        test_fraction = 0.01
        self.test_size = int(test_fraction * num_samples)
        self.train_size = num_samples - self.test_size
        # review
        self.X_trn = self.X[0:self.train_size]
        self.X_tst = self.X[self.train_size:num_samples]
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
        self.X, self.Y = self.mapper.get_tensor()
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
            self.enc_inp = [tf.placeholder(tf.int32, shape=(None,),
                                           name="input%i" % t)
                            for t in range(self.seq_length)]
            # desired output
            self.labels = [tf.placeholder(tf.int32, shape=(None,),
                                          name="labels%i" % t)
                           for t in range(self.seq_length)]
            # weight of the hidden layer
            self.weights = [tf.ones_like(labels_t, dtype=tf.float32)
                            for labels_t in self.labels]

            # Decoder input: prepend some "GO" token and drop the final
            # token of the encoder input
            self.dec_inp = ([tf.zeros_like(self.labels[0], dtype=np.int32, name="GO")]
                            + self.labels[:-1])

    def _load_model(self):
        """
        Creates the encoder decoder model

        :return: None
        """
        # Initial memory value for recurrence.
        self.prev_mem = tf.zeros((self.train_batch_size, self.memory_dim))

        # choose RNN/GRU/LSTM cell
        with tf.variable_scope("train_test", reuse=True):
            cell = self.get_cell()
            # Stacks layers of RNN's to form a stacked decoder
            self.cell = tf.nn.rnn_cell.MultiRNNCell([cell] * self.num_layers)

        # embedding model
        if not self.attention:
            with tf.variable_scope("train_test"):
                self.dec_outputs, self.dec_memory = tf.nn.seq2seq.embedding_rnn_seq2seq(
                                self.enc_inp, self.dec_inp, self.cell,
                                self.vocab_size, self.vocab_size, self.seq_length)
            with tf.variable_scope("train_test", reuse=True):
                self.dec_outputs_tst, _ = tf.nn.seq2seq.embedding_rnn_seq2seq(
                                self.enc_inp, self.dec_inp, self.cell,
                                self.vocab_size, self.vocab_size, self.seq_length, feed_previous=True)

        else:
            with tf.variable_scope("train_test"):
                self.dec_outputs, self.dec_memory = tf.nn.seq2seq.embedding_attention_seq2seq(
                                self.enc_inp, self.dec_inp, self.cell,
                                self.vocab_size, self.vocab_size, self.seq_length)
            with tf.variable_scope("train_test", reuse=True):
                self.dec_outputs_tst, _ = tf.nn.seq2seq.embedding_attention_seq2seq(
                                self.enc_inp, self.dec_inp, self.cell,
                                self.vocab_size, self.vocab_size, self.seq_length, feed_previous=True)

    def _load_optimizer(self):
        """
        Load the SGD optimizer

        :return: None
        """
        # loss function
        self.loss = tf.nn.seq2seq.sequence_loss(self.dec_outputs, self.labels, self.weights, self.vocab_size)

        # optimizer
        self.optimizer = tf.train.MomentumOptimizer(self.learning_rate, self.momentum)
        self.train_op = self.optimizer.minimize(self.loss)

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
            batch_data = self.X_trn[offset:(offset + self.train_batch_size), :].T
            batch_labels = self.Y_trn[offset:(offset + self.train_batch_size), :].T

            loss_t = self._train_batch(batch_data, batch_labels)
            print "Present Loss:", loss_t

            # check results on 2 tasks - Visual Validation
            print 'Train Data Validation\n'
            self._visual_validate(self.X_trn[301, :], self.Y_trn[301, :])
            print
            print
            print 'Test Data Validation\n'
            self._visual_validate(self.X_tst[56, :], self.Y_tst[56, :])
            print
            print

            # Store prediction after certain number of steps
            # This will be useful for the graph construction
            '''
            if(step % self.checkpointer.get_prediction_checkpoint_steps() == 0):
                self.predict()
                self.store_test_predictions('_' + str(step))
            '''

    def _train_batch(self, review, summary):
        """
        Train a batch of the data

        :param review: The input review data (X) shape[seq_length x batch_length]
        :param summary: The target tip data (Y) shape[seq_length x batch_length]
        :return: None
        """
        # feed in the data
        feed_dict = {self.enc_inp[t]: review[t] for t in range(self.seq_length)}
        feed_dict.update({self.labels[t]: summary[t] for t in range(self.seq_length)})

        # train
        _, loss_t = self.sess.run([self.train_op, self.loss], feed_dict)
        return loss_t

    def _visual_validate(self, review, true_summary):
        """
        Validate Result and display them on a sample

        :param review: The input review sentence
        :param true_summary: The true summary (target)
        :return: None
        """
        # review
        print 'Original Review'
        print self._index2sentence(review)
        print
        # True summary
        print 'True Summary'
        print self._index2sentence(true_summary)
        print
        # Generated Summary
        rev_out = self.generate_one_summary(review)
        print 'Generated Summary'
        print self._index2sentence(rev_out)
        print

    def generate_one_summary(self, review):
        """
        Create summary for one review using Encoder Decoder Seq2Seq model

        :param review: The input review
        :return: Output Summary of the model
        """
        review = review.T
        review = [np.array([int(x)]) for x in review]
        feed_dict_rev = {self.enc_inp[t]: review[t] for t in range(self.seq_length)}
        feed_dict_rev.update({self.labels[t]: review[t] for t in range(self.seq_length)})
        summary = self.sess.run(self.dec_outputs_tst, feed_dict_rev)
        summary = [logits_t.argmax(axis=1) for logits_t in summary]
        summary = [x[0] for x in summary]

        return summary

    def predict(self):
        """
        Make test time predictions of summary

        :return: None
        """
        self.predicted_test_summary = []
        for step in xrange(0, self.test_size // self.test_batch_size):
            print 'Predicting Batch No.:', step
            offset = (step * self.test_batch_size) % self.test_size
            batch_data = self.X_tst[offset:(offset + self.test_batch_size), :].T
            summary_test_out = self._predict_batch(batch_data)
            self.predicted_test_summary.extend(summary_test_out)

        print 'Prediction Complete. Moving Forward..'

        # test answers
        self.test_review = self.X_tst
        self.predicted_test_summary = self.predicted_test_summary
        self.true_summary = self.Y_tst

    def _predict_batch(self, review):
        """
        Predict test reviews in batches
        
        :param review: Input review batch
        :return: None
        """
        summary_out = []
        feed_dict_test = {self.enc_inp[t]: review[t] for t in range(self.seq_length)}
        feed_dict_test.update({self.labels[t]: review[t] for t in range(self.seq_length)})
        summary_test_prob = self.sess.run(self.dec_outputs_tst, feed_dict_test)

        # Do a softmax layer to get the final result
        summary_test_out = [logits_t.argmax(axis=1) for logits_t in summary_test_prob]

        for i in range(self.test_batch_size):
            summary_out.append([x[i] for x in summary_test_out])

        return summary_out
