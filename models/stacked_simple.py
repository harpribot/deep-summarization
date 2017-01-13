import tensorflow as tf
from models.sequenceNet import NeuralNet
from abc import abstractmethod
import cPickle as Pickle
import numpy as np
import random


class StackedSimple(NeuralNet):
    def __init__(self, review_summary_file, checkpointer, num_layers, attention=False):
        """

        :param review_summary_file:
        :param checkpointer:
        :param attention:
        """
        self.num_layers = num_layers
        NeuralNet.__init__(self, review_summary_file, checkpointer, attention)

    @abstractmethod
    def get_cell(self):
        pass

    def __split_train_tst(self):
        """
        divide the data into training and testing data
        Create the X_trn, X_tst, for both forward and backward, and Y_trn and Y_tst_fwd
        Note that only the reviews are changed, and not the summary.
        :return:
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

    def __load_data_graph(self):
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

    def __load_model(self):
        # Initial memory value for recurrence.
        self.prev_mem = tf.zeros((self.train_batch_size, self.memory_dim))

        # choose RNN/GRU/LSTM cell
        with tf.variable_scope("train_test", reuse=True):
            cell = self.get_cell
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

    def __load_optimizer(self):
        """

        :return:
        """
        # loss function
        self.loss = tf.nn.seq2seq.sequence_loss(self.dec_outputs, self.labels, self.weights, self.vocab_size)

        # optimizer
        self.optimizer = tf.train.MomentumOptimizer(self.learning_rate, self.momentum)
        self.train_op = self.optimizer.minimize(self.loss)

    def fit(self):
        """

        :return:
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

            loss_t = self.__train_batch(batch_data, batch_labels)
            print "Present Loss:", loss_t

            # check results on 2 tasks - Visual Validation
            print 'Train Data Validation\n'
            self.__visual_validate(self.X_trn[301, :], self.Y_trn[301, :])
            print
            print
            print 'Test Data Validation\n'
            self.__visual_validate(self.X_tst[56, :], self.Y_tst[56, :])
            print
            print

            # Store prediction after certain number of steps
            # This will be useful for the graph construction
            '''
            if(step % self.checkpointer.get_prediction_checkpoint_steps() == 0):
                self.predict()
                self.store_test_predictions('_' + str(step))
            '''

    def __train_batch(self, review, summary):
        """

        :param review: shape[seq_length x batch_length]
        :param summary: shape[seq_length x batch_length]
        :return:
        """
        # feed in the data
        feed_dict = {self.enc_inp[t]: review[t] for t in range(self.seq_length)}
        feed_dict.update({self.labels[t]: summary[t] for t in range(self.seq_length)})

        # train
        _, loss_t = self.sess.run([self.train_op, self.loss], feed_dict)
        return loss_t

    def __visual_validate(self, review, true_summary):
        """

        :param review:
        :param true_summary:
        :return:
        """
        # review
        print 'Original Review'
        print self.__index2sentence(review)
        print
        # True summary
        print 'True Summary'
        print self.__index2sentence(true_summary)
        print
        # Generated Summary
        rev_out = self.generate_one_summary(review)
        print 'Generated Summary'
        print self.__index2sentence(rev_out)
        print

    def generate_one_summary(self, rev):
        """

        :param rev:
        :return:
        """
        rev = rev.T
        rev = [np.array([int(x)]) for x in rev]
        feed_dict_rev = {self.enc_inp[t]: rev[t] for t in range(self.seq_length)}
        feed_dict_rev.update({self.labels[t]: rev[t] for t in range(self.seq_length)})
        rev_out = self.sess.run(self.dec_outputs_tst, feed_dict_rev)
        rev_out = [logits_t.argmax(axis=1) for logits_t in rev_out]
        rev_out = [x[0] for x in rev_out]

        return rev_out

    def predict(self):
        """

        :return:
        """
        self.predicted_test_summary = []
        for step in xrange(0, self.test_size // self.test_batch_size):
            print 'Predicting Batch No.:', step
            offset = (step * self.test_batch_size) % self.test_size
            batch_data = self.X_tst[offset:(offset + self.test_batch_size), :].T
            summary_test_out = self.__predict_batch(batch_data)
            self.predicted_test_summary.extend(summary_test_out)

        print 'Prediction Complete. Moving Forward..'

        # test answers
        self.test_review = self.X_tst
        self.predicted_test_summary = self.predicted_test_summary
        self.true_summary = self.Y_tst

    def __predict_batch(self, review):
        """

        :param review:
        :return:
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
