import tensorflow.python.framework
from tensorflow.python.framework import ops
import tensorflow as tf
from tensorflow.models.rnn import seq2seq, rnn_cell
import numpy as np
from helpers.data2tensor import Mapper
from sklearn.cross_validation import train_test_split
import tempfile
import pandas as pd

class NeuralNet:
    def __init__(self,review_summary_file, attention = False):
        # Set attention flag
        self.attention = attention
        # Get the input labels and output review
        self.review_summary_file = review_summary_file
        self.__load_data()

        # Split into test and train data
        self.__split_train_tst()

        # Load all the parameters
        self.__load_model_params()

    def set_parameters(self, batch_size, memory_dim, learning_rate):
        self.batch_size = batch_size
        self.memory_dim = memory_dim
        self.learning_rate = learning_rate

    def __load_data(self):
        self.mapper = Mapper()
        self.mapper.generate_vocabulary(self.review_summary_file)
        self.X,self.Y = self.mapper.get_tensor()

    def __split_train_tst(self):
        # divide the data into training and testing data
        self.X_trn, self.X_tst, self.Y_trn, self.Y_tst = \
                train_test_split(self.X, self.Y, test_size=0.05, random_state=42)
        self.train_size = self.X_trn.shape[0]
        self.test_size = self.X_tst.shape[0]

    def __load_model_params(self):
        # parameters
        self.seq_length = self.mapper.get_seq_length()
        self.vocab_size = self.mapper.get_vocabulary_size()
        self.momentum = 0.9

    def begin_session(self):
        # start the tensorflow session
        ops.reset_default_graph()
    	# assign efficient allocator
    	config = tf.ConfigProto()
    	config.gpu_options.allocator_type = 'BFC'
    	# initialize interactive session
        self.sess = tf.InteractiveSession(config=config)


    def form_model_graph(self, num_layers=1):
        self.__load_data_graph()
        self.__load_model(num_layers)
        self.__load_optimizer()
        self.__start_session()

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


    def __load_model(self, num_layers):
        # Initial memory value for recurrence.
        self.prev_mem = tf.zeros((self.batch_size, self.memory_dim))

        # choose RNN/GRU/LSTM cell
        with tf.variable_scope("train_test", reuse=True):
            gru = rnn_cell.GRUCell(self.memory_dim)
            # Stacks layers of RNN's to form a stacked decoder
            self.cell = rnn_cell.MultiRNNCell([gru] * num_layers)

        # embedding model
        if not self.attention:
            with tf.variable_scope("train_test"):
                self.dec_outputs, self.dec_memory = seq2seq.embedding_rnn_seq2seq(\
                                self.enc_inp, self.dec_inp, self.cell, \
                                self.vocab_size, self.vocab_size, self.seq_length)
            with tf.variable_scope("train_test", reuse = True):
                self.dec_outputs_tst, _ = seq2seq.embedding_rnn_seq2seq(\
                                self.enc_inp, self.dec_inp, self.cell, \
                                self.vocab_size, self.vocab_size, self.seq_length, feed_previous=True)

        else:
            with tf.variable_scope("train_test"):
                self.dec_outputs, self.dec_memory = seq2seq.embedding_attention_seq2seq(\
                                self.enc_inp, self.dec_inp, self.cell, \
                                self.vocab_size, self.vocab_size, self.seq_length)
            with tf.variable_scope("train_test", reuse = True):
                self.dec_outputs_tst, _ = seq2seq.embedding_attention_seq2seq(\
                                self.enc_inp, self.dec_inp, self.cell, \
                                self.vocab_size, self.vocab_size, self.seq_length, feed_previous=True)

    def __load_optimizer(self):
        # loss function
        self.loss = seq2seq.sequence_loss(self.dec_outputs, self.labels, \
                                                self.weights, self.vocab_size)

        # optimizer
        self.optimizer = tf.train.MomentumOptimizer(self.learning_rate, \
                                                    self.momentum)
        self.train_op = self.optimizer.minimize(self.loss)


    def __start_session(self):
        self.sess.run(tf.initialize_all_variables())

    def fit(self):
        # Iterate and train.
        for step in xrange(self.train_size // self.batch_size):
            print 'Step No.:', step
            offset = (step * self.batch_size) % self.train_size
            batch_data = self.X_trn[offset:(offset + self.batch_size), :].T
            batch_labels = self.Y_trn[offset:(offset + self.batch_size),:].T

            loss_t = self.__train_batch(batch_data, batch_labels)
            print "Present Loss:", loss_t

            ###### check results on 2 tasks - Visual Validation
            print 'Train Data Validation\n'
            self.__visual_validate(self.X_trn[301,:],self.Y_trn[301,:])
            print
            print
            print 'Test Data Validation\n'
            self.__visual_validate(self.X_tst[156,:],self.Y_tst[156,:])
            print
            print



    def __train_batch(self,review,summary):
        '''
            review : shape[seq_length x batch_length]
            summary : shape[seq_length x batch_length]
        '''
        # feed in the data
        feed_dict = {self.enc_inp[t]: review[t] for t in range(self.seq_length)}
        feed_dict.update({self.labels[t]: summary[t] for t in range(self.seq_length)})

        # train
        _, loss_t = self.sess.run([self.train_op, self.loss], feed_dict)
        return loss_t

    def __visual_validate(self,review,true_summary):
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


    def __index2sentence(self,list_):
        rev_map = self.mapper.get_reverse_map()
        sentence = ""
        for entry in list_:
            if entry != 0:
                sentence += (rev_map[entry] + " ")

        return sentence



    def generate_one_summary(self,rev):
        rev = rev.T
        rev = [np.array([int(x)]) for x in rev]
        feed_dict_rev = {self.enc_inp[t]: rev[t] for t in range(self.seq_length)}
        feed_dict_rev.update({self.labels[t]: rev[t] for t in range(self.seq_length)})
        rev_out = self.sess.run(self.dec_outputs_tst, feed_dict_rev )
        rev_out = [logits_t.argmax(axis=1) for logits_t in rev_out]
        rev_out = [x[0] for x in rev_out]

        return rev_out

    def predict(self):
        self.X_tst = self.X_tst.T
        feed_dict_test = {self.enc_inp[t]: X_tst[t] for t in range(self.seq_length)}
        feed_dict_test.update({self.labels[t]: X_tst[t] for t in range(self.seq_length)})
        summary_test_prob = self.sess.run(self.dec_outputs_tst, feed_dict_test)

        # Do a softmax layer to get the final result
        summary_test_out = [logits_t.argmax(axis=1) for logits_t in summary_test_prob]
        summary_test_out = [x[0] for x in summary_test_out]
        # test answers
        self.test_review = self.X_tst
        self.predicted_test_summary = summary_test_out
        self.true_summary = self.Y_tst

    def store_test_predictions(self, outfile):
        review = []
        true_summary = []
        generated_summary = []
        for i in range(self.test_size):
            review.append(self.__index2sentence(self.test_review[i]))
            true_summary.append(self.__index2sentence(self.true_summary))
            generated_summary.append(self.__index2sentence(self.predicted_test_summary))

        df = pd.DataFrame()
        df['review'] = np.array(review)
        df['true_summary'] = np.array(true_summary)
        df['generated_summary'] = np.array(generated_summary)
        df.to_csv(outfile, index=False)

    def close_session(self):
	self.sess.close()
