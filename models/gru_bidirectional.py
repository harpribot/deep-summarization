import tensorflow.python.framework
from tensorflow.python.framework import ops
import tensorflow as tf
from tensorflow.models.rnn import seq2seq, rnn_cell, rnn
import numpy as np
from helpers.data2tensor import Mapper
from sklearn.cross_validation import train_test_split
import tempfile
import pandas as pd
import random
import cPickle as pickle

class NeuralNet:
    def __init__(self,review_summary_file, checkpointer, attention = False):
        # Set attention flag
        self.attention = attention
        # Store the provided checkpoint (if any)
        self.checkpointer= checkpointer
        # Get the input labels and output review
        self.review_summary_file = review_summary_file
        self.__load_data()

        # Load all the parameters
        self.__load_model_params()

    def set_parameters(self, batch_size, memory_dim, learning_rate):
        self.batch_size = batch_size
        self.memory_dim = memory_dim
        self.learning_rate = learning_rate

    def __load_data(self):
        '''
            Load data only if the present data is not checkpointed,
            else, just load the checkpointed data
        '''
        self.mapper = Mapper()
        self.mapper.generate_vocabulary(self.review_summary_file)
        self.X_fwd,self.X_bwd, self.Y = self.mapper.get_tensor(reverseFlag=True)
        # Store all the mapper values in a dict for later recovery
        self.mapper_dict = {}
        self.mapper_dict['seq_length'] = self.mapper.get_seq_length()
        self.mapper_dict['vocab_size'] = self.mapper.get_vocabulary_size()
        self.mapper_dict['rev_map'] = self.mapper.get_reverse_map()
        # Split into test and train data
        self.__split_train_tst()

    def __split_train_tst(self):
        # divide the data into training and testing data
        # Create the X_trn, X_tst, for both forward and backward, and Y_trn and Y_tst_fwd
        # Note that only the reviews are changed, and not the summary.
        num_samples = self.Y.shape[0]
        mapper_file = self.checkpointer.get_mapper_file_location()
        if(not self.checkpointer.is_mapper_checkpointed()):
            print 'No mapper checkpoint found. Fresh loading in progress ...'
            # Now shuffle the data
            sample_id = range(num_samples)
            random.shuffle(sample_id)
            print 'Dumping the mapper shuffle for reuse.'
            pickle.dump(sample_id,open(mapper_file,'wb'))
            print 'Dump complete. Moving Forward...'
        else:
            print 'Mapper Checkpoint found... Reading from mapper dump'
            sample_id = pickle.load(open(mapper_file,'rb'))
            print 'Mapping unpickling complete.. Moving forward...'

        self.X_fwd = self.X_fwd[sample_id]
        self.X_bwd = self.X_bwd[sample_id]
        self.Y = self.Y[sample_id]
        # Now divide the data into test ans train set
        test_fraction = 0.05
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


    def __load_model_params(self):
        # parameters
        self.seq_length = self.mapper_dict['seq_length']
        self.vocab_size = self.mapper_dict['vocab_size']
        self.momentum = 0.9

    def begin_session(self):
        # start the tensorflow session
        ops.reset_default_graph()
    	# assign efficient allocator
    	config = tf.ConfigProto()
    	config.gpu_options.allocator_type = 'BFC'
    	# initialize interactive session
        self.sess = tf.InteractiveSession(config=config)


    def form_model_graph(self):
        self.__load_data_graph()
        self.__load_model()
        self.__load_optimizer()
        self.__start_session()

    def __load_data_graph(self):
        # input
        with tf.variable_scope("train_test", reuse=True):
            # review input - Both original and reversed
            self.enc_inp_fwd = [tf.placeholder(tf.int32, shape=(None,),
                              name="input%i" % t)
                      for t in range(self.seq_length)]
            self.enc_inp_bwd = [tf.placeholder(tf.int32, shape=(None,),
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
        self.prev_mem = tf.zeros((self.batch_size, self.memory_dim))

        # choose RNN/GRU/LSTM cell
        with tf.variable_scope("forward"):
            self.forward_cell = rnn_cell.GRUCell(self.memory_dim)
        with tf.variable_scope("backward"):
            self.backward_cell = rnn_cell.GRUCell(self.memory_dim)


        # embedding model
        if not self.attention:
            with tf.variable_scope("forward"):
                self.dec_outputs_fwd, _ = seq2seq.embedding_rnn_seq2seq(\
                                self.enc_inp_fwd, self.dec_inp, self.forward_cell, \
                                self.vocab_size, self.vocab_size, self.seq_length)
            with tf.variable_scope("forward", reuse = True):
                self.dec_outputs_fwd_tst, _ = seq2seq.embedding_rnn_seq2seq(\
                                self.enc_inp_fwd, self.dec_inp, self.forward_cell, \
                                self.vocab_size, self.vocab_size, self.seq_length, feed_previous=True)

            with tf.variable_scope("backward"):
                self.dec_outputs_bwd, _ = seq2seq.embedding_rnn_seq2seq(\
                                self.enc_inp_bwd, self.dec_inp, self.backward_cell, \
                                self.vocab_size, self.vocab_size, self.seq_length)

            with tf.variable_scope("backward", reuse = True):
                self.dec_outputs_bwd_tst, _ = seq2seq.embedding_rnn_seq2seq(\
                                self.enc_inp_bwd, self.dec_inp, self.backward_cell, \
                                self.vocab_size, self.vocab_size, self.seq_length, feed_previous=True)

        else:
            with tf.variable_scope("forward"):
                self.dec_outputs_fwd, _ = seq2seq.embedding_attention_seq2seq(\
                                self.enc_inp_fwd, self.dec_inp, self.forward_cell, \
                                self.vocab_size, self.vocab_size, self.seq_length)
            with tf.variable_scope("forward", reuse = True):
                self.dec_outputs_fwd_tst, _ = seq2seq.embedding_attention_seq2seq(\
                                self.enc_inp_fwd, self.dec_inp, self.forward_cell, \
                                self.vocab_size, self.vocab_size, self.seq_length, feed_previous=True)

            with tf.variable_scope("backward"):
                self.dec_outputs_bwd, _ = seq2seq.embedding_attention_seq2seq(\
                                self.enc_inp_bwd, self.dec_inp, self.backward_cell, \
                                self.vocab_size, self.vocab_size, self.seq_length)

            with tf.variable_scope("backward", reuse = True):
                self.dec_outputs_bwd_tst, _ = seq2seq.embedding_attention_seq2seq(\
                                self.enc_inp_bwd, self.dec_inp, self.backward_cell, \
                                self.vocab_size, self.vocab_size, self.seq_length, feed_previous=True)


    def __load_optimizer(self):
        # loss function
        with tf.variable_scope("forward"):
            self.loss_fwd = seq2seq.sequence_loss(self.dec_outputs_fwd, self.labels, \
                                                self.weights, self.vocab_size)

            # optimizer
            self.optimizer_fwd = tf.train.MomentumOptimizer(self.learning_rate, \
                                                    self.momentum)
            self.train_op_fwd = self.optimizer_fwd.minimize(self.loss_fwd)

        with tf.variable_scope("backward"):
            self.loss_bwd = seq2seq.sequence_loss(self.dec_outputs_bwd, self.labels, \
                                                self.weights, self.vocab_size)

            # optimizer
            self.optimizer_bwd = tf.train.MomentumOptimizer(self.learning_rate, \
                                                    self.momentum)
            self.train_op_bwd = self.optimizer_bwd.minimize(self.loss_bwd)


    def __start_session(self):
        self.sess.run(tf.initialize_all_variables())
        # initialize the saver node
        self.saver = tf.train.Saver()
        # get the latest checkpoint
        last_checkpoint_path = self.checkpointer.get_last_checkpoint()
        if last_checkpoint_path is not None:
            print 'Previous saved tensorflow objects found... Extracting...'
            # restore the tensorflow variables
            self.saver.restore(self.sess, last_checkpoint_path)
            print 'Extraction Complete. Moving Forward....'

    def fit(self):
        # Iterate and train.
        step_file = self.checkpointer.get_step_file()
        start_step = pickle.load(open(step_file,'rb'))
        for step in xrange(start_step,self.train_size // self.batch_size):
            print 'Step No.:', step
            # Checkpoint tensorflow variables for recovery
            if(step % self.checkpointer.get_checkpoint_steps() == 0):
                print 'Checkpointing: Saving Tensorflow variables'
                self.saver.save(self.sess, self.checkpointer.get_save_address())
                pickle.dump(step + 1, open(step_file, 'wb'))
                print 'Checkpointing Complete. Deleting historical checkpoints....'
                self.checkpointer.delete_previous_checkpoints(num_previous=2)
                print 'Deleted.. Moving forward...'

            offset = (step * self.batch_size) % self.train_size
            batch_data_fwd = self.X_trn_fwd[offset:(offset + self.batch_size), :].T
            batch_data_bwd = self.X_trn_bwd[offset:(offset + self.batch_size), :].T
            batch_labels = self.Y_trn[offset:(offset + self.batch_size),:].T

            loss_t_forward, loss_t_backward = self.__train_batch(batch_data_fwd, batch_data_bwd, batch_labels)
            print "Present Loss Forward:", loss_t_forward
            print "Present Loss Backward:", loss_t_backward

            ###### check results on 2 tasks - Visual Validation
            print 'Train Data Validation\n'
            self.__visual_validate(self.X_trn_fwd[301,:],self.X_trn_bwd[301,:],self.Y_trn[301,:])
            print
            print
            print 'Test Data Validation\n'
            self.__visual_validate(self.X_tst_fwd[156,:],self.X_tst_bwd[156,:],self.Y_tst[156,:])
            print
            print



    def __train_batch(self,review_fwd, review_bwd,summary):
        '''
            review : shape[seq_length x batch_length]
            summary : shape[seq_length x batch_length]
        '''
        # feed in the data for forward model
        feed_dict_fwd = {self.enc_inp_fwd[t]: review_fwd[t] for t in range(self.seq_length)}
        feed_dict_fwd.update({self.labels[t]: summary[t] for t in range(self.seq_length)})

        # feed in the data for the backward model
        feed_dict_bwd = {self.enc_inp_bwd[t] : review_bwd[t] for t in range(self.seq_length)}
        feed_dict_bwd.update({self.labels[t]: summary[t] for t in range(self.seq_length)})

        # train forward model
        print 'Forward Batch Training.......'
        _, loss_t_forward = self.sess.run([self.train_op_fwd, self.loss_fwd], feed_dict_fwd)

        # train backward model
        print 'Backward Batch Training.......'
        _, loss_t_backward = self.sess.run([self.train_op_bwd, self.loss_bwd], feed_dict_bwd)

        return loss_t_forward, loss_t_backward

    def __visual_validate(self,review_fwd, review_bwd,true_summary):
        # review
        print 'Original Review'
        print self.__index2sentence(review_fwd)
        print
        # True summary
        print 'True Summary'
        print self.__index2sentence(true_summary)
        print
        # Generated Summary
        summary_out = self.generate_one_summary(review_fwd, review_bwd)
        print 'Generated Summary'
        print self.__index2sentence(summary_out)
        print


    def __index2sentence(self,list_):
        rev_map = self.mapper_dict['rev_map']
        sentence = ""
        for entry in list_:
            if entry != 0:
                sentence += (rev_map[entry] + " ")

        return sentence



    def generate_one_summary(self,review_fwd, review_bwd):
        review_fwd = review_fwd.T
        review_bwd = review_bwd.T
        review_fwd = [np.array([int(x)]) for x in review_fwd]
        review_bwd = [np.array([int(x)]) for x in review_bwd]
        feed_dict_review_fwd = {self.enc_inp_fwd[t]: review_fwd[t] for t in range(self.seq_length)}
        feed_dict_review_fwd.update({self.labels[t]: review_fwd[t] for t in range(self.seq_length)})# Adds dummy label # Not used

        feed_dict_review_bwd = {self.enc_inp_bwd[t]: review_bwd[t] for t in range(self.seq_length)}
        feed_dict_review_bwd.update({self.labels[t]: review_bwd[t] for t in range(self.seq_length)})# Adds dummy label # Not used

        summary_prob_fwd = self.sess.run(self.dec_outputs_fwd_tst, feed_dict_review_fwd)
        summary_prob_bwd = self.sess.run(self.dec_outputs_bwd_tst, feed_dict_review_bwd)

        summary_sum_pool = (summary_prob_bwd + summary_prob_bwd)
        summary_avg_pool = [x/2. for x in summary_sum_pool]
        summary_out = [logits_t.argmax(axis=1) for logits_t in summary_avg_pool]
        summary_out = [x[0] for x in summary_out]

        return summary_out

    def predict(self):
        self.X_tst_fwd = self.X_tst_fwd.T
        self.X_tst_bwd = self.X_tst_bwd.T
        #### Forward probability
        feed_dict_test_fwd = {self.enc_inp_fwd[t]: X_tst_fwd[t] for t in range(self.seq_length)}
        # This is dummy label, same as input here. We are not using it, as it is test and feed_previous = True
        feed_dict_test.update({self.labels[t]: X_tst_fwd[t] for t in range(self.seq_length)})
        summary_test_prob_fwd = self.sess.run(self.dec_outputs_fwd_tst, feed_dict_test_fwd)

        #### Backward Probability
        feed_dict_test_bwd = {self.enc_inp_bwd[t]: X_tst_bwd[t] for t in range(self.seq_length)}
        # This is dummy label, same as input here. We are not using it, as it is test and feed_previous = True
        feed_dict_test_bwd.update({self.labels[t]: X_tst_bwd[t] for t in range(self.seq_length)})
        summary_test_prob_fwd = self.sess.run(self.dec_outputs_fwd_tst, feed_dict_test_fwd)

        # Average the forward and backward probability
        summary_sum_pool = (summary_prob_bwd + summary_prob_bwd)
        summary_avg_pool = [x/2. for x in summary_sum_pool]

        # Do a softmax layer to get the final result
        summary_out = [logits_t.argmax(axis=1) for logits_t in summary_avg_pool]
        summary_out = [x[0] for x in summary_out]

        # test answers
        self.test_review = self.X_tst
        self.predicted_test_summary = summary_out
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
