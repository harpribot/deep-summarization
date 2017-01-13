import os
import sys
sys.path.append(os.path.abspath(os.path.dirname(__file__) + '/' + '..'))
from models import lstm_bidirectional
from helpers import checkpoint

# Get the review summary file
review_summary_file = 'extracted_data/review_summary.csv'

# Initialize Checkpointer to ensure checkpointing
checkpointer = checkpoint.Checkpointer('bidirectional', 'lstm', 'noAttention')
checkpointer.steps_per_checkpoint(500)
checkpointer.steps_per_prediction(2000)
# Do using GRU cell - without attention mechanism
out_file = 'result/bidirectional/lstm/no_attention.csv'
checkpointer.set_result_location(out_file)
lstm_net = lstm_bidirectional.LstmBidirectional(review_summary_file, checkpointer)
lstm_net.set_parameters(train_batch_size=10, test_batch_size=10, memory_dim=6, learning_rate=0.8)
lstm_net.begin_session()
lstm_net.form_model_graph()
lstm_net.fit()
lstm_net.predict()
lstm_net.store_test_predictions()
lstm_net.close_session()
