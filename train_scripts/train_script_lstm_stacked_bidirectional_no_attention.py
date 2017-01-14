import os
import sys
sys.path.append(os.path.abspath(os.path.dirname(__file__) + '/' + '..'))
from models import lstm_stacked_bidirectional
from helpers import checkpoint
# Get the review summary file
review_summary_file = 'extracted_data/review_summary.csv'

# Initialize Checkpointer to ensure checkpointing
checkpointer = checkpoint.Checkpointer('stackedBidirectional', 'lstm', 'noAttention')
checkpointer.steps_per_checkpoint(1000)
checkpointer.steps_per_prediction(1000)
# Do using GRU cell - without attention mechanism
out_file = 'result/stacked_bidirectional/lstm/no_attention.csv'
checkpointer.set_result_location(out_file)
lstm_net = lstm_stacked_bidirectional.LstmStackedBidirectional(review_summary_file, checkpointer, num_layers=2)
lstm_net.set_parameters(train_batch_size=128, test_batch_size=128, memory_dim=128, learning_rate=0.05)
lstm_net.begin_session()
lstm_net.form_model_graph()
lstm_net.fit()
lstm_net.predict()
lstm_net.store_test_predictions()
