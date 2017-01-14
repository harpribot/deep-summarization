import os
import sys
sys.path.append(os.path.abspath(os.path.dirname(__file__) + '/' + '..'))
from models import gru_stacked_simple
from helpers import checkpoint

# Get the review summary file
review_summary_file = 'extracted_data/review_summary.csv'

# Initialize Checkpointer to ensure checkpointing
checkpointer = checkpoint.Checkpointer('stackedSimple', 'gru', 'noAttention')
checkpointer.steps_per_checkpoint(1000)
checkpointer.steps_per_prediction(1000)
# Do using GRU cell - without attention mechanism
out_file = 'result/stacked_simple/gru/no_attention.csv'
checkpointer.set_result_location(out_file)
gru_net = gru_stacked_simple.GruStackedSimple(review_summary_file, checkpointer, num_layers=2)
gru_net.set_parameters(train_batch_size=128, test_batch_size=128, memory_dim=128, learning_rate=0.05)
gru_net.begin_session()
gru_net.form_model_graph()
gru_net.fit()
gru_net.predict()
gru_net.store_test_predictions()
