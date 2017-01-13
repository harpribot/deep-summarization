from models import lstm_simple
from helpers import checkpoint
# Get the review summary file
review_summary_file = 'extracted_data/review_summary.csv'

# Initialize Checkpointer to ensure checkpointing
checkpointer = checkpoint.Checkpointer('simple','lstm','noAttention')
checkpointer.steps_per_checkpoint(1000)
checkpointer.steps_per_prediction(2000)
# Do using LSTM cell - without attention mechanism
out_file = 'result/simple/lstm/no_attention.csv'
checkpointer.set_result_location(out_file)
lstm_net = lstm_simple.LstmSimple(review_summary_file, checkpointer)
lstm_net.set_parameters(train_batch_size=10, test_batch_size=10, memory_dim=10, learning_rate=0.008)
lstm_net.begin_session()
lstm_net.form_model_graph()
lstm_net.fit()
lstm_net.predict()
lstm_net.store_test_predictions()
lstm_net.close_session()
