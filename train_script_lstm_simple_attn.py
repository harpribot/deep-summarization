from models import lstm_simple
from helpers import checkpoint
# Get the review summary file
review_summary_file = 'extracted_data/review_summary.csv'

# Initialize Checkpointer to ensure checkpointing
checkpointer = checkpoint.Checkpointer('simple','lstm','Attention')
checkpointer.steps_per_checkpoint(500)
checkpointer.steps_per_prediction(2000)
# Do using LSTM cell - with attention mechanism
out_file = 'result/simple/lstm/attention.csv'
checkpointer.set_result_location(out_file)
lstm_net = lstm_simple.NeuralNet(review_summary_file, checkpointer, attention = True)
lstm_net.set_parameters(train_batch_size=5,test_batch_size=25, memory_dim=50,learning_rate=0.05)
lstm_net.begin_session()
lstm_net.form_model_graph()
lstm_net.fit()
lstm_net.predict()
lstm_net.store_test_predictions()
lstm_net.close_session()
