from models import gru_simple

# Get the review summary file
review_summary_file = 'extracted_data/review_summary.csv'

# Do using GRU cell - with attention mechanism
out_file = 'result/simple/gru/attention.csv'
gru_net = gru_simple.NeuralNet(review_summary_file, attention = True)
gru_net.set_parameters(batch_size=15, memory_dim=15,learning_rate=0.05)
gru_net.begin_session()
gru_net.form_model_graph()
gru_net.fit()
gru_net.predict()
gru_net.store_test_predictions(out_file)
gru_net.close_session()
