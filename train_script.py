from algorithms import lstm, gru

# Get the review summary file
review_summary_file = 'extracted_data/review_summary.csv'

'''
# Do using GRU cell - without attention mechanism
out_file = 'result/test_results_gru_absence_attention.csv'
gru_net = gru.NeuralNet(review_summary_file)
gru_net.set_parameters(batch_size=5, memory_dim=15,learning_rate=0.05)
gru_net.begin_session()
gru_net.form_model_graph()
gru_net.fit()
gru_net.predict()
gru_net.store_test_predictions(out_file)
gru_net.close_session()

# Do using LSTM cell - without attention mechanism
out_file = 'result/test_results_lstm_absence_attention.csv'
lstm_net = lstm.NeuralNet(review_summary_file)
lstm_net.set_parameters(batch_size=5, memory_dim=15,learning_rate=0.05)
lstm_net.begin_session()
lstm_net.form_model_graph()
lstm_net.fit()
lstm_net.predict()
lstm_net.store_test_predictions(out_file)
lstm_net.close_session()

# Do using GRU cell - with attention mechanism
out_file = 'result/test_results_gru_with_attention.csv'
gru_net = gru.NeuralNet(review_summary_file, attention = True)
gru_net.set_parameters(batch_size=5, memory_dim=15,learning_rate=0.05)
gru_net.begin_session()
gru_net.form_model_graph()
gru_net.fit()
gru_net.predict()
gru_net.store_test_predictions(out_file)
gru_net.close_session()
'''
# Do using LSTM cell - with attention mechanism
out_file = 'result/test_results_lstm_with_attention.csv'
lstm_net = lstm.NeuralNet(review_summary_file, attention = True)
lstm_net.set_parameters(batch_size=5, memory_dim=15,learning_rate=0.05)
lstm_net.begin_session()
lstm_net.form_model_graph()
lstm_net.fit()
lstm_net.predict()
lstm_net.store_test_predictions(out_file)
lstm_net.close_session()

