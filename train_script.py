from algorithms.lstm import NeuralNet

review_summary_file = 'extracted_data/review_summary.csv'
out_file = 'result/test_results.csv'
net = NeuralNet(review_summary_file)
net.begin_session()
net.form_model_graph()
net.fit()
net.predict()
net.store_test_predictions(out_file)
