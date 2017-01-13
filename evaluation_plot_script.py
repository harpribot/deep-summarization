from helpers.plotter import Plotter
from helpers.metric import Calculator
import matplotlib.pyplot as plt

############## ALL GRU PLOTS ############################
result_file_1 = 'result/simple/gru/no_attention.csv'
result_file_2 = 'result/bidirectional/gru/no_attention.csv'
result_file_3 = 'result/stacked_simple/gru/no_attention.csv'
result_file_4 = 'result/stacked_bidirectional/gru/no_attention.csv'


result_file_description = ['gru_smpl', 'gru_bidr', 'gru_stack_smpl', 'gru_stack_bidr']
hypothesis_dir = 'metrics/hypothesis'
reference_dir = 'metrics/reference'

bleu_1 = []
bleu_2 = []
bleu_3 = []
bleu_4 = []
rouge = []


calculator = Calculator(3,hypothesis_dir,reference_dir)
calculator.load_result(result_file_1)
calculator.evaluate_all_ref_hyp_pairs()
bleu_1_val,bleu_2_val,bleu_3_val,bleu_4_val,rouge_val = calculator.get_all_metrics()
bleu_1.append(bleu_1_val)
bleu_2.append(bleu_2_val)
bleu_3.append(bleu_3_val)
bleu_4.append(bleu_4_val)
rouge.append(rouge_val)

calculator = Calculator(3,hypothesis_dir,reference_dir)
calculator.load_result(result_file_2)
calculator.evaluate_all_ref_hyp_pairs()
bleu_1_val,bleu_2_val,bleu_3_val,bleu_4_val,rouge_val = calculator.get_all_metrics()
bleu_1.append(bleu_1_val)
bleu_2.append(bleu_2_val)
bleu_3.append(bleu_3_val)
bleu_4.append(bleu_4_val)
rouge.append(rouge_val)

calculator = Calculator(3,hypothesis_dir,reference_dir)
calculator.load_result(result_file_3)
calculator.evaluate_all_ref_hyp_pairs()
bleu_1_val,bleu_2_val,bleu_3_val,bleu_4_val,rouge_val = calculator.get_all_metrics()
bleu_1.append(bleu_1_val)
bleu_2.append(bleu_2_val)
bleu_3.append(bleu_3_val)
bleu_4.append(bleu_4_val)
rouge.append(rouge_val)

calculator = Calculator(3,hypothesis_dir,reference_dir)
calculator.load_result(result_file_4)
calculator.evaluate_all_ref_hyp_pairs()
bleu_1_val,bleu_2_val,bleu_3_val,bleu_4_val,rouge_val = calculator.get_all_metrics()
bleu_1.append(bleu_1_val)
bleu_2.append(bleu_2_val)
bleu_3.append(bleu_3_val)
bleu_4.append(bleu_4_val)
rouge.append(rouge_val)

steps = calculator.get_steps()

plotter = Plotter()
plotter.set_metrics(bleu_1,bleu_2,bleu_3,bleu_4,rouge)
plotter.set_file_description(result_file_description)
plotter.set_steps(steps)
plotter.plot_all_metrics()


########## ALL LSTM PLOTS ####################
result_file_1 = 'result/simple/lstm/no_attention.csv'
result_file_2 = 'result/bidirectional/lstm/no_attention.csv'
result_file_3 = 'result/stacked_simple/lstm/no_attention.csv'
result_file_4 = 'result/stacked_bidirectional/lstm/no_attention.csv'


result_file_description = ['lstm_smpl','lstm_bidr','lstm_stack_smpl','lstm_stack_bidr']
hypothesis_dir = 'metrics/hypothesis'
reference_dir = 'metrics/reference'

bleu_1 = []
bleu_2 = []
bleu_3 = []
bleu_4 = []
rouge = []


calculator = Calculator(3,hypothesis_dir,reference_dir)
calculator.load_result(result_file_1)
calculator.evaluate_all_ref_hyp_pairs()
bleu_1_val,bleu_2_val,bleu_3_val,bleu_4_val,rouge_val = calculator.get_all_metrics()
bleu_1.append(bleu_1_val)
bleu_2.append(bleu_2_val)
bleu_3.append(bleu_3_val)
bleu_4.append(bleu_4_val)
rouge.append(rouge_val)

calculator = Calculator(3,hypothesis_dir,reference_dir)
calculator.load_result(result_file_2)
calculator.evaluate_all_ref_hyp_pairs()
bleu_1_val,bleu_2_val,bleu_3_val,bleu_4_val,rouge_val = calculator.get_all_metrics()
bleu_1.append(bleu_1_val)
bleu_2.append(bleu_2_val)
bleu_3.append(bleu_3_val)
bleu_4.append(bleu_4_val)
rouge.append(rouge_val)

calculator = Calculator(3,hypothesis_dir,reference_dir)
calculator.load_result(result_file_3)
calculator.evaluate_all_ref_hyp_pairs()
bleu_1_val,bleu_2_val,bleu_3_val,bleu_4_val,rouge_val = calculator.get_all_metrics()
bleu_1.append(bleu_1_val)
bleu_2.append(bleu_2_val)
bleu_3.append(bleu_3_val)
bleu_4.append(bleu_4_val)
rouge.append(rouge_val)

calculator = Calculator(3,hypothesis_dir,reference_dir)
calculator.load_result(result_file_4)
calculator.evaluate_all_ref_hyp_pairs()
bleu_1_val,bleu_2_val,bleu_3_val,bleu_4_val,rouge_val = calculator.get_all_metrics()
bleu_1.append(bleu_1_val)
bleu_2.append(bleu_2_val)
bleu_3.append(bleu_3_val)
bleu_4.append(bleu_4_val)
rouge.append(rouge_val)

steps = calculator.get_steps()

plotter = Plotter()
plotter.set_metrics(bleu_1,bleu_2,bleu_3,bleu_4,rouge)
plotter.set_file_description(result_file_description)
plotter.set_steps(steps)
plotter.plot_all_metrics()

#### GRU and LSTM Comparison plots #####

## SIMPLE
result_file_1 = 'result/simple/gru/no_attention.csv'
result_file_2 = 'result/simple/lstm/no_attention.csv'

result_file_description = ['gru_simple','lstm_simple']

bleu_1 = []
bleu_2 = []
bleu_3 = []
bleu_4 = []
rouge = []


calculator = Calculator(3,hypothesis_dir,reference_dir)
calculator.load_result(result_file_1)
calculator.evaluate_all_ref_hyp_pairs()
bleu_1_val,bleu_2_val,bleu_3_val,bleu_4_val,rouge_val = calculator.get_all_metrics()
bleu_1.append(bleu_1_val)
bleu_2.append(bleu_2_val)
bleu_3.append(bleu_3_val)
bleu_4.append(bleu_4_val)
rouge.append(rouge_val)

calculator = Calculator(3,hypothesis_dir,reference_dir)
calculator.load_result(result_file_2)
calculator.evaluate_all_ref_hyp_pairs()
bleu_1_val,bleu_2_val,bleu_3_val,bleu_4_val,rouge_val = calculator.get_all_metrics()
bleu_1.append(bleu_1_val)
bleu_2.append(bleu_2_val)
bleu_3.append(bleu_3_val)
bleu_4.append(bleu_4_val)
rouge.append(rouge_val)

steps = calculator.get_steps()

plotter = Plotter()
plotter.set_metrics(bleu_1,bleu_2,bleu_3,bleu_4,rouge)
plotter.set_file_description(result_file_description)
plotter.set_steps(steps)
plotter.plot_all_metrics()

## BIDIRECTIONAL
result_file_1 = 'result/bidirectional/gru/no_attention.csv'
result_file_2 = 'result/bidirectional/lstm/no_attention.csv'

result_file_description = ['gru_bidir','lstm_bidir']

bleu_1 = []
bleu_2 = []
bleu_3 = []
bleu_4 = []
rouge = []


calculator = Calculator(3,hypothesis_dir,reference_dir)
calculator.load_result(result_file_1)
calculator.evaluate_all_ref_hyp_pairs()
bleu_1_val,bleu_2_val,bleu_3_val,bleu_4_val,rouge_val = calculator.get_all_metrics()
bleu_1.append(bleu_1_val)
bleu_2.append(bleu_2_val)
bleu_3.append(bleu_3_val)
bleu_4.append(bleu_4_val)
rouge.append(rouge_val)

calculator = Calculator(3,hypothesis_dir,reference_dir)
calculator.load_result(result_file_2)
calculator.evaluate_all_ref_hyp_pairs()
bleu_1_val,bleu_2_val,bleu_3_val,bleu_4_val,rouge_val = calculator.get_all_metrics()
bleu_1.append(bleu_1_val)
bleu_2.append(bleu_2_val)
bleu_3.append(bleu_3_val)
bleu_4.append(bleu_4_val)
rouge.append(rouge_val)

steps = calculator.get_steps()

plotter = Plotter()
plotter.set_metrics(bleu_1,bleu_2,bleu_3,bleu_4,rouge)
plotter.set_file_description(result_file_description)
plotter.set_steps(steps)
plotter.plot_all_metrics()

## STACKED_SIMPLE
result_file_1 = 'result/stacked_simple/gru/no_attention.csv'
result_file_2 = 'result/stacked_simple/lstm/no_attention.csv'

result_file_description = ['gru_stacked','lstm_stacked']

bleu_1 = []
bleu_2 = []
bleu_3 = []
bleu_4 = []
rouge = []


calculator = Calculator(3,hypothesis_dir,reference_dir)
calculator.load_result(result_file_1)
calculator.evaluate_all_ref_hyp_pairs()
bleu_1_val,bleu_2_val,bleu_3_val,bleu_4_val,rouge_val = calculator.get_all_metrics()
bleu_1.append(bleu_1_val)
bleu_2.append(bleu_2_val)
bleu_3.append(bleu_3_val)
bleu_4.append(bleu_4_val)
rouge.append(rouge_val)

calculator = Calculator(3,hypothesis_dir,reference_dir)
calculator.load_result(result_file_2)
calculator.evaluate_all_ref_hyp_pairs()
bleu_1_val,bleu_2_val,bleu_3_val,bleu_4_val,rouge_val = calculator.get_all_metrics()
bleu_1.append(bleu_1_val)
bleu_2.append(bleu_2_val)
bleu_3.append(bleu_3_val)
bleu_4.append(bleu_4_val)
rouge.append(rouge_val)

steps = calculator.get_steps()

plotter = Plotter()
plotter.set_metrics(bleu_1,bleu_2,bleu_3,bleu_4,rouge)
plotter.set_file_description(result_file_description)
plotter.set_steps(steps)
plotter.plot_all_metrics()

## STACKED BIDIRECTIONAL
result_file_1 = 'result/stacked_bidirectional/gru/no_attention.csv'
result_file_2 = 'result/stacked_bidirectional/lstm/no_attention.csv'

result_file_description = ['gru_stack_bidir','lstm_stack_bidir']

bleu_1 = []
bleu_2 = []
bleu_3 = []
bleu_4 = []
rouge = []


calculator = Calculator(3,hypothesis_dir,reference_dir)
calculator.load_result(result_file_1)
calculator.evaluate_all_ref_hyp_pairs()
bleu_1_val,bleu_2_val,bleu_3_val,bleu_4_val,rouge_val = calculator.get_all_metrics()
bleu_1.append(bleu_1_val)
bleu_2.append(bleu_2_val)
bleu_3.append(bleu_3_val)
bleu_4.append(bleu_4_val)
rouge.append(rouge_val)

calculator = Calculator(3,hypothesis_dir,reference_dir)
calculator.load_result(result_file_2)
calculator.evaluate_all_ref_hyp_pairs()
bleu_1_val,bleu_2_val,bleu_3_val,bleu_4_val,rouge_val = calculator.get_all_metrics()
bleu_1.append(bleu_1_val)
bleu_2.append(bleu_2_val)
bleu_3.append(bleu_3_val)
bleu_4.append(bleu_4_val)
rouge.append(rouge_val)

steps = calculator.get_steps()

plotter = Plotter()
plotter.set_metrics(bleu_1,bleu_2,bleu_3,bleu_4,rouge)
plotter.set_file_description(result_file_description)
plotter.set_steps(steps)
plotter.plot_all_metrics()

# SHOW ALL PLOTS
plt.show()
