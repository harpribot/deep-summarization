import pandas as pd
from metrics import tester


class Calculator:
    def __init__(self,steps_per_prediction,hypothesis_store_loc, reference_store_loc):
        """

        :param steps_per_prediction:
        :param hypothesis_store_loc:
        :param reference_store_loc:
        """
        self.steps_per_prediction = steps_per_prediction
        self.hypothesis_store_loc = hypothesis_store_loc
        self.reference_store_loc = reference_store_loc
        self.result = None
        self.steps = None

    def load_result(self,result_file):
        """

        :param result_file:
        :return:
        """
        self.result = pd.read_csv(result_file, header=0)
        self.__scrape_reference()
        self.__scrape_all_hypotheses()

    def __scrape_reference(self):
        """

        :return:
        """
        self.reference = self.result['true_summary'].values

    def __scrape_all_hypotheses(self):
        """

        :return:
        """
        # Drop review and true summary
        self.hypotheses = self.result.drop(self.result.columns[[0, 1]], axis=1)
        self.num_hypothesis = self.hypotheses.shape[1]
        self.hypotheses = self.hypotheses.values

    def evaluate_all_ref_hyp_pairs(self):
        """

        :return:
        """
        self.bleu_1 = []
        self.bleu_2 = []
        self.bleu_3 = []
        self.bleu_4 = []
        self.rouge = []
        self.steps = range(0,
                           self.num_hypothesis * self.steps_per_prediction,
                           self.steps_per_prediction)

        for hypothesis in self.hypotheses.T:
            bleu_1,bleu_2, bleu_3, bleu_4, rouge = self.__evaluate_one_ref_hypothesis_pair(self.reference,hypothesis)
            self.bleu_1.append(bleu_1)
            self.bleu_2.append(bleu_2)
            self.bleu_3.append(bleu_3)
            self.bleu_4.append(bleu_4)
            self.rouge.append(rouge)

    def __evaluate_one_ref_hypothesis_pair(self, refs, hyps):
        """

        :param refs:
        :param hyps:
        :return:
        """
        # Dump the data into the corresponding files
        for index,pair in enumerate(zip(refs,hyps)):
            file_ref_nm = self.reference_store_loc + '/ref' + str(index) + '.txt'
            file_hyp_nm = self.hypothesis_store_loc + '/gen' + str(index) + '.txt'
            ref_file = open(file_ref_nm,'w')
            hyp_file = open(file_hyp_nm,'w')
            ref_file.write(str(pair[0]))
            if pair[1] != 'nan':
                hyp_file.write(str(pair[1]))
            else:
                hyp_file.write('')
        # Call the tester function to get the evaluations
        return tester.main()

    def get_all_metrics(self):
        """

        :return:
        """
        return self.bleu_1, self.bleu_2,self.bleu_3,self.bleu_4,self.bleu_4

    def get_steps(self):
        """

        :return:
        """
        return self.steps
