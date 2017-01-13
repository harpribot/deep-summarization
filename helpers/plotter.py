import matplotlib.pyplot as plt

class Plotter:
    def __init__(self):
        """

        """
        self.steps = None
        self.file_desc = None
        self.bleu_1 = self.bleu_2 = self.bleu_3 = self.bleu_4 = self.rouge = None

    def set_metrics(self, bleu_1, bleu_2, bleu_3, bleu_4, rouge):
        """

        :param bleu_1:
        :param bleu_2:
        :param bleu_3:
        :param bleu_4:
        :param rouge:
        :return:
        """
        self.bleu_1 = bleu_1
        self.bleu_2 = bleu_2
        self.bleu_3 = bleu_3
        self.bleu_4 = bleu_4
        self.rouge = rouge

    def set_steps(self,steps):
        """

        :param steps:
        :return:
        """
        self.steps = steps

    def set_file_description(self,file_desc):
        """

        :param file_desc:
        :return:
        """
        self.file_desc = file_desc

    def plot_all_metrics(self):
        """

        :return:
        """
        plt.figure()
        self.plot_one_metric(self.bleu_1, 'BLEU Score - 1-Gram')
        plt.figure()
        self.plot_one_metric(self.bleu_2, 'BLEU Score - 2-Gram')
        plt.figure()
        self.plot_one_metric(self.bleu_3, 'BLEU Score - 3-Gram')
        plt.figure()
        self.plot_one_metric(self.bleu_4, 'BLEU Score - 4-Gram')
        plt.figure()
        self.plot_one_metric(self.rouge, 'ROUGE Score')

    def plot_one_metric(self, models_metric, title):
        """

        :param models_metric:
        :param title:
        :return:
        """
        for index, model_metric in enumerate(models_metric):
            plt.plot(self.steps, model_metric, label=self.file_desc[index])
        plt.title(title)
        plt.legend()
        plt.xlabel('Number of batches')
        plt.ylabel('Score')

    def show_plots(self):
        """

        :return:
        """
        plt.show()
