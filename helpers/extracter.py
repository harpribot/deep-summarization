import numpy as np
import pandas as pd


class Spider:
    def __init__(self,num_reviews):
        """

        :param num_reviews:
        """
        self.num_reviews = num_reviews
        self.raw_data_file = None
        self.df = None

    def crawl_for_reviews_and_summary(self, input_file):
        """

        :param input_file:
        :return:
        """
        self.raw_data_file = input_file
        self.df = pd.DataFrame()
        self.df['Review'] = self.__crawl_review()
        self.df['Summary'] = self.__crawl_summary()

    def __crawl_review(self):
        """

        :return:
        """
        review_list = []
        print 'Crawling Reviews....'
        num_lines = 0
        with open(self.raw_data_file) as infile:
            for line in infile:
                if line.startswith('review/text'):
                    if num_lines >= self.num_reviews:
                        break
                    num_lines += 1
                    _,review = line.split('/text: ')
                    review_list.append(review)

        return np.array(review_list)

    def __crawl_summary(self):
        """

        :return:
        """
        summary_list = []
        print 'Crawling Summary....'
        num_lines = 0
        with open(self.raw_data_file) as infile:
            for line in infile:
                if line.startswith('review/summary'):
                    if num_lines >= self.num_reviews:
                        break
                    num_lines += 1
                    _,summary = line.split('/summary: ')
                    summary_list.append(summary)

        return np.array(summary_list)

    def save_review_summary_frame(self, output_file):
        """

        :param output_file:
        :return:
        """
        self.df.to_csv(output_file, index=False)
