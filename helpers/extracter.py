import numpy as np
import pandas as pd

class Spider:
    def __init__(self):
        pass

    def crawl_for_reviews_and_summary(self, input_file):
        self.raw_data_file = input_file
        self.df = pd.DataFrame()
        self.df['Review'] = self.__crawl_review()
        self.df['Summary'] = self.__crawl_summary()

    def __crawl_review(self):
        review_list = []
        print 'Crawling Reviews....'
        with open(self.raw_data_file) as infile:
            for line in infile:
                if(line.startswith('review/text')):
                    _,review = line.split('/text: ')
                    review_list.append(review)

        return np.array(review_list)

    def __crawl_summary(self):
        summary_list = []
        print 'Crawling Summary....'
        with open(self.raw_data_file) as infile:
            for line in infile:
                if(line.startswith('review/summary')):
                    _,summary = line.split('/summary: ')
                    summary_list.append(summary)

        return np.array(summary_list)


    def save_review_summary_frame(self, output_file):
        self.df.to_csv(output_file, index=False)
