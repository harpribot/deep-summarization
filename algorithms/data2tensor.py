import numpy as np
import pandas as pd
from nltk.tokenize import wordpunct_tokenize

class Mapper:
    def __init__(self):
        self.map = {}
        self.map["GO"] = 0
        self.revmap = {}
        self.revmap[0] = "GO"
        self.counter = 1
        self.review_max_words = 200
        self.summary_max_words = 200

    def generate_vocabulary(self, review_summary_file):
        self.rev_sum_pair = pd.read_csv(review_summary_file,header=0).values

        for review,summary in self.rev_sum_pair:
            rev_lst = wordpunct_tokenize(review)
            sum_lst = wordpunct_tokenize(summary)
            self.__add_list_to_dict(rev_lst)
            self.__add_list_to_dict(sum_lst)
            # Update the max words in reviews and summary
            '''
            if(len(rev_lst) > self.review_max_words):
                self.review_max_words = len(rev_lst)
            if(len(sum_lst) > self.summary_max_words):
                self.summary_max_words = len(sum_lst)
            '''

        # Now store the "" empty string as the last word of the voacabulary
        self.map[""] = len(self.map)
        self.revmap[len(self.map)] = ""

    def __add_list_to_dict(self,word_lst):
        for word in word_lst:
            word = word.lower()
            if word not in self.map:
                self.map[word] = self.counter
                self.revmap[self.counter] = word
                self.counter += 1


    def get_tensor(self):
        self.review_tensor = self.__generate_tensor(isReview=True)
        self.summary_tensor = self.__generate_tensor(isReview=False)

        return self.review_tensor, self.summary_tensor

    def __generate_tensor(self,isReview):
        seq_length = self.review_max_words if isReview else self.summary_max_words
        total_rev_summary_pairs = self.rev_sum_pair.shape[0]
        data_tensor = np.zeros([total_rev_summary_pairs,seq_length])

        sample = self.rev_sum_pair[0::,0] if isReview else self.rev_sum_pair[0::,1]

        for index, entry in enumerate(sample.tolist()):
            index_lst = np.array([self.map[word.lower()] for word in wordpunct_tokenize(entry)])
            if len(index_lst) <= seq_length:
                index_lst = np.lib.pad(index_lst, (0,seq_length - index_lst.size), 'constant', constant_values=(0, 0))
            else:
                index_lst = index_lst[0:seq_length]

            data_tensor[index] = index_lst

        return data_tensor

    def get_seq_length(self):
        return self.review_max_words


    def get_vocabulary_size(self):
        return len(self.map)

    def get_reverse_map(self):
        return self.revmap
