import numpy as np
import pandas as pd
from nltk.tokenize import wordpunct_tokenize


class Mapper:
    def __init__(self):
        """

        """
        self.map = dict()
        self.map["GO"] = 0
        self.revmap = dict()
        self.revmap[0] = "GO"
        self.counter = 1
        self.review_max_words = 100
        self.summary_max_words = 100
        self.rev_sum_pair = None
        self.review_tensor = None
        self.summary_tensor = None
        self.review_tensor_reverse = None

    def generate_vocabulary(self, review_summary_file):
        """

        :param review_summary_file:
        :return:
        """
        self.rev_sum_pair = pd.read_csv(review_summary_file, header=0).values

        for review,summary in self.rev_sum_pair:
            rev_lst = wordpunct_tokenize(review)
            sum_lst = wordpunct_tokenize(summary)
            self.__add_list_to_dict(rev_lst)
            self.__add_list_to_dict(sum_lst)

        # Now store the "" empty string as the last word of the voacabulary
        self.map[""] = len(self.map)
        self.revmap[len(self.map)] = ""

    def __add_list_to_dict(self, word_lst):
        """

        :param word_lst:
        :return:
        """
        for word in word_lst:
            word = word.lower()
            if word not in self.map:
                self.map[word] = self.counter
                self.revmap[self.counter] = word
                self.counter += 1

    def get_tensor(self, reverseflag=False):
        """

        :param reverseflag:
        :return:
        """
        self.review_tensor = self.__generate_tensor(is_review=True)
        if reverseflag:
            self.review_tensor_reverse = self.__generate_tensor(is_review=True, reverse=True)

        self.summary_tensor = self.__generate_tensor(is_review=False)

        if reverseflag:
            return self.review_tensor,self.review_tensor_reverse,self.summary_tensor
        else:
            return self.review_tensor, self.summary_tensor

    def __generate_tensor(self, is_review, reverse=False):
        """

        :param is_review:
        :param reverse:
        :return:
        """
        seq_length = self.review_max_words if is_review else self.summary_max_words
        total_rev_summary_pairs = self.rev_sum_pair.shape[0]
        data_tensor = np.zeros([total_rev_summary_pairs,seq_length])

        sample = self.rev_sum_pair[0::, 0] if is_review else self.rev_sum_pair[0::, 1]

        for index, entry in enumerate(sample.tolist()):
            index_lst = np.array([self.map[word.lower()] for word in wordpunct_tokenize(entry)])
            # reverse if want to get backward form
            if reverse:
                index_lst = index_lst[::-1]
            # Pad the list
            if len(index_lst) <= seq_length:
                index_lst = np.lib.pad(index_lst, (0,seq_length - index_lst.size), 'constant', constant_values=(0, 0))
            else:
                index_lst = index_lst[0:seq_length]

            data_tensor[index] = index_lst

        return data_tensor

    def get_seq_length(self):
        """

        :return:
        """
        return self.review_max_words

    def get_vocabulary_size(self):
        """

        :return:
        """
        return len(self.map)

    def get_reverse_map(self):
        """

        :return:
        """
        return self.revmap
