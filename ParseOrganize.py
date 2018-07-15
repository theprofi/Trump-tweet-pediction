import math
import random
from sklearn.neural_network import MLPClassifier
from sklearn import svm
import numpy as np
from sklearn.neighbors.nearest_centroid import NearestCentroid
from sklearn.externals import joblib
from nltk.tokenize import word_tokenize
from sklearn import linear_model
import string
from sklearn import decomposition

TWEETS_TO_TEST = 6000
# Data labels
AFTER = 1
BEFORE = 0
FOR_TESTS = 0
FOR_TRAIN_OTHERS = 1
FOR_TRAIN_ADABOOSTS = 2


class ParseOrganize:
    def __init__(self, tweets_before_path, tweets_after_path, tweets_before_year_before_path):
        """
        Each of the list tweets_before and tweets_after contains the tweets that are allocated for test
        in the first cell, the tweets that are allocated for the adaboost callsifiers to learn in the
        3rd cell and the tweets that are allocated for the other callsifers to learn in the 2nd cell.

        :param tweets_before_path: the path of the file containing the tweets before the announcement
        :param tweets_after_path: the path of the file containing the tweets after the announcement
        :param tweets_before_year_before_path: the path of the file containing the tweets
        """
        self.tweets_before = [[], [], []]
        self.tweets_after = [[], [], []]
        self.test_x_year_before = []
        self.organize_data(tweets_before_path, tweets_after_path, tweets_before_year_before_path)

    def organize_data(self, tweets_before_path, tweets_after_path, tweets_before_year_before_path):
        """
        Initializes the lists

        :param tweets_before_path: see __init__
        :param tweets_after_path: see __init__
        :param tweets_before_year_before_path: see __init__
        in the range June 2024 - June 2015
        :return: nothing
        """
        self.test_x_year_before = get_tweets(tweets_before_year_before_path)
        before = get_tweets(tweets_before_path)
        after = get_tweets(tweets_after_path)
        # Shuffle to eliminate any time related influences
        random.shuffle(before)
        random.shuffle(after)
        # From the all the tweets allocated TWEETS_TO_TEST for tests
        for i in range(TWEETS_TO_TEST // 2):
            self.tweets_before[FOR_TESTS].append(before.pop())
            self.tweets_after[FOR_TESTS].append(after.pop())
        # The remainig tweets allocate to adaboost and to other models
        self.tweets_before[FOR_TRAIN_OTHERS] = before[:len(before) // 2]
        self.tweets_after[FOR_TRAIN_OTHERS] = after[:len(before) // 2]
        self.tweets_before[FOR_TRAIN_ADABOOSTS] = before[len(before) // 2:]
        self.tweets_after[FOR_TRAIN_ADABOOSTS] = after[len(before) // 2:]

    # From the lists tweets_before, tweets_after get the tweets allocated for each model

    def get_tweets_before_for_others(self, ):
        return self.tweets_before[FOR_TRAIN_OTHERS]

    def get_tweets_after_for_others(self, ):
        return self.tweets_after[FOR_TRAIN_OTHERS]

    def get_tweets_before_for_adaboost(self, ):
        return self.tweets_before[FOR_TRAIN_ADABOOSTS]

    def get_tweets_after_for_adaboost(self, ):
        return self.tweets_after[FOR_TRAIN_ADABOOSTS]


def get_tweets(path):
    """
    :param path: the path of the file containing tweets seperated by lines
    :return: a list containing each tweet as a string
    """
    with open(path, encoding="utf8") as f:
        content = f.readlines()
    return [x.strip().upper() for x in content]


