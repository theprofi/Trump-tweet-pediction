import string
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import decomposition
from ParseOrganize import BEFORE, AFTER


class TweetToLetvec:
    def __init__(self,):
        self.len_before = 0
        self.len_after = 0
        self.train_x = []

    def init(self, tweets_before, tweets_after):
        for tweet in tweets_before + tweets_after:
            self.train_x.append(self.convert(tweet))
        self.len_before = len(tweets_before)
        self.len_after = len(tweets_after)

    def get_train_xy(self, ):
        train_y = [BEFORE] * self.len_before + [AFTER] * self.len_after
        return self.train_x, train_y

    def convert(self, tweet):
        # The vector is of the size of the english alphabet + 1 for space char
        letvec = np.zeros(27)
        for letter in tweet:
            if letter.lower() in 'abcdefghijklmnopqrstuvwxyz':
                letvec[string.ascii_lowercase.index(letter.lower())] += 1
            elif letter.lower() == " ":
                letvec[26] += 1
        return letvec


class TweetToBow:
    def __init__(self):
        self.len_before = 0
        self.len_after = 0
        self.train_x = []
        self.vectorizer = None

    def init(self, tweets_before, tweets_after):
        self.vectorizer = CountVectorizer()
        self.train_x = self.vectorizer.fit_transform(tweets_before + tweets_after).toarray()
        self.len_before = len(tweets_before)
        self.len_after = len(tweets_after)
        return self

    def get_train_xy(self, ):
        train_y = [BEFORE] * self.len_before + [AFTER] * self.len_after
        return self.train_x, train_y

    def convert(self, tweet):
        return self.vectorizer.transform([tweet]).toarray()[0]


class TweetToBowWithPca:
    def __init__(self, new_dim=900):
        self.len_before = 0
        self.len_after = 0
        self.train_x = []
        self.train_y = []
        self.new_dim = new_dim
        self.pca = None
        self.ttb = None

    def init(self, tweets_before, tweets_after):
        self.ttb = TweetToBow().init(tweets_before, tweets_after)
        self.train_x, self.train_y = self.ttb.get_train_xy()
        self.pca = decomposition.PCA(n_components=self.new_dim)
        self.train_x = self.pca.fit_transform(self.train_x)
        self.len_before = len(tweets_before)
        self.len_after = len(tweets_after)
        return self

    def get_train_xy(self, ):
        train_y = [BEFORE] * self.len_before + [AFTER] * self.len_after
        return self.train_x, train_y

    def convert(self, tweet):
        vec = self.ttb.convert(tweet)
        return self.pca.transform([vec])[0]


class TweetToTweet:
    def __init__(self):
        self.len_before = 0
        self.len_after = 0
        self.train_x = []

    def init(self, tweets_before, tweets_after):
        self.train_x = tweets_before + tweets_after
        self.len_before = len(tweets_before)
        self.len_after = len(tweets_after)
        return self

    def get_train_xy(self, ):
        train_y = [BEFORE] * self.len_before + [AFTER] * self.len_after
        return self.train_x, train_y

    def convert(self, tweet):
        return tweet