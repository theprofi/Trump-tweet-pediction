from ParseOrganize import BEFORE, AFTER
import math


class Adaboost:
    def __init__(self, algo_wrappers_list):
        """

        :param algo_wrappers_list: the models that are used as the classifiers for this adaboost
        """
        self.each_algo_preds = []
        self.algo_wrappers_list = algo_wrappers_list
        self.algos_weights = []
        self.tweets_weights = []
        self.trainx_len = 0
        self.train_y = []

    def get_best(self):
        lowest_err = 99999999
        best_algo_idx = 0
        mistakes = []
        corrects = []
        for alg_idx, pred_vec in enumerate(self.each_algo_preds):
            # For each classifier (alg_idx) check the on how much tweets it predicted correctly. The predictions are
            # stored in pred_vec and taked from each_algo_preds which was initialized in fit().
            cur_err = 0
            for i in range(self.trainx_len):
                if pred_vec[i] != self.train_y[i]:
                    cur_err += self.tweets_weights[i]
                    mistakes.append(i)
                else:
                    corrects.append(i)
            if cur_err < lowest_err:
                lowest_err = cur_err
                best_algo_idx = alg_idx
        return lowest_err, best_algo_idx, corrects, mistakes

    def fit(self, train_x, train_y):
        """
        Initializes the predictions of all the classifiers in advance and then that uses the results
        when it trains itself on those classifiers.
        :param train_x:
        :param train_y:
        :return:
        """
        self.train_y = train_y
        self.trainx_len = len(train_x)
        for algow in self.algo_wrappers_list:
            self.each_algo_preds.append(algow.predict(train_x))
        self.tweets_weights = [1 / self.trainx_len] * self.trainx_len
        self.algos_weights = [1] * len(self.algo_wrappers_list)
        self.train()

    def train(self):
        for _ in range(len(self.algo_wrappers_list)):
            lowest_err, best_algo_idx, corretcs, mistakes = self.get_best()
            if lowest_err >= 1:
                break
            self.algos_weights[best_algo_idx] = 0.5 * math.log((1 - lowest_err) / lowest_err)
            sum_of_weights = 0
            for tweet_idx in corretcs:
                self.tweets_weights[tweet_idx] *= math.exp(self.algos_weights[best_algo_idx])
                sum_of_weights += self.tweets_weights[tweet_idx]
            for tweet_idx in corretcs:
                self.tweets_weights[tweet_idx] *= math.exp(self.algos_weights[-best_algo_idx])
                sum_of_weights += self.tweets_weights[tweet_idx]
            # normalize
            for tweet in self.tweets_weights:
                tweet /= sum_of_weights

    def predict(self, tweets_list):
        """
        Predict a new tweet by considering the prediction of each classifier of this adaboost and its weight.
        :param tweets_list: list containing the tweets we want to predict
        :return: 0 or 1
        """
        predictions = []
        for tweet in tweets_list:
            answer = 0
            for i, algw in enumerate(self.algo_wrappers_list):
                cur_ans = -1 if algw.predict([tweet])[0] == BEFORE else AFTER
                answer += self.algos_weights[i] * cur_ans
            predictions.append(AFTER) if answer >= 0 else predictions.append(BEFORE)
        return predictions
