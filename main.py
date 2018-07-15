from sklearn.neural_network import MLPClassifier
from sklearn import svm
from sklearn.neighbors.nearest_centroid import NearestCentroid
from sklearn.externals import joblib
from sklearn import linear_model
import ParseOrganize
from ClassifiersWrapper import ClassifiersWrapper
import Converters
import IndexClassifier
import Adaboost

# Data labels
BEFORE = 0
AFTER = 1
# Paths
TWEETS_BEFORE_PATH = "datasets/before.txt"
TWEETS_BEFORE_YEAR_BEFORE_PATH = "datasets/year_before.txt"
TWEETS_AFTER_PATH = "datasets/after.txt"
SAVED_SESSIONS = "saved_sessions"


def run_tests(po, algs):
    # Three helper functions for this function
    def show_test_results(test_name, right_preds, total_tests, algs):
        print("=========", test_name, "=========")
        for i in range(len(right_preds)):
            print(algs[i], right_preds[i], "out of", total_tests, "=", str((right_preds[i] / total_tests) * 100) + "%")

    def test_on_year_before(po, algs):
        # The amount of correct predictions for each algorithm
        right_predictions = [0] * len(algs)
        for t in po.test_x_year_before:
            for i, alg in enumerate(algs):
                if alg.predict([t])[0] == BEFORE:
                    right_predictions[i] += 1
        show_test_results("Test on the tweets that has been tweeted in the range of June 2014 - June 2015",
                          right_predictions, len(po.test_x_year_before), algs)

    def test_random_without_year_before(po, algs):
        # The amount of correct predictions for each algorithm
        right_predictions = [0] * len(algs)
        for t in po.tweets_before[ParseOrganize.FOR_TESTS]:
            for i, alg in enumerate(algs):
                if alg.predict([t])[0] == BEFORE:
                    right_predictions[i] += 1
        for t in po.tweets_after[ParseOrganize.FOR_TESTS]:
            for i, alg in enumerate(algs):
                if alg.predict([t])[0] == AFTER:
                    right_predictions[i] += 1
        show_test_results("Test on random " + str(ParseOrganize.TWEETS_TO_TEST) +
                          " tweets. Not including those which has been tweeted in the range of June 2014 - June 2015",
                          right_predictions, len(po.tweets_before[ParseOrganize.FOR_TESTS])
                          + len(po.tweets_after[ParseOrganize.FOR_TESTS]), algs)

    # ======================================
    # ====== run_tests function body =======
    # ======================================
    test_on_year_before(po, algs)
    test_random_without_year_before(po, algs)


def get_trained_algs(is_load_saved=False):
    # Four helper functions for this function
    def create_new_other_classifiers():
        """
        Creates an object for a classifier with the class it uses to convert the raw data to a vector
        :return: Returns a list of wrappers which wraps the conversion object and the algorithm object
        """
        return [
            ClassifiersWrapper(MLPClassifier(hidden_layer_sizes=(500, 500)), Converters.TweetToBow(), "MLP with BOW"),
            ClassifiersWrapper(NearestCentroid(), Converters.TweetToBow(), "NearestCentroid with BOW"),
            ClassifiersWrapper(linear_model.LogisticRegression(), Converters.TweetToBow(), "Log Reg with BOW"),
            ClassifiersWrapper(svm.LinearSVC(), Converters.TweetToBow(), "SVM with BOW"),
            ClassifiersWrapper(MLPClassifier(hidden_layer_sizes=(500, 500)), Converters.TweetToLetvec(),
                               "MLP with LetVec"),
            ClassifiersWrapper(NearestCentroid(), Converters.TweetToLetvec(), "NearestCentroid with LetVec"),
            ClassifiersWrapper(linear_model.LogisticRegression(), Converters.TweetToLetvec(), "Log Reg with LetVec"),
            ClassifiersWrapper(svm.LinearSVC(), Converters.TweetToLetvec(), "SVM with LetVec")]

    def create_new_adaboost_classifiers(po, algs):
        """
        :param algs: The algorithms which will be the classifiers of one of the adaboosts
        :param po: The object which holds the tweets organized and parsed
        :return: Returns a list containing the two adaboost that are created in this function
        """
        adas = [ClassifiersWrapper(Adaboost.Adaboost(algs), Converters.TweetToTweet(),
                                   "Adaboost with all the algorithms from before as classifiers")]
        ic_list = []
        for i in range(900):
            ic = ClassifiersWrapper(IndexClassifier.IndexClassifier(my_index=i),
                                    Converters.TweetToTweet())
            ic.train(po.get_tweets_before_for_others(), po.get_tweets_after_for_others())
            ic_list.append(ic)
        adas.append(ClassifiersWrapper(Adaboost.Adaboost(ic_list), Converters.TweetToBowWithPca(new_dim=900),
                                       "Adaboost with every index as a classfier (after PCA to 900)"))
        return adas

    def load_saved():
        algs = joblib.load(SAVED_SESSIONS + "/algs")
        adas = joblib.load(SAVED_SESSIONS + "/adas")
        po = joblib.load(SAVED_SESSIONS + "/po")
        return po, algs, adas

    def create_new_models_save_and_load():
        # ===== Initialize =====
        po = ParseOrganize.ParseOrganize(TWEETS_BEFORE_PATH, TWEETS_AFTER_PATH, TWEETS_BEFORE_YEAR_BEFORE_PATH)
        algs = create_new_other_classifiers()
        adas = create_new_adaboost_classifiers(po, algs)
        # ===== Train all the algorithms =====
        for alg in algs:
            alg.train(po.get_tweets_before_for_others(), po.get_tweets_after_for_others())
        for ada in adas:
            ada.train(po.get_tweets_before_for_adaboost(), po.get_tweets_after_for_adaboost())
        # ===== Save the work above =====
        joblib.dump(algs, SAVED_SESSIONS + "/algs")
        joblib.dump(adas, SAVED_SESSIONS + "/adas")
        joblib.dump(po, SAVED_SESSIONS + "/po")
        return po, algs, adas

    # ======================================
    # ====== get_trained_algs function body =======
    # ======================================
    if is_load_saved:
        return load_saved()
    else:
        return create_new_models_save_and_load()


def main():
    po, algs, adas = get_trained_algs(is_load_saved=True)
    run_tests(po, algs + adas)


if __name__ == "__main__":
    main()
