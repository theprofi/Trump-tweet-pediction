from sklearn.externals import joblib
import ParseOrganize
from prettytable import PrettyTable

SAVED_SESSIONS = "saved_sessions"
TWEETS_PATH = "my_tweets.txt"
# Data labels
BEFORE = 0
AFTER = 1


def load_models():
    # ===== Load =====
    algs = joblib.load(SAVED_SESSIONS + "/algs")
    adas = joblib.load(SAVED_SESSIONS + "/adas")
    return algs + adas


def print_results(models, my_tweets):
    def most_common_label(lst):
        return max(set(lst), key=lst.count)

    t = PrettyTable(['Tweet/Model', 'Majority'] + [m.__repr__() for m in models])
    preds = []
    for m in models:
        preds.append(m.predict(my_tweets))
    for i in range(len(my_tweets)):
        pred_of_cur = [p[i] for p in preds]
        t.add_row([my_tweets[i].lower(), most_common_label(pred_of_cur)] + pred_of_cur)
    print(t)


def main():
    models = load_models()
    my_tweets = ParseOrganize.get_tweets(TWEETS_PATH)
    print_results(models, my_tweets)


if __name__ == "__main__":
    main()
