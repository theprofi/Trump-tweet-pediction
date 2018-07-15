class ClassifiersWrapper:
    def __init__(self, clf, converter, repr="no repr"):
        self.clf = clf
        self.converter = converter
        self.repr = repr

    def train(self, tweets_before, tweets_after):
        self.converter.init(tweets_before, tweets_after)
        train_x, train_y = self.converter.get_train_xy()
        self.clf.fit(train_x, train_y)

    def predict(self, tweets_list):
        vecs_list = []
        for tweet in tweets_list:
            vecs_list.append(self.converter.convert(tweet))
        return self.predict_converted(vecs_list)

    def predict_converted(self, vecs_list):
        return self.clf.predict(vecs_list)

    def __repr__(self):
        return self.repr
