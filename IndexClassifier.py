class IndexClassifier:
    def __init__(self, my_index):
        self.my_index = my_index

    def fit(self, train_x, train_y):
        pass

    def predict(self, vecs_list):
        predictions = []
        for vec in vecs_list:
            predictions.append(int(vec[self.my_index] > 0))
        return predictions
