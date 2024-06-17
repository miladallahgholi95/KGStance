class Classifier:
    def __init__(self, train_series, test_series, one_hot_column="", embedding_column_name=""):
        self.has_one_hot = one_hot_column and train_series[one_hot_column].count() > 0
        self.has_embedding = embedding_column_name and train_series[embedding_column_name].count() > 0


    # def one_hot_generator(self):
