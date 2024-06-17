from embedding import BERT


class Classifier:
    def __init__(self, train_series, test_series, one_hot_column="", embedding_column_name=""):
        self.has_one_hot = one_hot_column and train_series[one_hot_column].count() > 0
        self.has_embedding = embedding_column_name and train_series[embedding_column_name].count() > 0
        self.train_set = train_series
        self.test_set = test_series
        self.embedding_column = embedding_column_name
        self.one_hot_column = one_hot_column

    def one_hutter(self, value, uniqueValues):
        array = [0 for i in range(len(uniqueValues))]
        for i in range(len(uniqueValues)):
            if uniqueValues[i] == value:
                array[i] = 1
                return array

        return array

    def one_hot_generator(self):
        uniqueValuesx = self.train_set["Target"].unique()

        self.train_set["Target_One_Hot"] = self.train_set["Target"].apply(lambda x: self.one_hutter(x, uniqueValuesx))
        self.test_set["Target_One_Hot"] = self.test_set["Target"].apply(lambda x: self.one_hutter(x, uniqueValuesx))

        self.one_hot_column = "Target_One_Hot"

    def embedding_generator(self):
        myBert = BERT()

        self.train_set["Embeddings"] = self.train_set["Tweet"].apply(myBert.get_text_bert_embedding)
        self.test_set["Embeddings"] = self.test_set["Tweet"].apply(myBert.get_text_bert_embedding)

        self.embedding_column = "Embeddings"
