import torch
from transformers import AutoTokenizer, AutoModel

class BERT:
    def __init__(self):
        print("Loading Twitter BERT model...")
        self.twitter_tokenizer = AutoTokenizer.from_pretrained("vinai/bertweet-base")
        self.twitter_model = AutoModel.from_pretrained("vinai/bertweet-base")
        print("Twitter BERT model Loaded.")

        print("Loading Base BERT model...")
        self.base_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        self.base_model = AutoModel.from_pretrained("bert-base-uncased")
        print("Base BERT model Loaded.")

    def get_twitter_bert_embedding(self, input_text):
        inputs = self.twitter_tokenizer(input_text, return_tensors='pt', truncation=True, padding=True)
        with torch.no_grad():
            outputs = self.twitter_model(**inputs)
        return list(outputs.last_hidden_state.mean(dim=1).squeeze().numpy())

    def get_base_bert_embedding(self, input_text):
        inputs = self.base_tokenizer(input_text, return_tensors='pt', truncation=True, padding=True)
        with torch.no_grad():
            outputs = self.base_model(**inputs)
        return list(outputs.last_hidden_state.mean(dim=1).squeeze().numpy())


class OneHot:
    def __init__(self, categories):
        self.categories = categories
        self.category_to_index = {category: idx for idx, category in enumerate(categories)}
        self.index_to_category = {idx: category for idx, category in enumerate(categories)}
        self.num_categories = len(categories)

    def encode(self, value):
        if value not in self.category_to_index:
            raise ValueError(f"Value '{value}' is not in the list of categories.")

        one_hot_vector = [0] * self.num_categories
        one_hot_vector[self.category_to_index[value]] = 1
        return one_hot_vector

    def decode(self, one_hot_vector):
        if len(one_hot_vector) != self.num_categories:
            raise ValueError(
                f"One hot vector length {len(one_hot_vector)} does not match number of categories {self.num_categories}.")

        index = one_hot_vector.index(1)
        return self.index_to_category[index]