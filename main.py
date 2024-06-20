from tqdm import tqdm
import json
from dataset import SemEval2016
from features import AnoVa_V1, AnoVa_V2, NERExtractor
from embedding import BERT, OneHot
from utils import *

# read data
dataset = SemEval2016()
train_data, test_data = dataset.get_data("train"), dataset.get_data("test")


# calculate bert embedding and inset to input data
bert = BERT()
targets_list = list(set([row["Target"] for row in train_data]))
target_embedder = OneHot(targets_list)
ner_extractor = NERExtractor()

for row in tqdm(train_data, desc="Processing embeddings [Train]"):
    row["embedding"] = bert.get_twitter_bert_embedding(row["Text"])
    row["target_embedding"] = target_embedder.encode(row["Target"])
    row["ner"] = ner_extractor.extract_entities(row["Text"])

for row in tqdm(test_data, desc="Processing embeddings [Test]"):
    row["embedding"] = bert.get_twitter_bert_embedding(row["Text"])
    row["target_embedding"] = target_embedder.encode(row["Target"])
    row["ner"] = ner_extractor.extract_entities(row["Text"])


# Calculate Anova Top Words
anova = AnoVa_V2(train_data)
anova_words_dict = anova.get_anova_top_words(n_words=10)

