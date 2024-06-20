from tqdm import tqdm
import json
from dataset import SemEval2016
from features import AnoVa_V1, AnoVa_V2, NERExtractor
from embedding import BERT, OneHot
from knowledge_graph import WordNet
from utils import *

# read data
dataset = SemEval2016()
train_data, test_data = dataset.get_data("train"), dataset.get_data("test")

# Calculate Anova Top Words
anova = AnoVa_V2(train_data)
anova_words_dict = anova.get_anova_top_words(n_words=10)

# Create Knowledge Graph
wordnet_extractor = WordNet()
label_target_graph = {}
full_knowledge_graph = []
MAX_LEVEL = 2
for label, target_dict in tqdm(anova_words_dict.items(), desc="Create Knowledge Graph"):
    label_target_graph[label] = {}
    for target, words_dict in target_dict.items():
        words = [word for word, score in words_dict.items()]
        relations = []
        for word in words:
            relations += wordnet_extractor.extract_keywords_with_levels(word, MAX_LEVEL)
        relations = wordnet_extractor.remove_duplicates(relations)
        label_target_graph[label][target] = relations
        full_knowledge_graph += relations

full_knowledge_graph = wordnet_extractor.remove_duplicates(full_knowledge_graph)

# calculate bert embedding & one hot embedding & ner list
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




