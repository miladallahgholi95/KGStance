from tqdm import tqdm
import pickle
from dataset import SemEval2016
from features import AnoVa_V1, AnoVa_V2, NERExtractor
from embedding import BERT, OneHot
from knowledge_graph import WordNet
from utils import *
from graph_analysis import MultiSourceRandomWalk

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
MAX_LEVEL = 3
for label, target_dict in tqdm(anova_words_dict.items(), desc="Create Knowledge Graph"):
    label_target_graph[label] = {}
    for target, words_dict in target_dict.items():
        words = [word for word, score in words_dict.items()]
        relations = []
        for word in words:
            if len(word) > 2:
                relations += wordnet_extractor.extract_keywords_with_levels(word, MAX_LEVEL)
        relations = wordnet_extractor.remove_duplicates(relations)
        label_target_graph[label][target] = relations
        full_knowledge_graph += relations

# Extract NER and insert to KG
ner_extractor = NERExtractor()
for row in tqdm(train_data, desc="Processing NER"):
    row["ner"] = ner_extractor.extract_entities(row["Text"])
    words = [item["word"] for item in row["ner"]]
    relations = []
    for word in words:
        relations += wordnet_extractor.extract_keywords_with_levels(word, MAX_LEVEL)
    relations = wordnet_extractor.remove_duplicates(relations)
    full_knowledge_graph += relations

# remove duplicate from KG
full_knowledge_graph = wordnet_extractor.remove_duplicates(full_knowledge_graph)

# extract unique words in KG
unique_words = list(set([item[0] for item in full_knowledge_graph] + [item[1] for item in full_knowledge_graph]))

# extract KG words definition
words_definition = {word: wordnet_extractor.extract_word_definition(word) for word in unique_words}

# calculate bert embedding & one hot embedding & ner list
bert = BERT()
targets_list = list(set([row["Target"] for row in train_data]))
target_embedder = OneHot(targets_list)


for row in tqdm(train_data, desc="Processing embeddings [Train]"):
    row["embedding"] = bert.get_twitter_bert_embedding(row["Text"])
    row["target_embedding"] = target_embedder.encode(row["Target"])

multi_source_random_walk = MultiSourceRandomWalk(full_knowledge_graph)

for row in tqdm(test_data, desc="Processing embeddings [Test]"):
    row["target_embedding"] = target_embedder.encode(row["Target"])
    row["ner"] = ner_extractor.extract_entities(row["Text"])

    row["exists_words"] = [word for word in unique_words if word in row["Text"]]
    row["exists_words_definitions"] = [words_definition[word] for word in row["exists_words"]]
    row["enrich_words"] = []
    row["enrich_words_definitions"] = []

    IMPORTANT_WORDS_COUNT = 3
    if row["exists_words"]:
        walk_length, num_walks = 5, 100
        important_nodes_data = multi_source_random_walk.get_important_nodes(row["exists_words"], walk_length, num_walks)
        i = 0
        while row["enrich_words"].__len__() < IMPORTANT_WORDS_COUNT and i < important_nodes_data.__len__():
            if important_nodes_data[i][0] not in row["exists_words"]:
                row["enrich_words"].append(important_nodes_data[i][0])
                row["enrich_words_definitions"].append(words_definition[important_nodes_data[i][0]])
            i += 1

    exist_words_embedding = []
    for i in range(row["exists_words"].__len__()):
        for definition in row["exists_words_definitions"][i]:
            txt = row["exists_words"][i] + ": " + definition
            exist_words_embedding.append(bert.get_twitter_bert_embedding(txt))

    enrich_words_embedding = []
    for i in range(row["enrich_words"].__len__()):
        for definition in row["enrich_words_definitions"][i]:
            txt = row["enrich_words"][i] + ": " + definition
            enrich_words_embedding.append(bert.get_twitter_bert_embedding(txt))

    row["text_embedding"] = bert.get_twitter_bert_embedding(row["Text"])
    row["exists_words_definition_embedding"] = average_vectors(exist_words_embedding)
    row["enrich_words_definition_embedding"] = average_vectors(enrich_words_embedding)

    row["final_embedding"] = average_vectors([row["text_embedding"], row["exists_words_definition_embedding"], row["enrich_words_definition_embedding"]])


# save data in pickle
with open('train_data.pkl', 'wb') as file:
    pickle.dump(test_data, file)
with open('test_data.pkl', 'wb') as file:
    pickle.dump(test_data, file)







