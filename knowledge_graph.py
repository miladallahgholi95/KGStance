from nltk.corpus import wordnet as wn
import nltk
nltk.download('wordnet')


class WordNet:
    def __init__(self):
        pass

    def remove_duplicates(self, keyword_list):
        seen = set()
        unique_keywords = []
        for item in keyword_list:
            keyword_tuple = (item[0], item[1], item[2])
            reverse_keyword_tuple = (item[1], item[0], item[2])
            if keyword_tuple not in seen and reverse_keyword_tuple not in seen:
                unique_keywords.append(item)
                seen.add(keyword_tuple)
                seen.add(reverse_keyword_tuple)
        return unique_keywords

    def extract_word_keyword_list(self, word):
        keyword_list = []
        synsets = wn.synsets(word)

        for synset in synsets:
            hypernyms = synset.hypernyms()
            if hypernyms:
                for hypernym in hypernyms:
                    similar_word = hypernym.name().split(".")[0].replace("_", " ")
                    if len(similar_word) > 2 and similar_word != word and hypernym.pos() in ['n', 'a', 's']:
                        keyword_list.append([word, similar_word, "hypernym"])

            hyponyms = synset.hyponyms()
            if hyponyms:
                for hyponym in hyponyms:
                    similar_word = hyponym.name().split(".")[0].replace("_", " ")
                    if len(similar_word) > 2 and similar_word != word and hyponym.pos() in ['n', 'a', 's']:
                        keyword_list.append([word, similar_word, "hyponym"])

            coordinates = synset.similar_tos()
            if coordinates:
                for coordinate in coordinates:
                    similar_word = coordinate.name().split(".")[0].replace("_", " ")
                    if len(similar_word) > 2 and similar_word != word and coordinate.pos() in ['n', 'a', 's']:
                        keyword_list.append([word, similar_word, "coordinate"])

            holonyms = synset.part_holonyms()
            if holonyms:
                for holonym in holonyms:
                    similar_word = holonym.name().split(".")[0].replace("_", " ")
                    if len(similar_word) > 2 and similar_word != word and holonym.pos() in ['n', 'a', 's']:
                        keyword_list.append([word, similar_word, "holonym"])

            meronyms = synset.part_meronyms()
            if meronyms:
                for meronym in meronyms:
                    similar_word = meronym.name().split(".")[0].replace("_", " ")
                    if len(similar_word) > 2 and similar_word != word and meronym.pos() in ['n', 'a', 's']:
                        keyword_list.append([word, similar_word, "meronym"])

            antonyms = []
            for lemma in synset.lemmas():
                if lemma.antonyms():
                    antonyms.extend(lemma.antonyms())
            if antonyms:
                for antonym in antonyms:
                    similar_word = antonym.name().split(".")[0].replace("_", " ")
                    if len(similar_word) > 2 and similar_word != word:
                        keyword_list.append([word, similar_word, "antonym"])

            attributes = synset.attributes()
            if attributes:
                for attribute in attributes:
                    similar_word = attribute.name().split(".")[0].replace("_", " ")
                    if len(similar_word) > 2 and similar_word != word and attribute.pos() in ['n', 'a', 's']:
                        keyword_list.append([word, similar_word, "attribute"])

        return self.remove_duplicates(keyword_list)

    def extract_keywords_with_levels(self, word, level):
        if level < 1:
            return []

        keyword_list = self.extract_word_keyword_list(word)
        if level == 1:
            return keyword_list

        all_keywords = keyword_list.copy()
        for keyword in keyword_list:
            similar_word = keyword[1]
            all_keywords.extend(self.extract_keywords_with_levels(similar_word, level - 1))

        return self.remove_duplicates(all_keywords)