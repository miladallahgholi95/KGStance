from Preprocessing import TextPreprocessor
from collections import Counter
import math

def calculate_mean(numbers):
    if len(numbers) == 0:
        return 0
    return sum(numbers) / len(numbers)

def calculate_std(numbers):
    if len(numbers) == 0:
        return 0
    mean = calculate_mean(numbers)
    variance = sum((x - mean) ** 2 for x in numbers) / len(numbers)
    return math.sqrt(variance)

class AnoVa_V1:
    def __init__(self, input_texts):
        self.anova_score_dict = None
        self.target_list = None
        self.words_targets_docs_dict = None
        self.input_texts = input_texts

    def preprocessing(self):
        text_preprocessor = TextPreprocessor()
        for i in range(self.input_texts.__len__()):
            self.input_texts[i]["Words"] = text_preprocessor.preprocess(self.input_texts[i]["Text"], with_tokenize=True)

    def calculate_word_count(self):
        words_targets_docs_dict = {}
        target_docs_count = {}

        for i in range(self.input_texts.__len__()):
            words = self.input_texts[i]["Words"]
            target = self.input_texts[i]["Target"]
            label = self.input_texts[i]["Label"]

            if target not in target_docs_count:
                target_docs_count[target] = 0
            target_docs_count[target] += 1

            words_counter = Counter(words)

            for word, count in words_counter.items():
                if word not in words_targets_docs_dict:
                    words_targets_docs_dict[word] = {}

                if target not in words_targets_docs_dict[word]:
                    words_targets_docs_dict[word][target] = []

                words_targets_docs_dict[word][target].append(count)

        for word, values in words_targets_docs_dict.items():
            for target, target_count in target_docs_count.items():
                if target not in values:
                    values[target] = [0 for i in range(target_count)]
                else:
                    values[target] += [0 for i in range(target_count-values[target].__len__())]

        self.words_targets_docs_dict = words_targets_docs_dict
        self.target_list = list(target_docs_count.keys())

    def filter_words(self, filter_percent):
        final_words_targets_docs_dict = {}
        for word, values in self.words_targets_docs_dict.items():
            status = False
            for key, value in values.items():
                non_zero_count = sum(1 for x in value if x > 0)
                non_zero_percent = non_zero_count / value.__len__()
                if non_zero_percent >= filter_percent:
                    status = True
                    break
            if status:
                final_words_targets_docs_dict[word] = values

        self.words_targets_docs_dict = final_words_targets_docs_dict

    def calculate_words_anova_score(self):
        anova_score_dict = {}
        for target in self.target_list:

            if target not in anova_score_dict:
                anova_score_dict[target] = {}

            for word, values in self.words_targets_docs_dict.items():
                avg_main, std_main = calculate_mean(values[target]), calculate_std(values[target])

                others_list = []
                for key, value in values.items():
                    if key != target:
                        others_list += value

                avg_others, std_others = calculate_mean(others_list), calculate_std(others_list)

                try:
                    anova_score_dict[target][word] = calculate_std([avg_main, avg_others]) / calculate_mean([std_main, std_others])

                    if avg_others > avg_main:
                        anova_score_dict[target][word] *= -1

                except ZeroDivisionError:
                    continue

        self.anova_score_dict = anova_score_dict

    def get_anova_top_words(self, n_words=10):
        # preprocessing text
        self.preprocessing()

        # calculate words labels docs counts
        self.calculate_word_count()
        # print("All words count (before filtering):", self.words_labels_docs_dict.keys().__len__())

        # filter words
        self.filter_words(filter_percent=0.01)
        # print("All words count (after filtering):", self.words_labels_docs_dict.keys().__len__())

        # calculate words anova_score
        self.calculate_words_anova_score()

        label_target_anova_words = {}
        for target, words_score_dict in self.anova_score_dict.items():
            if target not in label_target_anova_words:
                label_target_anova_words[target] = {}

            words_score_sorted_dict = dict(sorted(words_score_dict.items(), key=lambda item: item[1], reverse=True))
            top_10_dict = dict(list(words_score_sorted_dict.items())[:10])
            label_target_anova_words[target] = top_10_dict

        print(label_target_anova_words)

class AnoVa_V2:
    def __init__(self, input_texts):
        self.anova_score_dict = None
        self.label_target_list = None
        self.words_labels_docs_dict = None
        self.input_texts = input_texts

    def preprocessing(self):
        text_preprocessor = TextPreprocessor()
        for i in range(self.input_texts.__len__()):
            self.input_texts[i]["Words"] = text_preprocessor.preprocess(self.input_texts[i]["Text"], with_tokenize=True)

    def calculate_word_count(self):
        words_labels_docs_dict = {}
        label_target_docs_count = {}

        for i in range(self.input_texts.__len__()):
            words = self.input_texts[i]["Words"]
            target = self.input_texts[i]["Target"]
            label = self.input_texts[i]["Label"]
            key = label + "__" + target

            if key not in label_target_docs_count:
                label_target_docs_count[key] = 0
            label_target_docs_count[key] += 1

            words_counter = Counter(words)

            for word, count in words_counter.items():
                if word not in words_labels_docs_dict:
                    words_labels_docs_dict[word] = {}

                if key not in words_labels_docs_dict[word]:
                    words_labels_docs_dict[word][key] = []

                words_labels_docs_dict[word][key].append(count)

        for word, values in words_labels_docs_dict.items():
            for label_target, label_target_count in label_target_docs_count.items():
                if label_target not in values:
                    values[label_target] = [0 for i in range(label_target_count)]
                else:
                    values[label_target] += [0 for i in range(label_target_count-values[label_target].__len__())]

        self.words_labels_docs_dict = words_labels_docs_dict
        self.label_target_list = list(label_target_docs_count.keys())

    def filter_words(self, filter_percent):
        final_words_labels_docs_dict = {}
        for word, values in self.words_labels_docs_dict.items():
            status = False
            for key, value in values.items():
                non_zero_count = sum(1 for x in value if x > 0)
                non_zero_percent = non_zero_count / value.__len__()
                if non_zero_percent >= filter_percent:
                    status = True
                    break
            if status:
                final_words_labels_docs_dict[word] = values

        self.words_labels_docs_dict = final_words_labels_docs_dict

    def calculate_words_anova_score(self):
        anova_score_dict = {}
        for label_target in self.label_target_list:

            if label_target not in anova_score_dict:
                anova_score_dict[label_target] = {}

            for word, values in self.words_labels_docs_dict.items():
                avg_main, std_main = calculate_mean(values[label_target]), calculate_std(values[label_target])

                others_list = []
                for key, value in values.items():
                    if key != label_target:
                        others_list += value

                avg_others, std_others = calculate_mean(others_list), calculate_std(others_list)

                try:
                    anova_score_dict[label_target][word] = calculate_std([avg_main, avg_others]) / calculate_mean([std_main, std_others])

                    if avg_others > avg_main:
                        anova_score_dict[label_target][word] *= -1

                except ZeroDivisionError:
                    continue

        self.anova_score_dict = anova_score_dict

    def get_anova_top_words(self, n_words=10):
        # preprocessing text
        self.preprocessing()

        # calculate words labels docs counts
        self.calculate_word_count()
        # print("All words count (before filtering):", self.words_labels_docs_dict.keys().__len__())

        # filter words
        self.filter_words(filter_percent=0.01)
        # print("All words count (after filtering):", self.words_labels_docs_dict.keys().__len__())

        # calculate words anova_score
        self.calculate_words_anova_score()

        label_target_anova_words = {}
        for label_target, words_score_dict in self.anova_score_dict.items():
            label, target = label_target.split("__")
            if label not in label_target_anova_words:
                label_target_anova_words[label] = {}

            if target not in label_target_anova_words[label]:
                label_target_anova_words[label][target] = {}

            words_score_sorted_dict = dict(sorted(words_score_dict.items(), key=lambda item: item[1], reverse=True))
            top_10_dict = dict(list(words_score_sorted_dict.items())[:10])
            label_target_anova_words[label][target] = top_10_dict

        print(label_target_anova_words)







