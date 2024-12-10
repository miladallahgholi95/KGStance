from Preprocessing import TextPreprocessor
from scipy.stats import f_oneway
from collections import Counter
import numpy as np

class AnoVaText:
    def __init__(self, texts, labels, num_important_words):
        # Ensure that the number of texts and labels match
        if len(texts) != len(labels):
            raise ValueError("The number of texts and labels must be the same.")

        # Initialize attributes
        self.texts = texts
        self.labels = labels
        self.num_important_words = num_important_words

        # Extract the unique words from the provided texts
        self.unique_words = self.extract_unique_words(texts)

        # Compute the word counts for each text
        self.word_counts = self.compute_word_counts(texts)

    def extract_unique_words(self, texts):
        """
        Extracts all unique words from the provided texts.
        - Splits each text into words.
        - Adds them to a list and removes duplicates to return only unique words.
        """
        words = []
        for text in texts:
            words.extend(text.split())  # Split text into words and add to list
        return list(set(words))  # Remove duplicates and return unique words

    def compute_word_counts(self, texts):
        """
        Computes the frequency of each word in each text.
        - For each text, counts the occurrences of every word and returns the counts.
        """
        word_counts = []
        for text in texts:
            count = Counter(text.split())  # Count occurrences of each word in the text
            word_counts.append(count)  # Append word count to the list
        return word_counts

    def analyze(self):
        """
        Perform ANOVA (F-test) on word frequencies to identify important words for each label.
        - For each label, compute the F-statistic comparing word frequencies between texts belonging to the label
          and texts belonging to other labels.
        - If the p-value is less than 0.05, the word is considered important and the F-statistic is used as the score.
        """
        label_set = set(self.labels)  # Identify unique labels (classes) in the dataset
        important_words = {}  # Dictionary to store the most important words for each label

        # Iterate over each unique label (class)
        for label in label_set:
            # Get indices of texts that belong to the current label
            label_indices = [i for i, lbl in enumerate(self.labels) if lbl == label]
            # Get indices of texts that belong to other labels
            other_indices = [i for i, lbl in enumerate(self.labels) if lbl != label]

            # Extract word counts for texts of the current label and other labels
            label_word_counts = [self.word_counts[i] for i in label_indices]
            other_word_counts = [self.word_counts[i] for i in other_indices]

            word_scores = {}  # Dictionary to store scores for each word
            for word in self.unique_words:
                # Get word frequencies for the current word in both label and other texts
                label_word_frequencies = [count.get(word, 0) for count in label_word_counts]
                other_word_frequencies = [count.get(word, 0) for count in other_word_counts]

                # Check if there is variance in both groups
                if np.var(label_word_frequencies) > 0 and np.var(other_word_frequencies) > 0:
                    # Perform ANOVA (F-test) to compare variances between the two groups (label vs others)
                    f_stat, p_value = f_oneway(label_word_frequencies, other_word_frequencies)

                    # If p-value is less than 0.05, there is a significant difference
                    if p_value < 0.05:
                        word_scores[word] = f_stat  # Assign the F-statistic as the word's score
                    else:
                        word_scores[word] = 0  # If no significant difference, assign score of 0
                else:
                    word_scores[word] = 0  # If no variance, assign score of 0

            # Sort the words by their score (higher F-statistic indicates more important words)
            sorted_words = sorted(word_scores.items(), key=lambda item: item[1], reverse=True)

            # Store the top `num_important_words` for the current label
            important_words[label] = sorted_words[:self.num_important_words]

        # Return the important words for each label
        return important_words


from dataset import SemEval2016
dataset = SemEval2016()
train_data, test_data = dataset.get_data("train"), dataset.get_data("test")


txt_prs = TextPreprocessor()

text_list, target_list = [], []
for row in train_data:
    text_list.append(txt_prs.preprocess(row["Text"], with_tokenize=False))
    target_list.append(row["Target"])

# ایجاد یک شیء از TextAnalyzerANOVA
analyzer = AnoVaText(text_list, target_list, 10)
important_words = analyzer.analyze()

# نمایش نتایج
for key, value in important_words.items():
    print(f"------- Target: {key} --------")
    for word, score in value:
        print(word, " = ", score)
