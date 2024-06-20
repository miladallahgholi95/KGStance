import re
import string
import spacy
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

nltk.download('stopwords')
nltk.download('punkt')

class TextPreprocessor:
    def __init__(self):
        self.nlp = spacy.load('en_core_web_sm')
        self.stop_words = set(stopwords.words('english'))
        self.stemmer = PorterStemmer()

    def to_lowercase(self, input_text):
        return input_text.lower()

    def remove_punctuation(self, input_text):
        return input_text.translate(str.maketrans('', '', string.punctuation))

    def remove_whitespace(self, input_text):
        return ' '.join(input_text.split())

    def remove_stopwords(self, input_text):
        words = nltk.word_tokenize(input_text)
        return ' '.join(word for word in words if word not in self.stop_words)

    def stem_text(self, input_text):
        words = nltk.word_tokenize(input_text)
        return ' '.join(self.stemmer.stem(word) for word in words)

    def lemmatize_text(self, input_text):
        doc = self.nlp(input_text)
        return ' '.join(token.lemma_ for token in doc)

    def remove_special_characters(self, input_text):
        return re.sub(r'[^A-Za-z\s]', '', input_text)

    def tokenize_text(self, input_text):
        return nltk.word_tokenize(input_text)

    def preprocess(self, input_text, with_tokenize=False):
        input_text = self.to_lowercase(input_text)
        input_text = self.remove_punctuation(input_text)
        input_text = self.remove_special_characters(input_text)
        input_text = self.remove_whitespace(input_text)
        input_text = self.remove_stopwords(input_text)
        if with_tokenize:
            input_text = self.tokenize_text(input_text)
        return input_text