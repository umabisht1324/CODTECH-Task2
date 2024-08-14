import numpy as np
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from gensim.models import KeyedVectors
import joblib


stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))


word2vec_model = KeyedVectors.load_word2vec_format('glove.6B.100d.word2vec.txt')

def preprocess_text(text):
    if not isinstance(text, str):
        text = str(text)
    tokens = word_tokenize(text.lower())
    processed_tokens = [
        lemmatizer.lemmatize(stemmer.stem(word))
        for word in tokens if word not in stop_words and word not in string.punctuation
    ]
    return ' '.join(processed_tokens)

def get_average_word2vec(tokens, model, num_features):
    feature_vector = np.zeros((num_features,), dtype="float32")
    num_words = 0
    index2word_set = set(model.index_to_key)
    for word in tokens:
        if word in index2word_set:
            num_words += 1
            feature_vector = np.add(feature_vector, model[word])
    if num_words > 0:
        feature_vector = np.divide(feature_vector, num_words)
    return feature_vector


joblib.dump(word2vec_model, 'word2vec_model.pkl')

print("Model saved successfully.")
