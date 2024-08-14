import streamlit as st
import numpy as np
import joblib
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
import string


word2vec_model = joblib.load('word2vec_model.pkl')


stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

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

def are_questions_similar(q1, q2):
    q1_processed = preprocess_text(q1)
    q2_processed = preprocess_text(q2)
    
    q1_vector = get_average_word2vec(q1_processed.split(), word2vec_model, 100)
    q2_vector = get_average_word2vec(q2_processed.split(), word2vec_model, 100)
    
    similarity = np.dot(q1_vector, q2_vector) / (np.linalg.norm(q1_vector) * np.linalg.norm(q2_vector))
    
    threshold = 0.7
    similar = similarity > threshold
    
    return similar, similarity

def main():
    st.title("Duplicate Question Pair Detector")
    st.write("Enter two questions to check if they are duplicates.")
    
    q1 = st.text_input("Question 1", "How can I learn machine learning?")
    q2 = st.text_input("Question 2", "What is the best way to study machine learning?")
    
    if st.button("Check Similarity"):
        similar, score = are_questions_similar(q1, q2)
        st.write(f"Questions are similar: {similar}")
        # st.write(f"Similarity score: {score:.2f}")

if __name__ == "__main__":
    main()



