# Duplicate Question Pair Detector

This project focuses on detecting duplicate question pairs using traditional NLP techniques and word embeddings like Word2Vec. The application preprocesses the questions using stemming and lemmatization, computes the similarity score using cosine similarity on word embeddings, and provides an interface for users to input questions and determine their similarity.

## Overview

The project aims to assist in identifying similar questions in large datasets, which can be particularly useful in forums or Q&A platforms to reduce redundancy and improve user experience.

##UI
![Duplicate Question Pair Detector](https://github.com/umabisht1324/CODTECH-Task2/blob/main/UI.png)

## Key Features

- **NLP Preprocessing**: Utilizes stemming, lemmatization, and stop-word removal for effective text processing.
- **Word Embeddings**: Employs Word2Vec embeddings to capture semantic similarity between questions.
- **Cosine Similarity**: Calculates similarity scores using cosine similarity between vector representations of questions.
- **Streamlit Interface**: Offers a simple and interactive web interface for users to test question pairs for similarity.

## Technologies Used

- **Natural Language Toolkit (NLTK)**: For tokenization, stemming, lemmatization, and stop-word removal.
- **Gensim**: For loading and working with pre-trained Word2Vec models.
- **Streamlit**: For building and deploying the web application.
- **Python**: The core programming language used in the project.

## Project Goals

- **Improve Redundancy Detection**: Aid in identifying duplicate questions to streamline content on Q&A platforms.
- **Enhance User Interaction**: Provide a user-friendly interface for checking question similarity.
- **Efficient Processing**: Implement techniques for fast and accurate similarity computation.

## Running the Application

1. Clone the repository.
2. Install the required dependencies using `pip install -r requirements.txt`.
3. Run the Streamlit app with `streamlit run app.py`.
4. Enter two questions to check for similarity.

## Future Work

- **Expand to More Models**: Integrate other word embedding models like GloVe or FastText.
- **Add Batch Processing**: Allow users to input and process multiple question pairs simultaneously.
- **Improve Accuracy**: Experiment with more advanced similarity metrics and preprocessing techniques.

## Conclusion

The Duplicate Question Pair Detector leverages traditional NLP techniques and word embeddings to provide a reliable solution for identifying duplicate questions, ultimately improving content management and user satisfaction on platforms dealing with large volumes of user-generated content.
