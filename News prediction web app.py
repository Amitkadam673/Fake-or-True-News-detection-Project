# -*- coding: utf-8 -*-
"""
Created on Tue Dec 17 14:33:44 2024

@author: Moxa
"""

import streamlit as st
import numpy as np
import pandas as pd
from scipy.sparse import hstack, csr_matrix
from joblib import load
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import warnings
from collections import Counter

# Initialize VADER analyzer outside the function for efficiency
vader_analyzer = SentimentIntensityAnalyzer()

# Function to preprocess text
def preprocess_text(text):
    return text.lower().strip()  # Convert text to lowercase and remove extra spaces

# Function to get TextBlob sentiment score
def get_textblob_sentiment(text):
    return TextBlob(text).sentiment.polarity

# Function to get VADER sentiment score
def get_vader_sentiment(text):
    return vader_analyzer.polarity_scores(text)['compound']

# Function to ensure feature compatibility
def adjust_features(input_vector, sentiment_features, loaded_model):
    expected_features = loaded_model.n_features_in_
    combined_features = hstack([input_vector, csr_matrix(sentiment_features)])
    current_features = combined_features.shape[1]

    if current_features < expected_features:
        warnings.warn(f"Padding features: Model expects {expected_features}, but got {current_features}.")
        padding = csr_matrix((combined_features.shape[0], expected_features - current_features))
        combined_features = hstack([combined_features, padding])
    elif current_features > expected_features:
        warnings.warn(f"Trimming features: Model expects {expected_features}, but got {current_features}.")
        combined_features = combined_features[:, :expected_features]

    return combined_features

# Function to predict whether the news is real or fake
def predict_news(user_input, loaded_model, tfidf_vectorizer):
    if user_input:
        try:
            # Preprocess input text
            processed_input = preprocess_text(user_input)

            # Transform text using TF-IDF Vectorizer
            input_vector_tfidf = tfidf_vectorizer.transform([processed_input])

            # Compute sentiment scores
            input_textblob_sentiment = get_textblob_sentiment(processed_input)
            input_vader_sentiment = get_vader_sentiment(processed_input)
            sentiment_features = np.array([[input_textblob_sentiment, input_vader_sentiment]])

            # Adjust features to match model input
            input_features = adjust_features(input_vector_tfidf, sentiment_features, loaded_model)

            # Make prediction using the trained model
            prediction = loaded_model.predict(input_features)[0]

            # Return the prediction result
            return "Real News" if prediction == 1 else "Fake News"
        except Exception as e:
            return f"An error occurred during prediction: {e}"
    else:
        return "Please enter text for prediction."

# Function to check class imbalance
def check_class_imbalance(data):
    try:
        label_counts = Counter(data['label'])  # Replace 'label' with the actual column name for labels
        st.write("Class Distribution in Training Data:")
        st.write(label_counts)
        return label_counts
    except Exception as e:
        st.error(f"An error occurred: {e}")

# Function to test with known examples
def test_with_known_examples(data, loaded_model, tfidf_vectorizer):
    try:
        st.write("Testing with Known Examples...")
        for index, row in data.sample(5).iterrows():  # Randomly test 5 examples
            text = row['text']  # Replace 'text' with your text column name
            label = "Real News" if row['label'] == 1 else "Fake News"  # Adjust label mapping if needed
            prediction = predict_news(text, loaded_model, tfidf_vectorizer)
            st.write(f"Text: {text}")
            st.write(f"Actual: {label}, Predicted: {prediction}")
            st.write("-" * 50)
    except Exception as e:
        st.error(f"An error occurred: {e}")

# Streamlit app
def main():
    st.title('News Prediction Web App')

    # File Upload for Model and Vectorizer
    uploaded_model = st.file_uploader("Upload Model File", type=["joblib"])
    uploaded_vectorizer = st.file_uploader("Upload Vectorizer File", type=["joblib"])

    if uploaded_model and uploaded_vectorizer:
        # Load the model and vectorizer
        loaded_model = load(uploaded_model)
        tfidf_vectorizer = load(uploaded_vectorizer)

        # Load training data if needed
        if st.checkbox("Check Class Imbalance"):
            training_data = pd.read_csv("path_to_training_data.csv")  # Replace with actual path to training data
            check_class_imbalance(training_data)

        if st.checkbox("Test with Known Examples"):
            training_data = pd.read_csv("path_to_training_data.csv")  # Replace with actual path to training data
            test_with_known_examples(training_data, loaded_model, tfidf_vectorizer)

        # User Input for Prediction
        user_input = st.text_area("Enter News Article Text:", "")
        if st.button("Predict"):
            result = predict_news(user_input, loaded_model, tfidf_vectorizer)
            st.write(f"Prediction: {result}")

            # Optional Debugging
            if st.checkbox("Enable Debugging"):
                processed_input = preprocess_text(user_input)
                input_vector_tfidf = tfidf_vectorizer.transform([processed_input])
                st.write("**Input Text:**", user_input)
                st.write("**Preprocessed Text:**", processed_input)
                st.write("**TF-IDF Features Shape:**", input_vector_tfidf.shape)
    else:
        st.warning("Please upload the model and vectorizer files to proceed.")

if __name__ == "__main__":
    main()
