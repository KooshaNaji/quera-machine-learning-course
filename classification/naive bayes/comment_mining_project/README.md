# Comment Mining

## Project Overview
This project implements a comment mining system that classifies user comments based on sentiment related to product pricing. It utilizes natural language processing (NLP) techniques with the Hazm library for Persian text normalization and tokenization, along with a Naive Bayes classification approach.

## Features
- **Data Preprocessing**: Normalization and tokenization of Persian text using Hazm.
- **Stopword & Punctuation Removal**: Cleans the text data for better analysis.
- **Probability Computation**: Uses prior probabilities for sentiment classification.
- **Parallel Processing**: Implements multithreading with `ThreadPoolExecutor` to speed up text processing.
- **Naive Bayes Classifier**: Computes class probabilities and predicts sentiment labels.
- **Performance Evaluation**: Uses `accuracy_score` to measure classification performance.
- **Submission File Generation**: Outputs predictions to a CSV file and compresses results into a ZIP file.
