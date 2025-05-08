# Sentiment Analysis with LSTM, BERT, and BART

## Overview
This project focuses on performing sentiment analysis on textual data using both traditional and transformer-based models. We experiment with different architectures, including an LSTM-based model, a DistilBERT-based model, and a BART-based model, to analyze text and classify the sentiment into predefined categories.

### Key Features:
- **LSTM-based Model**: An LSTM network for sentiment classification based on an embedding layer and bidirectional LSTM.
- **Transformer-based Models**: Fine-tuning a transformer model using **DistilBERT** and **BART** for sentiment analysis.
- **Preprocessing**: Text preprocessing techniques including tokenization, cleaning, and padding for deep learning models.
- **Model Evaluation**: Evaluation of the model performance on a test dataset and performance visualization.

## Table of Contents
- [Objective](#objective)
- [Problem Statement](#problem-statement)
- [Methodology](#methodology)
- [Requirements](#requirements)
- [Setup and Installation](#setup-and-installation)
- [Usage](#usage)
- [Model Architectures](#model-architectures)
- [Results](#results)
- [References](#references)

## Objective
The objective of this project is to develop and compare three models for sentiment analysis:
1. **LSTM-based model**: Traditional deep learning model using embedding layers and bidirectional LSTM for sentiment classification.
2. **DistilBERT-based model**: Fine-tuning a pre-trained **DistilBERT** model for sentiment analysis.
3. **BART-based model**: Fine-tuning a pre-trained **DistilBART** model for sentiment analysis.

## Problem Statement
Given a dataset of text (e.g., movie reviews or social media posts), the goal is to classify the sentiment of the text as:
- Positive
- Neutral
- Negative

The challenge is to accurately capture the sentiment of text using deep learning models, including the potential of transformer models (such as BERT and its smaller variant DistilBERT) and traditional models like LSTM.

## Methodology
The project proceeds as follows:

### Data Preprocessing
1. **Data Cleaning**: Text is cleaned by removing stopwords, punctuation, special characters, and unnecessary whitespaces.
2. **Tokenization**: Tokenize the cleaned text into word indices using tokenizers (such as the Tokenizer for LSTM or AutoTokenizer for BERT/DistilBERT).
3. **Padding**: All sequences are padded or truncated to a fixed length to ensure uniformity in the input size to the models.
4. **Splitting**: The dataset is split into training, validation, and test sets.

### Model Architectures

#### 1. **LSTM-based Model**
- The model is built using an embedding layer for transforming text into dense vectors, followed by a bidirectional LSTM layer for capturing sequential dependencies.
- The output layer uses a sigmoid activation function for binary classification.

#### 2. **DistilBERT Model**
- **DistilBERT** is a smaller version of the BERT model designed to be faster while retaining most of its performance. It is fine-tuned on the sentiment analysis task, using a classification head on top of the BERT encoder to predict sentiment labels.

    - Model used: [DistilBERT Base Uncased](https://huggingface.co/distilbert/distilbert-base-uncased)

#### 3. **DistilBART Model**
- **DistilBART** is a distilled version of BART, which is a sequence-to-sequence model trained to perform tasks like text generation, translation, and classification. We fine-tune **DistilBART** on sentiment analysis to classify the sentiment of text.
    
    - Model used: [DistilBART CNN 12-6](https://huggingface.co/sshleifer/distilbart-cnn-12-6)

### Evaluation
Models are evaluated using accuracy, loss, and other performance metrics, with results being recorded for each epoch.

## Requirements
- Python 3.x
- TensorFlow or PyTorch (depending on model type)
- Huggingface Transformers library for using pretrained models
- Pandas, NumPy for data handling
- Scikit-learn for evaluation metrics

### Installation
To set up the project, clone the repository and install the dependencies:

```bash
git clone https://github.com/yourusername/sentiment-analysis.git
cd sentiment-analysis
pip install -r requirements.txt
