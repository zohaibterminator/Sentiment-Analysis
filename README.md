# Sentiment Analysis with LSTM, BERT, and BART
This project focuses on performing sentiment analysis on textual data using both traditional and transformer-based models. We experiment with different architectures, including an LSTM model, DistilBERT model, and the DistilBART-CNN-12-6 model, to analyze text and classify the sentiment into predefined categories.

### Key Features:
- **LSTM Model**: An LSTM network for sentiment classification based on an embedding layer and bidirectional LSTM.
- **Transformer-based Models**: Fine-tuning a transformer model using **DistilBERT** and **DistilBART-CNN-12-6** for sentiment analysis.
- **Preprocessing**: Text preprocessing techniques including tokenization, cleaning, and padding for deep learning models.
- **Model Evaluation**: Evaluation of the model performance on a test dataset and performance visualization.

## Objective
The objective of this project is to develop and compare three models for sentiment analysis:
1. **Custom LSTM model**: Traditional deep learning model using embedding layers and bidirectional LSTM for sentiment classification.
2. **DistilBERT model**: Fine-tuning a pre-trained **DistilBERT** model for sentiment analysis.
3. **DistilBART-CNN-12-6**: Fine-tuning a pre-trained **DistilBART-CNN-12-6** model for sentiment analysis.

## Problem Statement
Given a dataset of text (social media posts), the goal is to classify the sentiment of the text as:
- Positive
- Negative

The challenge is to accurately capture the sentiment of text using deep learning models, including the potential of transformer models (such as DistilBERT) and traditional models like LSTM.

## Methodology
The project proceeds as follows:

### Data Preprocessing
1. **Data Cleaning**: Text is cleaned by removing stopwords, punctuation, special characters, and unnecessary whitespaces.
2. **Tokenization**: Tokenize the cleaned text into word indices using tokenizers (such as the Tokenizer for LSTM or AutoTokenizer for BERT/DistilBERT).
3. **Padding**: All sequences are padded or truncated to a fixed length to ensure uniformity in the input size to the models.
4. **Splitting**: The dataset is split into training, validation, and test sets.

### Model Architectures

#### 1. **LSTM Model**
- The model is built using an embedding layer for transforming text into dense vectors, followed by a bidirectional LSTM layer for capturing sequential dependencies.
- The output layer uses a sigmoid activation function for binary classification.

#### 2. **DistilBERT Model**
- **DistilBERT** is a smaller version of the BERT model designed to be faster while retaining most of its performance. It is fine-tuned on the sentiment analysis task, using a classification head on top of the BERT encoder to predict sentiment labels.

    - Model link: [DistilBERT Base Uncased](https://huggingface.co/distilbert/distilbert-base-uncased)

#### 3. **DistilBART-CNN-12-6 Model**
- **DistilBART-CNN-12-6** is a distilled version of BART, which is a sequence-to-sequence model trained to perform tasks like text generation, translation, and classification. We fine-tune **DistilBART-CNN-12-6** on sentiment analysis to classify the sentiment of text.

    - Model link: [DistilBART CNN 12-6](https://huggingface.co/sshleifer/distilbart-cnn-12-6)

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
