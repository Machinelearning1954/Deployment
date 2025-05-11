# Machine Learning Capstone Project: Sentiment Analysis of Movie Reviews

## 1. Problem Statement
The project aims to develop a machine learning model capable of performing sentiment analysis on movie reviews. The model will classify a given movie review text as either positive or negative. This addresses the common task of understanding public opinion from textual data.

## 2. Chosen Approach and ML Methods

### 2.1. Dataset
We will use the **IMDB Movie Review Dataset**. This dataset consists of 50,000 movie reviews, evenly split into 25,000 for training and 25,000 for testing. Each set has an equal number of positive and negative reviews.

### 2.2. Machine Learning Model
We will utilize a pre-trained **Transformer-based model**, specifically **DistilBERT (distilbert-base-uncased)**.
*   **Rationale**: Transformer models, like BERT and its distilled versions, have shown state-of-the-art performance on various NLP tasks, including sentiment analysis. DistilBERT offers a good balance between performance and computational efficiency, making it suitable for deployment.

### 2.3. Data Preprocessing and Feature Engineering
*   **Tokenization**: Input text will be tokenized using the DistilBERT tokenizer. This includes converting text to lowercase, splitting into tokens, adding special tokens (`[CLS]`, `[SEP]`), and converting tokens to IDs.
*   **Padding and Truncation**: Sequences will be padded or truncated to a fixed length to ensure uniform input size for the model.
*   **Attention Masks**: Attention masks will be generated to differentiate real tokens from padding tokens.

### 2.4. Model Training
*   The pre-trained DistilBERT model will be fine-tuned on the IMDB dataset for the sentiment classification task.
*   A classification head (a dense layer) will be added on top of the DistilBERT base model.
*   Training will involve optimizing a suitable loss function (e.g., Cross-Entropy Loss) using an optimizer like AdamW.

### 2.5. Evaluation Metrics
The model's performance will be evaluated using:
*   Accuracy
*   Precision
*   Recall
*   F1-score
*   Confusion Matrix

## 3. Interactive Interface
A web application will be developed using **Flask**. This application will provide a simple interface where users can:
*   Input a movie review text.
*   Submit the text to the deployed model.
*   Receive the predicted sentiment (positive/negative) as output.

## 4. GitHub Repository
All project components will be organized and hosted in a public GitHub repository. This will include:
*   Jupyter notebooks or Python scripts for data loading, preprocessing, model training, and evaluation.
*   The Flask application code for the interactive interface.
*   A `requirements.txt` file listing all dependencies.
*   A comprehensive `README.md` file detailing the project, setup instructions, and usage.

## 5. Deployment
The Flask web application will be deployed to a platform that allows public access. For this capstone, we will demonstrate this by exposing the locally running Flask app.

This design covers all aspects of the machine learning lifecycle, from problem definition to a deployed, interactive solution, as required by the project criteria.
