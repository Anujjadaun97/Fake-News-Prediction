# Fake News Prediction

This repository contains a Jupyter Notebook [Fake_news_prediction.ipynb](https://github.com/Anujjadaun97/Fake-News-Prediction/blob/main/Fake_news_prediction.ipynb) that demonstrates a fake news prediction model. The project involves data preprocessing, training, and evaluation of machine learning models for binary classification of news articles as real or fake.

## Table of Contents

  - Project Overview
  - Tasks Performed
  - Installation
  - Dataset
  - Data Preprocessing
  - Model Training and Evaluation
  - Dependencies

## Project Overview

This project aims to build a machine learning model to distinguish between "fake" and "true" news articles. It involves loading news data, performing necessary preprocessing steps such as handling duplicates and missing values, feature engineering, and finally, training a classification model (Logistic Regression, with potential for others) to predict the authenticity of news.

## Tasks Performed

The project notebook outlines the following key tasks:

1.  **News Data Loading**: Loading the fake and true news datasets.
2.  **Data Preprocessing**: Cleaning and preparing the data for model training.
3.  **Training and Test Split**: Dividing the dataset into training and testing sets.
4.  **Logistic Regression Model**: Implementing a Logistic Regression model for binary classification.
5.  **Evaluation and Prediction**: Evaluating the trained model and making news predictions.

## Installation

To run this notebook, you will need to have Python and Jupyter Notebook installed.
You can install the required libraries using pip:

```bash
pip install pandas numpy seaborn matplotlib scikit-learn tensorflow nltk scipy joblib wordcloud
```

Additionally, you will need to download NLTK data:

```python
import nltk
nltk.download('stopwords')
nltk.download('punkt_tab')
```

## Dataset

The project utilizes two CSV files:

  - [Fake.csv](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset?select=Fake.csv): Contains fake news articles.
  - [True.csv](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset?select=True.csv): Contains true news articles.

Both datasets contain `title`, `text`, `subject`, and `date` columns.

## Data Preprocessing

The preprocessing steps include:

  - Checking and dropping duplicate entries in both fake and true news datasets.
      - Fake News duplicates: 3
      - True News duplicates: 206
  - Verifying for missing values (it has been confirmed there are no missing values).
  - Assigning a 'class' label ('True' or 'Fake') to each dataset.
  - Concatenating the fake and true news dataframes into a single dataframe (`news_df`).
  - Applying `LabelEncoder` to convert the 'class' column into numeric labels (e.g., 0 for 'Fake', 1 for 'True').

## Model Training and Evaluation

The notebook primarily focuses on using a **Logistic Regression Model** for classification. The typical workflow involves:

  - Splitting the combined dataset into training and testing sets.
  - Training the Logistic Regression model on the training data.
  - Evaluating the model's performance using metrics such as `accuracy_score` and `classification_report`, `precision_score`, `recall_score`, `f1_score`, and `confusion_matrix`.

The notebook also imports other models like `MultinomialNB`, `SVC` for potential future use or comparison.

## Dependencies

The following Python libraries are used in this project:

  - `pandas`
  - `numpy`
  - `seaborn`
  - `matplotlib.pyplot`
  - `sklearn.preprocessing.LabelEncoder`
  - `sklearn.model_selection.train_test_split`
  - `sklearn.metrics` (accuracy\_score, classification\_report, precision\_score, recall\_score, f1\_score, confusion\_matrix)
  - `tensorflow`
  - `keras`
  - `unicodedata`
  - `nltk` (stopwords, word\_tokenize, PorterStemmer, WordNetLemmatizer)
  - `sklearn.feature_extraction.text.CountVectorizer`
  - `scipy.sparse`
  - `wordcloud.WordCloud`
  - `sklearn.naive_bayes.MultinomialNB`
  - `sklearn.linear_model.LogisticRegression`
  - `sklearn.svm.SVC`
  - `joblib`
  - `re`
  - `string`
