# Use the official tokenization script created by the Google team
from sklearn.model_selection import GridSearchCV, StratifiedKFold, RandomizedSearchCV
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import warnings
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import preprocessing, decomposition, model_selection, metrics, pipeline
from sklearn.metrics import f1_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn import model_selection
from nltk.corpus import stopwords
import nltk
import string
import re
import tokenization
import tensorflow_hub as hub
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense, Input
import tensorflow as tf
import pandas as pd
import numpy as np
# !wget - -quiet https: // raw.githubusercontent.com/tensorflow/models/master/official/nlp/bert/tokenization.py

# NOTES
# Term-frequency-inverse document frequency (TF-IDF) is another way to judge the topic of an article by the words it contains. With TF-IDF, words are given weight – TF-IDF measures relevance, not frequency. That is, wordcounts are replaced with TF-IDF scores across the whole dataset.
# The more documents a word appears in, the less valuable that word is as a signal to differentiate any given document. That’s intended to leave only the frequent AND distinctive words as markers. Each word’s TF-IDF relevance is a normalized data format that also adds up to one.
# Those marker words are then fed to the neural net as features in order to determine the topic covered by the document that contains them.
# Word2vec is great for digging into documents and identifying content and subsets of content. Its vectors represent each word’s context, the ngrams of which it is a part. BoW is a good, simple method for classifying documents as a whole.


# matplotlib and seaborn for plotting
sns.set(style="darkgrid")

warnings.filterwarnings('ignore')

# Training data
train = pd.read_csv("../data/train.csv")
print('Training data shape: ', train.shape)
train.head()

# Testing data
test = pd.read_csv("../data/test.csv")
print('Testing data shape: ', test.shape)
test.head()

# Missing values in training set
train.isnull().sum()
# Missing values in test set
test.isnull().sum()
train['target'].value_counts()

# take copies of the data to leave the originals for BERT
train1 = train.copy()
test1 = test.copy()

# Applying a first round of text cleaning techniques


def clean_text(text):
    '''Make text lowercase, remove text in square brackets,remove links,remove punctuation
    and remove words containing numbers.'''
    text = text.lower()  # make text lower case
    text = re.sub('\[.*?\]', '', text)  # remove text in square brackets
    text = re.sub('https?://\S+|www\.\S+', '', text)  # remove URLs
    text = re.sub('<.*?>+', '', text)  # remove html tags
    text = re.sub('[%s]' % re.escape(string.punctuation),
                  '', text)  # remove punctuation
    text = re.sub('\n', '', text)  # remove words conatinaing numbers
    text = re.sub('\w*\d\w*', '', text)
    text = re.sub('[‘’“”…]', '', text)

    return text

# emoji removal


def remove_emoji(text):
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                               u"\U00002702-\U000027B0"
                               u"\U000024C2-\U0001F251"
                               "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)


# Applying the de=emojifying function to both test and training datasets
train1['text'] = train1['text'].apply(lambda x: remove_emoji(x))
test1['text'] = test1['text'].apply(lambda x: remove_emoji(x))

# text preprocessing function


def text_preprocessing(text):
    """
    Cleaning and parsing the text.

    """
    tokenizer_reg = nltk.tokenize.RegexpTokenizer(r'\w+')

    nopunc = clean_text(text)
    tokenized_text = tokenizer_reg.tokenize(nopunc)
    remove_stopwords = [
        w for w in tokenized_text if w not in stopwords.words('english')]
    combined_text = ' '.join(remove_stopwords)
    return combined_text


# Applying the cleaning function to both test and training datasets
train1['text'] = train1['text'].apply(lambda x: text_preprocessing(x))
test1['text'] = test1['text'].apply(lambda x: text_preprocessing(x))

# Let's take a look at the updated text
# print(train1['text'].head())


#count_vectorizer = CountVectorizer()
count_vectorizer = CountVectorizer(ngram_range=(1, 1), min_df=1)
train_vectors = count_vectorizer.fit_transform(train1['text'])
test_vectors = count_vectorizer.transform(test1["text"])
# print(train_vectors)

# Keeping only non-zero elements to preserve space
train_vectors.shape

# Here we use 1 - and 2-grams where each terms has to appear at least twice and we ignore terms appearing in over 50 % of text examples.
tfidf = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_df=0.5)
train_tfidf = tfidf.fit_transform(train1['text'])
test_tfidf = tfidf.transform(test1["text"])

print(train_tfidf.shape)

# Fitting a simple Logistic Regression on BoW
logreg_bow = LogisticRegression(C=1.0)
scores = model_selection.cross_val_score(
    logreg_bow, train_vectors, train["target"], cv=5, scoring="f1")
print(scores.mean())

# Fit Logistic Regression and Multinomial Naive Bayes models with BoW and TF-IDF, giving four models in total. This is not an extensive list of vectorization options and models and I won't tune any of the models. It's more of an example framework for linear models in NLP

# Fitting a simple Logistic Regression on TFIDF
logreg_tfidf = LogisticRegression(C=1.0)
scores = model_selection.cross_val_score(
    logreg_tfidf, train_tfidf, train["target"], cv=5, scoring="f1")
print(scores.mean())

# Fitting a simple Naive Bayes on BoW
NB_bow = MultinomialNB()
scores = model_selection.cross_val_score(
    NB_bow, train_vectors, train["target"], cv=5, scoring="f1")
print(scores.mean())

# Fitting a simple Naive Bayes on TFIDF
NB_tfidf = MultinomialNB()
scores = model_selection.cross_val_score(
    NB_tfidf, train_tfidf, train["target"], cv=5, scoring="f1")
print(scores.mean())


# CONCLUSION
# The best score is when we use MNB on the bag of words vectors. It gives a training score of 0.6585 and a leaderboard score of 0.7945.
# Bag of Words is significantly better than TF-IDF in both models and it's a little bit surprising that 1-grams with no minumum limit seems to give the best results. I think this might be partly due to the nature of the data. Tweets are usually pretty short and probably don't have much of a 'standard' layout or structure. This might be why a fairly simple BoW model does really well compared to TF-IDF or higher gram models.
