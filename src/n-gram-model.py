import numpy as np
import csv as csv
import pandas as pd
import re as re
from collections import Counter
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import CountVectorizer

from gensim.models import Word2Vec

from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier

from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis


input_file_training = "../data/train.csv"
input_file_test = "../data/test.csv"

# Create a count vectorizer that uses common english words as stop words, and using min_df = 4
vectorizer = CountVectorizer(stop_words= "english", min_df = 4)
bigram = CountVectorizer(ngram_range = (1,2), token_pattern=r'\b\w+\b', min_df=2)
trigram = CountVectorizer(ngram_range = (1,3), token_pattern=r'\b\w+\b', min_df=2)

# load the training data as a matrix
dataset = pd.read_csv(input_file_training, header=0, encoding='ISO-8859-1')

# load the testing data
dataset2 = pd.read_csv(input_file_test, header=0, encoding = 'ISO-8859-1')

#cleaning data
temp_data = []
for i in range(0, len(dataset.text)):
    #remove all links
    temp = re.sub(r'(http|ftp|https)://([\w_-]+(?:(?:\.[\w_-]+)+))([\w.,@?^=%&:/~+#-]*[\w@?^=%&/~+#-])?', '', dataset.text[i])
    #remove any non ascii characters
    temp = re.sub(r'[^\x00-\x7F]+','', temp)
    temp_data.append(temp)

# fit the cleaned data into the vectorizer
X = vectorizer.fit_transform(temp_data)

# vectorizer vocabulary list
words = vectorizer.vocabulary_

# Create a dataframe with columns of words in the dictionary
dataset_extra = pd.DataFrame(X.todense(), columns = vectorizer.get_feature_names())

#dataframe with the features and id
dataset = dataset.merge(dataset_extra, left_index = True, right_index = True)

# remove unnecessary features
dataset = dataset.drop('id_x', axis=1)
dataset = dataset.drop('keyword', axis = 1)
dataset = dataset.drop('location_x', axis = 1)
dataset = dataset.drop('text_x', axis = 1)


#training target
target = dataset.target_x

#splitting the target into 80/20 --- Gonna try to use cross validation / K fold this weekend
train_x, test_x, train_y, test_y = train_test_split(dataset, target, test_size = 0.2, random_state = 1)

# Simple decision tree classifier with random parameters from another dataset --- Need further tuning?
decision_t = DecisionTreeClassifier(random_state=1, max_depth=20, max_leaf_nodes = 215,min_samples_split = 2)

# Fit the training data into the classifier and use it to predict 
test_pred = decision_t.fit(train_x, train_y).predict(test_x)

# Print classification report
print(classification_report(test_y, test_pred, labels=[0,1]))

# README: Bag of words and no location/keywords consideration. probably not the best approach.
print('----------------------MORE CLASSIFIERS-----------------------')
# List of classifiers that Minh used
classifiers = [
    KNeighborsClassifier(3),
    SVC(kernel="linear", C=0.025),
    SVC(gamma=2, C=1),
#     GaussianProcessClassifier(1.0 * RBF(1.0)),
    DecisionTreeClassifier(max_depth=5),
    RandomForestClassifier(max_depth=5, n_estimators=10),
    MLPClassifier(alpha=1, max_iter=1000),
    AdaBoostClassifier(),
    GaussianNB(),
    QuadraticDiscriminantAnalysis()]

names = ["Nearest Neighbors", "Linear SVM", "RBF SVM",
#          "Gaussian Process",
         "Decision Tree", "Random Forest", "Neural Net", "AdaBoost",
         "Naive Bayes", "QDA"]

for i in range(len(classifiers)):
    clas = classifiers[i]
    test_pred = clas.fit(train_x, train_y).predict(test_x)
    print(names[i])
    print(classification_report(test_y, test_pred, labels=[0,1]))

print('----------------------BIGRAM MODEL--------------------------')
print('----------------------TRIGRAM MODEL-------------------------')

