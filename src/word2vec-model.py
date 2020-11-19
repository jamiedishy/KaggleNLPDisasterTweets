import numpy as np
import csv as csv
import pandas as pd
import nltk
import re
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import classification_report
from gensim.models import Word2Vec
from sklearn.svm import SVC
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

input_file_training = pd.read_csv("../data/train.csv")
input_file_test = pd.read_csv("../data/test.csv")

print("Cleaning data")
temp_data = []
for i in range(0, len(input_file_training.text)):
    temp = re.sub(
        r"(http|ftp|https)://([\w_-]+(?:(?:\.[\w_-]+)+))([\w.,@?^=%&:/~+#-]*[\w@?^=%&/~+#-])?",
        "",
        input_file_training.text[i],
    )
    # remove any remaining non alphabet or non empty space character
    temp = re.sub(r"[^\x00-\x7F]+", "", temp)
    temp_data.append(temp)

clean_training_df = pd.DataFrame(temp_data, columns=["text"])
# clean_training_text = clean_training_df.text
input_file_training.text = clean_training_df

print("Removing special characters from commit dataframe")
spec_chars = [
    "!",
    '"',
    "#",
    "%",
    "&",
    "'",
    "(",
    ")",
    "*",
    "+",
    ",",
    "-",
    ".",
    "/",
    ":",
    ";",
    "<",
    "=",
    ">",
    "?",
    "@",
    "[",
    "\\",
    "]",
    "^",
    "_",
    "`",
    "{",
    "|",
    "}",
    "~",
    "–",
    "$",
    "ø",
    "å",
]
input_file_training.text = input_file_training.text.str.replace(
    "|".join(map(re.escape, spec_chars)), ""
)
# remove numbers
input_file_training.text = input_file_training.text.str.replace("\d+", "")

input_file_training.text = input_file_training.text.str.split().str.join(" ")

# Stemming, Remove stop words
print("Tokenizing comments from commit dataframe")
print("Lemmatizing, removing stop words and characters with a length of 1")
w_tokenizer = nltk.tokenize.WhitespaceTokenizer()
# stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()
stop = stopwords.words("english")


def stem_remove_stop_words(row):
    place_holder = ""
    for word in w_tokenizer.tokenize(row):
        # stem = stemmer.stem(word)
        lem = lemmatizer.lemmatize(word)
        if lem.lower() not in stop and len(lem) > 2:
            place_holder += lem + " "
    return place_holder


input_file_training.text = input_file_training.text.apply(
    lambda x: stem_remove_stop_words(x)
)
input_file_training.to_csv("cleaned_data.csv", index=False)

print("Collecting most frequent words from clean commit data with frequency > 3")
vectorizer = CountVectorizer(stop_words="english", min_df=4)
frequent_words = vectorizer.fit_transform(input_file_training.text)

print("Creating table for most frequent words given text column")


def bag_of_words():
    sentence_vectors = []
    for index, row in input_file_training.text.items():
        sentece_tokens = nltk.word_tokenize(row)
        sent_vec = []
        for token in vectorizer.get_feature_names():
            if token in sentece_tokens:
                sent_vec.append(1)
            else:
                sent_vec.append(0)
        sentence_vectors.append(sent_vec)
    return sentence_vectors


word_frequency_df = pd.DataFrame(bag_of_words(), columns=vectorizer.get_feature_names())
word_frequency_df = word_frequency_df.assign(id=input_file_training.id)


training_data_final_df = pd.merge(
    left=input_file_training,
    right=word_frequency_df,
    how="left",
    left_on="id",
    right_on="id",
)


train_target = training_data_final_df["target_x"]
training_data_final_df = training_data_final_df.drop("id", axis=1)
training_data_final_df = training_data_final_df.drop("keyword", axis=1)
training_data_final_df = training_data_final_df.drop("location_x", axis=1)
training_data_final_df = training_data_final_df.drop("text_x", axis=1)

training_data_final_df.to_csv("training_data_final.csv", index=False)

train_x, test_x, train_y, test_y = train_test_split(
    training_data_final_df, train_target, test_size=0.2, random_state=1
)

classifiers = [SVC(kernel="linear", C=0.025), SVC(gamma=2, C=1)]

names = ["Linear SVM", "RBF SVM"]

for i in range(len(classifiers)):
    clas = classifiers[i]
    test_pred = clas.fit(train_x, train_y).predict(test_x)
    print(names[i], "\n", classification_report(test_y, test_pred, labels=[0, 1]))
