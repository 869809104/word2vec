import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier


# data cleaning function: splitting a paragraph into words
def review_to_words(review):
    # 1).remove HTML tags or markup
    review_text = BeautifulSoup(review).get_text()
    # print(review_text)
    # 2).remove Punctuations, Numbers
    review_letters_only = re.sub("[^a-zA-Z]", " ", review_text)
    # print(review_letters_only)
    # 3).tokenization
    # convert to lowercase
    review_letters_lowercase = review_letters_only.lower()
    # split into individual words
    words = review_letters_lowercase.split()
    # 4).remove Stopwords
    # print(stopwords.words("english"))
    stopwords_set = set(stopwords.words("english"))  # searching sets in Python is much faster than searching lists
    words_meaningful = [w for w in words if not w in stopwords_set]
    # 5).Porter Stemming and Lemmatizing
    # 6).join the words and return a string separated by space
    return " ".join(words_meaningful)


# 1.Data reading
train = pd.read_csv(r"./data/labeledTrainData.tsv", header=0, delimiter="\t", quoting=3)  # 如实打印，保留双引号
test = pd.read_csv(r"./data/testData.tsv", header=0, delimiter="\t", quoting=3)
print("Read %d train reviews, and %d test reviews" % (train["review"].size, test["review"].size))
# print(train.shape)  # (25000, 3)
# print(train.columns.values)
# print(train["review"][0])
# print(type(train["review"]))  # Series
# print(review_to_words(train["review"][0]))

# 2.Data cleaning and text preprocessing
print("Cleaning and parsing the training set movie reviews...")
train_reviews_num = train["review"].size
clean_train_reviews = []
for i in range(0, train_reviews_num):
    if (i + 1) % 1000 == 0:
        print("Review %d of %d\n" % (i + 1, train_reviews_num))
    clean_train_reviews.append(review_to_words(train["review"][i]))
print("Cleaning and parsing the testing set movie reviews...")
test_reviews_num = test["review"].size
clean_test_reviews = []
for i in range(0, test_reviews_num):
    if (i+1) % 1000 == 0:
        print("Review %d of %d" % (i+1, test_reviews_num))
    clean_test_reviews.append(review_to_words(test["review"][i]))

# 3.Creating the features: count the number of times each vocabulary appears
print("Creating the bag of words model...")
vectorizer = CountVectorizer(analyzer="word",
                             tokenizer=None,
                             preprocessor=None,
                             stop_words=None,
                             max_features=5000)
train_data_features = vectorizer.fit_transform(clean_train_reviews)
train_data_features = train_data_features.toarray()
test_data_features = vectorizer.transform(clean_test_reviews)
test_data_features = test_data_features.toarray()
# print(train_data_features.shape)  # (25000, 5000)
vocabulary = vectorizer.get_feature_names()  # a list of the names of 5000 features
print(vocabulary)
# sum up the counts of each vocabulary word
dist = np.sum(train_data_features, axis=0)
for tag, count in zip(vocabulary, dist):
    print(tag, count)

# 4.Supervised learning: Random Forest
print("Fitting a Random Forest using 100 trees to training data...")
forest = RandomForestClassifier(n_estimators=100)
forest = forest.fit(train_data_features, train["sentiment"])

# 5.Run the trained Random Forest on testing set
print("Running the trained Random Forest on testing set...")
result = forest.predict(test_data_features)
output = pd.DataFrame(data={"id": test["id"], "sentiment": result})
output.to_csv(r"./data/Bag_of_Words_model.csv", index=False, quoting=3)
