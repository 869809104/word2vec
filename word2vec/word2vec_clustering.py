import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
import re
import nltk
from nltk.corpus import stopwords
from sklearn.ensemble import RandomForestClassifier
from gensim.models import word2vec
from sklearn.cluster import KMeans
import time


# data cleaning function: splitting a paragraph into words
def review_to_wordlist(review, remove_stopwords=False):
    # data cleaning
    # 1).remove HTML tags or mark up
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
    # 4).remove Stopwords(Optionally)
    if remove_stopwords:
        stopwords_set = set(stopwords.words("english"))  # searching sets in Python is much faster than searching lists
        # print(stopwords.words("english"))
        words = [w for w in words if not w in stopwords_set]
    # 5).Porter Stemming and Lemmatizing
    # 6).return a list of words
    return words


tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')


# data cleaning function: splitting a paragraph into sentences
def review_to_sentences(review, tokenizer, remove_stopwords=False):
    # 1).Use the NLTK tokenizer to split the paragraph into sentences
    sentences = tokenizer.tokenize(review.strip())
    clean_sentences = []
    for sentence in sentences:
        if len(sentence) > 0:
            clean_sentences.append(review_to_wordlist(sentence, remove_stopwords))
    return clean_sentences  # return a list of lists


# feature vector function
def create_bag_of_centroids(reviews, word_centroid_map):
    counter = 0
    centroids = np.zeros((len(reviews), clusters_num), dtype='float32')
    for review in reviews:
        if counter % 1000 == 0:
            print("Review %d of %d" % (counter, len(reviews)))
        centroids_num = max(word_centroid_map.values()) + 1
        bag_of_centroids = np.zeros(centroids_num, dtype='float32')
        for word in review:
            if word in word_centroid_map:
                bag_of_centroids[word_centroid_map[word]] += 1
        centroids[counter] = bag_of_centroids
        counter += 1
    return centroids


# 1.Data reading
labeled_train = pd.read_csv(r"./data/labeledTrainData.tsv", header=0, delimiter="\t", quoting=3)  # 保留双引号
test = pd.read_csv(r"./data/testData.tsv", header=0, delimiter="\t", quoting=3)
unlabeled_train = pd.read_csv(r"./data/unlabeledTrainData.tsv", header=0, delimiter="\t", quoting=3)
print("Read %d labeled train reviews, %d labeled test reviews, and %d unlabeled train reviews"
      % (labeled_train["review"].size, test["review"].size, unlabeled_train["review"].size))

# 2.Data cleaning and text preprocessing
clean_train_sentences = []
print("Parsing sentences from training set...")
for review in labeled_train["review"]:
    clean_train_sentences += review_to_sentences(review, tokenizer)
print("Parsing sentences from unlabeled training set...")
for review in unlabeled_train["review"]:
    clean_train_sentences += review_to_sentences(review, tokenizer)
print(len(clean_train_sentences))

# 3.Creating and training word2vec model
features_num = 300  # word vector dimensionality
min_word_count = 40  # minimum word count
workers_num = 4  # worker threads
context = 10  # context/window size
downsampling = 1e-3  # downsampling of frequent words
model_name = "300features_40minwords_10context"
# architecture = skip-gram / cbow
# training algorithm = hierarchical softmax / negative sampling
print("Training word2vec model...")
model = word2vec.Word2Vec(clean_train_sentences, workers=workers_num, size=features_num,
                          min_count=min_word_count, window=context, sample=downsampling)
model.init_sims(replace=True)

# model.save(model_name)
# model = Word2Vec.load(model_name)
type(model.wv.syn0)  # numpy array: a feature vector for each word in the model's vocabulary
type(model.wv.index2word)  # list: the names of the words in the model's vocabulary
print(model.wv.syn0.shape[0] == len(model.wv.index2word))
# model.syn0.shape  # rows: the number of words in the model's vocabulary, columns: the size of feature vector
# model['flower']  # access feature vectors, return a numpy array of 1*300
# model.doesnt_match("france england germany berlin".split())
# model.most_similar("awful")

# 4.Clustering
start = time.time()
clusters_num = model.wv.syn0.shape[0] // 5  # average of 5 words of per cluster
kmeans_clustering = KMeans(n_clusters=clusters_num)
idx = kmeans_clustering.fit_predict(model.wv.syn0)
end = time.time()
elapsed = end - start
print("Time taken for K-Means clustering: ", elapsed, "seconds")
# idx: the assignment of cluster number for each word
# index2word: the names of the words in the model's vocabulary
word_centroid_map = dict(zip(model.wv.index2word, idx))

# print details of the top 10 cluster
for cluster in range(0, 10):
    print("Cluster %d" % cluster)
    words = []
    for i in range(0, len(word_centroid_map.values())):
        if list(word_centroid_map.values())[i] == cluster:
            words.append(list(word_centroid_map.keys())[i])
    print(words)

print("Creating centroid for labeled training data...")
clean_train_reviews = []
for review in labeled_train["review"]:
    clean_train_reviews.append(review_to_wordlist(review, remove_stopwords=True))
train_data_features = create_bag_of_centroids(clean_train_reviews, word_centroid_map)
print("Creating centroid for testing data...")
clean_test_reviews = []
for review in test["review"]:
    clean_test_reviews.append(review_to_wordlist(review, remove_stopwords=True))
test_data_features = create_bag_of_centroids(clean_test_reviews, word_centroid_map)

# 5.Supervised learning: Random Forest
print("Fitting a random forest using 100 trees to labeled training data...")
forest = RandomForestClassifier(n_estimators=100)
forest = forest.fit(train_data_features, labeled_train["sentiment"])

# 6.Run the trained Random Forest on testing set
print("Running the trained Random Forest on testing set...")
result = forest.predict(test_data_features)
output = pd.DataFrame(data={"id": test["id"], "sentiment": result})
output.to_csv(r"./data/Word2Vec_Clustering.csv", index=False, quoting=3)
