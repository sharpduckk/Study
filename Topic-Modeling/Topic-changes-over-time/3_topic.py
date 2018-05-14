#-*- coding: utf-8 -*-
# Author: Olivier Grisel <olivier.grisel@ensta.org>
#         Lars Buitinck
#         Chyi-Kwei Yau <chyikwei.yau@gmail.com>
# License: BSD 3 clause

from __future__ import print_function
from time import time
from pprint import pprint
import json
from copy import copy
import pandas as pd
from pylab import scatter, show, legend, xlabel, ylabel
from matplotlib import pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation
from sklearn.datasets import fetch_20newsgroups

# n_samples = 2000
n_samples = 200
# n_features = 1000
n_features = 100
# n_components = 10  # number of topics
n_components = 2
n_top_words = 20

def print_top_words(model, feature_names, n_top_words):
    topic_set = []
    for topic_idx, topic in enumerate(model.components_):
        message = "Topic #%d: " % topic_idx
        message += " ".join([feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]])
        topic_set.append([feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]])
        # test1 = topic.argsort() # return sort result index (from 0 to max)
        # test2 = test1[:-n_top_words]
        # test3 = test1[1:-1]
        # test4 = test1[:-n_top_words - 1:-1] # 20 words index, from max to max-20
        print(message)
    print()
    return topic_set


# Load the 20 newsgroups dataset and vectorize it. We use a few heuristics
# to filter out useless terms early on: the posts are stripped of headers,
# footers and quoted replies, and common English words, words occurring in
# only one document or in at least 95% of the documents are removed.

print("Loading dataset...")
t0 = time()

# author example
"""
dataset = fetch_20newsgroups(shuffle=True, random_state=1, remove=('headers', 'footers', 'quotes'))
data_samples = dataset.data[:n_samples]
# pprint.pprint(data_samples)
"""

# ------------------------------------------------
# data load
with open('news_eng.json', encoding='UTF8') as data_file:
    data = json.load(data_file)

Corpus_set = []
for article in data:
    Corpus_set.append(article['eng_contents'])
data_samples = Corpus_set

print("done in %0.3fs." % (time() - t0))


# -----------------------------------------------------------------
# Use tf-idf features for NMF.
print("Extracting tf-idf features for NMF...")
tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=2,
                                   max_features=n_features,
                                   stop_words='english')
t0 = time()
tfidf = tfidf_vectorizer.fit_transform(data_samples)
print("done in %0.3fs." % (time() - t0))

# Use tf (raw term count) features for LDA.
print("Extracting tf features for LDA...")
tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2,
                                max_features=n_features,
                                stop_words='english')



# Fit the NMF model
print("Fitting the NMF model (Frobenius norm) with tf-idf features, "
      "n_samples=%d and n_features=%d..."
      % (n_samples, n_features))
t0 = time()
nmf = NMF(n_components=n_components, random_state=1,
          alpha=.1, l1_ratio=.5).fit(tfidf)
print("done in %0.3fs." % (time() - t0))

print("\nTopics in NMF model (Frobenius norm):")
tfidf_feature_names = tfidf_vectorizer.get_feature_names()
topic_set = print_top_words(nmf, tfidf_feature_names, n_top_words)


# -------------------------------------------------------------------
# Similarity
print(topic_set[0])
label_data = copy(data)
# topic 1: topic_set[0]
# topic 2: topic_set[1]
for idx, article in enumerate(data):
    cnt = 0

    for topic in topic_set[0]:  # topic 1
        cnt_topic = article['eng_contents'].count(topic)
        if cnt_topic > 0:
            cnt += 1+cnt_topic/10

    for topic in topic_set[1]:  # topic 1
        cnt_topic = article['eng_contents'].count(topic)
        if cnt_topic > 0:
            cnt -= 1+cnt_topic/10
    label_data[idx]['vec'] = cnt

# --------------------------
# plot
article_series = []
for article in label_data:
    article_series.append([article['write'], article['vec']])


df_series = pd.DataFrame(article_series)
df_series.columns = ["date", "vec"]
df_series['date'] = pd.to_datetime(df_series['date'])
print(df_series)
df_series.set_index('date', inplace=False)
scatter(df_series['date'].tolist(), df_series['vec'].tolist(), marker='o', c='b')
xlabel('Article insert time')
ylabel('Similarity Vector')
legend('vector')
# plt.plot(df_series['date'], df_series['vec'], color='r')
plt.xlim('2018-04-01 00:00:00','2018-04-16 23:59:59')
plt.show()