#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  5 21:49:58 2020

@author: georgehan
len(data)
Out[33]: 253272
"""

#%matplotlib inline

import os
os.chdir('/Users/georgehan/TDI/Miniproject/nlp')
file = '/Users/georgehan/TDI/Miniproject/nlp/yelp_train_academic_dataset_review_reduced.json.gz'

import matplotlib
import seaborn as sns
sns.set()
matplotlib.rcParams['figure.dpi'] = 144
import dill
import scipy.sparse
from scipy.sparse import csr_matrix
from scipy.sparse import lil_matrix
from sklearn.linear_model import SGDRegressor
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


from sklearn.feature_extraction.text import TfidfVectorizer
from spacy.lang.en.stop_words import STOP_WORDS
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.decomposition import TruncatedSVD

from sklearn.pipeline import Pipeline
from sklearn import base

import gzip
import ujson as json

with gzip.open(file, 'rb') as f:
    data = [json.loads(line) for line in f]



stars = [row['stars'] for row in data]
text = [row['text'] for row in data]



ng_tfidf=TfidfVectorizer(max_features=300,
                         ngram_range=(1,1),
                         stop_words=STOP_WORDS.union({'ll','ve'}))
#ng_tfidf=ng_tfidf.fit( fruit_sents + company_sents )
#print(ng_tfidf.get_feature_names()[100:105])
#print(ng_tfidf.transform(fruit_sents + company_sents))


t = time.process_time()
X = ng_tfidf.fit_transform(text)
print(time.process_time() - t)


t = time.process_time()
X = ng_tfidf.fit_transform(text)
print(time.process_time() - t)

t = time.process_time()
X_array = csr_matrix(X).toarray()
print(time.process_time() - t)

t = time.process_time()
reg = SGDRegressor(max_iter=1000, tol=1e-3)
reg.fit(X_array, stars)
print(time.process_time() - t)

class ColumnSelectTransformer(base.BaseEstimator, base.TransformerMixin):

    def __init__(self, col_names):
        self.col_names = col_names  # We will need these in transform()

    def fit(self, X, y=None):
        # This transformer doesn't need to learn anything about the data,
        # so it can just return self without any further processing
        return self

    def transform(self, X):
        # Return an array with the same number of rows as X and one
        # column for each in self.col_names
        return [row[column] for column in self.col_names for row in X]



bigram_est = Pipeline([
    # Column selector (remember the ML project?)
    ('ColumnSelectTransformer', ColumnSelectTransformer(['text'])),
    ('TfidfVectorizer', TfidfVectorizer(max_features=None,
                         ngram_range=(1,2),
                         stop_words=STOP_WORDS.union({'ll','ve'}))),
    ('TruncatedSVD', TruncatedSVD(n_components=100, n_iter=7, random_state=42)),
    #('TfidfTransformer', TfidfTransformer(use_idf = True,
    #                     sublinear_tf = True)),
    ('SGDRegressor', SGDRegressor(max_iter=1000, tol=1e-3))
    # Vectorizer
    # Frequency filter (if necessary)
    # Regressor
])


t = time.process_time()
bigram_est.fit(data, stars)
print(time.process_time() - t)
