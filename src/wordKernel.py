import os
import numpy as np
from scipy import spatial
from gensim import models
from gensim.models import Word2Vec
import nltk
import math
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import wordnet

# here we return all the feature vectors of the docs
nltk.download('wordnet')
wn_lemmas = set(wordnet.all_lemma_names())


def word_kernel(s, t):
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform([s, t])
    print(vectorizer.get_feature_names())
    feature_matrix = X.toarray()  # 2*K
    return np.dot(feature_matrix[0], feature_matrix[1]) / np.sqrt(np.dot(feature_matrix[0], feature_matrix[0]) * np.dot(feature_matrix[1], feature_matrix[1]))


def semantic_word_kernel(s, t):
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform([s, t])
    feature_matrix = list(X.toarray())  # 2*D

    wordlist = vectorizer.get_feature_names()
    for i in range(len(wordlist)):
        wordlist[i] = wordlist[i].encode('ascii', 'ignore')
    matrix_p = compute_Pmatrix(wordlist)

    tmp1 = np.dot(feature_matrix[0], matrix_p)
    tmp2 = np.dot(matrix_p, feature_matrix[1])
    res = np.dot(tmp1, tmp2)

    #feature_matrix = np.dot(feature_matrix, matrix_p)
    #res = np.dot(feature_matrix[0], feature_matrix[1]) / np.sqrt(np.dot(feature_matrix[0], feature_matrix[0]) * np.dot(feature_matrix[1], feature_matrix[1]))
    '''
    if math.isnan(res):
        res = 0.1
    '''
    return res


def compute_Pmatrix(wordlist):
    wordlen = len(wordlist)
    Pmatrix = np.zeros((wordlen, wordlen))
    for i in range(wordlen):
        for j in range(wordlen):
            s1 = wordlist[i]
            s2 = wordlist[j]
            if i == j:
                Pmatrix[i][j] = 1.0
                continue
            elif s1.isdigit() and s2.isdigit():
                Pmatrix[i][j] = 1.0
                continue
            elif s1.isdigit() or s2.isdigit():
                Pmatrix[i][j] = 0.0
                continue
            if s1 in wn_lemmas and s2 in wn_lemmas:
                s1 = wordnet.synsets(s1)[0]
                s2 = wordnet.synsets(s2)[0]
                similarity = s1.wup_similarity(s2)
                if similarity > 0.5:
                    Pmatrix[i][j] = 1
                else:
                    Pmatrix[i][j] = 0
    #Pmatrix = Pmatrix / Pmatrix.sum(axis=0)
    return Pmatrix


def load_pickled_data(train_data_path, test_data_path):
    with open(train_data_path) as fd:
        train_data = pickle.load(fd)

    with open(test_data_path) as fd:
        test_data = pickle.load(fd)

    return train_data, test_data


'''
train_path = 'train_data_newK.pkl'
test_path = 'test_data_newK.pkl'
train_data, test_data = load_pickled_data(train_path, test_path)

train_labels = [i[1] for i in train_data]
train_text = [i[0] for i in train_data]
test_labels = [i[1] for i in test_data]
test_text = [i[0] for i in test_data]

semantic_word_kernel(train_text[3], train_text[4])
'''

'''
s1 = 'happy22'
s2 = 'record'
if not s1 in wn_lemmas:
    print('s1 not here')
    res = 0.5
    print(res)
else:
    s1 += '.n.01'
    s2 += '.n.01'
    s1 = wordnet.synset(s1)
    s2 = wordnet.synset(s2)
    print(s1.wup_similarity(s2))
'''
'''
corpus = [
    #'This is the first document aa bb cc dd ee ff gg hh ii jj jj jj kk ll mm nn nn oo pp qq rr ss tt.',
    #'This document is the second document ss aa bb bb bb bb bb bb aa aa aa jj mm nn qq tt oo aa cc.',
    #'And this is the third one.',
    #'Is this the first document?',
    train_text[3], train_text[5]
]
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(corpus)
feature_matrix = X.toarray()  # 2*K
#feature_matrix = feature_matrix / feature_matrix.sum(axis=0)
res = np.dot(feature_matrix[0], feature_matrix[1]) / np.sqrt(np.dot(feature_matrix[0], feature_matrix[0]) * np.dot(feature_matrix[1], feature_matrix[1]))
print(feature_matrix)
print(res)
'''
