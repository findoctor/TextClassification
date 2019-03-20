'''
Using gram matrix to do multi-class SVM
'''
import wordKernel
import thread
from multiprocessing.dummy import Pool as ThreadPool
import ngk
import ssk
import pickle
import dataProcess
import os
from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import label_binarize
from sklearn.metrics import precision_recall_curve
from sklearn.multiclass import OneVsOneClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import precision_score
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import SGDClassifier
from sklearn import metrics


def save_semantic_kernel(train_text, train_labels, test_text, test_labels):
    n_train = len(train_data)
    n_test = len(test_data)
    feature_matrix = np.zeros((n_train, n_train))
    test_feature_matrix = np.zeros((n_test, n_train))

    for i in range(n_train):
        for j in range(n_train):
            print('Train Round ' + str(i + 1) + ',Sample ' + str(j + 1))
            feature_matrix[i][j] = wordKernel.semantic_word_kernel(train_text[i], train_text[j])

    for i in range(n_test):
        for j in range(n_train):
            print('Test Round ' + str(i + 1) + ',Sample ' + str(j + 1))
            test_feature_matrix[i][j] = wordKernel.semantic_word_kernel(test_text[i], train_text[j])

    with open('semantic_train_kernel.pkl', 'wb') as handle:
        pickle.dump(feature_matrix, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open('semantic_test_kernel.pkl', 'wb') as handle2:
        pickle.dump(test_feature_matrix, handle2, protocol=pickle.HIGHEST_PROTOCOL)
    semantic_train_kernel_path = 'semantic_train_kernel.pkl'
    semantic_test_kernel_path = 'semantic_test_kernel.pkl'
    return semantic_train_kernel_path, semantic_test_kernel_path


def save_wk_kernel(train_text, train_labels, test_text, test_labels):
    n_train = len(train_data)
    n_test = len(test_data)
    feature_matrix = np.zeros((n_train, n_train))
    test_feature_matrix = np.zeros((n_test, n_train))

    for i in range(n_train):
        for j in range(n_train):
            feature_matrix[i][j] = wordKernel.word_kernel(train_text[i], train_text[j])

    for i in range(n_test):
        for j in range(n_train):
            test_feature_matrix[i][j] = wordKernel.word_kernel(test_text[i], train_text[j])

    with open('wk_train_kernel.pkl', 'wb') as handle:
        pickle.dump(feature_matrix, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open('wk_test_kernel.pkl', 'wb') as handle2:
        pickle.dump(test_feature_matrix, handle2, protocol=pickle.HIGHEST_PROTOCOL)
    wk_train_kernel_path = 'wk_train_kernel.pkl'
    wk_test_kernel_path = 'wk_test_kernel.pkl'
    return wk_train_kernel_path, wk_test_kernel_path


def save_ngk_kernel(train_text, train_labels, test_text, test_labels, n):
    n_train = len(train_data)
    n_test = len(test_data)
    feature_matrix = np.zeros((n_train, n_train))
    test_feature_matrix = np.zeros((n_test, n_train))

    for i in range(n_train):
        for j in range(n_train):
            feature_matrix[i][j] = ngk.compute_ngk(train_text[i], train_text[j], n)

    for i in range(n_test):
        for j in range(n_train):
            test_feature_matrix[i][j] = ngk.compute_ngk(test_text[i], train_text[j], n)
    base_path1 = 'ngk_train_kernel'
    base_path2 = 'ngk_test_kernel'
    with open(base_path1 + str(n) + '.pkl', 'wb') as handle:
        pickle.dump(feature_matrix, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(base_path2 + str(n) + '.pkl', 'wb') as handle2:
        pickle.dump(test_feature_matrix, handle2, protocol=pickle.HIGHEST_PROTOCOL)


def save_ssk_kernel(train_text, train_labels, test_text, test_labels, n, l):
    n_train = len(train_data)
    n_test = len(test_data)
    feature_matrix = np.zeros((n_train, n_train))
    test_feature_matrix = np.zeros((n_test, n_train))

    for i in range(n_train):
        print('Train Gram, Round' + str(i + 1))
        for j in range(n_train):
            print('Train No.' + str(j + 1))
            feature_matrix[i][j] = ssk.ssk_kernel(train_text[i], train_text[j], n, l)

    for i in range(n_test):
        print('Test Gram, Round' + str(i + 1))
        for j in range(n_train):
            test_feature_matrix[i][j] = ssk.ssk_kernel(test_text[i], train_text[j], n, l)
    base_path1 = 'ssk_train_kernel'
    base_path2 = 'ssk_test_kernel'
    with open(base_path1 + str(n) + '.pkl', 'wb') as handle:
        pickle.dump(feature_matrix, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(base_path2 + str(n) + '.pkl', 'wb') as handle2:
        pickle.dump(test_feature_matrix, handle2, protocol=pickle.HIGHEST_PROTOCOL)


def load_wk_kernel(train_path, test_path):
    with open(train_path) as fd:
        train_kernel = pickle.load(fd)

    with open(test_path) as fd2:
        test_kernel = pickle.load(fd2)

    return train_kernel, test_kernel


def load_ngk_kernel(train_path, test_path):
    with open(train_path) as fd:
        train_kernel = pickle.load(fd)

    with open(test_path) as fd2:
        test_kernel = pickle.load(fd2)

    return train_kernel, test_kernel


def load_ssk_kernel(train_path, test_path):
    with open(train_path) as fd:
        train_kernel = pickle.load(fd)

    with open(test_path) as fd2:
        test_kernel = pickle.load(fd2)

    return train_kernel, test_kernel


def evaluate(feature_matrix, test_feature_matrix, train_labels, test_labels):
    classifier = svm.SVC(kernel='precomputed')
    classifier.fit(feature_matrix, train_labels)
    y_pred = list(classifier.predict(test_feature_matrix))
    # cm = precision_score(train_labels, y_pred, average=None)
    print(test_labels)
    print(y_pred)
    print(metrics.classification_report(test_labels, y_pred, target_names=['earn', 'acq', 'crude', 'corn']))



#########__main__#######
'''
wk_train_kernel_path = 'train_data_newK.pkl'
wk_test_kernel_path = 'test_data_newK.pkl'
train_data, test_data = dataProcess.load_pickled_data(wk_train_kernel_path, wk_test_kernel_path)

train_labels = [i[1] for i in train_data]
train_text = [i[0] for i in train_data]
test_labels = [i[1] for i in test_data]
test_text = [i[0] for i in test_data]
'''

'''
n = 3
l = 0.5
ssk_train_path = 'ssk_train_kernel' + str(n) + '.pkl'
ssk_test_path = 'ssk_test_kernel' + str(n) + '.pkl'
save_ssk_kernel(train_text, train_labels, test_text, test_labels, n, l)
ssk_train_gram, ssk_test_gram = load_ssk_kernel(ssk_train_path, ssk_test_path)
evaluate(train_gram, test_gram, train_labels, test_labels)
'''


# wk_train_kernel_path, wk_test_kernel_path = save_wk_kernel(train_text, train_labels, test_text, test_labels)
train_dataPath = 'train_data_newK.pkl'
test_dataPath = 'test_data_newK.pkl'
train_data, test_data = dataProcess.load_pickled_data(train_dataPath, test_dataPath)

train_labels = [i[1] for i in train_data]
train_text = [i[0] for i in train_data]
test_labels = [i[1] for i in test_data]
test_text = [i[0] for i in test_data]

pool = ThreadPool(4)
results = pool.map(save_semantic_kernel, (train_text, train_labels, test_text, test_labels))

#semantic_train_kernel_path, semantic_test_kernel_path = save_semantic_kernel(train_text, train_labels, test_text, test_labels)
#semantic_train_kernel_path = 'semantic_train_kernel.pkl'
#semantic_test_kernel_path = 'semantic_test_kernel.pkl'
train_gram, test_gram = load_wk_kernel(semantic_train_kernel_path, semantic_test_kernel_path)
print(train_gram)
evaluate(train_gram, test_gram, train_labels, test_labels)

'''
# evaluate ngk performance
n = 3
ngk_train_gram_path = 'ngk_train_kernel'
ngk_test_gram_path = 'ngk_test_kernel'
ngk_train_gram_path = ngk_train_gram_path + str(n) + '.pkl'
ngk_test_gram_path = ngk_test_gram_path + str(n) + '.pkl'
save_ngk_kernel(train_text, train_labels, test_text, test_labels, n)
train_ngk_gram, test_ngk_gram = load_ngk_kernel(ngk_train_gram_path, ngk_test_gram_path)
evaluate(train_ngk_gram, test_ngk_gram, train_labels, test_labels)
'''
