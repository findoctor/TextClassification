import os
import string
import pickle
import numpy as np
import nltk
from nltk import word_tokenize
from nltk.corpus import stopwords
import string
from bs4 import BeautifulSoup


category = {
    unicode('earn'): [30, 12, 0],
    unicode('acq'): [30, 12, 1],
    unicode('crude'): [30, 12, 2],
    unicode('corn'): [30, 12, 3]
}
# nltk.download('punkt')
# nltk.download('stopwords')


def load_sgml_data(data_dir):
    """
    Load and parse training and test data from the Reuters dataset in SGML format
    Use Modified Apte dataset split
    :return: List with training data and list with test data
    """

    train_data = []
    test_data = []

    for file_name in os.listdir(data_dir):
        if file_name.endswith('.sgm'):
            file_content = BeautifulSoup(open(os.path.join(data_dir, file_name)), 'lxml')
            train_tags = file_content.find_all('reuters', lewissplit='TRAIN', topics='YES')
            train_data += [parse_document(tag) for tag in train_tags]

            test_tags = file_content.find_all('reuters', lewissplit='TEST', topics='YES')
            test_data += [parse_document(tag) for tag in test_tags]

    return train_data, test_data


def parse_document(tag):
    """
    Retrieve article body and topic list from tag structure.
    :param tag: Tag structure with article data
    :return: Tuple holding the document body and topic list
    """

    topics = [unicode(d_tag.text) for d_tag in tag.find('topics').find_all('d')]
    text_tag = tag.find('text')
    text = text_tag.body.text if text_tag.body else text_tag.contents[-1]

    return unicode(text), topics


def trim_docs(text, blacklist):
    stop = stopwords.words('english') + list(string.punctuation)
    filtered = [i.lstrip('0123456789 ') for i in word_tokenize(text.lower()) if i not in stop]
    res = ' '.join(filtered)
    return res


def select_small_dataset(train_data, test_data, blacklist):
    train_small_data = []
    test_small_data = []
    nums_category = {
        unicode('earn'): [0, 0],
        unicode('acq'): [0, 0],
        unicode('crude'): [0, 0],
        unicode('corn'): [0, 0]
    }
    for key, val in category.items():
        for (text, topic) in train_data:
            if key in topic:
                nums_category.get(key)[0] += 1
                train_small_data.append((trim_docs(text, blacklist), category.get(key)[2]))
                if nums_category.get(key)[0] == category.get(key)[0]:
                    break
    for key, val in category.items():
        for (text, topic) in test_data:
            if key in topic:
                nums_category.get(key)[1] += 1
                test_small_data.append((trim_docs(text, blacklist), category.get(key)[2]))
                if nums_category.get(key)[1] == category.get(key)[1]:
                    break

    return train_small_data, test_small_data


def save_into_pickle(train_data, test_data):
    with open('train_data_newK.pkl', 'wb') as handle:
        pickle.dump(train_data, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open('test_data_newK.pkl', 'wb') as handle2:
        pickle.dump(test_data, handle2, protocol=pickle.HIGHEST_PROTOCOL)


def load_pickled_data(train_data_path, test_data_path):
    with open(train_data_path) as fd:
        train_data = pickle.load(fd)

    with open(test_data_path) as fd:
        test_data = pickle.load(fd)

    return train_data, test_data


datadir = 'reuters21578'
train_data, test_data = load_sgml_data(datadir)
train_small_data, test_small_data = select_small_dataset(train_data, test_data, stopwords)
save_into_pickle(train_small_data, test_small_data)

train_path = 'train_data_newK.pkl'
test_path = 'test_data_newK.pkl'
train_small_data, test_small_data = load_pickled_data(train_path, test_path)
print(train_small_data[1])
print(len(train_small_data))
print(len(test_small_data))
