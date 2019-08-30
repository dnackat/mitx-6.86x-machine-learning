from string import punctuation, digits
import csv
import numpy as np
import matplotlib.pyplot as plt

import sys

if sys.version_info[0] < 3:
    PYTHON3 = False
else:
    PYTHON3 = True

def load_data(path_data):
    """Return a dictionary for the state descriptions displayed to player"""
    global PYTHON3

    data = []
    if PYTHON3:
        f_data = open(path_data, encoding="latin1")
    else:
        f_data = open(path_data)

    reader = csv.reader(f_data, delimiter='\t')

    for row in reader:
    	data.append(row)

    f_data.close()

    return data


def ewma(a, alpha=0.9):
    """Computes the exponentially weighted moving average of a"""
    b = np.array(a)
    n = b.size
    w0 = np.ones(n) * alpha
    p = np.arange(n - 1, -1, -1)
    return np.average(b, weights=w0 ** p)


def extract_words(input_string):
    """
    Helper function for bag_of_words()
    Inputs a text string
    Returns a list of lowercase words in the string.
    Punctuation and digits are separated out into their own words.
    """
    for c in punctuation + digits:
        input_string = input_string.replace(c, ' ' + c + ' ')
    return input_string.lower().split()


def bag_of_words(texts):
    """
    Inputs a list of string descriptions
    Returns a dictionary of unique unigrams occurring over the input
    """
    dictionary = {}  # maps word to unique index
    for text in texts:
        word_list = extract_words(text[0])
        for word in word_list:
            if word not in dictionary:
                dictionary[word] = len(dictionary)
    return dictionary


def extract_bow_feature_vector(state_desc, dictionary):
    """
    Inputs a string state description
    Inputs the dictionary of words as given by bag_of_words
    Returns the bag-of-words vector representation of the state
    The returned vector is of dimension m, where m the total number of entries in the dictionary.
    """
    state_vector = np.zeros([len(dictionary)])
    word_list = extract_words(state_desc)
    for word in word_list:
        if word in dictionary:
            state_vector[dictionary[word]] += 1

    return state_vector
