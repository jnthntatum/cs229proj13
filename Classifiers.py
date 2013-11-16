#!/usr/bin/python 
"""
Jon Tatum
Classifiers.py
Classifiers for sentiment problem
"""

import numpy;

class Vectorizer(object):
    def __init__(self):
        pass
    def to_vector(self, input):
        return 

# initial model
class CorpusStatsVectorizer(Vectorizer):
    def __init__(self, token_map):
        super(NBVectorizer, self).__init__()

    def to_vector(self, tweet):
        n = len(token_map.keys())
        v = numpy.zeros([n, 1])
        for token in tweet['tokens']:
            if token in token_map:
                v[token_map[token]] += 1.0
            else:
                v["<OMIT>"] += 1.0
        return v


class Classifier(object):
    def __init__(self):
        pass
    def train(self, trainging_set):
        raise ValueError("Abstract Class")
        pass
    def classify(self, example):
        raise ValueError("Abstract Class")
        pass
    def classify_many(self, examples):
        result = numpy.zeros([len(examples), 1])
        for i, example in enumerate(examples):
            result[i] = self.classify(example)
        return result

# MULTINOMIAL NAIVE BAYES
class NBClassifier(Classifier):
    """
    multinomial bayes classifier (easy baseline for checking methods)
    """
    def __init__(self):
        super(NBClassifier, self).__init__()
        self.px_given_label = None
        self.p_label

    def train(self, training_set):
        #MLE
        labels = training_set.keys()
        pass
    
    def classify(self, example):
        pass

# SVM HERE!!!