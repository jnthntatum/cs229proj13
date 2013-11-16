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
class UnigramVectorizer(Vectorizer):
    def __init__(self, token_map):
        super(NBVectorizer, self).__init__()
        self.token_map = token_map

    def to_vector(self, tweet):
        n = len(self.token_map.keys())
        v = numpy.zeros([n, 1])
        for token in tweet['tokens']:
            if token in self.token_map:
                v[self.token_map[token]] += 1.0
            else:
                v["<OMIT>"] += 1.0
        return v

class BigramVectorizer(UnigramVectorizer):
    def __init__(self, token_map, bigram_map):
        super(BigramVectorizer, self).__init__(token_map)
        
    def to_vector(self, tweet):
        pass

class Classifier(object):
    class ClassifierException(ValueError):
        pass

    def __init__(self):
        pass
    def train(self, training_set, labels):
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
    parameters: n -- the size of the feature vectors
                k -- the number of classes to consider
    """
    def __init__(self, n, labels):
        super(NBClassifier, self).__init__()
        self.conditional_prob = None
        self.class_priors = None
        self.n = n
        self.k = len(labels)
        self.label_to_idx = {l:i for i, l in enumerate(labels)}
        self.idx_to_label = {i:l for i, l in enumerate(labels)}

    def train(self, training_set, labels):
        #MLE
        self.class_priors = numpy.zeros([self.k, 1])
        self.conditional_priors = numpy.zeros([self.k, self.n])
        #Convert parameters to logspace
        pass
    
    def classify(self, example):
        if self.class_priors is None:
            raise ClassifierException("ERROR: model not trained")
        


# SVM HERE!!!