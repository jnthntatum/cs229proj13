#!/usr/bin/python 
"""
Jon Tatum
Classifiers.py
Classifiers for sentiment problem
"""

import numpy

class Vectorizer(object):
    def __init__(self):
        pass
    def to_vector(self, input):
        return 

# initial model
class UnigramVectorizer(Vectorizer):
    def __init__(self, token_map):
        super(Vectorizer, self).__init__()
        self.token_map = token_map

    def to_vector(self, tweet):
        n = self.feature_size
        v = numpy.zeros(n)
        for token in tweet['tokens']:
            if token in self.token_map:
                v[self.token_map[token]] += 1.0
            else:
                v[self.token_map["<OMIT>"]] += 1.0
        return v
    
    @property
    def feature_size(self):
        return len(self.token_map.keys())

class KGramVectorizer(UnigramVectorizer):
    def __init__(self, token_map, kgram_map, k):
        super(UnigramVectorizer, self).__init__(token_map)
        self.kgram_map = None
        self.k = k
    
    def to_vector(self, tweet):
        toks = tweet['tokens']
        padlen = k-1
        padded = ["<NULL>" for i in xrange(padlen)] + toks + ["<NULL>" for i in xrange(padlen)]
        v = numpy.zeros(self.feature_size)
        for i in xrange(padlen, len(padded) - padlen):
            kgid = tuple(toks[i:i+padlen])
            if kgid in self.kgram_map:
                idx = self.kgram_map[kgid]  
            else: 
                idx = self.kgram_map["<OMIT>"]
            v[idx] += 1
        return v
            
    @property
    def feature_size(self):
        return len(self.kgram_map)
   
    

class BigramVectorizer(UnigramVectorizer):
    def __init__(self, token_map, bigram_map):
        super(BigramVectorizer, self).__init__(token_map)
        
    def to_vector(self, tweet):
        n = self.feature_size

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
        result = numpy.zeros(len(examples))
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
    def __init__(self, n, labels, use_priors=True):
        super(NBClassifier, self).__init__()
        self.theta = None
        self.class_priors = None
        self.n = n
        self.k = len(labels)
        self.label_to_idx = {l:i for i, l in enumerate(labels)}
        self.idx_to_label = {i:l for i, l in enumerate(labels)}
        self.use_priors = use_priors

    def train(self, training_set, labels):
        # MLE
        class_priors = numpy.zeros([self.k, 1])
        theta = numpy.zeros([self.k, self.n])
        num_examples = labels.shape[0]
        for label in self.label_to_idx:
            lbl_idx = self.label_to_idx[label]
            indices = numpy.where(labels == label)[0]
            class_examples = training_set[indices,:]
            theta[lbl_idx,:] = class_examples.sum(0) + 1 # sum over rows
            theta[lbl_idx,:] *= 1.0 / theta[lbl_idx,:].sum()
            class_priors[lbl_idx] = len(indices) / float(num_examples)
        #Convert parameters to logspace
        self.class_priors = numpy.log(class_priors)
        self.theta = numpy.log(theta).T
    
    def classify(self, example):
        if self.class_priors is None:
            raise ClassifierException("ERROR: model not trained")
        lls = example.reshape([1, self.n]).dot(self.theta)
        if self.use_priors:
            lls += self.class_priors.T
        return self.idx_to_label[numpy.argmax(lls)]


# SVM HERE!!!