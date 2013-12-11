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
        super(UnigramVectorizer, self).__init__()
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

class KGramVectorizer(Vectorizer):
    def __init__(self, token_map, kgram_map, k):
        super(KGramVectorizer, self).__init__()
        self.kgram_map = kgram_map
        self.token_map = token_map
        self.k = k
    
    def to_vector(self, tweet):
        toks = tweet['tokens']
        k = self.k
        padlen = k-1
        padded = ["<NULL>" for i in xrange(padlen)] + toks + ["<NULL>" for i in xrange(padlen)]
        v = numpy.zeros(self.feature_size) 
        for i in xrange(padlen-1, len(toks) + padlen) :
            kgid = tuple(toks[i:i+k])
            if kgid in self.kgram_map:
                idx = self.kgram_map[kgid]  
            else: 
                idx = self.kgram_map["<OMIT>"]
            v[idx] += 1
        return v
            
    @property
    def feature_size(self):
        return len(self.kgram_map)

class KGramUniGramVectorizer(Vectorizer):
    def __init__(self, token_map, kgram_map, k):
        super(KGramUniGramVectorizer, self).__init__()
        self.kgv = KGramVectorizer(token_map, kgram_map, k)
        self.ugv = UnigramVectorizer(token_map)

    def to_vector(self, tweet):
        v = numpy.zeros(self.feature_size)
        a = self.ugv.to_vector(tweet)
        b = self.kgv.to_vector(tweet)
        al = self.ugv.feature_size - 1 
        bl = self.kgv.feature_size
        v[0:al] = a[0:al]
        v[al:al+bl] = b[0:bl]
        return v

    @property
    def feature_size(self):
        return self.kgv.feature_size + self.ugv.feature_size - 1
    
    
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
        feature_priors = numpy.zeros([1, self.n]) 
        for label in self.label_to_idx:
            lbl_idx = self.label_to_idx[label]
            indices = numpy.where(labels == label)[0]
            class_examples = training_set[indices,:]
            theta[lbl_idx,:] = class_examples.sum(0) + 2 # sum over rows
            feature_priors += theta[lbl_idx,:]
            theta[lbl_idx,:] *= 1.0 / theta[lbl_idx,:].sum()
            class_priors[lbl_idx] = len(indices) / float(num_examples)
        #Convert parameters to logspace
        self.class_priors = numpy.log(class_priors)
        feature_priors *= 1.0 / feature_priors.sum() 
        self.feature_priors = numpy.log(feature_priors).T 
        self.theta = numpy.log(theta).T
    
    def classify(self, example):
        if self.class_priors is None:
            raise ClassifierException("ERROR: model not trained")
        ex = example.reshape([1, self.n])
        lls = ex.dot(self.theta) - ex.dot(self.feature_priors)
        if self.use_priors:
            lls += self.class_priors.T
        return self.idx_to_label[numpy.argmax(lls)]

class PCANBayes(NBClassifier):
    def __init__(self, n, labels, k=200, use_priors=True):
        super(PCANBayes, self).__init__(k, labels, use_priors)
        self.basis = None
    def train(self, training_set, labels):
        m = training_set.shape[0]
        n = training_set.shape[1]
        mus = training_set.sum(0) * 1 / m
        print mus.shape
        b = training - ones([m, 1]).dot(mus)
        cov = b.T.dot(b)
        z = numpy.diag(numpy.sqrt((m-1)/cov.diagonal()))
        cov = z * cov * z
        evals, evecs = numpy.linalg.eigh(cov)
        self.mus = mus
        self.cov = cov
        self.z = z
        self.evecs = evecs
        self.evals = evals
        self.basis = evecs[:,(-evals).argsort()[0: self.k]]
        projection = b.dot(z).dot(self.basis)    
        super(PCANBayes, self).train(projection, labels)
    def project(data):
        if self.basis is None: 
            raise ClassifierException("model not trained")
        m = data.shape[0] 
    def classify(self, example):
        if self.basis is None: 
            raise ClassifierException("model not trained")
        ex = example.reshape([1, n]) 
# SVM HERE!!!