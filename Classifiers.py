#!/usr/bin/python 
"""
Jon Tatum
Classifiers.py
Classifiers for sentiment problem
"""

import numpy;

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
            result[i] = classify(example)

