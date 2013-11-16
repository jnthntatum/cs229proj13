#!/usr/bin/python
"""
corpus_stats.py
compute statistics on word frequencies over our corpus of tweets.
throughout this file N refers to the number of documents
m refers to the number of toks overall
k refers to the number of toks per document 
"""

from numpy import *
from collections import Counter, defaultdict
from random import shuffle
import pylab
import json
import math
from classifiers import *

TWEET_DATA = "data/normalized_tweets.dat"


"""
def hist_counter(fname, counters):
    pylab.figure()
    pylab.plot(counters.keys(), counters.values())
"""


"""
Special token for words that are ignored because they are too
frequent or under frequent
"""

OMIT = "<OMIT>"

def sort_and_zip(counter):
    a = sorted(counter.keys())
    return zip(a, [counter[ai] for ai in a ]) 

def write_counter(fname, counter):
    with open(fname, 'w+') as out:
        keys = sorted(counter.keys(), key=lambda x: counter[x])
        for key in keys:
            out.write("%s\t%d\n" % (key,counter[key]))

def hist_counter(counter):
    hist = Counter()
    for key in counter:
        hist[counter[key]] += 1
    return hist


def cumulative_stats(tweets):
    """
    makes a pass over all documents counting general statistics
    and producing a list of unique tokens
    """  
    length_dist = Counter()
    tok_freq = Counter()
    tok_doc_freq = Counter()  
    m = 0

    for tweet in tweets:
        toks = tweet['tokens']
        k = len(toks)
        length_dist[k] += 1
        m += k
        for tok in toks:
            tok_freq[tok.lower()] += 1
        for tok in set(toks):
            tok_doc_freq[tok.lower()] += 1
    return length_dist, m, tok_freq, tok_doc_freq

def load_tweets(fname):
    tweets = []
    with open(fname, 'r') as fhandle:
        for line in fhandle:
            tweets.append(json.loads(line))
    return tweets


def split(full_set, k, i):
    """
    return a split of the data so we have one fold of 
    validation and the rest as training.

    return: training_set, validation_set
    """
    m = full_set.shape[0]
    fold_sz = int(math.ceil(m / float(k)))
    validation_idxs = range(i*fold_sz, min((i+1)*fold_sz, m))
    training_idxs = range(0, i*fold_sz) + range((i+1)*fold_sz, m)  
    return full_set[training_idxs,:], full_set[validation_idxs, :]

def unpack_labels(examples):
    """
    separate examples from labels
    returns y, a column vector of labels and X, the design matrix

    return: y, X
    """
    l = examples[:,0]
    e = examples[:,1:]
    return l, e

def pack_labels(tuples):
    m = len(tuples)
    n = tuples[0][1].size
    M = zeros([m, n + 1])
    for i, t in enumerate(tuples):
        M[i,1:] = t[1]
        M[i,0]  = t[0]
    return M 

def kfold_validation(tweets, vectorizer, classifier, k=3):
    # split data by label to preserve relative frequencies
    data = map( lambda t: (t['label'], vectorizer.to_vector(t)), tweets)    
    shuffle(data)
    data = pack_labels(data)
    
    # todo: ROC curve calculation
    tp = Counter()
    fp = Counter()
    fn = Counter()
    tn = Counter()
    
    errs = 0

    for i in xrange(k):
        train, validation = split(data, k, i)
        tl, te = unpack_labels(train)
        vl, ve = unpack_labels(validation)
        classifier.train(te, tl)
        predictions = classifier.classify_many(ve)
        
        for i in [-1, 0, 1]:
            tp[i] += ((vl == i) * (predictions == i)).sum()
            tn[i] += ((vl != i) * (predictions != i)).sum()
            fp_i = ((vl != i) * (predictions == i)).sum()
            fp[i] += fp_i
            fn_i = ((vl == i) * (predictions != i)).sum()
            fn[i] += fn_i
            errs += fn_i    
    m = data.shape[0]
    return float(errs) / m, tp, fp, fn, tn

def generate_dictionary(tf, min_threshold):
    result = {}
    index = 0
    for token in tf:
        if tf[token] > min_threshold:
            result[token] = index
            index += 1
    result[OMIT] = index
    return result

def print_stats(tp, fp, fn, tn):
    for i in tp:
        print "======= label=%d ========" % i
        print "tp: %d; tn: %d; fp: %d; fn: %d" % (tp[i], tn[i], fp[i], fn[i])
        print "accuracy:\t%0.3f" % ((tn[i] + tp[i]) / float(fp[i] + tn[i] + tp[i]+ fp[i]))
        print "specificity:\t%0.3f" % (tn[i]/ float(fp[i] + tn[i]))
        print "sensitivity:\t%0.3f" % (tp[i]/ float(tp[i] + fn[i]))
        print "negative predictive value:\t%0.3f" % (tn[i]/ float(fn[i] + tn[i]))
        print "positive predictive value:\t%0.3f" % (tp[i]/ float(fp[i] + tp[i]))
        print ""

if __name__ == "__main__":
    tweets = load_tweets(TWEET_DATA)
    ld, m, tf, tdf = cumulative_stats(tweets)
    models_learned = []
    for i in xrange(15):
        params = generate_dictionary(tdf, i)
        v = UnigramVectorizer(params)
        c = NBClassifier(v.feature_size, [-1, 0, 1])  
        acc, tp, fp, fn, tn = kfold_validation(tweets, v, c, 4)
        print "<><><><><><><><><><><>"
        print "======================"
        print "%d (%d features) -- %0.3f" % (i, v.feature_size, acc)
        print_stats(tp, fp, fn, tn)