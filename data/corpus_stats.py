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


TWEET_DATA = "normalized_tweets.dat"


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
    pass

def kfold_validation(k=3, tweets, vectorizer, classifierType):
    # split data by label to preserve relative frequencies
    v = vectorizer()
    c = classifierType()
    m = map( lambda t: (t['label'], v.to_vector(t)), tweets)    
    shuffle(m) # don't really need to recover order
    data = reduce(lambda r, t: r[t[0]].append(t[1]), m, defaultdict(list))
    
    tp = Counter()
    fp = Counter()

    for i in xrange(k):
        train, validation = split(data)
        c.train(train)


if __name__ == "__main__":
    tweets = load_tweets(TWEET_DATA)
    ld, m, tf, tdf = cumulative_stats(tweets)
