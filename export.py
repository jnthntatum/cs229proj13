#!/usr/bin/python
"""
Export.python 
"""

import numpy
from corpus_stats import *
import classifiers

def export_pca():
    tweets = load_tweets(TWEET_DATA)
    tf, tdf = count_tokens(tweets)
    # bigrams
    kgram_freq, kgram_doc_freq = count_kmers(tweets,2,2)
    print "finished with corp stats"
    for p in [l * 100 for l in xrange(2, 8)]: 
        for i in xrange(2, 10):
            token_map = generate_dictionary(tf, i)
            kgram_map = generate_dictionary(kgram_freq, i)
            v3 = KGramUniGramVectorizer(token_map, kgram_map, 2)
            v1 = UnigramVectorizer(token_map) 
            v2 = KGramVectorizer(token_map, kgram_map, 2)
            vs = [v1, v2, v3]
            pca = classifiers.PCA(p)
            for j, v in enumerate(vs):
                if v.feature_size > 5000:
                    continue
                header = ",".join([str(k+1) for k in xrange(p + 1)])    
                suffix = ".%d.%d.%d.csv" % (i, j, p)      
                d = to_dataset(tweets, v)
                trd, ted = split(d, 5, 4)
                trl, tre = unpack_labels(trd)
                tel, tee = unpack_labels(ted)
                trep = pca.train(tre)
                teep = pca.project(tee)
                tro = repack_labels(trl, trep)
                teo = repack_labels(tel, teep) 
                numpy.savetxt("train"+suffix, tro ,delimiter=",", header=header, comments="")
                numpy.savetxt("test"+suffix, teo ,delimiter=",", header=header, comments="")

def export_n_grams():
    tweets = load_tweets(TWEET_DATA)
    tf, tdf = count_tokens(tweets)
    # bigrams
    kgram_freq, kgram_doc_freq = count_kmers(tweets,2,2)
    print "finished with corp stats"
    lbls = {0: "unigram", 1:"bigram", 2: "uni-bigram" }   
    for i in xrange(2, 10):
        token_map = generate_dictionary(tf, i)
        kgram_map = generate_dictionary(kgram_freq, i)
        v3 = KGramUniGramVectorizer(token_map, kgram_map, 2)
        v1 = UnigramVectorizer(token_map) 
        v2 = KGramVectorizer(token_map, kgram_map, 2)
        vs = [v1, v2, v3]
        for j, v in enumerate(vs):
            if v.feature_size > 5000:
                continue
            header = ",".join([str(k+1) for k in xrange(v.feature_size + 1)])    
            suffix = "%d.%s.csv" % (i, lbls[j])      
            d = to_dataset(tweets, v)
            trd, ted = split(d, 5, 4) 
            numpy.savetxt("train"+suffix, trd ,delimiter=",", header=header, comments="")
            numpy.savetxt("test"+suffix, ted ,delimiter=",", header=header, comments="")

def load_file(fname):
    return numpy.load_text(fname, delimiter=',', skiprows=1)

if __name__ == "__main__":
    # export_pca
    # export_n_grams()