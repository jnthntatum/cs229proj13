#!/usr/bin/python
"""
normalize.py
"""

import nltk
from numpy import *
import json
import csv
from time import strftime, strptime

TWEETS = './twitter_info.dat' 
LABELS = './tweet_sentiment.csv'

SENTIMENTS = {
    "positive": 1,
    "negative": -1,
    "neutral" : 0,
    "irrelevant": 0 
}

def load_labels(fname):
    """
    return a map of twitter ids to (sentiment, topic)
    """
    result = {}
    with open(fname, 'rb') as fhandle:
        labels = csv.reader(fhandle)
        for label in labels:
            result[label[2]] = (SENTIMENTS[label[1]], label[0])
    return result

def load_tweets(fname):
    result = []
    with open(fname, 'r') as fhandle:
        for row in fhandle:
            if row == "false\n":
                # removed or otherwise unavailable tweets
                continue
            result.append(json.loads(row))

    return result
def normalize_tweets(tweet_list, sentiment):
    """
    extract just the information we want to work on
    1. id
    2. text
        - tokenized
        - stemmed
        - mentions and hashtags removed (eventually want to predict
            things about these so we should avoid using them as 
            training features) 
    3. timestamp
    4. number of followers (proxy for how reputable the user is)

    ugly and uninteresting. don't look too closely...

    """ 
    result = [] 
    stemmer = nltk.stem.PorterStemmer()
    for tweet in tweet_list:
        if not tweet:
            continue
        short_tweet = {}
        tid = short_tweet["id"] = tweet["id"]
        short_tweet["created_at"] = strftime('%Y-%m-%d %H:%M:%S', strptime(tweet['created_at'],'%a %b %d %H:%M:%S +0000 %Y'))
        tid = str(tid)
        short_tweet["label"] = sentiment[tid][0]
        short_tweet["topic"] = sentiment[tid][1]
        short_tweet["retweets"] = tweet["retweet_count"]
        short_tweet["friends"] = tweet["user"]["friends_count"]
        short_tweet["followers"] = tweet["user"]["followers_count"]
        short_tweet["hashtags"] = map(
            lambda x: x['text'], tweet["entities"]["hashtags"])
        short_tweet["mentions"] = map(
            lambda x: (x['name']), tweet["entities"]["user_mentions"]) 

        # first pass at text normalization
        tokens = nltk.word_tokenize(tweet["text"])
        stems = []
        i = 0
        while i < len(tokens): 
            # strip mentions
            token = tokens[i] 
            if token == "@": 
                i += 1
            elif token != "#":
                stems.append(stemmer.stem_word(token))
            i += 1
        short_tweet['tokens'] = stems
        result.append(short_tweet)
    return result

if __name__ == "__main__":
    l = load_labels(LABELS)
    t = load_tweets(TWEETS)
    normalized = normalize_tweets(t,l)
    for t in normalized:
        print json.dumps(t) 