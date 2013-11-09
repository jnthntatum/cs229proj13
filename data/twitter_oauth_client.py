#!/usr/bin/python
"""
OAUTH 2.0 python scraper.
Read the README.
(tl;dr -- don't publish anything other than twitter IDs and your derivative information relating to those ids)
"""

import rauth
import sys
import csv
import time
import json

TWITTER={
    "v1.1": {
        "root":"https://api.twitter.com/1.1/",
        "request_token":"https://api.twitter.com/oauth/request_token",
        "authorize":"https://api.twitter.com/oauth/authorize",
        "access_token":"https://api.twitter.com/oauth/access_token",
        "GET": { 
            "tweet": "statuses/show"
        },
        "POST": {

        }
    }
}


USAGE = """
usage: 
    python twitter_oauth_client.py -id <credentials> -tweets <tweets> [-col <n>]

    -id:        identity file csv 
    -tweets:    csv file of tweet data
    -col:       column  of the tweet ids 
"""

TIMEOUT = 15 * 60
REQUESTS_PER_TO = 150

class OAUTHCredentials:
    def __init__(self, name, client_key, client_secret, access_token, access_token_secret, num_requests = 0):
        self.name = name
        self.client_key = client_key
        self.client_secret = client_secret
        self.access_token = access_token
        self.access_token_secret = access_token_secret
        self.num_requests = num_requests
    
    def create_service(self):
        serv = rauth.OAuth1Service(
            name=self.name,
            consumer_key=self.client_key,
            consumer_secret=self.client_secret,
            request_token_url=TWITTER["v1.1"]["request_token"],
            access_token_url=TWITTER["v1.1"]["access_token"],
            authorize_url=TWITTER["v1.1"]["authorize"],
            base_url=TWITTER["v1.1"]["root"]
        )
        request_token, request_token_secret = serv.get_request_token()
        print request_token, request_token_secret
        session = rauth.OAuth1Session(
            self.client_key, 
            self.client_secret, 
            access_token = self.access_token, 
            access_token_secret = self.access_token_secret)
        self.oaservice = serv
        self.oasession = session
        self.timestamp = time.time() 
        return serv
    def get(self, resource, rid):
        self.request()
        resource_path = TWITTER["v1.1"]["GET"][resource]+"/"+rid+".json" 
        resource_url = TWITTER["v1.1"]["root"] + resource_path
        result = self.oasession.get(resource_url, params={"format": "json"})
        if result.ok:
            return result.json()
        return False

    def request(self):
        """
        handle twitter throttling...
        """
        if self.num_requests >= REQUESTS_PER_TO:
            time_left = TIMEOUT - (time.time() - self.timestamp)
            while time_left > 0: 
                time.sleep(time_left + 1)
                time_left = TIMEOUT - (time.time() - self.timestamp)
            self.timestamp = time.time()
            self.num_requests = 0
        self.num_requests += 1
        

def load_credential_file(fname):
    """

    """
    handle = open(fname, 'rb')
    reader = csv.reader(handle, delimiter=",")
    cred = reader.next();
    handle.close() 
    
    return OAUTHCredentials(cred[0], cred[1], cred[2], cred[3], cred[4]) 


def load_tweets_file(fname, col=0): 
    """

    """
    handle = open(fname, 'rb')
    reader = csv.reader(handle, delimiter=",")
    tweets = [] 
    for row in reader:
        tweets.append(row[col])
    handle.close()
    return tweets

def test(oauth, tid="126415614616154112"):
    r = oauth.get("tweet", tid)
    return r

def scrape(oauth, tweets, out_file):
    handle = open(out_file, "w+")
    for tweet_id in tweets: 
        result = oauth.get("tweet", tweet_id)
        blob = ""
        if result is not None:
            blob = json.dumps(result).replace("\n", "\\n") 
        handle.write("%s\n" % blob)
    handle.close()

def parse_args(arr, results):
    n = len(arr) 
    
    if n % 2 != 0:  
        raise ValueError("invalid number of arguments");    

    for i in xrange(n / 2):
        flag = arr[2 * i][1:]
        param = arr[2 * i + 1]
        results[flag] = param
    return results

if __name__ == "__main__":
    print USAGE
    defaults = {
        "id": None,
        "tweets": None, 
        "col": 0,
        "out-file": "out.dat"
    }
    args = parse_args (sys.argv[1:], defaults)

    if args["id"] is None:
        raise ValueError("Need OAUTH information to connect to twitter")
    credentials = load_credential_file(args["id"])
    credentials.create_service()
    if "test" in args:
        test(credentials)
    else:
        if args["tweets"] is None: 
            raise ValueError("need at least OAUTH info and tweets to fetch data")
        tweets = load_tweets_file(args["tweets"], int(args["col"]))
        scrape(credentials, tweets, args["out-file"])
