# fetch and preprocess datasets
#
#

all: fetch unzip

fetch: 
	wget -O san_twitter_sentiment.zip  http://www.sananalytics.com/lab/twitter-sentiment/sanders-twitter-0.2.zip
	
unzip: 
	unzip san_twitter_sentiment.zip
	mv ./sanders-twitter-* ./sanders-twitter
	cp ./sanders-twitter/corpus.csv ./tweet-sentiment.csv

scrape_tweets: tweet-sentiment.csv
	python ./twitter_oauth_client.py -id ./id.cred -tweets ./tweet-sentiment.csv -col 3 -out-file twitter_info.dat

.PHONY clean:
	rm -rf *.zip

.PHONY strip-data:
	rm -rf ./sanders-twitter*

