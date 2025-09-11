import tweepy
import time

consumer_key = 'CONSUMER_KEY'
consumer_secret = 'CONSUMER_SECRET'
access_token = 'ACCESS_TOKEN'
access_token_secret = 'ACCESS_TOKEN_SECRET'

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth)

query = '#example'

def twitter_bot():
    for tweet in tweepy.Cursor(api.search_tweets, q=query, lang='en').items(10):
        try:
            if not tweet.favorited:
                tweet.favorite()
            if not tweet.retweeted:
                tweet.retweet()
            if not tweet.user.following:
                tweet.user.follow()
            time.sleep(10)
        except tweepy.TweepError as e:
            print(e.reason)
        except StopIteration:
            break

if __name__ == "__main__":
    twitter_bot()

