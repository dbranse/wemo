import json
import tweepy
from langdetect import detect
import got3 as got
from tweepy import OAuthHandler, Stream
from tweepy.streaming import StreamListener
from datetime import timedelta, date
from multiprocessing import Pool


 
auth = OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_secret)


hashtags = {
    # 'happy':set(["happy", "excited", "joy", "elated", "proud", "amused", "bliss", "optimism"]),
    # 'sad'  :set(["sad", "depressing", "lonely", "unhappy", "sorrow", "suffer", "suffering", "anguish", "disappointed", "regret"]),
    'angry':set(["angry", "anger", "irritating", "irritated", "annoyed", "annoying", "frustrate", "frustrated", "furious", "hate", "disgusted", "disgust", "fuming"]),
    'other':set(["the", "and", "a", "is", "or", "I"])
}

start_date = date(year = 2018, month = 1, day = 1)
end_date = date.today()
max_tweets = 25
 
# api = tweepy.API(auth)

def countHashtags(t):
    return t.count('#') >= 5

def parseTweet(t, e):
    if countHashtags(t):
        return 'nis'
    p = ''
    words = t.split()
    i = 0
    while i < len(words):
        if words[i] == '#' and i+1 < len(words):
            if words[i+1].lower() != e and '.com' not in words[i+1] and 'http' not in words[i+1] and 'www' not in words[i+1]:
                p += '#'+words[i+1].lower() + ' '
            i += 2
            continue
        elif 'http' in words[i] or 'www' in words[i] or '.com' in words[i]:
            i += 1
            continue
        p += words[i].lower() + ' '
        i += 1
    return p

def saveTweet(e, t, s):
    with open(e+'.txt', 'a') as f:
        p = parseTweet(t, e)
        if p is not 'nis':
            if p not in s:
                s.add(p)
                f.write(p + '\n')

def queryEmotion(e):
    d = start_date
    delta = timedelta(days=1)
    tweets = set()
    tagged = '#' if e not in hashtags['other'] else ''
    while d <= end_date:
        since = d.strftime("%Y-%m-%d")
        d += delta
        until = d.strftime("%Y-%m-%d")
        try:
            tweetCriteria = got.manager.TweetCriteria().setQuerySearch(tagged+e).setSince(since).setUntil(until).setMaxTweets(25)
            for tweet in got.manager.TweetManager.getTweets(tweetCriteria):
                text = tweet.text
                if detect(text) == 'en':
                    saveTweet(e, text, tweets)
        except Exception as e:
            print('woops, error: ' + str(e))
    print(len(tweets))

def main():
    for emotion, tags in hashtags.items():
        print('emotion: '+emotion)
        pool = Pool(processes=len(tags))
        pool.map(queryEmotion, tags)
        # for tag in tags:
        #     print('tag: '+tag)
        #     queryEmotion(tag)


# class TweetListener(StreamListener):
 
#     def on_data(self, data):
#         j = json.loads(data)
#         if 'media' not in j['entities']:
#             print(j['text'])
#         return True
 
#     def on_error(self, status):
#         print(status)
#         return True
 
# twitter_stream = Stream(auth, TweetListener())
# twitter_stream.filter(languages=["en"], track=['#happy'])
# twitter_stream.filter(languages=["en"], track=['#sad'])
# twitter_stream.filter(languages=["en"], track=['#angry'])
# twitter_stream.filter(languages=["en"], track=['#angerey'])
if __name__ == '__main__':
    main()
