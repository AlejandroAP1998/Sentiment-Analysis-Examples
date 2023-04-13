import matplotlib as plt
import numpy as np
import nltk
from nltk.corpus import twitter_samples
from nltk.tag import pos_tag # It helps determining context of words
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
import re
import string

# Print tweets in different json files to string format
positive_tweets = twitter_samples.strings('positive_tweets.json')
negative_tweets = twitter_samples.strings('negative_tweets.json')
unlabeled_tweets = twitter_samples.strings('tweets.20150430-223406.json')


# Now we're going to remove the noise in the sentences. By default, we consider stopWords in English but you can enter custom ones
def remove_noise(tokens, stopWords = stopwords.words('english')):
    cleaned_tokens = []
    for token, tag in pos_tag(tokens):
        norm_token = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+#]|[!*\(\),]|'\
                       '(?:%[0-9a-fA-F][0-9a-fA-F]))+','', token) # We remove hyperlinks
        norm_token = re.sub("(@[A-Za-z0-9_]+)","", token) # We also remove mentions to other users
        
        # We normalize the token by using a lemmatizer, also converting it to lowercase
        if tag.startswith('NN'):
            pos = 'n'
        elif tag.startswith('VB'):
            pos = 'v'
        else:
            pos = 'a'
        
        lemmatizer = WordNetLemmatizer()
        norm_token = lemmatizer.lemmatize(norm_token,pos).lower
        
        # If the normalized token is not empty and is nor a punctuation sign neither a stop word, we include it as a cleaned token
        if len(norm_token)>0 and token not in string.punctuation and token not in stopWords:
            cleaned_tokens.append(norm_token)
        
    return cleaned_tokens
     




