import nltk
from nltk.corpus import twitter_samples
from nltk.tag import pos_tag # It helps determining context of words
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk import classify
from nltk import NaiveBayesClassifier
from nltk.tokenize import word_tokenize
import re
import string
from random import shuffle
from random import randint

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
        norm_token = lemmatizer.lemmatize(norm_token,pos).lower()
        
        # If the normalized token is not empty and is nor a punctuation sign neither a stop word, we include it as a cleaned token
        if len(norm_token)>0 and token not in string.punctuation and token not in stopWords:
            cleaned_tokens.append(norm_token)
        
    return cleaned_tokens

# We tokenize and clean all positive and negative tweets
pos_tweet_tokens = twitter_samples.tokenized('positive_tweets.json')
pos_cleaned_tokens = []
for tokens in pos_tweet_tokens:
    pos_cleaned_tokens.append(remove_noise(tokens))


neg_tweet_tokens = twitter_samples.tokenized('negative_tweets.json')
neg_cleaned_tokens = []
for tokens in neg_tweet_tokens:
    neg_cleaned_tokens.append(remove_noise(tokens))

# Now we're going to see what the most frequent words are. First, we create a generator function to get all the words in the different tweets

def get_all_words(token_list):
    for tokens in token_list:
        for token in tokens:
            yield token



 # As we're going to use Naive Bayes Classifier, we need to express the data in a suitable format to train the model.
 # Naive Bayes Classifier asks for a dictionary with keys as tokens and True as values.
def prepare_tweets(token_list):
    for tokens in token_list:
        yield dict([token, True] for token in tokens)

prepared_pos = prepare_tweets(pos_cleaned_tokens)
prepared_neg = prepare_tweets(neg_cleaned_tokens)


# We label each dataset and concatenate them to build the final dataset
pos_dataset = [(dic, "Positive") for dic in prepared_pos]
neg_dataset = [(dic, "Negative") for dic in prepared_neg]
dataset = pos_dataset + neg_dataset

# We shuffle the dataset to avoid bias and split it into train (70%) and validation (30%) sets (it is 10000 tweets long)
shuffle(dataset)
train_set = dataset[:7000]
test_set = dataset[7000:]

# We define and train the classifier using Naive Bayes classifier, and then we check the accuracy of the model
classifier = NaiveBayesClassifier.train(train_set)
#print("Accuracy is:", classify.accuracy(classifier, test_set))
#print(classifier.show_most_informative_features(10))  # This statement prints the top 10 most relevant tokens to the model

# We can try the model with a custom tweet
def analyzeCustomTweet(custom_tweet):
    prepared_custom_tokens = remove_noise(word_tokenize(custom_tweet))
    print("Custom tweet: "+custom_tweet)
    print("Random tweet score: "+classifier.classify(dict([token, True] for token in prepared_custom_tokens)))

custom_tweet = "I love being in this place"
analyzeCustomTweet(custom_tweet)

# We also define a function to analyze a random tweet from the unlabeled tweets dataset
def analyzeRandomTweet():
    x = randint(0,len(unlabeled_tweets)-1)
    random_tweet = unlabeled_tweets[x]
    prepared_random_tokens = remove_noise(word_tokenize(random_tweet))
    print("Tweet chosen at random: "+random_tweet)
    print("Random tweet score: "+classifier.classify(dict([token, True] for token in prepared_random_tokens)))
 
analyzeRandomTweet()



