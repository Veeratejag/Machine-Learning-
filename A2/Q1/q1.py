import string
import numpy as np,pandas as pd,matplotlib.pyplot as plt
import re
from wordcloud import WordCloud, STOPWORDS
from collections import defaultdict
from sklearn.metrics import confusion_matrix  
import nltk
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
classes = {'Positive':0,'Negative':1,'Neutral':2}
P = {0:'Positive',1:'Negative',2:'Neutral'}

def unigram(text:str):
    return text.split()

def train(X,Y,split_fn):
    d=defaultdict(lambda:np.ones(3))
    probability = {'Positive':0,'Negative':0,'Neutral':0}
    for i in range(len(X)):
        tweet = split_fn(X[i])
        # print(tweet[2])
        for word in tweet:
            d[word][classes[Y[i]]]+=1
        probability[Y[i]]+=1
    total = sum(probability.values())

    for i in probability:
        probability[i]/=total
    den = sum(d.values())
    return d,probability

def theta(l,d,den):
    num = d[l]
    return num/den

def predict(tweet,probability,d,split_fn):
    den = sum(d.values())
    prob = np.log([probability['Positive'],probability['Negative'],probability['Neutral']])
    for word in split_fn(tweet):
        prob += np.log(theta(word,d,den))
    if np.max(prob)==prob[0]:
        return "Positive"
    elif np.max(prob)==prob[1]:
        return "Negative"
    else:
        return "Neutral"

def accuracy(X_test,Y_test,d,probability,split_fn):
    pred = [];ans=0
    for i in range(len(X_test)):
        y  = predict(X_test[i],probability,d,split_fn)
        pred.append(y)
        ans+=(y==Y_test[i])
    return pred,ans/len(Y_test)

test_data = pd.read_csv('Corona_validation.csv')
# confusion =confusion_matrix(pred,test_data['Sentiment'],labels=["Positive","Negative","Neutral"])


def random_assign(tweets,labels):
    pred = [np.random.choice(['Positive','Negative','Neutral']) for tweet in tweets]
    return [confusion_matrix(labels,pred),np.mean(pred==labels)]

def pred_pos(tweets,labels):
    pred = ['Positive']*len(tweets)
    return [confusion_matrix(labels,pred),np.mean(pred==labels)]

def stem(data,dest):
    stemmer = PorterStemmer()
    stop_words = set(stopwords.words('english'))
    X = [word_tokenize(re.sub('[{}]'.format(string.punctuation), ' ', tweet)) for tweet in data['CoronaTweet']]
    with open(dest,"w", encoding='utf-8') as f:
        f.write("ID,Sentiment,CoronaTweet\n")
    for i in range(len(X)):
        X[i] = " ".join([stemmer.stem(word) for word in X[i] if word.lower() not in stop_words])
        with open(dest,"a", encoding='utf-8') as f:
            f.write(str(data['ID'][i])+","+data['Sentiment'][i]+',"'+X[i]+'"\n')
    return X


train_data = pd.read_csv('Corona_train.csv')
test_data = pd.read_csv('Corona_validation.csv')
vocab,probability = train(train_data['CoronaTweet'],train_data['Sentiment'],unigram)
pred,train_accuracy = accuracy(train_data['CoronaTweet'],train_data['Sentiment'],vocab,probability,unigram)
confusion_matrix_train =  confusion_matrix(pred,train_data['Sentiment'],labels=["Positive","Negative","Neutral"])

pred,test_accuracy = accuracy(test_data['CoronaTweet'],test_data['Sentiment'],vocab,probability,unigram)
confusion_matrix_text =  confusion_matrix(pred,test_data['Sentiment'],labels=["Positive","Negative","Neutral"])

print("Train Accuracy: ",train_accuracy)
print("Confusion Matrix: ",confusion_matrix_train)
print("Test Accuracy: ",test_accuracy)
print("Confusion Matrix: ",confusion_matrix_text)

print("Random Assign: ",random_assign(test_data['CoronaTweet'],test_data['Sentiment']))
print("Predicting Positive: ",pred_pos(test_data['CoronaTweet'],test_data['Sentiment']))



################# part e #####################



def bigram(text:str):
    text = text.split()
    # bigrams = []
    n=len(text)
    for i in range(n-1):
        text.append((text[i]+" "+text[i+1]))
    return text

def train_bigram(X,Y):
    return train(X,Y,bigram)

def predict_bigram(X,Y,d,den):
    return predict(X,Y,d,den,bigram)
    
def trigram(text):
    text = text.split()
    n=len(text)
    for i in range(n-1):
        text.append((text[i]+" "+text[i+1]))
    for i in range(n-2):
        text.append(text[i]+" "+text[i+1]+" "+text[i+2])
    return text

########### part f ##############

def train_domain(X,Y,X_domain,Y_domain,source:bool,split_fn):
    if source:
        return train(X+X_domain,Y+Y_domain,split_fn)
    else:
        return train(X_domain,Y_domain,split_fn)

def predict_domain(X,prob,d,split_fn):
    return predict(X,prob,d,split_fn)
