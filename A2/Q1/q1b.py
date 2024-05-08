import string
import numpy as np,pandas as pd,matplotlib.pyplot as plt
from collections import defaultdict
from sklearn.metrics import confusion_matrix  
from wordcloud import WordCloud
from q1_util import train_data,test_data,draw_matrix

import random



cls = ['Positive','Negative','Neutral']
def random_assign(tweets,labels):
    pred = [random.choice(cls) for tweet in tweets]
    return [confusion_matrix(labels,pred,labels=cls),np.mean(pred==labels)]

conf,acc = random_assign(train_data['CoronaTweet'],train_data['Sentiment'])
print("Training  Accuracy of randomly guessing a class is {} ".format(acc))
draw_matrix(conf)


conf,acc = random_assign(test_data['CoronaTweet'],test_data['Sentiment'])
print("Validation Accuracy of randomly guessing a class is {} ".format(acc))
draw_matrix(conf)


def pred_pos(tweets,labels):
    pred = ['Positive']*len(tweets)
    return [confusion_matrix(labels,pred,labels=cls),np.mean(pred==labels)]

conf,acc = pred_pos(train_data['CoronaTweet'],train_data['Sentiment'])
print("Training  Accuracy of guessing a Positive for all Tweets is {} ".format(acc))
draw_matrix(conf)

conf,acc=pred_pos(test_data['CoronaTweet'],test_data['Sentiment'])
print("Accuracy of  guessing a Positive for all Tweets is {} ".format(acc))
draw_matrix(conf)

