import string,seaborn as sns
import numpy as np,pandas as pd,matplotlib.pyplot as plt
from collections import defaultdict
from sklearn.metrics import confusion_matrix  
from wordcloud import WordCloud
import string,re
import numpy as np,pandas as pd,matplotlib.pyplot as plt
from collections import defaultdict
from sklearn.metrics import confusion_matrix  
from wordcloud import WordCloud

import random
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string

test_data = pd.read_csv('Corona_validation.csv')
train_data = pd.read_csv('Corona_train.csv')
def draw_matrix(confusion_matrix):
    class_labels = ["Positive", "Negative", "Neutral"]

    plt.figure(figsize=(6,6))
    sns.set(font_scale=1)
    sns.heatmap(confusion_matrix, annot=True, cmap="Blues", fmt='g',
                xticklabels=class_labels, yticklabels=class_labels,
                linewidths=.5, square=True, cbar=False)

    plt.xlabel('Predicted',color='red')
    plt.ylabel('Actual',color='red')
    plt.title('Confusion Matrix',color='darkblue')

    plt.show()

def unigram(text:str):
    text = text.split()
    return text

def bigram(text:str):
    text = text.split()
    n=len(text)
    for i in range(n-1):
        text.append((text[i]+" "+text[i+1]))
    return text


def trigram(text:str):
    text = text.split()
    n=len(text)
    for i in range(n-1):
        text.append((text[i]+" "+text[i+1]))
    for i in range(n-2):
        text.append((text[i]+" "+text[i+1]+" "+text[i+2]))
    return text



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
