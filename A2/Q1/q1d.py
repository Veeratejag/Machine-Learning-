import numpy as np,pandas as pd,matplotlib.pyplot as plt
from wordcloud import WordCloud
from collections import defaultdict

from q1_util import stem,draw_matrix,stem,unigram

classes = {'Positive':0,'Negative':1,'Neutral':2}
P = {0:'Positive',1:'Negative',2:'Neutral'}
d=defaultdict(lambda:np.ones(3))
probability = {'Positive':0,'Negative':0,'Neutral':0}


def train(X,Y,split_fn):
    
    for i in range(len(X)):
        tweet = split_fn(X[i])
        # print(tweet[2])
        for word in tweet:
            d[word][classes[Y[i]]]+=1
        probability[Y[i]]+=1
    total = sum(probability.values())

    for i in probability:
        probability[i]/=total
    # den = sum(d.values())
    return d,probability

def theta(l,den):
    num = d[l]
    return num/den

def predict(tweet,probability,den,split_fn):
    
    prob = np.log([probability['Positive'],probability['Negative'],probability['Neutral']])
    for word in split_fn(tweet):
        prob += np.log(theta(word,den))
    if np.max(prob)==prob[0]:
        return "Positive"
    elif np.max(prob)==prob[1]:
        return "Negative"
    else:
        return "Neutral"
    
def accuracy(X_test,Y_test,probability,split_fn):
    pred = [];ans=0
    den = sum(d.values())
    for i in range(len(X_test)):
        y  = predict(X_test[i],probability,den,split_fn)
        pred.append(y)
        ans+=(y==Y_test[i])
    return pred,ans/len(Y_test)


data = pd.read_csv('Corona_train.csv')
dest = 'Corona_train_stemmed.csv'
stemmed_train = stem(data,dest)

data = pd.read_csv('Corona_validation.csv')
dest = 'Corona_validation_stemmed.csv'
stemmed_validation = stem(data,dest)

train_data = pd.read_csv('Corona_train_stemmed.csv')
validation_data = pd.read_csv('Corona_validation_stemmed.csv')

X_train,Y_train = train_data['CoronaTweet'],train_data['Sentiment']
X_validation,Y_validation = validation_data['CoronaTweet'],validation_data['Sentiment']

vocab,probability = train(X_train,Y_train,unigram)
pred,acc = accuracy(X_train,Y_train,probability,unigram)

print("Train Accuracy after stemming : ",acc)

pred,acc = accuracy(X_validation,Y_validation,probability,unigram)
print("Validation Accuracy after stemming : ",acc)


pos = defaultdict(lambda:0)
neg = defaultdict(lambda:0)
neu = defaultdict(lambda:0)
for k in vocab:
    if vocab[k][0]>0:pos[k]+=int(vocab[k][0])
    if vocab[k][1]>0:neg[k]+=int(vocab[k][1])
    if vocab[k][2]>0:neu[k]+=int(vocab[k][2]) 

    
class1 = WordCloud(
    background_color='white',
    max_words=2000
)
class1.generate_from_frequencies(pos)
class2 = WordCloud(
    background_color='white',
    max_words=2000
)
class2.generate_from_frequencies(neg)
class3  = WordCloud(
    background_color='white',
    max_words=2000
)
class3.generate_from_frequencies(neu)

for c,sent in zip([class1,class2,class3],["Positive","Negative","Neutral"]):
    plt.figure(figsize=(10,8))
    plt.imshow(c, interpolation='bilinear')
    plt.axis('off')
    plt.title("Word Cloud for {} stemmed data".format(sent))
    plt.show()


