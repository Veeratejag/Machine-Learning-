from q1_util import *


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


train_data = pd.read_csv('Corona_train_stemmed.csv')
test_data = pd.read_csv('Corona_validation_stemmed.csv')

vocab,probability = train(train_data['CoronaTweet'],train_data['Sentiment'],bigram)
pred,train_accuracy = accuracy(train_data['CoronaTweet'],train_data['Sentiment'],probability,bigram)

print("Train Accuracy after using bigrams: ",train_accuracy)

pred,test_accuracy = accuracy(test_data['CoronaTweet'],test_data['Sentiment'],probability,bigram)

print("Validation Accuracy after using bigrams: ",test_accuracy)


vocab,probability = train(train_data['CoronaTweet'],train_data['Sentiment'],trigram)
pred,train_accuracy = accuracy(train_data['CoronaTweet'],train_data['Sentiment'],probability,trigram)
print("Train Accuracy after using trigrams: ",train_accuracy)

pred,test_accuracy = accuracy(test_data['CoronaTweet'],test_data['Sentiment'],probability,trigram)
print("Validation Accuracy after using trigrams: ",test_accuracy)

