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

def stem(data,dest):
    stemmer = PorterStemmer()
    stop_words = set(stopwords.words('english'))
    X = [word_tokenize(re.sub('[{}]'.format(string.punctuation), ' ', tweet)) for tweet in data['Tweet']]
    with open(dest,"w", encoding='utf-8') as f:
        f.write("ID,Sentiment,Tweet\n")
    for i in range(len(X)):
        X[i] = " ".join([stemmer.stem(word) for word in X[i] if word.lower() not in stop_words])
        with open(dest,"a", encoding='utf-8') as f:
            f.write(str(data['ID'][i])+","+data['Sentiment'][i]+',"'+X[i]+'"\n')
    return X
path = 'Domain_Adaptation/Twitter_train_{}.csv'
destin = 'domain/Twitter_train_{}.csv'
for x in [1,2,5,10,25,50,100]:
    data = pd.read_csv(path.format(x))

    dest = destin.format(x)
    stemmed_train = stem(data,dest)

data = pd.read_csv('Domain_Adaptation/Twitter_validation.csv')
dest = 'domain/Twitter_validation.csv'
stem(data,dest)

################## with source domain #####################
train_data = pd.read_csv('Corona_train_stemmed.csv')
path = 'Domain_Adaptation/Twitter_train_{}.csv'
validation_data = pd.read_csv('domain/Twitter_validation.csv')
accuracies_with_source = []
for x in [1,2,5,10,25,50,100]:
    domain_data = pd.read_csv(path.format(x))
    X_domain,Y_domain = domain_data['Tweet'],domain_data['Sentiment']
    X_train,Y_train = train_data['CoronaTweet'],train_data['Sentiment']
    X_trian = np.concatenate((X_train,X_domain))
    Y_train = np.concatenate((Y_train,Y_domain))
    vocab,probability = train(X_train,Y_train,unigram)
    pred,validation = accuracy(validation_data['Tweet'],validation_data['Sentiment'],probability,unigram)
    accuracies_with_source.append(validation)
    print("Validation Accuracy after using {} target training  data: ".format(x),validation)


################## without source domain #####################
path = 'Domain_Adaptation/Twitter_train_{}.csv'
validation_data = pd.read_csv('domain/Twitter_validation.csv')
accuracies_without_source=[]
for x in [1,2,5,10,25,50,100]:
    domain_data = pd.read_csv(path.format(x))
    X_train,Y_train = domain_data['Tweet'],domain_data['Sentiment']
    vocab,probability = train(X_train,Y_train,unigram)
    pred,validation = accuracy(validation_data['Tweet'],validation_data['Sentiment'],probability,unigram)
    accuracies_without_source.append(validation)
    print("Validation Accuracy after using {} target training  data: ".format(x),validation)

import matplotlib.pyplot as plt

plt.plot([1,2,5,10,25,50,100],accuracies_with_source,label='With Source Domain')
plt.plot([1,2,5,10,25,50,100],accuracies_without_source,label='Without Source Domain')
plt.xlabel('Number of Target Domain Training Data')
plt.ylabel('Validation Accuracy')
plt.legend()
plt.show()


