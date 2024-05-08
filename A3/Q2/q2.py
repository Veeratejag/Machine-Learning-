import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.preprocessing import OneHotEncoder
from time import time
from collections import Counter

x_train = np.load('x_train.npy')
y_train = np.load('y_train.npy')
x_test = np.load('x_test.npy')
y_test = np.load('y_test.npy')
x_train=2*(0.5 - x_train/255)
x_test=2*(0.5 - x_test/255)



def sigmoid(X):
    return 1/(1+np.exp(-X))
def sigmoid_derivative(X):
    return sigmoid(X)*(1-sigmoid(X))
def relu(X):
    return np.maximum(0,X)
def relu_derivative(X):
    return X>0

def softmax(X):
    return np.exp(X)/sum(np.exp(X))
def one_hot(Y):
    encoder = OneHotEncoder(sparse_output=False)
    return encoder.fit_transform(np.array(Y).reshape(-1, 1))


y_train_one_hot=one_hot(y_train)



class NN:
    def __init__(self,input_f,output_f,hidden_layers,alpha=0.01,batch_size=32,activation=sigmoid,activation_prime = sigmoid_derivative):
        np.random.seed(99)
        self.layers=[input_f]+hidden_layers+[output_f]
        self.alpha=alpha
        self.batch_size=batch_size
        self.activation=activation
        self.activation_prime=activation_prime
        self.theta={}
        for i in range(len(self.layers)-1):
            self.theta[f"W{i+1}"]=np.random.rand(self.layers[i],self.layers[i+1])-0.5
            self.theta[f"b{i+1}"]=np.random.rand(self.layers[i+1],1)-0.5

    def forward_prop(self,X):
        forward={}
        forward["A0"]=X
        Acurr = forward["A0"]
        for i in range(len(self.layers)-1):
            W,b = self.theta[f"W{i+1}"],self.theta[f"b{i+1}"]
            Zcurr = W.T.dot(forward[f"A{i}"])+b
            Acurr = self.activation(Zcurr) if i!=len(self.layers)-2 else softmax(Zcurr)
            
            forward[f"Z{i+1}"]=Zcurr
            forward[f"A{i+1}"]=Acurr
        return (forward,Acurr)
    
    def predict(self,X):
        return self.forward_prop(X)[1].argmax(axis=1)+1
    
    def loss(self,y,A):
        return np.mean(-np.sum(y*np.log(A),axis=0))
    
    def back_prop(self,X,y,eta=0.01):
        (forward,_)=self.forward_prop(X)
        weights = self.theta
        backward={}
        m = y.shape[0]
        num_layers = len(self.layers)-1
        backward[f'dZ{num_layers}'] = forward[f'A{num_layers}'] - y
        i = num_layers - 1
        while(i > 0):
            zi = backward[f'dZ{i + 1}']
            wi = weights[f'W{i + 1}']
            g_prime = self.activation_prime(forward[f'Z{i}'])
            backward[f'dZ{i}'] = (np.dot(wi,zi))*g_prime
            i -= 1
        for i in range(1, num_layers + 1):
            dw = np.dot(backward[f"dZ{i}"],forward[f"A{i-1}"].T).T
            db = np.sum(backward[f'dZ{i}'])
            weights[f'W{i}'] = weights[f'W{i}'] - (eta/m)*dw
            weights[f'b{i}'] = weights[f'b{i}'] - (eta/m)*db
        self.theta=weights
        return weights,backward

