# from q2 import *
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from time import time


x_train = np.load('x_train.npy')
y_train = np.load('y_train.npy')
x_test = np.load('x_test.npy')
y_test = np.load('y_test.npy')
x_train=2*(0.5 - x_train/255)
x_test=2*(0.5 - x_test/255)


from sklearn.neural_network import MLPClassifier
for layers in [[512,256],[512,256,128],[512,256,128,64],[512]]:
    start = time()
    
    clf = MLPClassifier(
        hidden_layer_sizes=layers,
        activation='relu',
        solver='sgd',
        alpha=0,
        batch_size=32,
        learning_rate='invscaling',
        max_iter=1000
    )

    clf.fit(x_train, y_train)
    print("Training Time taken:",time()-start)
    y_pred = clf.predict(x_train)
    print("Train Accuracy:",accuracy_score(y_train, y_pred))
    print(classification_report(y_train, y_pred,zero_division=0))
    y_pred = clf.predict(x_test)
    print("Test Accuracy:",accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred,zero_division=0))

