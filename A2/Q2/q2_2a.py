import numpy as np
import cv2,sklearn
from libsvm.svmutil import svm_problem,svm_parameter,svm_train,svm_predict
from sklearn.metrics import accuracy_score,confusion_matrix as cm
import os,cvxopt
from cvxopt import matrix, solvers
# from q2_1a import find_P
kC2classes = [['1','2'],['1','3'],['1','4'],['1','5'],['1','6'],
              ['2','3'],['2','4'],['2','5'],['2','6'],['3','4'],
              ['3','5'],['3','6'],['4','5'],['4','6'],['5','6']]

from q2_1_pre import x_train_tot,y_train_tot,x_test_tot,y_test_tot

X_train = [];Y_train = []
X_test = [];Y_test = []
for i in range(len(x_train_tot)):
    X_train.append(np.array(x_train_tot[i]))
    Y_train.append(np.array(y_train_tot[i]))
for i in range(len(x_test_tot)):
    X_test.append(np.array(x_test_tot[i]))
    Y_test.append(np.array(y_test_tot[i]))



def kernel_gauss(x_i,x_j):
    return np.exp(-0.001*(np.linalg.norm(x_i-x_j)**2))
def wtx(alpha, Y, X, x_j):
    return np.sum(alpha * Y * kernel_gauss(X, x_j))

def gaussian(X1,X2, gamma=0.001):
    prod1 = np.reshape(np.einsum('ij,ij->i', X1, X1), (X1.shape[0], 1))
    prod2 = np.reshape(np.einsum('ij,ij->i', X2, X2), (X2.shape[0], 1))
    prod = prod1 + prod2.T - 2 * np.matmul(X1, X2.T)
    return np.exp(-gamma * prod)

def Kernel(X1,X2,gamma):
    if gamma:
        return gaussian(X1,X2, gamma)
    else:
        return np.dot(X1, X2.T)
def find_P(X, Y, gamma):
    K = Kernel(X,X, gamma)
    P=np.outer(Y,Y)*K
    return P



from time import time
print("Multi class classification using one vs one approach with CVXOPT")
start = time()
Abs = []
m=len(x_train_tot[0])*2
q = -np.ones((m, 1))
q = matrix(q)
B = matrix(0.0, tc='d')
G = np.vstack((-np.eye(m), np.eye(m)))
G = matrix(G)
gamma = 0.001
C=1.0
h = np.zeros((2 * m, 1))
h[m:, :] = C   
h = matrix(h)


for X in kC2classes:
    start = time()
    # classes = X
    data = x_train_tot[int(X[0])-1]+x_train_tot[int(X[1])-1]
    labels = [-1]*len(x_train_tot[int(X[0])-1])+[1]*len(x_train_tot[int(X[1])-1])
    
    x_train = np.array(data)
    y_train = np.array(labels)
    m = x_train.shape[0]
    P = find_P(x_train,y_train, 0.001)
    P = matrix(P)
    A = matrix(y_train.reshape(1, -1), tc='d')
    sol = solvers.qp(P, q,G, h, A, B,options={'show_progress': False}) 
    alpha = np.reshape(np.array(sol['x']), (x_train.shape[0],1))
    indices = np.where((alpha > 1e-5) & (alpha < C ))[0]
    b = np.sum(labels[indices])
    b -= np.sum(alpha[indices] * y_train[indices] * gaussian(x_train[indices], x_train[indices[0]],0.001))
    b/=len(indices)
    Abs.append([alpha,b])
    print("done",X,len(Abs),time()-start)

print("Training time",time()-start)


def predict_pair(alpha,x_train,y_train,b,x_test):
    K = Kernel(x_train,x_test, gamma)
    return np.sign(np.dot((alpha * y_train).T, K) + b)

def predict_kC2(alphas,x_train,y_train,x_test,y_test):
    num_classes = 6
    y_pred = np.zeros((len(x_test), num_classes), dtype=int)
    gamma = 0.001
    for i, y in enumerate(alphas):
        alpha,b = y
        indices =np.where((alpha > 1e-5) & (alpha < C ))[0]
        y_pred = predict_pair(alpha[indices],x_train[indices],y_train[indices],b,x_test)
    y_pred = np.argmax(y_pred, axis=1) + 1
    # print("Predicted labels:", y_pred)
    
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)
    
    return y_pred








