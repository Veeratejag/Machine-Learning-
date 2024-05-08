import numpy as np
from cvxopt import matrix, solvers
from q2_1_pre import x_train_tot,y_train_tot,x_test_tot,y_test_tot
import matplotlib.pyplot as plt

"""
    entry no. 2021CS10107 ==> 7 %6=1 ==> classes 1 and 2

"""


from time import time
x_train = np.concatenate((x_train_tot[1],x_train_tot[2]))
y_train = np.concatenate((y_train_tot[1],y_train_tot[2]))
x_test = np.concatenate((x_test_tot[1],x_test_tot[2]))
y_test = np.concatenate((y_test_tot[1],y_test_tot[2]))

def gaussian(X1,X2, gamma):
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


def train_CVOPT(X,Y,C,gamma):
    n_samples, n_features = X.shape
    P = find_P(X, Y, gamma)
    q = -np.ones((n_samples, 1))
    G = np.vstack((-np.eye(n_samples), np.eye(n_samples)))
    h = np.vstack((np.zeros((n_samples, 1)), C * np.ones((n_samples, 1))))
    A = Y.reshape(1, -1)
    b = np.zeros((1, 1))
    P = matrix(P)
    q = matrix(q)
    G = matrix(G)
    h = matrix(h)
    A = matrix(A,tc='d')
    b = matrix(b,tc='d')
    sol = solvers.qp(P, q, G, h, A, b,options={'show_progress': False})
    alphas = np.array(sol['x'])
    return alphas

def find_w_b(X,Y,alphas,C=1.0):
    n_samples, n_features = X.shape
    w = np.sum(alphas * Y.reshape(-1, 1) * X, axis=0)
    sv = (C>alphas >1e-5).flatten()
    b = np.mean(Y[sv] - np.dot(X[sv], w))
    return w, b
def no_of_sv(alpha,C=1.0):
    SVs = np.where((alpha > 1e-5) & (alpha < C))[0]
    return len(SVs)

def kernel(x_i,x_j):
    return np.exp(-0.001*(np.linalg.norm(x_i-x_j)**2))
def wtx(alpha, Y, X, x_j):
    return np.sum(alpha * Y * kernel(X, x_j))
def find_b_gauss(alpha,X,Y,C=1.0):
    b = 0.0
    indices = np.where((alpha > 1e-5) & (alpha < C ))[0]
    j=0
    for idx in indices:
        b += Y[idx] - wtx(alpha,Y,X,X[idx])
    b /= len(indices)
    return b

def linear_predict(X,w,b):
    return np.sign(np.dot(X, w) + b)

def gauss_predict(X,alphas,X_train,Y_train,b,gamma):
    K = Kernel(X_train,X, gamma)
    return np.sign(np.dot((alphas * Y_train).T, K) + b)

def accuracy_gauss(X,Y,alphas,X_train,Y_train,b,gamma):
    pred = gauss_predict(X,alphas,X_train,Y_train,b,gamma)
    return np.mean(pred==Y)


def accuracy_linear(X,Y,w,b):
    pred = linear_predict(X,w,b)
    return np.mean(pred==Y)

def plot_top6(alpha,x_train,y_train):
    svis = np.where((alpha > 1e-5) & (alpha < C))[0]
    tsvi = svis[np.argsort(np.abs(alpha[svis]))[-6:][::-1]]

    plt.figure(figsize=(10, 6))

    for i, idx in enumerate(tsvi):
        sv = x_train[idx]
        sv_image = sv.reshape((16, 16, 3))

        plt.subplot(2, 3, i + 1)  
        plt.imshow(sv_image)
        plt.title(f"Support Vector {i+1}")

    plt.tight_layout()
    plt.show()


def plot_w(alpha,x_train,y_train):
    w,b = find_w_b(x_train,y_train,alpha)
    w = w.reshape(16,16,3)
    plt.imshow(w)
    plt.show()


print("Using CVXOPT for training SVMS ")

print("Linear SVM")
C = 1
start = time()
alphas = train_CVOPT(x_train,y_train,C,0)
print("Time taken for training: ",time()-start)
print("No of SVs: ",no_of_sv(alphas,C))
w_cvx_lin,b_cvx_lin = find_w_b(x_train,y_train,alphas)

print("Training Accuracy: ",accuracy_linear(x_train,y_train,w_cvx_lin,b_cvx_lin))
print("Test Accuracy: ",accuracy_linear(x_test,y_test,w_cvx_lin,b_cvx_lin))

plot_top6(alphas,x_train,y_train)
plot_w(alphas,x_train,y_train)



print("Gaussian SVM")
C = 1
gamma = 0.001
start = time()
alphas = train_CVOPT(x_train,y_train,C,gamma)
print("No of SVs: ",no_of_sv(alphas,C))
b= find_b_gauss(alphas,x_train,y_train,gamma)
print("Time taken for training: ",time()-start)
print("Training Accuracy: ",accuracy_gauss(x_train,y_train,alphas,x_train,y_train,b,gamma))
print("Validation Accuracy: ",accuracy_gauss(x_test,y_test,alphas,x_train,y_train,b,gamma))
plot_top6(alphas,x_train,y_train)


print("----------------------------------------------")
print("Using LIBSVM for training SVMS ")
from libsvm.svmutil import svm_problem,svm_parameter,svm_train,svm_predict

def sv_libsvm(x_train,y_train,kernel,gamma):
    prob = svm_problem(y_train,x_train)
    param = svm_parameter('-t {} -c 1'.format(kernel))
    if gamma:
        param = svm_parameter('-t {} -c 1 -g {}'.format(kernel,gamma))
    model = svm_train(prob,param)
    return model
def accuracy_libsvm(model,x_train,y_train,x_test,y_test):
    p_label,p_acc_train,p_val = svm_predict(y_train,x_train,model)
    print("Training Accuracy: ",p_acc_train[0])
    p_label,p_acc_validation,p_val = svm_predict(y_test,x_test,model)
    print("Validation Accuracy: ",p_acc_validation[0])
    return p_acc_train,p_acc_validation[0]


print("Linear SVM")
start = time()
model = sv_libsvm(x_train,y_train,2,0)
print("Time taken for training: ",time()-start)
print("Number of Svs : ",model.get_nr_sv())
alpha = np.array(model.get_sv_coef()).reshape(-1,1)
W_lib_lin,b_lib_lin = find_w_b(x_train,y_train,alpha)

accuracy_libsvm(model,x_train,y_train,x_test,y_test)


print("Gaussian SVM")
start = time()
model = sv_libsvm(x_train,y_train,2,0.001)
print("Time taken for training: ",time()-start)
print("Number of Svs : ",model.get_nr_sv())
accuracy_libsvm(model,x_train,y_train,x_test,y_test)

