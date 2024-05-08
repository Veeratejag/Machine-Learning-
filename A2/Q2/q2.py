import cvxopt,numpy as np,pandas as pd,matplotlib.pyplot as plt,os,cv2
from cvxopt import matrix,solvers
from sklearn.base import accuracy_score
from sklearn.metrics import confusion_matrix
from libsvm import svm_problem,svm_parameter,svm_train,svm_predict

def resize(classes,dataset_path):
    data = []
    labels = [] 
    # dataset_path = 'svm/train/'
    for x in classes:
        path = os.path.join(dataset_path, x)
        for y in os.listdir(path):
            image = cv2.imread(os.path.join(path, y))
            # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            image = cv2.resize(image, (16,16))
            image = image.reshape(768)
            image=image/255.0
            data.append(image)
            labels.append((int(x)+1))
    labels = labels.reshape(labels.shape[0],1)
    return data,labels

def gaussian(X, gamma):
    squared_distances = np.sum(X**2, axis=1).reshape(-1, 1) + np.sum(X**2, axis=1) - 2 * np.dot(X, X.T)
    K = np.exp(-gamma * squared_distances)
    return K
def Kernel(X,gamma):
    if gamma:
        return gaussian(X, gamma)
    else:
        return np.dot(X, X.T)
def find_P(X, Y, gamma):
    K = Kernel(X, gamma)
    P=np.outer(Y,Y)*K
    return P


def find_alpha(P,X, Y,c):
    m = len(X)
    P = matrix(P)
    
    q = -1*np.ones((m,1));q=matrix(q)
    
    G = np.vstack((np.eye(m)*-1,np.eye(m)))
    G = matrix(G)

    h = np.hstack((np.zeros(m),np.ones(m)*c))
    h=matrix(h)

    A =  matrix(Y.reshape(1, -1), tc='d')
    b = matrix(0.0, tc='d')
    
    sol = solvers.qp(P,q,G,h,A,b)
    alpha = np.reshape(np.array(sol['x']), (X.shape[0],1))
    return alpha

def find_w_b(alpha,X,Y):
    w = np.sum(alpha*Y*X,axis=0)
    S = (1>alpha>1e-5).flatten()
    b = Y[S]-np.dot(X[S],w)
    b = np.mean(b)
    return w,b

def Kernel(X,gamma):
    if gamma:
        return gaussian(X, gamma)
    else:
        return np.dot(X, X.T)

def predict(W,b,x_test,y_test):
    y_pred=[]
    for i in range(len(x_test)):
        y_pred.append(np.sign(np.dot(W.T,x_test[i])+b))
    # y_pred = [(y+3)//2 for y in y_pred]
    y_pred=np.array(y_pred)
    y_pred=y_pred.reshape(y_test.shape)
    print("accuracy: ",accuracy_score(y_test,y_pred))
    return y_pred

def no_of_sv(alpha,C):
    count=0
    for i in range(len(alpha)):
        if(C>alpha[i]>1e-5):
            count+=1
    return count , count/len(alpha)


######### linear kernel #########

def linear_kernel(X_train,X_test,Y_train,Y_test,C):
    P = find_P(X_train,Y_train,0)
    alpha = find_alpha(P,X_train,Y_train,C)
    W,b = find_w_b(alpha,X_train,Y_train)
    y_pred = predict(W,b,X_test,Y_test)
    print("Linear Kernel\n")
    print("C = 1.0\n")
    print("alpha: ",alpha)
    print("Number of support vectors: ",no_of_sv(alpha,1.0))
    print("W: ",W)
    print("b: ",b)
    print("Accuracy: ",accuracy_score(Y_test,y_pred))
    return y_pred


######### gaussian kernel #########

def gauss_kernel(X_train,X_test,Y_train,Y_test,C,gamma):
    P = find_P(X_train,Y_train,gamma)
    alpha = find_alpha(P,X_train,Y_train,C)
    W,b = find_w_b(alpha,X_train,Y_train)
    y_pred = predict(W,b,X_test,Y_test)
    print("Gaussian Kernel\n")
    print("C = 1.0\n")
    print("alpha: ",alpha)
    print("Number of support vectors: ",no_of_sv(alpha,1.0))
    print("W: ",W)
    print("b: ",b)
    print("Accuracy: ",accuracy_score(Y_test,y_pred))
    return y_pred


######### LIBSVM #########

def svm_libsvm(X_train,Y_train,gamma=0.001):
    param = ""
    if gamma:
        param += f"-t 2 -c 1 -q -g {gamma}"
    else:
        param += f"-t 0 -c 1 -q"
    model = svm_train(Y_train.T, X_train, param)
    indices = model.get_sv_indices()
    for i in range(len(indices)):
        indices[i] -= 1
    alpha = np.abs(np.array(model.get_sv_coef(), ndmin=2))
    X, Y = X_train[indices], Y_train[indices]
    inner_prod = np.sum(alpha * Y * Kernel(X, gamma), 0)
    M = max(range(len(alpha)), key=lambda i: -float("inf") if Y[i] == 1 else inner_prod[i])
    m = min(range(len(alpha)), key=lambda i: float("inf") if Y[i] == -1 else inner_prod[i])
    b = -(inner_prod[M] + inner_prod[m]) / 2

    return indices, alpha, model, b


def accuracy_util_libsvm(X_train, Y_train, X_test, Y_test, model):
    train_accuracy = svm_predict(Y_train, X_train, model, '-q')[1][0]
    test_accuracy = svm_predict(Y_test, X_test, model, '-q')[1][0]
    return train_accuracy, test_accuracy

######### part1 ends here #########

######### part2 starts here #########

x_test = []; y_test = []
classes = ['0','1','2','3','4','5']
dataset_path = 'svm/val/'
for x in classes:
    path = os.path.join(dataset_path, x)
    for y in os.listdir(path):
        image = cv2.imread(os.path.join(path, y))
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.resize(image, (16,16))
        image = image.reshape(768)
        image=image/255.0
        x_test.append(image)
        y_test.append(int(x)+1)
x_test = np.array(x_test)
y_test = np.array(y_test)

x_train = [];y_train = []
dataset_path = 'svm/train/'
for x in classes:
    data=[];labels=[]
    path = os.path.join(dataset_path, x)
    for y in os.listdir(path):
        image = cv2.imread(os.path.join(path, y))
        image = cv2.resize(image, (16,16))
        image = image.reshape(768)
        image=image/255.0
        data.append(image)
        labels.append(int(x)+1)
    x_train.append(data)
    y_train.append(labels)
x_train = np.array(x_train)
y_train = np.array(y_train)



kC2classes = [['1','2'],['1','3'],['1','4'],['1','5'],['1','6'],
              ['2','3'],['2','4'],['2','5'],['2','6'],['3','4'],
              ['3','5'],['3','6'],['4','5'],['4','6'],['5','6']]

def predict(Wbs, x_test, y_test):
    num_classes = 6
    y_pred = np.zeros((len(x_test), num_classes), dtype=int)
    
    for i in range(len(Wbs)):
        Wb = Wbs[i]
        W, b = Wb[0], Wb[1]
        
        for j in range(len(x_test)):
            val = np.sign(np.dot(W, x_test[j]) + b)
            if val > 0:
                y_pred[j, int(kC2classes[i][1]) - 1] += 1
            else:
                y_pred[j, int(kC2classes[i][0]) - 1] += 1
    
    y_pred = np.argmax(y_pred, axis=1) + 1
    print("Predicted labels:", y_pred)
    
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)
    
    return y_pred

def calc_Wbs(x_train,y_train,gamma):
    """
    this method calculates margin and bias for each pair of classes.
    it takes a lot of time to run. ~ 28-35 minutes
    accuracy: 0.55 so far
    """

    Wbs = []
    m=len(x_train[0])*2
    q = -np.ones((m, 1))
    q = matrix(q)
    B = matrix(0.0, tc='d')
    G = np.vstack((-np.eye(m), np.eye(m)))
    G = matrix(G)

    C=1.0
    h = np.zeros((2 * m, 1))
    h[m:, :] = C   
    h = matrix(h)
    # from time import time

    for X in kC2classes:
        # start = time()
        # classes = X
        data = x_train[int(X[0])-1]+x_train[int(X[1])-1]
        labels = y_train[int(X[0])-1]+y_train[int(X[1])-1]
        
        data = np.array(data)
        labels = np.array(labels)
        m = data.shape[0]
        P = find_P(data,labels, gamma)
        P = matrix(P)
        A = matrix(labels.reshape(1, -1), tc='d')

        sol = solvers.qp(P, q,G, h, A, B,options={'show_progress': False}) 
        alpha = np.reshape(np.array(sol['x']), (data.shape[0],1))
        W = sum(alpha[i]*labels[i]*data[i] for i in range(len(alpha)))
        W = np.array(W)  
        b = 0.0
        support_vector_indices = np.where((alpha > 1e-5) & (alpha < C ))[0]
        for idx in support_vector_indices:
            b += labels[idx] - np.dot(W.T, data[idx])
        b /= len(support_vector_indices)
        Wbs.append([W,b])
        # print("done",X,len(Wbs),time()-start)
        print("done",X,len(Wbs))
    return Wbs



