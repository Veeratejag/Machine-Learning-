from q2_1_pre import x_train_tot,y_train_tot,x_test_tot,y_test_tot
import numpy as np
from sklearn import svm

X_train = [];Y_train = []
X_test = [];Y_test = []
for i in range(len(x_train_tot)):
    X_train.append(np.array(x_train_tot[i]))
    Y_train.append(np.array(y_train_tot[i]))
for i in range(len(x_test_tot)):
    X_test.append(np.array(x_test_tot[i]))
    Y_test.append(np.array(y_test_tot[i]))

def k_fold(X,Y,c,k):
    m = X.shape[0]
    m //= k
    X_test, X_train = np.split(X, [m])
    Y_test, Y_train = np.split(Y, [m])
    accuracy = 0
    for i in range(k):
        clf = svm.SVC(kernel='rbf', gamma=0.001, C=c)
        clf.fit(X_train,Y_train)
        pred = clf.predict(X_test)
        accuracy += np.mean(pred == Y_test)
        if i < k - 1:
            X_train[i * m:(i + 1) * m], X_test = X_test, X_train[i * m:(i + 1) * m].copy()
            Y_train[i * m:(i + 1) * m], Y_test = Y_test, Y_train[i * m:(i + 1) * m].copy()
    return accuracy / k

def accuracy_libsvm(X,Y,X_test,Y_test,C):
    model = svm.SVC(kernel='rbf', gamma=0.001, C=C)
    model.fit(X,Y)
    pred = model.predict(X_test)
    # print("Validation Accuracy: ",np.mean(pred==Y_test))
    return pred

kf_Acc = [];v_Acc = []
C_values=[1e-5,1e-3,1,5,10]
for c in C_values:
    pred = accuracy_libsvm(X_train,Y_train,X_test,Y_test,c) 
    v_Acc.append(np.mean(pred==Y_test))
    print("C: ",c," Validation Accuracy ",np.mean(pred==Y_test))

indices = np.arange(len(X_train))
np.random.shuffle(indices)
X_train = X_train[indices]
Y_train = X_train[indices]

for c in C_values:
    k_fold_Accuracy = k_fold(X_train,Y_train,c,5)
    kf_Acc.append(k_fold_Accuracy)
    print("C: ",c,"k_fold Accuracy: ",k_fold_Accuracy)
    

print("Kfold accuracies: ",kf_Acc)
print("Validation Accuracies: ",v_Acc)

import matplotlib.pyplot as plt
plt.plot(C_values,kf_Acc,label="Kfold Accuracy")
plt.plot(C_values,v_Acc,label="Validation Accuracy")
plt.xscale("log")
plt.xlabel("C")
plt.ylabel("Accuracy")
plt.legend()
plt.show()