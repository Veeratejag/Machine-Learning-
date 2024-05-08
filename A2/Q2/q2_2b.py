from q2_1a import sv_libsvm,accuracy_libsvm
from q2_1_pre import x_train_tot,y_train_tot,x_test_tot,y_test_tot
import numpy as np
from time import time

X_train = [];Y_train = []
X_test = [];Y_test = []
for i in range(len(x_train_tot)):
    X_train.append(np.array(x_train_tot[i]))
    Y_train.append(np.array(y_train_tot[i]))
for i in range(len(x_test_tot)):
    X_test.append(np.array(x_test_tot[i]))
    Y_test.append(np.array(y_test_tot[i]))

print("Multi class classification using one vs one approach with LIBSVM")
start = time()
model = sv_libsvm(X_train,Y_train[0],2,0.001)
print("Time taken for training: ",time()-start)
accuracy_libsvm(model,X_train,Y_train,X_test,Y_test)


