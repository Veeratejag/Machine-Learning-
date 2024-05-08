from q2 import *

# network = NN(1024,5,[5])
# print(network.theta.keys())
# forward,A=network.forward_prop(x_train.T)
# np.mean(A.argmax(axis=0)+1==y_train)
# network.theta['W1']
# Counter(A.argmax(axis=0)+1)
# print(np.mean((A.argmax(axis=1)+1)==y_train))
# confusion_matrix(A.argmax(axis=1)+1,y_train)

epochs=400


for size in [1,5,10,50,100]:
    network = NN(1024,5,[size],activation=relu,activation_prime=relu_derivative)
    times=[]
    print("Hidden Layer Size:",size)
    eta = 0.01
    for i in range(epochs):
        # k=0
        start = time()
        for j in range(0,len(x_train),32):
            # print(j)
            backward=network.back_prop(x_train[j:j+32].T,y_train_one_hot[j:j+32].T,eta=eta)
        times.append(time()-start)
        print(times[0]) if i==0 else None
    print("Time Taken per epoch:",np.mean(times))
        # print("Epoch:",i+1,"Loss:",np.sum(np.square(network.forward_prop(x_train.T)[1].argmax(axis=0)-y_train))/len(x_train)) if i%50==0 else None
    forward,A=network.forward_prop(x_train.T)
    train_acc = np.mean(A.argmax(axis=0)+1==y_train)
    print("Train Accuracy:",train_acc)
    print(classification_report(A.argmax(axis=0)+1,y_train,zero_division=0))
    forward,A=network.forward_prop(x_test.T)
    test_acc = np.mean(A.argmax(axis=0)+1==y_test)
    print("Test Accuracy:",test_acc)
    print(classification_report(A.argmax(axis=0)+1,y_test,zero_division=0))
    print()

