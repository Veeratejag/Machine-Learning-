from q2 import *

M=32
epochs=400
for size in [[512],[512,256],[512,256,128],[512,256,128,64]]:
    network = NN(1024,5,size)
    times=[]
    eta = 0.01
    l_prev,l_curr=float('inf'),0
    print("Hidden Layer Size:",size)
    for i in range(epochs):
        l_prev=l_curr
        start = time()
        for j in range(0,len(x_train),M):
            backward=network.back_prop(x_train[j:j+M].T,y_train_one_hot[j:j+M].T)
        l_curr = network.loss(y_train_one_hot.T,network.forward_prop(x_train.T)[1])
        times.append(time()-start)
        if abs(l_prev-l_curr)<1e-10:
            break
        print(np.mean(times)) if i==0 else None
    print("Time Taken per epoch:",np.mean(times),"epochs:",i+1)
    forward,A=network.forward_prop(x_train.T)
    train_acc = np.mean(A.argmax(axis=0)+1==y_train)
    print(classification_report(A.argmax(axis=0)+1,y_train,zero_division=0))
    print("Train Accuracy:",train_acc)
    forward,A=network.forward_prop(x_test.T)
    test_acc = np.mean(A.argmax(axis=0)+1==y_test)
    print("Test Accuracy:",test_acc)
    print(classification_report(A.argmax(axis=0)+1,y_test,zero_division=0))
    print()

