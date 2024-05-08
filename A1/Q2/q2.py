from time import time
import numpy as np,matplotlib.pyplot as plt
from collections import deque
def sample(x,theta,noise,m):
    # x = np.insert(x, 0, [1, 0], axis=0)
    X = np.random.normal(np.insert(x, 0, [1, 0], axis=0).T[0],np.sqrt(np.insert(x, 0, [1, 0], axis=0).T[1]),(m,np.insert(x, 0, [1, 0], axis=0).shape[0]))
    # Y = np.dot(X,theta) + np.random.normal(0,np.sqrt(noise),(m,1))
    return X,np.dot(X,theta) + np.random.normal(0,np.sqrt(noise),(m,1))

def loss(X,Y,theta):
    return np.sum(np.square(np.dot(X,theta)-Y))/(2*X.shape[0])

def loss_grad(X,Y,theta):
    return np.dot(X.T,np.dot(X,theta)-Y)/(X.shape[0])



def stoch_grad_descent(X,Y,learning_rate,epsilon,r):
    # m=
    X=X[:,1:]
    X=np.insert(X,0,np.ones(X.shape[0]),axis=1)
    k=1000
    q=deque()
    theta = np.array([[0.0],[0.0],[0.0]])
    
    theta_history=[]
    
    loss_prev,loss_curr=loss(X,Y,theta),0
    
    epoch,t=0,0
    
    while abs(loss_prev-loss_curr)>epsilon:
        loss_prev=loss_curr;loss_curr=0
        for i in range(X.shape[0]//r):

            X_batch=X[i*r:(i+1)*r];Y_batch=Y[i*r:(i+1)*r]
            theta -= learning_rate*loss_grad(X_batch,Y_batch,theta)
            loss_curr+=loss(X_batch,Y_batch,theta)

            if len(q)==k and  abs((sum(q)/k)-loss(X_batch,Y_batch,theta))<=epsilon:
                copy=np.copy(theta)
                theta_history.append(copy.T[0].tolist())
                return theta,t+1,epoch,np.array(theta_history)
            
            if len(q)>=k:
                q.popleft()
            
            q.append(loss(X_batch,Y_batch,theta))
            copy=np.copy(theta)
            theta_history.append(copy.T[0].tolist())
            t+=1
            
        loss_curr /= X.shape[0]//r
        epoch+=1

        if abs(loss_prev-loss_curr)>1e20:
            print('Diverged')
            return None,t,epoch,theta_history
        
    return theta,t,epoch,np.array(theta_history)


def normalise(X):
    return (X - np.mean(X, axis=0)) / np.std(X, axis=0)
X, Y = sample(np.array([[3, 4], [-1, 4]]),np.array([[3], [1], [2]]), 2, 1000000)

# np.savetxt( "q2_X.csv", X, delimiter=',')
# np.savetxt( "q2_Y.csv", Y)
# X = np.loadtxt(open('q2_X.csv'), ndmin=2, delimiter=',')
# Y = np.loadtxt(open('q2_Y.csv'), ndmin=2)
# Y = np.reshape(Y, (X.shape[0], 1))
Thetas = np.empty((3, 4))
times = ["batch size,time taken,iterations,epochs"]
print(len(X),len(Y))

test_data=np.genfromtxt('q2test.csv',delimiter=',')
test_data=test_data[1:]
X_test = test_data[:, :2]
Y_test = test_data[:, 2:]
X_test=np.insert(X_test,0,np.ones(X_test.shape[0]),axis=1)
bachsizes=[1,100,10000,1000000]
for i in range(4):
    r=bachsizes[i]
    
    t=time()
    Theta, iters, epochs, Thetass = stoch_grad_descent(X, Y, 0.01, 1e-5, r)
    Theta = np.reshape(Theta, (-1, 1))
    print("Loss against given dataset: ",loss(X_test,Y_test,Theta))
    print("Learnt for batch size {} in {} seconds".format(r, time() - t))
    print(Theta,iters,epochs)
    theta1 = Thetass.T[0, :]
    theta2 = Thetass.T[ 1,:]
    theta3 = Thetass.T[2,:]
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(theta1, theta2, theta3, marker='o')
    ax.plot(0, 0,0, marker='o', color='red', label='First Point')  
    ax.plot(theta1[-1], theta2[-1], theta3[-1], marker='o', color='green', label='Last Point')  

    ax.set(xlabel='theta1', ylabel='theta2', zlabel='theta3', title='theta plot for batch size ' + str(r))
    plt.savefig('r='+str(r)+'.png')
    ax.legend()

    plt.show()
theta = np.array([[3],[1],[2]]).T
Theta = np.reshape(theta, (-1, 1))
print(loss(X_test,Y_test,Theta))

