import numpy as np,matplotlib.pyplot as plt
X = np.loadtxt(open('logisticX.csv'), ndmin=2,delimiter=',')
Y = np.loadtxt(open('logisticY.csv'), ndmin=2)
Y = np.reshape(Y, (X.shape[0], 1))

def normalise(X):
    return (X - np.mean(X, axis=0)) / np.std(X, axis=0)
X = normalise(X)

def model(X,T): 
    # print(T)
    return 1/(1+np.exp(-np.matmul(X,T)))

def hessian(x,T):
    grad = np.exp(-np.matmul(x,T).T)
    t = 1+grad
    return -np.matmul(x.T * grad / (t*t),x)
    # pass


def likelihood(X,Y,T):
    h_T = model(X,T)
    ans = (np.matmul(Y.T,np.log(h_T))+np.matmul(1-Y.T,np.log(1-h_T))) # Generate predicted values using the model
    return ans[0][0]



def likelihood_grad(X,Y,T):
    return np.dot(X.T,Y-model(X,T))



def gradient_descent(X, Y, e: float, n= 0.29):
    X=normalise(X)
    X = np.insert(X, 0, np.ones(X.shape[0]), axis=1)
    T = np.array([[0], [0], [0]])
    prev_likelihood= 0
    curr_likelihood = 1+e
    iterations = 0
    while 1:
        # print(curr_likelihood,prev_likelihood)
        if abs(curr_likelihood - prev_likelihood) < e :
            # print("iteration: ",iterations," likelihood: ",curr_likelihood)
            break
        # print("iteration: ",iterations," likelihood: ",curr_likelihood)
        d_j = likelihood_grad(X, Y, T)
        
        T = T - np.matmul(np.linalg.inv(hessian(X,T)),likelihood_grad(X, Y, T))
        # T = T + n*d_j
        prev_likelihood = curr_likelihood 
        curr_likelihood = likelihood(X, Y, T)
        iterations+= 1
        # ans.append((T[0][0], T[1][0],T[2][0], curr_likelihood))

    return T, iterations
    # return T, i,ans



T, t = gradient_descent(X, Y, 1e-20, 0.29)
print("Theta: ",T, "iterations",t)
# print(Ts)

print("\n")
fig, axes = plt.subplots()
axes.set(xlabel='x1', ylabel='x2', title='Logistic Regression')

class_1= X[Y[:, 0] == 0]
axes.scatter(class_1[:, 0], class_1[:, 1], c='blue', marker='*', label='Class 1')

class_2 = X[ Y[:, 0] == 1]
axes.scatter(class_2[:, 0], class_2[:, 1], c='green', marker='D', label='Class 2')

# x_values = X[:, 0]
y_values = -(T[0][0] + T[1][0] *  X[:, 0]) / T[2][0]

axes.plot( X[:, 0], y_values, c='violet', label='Logistic Separator')

axes.set_xlabel('Feature 1')
axes.set_ylabel('Feature 2')
axes.set_title('Binary Classification with Logistic Separator')

axes.legend()
plt.show()
fig.savefig("logistic.png")
plt.close()