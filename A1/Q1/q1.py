from matplotlib import animation
import numpy as np,matplotlib.pyplot as plt
X = np.loadtxt(open('./linearX.csv'), ndmin=2)
Y = np.loadtxt(open('./linearY.csv'), ndmin=2)


def normalise(X):
    return (X - np.mean(X, axis=0)) / np.std(X, axis=0)

X = normalise(X)
def loss(X,Y,theta):
    return np.sum(np.square(np.dot(X,theta)-Y))/(200)

def loss_grad(X,Y,theta):
    return np.dot(X.T,np.dot(X,theta)-Y)/100


def gradient_descent(X, Y, n, e):
    T = [[0.0], [0.0]]
    prev_loss= 1+e
    curr_loss = 0
    ans = [(T[0][0], T[1][0], loss(X, Y, T))]
    i = 0
    while 1:
        if abs(curr_loss - prev_loss) < e:
            break
        prev_loss = curr_loss
        grad = loss_grad(X, Y, T)
        T = T - n*grad
        curr_loss = loss(X, Y, T)
        i += 1
        ans.append((T[0][0], T[1][0], curr_loss))


    return T, i,ans


X_new = np.insert(X, 0, np.ones(X.shape[0]), axis=1)
T, t, Ts = gradient_descent(X_new, Y, 0.05, 1e-15)
print("eta: 0.05",T[0][0],T[1][0], t)
print("\n")
# part b
regression, graph = plt.subplots()
graph.set(title='Linear Regression',xlabel='theta0',ylabel='theta1')
graph.scatter(X, Y, label='data', color='green')
graph.plot(X, X * T[1] + T[0], color='red', label='regression line')
graph.legend()
regression.tight_layout()
plt.show()
# regression.savefig( "b.png")
plt.close()

print("\n")

# part c
X = np.insert(X, 0, np.ones(X.shape[0]), axis=1)
T, t, Ts = gradient_descent(X, Y, 0.05, 1e-15)
fig = plt.figure()
axes = fig.add_subplot(111, projection='3d',title='Loss Function',xlabel='theta0',ylabel='theta1',zlabel='loss')

X_data,Y_data = np.meshgrid(*np.linspace(-T.T[0], 2*T.T[0], 100).T)

J = np.apply_along_axis(
        lambda theta: loss(X, Y, np.reshape(theta, (-1, 1))),
        2, np.stack([X_data, Y_data], axis=-1))
axes.plot_surface(X_data,Y_data,J,cmap='viridis')

sc = axes.scatter([], [], [], c='b', marker='o') 
Ts = np.array([(theta0, theta1, loss) for theta0,theta1, loss in Ts]).T
axes.scatter(Ts[0][0], Ts[1][0], Ts[2][0], c='r', marker='o', label='Starting Point')
axes.scatter(Ts[0][-1], Ts[1][-1], Ts[2][-1], c='g', marker='o', label='Final Point')

for i in range(Ts.shape[1]):  # Start from 1 since we already plotted the initial point
    axes.scatter(Ts.T[i][0], Ts.T[i][1], Ts.T[i][2],c='red', alpha=1, marker='o')
    plt.pause(1e-6)

plt.show()
# plt.savefig("c.png")
plt.close()

# part d

fig2, ax = plt.subplots()
ax.set_xlim(0,2)
ax.set_ylim(-1,1)
x_range = np.linspace(0,2, 100)
y_range = np.linspace(-1, 1, 100)
X_data,Y_data = np.meshgrid(x_range,y_range)
f=lambda theta: loss(X, Y, np.reshape(theta, (-1, 1)))
J = np.apply_along_axis(f,2, np.stack([X_data, Y_data], axis=-1))

plt.contour(X_data,Y_data,J,levels=100,cmap='viridis')

plt.title('eta = 0.05')
plt.xlabel('theta0')
plt.ylabel('theta1')
for i in range(Ts.shape[1]):
    ax.scatter(Ts.T[i][0], Ts.T[i][1], c='r', marker='o')
    plt.pause(0.00001)

plt.show()
# plt.savefig("d.png")
plt.close()


for i in {0.001,0.025,0.1}:
    T, t, Ts = gradient_descent(X, Y, i, 1e-15)
    print("eta:",i,T[0][0],T[1][0], t)
    Ts = np.array([(theta0, theta1, loss) for theta0,theta1, loss in Ts]).T
    fig2, ax = plt.subplots()
    ax.set_xlim(0,2)
    ax.set_ylim(-1,1)
    x_range = np.linspace(0,2, 100)
    y_range = np.linspace(-1, 1, 100)
    X_data,Y_data = np.meshgrid(x_range,y_range)
    f=lambda theta: loss(X, Y, np.reshape(theta, (-1, 1)))
    J = np.apply_along_axis(f,2, np.stack([X_data, Y_data], axis=-1))

    plt.contour(X_data,Y_data,J,levels=100,cmap='viridis')
    plt.xlabel('theta0')
    plt.ylabel('theta1')
    plt.title('eta = '+str(i))
    for i in range(Ts.shape[1]):
        ax.scatter(Ts.T[i][0], Ts.T[i][1], c='r', marker='o')
        plt.pause(1e-6)
     
    plt.show()
    # plt.savefig(f"e{i}.png")
    plt.close()
    

    

