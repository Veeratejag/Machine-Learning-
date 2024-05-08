import pandas as pd,numpy as np,matplotlib.pyplot as plt
X = np.loadtxt(open('q4x.dat'), ndmin=2)
Y = pd.read_csv('q4y.dat', header=None, names=['country'])
def normalise(X):
    return (X - np.mean(X, axis=0)) / np.std(X, axis=0)

X = normalise(X)


fig, axes = plt.subplots()
Alaska=[[],[]];Canada=[[],[]]
for i in range(X.shape[0]):
    if Y['country'][i] == "Alaska":
        Alaska[0].append(X[i][0])
        Alaska[1].append(X[i][1])
    else:
        Canada[0].append(X[i][0])
        Canada[1].append(X[i][1])

Alaska[0]=np.array(Alaska[0]);Alaska[1]=np.array(Alaska[1])
Canada[0]=np.array(Canada[0]);Canada[1]=np.array(Canada[1])


plt.plot(Alaska[0],Alaska[1],label='Alaska',marker='o',linestyle='None')
plt.plot(Canada[0],Canada[1],label='Canada',marker='*',linestyle='None')
axes.set(title='Distribution of Salmons',xlabel='diameter in Fresh Water',ylabel='diameter in Marine Water')


# plt.show()


def bernoulli(phi,y):
    return phi**y*(1-phi)**(1-y)

def gaussian(x,u,sigma):
    temp=np.matmul(np.exp(-(x-u).T),np.linalg.inv(sigma))
    return (((2*np.pi)**(x.shape[0]))*np.linalg.det(sigma))**(-0.5)*np.matmul(temp,(x-u)/2)
def mu(x,y,z):
    return np.matmul(np.matmul(x.T,y),z)


def gaussian_linear(x,u0,u1,sigma,phi):
    log_part = np.log((1-phi)/phi)
    u_part = np.matmul(np.matmul(u0.T,np.linalg.inv(sigma)),u0)- np.matmul(np.matmul(u1,np.linalg.inv(sigma)),u1)
    ux_part = 2*mu(x,np.linalg.inv(sigma),u1-u0)
    return log_part-0.5*(u_part+ux_part)


def gaussan_general(x,u0,u1,sigma0,sigma1,phi):
    log_part=np.log((1-phi)*np.sqrt(np.linalg.det(sigma1))/(phi*np.sqrt(np.linalg.det(sigma0))))
    x_part = mu(x,np.linalg.inv(sigma0),x)- mu(x,np.linalg.inv(sigma1),x)
    u_part = mu(u0,np.linalg.inv(sigma0),u0)- mu(u1,np.linalg.inv(sigma1),u1)
    ux_part=mu(x,np.linalg.inv(sigma1),u1)- mu(x,np.linalg.inv(sigma0),u0)
    return log_part-0.5*(x_part+u_part+ux_part*2)
    
def indicator(cond:bool):
    return 1 if cond else 0

def mul(X):
    return np.array([[X[0]*X[0],X[0]*X[1]],[X[1]*X[0],X[1]*X[1]]])
def cal_sigma(X,u0,u1):
    sigma = np.zeros((2,2))
    for i in range(X.shape[0]):
        sigma += mul(X[i]-u0)*indicator(Y['country'][i]=="Alaska")
        sigma += mul(X[i]-u1)*indicator(Y['country'][i]=="Canada")
    return sigma/X.shape[0]

def cal_phi(Y):
    return np.sum(Y['country']=="Canada")/Y.shape[0]

def cal_u(X,Y):
    alaska=[0,0];canada=[0,0]
    a,c=0,0
    for i in range(X.shape[0]):
        alaska += X[i]*indicator(Y['country'][i]=="Alaska")
        canada += X[i]*indicator(Y['country'][i]=="Canada")
        a+=indicator(Y['country'][i]=="Alaska")
        c+=indicator(Y['country'][i]=="Canada")
    
    return alaska/a,canada/c
u0,u1=cal_u(X,Y)

sigma=cal_sigma(X,u0,u1)


def sigma0(X,u0):
    sigma = np.zeros((2,2))
    a=0
    for i in range(X.shape[0]):
        sigma += mul(X[i]-u0)*indicator(Y['country'][i]=="Alaska")
        a+=indicator(Y['country'][i]=="Alaska")
    return sigma/a
def sigma1(X,u1):
    sigma = np.zeros((2,2))
    c=0
    for i in range(X.shape[0]):
        sigma += mul(X[i]-u1)*indicator(Y['country'][i]=="Canada")
        c+=indicator(Y['country'][i]=="Canada")
    return sigma/c

sigma0 = sigma0(X,u0)
sigma1 = sigma1(X,u1)

x=np.linspace(-2,2,100)
y=np.linspace(-2,2,100)
x1,x2=np.meshgrid(x,y)


Z = np.apply_along_axis(lambda x: gaussian_linear(x, u0, u1, sigma, 0.5), 0, np.stack([x1, x2]))

contour1 = plt.contour(x1,x2,Z,levels=[0],colors='red')
plt.plot([], [], 'red', label='linear gaussian')
plt.plot()


Z1 = np.apply_along_axis(lambda x: gaussan_general(x, u0, u1, sigma0, sigma1, 0.5), 0, np.stack([x1, x2]))
contour2=plt.contour(x1,x2,Z1,levels=[0],colors='blue')
plt.plot([], [], 'blue', label='general gaussian')
plt.legend()
plt.show()
# fig.savefig( "q4.png")



print("phi:\n",cal_phi(Y))
print("u0:\n",u0.T,"\n","u1:\n",u1.T )
print("Sigma: \n",sigma.T)
print("Sigma0: \n",sigma0)
print("Sigma1: \n",sigma1)