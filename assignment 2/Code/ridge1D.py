import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import scipy as sc
from scipy import linalg as LA
from PIL import Image
import math

X = []
T = []
f = open("1d_team_25_train.txt", "r")
for line in f.readlines():
    x, t = line.strip().split(' ')
    X.append(x)
    T.append(t)

X = list(map(float, X))
X = np.array(X)

T = list(map(float, T))
T = np.array(T)

X_dev = []
T_dev = []
f_dev = open("1d_team_25_dev.txt", "r")
for line in f_dev.readlines():
    x, t = line.strip().split(' ')
    X_dev.append(x)
    T_dev.append(t)

X_dev = list(map(float, X_dev))
X_dev = np.array(X_dev)

T_dev = list(map(float, T_dev))
T_dev = np.array(T_dev)

n = 10
error_trn = np.array([])
error_dev = np.array([])
m = 9
lam_array=np.arange(-20,-10,1)
for lam in lam_array:
    lam1=math.exp(lam)
    # Phi = np.array([[0.0 for j in range(m)] for i in range(n+m)])
    # for i in range(n):
    #     for j in range(m):
    #         Phi[i][j]=X[i]**j
    # for i in range(n,n+m):
    #     Phi[i][i-n]=math.sqrt(lam)
    # T_trn=np.array([0.0 for i in range(n+m)])
    # for i in range(n):
    #     T_trn[i]=T[i]
    # Phi_pinv=LA.pinv(Phi)
    Phi=[]
    X_trn = np.array([X[i] for i in range(n)])
    for i in range(n):
        for j in range(m):
            Phi.append(X_trn[i]**j)
    Phi=np.array(Phi)
    Phi.resize(n,m)
    temp=(Phi.T)@Phi
    Idnt=np.identity(m,dtype='float64')
    Idnt=Idnt*lam1
    temp1=temp+Idnt
    temp2=LA.inv(temp1)
    Phi_final=temp2@(Phi.T)


    T_trn=np.array([T[i] for i in range(n)])
    W = Phi_final @ T_trn
    Y = Phi @ W.T
    # print("ln Lambda is "+str(lam))
    # print(W)
    # print("#############################################################")
    # plt.scatter(X_trn, T_trn)
    # plt.plot(X_trn, Y)
    # plt.title("m value is " + str(m))
    # plt.title("Lambda is "+ str(lam))
    # plt.show()
    # plt.scatter(T_trn, Y)
    # plt.xlabel("Target Data")
    # plt.ylabel("Trained Data")
    # plt.title(str(n) + "," + str(m))
    # plt.show()

    E_trn = Y - T_trn
    E_trn=E_trn**2
    err_trn=np.sum(E_trn)/n
    err_trn=math.sqrt(err_trn)
    error_trn = np.append(error_trn, err_trn)
    Phi_dev = np.array([[X_dev[i] ** j for j in range(m)] for i in range(n)])
    T_dev_samp = np.array([T_dev[i] for i in range(n)])
    Y_dev = Phi_dev @ W.T
    E_dev = Y_dev - T_dev_samp
    E_dev=E_dev**2
    err_dev=np.sum(E_dev)/n
    err_dev=math.sqrt(err_dev)
    error_dev = np.append(error_dev, err_dev)
    # plt.scatter(T_dev_samp, Y_dev)
    # plt.xlabel("Target Data dev")
    # plt.ylabel("Trained Data dev")
    # plt.title(str(n) + "," + str(m))
    # plt.show()

print("error_train:")
print(error_trn)
print("#################################################################################################")

print("error_dev:")
print(error_dev)


plt.plot(lam_array, error_trn)
plt.plot(lam_array, error_dev)
plt.title("ridge regression")
plt.show()