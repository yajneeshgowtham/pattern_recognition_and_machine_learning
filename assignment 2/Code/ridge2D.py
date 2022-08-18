import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import scipy as sc
from scipy import linalg as LA
from PIL import Image
import math

X1 = []
X2 = []
T = []
f = open("2d_team_25_train.txt", "r")
for line in f.readlines():
    x1, x2, t = line.strip().split(' ')
    X1.append(x1)
    X2.append(x2)
    T.append(t)

X1 = list(map(float, X1))
X1 = np.array(X1)

X2 = list(map(float, X2))
X2 = np.array(X2)

T = list(map(float, T))
T = np.array(T)

X1_dev = []
X2_dev = []
T_dev = []
f = open("2d_team_25_dev.txt", "r")
for line in f.readlines():
    x1, x2, t = line.strip().split(' ')
    X1_dev.append(x1)
    X2_dev.append(x2)
    T_dev.append(t)

X1_dev = list(map(float, X1_dev))
X1_dev = np.array(X1_dev)

X2_dev = list(map(float, X2_dev))
X2_dev = np.array(X2_dev)

T_dev = list(map(float, T_dev))
T_dev = np.array(T_dev)

n = 1000

m= 15

error_trn = np.array([])
error_dev = np.array([])
lam_array=np.arange(-40,0,1)
for lam in lam_array:
    lam1 = math.exp(lam)
    Phi = np.array([[0. for j in range((m * (m + 1)) // 2)] for i in range(n)])

    start = 0
    end = 0
    for idx in range(m):
        for j in range(start, end + 1):
            for i in range(n):
                Phi[i][j] = (X1[i] ** (idx - j + start)) * (X2[i] ** (j - start))
        start = end + 1
        end = start + idx + 1
    Phi_t = Phi.T
    temp1 = Phi_t @ Phi
    Idnt = np.identity((m*(m+1))//2, dtype='longdouble')
    Idnt = Idnt * lam1
    temp2 = np.add(temp1,Idnt)
    temp3 = LA.inv(temp2)
    Phi_pinv = temp3 @ Phi_t
    # Phi_pinv=LA.pinv(Phi)
    T_trn = np.array([T[i] for i in range(n)])
    W = Phi_pinv @ T_trn
    Y = Phi @ W.T
    print("ln Lambda value is "+str(lam))
    print(W)
    print("##############################################################")
    plt.scatter(T_trn,Y)
    plt.xlabel("Target Data")
    plt.ylabel("Trained Data")
    plt.title(str(n)+","+str(m))
    plt.show()

    E_trn = Y - T_trn
    E_trn = E_trn ** 2
    err_trn = np.sum(E_trn) / n
    err_trn = math.sqrt(err_trn)
    error_trn = np.append(error_trn, err_trn)

    Phi_dev = np.array([[0. for j in range((m * (m + 1)) // 2)] for i in range(n)])

    start = 0
    end = 0
    for idx in range(m):
        for j in range(start, end + 1):
            for i in range(n):
                Phi_dev[i][j] = (X1_dev[i] ** (idx - j + start)) * (X2_dev[i] ** (j - start))
        start = end + 1
        end = start + idx + 1

    T_dev_samp = np.array([T_dev[i] for i in range(n)])
    Y_dev = Phi_dev @ W.T
    E_dev = Y_dev - T_dev_samp
    E_dev = E_dev ** 2
    err_dev = np.sum(E_dev) / n
    err_dev = math.sqrt(err_dev)
    error_dev = np.append(error_dev, err_dev)

    plt.scatter(T_dev_samp,Y_dev)
    plt.xlabel("Target Data")
    plt.ylabel("Trained Data")
    plt.title(str(n)+","+str(m))
    plt.show()

print("error_trn is :" + str(error_trn))
print("########################################")
print("error_dev is :" + str(error_dev))

plt.plot(lam_array, error_trn)
plt.plot(lam_array, error_dev)
plt.show()

