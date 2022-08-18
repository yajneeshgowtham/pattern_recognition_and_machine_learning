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
    X.append(np.longdouble(x))
    T.append(np.longdouble(t))

X = np.array(X)
T = np.array(T)

X_dev = []
T_dev = []
f_dev = open("1d_team_25_dev.txt", "r")
for line in f_dev.readlines():
    x, t = line.strip().split(' ')
    X_dev.append(np.longdouble(x))
    T_dev.append(np.longdouble(t))

X_dev = np.array(X_dev)
T_dev = np.array(T_dev)

n =200
error_trn = np.array([])
error_dev = np.array([])
m_array = np.arange(1,15,1)
for m in m_array:

    X_trn = np.array([X[i] for i in range(n)])
    Phi = np.array([[X_trn[i] ** j for j in range(m)] for i in range(n)])
    Phi_t = Phi.T
    temp1 = Phi_t @ Phi
    temp2 = LA.inv(temp1)
    Phi_pinv = temp2 @ Phi_t
    # Phi_pinv = LA.pinv(Phi)
    T_trn = np.array([T[i] for i in range(n)])
    W = Phi_pinv @ T_trn
    Y = Phi @ W.T
    plt.scatter(T_trn, Y)
    plt.xlabel("target output")
    plt.ylabel("model output")
    plt.title(" target output vs model output for training data for M="+str(m))
    plt.show()
    plt.scatter(X_trn,T_trn)
    plt.plot(X_trn,Y)
    plt.title("m value is "+str(m)+ " for training data")
    plt.show()
    E_trn = Y - T_trn
    E_trn = E_trn ** 2
    err_trn = np.sum(E_trn) / n
    err_trn = math.sqrt(err_trn)
    error_trn = np.append(error_trn, err_trn)
    Phi_dev = np.array([[X_dev[i] ** j for j in range(m)] for i in range(n)])
    T_dev_samp = np.array([T_dev[i] for i in range(n)])
    X_dev_samp=np.array([X_dev[i] for i in range(n)])
    Y_dev = Phi_dev @ W.T
    E_dev = Y_dev - T_dev_samp
    E_dev = E_dev ** 2
    err_dev = np.sum(E_dev) / n
    err_dev = math.sqrt(err_dev)
    error_dev = np.append(error_dev, err_dev)
    plt.scatter(T_dev_samp, Y_dev)
    plt.xlabel("target output")
    plt.ylabel("model output")
    plt.title(" target output vs model output for test data for M=" + str(m))
    plt.show()
    plt.scatter(X_dev_samp, T_dev_samp)
    plt.plot(X_dev_samp, Y_dev)
    plt.title("m value is " + str(m)+" for test data")
    plt.show()


print("error_train:")
print(error_trn)
print("#################################################################################################")

print("error_dev:")
print(error_dev)



plt.plot(m_array,error_trn)
plt.plot(m_array,error_dev)
plt.ylabel("e_rms")
plt.xlabel("value of m")
plt.show()