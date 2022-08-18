import os
import matplotlib.pyplot as plt
import numpy as np
import random
import mpmath
from collections import OrderedDict
from numpy import linalg as LA
import math
from sklearn.metrics import DetCurveDisplay

directory1 = 'coast/train'
directory2 = 'forest/train'
directory3 = 'highway/train'
directory4 = 'mountain/train'
directory5 = 'opencountry/train'

directory6 = 'coast/dev'
directory7 = 'forest/dev'
directory8 = 'highway/dev'
directory9 = 'mountain/dev'
directory10 = 'opencountry/dev'

def extractData(directory):
    data=[]
    for filename in os.listdir(directory):
        f = os.path.join(directory, filename)
        f=open(f,"r")
        m=[]
        for line in f.readlines():
            x=line.split()
            x = list(map(float, x))
            data.append(x)
    return np.array(data)
def extrctDataDev(directory):
    data = []
    for filename in os.listdir(directory):
        f = os.path.join(directory, filename)
        f = open(f, "r")
        m = []
        for line in f.readlines():
            x = line.split()
            x = list(map(float, x))
            m.append(x)
        data.append(m)
    return np.array(data)


def initialParam(data,K):
    len=data[:,:1].size
    idx=[i for i in range(len)]
    idx=random.sample(idx, K)
    idx.sort()
    mu=[]
    mu1=[]
    for x in idx:
        mu.append(data[x])
        mu1.append(data[x])
    return (np.array(mu),np.array(mu1))
def nearCluster(x,mu):
    dist=np.array([math.dist(x,u) for u in mu])
    return np.argmin(dist),dist[np.argmin(dist)]
def meanOfCluster(cluster):
    r,c=cluster.shape
    x=np.array([0. for i in range(c)])
    for y in cluster:
        x=np.add(x,y)
    x=x/r
    return x
def K_Means(mu,data,K):
    Change=True
    itr=0
    while Change and itr<=10:
        Cluster = {}
        for x in data:
            i,dist=nearCluster(x,mu)
            if i not in Cluster.keys():
                Cluster[i]=[x]
            else:
                Cluster[i].append(x)
        cnt=0
        Cluster=OrderedDict(sorted(Cluster.items()))
        for i in Cluster.keys():
            Cluster[i]=np.array(Cluster[i])
            copy = np.empty_like(mu[i])
            copy[:] = mu[i]
            mu[i]=meanOfCluster(Cluster[i])
            if np.array_equal(copy,mu[i]):
                cnt+=1
        if cnt==K:
            Change=False
        itr+=1
    return mu
def finalClusters(data,mu):
    Cluster={}
    for x in data:
        i, dist = nearCluster(x, mu)
        if i not in Cluster.keys():
            Cluster[i] = [x]
        else:
            Cluster[i].append(x)

    Cluster = OrderedDict(sorted(Cluster.items()))
    for i in Cluster.keys():
        Cluster[i] = np.array(Cluster[i])
    return Cluster
def finalsigma(cluster,mu):
    sig=np.array([[0. for j in range(np.size(mu))]for i in range(np.size(mu))])
    cnt=0
    for x in cluster:
        a=x-mu
        a=np.reshape(a,(np.size(x),1))
        sig+=a@a.T
        cnt+=1
    sig=sig/(cnt)
    return sig
def finalSigmas(Cluster,MU):
        Sig=[finalsigma(Cluster[i],MU[i]) for i in Cluster.keys()]
        return Sig
def finalPhis(Cluster):
    Phi = np.array([Cluster[i].size for i in Cluster.keys()])
    Phi=Phi/np.sum(Phi)
    return Phi
def gauss(x,mu,sig):
    a=x-mu
    a=np.reshape(a,(np.size(x),1))
    sig=np.array(sig)
    r=a.T@LA.inv(sig)@a
    res=mpmath.exp(-r[0][0]/2)/mpmath.sqrt(LA.det(sig))
    return res
def estimationSTEP(MU,Sigma,Phi,K,data):
    n=len(data)
    gam = np.array([[0. for j in range(K)] for i in range(n)])
    for i in range(n):
        cnt = 0
        for j in range(K):
            gam[i][j] = Phi[j] * gauss(data[i], MU[j], Sigma[j])
            cnt += gam[i][j]
        for j in range(K):
            gam[i][j] = gam[i][j] / cnt
    N = np.array([0. for j in range(K)])
    for j in range(K):
        for i in range(n):
            N[j] += gam[i][j]
    return (gam,N)
def newMean(gam,N,data,n):
    c = 23
    mu=np.array([0. for i in range(c)])
    for i in range(n):
        mu=mu+gam[i]*data[i]
    mu=mu/N
    return mu
def newSig(gam,N,data,n,mu):
    c=23
    sig=np.array([[0. for j in range(c)]for i in range(c)])
    for i in range(n):
        a=data[i]-mu
        a=np.reshape(a,(c,1))
        sig=sig+gam[i]*(a@a.T)
    sig=sig/N
    return sig
def GMM(MU,Sigma,Phi,K,data):
    Change=True
    itr=0
    n=len(data)
    while Change and itr<=2:
        Change=True
        ###################################ESTIMATION STEP#####################################################
        gam,N=estimationSTEP(MU,Sigma,Phi,K,data)
        #################################MAXIMAIZATION STEP###############################################
        cnt=0
        for j in range(K):
            copy = np.empty_like(MU[j])
            copy[:] = MU[j]
            MU[j]=newMean(gam[:,j],N[j],data,n)
            Sigma[j]=newSig(gam[:,j],N[j],data,n,MU[j])
            Phi[j]=N[j]/np.sum(N)
            if np.array_equal(copy,MU[j]):
                cnt+=1
        if cnt==K:
            Change=False
        itr+=1

    return MU,Sigma,Phi
def classifyGMM(MU,Sigmas,dev,K,Phis):
    SC=[0.,0.,0.,0.,0.]
    for x in dev:
        d1=np.sum(np.array([Phis[0][j]*gauss(x,MU[0][j],Sigmas[0][j]) for j in range(K)]))
        d2=np.sum(np.array([Phis[1][j]*gauss(x,MU[1][j],Sigmas[1][j]) for j in range(K)]))
        d3=np.sum(np.array([Phis[2][j] * gauss(x, MU[2][j], Sigmas[2][j]) for j in range(K)]))
        d4=np.sum(np.array([Phis[3][j] * gauss(x, MU[3][j], Sigmas[3][j]) for j in range(K)]))
        d5=np.sum(np.array([Phis[4][j]*gauss(x,MU[4][j],Sigmas[4][j]) for j in range(K)]))
        SC[0]+=mpmath.log(d1)
        SC[1]+=mpmath.log(d2)
        SC[2] += mpmath.log(d3)
        SC[3] += mpmath.log(d4)
        SC[4] += mpmath.log(d5)
    return SC
def plot_ROC_curve(scores, n,K):
    scores_mod = scores.flatten()
    scores_mod = np.sort(scores_mod)
    tpr = np.array([])
    fpr = np.array([])
    for threshold in scores_mod:
        tp = 0
        fp = 0
        tn = 0
        fn = 0
        for i in range(n):
            ground_truth = C_dev[i]
            for j in range(2):
                if scores[i][j] >= threshold:
                    if ground_truth == j + 1:
                        tp += 1
                    else:
                        fp += 1
                else:
                    if ground_truth == j + 1:
                        fn += 1
                    else:
                        tn += 1
        tpr = np.append(tpr, tp / (tp + fn))
        fpr = np.append(fpr, fp / (fp + tn))
    plt.plot(fpr,tpr,label="K="+str(K))
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve for different values of K")
    fnr=1-tpr
    return (fpr,fnr)

K=12

train=[]
train.append(extractData(directory1))
train.append(extractData(directory2))
train.append(extractData(directory3))
train.append(extractData(directory4))
train.append(extractData(directory5))
dev=[]
dev.append(extrctDataDev(directory6))
dev.append(extrctDataDev(directory7))
dev.append(extrctDataDev(directory8))
dev.append(extrctDataDev(directory9))
dev.append(extrctDataDev(directory10))
FPR=[]
FNR=[]
for K in [5,10,15,20]:
    print(" K VALUES IS "+str(K))
    MU=[]
    MU_diag=[]
    C_dev=[]
    for i in range(5):
        a,b=initialParam(train[i],K)
        MU.append(a)
        MU_diag.append(b)
    MU=[K_Means(MU[i],train[i],K) for i in range(5)]
    Clusters=[finalClusters(train[i],MU[i]) for i in range(5)]
    Sigmas=[finalSigmas(Clusters[i],MU[i]) for i  in range(5)]
    Phis=[finalPhis(Clusters[i]) for i in range(5)]
    print([[LA.det(Sigmas[i][j]) for j in range(K)]for i in range(5)])
    for i in range(5):
        print("hi")
        MU[i],Sigmas[i],Phis[i]=GMM(MU[i],Sigmas[i],Phis[i],K,train[i])
    print([[LA.det(Sigmas[i][j]) for j in range(K)]for i in range(5)])
    print("hello")
    scores=[]
    cnt=0
    num=0
    itr=0
    for i in range(5):
        d=dev[i]
        # print(np.array(d).shape)
        for p in d:
            itr += 1
            print(itr)
            C_dev.append(i+1)
            num+=1
            sc=classifyGMM(MU,Sigmas,p,K,Phis)
            if np.argmax(np.array(sc))==i:
                cnt+=1
            scores.append(sc)
    print(cnt,num)
    scores=np.array(scores)
    a,b=plot_ROC_curve(scores,len(C_dev),K)
    FPR.append(a)
    FNR.append(b)
plt.legend()
plt.show()

fig, ax_det = plt.subplots(1,1)
d1=DetCurveDisplay(fpr=FPR[0],fnr=FNR[0],estimator_name="K value is 5").plot(ax=ax_det)
d2=DetCurveDisplay(fpr=FPR[1],fnr=FNR[1],estimator_name="K value is 10").plot(ax=ax_det)
d3=DetCurveDisplay(fpr=FPR[2],fnr=FNR[2],estimator_name="K value is 15").plot(ax=ax_det)
d4=DetCurveDisplay(fpr=FPR[3],fnr=FNR[3],estimator_name="K value is 20").plot(ax=ax_det)
plt.show()
