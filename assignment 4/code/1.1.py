import math
import numpy as np
import os
import matplotlib.pyplot as plt
from numpy import linalg as LA
from sklearn.preprocessing import MinMaxScaler
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
            m.extend(x)
        data.append(m)
    return np.array(data)


def ROC(scores,actual):
    n=len(actual)
    sc=np.array(scores).flatten()
    sc=np.sort(sc)
    tpr=[]
    fpr=[]
    for threshold in sc:
        tp=0
        fp=0
        tn=0
        fn=0
        for i in range(n):
            ground_truth=actual[i]
            for j in range(5):
                if scores[i][j]>=threshold:
                    if ground_truth == j:
                        tp+=1
                    else:
                        fp+=1
                else:
                    if ground_truth==j:
                        fn+=1
                    else:
                        tn+=1
        tpr.append(tp/(tp+fn))
        fpr.append(fp/(fp+tn))
    # plt.plot(fpr,tpr)
    return (np.array(fpr),np.array(tpr))



train=[]
train.append(extractData(directory1))
train.append(extractData(directory2))
train.append(extractData(directory3))
train.append(extractData(directory4))
train.append(extractData(directory5))
dev=[]
dev.append(extractData(directory6))
dev.append(extractData(directory7))
dev.append(extractData(directory8))
dev.append(extractData(directory9))
dev.append(extractData(directory10))


r=[]
for i in range(5):
    r.extend(train[i])
scl=MinMaxScaler().fit(r)
for i in range(5):
    train[i]=scl.transform(train[i])
    dev[i]=scl.transform(dev[i])

def findMU(train):
    a=np.array(train[0])
    b,l=a.shape
    mu=np.array([0. for i in range(l)])
    cnt=0
    for i in range(5):
        for t in train[i]:
            mu=mu+np.array(t)
            cnt+=1
    mu=mu/cnt
    return mu


def findCls(dist,K):
    cnt=[0.,0.,0.,0.,0.]
    for i in range(K):
        cnt[dist[i][1]]+=1
    cnt=np.array(cnt)
    s=np.sum(cnt)
    return np.argmax(cnt),cnt

def meanNormalize(data,mu):
    res=[]
    for i in range(len(data)):
        res.append(np.array(data[i])-np.array(mu))
    return res



def findSig(train,mu):
    sig=np.zeros((len(train[0][0]),len(train[0][0])))
    for i in range(2):
        for d in train[i]:
            d1=np.array(np.array(d)-np.array(mu))
            d1=np.reshape(d1,(len(d1),1))
            sig+=d1@d1.T
    sig=sig/(len(train[0])+len(train[1]))
    return sig
def eigValSorter(m):
  return -abs(m)


FPR_K=[]
TPR_K=[]


def PCA(data,evec):
    res=[]
    c=len(evec[0])
    for d in data:
        temp=[]
        for i in range(c):
            temp.append(np.real(d@evec[:,i]))
        res.append(temp)
    return res

print("KNN accuracies: ")
for K in [15]:
    scores=[]
    actual=[]
    acc = np.array([0, 0, 0, 0, 0])
    tot = np.array([0, 0, 0, 0, 0])
    for i in range(5):
        for d in dev[i]:
            actual.append(i)
            dist=[]
            for j in range(5):
                for t in train[j]:
                    dist.append([math.dist(d,t),j])
            dist.sort()
            cls,sc=findCls(dist,K)
            sc=sc/sum(sc)
            scores.append(sc)
            if cls==i:
                acc[i]+=1
            tot[i]+=1
    print(K, np.sum(acc) * 100 / np.sum(tot))
    fpr,tpr=ROC(scores,actual)
    FPR_K.append(fpr)
    TPR_K.append(tpr)
    plt.plot(fpr,tpr,label="KNN")



print("PCA and KNN accuracies :")
mu=findMU(train)
sig2=findSig(train,mu)
train1=[meanNormalize(train[i],mu) for i in range(5)]
dev1=[meanNormalize(dev[i],mu) for i in range(5)]
mu1=findMU(train1)
sig1=findSig(train1,mu1)
K_array=[15]
L_array=np.array([i for i in range(60,201,10)])
ACC=np.array([])
eval, evec = LA.eig(sig1)
for K in K_array:
    scores=[]
    actual=[]
    L=80
    sortedIdx_order = np.argsort(eigValSorter(eval))
    eval = eval[sortedIdx_order]
    evec = evec[:, sortedIdx_order]
    evec = evec[:, :L]
    train2 = [PCA(train1[i], evec) for i in range(5)]
    dev2 = [PCA(dev1[i], evec) for i in range(5)]
    acc = np.array([0, 0, 0, 0, 0])
    tot = np.array([0, 0, 0, 0, 0])
    for i in range(5):
        for d in dev2[i]:
            dist=[]
            actual.append(i)
            for j in range(5):
                for t in train2[j]:
                    dist.append([math.dist(d,t),j])
            dist.sort()
            cls,sc=findCls(dist,K)
            sc = sc / sum(sc)
            scores.append(sc)
            if cls==i:
                acc[i]+=1
            tot[i]+=1

    print(K,np.sum(acc)*100/np.sum(tot))
    fpr,tpr=ROC(scores,actual)
    FPR_K.append(fpr)
    TPR_K.append(tpr)
    plt.plot(fpr,tpr,label="PCA  KNN")



def findMUi(train):
    mu=np.array([0. for i in range(len(train[0]))])
    cnt=0
    for t in train:
        mu=mu+t
        cnt+=1
    mu=mu/cnt
    return mu


print("PCA and LDA and KNN accuracies are :")

mu=findMU(train)
sig2=findSig(train,mu)
train1=[meanNormalize(train[i],mu) for i in range(5)]
dev1=[meanNormalize(dev[i],mu) for i in range(5)]
mu1=findMU(train1)
sig1=findSig(train1,mu1)
L=80
eval, evec = LA.eig(sig1)
sortedIdx_order = np.argsort(eigValSorter(eval))
eval = eval[sortedIdx_order]
evec = evec[:, sortedIdx_order]
evec = evec[:, :L]
train2 = [PCA(train1[i], evec) for i in range(5)]
dev2 = [PCA(dev1[i], evec) for i in range(5)]

mu=findMU(train2)
train3=[meanNormalize(train2[i],mu) for i in  range(5)]
dev3=[meanNormalize(dev2[i],mu) for i in  range(5)]
mus=np.array([findMUi(train3[i]) for i in range(5)])
muall=np.array(findMU(train3))
Sw=np.array([[0. for j in range(L)]for i in range(L)])
St=np.array([[0. for j in range(L)]for i in range(L)])

for i in range(5):
    for x in train3[i]:
        a=x-mus[i]
        b=x-muall[i]
        a=np.reshape(a,(L,1))
        b=np.reshape(b,(L,1))
        Sw+=a@a.T
        St+=b@b.T
Sb=St-Sw
mat=LA.inv(Sw)@Sb

eval,evec=LA.eig(mat)
L=4
for K in [15]:
    scores=[]
    actual=[]
    sortedIdx_order = np.argsort(eigValSorter(eval))
    eval1 = eval[sortedIdx_order]
    evec1 = evec[:, sortedIdx_order]
    evec1 = evec1[:, :L]
    train4 = [PCA(train3[i], evec1) for i in range(5)]
    dev4 = [PCA(dev3[i], evec1) for i in range(5)]
    acc = np.array([0, 0, 0, 0, 0])
    tot = np.array([0, 0, 0, 0, 0])
    for i in range(5):
        for d in dev4[i]:
            actual.append(i)
            dist=[]
            for j in range(5):
                for t in train4[j]:
                    dist.append([math.dist(d,t),j])
            dist.sort()
            cls,sc=findCls(dist,K)
            sc = sc / sum(sc)
            scores.append(sc)
            if cls==i:
                acc[i]+=1
            tot[i]+=1
    print(K,np.sum(acc)*100/np.sum(tot))
    fpr,tpr=ROC(scores,actual)
    FPR_K.append(fpr)
    TPR_K.append(tpr)
    plt.plot(fpr,tpr,label="PCA LDA KNN")
plt.legend()
plt.show()


estimate=["KNN","PCA KNN","PCA LDA KNN"]
fig, ax_det = plt.subplots(1,1)
for i in range(len(FPR_K)):
    print(FPR_K[i])
    print(TPR_K[i])
    d1 = DetCurveDisplay(fpr=np.array(FPR_K[i]), fnr=1-np.array(TPR_K[i]),estimator_name=estimate[i]).plot(ax=ax_det)
plt.show()








#################################################################################################################

def calculateProbability(p,numcls,w):
    aK_s=[]
    for i in range(numcls):
        P=[1.]
        for t in p:
            P.append(t)
        aK_s.append(w[i].T@P)
    logit=[math.exp(a) for a in aK_s]
    s=np.sum(np.array(logit))
    for i in range(len(logit)):
        logit[i]=logit[i]/s
    return logit


print("Logistic regression accuracies are : ")
m=828
iter=300
eta=10**(-5)
for i in range(5):
    print(len(train[i]))
w=np.zeros((len(train),m+1))
sum=[0,0,0,0,0]
for i in range(1,5):
    sum[i]=len(train[i-1])+sum[i-1]
print(sum)
ijdis=0
for _ in range(iter):
    print(ijdis)
    ijdis +=1
    probabilities=[]
    for i in range(5):
        for p in train[i]:
            probabilities.append(calculateProbability(p,len(train),w))
    en=np.zeros((len(train),m+1))
    for idx in range(5):
        for j in range(len(train[idx])):
            p=train[idx][j]
            P = [1.]
            for t in p:
                P.append(t)
            P=np.array(P)
            for i in range(5):
                if i==idx:
                    # en[i]+=1
                    en[i]+=P*(probabilities[j+sum[idx]][i]-1.)
                else:
                    # print(j+idx*len(train[idx]))
                    # en[i]+=1
                    en[i]+=P*(probabilities[j+sum[idx]][i])
    # print("w_j is :" + str(en))
    w=w-eta*en
scores=[]
FPR_L=[]
TPR_L=[]
accuracy=0.
true=0.
total=0.
actual=[]
for i in range(5):
    for d in dev[i]:
        score=calculateProbability(d,len(dev),w)
        if score.index(max(score))==i:
            true+=1
        total+=1
        scores.append(score)
        actual.append(i)
fpr,tpr=ROC(scores,actual)
FPR_L.append(fpr)
TPR_L.append(tpr)
plt.plot(fpr,tpr,label="Logistic")
accuracy=true/total
print("accuracy of the model is : ",accuracy*100)





print("Logistic Regression and PCA accuracies : ")
mu=findMU(train)
sig2=findSig(train,mu)
train1=[meanNormalize(train[i],mu) for i in range(5)]
dev1=[meanNormalize(dev[i],mu) for i in range(5)]
mu1=findMU(train1)
sig1=findSig(train1,mu1)
L=35
eval, evec = LA.eig(sig1)
sortedIdx_order = np.argsort(eigValSorter(eval))
eval = eval[sortedIdx_order]
evec = evec[:, sortedIdx_order]
evec = evec[:, :L]
train2 = [PCA(train1[i], evec) for i in range(5)]
dev2 = [PCA(dev1[i], evec) for i in range(5)]
m=L
iter=1000
eta=10**(-4.2)
for i in range(5):
    print(len(train2[i]))
w=np.zeros((len(train2),m+1))
sum=[0,0,0,0,0]
for i in range(1,5):
    sum[i]=len(train2[i-1])+sum[i-1]
print(sum)
ijdis=0
for _ in range(iter):
    ijdis +=1
    probabilities=[]
    for i in range(5):
        for p in train2[i]:
            probabilities.append(calculateProbability(p,len(train2),w))
    en=np.zeros((len(train2),m+1))
    for idx in range(5):
        for j in range(len(train2[idx])):
            p=train2[idx][j]
            P = [1.]
            for t in p:
                P.append(t)
            P=np.array(P)
            for i in range(5):
                if i==idx:
                    # en[i]+=1
                    en[i]+=P*(probabilities[j+sum[idx]][i]-1.)
                else:
                    # print(j+idx*len(train[idx]))
                    # en[i]+=1
                    en[i]+=P*(probabilities[j+sum[idx]][i])
    # print("w_j is :" + str(en))
    w=w-eta*en
scores=[]
accuracy=0.
true=0.
total=0.
actual=[]
for i in range(5):
    for d in dev2[i]:
        score=calculateProbability(d,len(dev2),w)
        if score.index(max(score))==i:
            true+=1
        total+=1
        scores.append(score)
        actual.append(i)
fpr,tpr=ROC(scores,actual)
FPR_L.append(fpr)
TPR_L.append(tpr)
plt.plot(fpr,tpr,label="PCA Logistic")
accuracy=true/total
print("accuracy of the model is : ",accuracy*100)






print("Logistic Regression and LDA and PCA accuracies : ")
mu=findMU(train)
sig2=findSig(train,mu)
train1=[meanNormalize(train[i],mu) for i in range(5)]
dev1=[meanNormalize(dev[i],mu) for i in range(5)]
mu1=findMU(train1)
sig1=findSig(train1,mu1)
L=35
eval, evec = LA.eig(sig1)
sortedIdx_order = np.argsort(eigValSorter(eval))
eval = eval[sortedIdx_order]
evec = evec[:, sortedIdx_order]
evec = evec[:, :L]
train2 = [PCA(train1[i], evec) for i in range(5)]
dev2 = [PCA(dev1[i], evec) for i in range(5)]
mu=findMU(train2)
train3=[meanNormalize(train2[i],mu) for i in  range(5)]
dev3=[meanNormalize(dev2[i],mu) for i in  range(5)]
mus=np.array([findMUi(train3[i]) for i in range(5)])
muall=np.array(findMU(train3))
Sw=np.array([[0. for j in range(L)]for i in range(L)])
St=np.array([[0. for j in range(L)]for i in range(L)])

for i in range(5):
    for x in train3[i]:
        a=x-mus[i]
        b=x-muall[i]
        a=np.reshape(a,(L,1))
        b=np.reshape(b,(L,1))
        Sw+=a@a.T
        St+=b@b.T
Sb=St-Sw
mat=LA.inv(Sw)@Sb
eval,evec=LA.eig(mat)
L=4
evec=evec[:,:L]
train4=[PCA(train3[i],evec) for i in range(5)]
dev4=[PCA(dev3[i],evec)for i in range(5)]
m=L
iter=1000
eta=10**(-4.2)
for i in range(5):
    print(len(train4[i]))
w=np.zeros((len(train4),m+1))
sum=[0,0,0,0,0]
for i in range(1,5):
    sum[i]=len(train4[i-1])+sum[i-1]
print(sum)
ijdis=0
for _ in range(iter):
    ijdis +=1
    probabilities=[]
    for i in range(5):
        for p in train4[i]:
            probabilities.append(calculateProbability(p,len(train4),w))
    en=np.zeros((len(train4),m+1))
    for idx in range(5):
        for j in range(len(train4[idx])):
            p=train4[idx][j]
            P = [1.]
            for t in p:
                P.append(t)
            P=np.array(P)
            for i in range(5):
                if i==idx:
                    # en[i]+=1
                    en[i]+=P*(probabilities[j+sum[idx]][i]-1.)
                else:
                    # print(j+idx*len(train[idx]))
                    # en[i]+=1
                    en[i]+=P*(probabilities[j+sum[idx]][i])
    # print("w_j is :" + str(en))
    w=w-eta*en
scores=[]
accuracy=0.
true=0.
total=0.
actual=[]
for i in range(5):
    for d in dev4[i]:
        score=calculateProbability(d,len(dev4),w)
        if score.index(max(score))==i:
            true+=1
        total+=1
        scores.append(score)
        actual.append(i)
fpr,tpr=ROC(scores,actual)
FPR_L.append(fpr)
TPR_L.append(tpr)
plt.plot(fpr,tpr,label="PCA LDA Logistic")

accuracy=true/total
print("accuracy of the model is : ",accuracy*100)
plt.legend()
plt.show()


estimate=["Logistic","PCA Logistic","PCA LDA Logistic"]
fig, ax_det = plt.subplots(1,1)
for i in range(len(FPR_L)):
    print(FPR_L[i])
    print(TPR_L[i])
    d1 = DetCurveDisplay(fpr=np.array(FPR_L[i]), fnr=1-np.array(TPR_L[i]),estimator_name=estimate[i]).plot(ax=ax_det)
plt.show()
