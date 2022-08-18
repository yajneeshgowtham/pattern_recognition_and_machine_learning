import math
import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as LA
from sklearn.metrics import DetCurveDisplay


def extractData(filename):
    f = open(filename, "r")
    c1 = []
    c2 = []
    for line in f.readlines():
        a, b, c = line.strip().split(',')
        a = float(a)
        b = float(b)
        if c =="1":
            c1.append([a, b])
        else:
            c2.append([a, b])
    data = []
    data.append(c1)
    data.append(c2)
    return np.array(data)
def findMU(train):
    a=np.array(train[0])
    b,l=a.shape
    mu=np.array([0. for i in range(l)])
    cnt=0
    for i in range(2):
        for t in train[i]:
            mu=mu+np.array(t)
            cnt+=1
    mu=mu/cnt
    return mu
def findCls(dist,K):
    cnt=[0,0]
    for i in range(K):
        cnt[dist[i][1]]+=1
    cnt=np.array(cnt)
    return (np.argmax(cnt),cnt)
def meanNormalize(data,mu):
    for i in range(len(data)):
        data[i]=np.array(data[i])-np.array(mu)
    return data
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

# def LDA_2D()


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
            for j in range(2):
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
    return (fpr,tpr)


train = extractData("train.txt")
dev = extractData("dev.txt")


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


##############################KMEANS###############################################
estimate=[15]
print("KNN accuracies: ")
for K in estimate:
    scores=[]
    actual=[]
    acc = np.array([0, 0, 0, 0, 0])
    tot = np.array([0, 0, 0, 0, 0])
    for i in range(2):
        for d in dev[i]:
            actual.append(i)
            dist=[]
            for j in range(2):
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
# plt.legend()
# plt.show()
##########################################################################################

# fig, ax_det = plt.subplots(1,1)
# for i in range(len(FPR_K)):
#     d1 = DetCurveDisplay(fpr=np.array(FPR_K[i]), fnr=1-np.array(TPR_K[i]),estimator_name=estimate[i]).plot(ax=ax_det)
# plt.show()



print("PCA and KNN accuracies :")
mu=findMU(train)
sig2=findSig(train,mu)
train1=[meanNormalize(train[i],mu) for i in range(2)]
dev1=[meanNormalize(dev[i],mu) for i in range(2)]
mu1=findMU(train1)
sig1=findSig(train1,mu1)
K_array=[15]
L_array=np.array([i for i in range(60,201,10)])
ACC=np.array([])
eval, evec = LA.eig(sig1)
for K in K_array:
    scores=[]
    actual=[]
    L=1
    sortedIdx_order = np.argsort(eigValSorter(eval))
    eval = eval[sortedIdx_order]
    evec = evec[:, sortedIdx_order]
    evec = evec[:, :L]
    train2 = [PCA(train1[i], evec) for i in range(2)]
    dev2 = [PCA(dev1[i], evec) for i in range(2)]
    acc = np.array([0, 0])
    tot = np.array([0, 0])
    for i in range(2):
        for d in dev2[i]:
            dist=[]
            actual.append(i)
            for j in range(2):
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
L=2
mu=findMU(train)
train3=[meanNormalize(train[i],mu) for i in  range(2)]
dev3=[meanNormalize(dev[i],mu) for i in  range(2)]
mus=np.array([findMUi(train3[i]) for i in range(2)])
muall=np.array(findMU(train3))
Sw=np.array([[0. for j in range(L)]for i in range(L)])
St=np.array([[0. for j in range(L)]for i in range(L)])

for i in range(2):
    for x in train3[i]:
        a=x-mus[i]
        b=x-muall[i]
        a=np.reshape(a,(L,1))
        b=np.reshape(b,(L,1))
        Sw+=a@a.T
        St+=b@b.T
Sb=St-Sw
mat=LA.inv(Sw)@Sb
print("mat is : ",mat)
eval,evec=LA.eig(mat)
L=1
for K in [15]:
    scores=[]
    actual=[]
    sortedIdx_order = np.argsort(eigValSorter(eval))
    eval1 = eval[sortedIdx_order]
    evec1 = evec[:, sortedIdx_order]
    evec1 = evec1[:, :L]
    train4 = [PCA(train3[i], evec1) for i in range(2)]
    dev4 = [PCA(dev3[i], evec1) for i in range(2)]
    acc = np.array([0, 0])
    tot = np.array([0, 0])
    for i in range(2):
        for d in dev4[i]:
            actual.append(i)
            dist=[]
            for j in range(2):
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
    d1 = DetCurveDisplay(fpr=np.array(FPR_K[i]), fnr=1-np.array(TPR_K[i]),estimator_name=estimate[i]).plot(ax=ax_det)
plt.show()


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
m=2
iter=100
eta=10**(-5)
for i in range(2):
    print(len(train[i]))
w=np.zeros((len(train),m+1))
sum=[0,0]
for i in range(1,2):
    sum[i]=len(train[i-1])+sum[i-1]
print(sum)
ijdis=0
for _ in range(iter):
    ijdis +=1
    probabilities=[]
    for i in range(2):
        for p in train[i]:
            probabilities.append(calculateProbability(p,len(train),w))
    en=np.zeros((len(train),m+1))
    for idx in range(2):
        for j in range(len(train[idx])):
            p=train[idx][j]
            P = [1.]
            for t in p:
                P.append(t)
            P=np.array(P)
            for i in range(2):
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
for i in range(2):
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
plt.show()
fig, ax_det = plt.subplots(1,1)
d1 = DetCurveDisplay(fpr=np.array(FPR_L[0]), fnr=1-np.array(TPR_L[0]),estimator_name=estimate[i]).plot(ax=ax_det)
plt.show()


print("Logistic Regression and PCA accuracies : ")
L=2
mu=findMU(train)
sig2=findSig(train,mu)
train1=[meanNormalize(train[i],mu) for i in range(2)]
dev1=[meanNormalize(dev[i],mu) for i in range(2)]
mu1=findMU(train1)
sig1=findSig(train1,mu1)
eval, evec = LA.eig(sig1)
sortedIdx_order = np.argsort(eigValSorter(eval))
eval = eval[sortedIdx_order]
evec = evec[:, sortedIdx_order]
evec = evec[:, :L]
mu=findMU(train)
sig1=findSig(train,mu)
train2 = [PCA(train[i], evec) for i in range(2)]
dev2 = [PCA(dev[i], evec) for i in range(2)]
m=L
iter=500
eta=10**(-5)
for i in range(2):
    print(len(train2[i]))
w=np.zeros((len(train2),m+1))
sum=[0,0]
for i in range(1,2):
    sum[i]=len(train2[i-1])+sum[i-1]
ijdis=0
for _ in range(iter):
    ijdis +=1
    probabilities=[]
    for i in range(2):
        for p in train2[i]:
            probabilities.append(calculateProbability(p,len(train2),w))
    en=np.zeros((len(train2),m+1))
    for idx in range(2):
        for j in range(len(train2[idx])):
            p=train2[idx][j]
            P = [1.]
            for t in p:
                P.append(t)
            P=np.array(P)
            for i in range(2):
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
for i in range(2):
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
