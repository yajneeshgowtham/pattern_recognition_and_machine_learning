import numpy as np
import numpy.linalg as LA
import seaborn as sn
import matplotlib.pyplot as plt
from sklearn.svm import SVC
import os
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import DetCurveDisplay
from pca import PCA

train=['coast/train','forest/train','highway/train','mountain/train','opencountry/train']
dev=['coast/dev','forest/dev','highway/dev','mountain/dev','opencountry/dev']


def plot_ROC_curve(scores, C_dev, n,clusters):
    scores_mod = scores.flatten()
    scores_mod = np.sort(scores_mod)
    tpr = np.array([])
    fpr = np.array([])
    fnr = np.array([])
    for threshold in scores_mod:
        tp = 0
        fp = 0
        tn = 0
        fn = 0
        for i in range(n):
            ground_truth = C_dev[i]
            for j in range(5):
                if (scores[i][j] >= threshold):
                    if ground_truth == j:
                        tp += 1
                    else:
                        fp += 1
                else:
                    if ground_truth == j:
                        fn += 1
                    else:
                        tn += 1
        tpr = np.append(tpr, tp / (tp + fn))
        fpr = np.append(fpr, fp / (fp + tn))
    plt.plot(fpr,tpr,label=clusters)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    return ([tpr,fpr])

def SVM(trn_data,dev_data,lab_train,lab_dev,st):
    mdl=SVC(kernel="rbf",C=20,probability=True)

    scl=MinMaxScaler().fit(trn_data)
    trn_scaled=scl.transform(trn_data)
    dev_scaled=scl.transform(dev_data)
    mdl.fit(trn_scaled,lab_train)
    prd=mdl.predict(dev_scaled)
    prd=list(prd)
    scores=mdl.predict_proba(dev_scaled)
    l=plot_ROC_curve(scores,lab_dev,len(lab_dev),st)
    return l

def det(tpr,fpr,label):
    d1 = DetCurveDisplay(fpr=fpr, fnr=1 - tpr, estimator_name=label).plot(ax=ax_det)

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
    return data

lab_train=[]
lab_dev=[]
trn_data=[]
dev_data=[]
arr1=[]
arr2=[]
tpr=[]
fpr=[]
label=[]

for i in range(5):
    l=extractData(train[i])
    trn_data.extend(l)
    arr1.append(np.array(l))
    for j in range(len(l)):
        lab_train.append(i)

for i in range(5):
    l=extractData(dev[i])
    dev_data.extend(l)
    arr2.append(np.array(l))
    for j in range(len(l)):
        lab_dev.append(i)

l=SVM(trn_data,dev_data,lab_train,lab_dev,"original")
tpr.append(l[0])
fpr.append(l[1])
label.append("original")

# def PCA(data,evec):
#     res=[]
#     c=len(evec[0])
#     for d in data:
#         temp=[]
#         for i in range(c):
#             temp.append(np.real(d@evec[:,i]))
#         res.append(temp)
#     return res
#
# def meanNormalize(data,mu):
#     res=[]
#     for i in range(len(data)):
#         res.append(np.array(data[i])-np.array(mu))
#     return res
#
# def findMU(train):
#     a=np.array(train[0])
#     b,l=a.shape
#     mu=np.array([0. for i in range(l)])
#     cnt=0
#     for i in range(5):
#         for t in train[i]:
#             mu=mu+np.array(t)
#             cnt+=1
#     mu=mu/cnt
#     return mu
#
#
# def findSig(train,mu):
#     sig=np.zeros((len(train[0][0]),len(train[0][0])))
#     for i in range(5):
#         for d in train[i]:
#             d1=np.array(np.array(d)-np.array(mu))
#             d1=np.reshape(d1,(len(d1),1))
#             sig+=d1@d1.T
#     sig=sig/(len(train[0])+len(train[1]))
#     return sig
#
# def eigValSorter(m):
#   return -abs(m)
#
# mu=findMU(arr1)
# train0=[meanNormalize(arr1[i],mu) for i in range(5)]
# dev0=[meanNormalize(arr2[i],mu) for i in range(5)]
# mu1=findMU(train0)
# sig1=findSig(train0,mu1)
# L=120
# eval, evec = LA.eig(sig1)
# sortedIdx_order = np.argsort(eigValSorter(eval))
# eval = eval[sortedIdx_order]
# evec = evec[:, sortedIdx_order]
# evec = evec[:, :L]
# train1 = [PCA(train0[i], evec) for i in range(5)]
# dev1 = [PCA(dev0[i], evec) for i in range(5)]
#
# lab_train=[]
# lab_dev=[]
# trn_data=[]
# dev_data=[]
# for i in range(5):
#     trn_data.extend(train1[i])
#     dev_data.extend(dev1[i])
#     for j in range(len(train1[i])):
#         lab_train.append(i)
#     for j in range(len(dev1[i])):
#         lab_dev.append(i)
#
# # print(len(trn_data),len(lab_train),len(dev_data),len(lab_dev))
# l=SVM(trn_data,dev_data,lab_train,lab_dev,"with PCA")
# tpr.append(l[0])
# fpr.append(l[1])
# label.append("with PCA")
# # #########################################################PCA over,LDA start
#
# def findMUi(train):
#     mu=np.array([0. for i in range(len(train[0]))])
#     cnt=0
#     for t in train:
#         mu=mu+t
#         cnt+=1
#     mu=mu/cnt
#     return mu
#
# mus=np.array([findMUi(train1[i]) for i in range(5)])
# muall=np.array(findMU(train1))
# Sw=np.array([[0. for j in range(120)]for i in range(120)])
# St=np.array([[0. for j in range(120)]for i in range(120)])
#
# for i in range(5):
#     for x in train1[i]:
#         a=x-mus[i]
#         b=x-muall[i]
#         a=np.reshape(a,(120,1))
#         b=np.reshape(b,(120,1))
#         Sw+=a@a.T
#         St+=b@b.T
# Sb=St-Sw
# mat=LA.inv(Sw)@Sb
# eval,evec=LA.eig(mat)
# L=4
#
# evec=evec[:,:L]
# train2=[PCA(train1[i],evec) for i in range(5)]
# dev2=[PCA(dev1[i],evec)for i in range(5)]
#
# trn_data=[]
# dev_data=[]
# lab_train=[]
# lab_dev=[]
#
# for i in range(5):
#     trn_data.extend(train2[i])
#     dev_data.extend(dev2[i])
#     for j in range(len(train2[i])):
#         lab_train.append(i)
#     for j in range(len(dev2[i])):
#         lab_dev.append(i)
#
# mini=min(trn_data)
# maxi=max(trn_data)
# mini_d=min(dev_data)
# maxi_d=max(dev_data)
#
# for i in range(len(trn_data)):
#     l=trn_data[i]
#     for j in range(len(l)):
#         if maxi[j]!=mini[j]:
#             trn_data[i][j]=(trn_data[i][j]-mini[j])/(maxi[j]-mini[j])
#
# for i in range(len(dev_data)):
#     l=dev_data[i]
#     for j in range(len(l)):
#         if maxi_d[j]!=mini_d[j]:
#             dev_data[i][j]=(dev_data[i][j]-mini_d[j])/(maxi_d[j]-mini_d[j])
#
# l=SVM(trn_data,dev_data,lab_train,lab_dev,"with LDA")
# tpr.append(l[0])
# fpr.append(l[1])
# label.append("with LDA")
plt.legend()
plt.show()

fig, ax_det = plt.subplots(1, 1)
for i in range(1):
    det(tpr[i], fpr[i], label[i])
plt.show()

# mdl=SVC(kernel="rbf",C=20,probability=True)
#
# scl=MinMaxScaler().fit(trn_data)
# trn_scaled=scl.transform(trn_data)
# dev_scaled=scl.transform(dev_data)
# mdl.fit(trn_scaled,lab_train)
# prd=mdl.predict(dev_scaled)
# prd=list(prd)
# scores=scl.predict_proba(dev_scaled)
# plot_ROC_curve(scores,lab_dev,st)
#
# n=len(prd)
# x=0
# for i in range(n):
#     if prd[i]==lab_dev[i]:
#         x+=1

# def draw_confusion(res1,C_dev):
#     conf= np.array([[0 for j in range(5)] for i in range(5)])
#     for i in range(np.size(res1)):
#         conf[C_dev[i]-1][res1[i]-1] += 1
#     df_cm = pd.DataFrame(conf)
#     as1 = sn.heatmap(df_cm, annot=True,fmt=".1f")
#     as1.set_xlabel('true class')
#     as1.set_ylabel('predicted class')
#
# draw_confusion(prd,lab_dev)
# plt.show()

# print("Accuracy is : ",x/n * 100)
