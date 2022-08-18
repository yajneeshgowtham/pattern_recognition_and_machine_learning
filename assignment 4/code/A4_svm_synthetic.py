import numpy as np
import seaborn as sn
import matplotlib.pyplot as plt
from sklearn.svm import SVC
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import os
from sklearn.preprocessing import MinMaxScaler
import numpy.linalg as LA
from sklearn.metrics import DetCurveDisplay

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
            for j in range(2):
                if (scores[i][j] >= threshold):
                    if ground_truth == j+1:
                        tp += 1
                    else:
                        fp += 1
                else:
                    if ground_truth == j+1:
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
    l = plot_ROC_curve(scores, lab_dev, len(lab_dev), st)
    return l

def det(tpr,fpr,label):
    d1 = DetCurveDisplay(fpr=fpr, fnr=1 - tpr, estimator_name=label).plot(ax=ax_det)

def extractData(filename):
    f = open(filename, "r")
    c1 = []
    c2 = []
    c0=[]
    c3=[]
    for line in f.readlines():
        a, b, c = line.strip().split(',')
        a = float(a)
        b = float(b)
        c = int(c)
        if c ==1:
            c1.append([a, b])
            c0.append([a, b,c])
        else:
            c2.append([a, b])
            c3.append([a, b,c])
    data = []
    data.extend(c0)
    data.extend(c3)
    return np.array(c1),np.array(c2),data

c1_trn,c2_trn,train_data=extractData("25/train.txt")
c1_dev,c2_dev,dev_data=extractData("25/dev.txt")
lab_train=[]
lab_dev=[]
arr1=[]
arr2=[]
arr1.append(c1_trn)
arr1.append(c2_trn)
arr2.append(c1_dev)
arr2.append(c2_dev)
c1t=list(c1_trn)
c2t=list(c2_trn)
c1d=list(c1_dev)
c2d=list(c2_dev)
ct=[]
cd=[]
ct.extend(c1t)
ct.extend(c2t)
cd.extend(c1d)
cd.extend(c2d)
tpr=[]
fpr=[]
label=[]

for i in range(len(train_data)):
    lab_train.append(train_data[i][2])

for i in range(len(dev_data)):
    lab_dev.append(dev_data[i][2])

l=SVM(train_data,dev_data,lab_train,lab_dev,"original")
tpr.append(l[0])
fpr.append(l[1])
label.append("original")
#
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
#     for i in range(2):
#         for t in train[i]:
#             mu=mu+np.array(t)
#             cnt+=1
#     mu=mu/cnt
#     return mu
#
#
# def findSig(train,mu):
#     sig=np.zeros((len(train[0][0]),len(train[0][0])))
#     for i in range(2):
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
# train0=[meanNormalize(arr1[i],mu) for i in range(2)]
# dev0=[meanNormalize(arr2[i],mu) for i in range(2)]
# mu1=findMU(train0)
# sig1=findSig(train0,mu1)
# L=120
# eval, evec = LA.eig(sig1)
# sortedIdx_order = np.argsort(eigValSorter(eval))
# eval = eval[sortedIdx_order]
# evec = evec[:, sortedIdx_order]
# evec = evec[:, :L]
# train1 = [PCA(train0[i], evec) for i in range(2)]
# dev1 = [PCA(dev0[i], evec) for i in range(2)]
#
# lab_train=[]
# lab_dev=[]
# trn_data=[]
# dev_data=[]
# for i in range(2):
#     trn_data.extend(train1[i])
#     dev_data.extend(dev1[i])
#     for j in range(len(train1[i])):
#         lab_train.append(i)
#     for j in range(len(dev1[i])):
#         lab_dev.append(i)
#
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
# mus=np.array([findMUi(train1[i]) for i in range(2)])
# muall=np.array(findMU(train1))
# Sw=np.array([[0. for j in range(2)]for i in range(2)])
# St=np.array([[0. for j in range(2)]for i in range(2)])
#
# for i in range(2):
#     for x in train1[i]:
#         a=x-mus[i]
#         b=x-muall[i]
#         a=np.reshape(a,(2,1))
#         b=np.reshape(b,(2,1))
#         Sw+=a@a.T
#         St+=b@b.T
# Sb=St-Sw
# mat=LA.inv(Sw)@Sb
# eval,evec=LA.eig(mat)
# L=4
#
# evec=evec[:,:L]
# train2=[PCA(train1[i],evec) for i in range(2)]
# dev2=[PCA(dev1[i],evec)for i in range(2)]
#
# trn_data=[]
# dev_data=[]
# lab_train=[]
# lab_dev=[]
#
# for i in range(2):
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
for i in range(3):
    det(tpr[i], fpr[i], label[i])
plt.show()
# scl=MinMaxScaler().fit(train_data)
# trn_scaled=scl.transform(train_data)
# dev_scaled=scl.transform(dev_data)
#
# clf = MLPClassifier(random_state=10, max_iter=5000).fit(trn_scaled, lab_train)
# prediction = clf.predict(dev_scaled)
# # proba = clf.predict_proba(x_test)
# # print(proba[1])
# prd = list(prediction)
#
# n=len(prd)
# x=0
# for i in range(n):
#     if prd[i]==lab_dev[i]:
#         x+=1
#
# def draw_confusion(res1,C_dev):
#     conf= np.array([[0 for j in range(2)] for i in range(2)])
#     for i in range(np.size(res1)):
#         conf[C_dev[i]-1][res1[i]-1] += 1
#     df_cm = pd.DataFrame(conf)
#     as1 = sn.heatmap(df_cm, annot=True,fmt=".1f")
#     as1.set_xlabel('true class')
#     as1.set_ylabel('predicted class')
#
# draw_confusion(prd,lab_dev)
# plt.show()
#
# print("Accuracy is : ",x/n * 100)