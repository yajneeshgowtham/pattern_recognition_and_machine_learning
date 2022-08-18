import numpy as np
import math
from numpy import linalg as LA
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sn
from scipy.stats import multivariate_normal
from sklearn.metrics import DetCurveDisplay
import random
import os
from sklearn.cluster import KMeans

trn=["2/train","3/train","4/train","6/train","o/train"]
dev=["2/dev","3/dev","4/dev","6/dev","o/dev"]

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
    plt.plot(fpr,tpr,label=str(clusters))
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    # plt.title("ROC Curve for clusters :"+str(clusters))
    # plt.show()

    # fig, ax_det = plt.subplots(1, 1)
    # d1 = DetCurveDisplay(fpr=fpr, fnr=1 - tpr, estimator_name="DTW").plot(ax=ax_det)
    # plt.show()
    return ([tpr,fpr])

def det(tpr,fpr,label):
    d1 = DetCurveDisplay(fpr=fpr, fnr=1 - tpr, estimator_name=str(label)).plot(ax=ax_det)
    # plt.show()


def draw_confusion(res1,C_dev):
    conf= np.array([[0 for j in range(5)] for i in range(5)])
    for i in range(np.size(res1)):
        conf[C_dev[i]-1][res1[i]-1] += 1
    df_cm = pd.DataFrame(conf)
    as1 = sn.heatmap(df_cm, annot=True,fmt=".1f")
    as1.set_xlabel('true class')
    as1.set_ylabel('predicted class')

def extractData(directory):
    data=[]
    for filename in os.listdir(directory):
        (filename,extension) = os.path.splitext(filename)
        if extension == '.mfcc':
            f=open(directory+'/'+filename+extension,"r")
            lines = f.readlines()
            for line in lines[1:]:
                x=line.split()
                x = list(map(float, x))
                data+=[x]
    return data

def extractData1(directory):
    data=[]
    for filename in os.listdir(directory):
        (filename,extension) = os.path.splitext(filename)
        if extension == '.mfcc':
            f=open(directory+'/'+filename+extension,"r")
            m=[]
            lines = f.readlines()
            for line in lines[1:]:
                x=line.split()
                x = list(map(float, x))
                m.append(x)
            data.append(np.array(m))
    return data

def probab(same,same_prob,next,next_prob,arr):
    n=len(arr)
    state=len(same)
    alp=np.zeros(n+1)
    alp[0]=1
    for i in range(n):
        nalp=np.zeros(n+1)
        for j in range(state):
            nalp[j]+=alp[j]*same[j]*same_prob[j][arr[i]]
            if j>0:
                nalp[j]+=alp[j-1]*next[j-1]*next_prob[j-1][arr[i]]
        alp,nalp=nalp,alp
    return np.sum(alp)

trn_data=[]
dev_data=[extractData1(dev[i]) for i in range(5)]
for i in range(5):
    l=extractData(trn[i])
    trn_data+=l
# print(trn_data)
clusters=15
dt=np.array(trn_data)
tpr=[]
fpr=[]

lab=KMeans(n_clusters=clusters,max_iter=50,random_state=0).fit(dt)
labels=lab.labels_
# print(len(labels))
state=[3,4,5,6,10]
for states in state:
    k = 0
    same_=[[],[],[],[],[]]
    next_=[[],[],[],[],[]]
    same_prob_=[[],[],[],[],[]]
    next_prob_=[[],[],[],[],[]]
    for i in range(5):
        dir=trn[i]
        file_ = open("clusters.seq", "w")
        for filename in os.listdir(dir):
            (filename, extension) = os.path.splitext(filename)
            if extension == '.mfcc':
                f = open(dir + '/' + filename + extension, "r")
                lines = f.readlines()
                n=len(lines)
                n-=1
                st=""
                for j in range(n-1):
                    st+=str(labels[k])+" "
                    k+=1
                st+=str(labels[k])
                k+=1
                file_.write(st+"\n")
        file_.close()
        cmd="./train_hmm clusters.seq 100000 " + str(states) + " " + str(clusters) + " 0.001"
        # print(cmd)
        os.system(cmd)
        file_=open("clusters.seq.hmm","r")
        lines=file_.readlines()
        same=[]
        next=[]
        same_prob=[]
        next_prob=[]
        n=len(lines)
        j=2
        p=0
        while j<n:
            s=lines[j]
            l=s.split()
            l=list(map(float,l))
            same.append(l[0])
            same_prob.append(l[1:])
            j+=1
            s = lines[j]
            l = s.split()
            l = list(map(float, l))
            next.append(l[0])
            next_prob.append(l[1:])
            j+=2
            p+=1
            if p==states:
                break
        same_[i]=same
        next_[i]=next
        same_prob_[i]=same_prob
        next_prob_[i]=next_prob
        # print(same)
        # print(next)
        # print(same_prob)
        # print(next_prob)

    scores=[]
    C_dev=[]
    C_dev_found=[]
    acc=np.array([0 for i in range(5)])
    cnt=np.array([0 for i in range(5)])
    conf=np.array([[0 for i in range(5)] for j in range(5)])
    for i in range(5):
        l=dev_data[i]
        for j in range(len(l)):
            mat=l[j]
            arr=lab.predict(mat)
            Min=[]
            for k in range(5):
                x=probab(same_[k],same_prob_[k],next_[k],next_prob_[k],arr)
                Min.append(x)
            cls=Min.index(max(Min))
            conf[cls][i]+=1
            scores.append(Min)
            if cls==i:
                acc[i]+=1
            cnt[i]+=1
            C_dev.append(i+1)
            C_dev_found.append(cls+1)
    #     print("accuracy:", acc[i], cnt[i])
    # print("final accuracy:", np.sum(acc) * 100 / np.sum(cnt))

    l = plot_ROC_curve(np.array(scores), C_dev, len(C_dev), clusters)
    tpr.append(l[0])
    fpr.append(l[1])
plt.title("ROC Curve for state values :" + str(state))
plt.legend()
plt.show()
fig, ax_det = plt.subplots(1, 1)
for states in range(len(state)):
    det(tpr[states], fpr[states], state[states])
plt.show()
# draw_confusion(C_dev_found,C_dev)
# plt.show()