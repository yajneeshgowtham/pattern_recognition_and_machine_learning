# import numpy as np
# import os
# import math
# import pandas as pd
# import matplotlib.pyplot as plt

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

trn=["a/train","ai/train","chA/train","dA/train","tA/train"]
dev=["a/dev","ai/dev","chA/dev","dA/dev","tA/dev"]


def extractData(directory):
    data=[]
    for filename in os.listdir(directory):
        (filename,extension) = os.path.splitext(filename)
        if extension == '.txt':
            f=open(directory+'/'+filename+extension,"r")
            m=[]
            lines = f.readlines()
            s=lines[0]
            l=s.split()
            for i in range(1,len(l),2):
                x = float(l[i])
                y = float(l[i+1])
                m.append([x,y])
            d=pd.DataFrame(m)
            Min=d.min()
            Max=d.max()
            x_min=Min[0]
            y_min=Min[1]
            x_max=Max[0]
            y_max=Max[1]
            for i in range(len(m)):
                m[i][0]=(m[i][0]-x_min)/(x_max-x_min)
                m[i][1] = (m[i][1] - y_min) / (y_max - y_min)
            data.append(np.array(m))
    return data

def dtw(X,Y):
    r1,c1=X.shape
    r2,c2=Y.shape
    DTW=np.array([[np.inf for j in range(r2+1)]for i in range(r1+1)])
    DTW[0][0]=0.
    for i in range(1,r1+1):
        for j in range(1,r2+1):
            DTW[i,j]=min(min(DTW[i-1,j],DTW[i,j-1]),DTW[i-1,j-1])+math.dist(X[i-1],Y[j-1])
    return DTW[r1][r2]


def plot_ROC_curve(scores, C_dev, n):
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
    plt.plot(fpr,tpr)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    return ([tpr,fpr])

def det(tpr,fpr,label):
    d1 = DetCurveDisplay(fpr=fpr, fnr=1 - tpr, estimator_name=str(label)).plot(ax=ax_det)
# def findClass(dist,K):
#     cls=np.array([0 for i in range(5)])
#     for i in range(K):
#         cls[dist[i][1]]+=1
#     return [cls,np.argmax(cls)]

def avg(d,k):
    sum=0.0
    for i in range(k):
        sum+=d[i][0]
    sum=sum/k
    return sum

def draw_confusion(res1,C_dev):
    conf= np.array([[0 for j in range(5)] for i in range(5)])
    for i in range(np.size(res1)):
        conf[C_dev[i]-1][res1[i]-1] += 1
    df_cm = pd.DataFrame(conf)
    as1 = sn.heatmap(df_cm, annot=True,fmt=".1f")
    as1.set_xlabel('true class')
    as1.set_ylabel('predicted class')

trn_data=[extractData(trn[i]) for i in range(5)]
dev_data=[extractData(dev[i]) for i in range(5)]

# K=[10,14,18,20,24,28,30,50]
K=[5,15,25,35,50]
scores=[]
conf_mat=np.array([[0 for i in range(5)] for j in range(5)])

acc=np.array([0 for i in range(5)])
cnt=np.array([0 for i in range(5)])
tpr=[]
fpr=[]
for k in K:
    C_dev=[]
    C_dev_found=[]
    for i in range(5):
        for Y in dev_data[i]:
            dist=[[],[],[],[],[]]
            for j in range(5):
                for X in trn_data[j]:
                    dist[j].append([dtw(X,Y),j])
            averages=[]
            for j in range(5):
                dist[j].sort()
                x=avg(dist[j],k)
                averages.append(x)
            # cls_p=findClass(dist,k)
            cls=averages.index(min(averages))
            if cls==i:
                acc[i]+=1
            cnt[i]+=1
            conf_mat[cls][i]+=1
            inv_average=[]
            for j in range(5):
                inv_average.append(1/averages[j])
            scores.append(inv_average)
            C_dev.append(i+1)
            C_dev_found.append(cls+1)
    #     print("accuracy:",acc[i] , cnt[i])
    # print("final accuracy for k:",k,np.sum(acc)*100/np.sum(cnt))
    l=plot_ROC_curve(np.array(scores),C_dev,len(C_dev))
    tpr.append(l[0])
    fpr.append(l[1])

plt.title("ROC Curve for k values: "+str(K))
plt.legend()
plt.show()

fig, ax_det = plt.subplots(1, 1)
for k in range(len(K)):
    det(tpr[k],fpr[k],K[k])
plt.show()
# draw_confusion(C_dev_found,C_dev)
# plt.show()