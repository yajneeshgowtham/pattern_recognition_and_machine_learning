import numpy as np
import random
import math
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from collections import OrderedDict
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
    sig=sig/(cnt-1)
    return sig
def finalSigmas(Cluster,MU):
        Sig=[finalsigma(Cluster[i],MU[i]) for i in Cluster.keys()]
        return Sig
def finalPhis(Cluster):
    Phi = np.array([Cluster[i].size for i in Cluster.keys()])
    Phi=Phi/np.sum(Phi)
    return Phi
def classify(MU,dev):
    cnt1=0
    cnt2=0
    cnt=0
    scores=[]
    for x in dev[0]:
        cnt+=1
        id1,dist1=nearCluster(x,MU[0])
        id2,dist2=nearCluster(x,MU[1])
        if dist1<=dist2:
            cnt1+=1
            plt.plot([x[0]], [x[1]],color='yellow',marker='.',alpha=0.5)
        else:
            plt.plot([x[0]], [x[1]], color='blue', marker='.', alpha=0.5)
        scores.append([1/dist1,1/dist2])
    for x in dev[1]:
        cnt+=1
        id1, dist1 = nearCluster(x, MU[0])
        id2, dist2 = nearCluster(x, MU[1])
        if dist2 <=dist1:
            cnt2 += 1
            plt.plot([x[0]], [x[1]],color='blue',marker='.',alpha=0.5)
        else:
            plt.plot([x[0]], [x[1]], color='yellow', marker='.', alpha=0.5)
        scores.append([1/dist1, 1/dist2])
    print("accuracy is ",(cnt1+cnt2)*100/cnt)
    return scores
def decisionSurface1(MU):
    x = np.linspace(-16, 16, 150)
    y = np.linspace(-16, 16, 150)
    X, Y = np.meshgrid(x, y)
    pdf = []
    for q in y:
        for p in x:
            P = np.array([p,q])
            id1, dist1 = nearCluster(P, MU[0])
            id2, dist2 = nearCluster(P, MU[1])
            if dist1<=dist2:
                pdf.append(1)
            else:
                pdf.append(2)
    pdf=np.array(pdf)
    Z=pdf.reshape(X.shape)
    plt.contourf(X,Y,Z,cmap='Paired')
def decisionSurface(MU):
    x = np.linspace(-16,16,150)
    y = np.linspace(-16,16,150)
    X,Y=np.meshgrid(x,y)
    pdf1=[]
    pdf2=[]
    for i  in range(X.shape[0]):
        for j in range(X.shape[1]) :
            p=np.array([X[i,j],Y[i,j]])
            id1, dist1 = nearCluster(p, MU[0])
            id2, dist2 = nearCluster(p, MU[1])
            if dist1<=dist2:
                pdf1.append(p)
            else:
                pdf2.append(p)
    pdf1=np.array(pdf1)
    pdf2=np.array(pdf2)
    plt.plot(pdf1[:,0],pdf1[:,1],marker='o',color='lightsteelblue',alpha=0.1)
    plt.plot(pdf2[:,0],pdf2[:,1],marker='o',color='beige',alpha=0.1)

def plotContour(mu,cluster,sig):
    x_min=np.min(cluster[:,0])
    x_max=np.max(cluster[:,0])
    y_min=np.min(cluster[:,1])
    y_max=np.max(cluster[:,1])
    x = np.linspace(x_min-1, x_max+1, 100)
    y = np.linspace(y_min-1, y_max+1, 100)
    distr = multivariate_normal(cov=sig, mean=mu)
    X, Y = np.meshgrid(x, y)
    pdf = np.zeros(X.shape)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            pdf[i, j] = distr.pdf([X[i, j], Y[i, j]])
    pdf_list = []
    pdf_list.append(pdf)
    for idx, val in enumerate(pdf_list):
        plt.contour(X, Y, val,cmap='plasma')
def plotContours(MU,Cluster,Sig):
    for i in Cluster.keys():
        plotContour(MU[i],Cluster[i],Sig[i])
def gauss(x,mu,sig):
    a=x-mu
    a=np.reshape(a,(np.size(x),1))
    r=a.T@LA.inv(sig)@a
    res=math.exp(-r[0][0]/2)/math.sqrt(LA.det(sig))
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
    c = 2
    mu=np.array([0. for i in range(c)])
    for i in range(n):
        mu=mu+gam[i]*data[i]
    mu=mu/N
    return mu
def newSig(gam,N,data,n,mu):
    c=2
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
    cnt1=0
    cnt2=0
    cnt=0
    scores=[]
    for x in dev[0]:
        cnt+=1
        dist1=np.array([Phis[0][j]*gauss(x,MU[0][j],Sigmas[0][j]) for j in range(K)])
        dist2=np.array([Phis[1][j]*gauss(x,MU[1][j],Sigmas[1][j]) for j in range(K)])
        if np.sum(dist1)>np.sum(dist2):
            cnt1+=1
            plt.plot([x[0]], [x[1]], color='yellow', marker='.', alpha=0.5)
        else:
            plt.plot([x[0]], [x[1]], color='blue', marker='.', alpha=0.5)
        scores.append([np.sum(dist1),np.sum(dist2)])
    for x in dev[1]:
        cnt+=1
        dist1=np.array([Phis[0][j]*gauss(x,MU[0][j],Sigmas[0][j]) for j in range(K)])
        dist2=np.array([Phis[1][j]*gauss(x,MU[1][j],Sigmas[1][j]) for j in range(K)])
        if np.sum(dist2)>=np.sum(dist1):
            cnt2+=1
            plt.plot([x[0]], [x[1]], color='blue', marker='.', alpha=0.5)
        else:
            plt.plot([x[0]], [x[1]], color='yellow', marker='.', alpha=0.5)
        scores.append([np.sum(dist1), np.sum(dist2)])
    print("accuracy is ", (cnt1 + cnt2) * 100 / cnt)
    return scores
def decisionSurfaceGMM1(MU,Sigmas,Phis):
    x = np.linspace(-16, 16, 150)
    y = np.linspace(-16, 16, 150)
    X, Y = np.meshgrid(x, y)
    pdf = []
    for q in y:
        for p in x:
            P = np.array([p, q])
            dist1 = np.array([Phis[0][j] * gauss(P, MU[0][j], Sigmas[0][j]) for j in range(K)])
            dist2 = np.array([Phis[1][j] * gauss(P, MU[1][j], Sigmas[1][j]) for j in range(K)])
            if np.max(dist1)>np.max(dist2):
                pdf.append(1)
            else:
                pdf.append(2)
    pdf=np.array(pdf)
    Z=pdf.reshape(X.shape)
    plt.contourf(X,Y,Z,cmap='Paired')
def decisionSurfaceGMM(MU,Sigmas,Phis):
    x = np.linspace(-16, 16, 150)
    y = np.linspace(-16, 16, 150)
    X, Y = np.meshgrid(x, y)
    pdf1 = []
    pdf2 = []
    for i  in range(X.shape[0]):
        for j in range(X.shape[1]) :
            p=np.array([X[i,j],Y[i,j]])
            dist1 = np.array([Phis[0][k]*gauss(p, MU[0][k], Sigmas[0][k]) for k in range(K)])
            dist2 = np.array([Phis[1][k]*gauss(p, MU[1][k], Sigmas[1][k]) for k in range(K)])
            if np.max(dist1)>np.max(dist2):
                pdf1.append(p)
            else:
                pdf2.append(p)
    pdf1 = np.array(pdf1)
    pdf2 = np.array(pdf2)
    plt.plot(pdf1[:, 0], pdf1[:, 1], marker='o', color='lightsteelblue', alpha=0.1)
    plt.plot(pdf2[:, 0], pdf2[:, 1], marker='o', color='beige', alpha=0.1)


def normalize(data):
    mu=np.array([0.,0.])
    mu[0]=np.sum(train[:,0])/len(train[:,0])
    mu[1] = np.sum(train[:, 1]) / len(train[:, 1])
    print(mu)
    sig=np.array([0.,0.])
    for i in range(len(data)):
        x=np.array(data[i])
        a=x-mu
        sig[0]+=a[0]*a[0]
        sig[1]+=a[1]*a[1]
    sig/=len(data)
    for i in range(len(data)):
        x=data[i]
        data[i][0]=(x[0]-mu[0])/sig[0]
        data[i][1]=(x[1]-mu[1])/sig[1]
    return data


train = extractData("train.txt")
dev = extractData("dev.txt")
# train=np.array([normalize(train[i]) for i in range(2)])
# dev=np.array([normalize(dev[i]) for i in range(2)])
C_dev=[]
for i in range(2):
    d=dev[i]
    for a,b in d:
        C_dev.append(i+1)
K=25
MU=[]
MU_diag=[]
for i in range(2):
    a,b=initialParam(train[i],K)
    MU.append(a)
    MU_diag.append(b)
MU=[K_Means(MU[i],train[i],K) for i in range(2)]
Clusters=[finalClusters(train[i],MU[i]) for i in range(2)]
Sigmas=[finalSigmas(Clusters[i],MU[i]) for i  in range(2)]
Phis=[finalPhis(Clusters[i]) for i in range(2)]
plt.figure(figsize=(14,10))
plt.suptitle("Decision Boundaries for K="+str(K))
plt.subplot(2, 2, 1)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
decisionSurface1(MU)
scores_Kmeans=classify(MU,dev)
plotContours(MU[0],Clusters[0],Sigmas[0])
plotContours(MU[1],Clusters[1],Sigmas[1])
plt.xlabel("dimension 1")
plt.ylabel("dimension 2")
plt.title("K means for non diagonal  covariance matrices")
plt.subplot(2, 2, 2)
for i in range(2):
    MU[i],Sigmas[i],Phis[i]=GMM(MU[i],Sigmas[i],Phis[i],K,train[i])

decisionSurfaceGMM1(MU,Sigmas,Phis)
scores_GMM=classifyGMM(MU,Sigmas,dev,K,Phis)
plotContours(MU[0],Clusters[0],Sigmas[0])
plotContours(MU[1],Clusters[1],Sigmas[1])
plt.xlabel("dimension 1")
plt.ylabel("dimension 2")
plt.title("GMM for non diagonal covariance matrices")
# plt.show()




def diagnolize(Sigma):
    for i in range(len(Sigma)):
        sig=Sigma[i]
        sig[0][1]=0.
        sig[1][0]=0.
        Sigma[i]=sig
    return Sigma



def GMM_diag(MU,Sigma,Phi,K,data):
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
            sig=Sigma[j]
            sig[0][1]=0.
            sig[1][0]=0.
            Sigma[j]=sig
            Phi[j]=N[j]/np.sum(N)
            if np.array_equal(copy,MU[j]):
                cnt+=1
        if cnt==K:
            Change=False
        itr+=1

    return MU,Sigma,Phi

# train = extractData("train.txt")
# dev = extractData("dev.txt")
# MU=[initialParam(train[i],K) for i in range(2)]
MU_diag=[K_Means(MU_diag[i],train[i],K) for i in range(2)]
Clusters_diag=[finalClusters(train[i],MU_diag[i]) for i in range(2)]
Sigmas_diag=[finalSigmas(Clusters_diag[i],MU_diag[i]) for i  in range(2)]
Sigmas_diag=[diagnolize(Sigmas_diag[i]) for i in range(2)]
Phis_diag=[finalPhis(Clusters_diag[i]) for i in range(2)]

plt.subplot(2, 2, 3)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
decisionSurface1(MU_diag)
scores_Kmeans_diag=classify(MU_diag,dev)
plotContours(MU_diag[0],Clusters_diag[0],Sigmas_diag[0])
plotContours(MU_diag[1],Clusters_diag[1],Sigmas_diag[1])
# plt.suptitle("Decision Boundaries for K="+str(K))
plt.xlabel("dimension 1")
plt.ylabel("dimension 2")
plt.title("K Means with diagonal covariance matrices")
plt.subplot(2, 2, 4)
for i in range(2):
    MU_diag[i],Sigmas_diag[i],Phis_diag[i]=GMM_diag(MU_diag[i],Sigmas_diag[i],Phis_diag[i],K,train[i])

decisionSurfaceGMM1(MU_diag,Sigmas_diag,Phis_diag)
scores_GMM_diag=classifyGMM(MU_diag,Sigmas_diag,dev,K,Phis_diag)
plotContours(MU_diag[0],Clusters_diag[0],Sigmas_diag[0])
plotContours(MU_diag[1],Clusters_diag[1],Sigmas_diag[1])

plt.xlabel("dimension 1")
plt.ylabel("dimension 2")
plt.title("GMM with diagonal covariance matrices")
plt.show()

def plot_ROC_curve(scores, n,cls):
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
    plt.plot(fpr,tpr,label=cls)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    fnr=1-tpr
    return (fpr,fnr)

scores_Kmeans=np.array(scores_Kmeans)
scores_GMM=np.array(scores_GMM)
scores_Kmeans_diag=np.array(scores_Kmeans_diag)
scores_GMM_diag=np.array(scores_GMM_diag)


fpr1,fnr1=plot_ROC_curve(scores_Kmeans,len(C_dev),"k Means(non diagonal covariance matrices)")
fpr2,fnr2=plot_ROC_curve(scores_GMM,len(C_dev),"GMM(non diagonal covariance matrices)")
fpr3,fnr3=plot_ROC_curve(scores_Kmeans_diag,len(C_dev),"K means(diagonal covariance matrices)")
fpr4,fnr4=plot_ROC_curve(scores_GMM_diag,len(C_dev),"GMM (diagonal covariance matrices)")
plt.legend()
plt.show()

fig, ax_det = plt.subplots(1,1)
d1=DetCurveDisplay(fpr=fpr1,fnr=fnr1,estimator_name="k Means(non diagonal covariance matrices)").plot(ax=ax_det)
d2=DetCurveDisplay(fpr=fpr2,fnr=fnr2,estimator_name="GMM(non diagonal covariance matrices)").plot(ax=ax_det)
d3=DetCurveDisplay(fpr=fpr3,fnr=fnr3,estimator_name="K means(diagonal covariance matrices)").plot(ax=ax_det)
d4=DetCurveDisplay(fpr=fpr4,fnr=fnr4,estimator_name="GMM (diagonal covariance matrices)").plot(ax=ax_det)
plt.show()
