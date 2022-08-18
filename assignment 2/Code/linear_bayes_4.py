import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import scipy as sc
from scipy import linalg as LA
from PIL import Image
import math
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import multivariate_normal
from sklearn.metrics import det_curve
from sklearn import preprocessing
import seaborn as sn
import pandas as pd
from sklearn.metrics import DetCurveDisplay

X = []
C = []
X1=[]
X2=[]
X3=[]
f = open("trian.txt", "r")
for line in f.readlines():
    x1, x2, c = line.strip().split(',')


    if c== '1' :
        X1.append((np.longdouble(x1),np.longdouble(x2)))
    elif c=='2':
        X2.append((np.longdouble(x1), np.longdouble(x2)))
    elif c=='3':
        X3.append((np.longdouble(x1), np.longdouble(x2)))
    C.append(np.longdouble(c))


X1=np.array(X1)
X2=np.array(X2)
X3=np.array(X3)
C=np.array(C)

C_dev=[]
f=open("dev.txt","r")
for line in f.readlines():
    x1, x2, c = line.strip().split(',')
    X.append((np.longdouble(x1), np.longdouble(x2)))
    C_dev.append(np.int32(c))

X=np.array(X)
C_dev=np.array(C_dev)
# print(X[:,0])


n1=np.size(X1)//2
mu1=np.array([0.,0.])
mu1[0]=np.sum(X1[:,0])
mu1[1]=np.sum(X1[:,1])
mu1=mu1/n1


n2=np.size(X2)//2
mu2=np.array([0.,0.])
mu2[0]=np.sum(X2[:,0])
mu2[1]=np.sum(X2[:,1])
mu2=mu2/n2

n3=np.size(X3)//2
mu3=np.array([0.,0.])
mu3[0]=np.sum(X3[:,0])
mu3[1]=np.sum(X3[:,1])
mu3=mu3/n3

sig=np.array([[0.,0.],[0.,0.]])

mu=(mu1+mu2+mu3)/3

for i in range(n1):
    # print(X1[i],i,n1)
    temp1=np.subtract(X1[i],mu1)
    sig[0][0]+=temp1[0]*temp1[0]
    sig[1][1]+=temp1[1]*temp1[1]
    sig[0][1]+=temp1[0]*temp1[1]
    sig[1][0]+=temp1[0]*temp1[1]

for i in range(n2):
    # print(X1[i],i,n1)
    temp1=np.subtract(X2[i],mu2)
    sig[0][0]+=temp1[0]*temp1[0]
    sig[1][1]+=temp1[1]*temp1[1]
    sig[0][1]+=temp1[0]*temp1[1]
    sig[1][0]+=temp1[0]*temp1[1]

for i in range(n3):
    # print(X1[i],i,n1)
    temp1=np.subtract(X3[i],mu3)
    sig[0][0]+=temp1[0]*temp1[0]
    sig[1][1]+=temp1[1]*temp1[1]
    sig[0][1]+=temp1[0]*temp1[1]
    sig[1][0]+=temp1[0]*temp1[1]
sig=sig/(n1+n2+n3-1)
sig[1][0]=0.
sig[0][1]=0.
print(sig)
############################################################################################################

res=[]
n=np.size(X)//2
s=np.array([[0.0 for j in range(3)] for i in range(n)])
i=0
for (x1,x2) in X:
    t1=np.row_stack([[x1,x2]])
    t1=np.subtract(t1,mu1)
    g1=0
    g1-=np.log(np.sqrt(2*math.pi))/2
    g1-=np.log(math.sqrt(np.linalg.det(sig)))
    g1-=t1@LA.inv(sig)@t1.T/2
    #print(g1[0][0])
    p1=math.exp(g1)
    p1=p1/3
    s[i][0]=p1

    t2=np.row_stack([[x1,x2]])
    t2=np.subtract(t2,mu2)
    g2 = 0
    g2 -= np.log(np.sqrt(2 * math.pi)) / 2
    g2 -= np.log(math.sqrt(np.linalg.det(sig)))
    g2 -= t2 @ LA.inv(sig) @ t2.T/2
    #print(g1[0][0])
    p2 = math.exp(g2)
    p2 = p2 / 3
    s[i][1] = p2

    t3 = np.row_stack([[x1, x2]])
    t3 = np.subtract(t3, mu3)
    g3 = 0
    g3 -= np.log(np.sqrt(2 * math.pi)) / 2
    g3 -= np.log(math.sqrt(np.linalg.det(sig)))
    g3 -= t3 @ LA.inv(sig) @ t3.T / 2
    p3 = math.exp(g3)
    p3 = p3 / 3
    s[i][2] = p3
    i+=1

    if g1 >= g2 and g1 >= g3:
        res.append(1)
    elif g2 >= g3:
        res.append(2)
    else:
        res.append(3)
res = np.array(res)
confusion = np.array([[0 for j in range(3)] for i in range(3)])

for i in range(np.size(res)):
    # print(C_dev,res)
    confusion[C_dev[i] - 1][res[i] - 1] += 1
# print(confusion)
df_cm = pd.DataFrame(confusion)
# plt.figure(figsize=(10, 7))
# sn.heatmap(df_cm, annot=True, fmt=".1f")
as1 = sn.heatmap(df_cm, annot=True,fmt=".1f")
as1.set_xlabel('true class')
as1.set_ylabel('predicted class')
plt.show()

s_mod=s.flatten()
s_mod=np.sort(s_mod)

tpr=np.array([])
fpr=np.array([])
fnr=np.array([])

for threshold in s_mod:
    tp=0
    fp=0
    tn=0
    fn=0
    for i in range(n):
        ground_truth=C_dev[i]
        for j in range(3):
            if s[i][j] >= threshold:
                if ground_truth == j+1:
                    tp+=1
                else:
                    fp+=1
            else:
                if ground_truth == j+1:
                    fn+=1
                else:
                    tn+=1
    tpr=np.append(tpr,tp/(tp+fn))
    fpr=np.append(fpr,fp/(fp+tn))

fnr=1-tpr

plt.scatter(X1[:,0],X1[:,1])
plt.scatter(X2[:,0],X2[:,1])
plt.scatter(X3[:,0],X3[:,1])
plt.show()
plt.plot(fpr,tpr)
plt.show()
tpr_lis=tpr.tolist()
fpr_lis=fpr.tolist()
f2=open("4.txt","a")
for item in fpr_lis:
        f2.write("%s\n" % item)
for item in tpr_lis:
        f2.write("%s\n" % item)

display = DetCurveDisplay(fpr=fpr,fnr=fnr).plot()
######################################################################################################################

import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn-dark')
# plt.rcParams['figure.figsize'] = 14, 6
fig = plt.figure()
ax = plt.axes(projection='3d')


random_seed=1000
pdf_list = []
cov = sig
mean = mu1
distr = multivariate_normal(cov=cov, mean=mean,
                            seed=random_seed)

# # Generating a meshgrid complacent with
# # the 3-sigma boundary
mean_1, mean_2 = mean[0], mean[1]
sigma_1, sigma_2 = cov[0, 0], cov[1, 1]
x = np.linspace(-20, 30, num=100)
y = np.linspace(-20, 30, num=100)
X, Y = np.meshgrid(x, y)
X11=X
Y11=Y
pdf1 = np.zeros(X.shape)
for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        pdf1[i, j] = distr.pdf([X[i, j], Y[i, j]])
pdf_list.append(pdf1)
ax.axes.zaxis.set_ticks([])
plt.tight_layout()


random_seed=1000
# pdf_list = []
cov = sig
mean = mu2
distr = multivariate_normal(cov=cov, mean=mean,
                            seed=random_seed)

# Generating a meshgrid complacent with
# the 3-sigma boundary
mean_1, mean_2 = mean[0], mean[1]
sigma_1, sigma_2 = cov[0, 0], cov[1, 1]
x = np.linspace(-20, 30, num=100)
y = np.linspace(-20, 30, num=100)
X, Y = np.meshgrid(x, y)
X22=X
Y22=Y
pdf2 = np.zeros(X.shape)
for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        pdf2[i, j] = distr.pdf([X[i, j], Y[i, j]])

ax.plot_surface(X22,Y22, pdf2, cmap='inferno')
pdf_list.append(pdf2)
ax.axes.zaxis.set_ticks([])
plt.tight_layout()
# plt.show()


random_seed=1000
# pdf_list = []
cov = sig
mean = mu3
distr = multivariate_normal(cov=cov, mean=mean,
                            seed=random_seed)

# Generating a meshgrid complacent with
# the 3-sigma boundary
mean_1, mean_2 = mean[0], mean[1]
sigma_1, sigma_2 = cov[0, 0], cov[1, 1]

x = np.linspace(-20, 30, num=100)
y = np.linspace(-20, 30, num=100)
X, Y = np.meshgrid(x, y)
X33=X
Y33=Y

pdf3 = np.zeros(X.shape)
for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        pdf3[i, j] = distr.pdf([X[i, j], Y[i, j]])


pdf_list.append(pdf3)
plt.xlabel("x1")
plt.ylabel("x2")
plt.title('PDF functions for three classes')

PDF=np.maximum(pdf1,pdf2)
PDF=np.maximum(PDF,pdf3)
ax.plot_surface(X, Y, PDF, cmap='viridis')
ax.axes.zaxis.set_ticks([])
plt.tight_layout()
plt.show()
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
plt.contour(X11, Y11, np.array(pdf_list[0]), cmap='viridis')
plt.contour(X22, Y22, np.array(pdf_list[1]), cmap='inferno')
plt.contour(X33, Y33, np.array(pdf_list[2]), cmap='plasma')
plt.tight_layout()



plt.scatter(X1[:,0],X1[:,1])
plt.scatter(X2[:,0],X2[:,1])
plt.scatter(X3[:,0],X3[:,1])

mu1=np.row_stack(mu1)
mu2=np.row_stack(mu2)
mu3=np.row_stack(mu3)

mu12_s=np.subtract(mu1,mu2)
mu12_s=sig@mu12_s

mu12_s=mu12_s.T
mu12_a=np.add(mu1,mu2)/2
print(mu12_a,mu12_s)
x=np.arange(-10,9,0.01)
y=(-mu12_s[0][0]/mu12_s[0][1])*(x-mu12_a[0][0])+mu12_a[1][0]
plt.plot(x,y)


mu23_s=np.subtract(mu2,mu3)
mu23_s=sig@mu23_s
mu23_s=mu23_s.T
mu23_a=np.add(mu2,mu3)/2
x=np.arange(9,30,0.01)
y=(-mu23_s[0][0]/mu23_s[0][1])*(x-mu23_a[0][0])+mu23_a[1][0]
plt.plot(x,y)


mu31_s=np.subtract(mu3,mu1)
mu32_s=sig@mu31_s
mu31_s=mu31_s.T
mu31_a=np.add(mu3,mu1)/2
x=np.arange(-10,9,0.01)
y=(-mu31_s[0][0]/mu31_s[0][1])*(x-mu31_a[0][0])+mu31_a[1][0]
plt.plot(x,y)
plt.show()
#############################################################################################################################################################

plt.contour(X11, Y11, np.array(pdf_list[0]), cmap='viridis')
plt.contour(X22, Y22, np.array(pdf_list[1]), cmap='inferno')
plt.contour(X33, Y33, np.array(pdf_list[2]), cmap='plasma')
plt.tight_layout()

eig,ev=LA.eig(sig)
x=[mu1[0],mu1[0]+8*ev[0][0]]
y=[mu1[1],mu1[1]+8*ev[0][1]]
plt.plot(x,y)

x=[mu1[0],mu1[0]+10*ev[1][0]]
y=[mu1[1],mu1[1]+10*ev[1][1]]
plt.plot(x,y)

eig,ev=LA.eig(sig)
x=[mu2[0],mu2[0]+8*ev[0][0]]
y=[mu2[1],mu2[1]+8*ev[0][1]]
plt.plot(x,y)

x=[mu2[0],mu2[0]+10*ev[1][0]]
y=[mu2[1],mu2[1]+10*ev[1][1]]
plt.plot(x,y)

eig,ev=LA.eig(sig)
x=[mu3[0],mu3[0]+8*ev[0][0]]
y=[mu3[1],mu3[1]+8*ev[0][1]]
plt.plot(x,y)

x=[mu3[0],mu3[0]+10*ev[1][0]]
y=[mu3[1],mu3[1]+10*ev[1][1]]
plt.plot(x,y)
plt.show()
