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
from sklearn.metrics import DetCurveDisplay

fpr1=np.array([])
tpr1=np.array([])
fpr2=np.array([])
tpr2=np.array([])
fpr3=np.array([])
tpr3=np.array([])
fpr4=np.array([])
tpr4=np.array([])
fpr5=np.array([])
tpr5=np.array([])
f=open("1.txt","r")
count=0
for line in f.readlines():
    if count <900:
        fpr1=np.append(fpr1,np.float64(line))
    else:
        tpr1=np.append(tpr1,np.float64(line))
    count+=1
f=open("2.txt","r")
count=0
for line in f.readlines():
    if count <900:
        fpr2=np.append(fpr2,np.float64(line))
    else:
        tpr2=np.append(tpr2,np.float64(line))
    count+=1
f=open("3.txt","r")
count=0
for line in f.readlines():
    if count <900:
        fpr3=np.append(fpr3,np.float64(line))
    else:
        tpr3=np.append(tpr3,np.float64(line))
    count+=1
f=open("4.txt","r")
count=0
for line in f.readlines():
    if count <900:
        fpr4=np.append(fpr4,np.float64(line))
    else:
        tpr4=np.append(tpr4,np.float64(line))
    count+=1
f=open("5.txt","r")
count=0
for line in f.readlines():
    if count <900:
        fpr5=np.append(fpr5,np.float64(line))
    else:
        tpr5=np.append(tpr5,np.float64(line))
    count+=1
plt.plot(fpr1,tpr1,label='case 1')
plt.plot(fpr2,tpr2,label='case 2')
plt.plot(fpr3,tpr3,label='case 3')
plt.plot(fpr4,tpr4,label='case 4')
plt.plot(fpr5,tpr5,label='case 5')
plt.legend()
plt.show()
fig, ax_det = plt.subplots(1,1)
d1=DetCurveDisplay(fpr=fpr1,fnr=1-tpr1,estimator_name="case 1").plot(ax=ax_det)
d2=DetCurveDisplay(fpr=fpr2,fnr=1-tpr2,estimator_name="case 2").plot(ax=ax_det)
d3=DetCurveDisplay(fpr=fpr3,fnr=1-tpr3,estimator_name="case 3").plot(ax=ax_det)
d4=DetCurveDisplay(fpr=fpr4,fnr=1-tpr4,estimator_name="case 4").plot(ax=ax_det)
d5=DetCurveDisplay(fpr=fpr5,fnr=1-tpr5,estimator_name="case 5").plot(ax=ax_det)
plt.show()