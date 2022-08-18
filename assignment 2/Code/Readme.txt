PART 1:
--First we need to take the codes linear 1d and ridge 1d and take the input files and change the file names in open accordingly for 1D data.
--Now for different values of m we need to change m_array accordingly and get the values of x vs t ,x vs y and y vs t which ever graph or error train and error test .
--To get for different values of n change the value of n and get the results
--Now for ridge regression change the value of lam_array to get the values for different values of lambda
--simillarly for 2d data change the value sof m_array and lam_array to get the error values.
--1-D data-->linear1D,ridge1D
--2-D data-->linear2D,ridge2D
--------------------------------------------------------------------------------------------------------------------------------------------------------------
PART 2:
2.
--There will be three folders for each linearly seperable,linearly non-seperable,real data.
--Now, for each folder there will be .py files within them along with the dataset which we are given, we will take this input by open() function in the .py codes. So, we can give any dataset we want as input.
--Now, if we run the code we will get outputs as confusion matrix, contours, eigen vectors, decision curves. If we want roc and det curves then we can run the roc.py in the same folder and can obtain the ROC and DET curves for all the five cases given.
--To run the roc we get the values of fpr,tpr in 1.py,2.py,3.py,4.py,5.py make sure it is empty before we run the file sof particular ones
--Linearly seperable data---->linear_bayes1.py,linear_bayes2.py,linear_bayes3.py,linear_bayes4.py,linear_bayes5.py,linear_roc.py
-- Non Linearly seperable data---->nonlinear_bayes1.py,nonlinear_bayes2.py,nonlinear_bayes3.py,nonlinear_bayes4.py,nonlinear_bayes5.py,nonlinear_roc.py
--Real data---->Real_bayes1.py,Real_bayes2.py,Real_bayes3.py,Real_bayes4.py,Real_bayes5.py,real_roc.py