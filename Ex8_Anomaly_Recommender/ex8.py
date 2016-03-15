# -*- coding: utf-8 -*-
"""
Created on Wed Dec  2 10:33:39 2015

@author: Hanna
"""

import numpy as np
import scipy.io as io
import matplotlib.pyplot as plt
from scipy import linalg

def estimateGaussian(X):
    mu = np.mean(X,axis=0)
    sigma2 	= np.var(X,axis=0)
    
    return mu, sigma2
    
    
def multivariateGaussian(X, mu, sigma2):
    #this is multivariate gaussian so sigma2 is a covariance matrix;
    #if supplied sigma2 is a vector (features were assumed independent), 
    #form corresponding covariance matrix 
    #(although using multivariate Gaussian is computationally inefficient in this case)
    if sigma2.ndim == 1: 
        sigma2 = np.diag(sigma2)
    n = len(mu)
    invsigma2 = linalg.inv(sigma2)
    #should work for vector (individual example) and matrix X
    if X.ndim == 1:
            p = np.exp(-np.dot(np.dot((X-mu).T,invsigma2),X-mu)/2.0)/((2.0*np.pi)**(n/2.0)*np.sqrt(linalg.det(sigma2)))
    else:
        m = np.shape(X)[0]
        p = np.zeros(m)
        for i in range(0,m):
            p[i] = np.exp(-np.dot(np.dot((X[i]-mu).T,invsigma2),X[i]-mu)/2.0)/((2.0*np.pi)**(n/2.0)*np.sqrt(linalg.det(sigma2)))
            
    return p
    
    
def visualizeFit(X, mu, sigma2):
    X1 = np.linspace(0,35,35/0.5+1)
    X2 = np.linspace(0,35,35/0.5+1)
    X_vals = np.zeros((len(X1)*len(X2),2))
    stride = len(X1)
    for i, elem1 in enumerate(X1):
        for j, elem2 in enumerate(X2):
            X_vals[i*stride+j] = np.array([elem1, elem2])
    Z = multivariateGaussian(X_vals, mu, sigma2)
    Z = Z.reshape(len(X1),len(X2))
    X1, X2 = np.meshgrid(X1,X2)

    plt.figure()
    plt.scatter(X[:,0], X[:,1], c='b', marker='x', s=10, linewidths=0.5)
    plt.contour(X1, X2, Z.T, levels=10.0**np.arange(-20,0,3))
    plt.xlabel("Latency (ms)")
    plt.ylabel("Throughput (mb/s)")
    plt.xlim([0, 30])
    plt.ylim([0, 30])
    
   
def selectThreshold(yval, pval):
    bestEpsilon = 0.0
    bestF1 = 0.0
    F1 = 0.0

    #stepsize = (np.max(pval) - np.min(pval))/1000.0
    epsilons = np.linspace(np.min(pval),np.max(pval),1001)
    yval = yval.reshape(np.shape(pval))
    
    for epsilon in epsilons:
        predictions = (pval < epsilon)
        tp = np.sum((predictions == 1) & (yval == 1))
        fp = np.sum((predictions == 1) & (yval == 0))
        fn = np.sum((predictions == 0) & (yval == 1))
        #take care of the case when model does not predict any anomalies,
        #i.e. tp == 0 & fp == 0
        if tp == 0 & fp == 0:
            prec = 1.0
        else:
            prec = tp/(tp + fp)
        #tp and fn cannot be zero simultaneously if there are true anomalies in CV set
        rec = tp/(tp + fn)
        F1 = 2.0*prec*rec/(prec+rec)
        if F1 > bestF1:
            bestF1 = F1
            bestEpsilon = epsilon
            
    return bestEpsilon, bestF1
        
	
if __name__ == '__main__':
    
    #load and visualize the example dataset
    mat = io.loadmat("ex8data1.mat")
    X, Xval, yval = mat['X'], mat['Xval'], mat['yval']
    #print(np.shape(X))
    #print(np.shape(Xval))
    #print(np.shape(yval))
    #print(X[:2],Xval[:2],yval[:2])
    
    plt.figure()
    plt.scatter(X[:,0], X[:,1], c='b', marker='x', s=10, linewidths=0.5)
    plt.xlabel("Latency (ms)")
    plt.ylabel("Throughput (mb/s)")
    plt.xlim([0, 30])
    plt.ylim([0, 30])
    plt.savefig('ExampleDataSet1.pdf')
    
    #estimate parameters for each feature's Gaussian
    mu, sigma2 = estimateGaussian(X)
    
    #get probability density for each example in training set
    p = multivariateGaussian(X, mu, sigma2)
    #print(p[:5])
    
    #visualize the dataset and its estimated distribution fit
    visualizeFit(X, mu, sigma2)
    plt.savefig('FitDataSet1.pdf')
    
    #get probability density for each examples in CV set
    pval = multivariateGaussian(Xval, mu, sigma2)
    #select threshold using pval 
    epsilon, F1 = selectThreshold(yval, pval)
    print("Threshold is epsilon = ", epsilon, " with F1 = ", F1)  # get 8.99085277927e-05, 0.875
    print("epsilon should be about 8.99e-05")
    
    #find anomalies in the training set based on threshold
    outliers = np.where(p < epsilon)
    #draw red circles around outliers
    visualizeFit(X, mu, sigma2)
    plt.scatter(X[outliers, 0], X[outliers, 1], facecolors='none', edgecolors='r', s=40, linewidths=2)
    plt.savefig('OutliersDataSet1.pdf')
    plt.show()
    
    #find anomalies in multidimensional dataset
    
    #load the dataset
    mat = io.loadmat("ex8data2.mat")
    X, Xval, yval = mat['X'], mat['Xval'], mat['yval']
    
    #estimate parameters for each feature's Gaussian
    mu, sigma2 = estimateGaussian(X)
    
    #get probability density for each example in training set
    p = multivariateGaussian(X, mu, sigma2)
    
    #get probability density for each examples in CV set
    pval = multivariateGaussian(Xval, mu, sigma2)
    #select threshold using pval 
    epsilon, F1 = selectThreshold(yval, pval)
    print("Threshold for 11D dataset is epsilon = ", epsilon, " with F1 = ", F1)  # get 1.37722889076e-18, 0.615384615385
    print("epsilon should be about 1.38e-18")
    
    #find # of anomalies in the training set based on threshold
    print("Found ", np.sum(p < epsilon), " outliers")  # get 117
    print("should be 117")
    