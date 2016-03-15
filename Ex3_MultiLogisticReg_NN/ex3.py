# -*- coding: utf-8 -*-
"""
Created on Wed Nov 18 18:09:18 2015

@author: Hanna
"""
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import scipy.optimize as op
import scipy.io as io

def displayData(X): 
    pixels = 20  # images are 20x20 pixels
    #100 examples shown on a  10 by 10 square
    display_rows = 10
    display_cols = 10
    out = np.zeros((pixels*display_rows,pixels*display_cols))
    rand_indices = np.random.permutation(5000)[0:display_rows*display_cols]
    for j in range(0, display_rows):
        for i in range(0, display_cols):
            start_i = i*pixels
            start_j = j*pixels
            out[start_i:start_i+pixels, start_j:start_j+pixels] = X[rand_indices[display_rows*j+i]].reshape(pixels, pixels).T    
    fig = plt.figure()
    ax = fig.gca()
    ax.imshow(out,cmap="Greys_r")
    ax.set_axis_off()
    plt.savefig("100dataExamples.pdf")
    plt.show()
  
  
def sigmoid(z):
    #works elementwise for any array z
    
    return sp.special.expit(z)  # expit(x) = 1/(1+exp(-x)) elementwise for array x
    
    
def hypothesis(theta,X):
    #works for any # of features

    return sigmoid(np.dot(X,theta))
 
 
def computeCostReg(theta,X,y,lambbda):  # J(theta) with regularization
    #works for any # of features
    m = len(y)
    h = hypothesis(theta,X)
    thetaReg = theta[1:]  # remove the first row(corresponding to theta0) from matrix theta
    cost = (-np.dot(y.T,sp.log(h))-np.dot((1.0-y).T,sp.log(1.0-h)))/m + lambbda*np.dot(thetaReg.T,thetaReg)/(2.0*m)
    #check for nans: can occur since have log
    #if skip this, precision loss occurs
    if np.isnan(cost):
        return np.inf
        
    return cost
    
    
def computeCostDerivReg(theta,X,y,lambbda):  # partial derivatives of J(theta) with regularization
    #works for any # of features
    n = np.shape(X)[1]  # # of features including itercept; in main n is the # of original features
    m = np.shape(X)[0]
    #minimize routine changes dimensions of theta from matrix to vector, 
    #i.e from (3,1) to (3,) in this case
    #so need to reshape to original dimensions
    #since in computeCostDeriv routine it is important that theta is matrix
    #does not matter for computeCost routine
    theta = theta.reshape(n,1)
    h = hypothesis(theta,X)
    #initialize matrix of gradients
    gradient = np.zeros(np.shape(theta))
    #use formula for gradient of regularized cost function
    gradient = np.dot(X.T,h-y)/m + lambbda*theta/m
    #do not need to regularize theta0
    #it's computationally more efficient to compute all elements with regularization
    #and then compensate for theta0
    gradient[0,0] = gradient[0,0] - lambbda*theta[0,0]/m
    
    return gradient.flatten()  # minimize needs vector of gradients not matrix


def oneVsAll(X,y,num_classes,lambbda):
    n = np.shape(X)[1]  # # of features including intercept
    all_theta = np.zeros((num_classes,n))
    for k in range(0, num_classes):
            theta = np.zeros((n,1))
            y_k = (y == k+1).astype(int)
            #below are two absolutely equivalent ways to call nonlinear conjugate gradient
            #analogous to matlab fmincg
            #result = op.fmin_cg(computeCostReg, theta, fprime=computeCostDerivReg, args=(X,y_k,lambbda))
            #all_theta[k,:] = result
            result = op.minimize(computeCostReg, theta, args=(X,y_k,lambbda), method='CG', jac=computeCostDerivReg, options={'disp': True})
            all_theta[k,:] = result.x

    return all_theta


def predictOneVsAll(all_theta,X):
    all_probability = hypothesis(all_theta.T,X)
    prediction = np.argmax(all_probability,axis=1) + 1  # we do not have class 0 but have class 10    
    
    return prediction.reshape(np.shape(X)[0],1)
       
       
if __name__ == '__main__':
    
    mat = io.loadmat("ex3data1.mat")
    X, y = mat['X'], mat['y']
    displayData(X)
    
    m = np.shape(X)[0]  # # of examples
    X = np.hstack((np.ones((m,1)),X))
    lambbda = 0.1
    all_theta = oneVsAll(X,y,10,lambbda)
    prediction = predictOneVsAll(all_theta,X)
    training_accuracy = np.mean(prediction == y) * 100.0
    print("training set prediction accuracy = ", training_accuracy,"%")  # get 96.46 %
    print("supposed to be 94.9%, obtained accuracy is a bit higher since did not restrict max # of iterations")
    