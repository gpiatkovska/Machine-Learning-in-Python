# -*- coding: utf-8 -*-
"""
Created on Tue Nov 17 14:28:30 2015

@author: Hanna
"""
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import scipy.optimize as op

def plotData(data):
    pos = data[data[:,2] == 1]
    neg = data[data[:,2] == 0]
    
    plt.figure()
    plt.scatter(pos[:,0], pos[:,1], c='k', marker='+', s=40, linewidths=2)
    plt.scatter(neg[:,0], neg[:,1], c='y', marker='o', s=40, linewidths=1)
    plt.legend(['y=1', 'y=0'], loc='upper right', scatterpoints=1)
    plt.xlabel("Microchip test 1")
    plt.ylabel("Microchip test 2")
    plt.xlim([-1, 1.5])
    plt.ylim([-0.8, 1.2])
    
    
def plotDecisionBdryNonlin(data,X,theta,lambbda):
    plotData(data)
    u = np.linspace(-1,1.5,50)
    v = np.linspace(-1,1.5,50)
    #print(np.shape(theta))
    z = np.zeros((len(u),len(v)))
    for i in range(0, len(u)): 
        for j in range(0, len(v)):
            mapped = mapFeature(u[i],v[j],6)
            #print(np.shape(mapped))
            z[i,j] = np.dot(mapped,theta)
    u, v = np.meshgrid(u,v)	
    plt.contour(u, v, z.T, [0.0, 0.0], colors = 'Green')
    plt.annotate('—', xy=(1.02, 0.79), color='Green')
    plt.annotate('Decision bdry', xy=(1.11, 0.79),fontsize=10)
    plt.title("lambda=" + str(lambbda))
    plt.savefig("NonLinDecisionBdry_lambda"+str(lambbda)+".pdf")
    
    
def mapFeature(X1,X2,degree):
    #maps features X1,X2 into polynomial of degree 6
    #Returns a new feature array with more features, comprising of 
    #X1, X2, X1.^2, X2.^2, X1*X2, X1*X2.^2, etc..
    #adds intercept to a new feature matrix
    out = np.ones((np.shape(X1)))
    for i in range(1,degree+1):
        for j in range(0,i+1):
            out=np.hstack((out,(X1**(i-j))*(X2**j)))
            
    return out
 
    
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
    thetaReg = theta[1:]  #remove the first row(corresponding to theta0) from matrix theta
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
    
    
def predict(theta,X):
    if hypothesis(theta,X) >= 0.5:
        prediction = 1 
    else:
        prediction = 0
      
    return prediction


if __name__ == '__main__':
 
    data = np.genfromtxt("ex2data2.txt", delimiter=',')
    m = np.shape(data)[0]  # number of examples in training set
    n = np.shape(data)[1]-1  # number of features
    X, y = data[:,:n], data[:,n:]
    
    #plot data
    plotData(data)
    plt.savefig("scatterData2.pdf")
    
    #bdry looks nonlinear so enlarge feature space to degree 6 polynomial
    X = mapFeature(X[:,0:1],X[:,1:2],6)
    
    #initialize theta
    theta = np.zeros((np.shape(X)[1],1))
    #set regularization parameter
    lambbda = 1.0
    
    #test cost function on initial theta, should be 0.693
    print("cost function for initial theta = ",computeCostReg(theta,X,y,lambbda),"should be 0.693")  # get [[ 0.69314718]]
    
    #find minimum of cost function using BFGS
    result = op.minimize(computeCostReg, theta, args=(X,y,lambbda), method='BFGS', jac=computeCostDerivReg, options={'disp': True})
    theta = result.x  # note, result.x is a vector so below theta is a vector ie when passed to plotDecisionBdryNonlin
    cost = result.fun
    print("optimal theta =",theta)
    print("optimal cost funstion =",cost)
    
    #plot decision bdry
    plotDecisionBdryNonlin(data,X,theta,lambbda)
    
    #try other values of lambda
    lambbdas = [0.0,0.5,100]
    for lambbda in lambbdas:
        theta = np.zeros((np.shape(X)[1],1))
        #using TNC (truncated Newton) gives decision bdry similar to matlab's fmiunc for lambbda=0;
        #using BFGS gives different boundary then fminunc for lambbda=0 although fminunc uses BFGS;
        #for other tested values of lambda there is no difference between TNC and BFGS
        #result = op.minimize(computeCostReg, theta, args=(X,y,lambbda), method='BFGS', jac=computeCostDerivReg, options={'disp': True})
        result = op.minimize(computeCostReg, theta, args=(X,y,lambbda), method='TNC', jac=computeCostDerivReg, options={'disp': True})
        theta = result.x
        plotDecisionBdryNonlin(data,X,theta,lambbda)
        
    plt.show()
        