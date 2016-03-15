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
    plt.legend(['Admitted', 'Not admitted'], loc='upper right', prop={'size':10}, scatterpoints=1)
    plt.xlabel("Exam 1 score")
    plt.ylabel("Exam 2 score")
    plt.xlim([28, 102])
    plt.ylim([28, 102])
    
    
def plotDecisionBoundary(data,X,theta):
    plotData(data)
    #decision boundary is a line where hypothesis=0.5
    #i.e. sigmoid(theta0+theta1*x1+theta2*x2)=0.5
    #i.e. theta0+theta1*x1+theta2*x2=0
    #x1 -- exam1, need only two points since decision bdry is a line:
    plt.plot_x = np.array([min(X[:,1]),max(X[:,1])])
    #x2 -- exam2:
    plt.plot_y = (-theta[0]-theta[1]*plt.plot_x)/theta[2]
    plt.plot(plt.plot_x, plt.plot_y,'b-',linewidth=1)
    plt.savefig("LogisticRegFit.pdf")
    
       
def sigmoid(z):
    #works elementwise for any array z
    
    return sp.special.expit(z)  # expit(x) = 1/(1+exp(-x)) elementwise for array x
    
    
def hypothesis(theta,X):
    #works for any # of features

    return sigmoid(np.dot(X,theta))
 
 
def computeCost(theta,X,y):  # J(theta)
    #works for any # of features
    m = len(y)
    h = hypothesis(theta,X)
    cost = (-np.dot(y.T,sp.log(h))-np.dot((1.0-y).T,sp.log(1.0-h)))/m
    #check for nans: can occur since have log
    #if skip this, precision loss occurs
    if np.isnan(cost):
        return np.inf
        
    return cost
    
    
def computeCostDeriv(theta,X,y):  # partial derivatives of J(theta)
    #works for any # of features
    n = np.shape(X)[1]  # # of features including itercept; in main n is the # of original features
    m = np.shape(X)[0]
    #minimize routine changes dimensions of theta from matrix to vector, 
    #i.e from (3,1) to (3,) in this case
    #so need to reshape theta to original matrix dimensions
    #since in computeCostDeriv routine it is important that theta and thus hypothesis is matrix
    #if theta is a vector dot product in hypothesis works anyway but resulting hypothesis is a vector
    # then h-y has wrong dimension, (100,100) in this case: h is (100,), y is (100,1)
    #does not matter for computeCost routine
    theta = theta.reshape(n,1)
    h = hypothesis(theta,X)
    
    return (np.dot(X.T,h-y)/m).flatten()  # minimize needs vector of gradients not matrix

    
def predict(theta,X):
    if hypothesis(theta,X) >= 0.5:
        prediction = 1 
    else:
        prediction = 0
      
    return prediction


if __name__ == '__main__':
 
    data = np.genfromtxt("ex2data1.txt", delimiter=',')
    m = np.shape(data)[0]  # number of examples in training set
    n = np.shape(data)[1]-1  # number of features
    X, y = data[:,:n], data[:,n:]
    
    plotData(data)
    plt.savefig("scatterData.pdf")
    
    #test sigmoid
    #print(sigmoid(-100))#should be close to 0
    #print(sigmoid(100))#should be close to 1
    #print(sigmoid(0))#should equal 0.5
    
    X = np.hstack((np.ones((m,1)),X))  # add intercept column
    theta = np.zeros((n+1,1))  # initial guess for theta
    
    #test cost function, should be 0.693
    print("cost function for initial theta = ",computeCost(theta,X,y),"should be 0.693")  # get [[ 0.69314718]]
    
    #find minimum of cost function using BFGS
    result = op.minimize(computeCost, theta, args=(X,y), method='BFGS', jac=computeCostDeriv, options={'disp': True})
    theta = result.x  # note, result.x is a vector so below theta is a vector eg when passed to plotDecisionBoundary
    cost = result.fun
    print("optimal theta =",theta)  # get [-25.16133446   0.20623173   0.20147159]
    print("optimal cost funstion =",cost,"should be 0.203")  # get 0.20349770158945538
    
    #plot decision bdry
    plotDecisionBoundary(data,X,theta)
    
    #make a prediction
    #do not need to reshape theta to matrix form theta.reshape(n+1,1)
    test_student = [[1,45,85]]  # try also [[1,45,85],[1,40,50],[1,70,90]]
    print("if scores are 45 for exam1 and 85 for exam2, the admission probability is")
    print(hypothesis(theta,test_student),"should be 0.776")  # get [ 0.77629051]
    
    plt.show()
