# -*- coding: utf-8 -*-
"""
Created on Mon Nov 16 15:24:48 2015

@author: Hanna
"""
import numpy as np
import matplotlib.pyplot as plt

def featureNormalize(X):
    mu = np.mean(X,axis=0)
    X_norm = X-mu
    sigma = np.std(X_norm,axis=0)
    X_norm = X_norm/sigma
    
    return X_norm, mu, sigma


def hypothesis(theta,X):
    #works for any # of features

    return np.dot(X,theta)


def computeCost(X,y,theta):  # J(theta)
    #works for any # of features
    m = len(y)
    h = hypothesis(theta,X)
    
    return (np.dot((h-y).T,h-y))/(2.0*m)


def computeCostDeriv(X,y,theta):  # partial derivatives of J(theta)
    #works for any # of features
    m = len(y)
    h = hypothesis(theta,X)
    
    return np.dot(X.T,h-y)/m


def gradientDescent(X,y,theta,alpha,iterations):
    #works for any # of features
    
    #J(theta) as a function of # of iterations
    costHistory = np.zeros((iterations+1,1))  # for debugging
    costHistory[0,0] = computeCost(X,y,theta)  # for debugging
    
    #gradient descent
    for iter in range(0,iterations):
        theta = theta - alpha*computeCostDeriv(X,y,theta)
        #or, without separately calculating derivatives
        #theta = theta - np.dot(X.T,hypothesis(theta,X)-y)*alpha/len(y)
        
        costHistory[iter+1,0] = computeCost(X,y,theta)  # for debugging
    
    return theta, costHistory


def normalEquation(X,y):#analytical solution for theta which minimizes J(theta)
    #works for any # of features
    
    return np.dot(np.dot(np.linalg.inv(np.dot(X.T,X)),X.T),y)

    
if __name__ == '__main__':
    
    data = np.genfromtxt("ex1data2.txt", delimiter=',')
    X, y = data[:,:2], data[:,2:]
    m = np.shape(y)[0]  # number of examples in training set
    n = np.shape(X)[1]  # number of features
    
    #normalize features
    #print(X[0:10,:])  # first ten examples of original data
    X, mu, sigma = featureNormalize(X)
    #print(X[0:10,:])  # first ten examples of normalized data
    
    #multivariate linear regression    
    X = np.hstack((np.ones((m,1)),X))  # add intercept column
    
    #try for many alphas to select optimal learning rate
    alphas = [1.0,0.3,0.1,0.03,0.01]
    iterations = 100  # 500
    #vector of # of iterations
    iter_num = np.ones((iterations+1,1))
    for i in range(0,iterations+1):
        iter_num[i,0] = i
    #plot of J(theta) vs # of iterations for alphas
    plt.figure()
    for alpha in alphas:
        theta = np.zeros((n+1,1))  # initial guess for theta
        theta, costHistory = gradientDescent(X,y,theta,alpha,iterations)
        plt.plot(iter_num[:,0],costHistory[:,0])
    plt.legend(['alpha=1.0', 'alpha=0.3', 'alpha=0.1', 'alpha=0.03', 'alpha=0.01'], loc='upper right')
    plt.ylabel('cost function')
    plt.xlabel('# of iterations')
    plt.savefig("cost_vs_iter_vs_alpha.pdf")
    plt.show()
    
    #select alpha = 0.3 seens to converge at 50 iterations
    alpha = 0.3
    theta = np.zeros((n+1,1))  # initial guess for theta
    theta, costHistory = gradientDescent(X,y,theta,alpha,iterations)
    
    # prediction for 1650 sq feet 3 bedroom house
    test = np.array([[1650.0, 3.0]])
    #normalize 
    test = (test-mu)/sigma
    #add intercept
    test = np.hstack((np.ones((1,1)),test))
    predict = hypothesis(theta,test)
    print("prediction for 1650 sq feet 3 bedroom house is",predict)
    
    #normal equation
    #do not need to normalize data
    X, y = data[:,0:2], data[:,2:3]
    X = np.hstack((np.ones((m,1)),X))  # add intercept column
    theta = normalEquation(X,y)
    # prediction for 1650 sq feet 3 bedroom house with added intercept
    test = np.array([[1.0,1650.0, 3.0]])
    predict = hypothesis(theta,test)
    print("prediction from normal equation for 1650 sq feet 3 bedroom house is",predict)
    
    #results: gradient descent:[[ 293081.47339913]], normal eq:[[ 293081.4643349]]
    #if use 500 iterations: gradient descent:[[ 293081.4643349]] coincides with normal eq
