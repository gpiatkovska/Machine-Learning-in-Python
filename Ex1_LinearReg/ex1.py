# -*- coding: utf-8 -*-
"""
Created on Mon Nov 16 15:24:48 2015

@author: Hanna
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

def warmUpExercise():
    A = np.eye(5)
    print(A)


def plotData(X,y):
    plt.figure(1)
    plt.plot(X, y, 'rx', markersize=5)
    plt.xlim([4, 24])
    plt.ylim([-5, 25])
    plt.ylabel('Profit in $10,000s')
    plt.xlabel('Population of City in 10,000s')
    plt.savefig("data.pdf")


def plotLinReg(X,y,theta):
    plt.figure(3)
    plt.plot(X[:,1], y, 'rx', markersize=5)  # do not plot added column of ones
    plt.plot(X[:,1], hypothesis(theta,X), 'b-')
    plt.xlim([4, 24])
    plt.ylim([-5, 25])
    plt.legend(['Training data', 'Linear regression'], loc='lower right', numpoints=1)
    plt.ylabel('Profit in $10,000s')
    plt.xlabel('Population of City in 10,000s')
    plt.savefig("linRegFit.pdf")


def hypothesis(theta,X):

    return np.dot(X,theta)


def computeCost(X,y,theta):  # J(theta)
    m = len(y)
    h = hypothesis(theta,X)
    
    return (np.dot((h-y).T,h-y))/(2.0*m)


def gradientDescent(X,y,theta,alpha,iterations):
    m = len(y)
    
    costHistory = np.zeros((iterations+1,1))  # for debugging
    costHistory[0,0] = computeCost(X,y,theta)  # for debugging
    
    for iter in range(0,iterations):
        theta = theta - np.dot(X.T,hypothesis(theta,X)-y)*alpha/m
        
        costHistory[iter+1,0] = computeCost(X,y,theta)  # for debugging
    
    #plot J(theta) as a function pf # of iterations -- should monotonically decrease
    ind = np.ones((iterations+1,1))  # for debugging
    for i in range(0,iterations+1):  # for debugging
        ind[i,0] = i  # for debugging
    plt.figure(2)  # for debugging
    plt.plot(ind[:,0],costHistory[:,0],'ro')  # for debugging
    plt.ylabel('cost function')  # for debugging
    plt.xlabel('# of iterations')  # for debugging
    plt.savefig("cost_vs_iter.pdf")  #for debugging
    
    return theta


def visualCost(X,y,theta):
    theta0_vals = np.linspace(-10, 10, 500)
    theta1_vals = np.linspace(-1, 4, 500)
    J_vals = np.zeros((len(theta0_vals),len(theta1_vals)))
    for i, elem0 in enumerate(theta0_vals):
        for j, elem1 in enumerate(theta1_vals):
            theta_vals = np.array([[elem0], [elem1]])
            J_vals[i, j] = computeCost(X, y, theta_vals)       
    theta0_vals, theta1_vals = np.meshgrid(theta0_vals,theta1_vals)

    J_vals = J_vals.T
    #surface plot
    fig4 = plt.figure()
    ax = fig4.gca(projection='3d')
    ax.plot_surface(theta0_vals, theta1_vals, J_vals, cmap=cm.jet)
    ax.view_init(azim = 180+40,elev = 25)
    plt.xlabel("theta_0")
    plt.ylabel("theta_1")
    plt.savefig("cost3D.pdf")
    
    #contour plot
    plt.figure(5)
    plt.contour(theta0_vals, theta1_vals, J_vals, levels=np.logspace(-2, 3, 20))
    plt.scatter(theta[0,0], theta[1,0],  c='r', marker='x', s=20, linewidths=1)  # mark converged minimum
    plt.xlim([-10, 10])
    plt.ylim([-1, 4])
    plt.xlabel("theta_0")
    plt.ylabel("theta_1")
    plt.savefig("costContour.pdf")
    
    
if __name__ == '__main__':
    
    warmUpExercise()
    
    #load the data
    data = np.genfromtxt("ex1data1.txt", delimiter=',')
    X, y = data[:,:1], data[:,1:]
    m = len(y)  # number of examples in training set
    
    #see what the data looks like
    plotData(X,y)
    
    #linear regression
    X = np.hstack((np.ones((m,1)),X))
    
    theta = np.zeros((2, 1))  # initial guess for theta
    iterations = 1500
    alpha = 0.01
    
    cost = computeCost(X,y,theta)  # debug J(theta)
    print("cost for initial theta =",cost,"should be 32.07 for this dataset")  # get [[ 32.07273388]]
    
    theta = gradientDescent(X,y,theta,alpha,iterations)  # contains plot of J(theta) vs #iterations for debugging
    cost = computeCost(X, y, theta)  # minimized J(theta)
    print("optimal theta =",theta)  # get [[-3.63029144] [ 1.16636235]]
    print("optimal cost =",cost)  # get [[ 4.48338826]]
    
    #plot the resulting model
    plotLinReg(X,y,theta)
    
    #use model to make predictions
    predict1 = np.dot(np.array([[1,3.5]]),theta)
    predict2 = np.dot(np.array([[1,7]]),theta)
    print("prediction1 =",predict1)  # get [[ 0.45197679]]
    print("prediction2 =",predict2)  # get [[ 4.53424501]]
    
    #visualize cost function
    visualCost(X,y,theta)

    plt.show()
    