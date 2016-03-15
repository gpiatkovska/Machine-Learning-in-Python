# -*- coding: utf-8 -*-
"""
Created on Tue Nov 24 18:33:09 2015

@author: Hanna
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as io
import scipy.optimize as op

def plotData(X,y):
    plt.figure(1)
    plt.plot(X, y, 'rx', markersize=6)
    plt.title('Training data')
    plt.ylabel('Water flowing out of the dam(y)')
    plt.xlabel('Change in water level(x)')
    plt.savefig("data.pdf")


def plotLinReg(X,y,theta):
    plt.figure(2)
    plt.plot(X[:,1], y, 'rx', markersize=6)  # do not plot added column of ones
    plt.plot(X[:,1], hypothesis(theta,X), 'b-')
    plt.title('Linear regression fit')
    plt.ylabel('Water flowing out of the dam(y)')
    plt.xlabel('Change in water level(x)')
    plt.savefig("linRegFit.pdf")


def hypothesis(theta,X):

    return np.dot(X,theta)


def linearRegCostFunction(X,y,theta,lambbda):  # J(theta)
    m = np.shape(X)[0]  # # of examples
    n = np.shape(X)[1]  # # of features including itercept; in main n is the # of original features
    
    theta = theta.reshape(n,1)  # optimization routines use vector theta
    h = hypothesis(theta,X)
    
    #compute cost
    
    cost = (np.dot((h-y).T,h-y))/(2.0*m) + lambbda*np.dot(theta[1:].T,theta[1:])/(2.0*m)
    
    #compute gradient
    
    gradient = np.zeros(np.shape(theta))
    
    gradient = np.dot(X.T,h-y)/m + lambbda*theta/m
    
    gradient[0,0] = gradient[0,0] - lambbda*theta[0,0]/m
    
    #alternatively
    #gradient = np.dot(X.T,h-y)/m
    #gradient[1:] = gradient[1:] + lambbda*theta[1:]/m
    
    return (cost.flatten(), gradient.flatten())
    
    
def trainLinearReg(X,y,lambbda):
    #initialize theta
    theta = np.zeros((np.shape(X)[1],1))
    
    #function and gradient to pass to the optimization routine
    cost = lambda theta: linearRegCostFunction(X,y,theta,lambbda)[0]
    grad = lambda theta: linearRegCostFunction(X,y,theta,lambbda)[1]
    
    #minimize using nonlinear conjugate gradient analogous to matlab fmincg
    result = op.minimize(cost, theta, method='CG', jac=grad, options={'maxiter': 4000})#,'disp': True})
    #result = op.minimize(cost, theta, method='CG', options={'maxiter': 4000,'disp': True})  # use numerical gradient
    #result = op.minimize(cost, theta, method='BFGS', jac=grad, options={'maxiter': 4000,'disp': True})
    #result = op.minimize(cost, theta, method='COBYLA', options={'maxiter': 4000,'disp': True})
    theta = result.x
    
    return theta
    
    
def learningCurve(X,y,X_val,y_val,lambbda):
    m = len(y)  # # of examples in training set
    error_train = np.zeros(m)
    error_val = np.zeros(m)

    for i in range(0,m):
        Xset = X[0:i+1,:]
        yset = y[0:i+1,:]
        theta = trainLinearReg(Xset,yset,lambbda)
        error_train[i] = linearRegCostFunction(Xset,yset,theta,0)[0]  # set lambbda to zero
        error_val[i] = linearRegCostFunction(X_val,y_val,theta,0)[0]  # set lambbda to zero

    return error_train, error_val
    
    
def learningCurveRand(X,y,X_val,y_val,lambbda):  # uses bootstrap
    m = len(y)  # # of examples in training set
    error_train = np.zeros(m)
    error_val = np.zeros(m)

    for i in range(0,m):
        for k in range(0,50):
            ind = np.random.choice(X.shape[0], i+1)
            ind_val = np.random.choice(X_val.shape[0], i+1)
            Xset = X[ind,:]
            yset = y[ind,:]
            Xset_val = X_val[ind_val,:]
            yset_val = y_val[ind_val,:]
            theta = trainLinearReg(Xset,yset,lambbda)
            error_train[i] = error_train[i] + linearRegCostFunction(Xset,yset,theta,0)[0]  # set lambbda to zero
            error_val[i] = error_val[i] + linearRegCostFunction(Xset_val,yset_val,theta,0)[0]  # set lambbda to zero

    return error_train/50.0, error_val/50.0
    
    
def polyFeatures(X,p):
    out = X
    for i in range(1,p):
        out = np.hstack((out,(X**(i+1))))
            
    return out


def featureNormalize(X):
    mu = np.mean(X,axis=0)
    X_norm = X-mu
    sigma = np.std(X_norm,axis=0)
    X_norm = X_norm/sigma
    
    return X_norm, mu, sigma
    
    
def plotFit(min_x,max_x,mu,sigma,theta,p):
    x = np.linspace(min_x-15,max_x+25,(max_x+25-(min_x-15))/0.05)
    x = x.reshape(len(x),1)
    x_poly = polyFeatures(x,p)
    x_poly = (x_poly - mu)/sigma
    x_poly = np.hstack((np.ones((np.shape(x)[0],1)),x_poly))
    plt.plot(x, hypothesis(theta,x_poly), 'b--')
    plt.ylabel('Water flowing out of the dam(y)')
    plt.xlabel('Change in water level(x)')
    

def validationCurve(X,y,X_val,y_val):
    lambbda_vec = np.array([0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0])
    error_train = np.zeros(np.shape(lambbda_vec))
    error_val = np.zeros(np.shape(lambbda_vec))

    for i in range(0,len(lambbda_vec)):
         theta = trainLinearReg(X,y,lambbda_vec[i])
         error_train[i] = linearRegCostFunction(X,y,theta,0)[0]  # set lambbda to zero
         error_val[i] = linearRegCostFunction(X_val,y_val,theta,0)[0]  # set lambbda to zero

    return lambbda_vec, error_train, error_val
    
    
if __name__ == '__main__':
    
    #load the data
    mat = io.loadmat("ex5data1.mat")
    X, y = mat['X'], mat['y']
    X_val, y_val = mat['Xval'], mat['yval']
    X_test, y_test = mat['Xtest'], mat['ytest']
    
    #visualize the training data
    plotData(X,y)
    
    #linear regression
    
    m = len(y) # number of examples in training set
    #add bias column
    X = np.hstack((np.ones((m,1)),X))
    
    #initialize theta
    theta = np.array([[1,1]]).T
    #regularization parameter
    lambbda = 1.0
    
    #debug J(theta)
    cost = linearRegCostFunction(X,y,theta,lambbda)[0]
    print("cost for initial theta =",cost,"should be about 303.993 for this dataset")  # gives [ 303.99319222]
    gradient = linearRegCostFunction(X,y,theta,lambbda)[1]
    print("gradient for initial theta =",gradient,"should be about [-15.30; 598.250] for this dataset")  # gives [ -15.30301567  598.25074417]
    
    #train linear regression with lambda = 0
    lambbda = 0.0
    theta = trainLinearReg(X,y,lambbda)
    plotLinReg(X,y,theta)
    
    #learning curve
    m_val = len(y_val) # number of examples in validation set
    #add bias column
    X_val = np.hstack((np.ones((m_val,1)),X_val))
    #calculate training and validation errors
    error_train, error_val = learningCurve(X,y,X_val,y_val,lambbda)
    #plot learning curves
    examples_num = np.zeros(m)
    for i in range(0,m):
        examples_num[i] = i+1
    plt.figure(3)
    plt.plot(examples_num, error_train, 'b-')
    plt.plot(examples_num, error_val, 'g-')
    plt.xlim([0,13])
    plt.ylim([0,150])
    plt.legend(['Train', 'Validation'], loc='upper right')
    plt.title('Learning curve for linear regression')
    plt.ylabel('Error')
    plt.xlabel('Number of training examples')
    plt.savefig("linRegLearnCurve.pdf")
    
    #polynomial regression
    
    #map and normalize features
    p = 8
    
    #training set
    X_poly 	= polyFeatures(X[:,1:],p)  # do not map bias column
    X_poly, mu, sigma 	= featureNormalize(X_poly)
    X_poly 	= np.hstack((np.ones((m,1)),X_poly))  # add bias
    
    #validation set
    X_poly_val = polyFeatures(X_val[:,1:],p)  # do not map bias column
    #normalize using mu and sigma from training set
    X_poly_val = X_poly_val - mu
    X_poly_val = X_poly_val/sigma
    X_poly_val 	= np.hstack((np.ones((m_val,1)),X_poly_val))  # add bias
    
    #test set
    X_poly_test = polyFeatures(X_test,p)  # X_test is not biased yet
    #normalize using mu and sigma from training set
    X_poly_test = X_poly_test - mu
    X_poly_test = X_poly_test/sigma
    m_test = len(y_test) # # of test examples
    X_poly_test 	= np.hstack((np.ones((m_test,1)),X_poly_test))  # add bias
    
    #mapped and normalized first training example
    print(X_poly[0, :])
    
    #train polynomial regression with lambda = 0
    theta = trainLinearReg(X_poly,y,lambbda)
    #print(theta)
    plt.figure(4)
    plt.plot(X[:,1], y, 'rx', markersize=6)
    plotFit(np.min(X[:,1:]),np.max(X[:,1:]),mu,sigma,theta,p)
    plt.title('Polynomial regression fit, lambda = 0 ')
    plt.savefig("PolyRegFit_lambda0.pdf")
    #Note that this plot is not the same as provided in the assigmnent (however for the cases lambda=1   
    #and lambda=100 (below), as well as for the linear regression the fits are exactly the same as given in the assignment text);
    #this is apparently due to the difference in the matlab and python fmincg (conjugate gradient) implementation;
    #using method='COBYLA' instead of CG gives the fit and the learning curve very similar to the ones in the assignment
    #(although cobyla is a quite different method and does not use gradient info);
    #other methods (TNC,BFGS,Newton-CG,L-BFGS-B,SLSQP,Powell) give fits similar to CG 
    #(Nelder-Mead gives a fit different from both cobyla and CG).
    #Also, sometimes "Warning: Desired error not necessarily achieved due to precision loss" appears when using CG,
    #however BFGS does not have this problem and overall produces nicer learning curves (e.g. without peaks at 11 training examples for lambbda=1).
    #The confidence in the supplied gradient (jac=grad) correctness is established by allowing gradient calculation
    #numerically, i.e., result = op.minimize(cost, theta, method='CG', options={'maxiter': 4000,'disp': True})
    #gives the same results (including the warning).
    
    #calculate training and validation errors
    error_train, error_val = learningCurve(X_poly,y,X_poly_val,y_val,lambbda)
    #plot learning curves
    plt.figure(5)
    plt.plot(examples_num, error_train, 'b-')
    plt.plot(examples_num, error_val, 'g-')
    plt.xlim([0,13])
    plt.ylim([0,100])
    plt.legend(['Train', 'Validation'], loc='upper right')
    plt.title('Learning curve for polynomial regression, lambda=0')
    plt.ylabel('Error')
    plt.xlabel('Number of training examples')
    plt.savefig("PolyRegLearnCurve_lambda0.pdf")
    
    #train polynomial regression with lambda = 1.0
    lambbda = 1.0
    theta = trainLinearReg(X_poly,y,lambbda)
    #print(theta)
    plt.figure(6)
    plt.plot(X[:,1], y, 'rx', markersize=6)
    plotFit(np.min(X[:,1:]),np.max(X[:,1:]),mu,sigma,theta,p)
    plt.title('Polynomial regression fit, lambda = 1.0 ')
    plt.savefig("PolyRegFit_lambda1.pdf")
    #calculate training and validation errors
    error_train, error_val = learningCurve(X_poly,y,X_poly_val,y_val,lambbda)
    #plot learning curves
    plt.figure(7)
    plt.plot(examples_num, error_train, 'b-')
    plt.plot(examples_num, error_val, 'g-')
    plt.xlim([0,13])
    plt.ylim([0,100])
    plt.legend(['Train', 'Validation'], loc='upper right')
    plt.title('Learning curve for polynomial regression, lambda=1.0')
    plt.ylabel('Error')
    plt.xlabel('Number of training examples')
    plt.savefig("PolyRegLearnCurve_lambda1.pdf")
    
    #train polynomial regression with lambda = 100.0
    lambbda=100.0
    theta = trainLinearReg(X_poly,y,lambbda)
    #print(theta)
    plt.figure(8)
    plt.plot(X[:,1], y, 'rx', markersize=6)
    plotFit(np.min(X[:,1:]),np.max(X[:,1:]),mu,sigma,theta,p)
    plt.title('Polynomial regression fit, lambda = 100.0 ')
    plt.savefig("PolyRegFit_lambda100.pdf")
    #calculate training and validation errors
    error_train, error_val = learningCurve(X_poly,y,X_poly_val,y_val,lambbda)
    #plot learning curves
    plt.figure(9)
    plt.plot(examples_num, error_train, 'b-')
    plt.plot(examples_num, error_val, 'g-')
    plt.legend(['Train', 'Validation'], loc='upper right')
    plt.title('Learning curve for polynomial regression, lambda=100.0')
    plt.ylabel('Error')
    plt.xlabel('Number of training examples')
    plt.savefig("PolyRegLearnCurve_lambda100.pdf")
    
    #plot validation curve for selecting lambbda
    lambbda_vec, error_train, error_val = validationCurve(X_poly,y,X_poly_val,y_val)
    print(error_train)
    print(error_val)
    plt.figure(10)
    plt.plot(lambbda_vec, error_train, 'b-')
    plt.plot(lambbda_vec, error_val, 'g-')
    plt.xlim([0,10])
    plt.ylim([0,20])
    plt.legend(['Train', 'Validation'], loc='upper right')
    plt.title('Validation curve for polynomial regression')
    plt.ylabel('Error')
    plt.xlabel('lambda')
    plt.savefig("PolyRegValidCurve.pdf")
    #the resulting optimal lambbda is 3.0 regardless of the used optimization method with 
    #corresponding minimal validation error of 3.83216872 using CG, 3.83218137 for CG with numerical grad,
    #and 3.83211428 using COBYLA, 3.83218056 using BFGS
    
    #compute the test set error using lambbda = 3.0
    lambbda = 3.0
    theta = trainLinearReg(X_poly,y,lambbda)
    error_test = linearRegCostFunction(X_poly_test,y_test,theta,0)[0]  # set lambbda to zero
    print("test error =",error_test," for lambda = 3.0, should be 3.8599 for this dataset")
    # gives 3.57202245 (CG), 3.572016053 (CG with numerical grad), 3.572024 (BFGS), 3.57190019(cobyla)
    
    #plot learning curves for lambda = 0.01 using bootstrap
    lambbda = 0.01
    error_train, error_val = learningCurveRand(X_poly,y,X_poly_val,y_val,lambbda)
    plt.figure(11)
    plt.plot(examples_num, error_train, 'b-')
    plt.plot(examples_num, error_val, 'g-')
    plt.xlim([0,13])
    plt.ylim([0,100])
    plt.legend(['Train', 'Validation'], loc='upper right')
    plt.title('Learning curve for polynomial regression, lambda=0.01')
    plt.ylabel('Error')
    plt.xlabel('Number of training examples')
    plt.savefig("PolyRegLearnCurve_lambda001.pdf")
    
    plt.show()
      