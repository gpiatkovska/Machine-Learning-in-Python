# -*- coding: utf-8 -*-
"""
Created on Thu Nov 26 14:15:32 2015

@author: Hanna
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as io
from sklearn import svm

def plotData(data):
    plt.figure()
    
    pos = data[data[:,2] == 1]
    neg = data[data[:,2] == 0]

    plt.scatter(pos[:,0], pos[:,1], c='k', marker='+', s=40, linewidths=2)
    plt.scatter(neg[:,0], neg[:,1], c='y', marker='o', s=40, linewidths=1)
    
    
def visualizeBoundaryLinear(X,y,clf):
    #get the separating hyperplane
    w = clf.coef_[0]
    a = -w[0] / w[1]
    xx = np.linspace(min(X[:, 0]), max(X[:, 0]), 100)
    yy = a * xx - (clf.intercept_[0]) / w[1]
    plotData(np.hstack((X,y)))  # plot the examples
    plt.plot(xx, yy, 'b-')  # plot the boundary
    
    
def visualizeBoundary(X,y,clf):  
    h = .002  # step size in the mesh
    # create a mesh to plot in
    x_min, x_max = X[:, 0].min(), X[:, 0].max()
    y_min, y_max = X[:, 1].min(), X[:, 1].max()
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),np.arange(y_min, y_max, h))

    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, m_max]x[y_min, y_max].
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plotData(np.hstack((X,y)))  # plot the examples
    #plt.contour(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.8)
    plt.contour(xx, yy, Z, colors='blue')  # plot the boundary
    
    
def gaussianKernel(x1,x2,sigma):
    K = np.zeros((np.shape(x1)[0],np.shape(x2)[0]))
    for i in range(np.shape(x1)[0]):
        for j in range(np.shape(x2)[0]):
            K[i,j] = np.exp(-np.dot((x1[i]-x2[j]),(x1[i]-x2[j]).T)/(2.0*sigma**2.0))
    
    return K
    
  
def dataset3Params(X,y,X_val,y_val):
    C_vals = np.array([0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30])
    sigma_vals = np.array([0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30])
    scores = np.zeros((len(C_vals),len(sigma_vals)))
    for i in range(0,len(C_vals)):
        for j in range(0,len(sigma_vals)):
            gamma = 1.0/(2.0*sigma_vals[j]**2)
            clf = svm.SVC(kernel='rbf',gamma=gamma,C=C_vals[i])
            clf.fit(X,y.ravel())  # train svm on training set
            scores[i,j] = clf.score(X_val,y_val.ravel())  # mean prediction accuracy on validation set
    i,j = np.unravel_index(scores.argmax(), scores.shape)  # find indices for which accuracy is max

    return C_vals[i], sigma_vals[j]
    

if __name__ == '__main__':
 
    #load the data from dataset 1
    mat = io.loadmat("ex6data1.mat")
    X, y = mat['X'], mat['y']
    
    #plot the data
    data = np.hstack((X,y))
    plotData(data)
    plt.xlim(0, 4.5)
    plt.ylim(1.5, 5)
    plt.savefig("dataSet1.pdf")
    
    #train SVM with linear kernel and C = 1
    svc = svm.SVC(kernel='linear',C=1)
    svc.fit(X,y.ravel())
    #plot decision boundary
    visualizeBoundaryLinear(X,y,svc)
    plt.xlim(0, 4.5)
    plt.ylim(1.5, 5)
    plt.savefig("dataSet1DecisionC1.pdf")
    
    #train SVM with linear kernel and C = 100
    svc = svm.SVC(kernel='linear',C=100)
    svc.fit(X,y.ravel())
    #plot decision boundary
    visualizeBoundaryLinear(X,y,svc)
    plt.xlim(0, 4.5)
    plt.ylim(1.5, 5)
    plt.savefig("dataSet1DecisionC100.pdf")
    
    #test gaussian kernel
    #note that sklearn library requires custom kernels to be implemented differently
    #than described in the assignment text
    x1 = np.array([1, 2, 1]).reshape(1,3)
    x2 = np.array([0, 4, -1]).reshape(1,3)
    sigma = 2.0
    print("Gaussian kernel between x1 = [1; 2; 1], x2 = [0; 4; -1] with sigma = 2 is ")
    print(gaussianKernel(x1,x2,sigma),", should be 0.324652")  # get [[ 0.32465247]]
    
    #load the data from dataset 2
    mat = io.loadmat("ex6data2.mat")
    X, y = mat['X'], mat['y']
    #print(np.shape(X))
    #print(np.shape(y))
    
    #plot the data
    data = np.hstack((X,y))
    plotData(data)
    plt.xlim(0, 1.0)
    plt.ylim(0.4, 1.0)
    plt.savefig("dataSet2.pdf")
    
    #train SVM with gaussian kernel
    C = 1.0
    sigma = 0.1
    #use custom Gaussian kernel, much slower than sklearn's kernel='rbf'
    mykernel = lambda x1,x2: gaussianKernel(x1,x2,sigma) 
    svc = svm.SVC(kernel=mykernel,C=C)
    svc.fit(X,y.ravel())
    #plot decision boundary
    visualizeBoundary(X,y,svc)
    plt.xlim(0, 1.0)
    plt.ylim(0.4, 1.0)
    plt.savefig("dataSet2DecisionCustom.pdf")
    #use ready rbf kernel
    gamma = 1.0/(2.0*sigma**2)
    svc = svm.SVC(kernel='rbf',gamma=gamma,C=C)
    svc.fit(X,y.ravel())
    #plot decision boundary
    visualizeBoundary(X,y,svc)
    plt.xlim(0, 1.0)
    plt.ylim(0.4, 1.0)
    plt.savefig("dataSet2Decision.pdf")
    
    #load the data from dataset 3
    mat = io.loadmat("ex6data3.mat")
    X, y = mat['X'], mat['y']
    X_val, y_val = mat['Xval'], mat['yval']
    
    #plot the data
    data = np.hstack((X,y))
    plotData(data)
    plt.xlim(-0.6, 0.3)
    plt.ylim(-0.8, 0.6)
    plt.savefig("dataSet3.pdf")
    
    #find the best parameters using validation set
    C, sigma = dataset3Params(X,y,X_val,y_val)
    print("best C = ",C)
    print("best sigma = ",sigma)
    
    #train SVM with gaussian kernel and found best parameters
    gamma = 1.0/(2.0*sigma**2)
    svc = svm.SVC(kernel='rbf',gamma=gamma,C=C)
    svc.fit(X,y.ravel())
    #plot decision boundary
    visualizeBoundary(X,y,svc)
    plt.xlim(-0.6, 0.3)
    plt.ylim(-0.8, 0.6)
    plt.savefig("dataSet3Decision.pdf")
    
    plt.show()
