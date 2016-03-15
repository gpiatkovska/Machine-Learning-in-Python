# -*- coding: utf-8 -*-
"""
Created on Mon Nov 30 16:44:28 2015

@author: Hanna
"""
import numpy as np
import scipy.io as io
import matplotlib.pyplot as plt
import scipy.misc as misc
from ex7 import kMeansInitCentroids, runkMeans
from mpl_toolkits.mplot3d import Axes3D

def featureNormalize(X):
    mu = np.mean(X,axis=0)
    X_norm = X-mu
    sigma = np.std(X_norm,axis=0,ddof=1) # to get the same results as matlab
    X_norm = X_norm/sigma
    
    return X_norm, mu, sigma
    
    
def pca(X):
    #covariance matrix
    Sigma = np.dot(X.T,X)/np.shape(X)[0]
    #singular value decomposition
    U, s, V = np.linalg.svd(Sigma)
    
    return U, s
    
    
def projectData(X, U, K):
    
    return np.dot(X, U[:,:K])
    
    
def recoverData(Z, U, K):
    
    return np.dot(Z, U[:,:K].T)
    
    
def displayData(X):
    pixels = np.sqrt(np.shape(X)[1]).astype(np.int)  # images are pixels by pixels size
    #images are shown on a  display_rows by display_cols square
    display_rows = np.sqrt(np.shape(X)[0]).astype(np.int)
    display_cols = display_rows
    #print(pixels,display_rows,display_cols)
    out = np.zeros((pixels*display_rows,pixels*display_cols))
    indices = range(0,display_rows*display_cols)
    for j in range(0, display_rows):
        for i in range(0, display_cols):
            start_i = i*pixels
            start_j = j*pixels
            out[start_i:start_i+pixels, start_j:start_j+pixels] = X[indices[display_rows*j+i]].reshape(pixels, pixels).T    
    
    return out


if __name__ == '__main__':
    
    #load and plot the example dataset
    mat = io.loadmat("ex7data1.mat")
    X = mat['X'] 
    
    plt.figure()
    plt.scatter(X[:,0], X[:,1], facecolors='none', edgecolors='b')
    plt.xlim([0.5, 6.5])
    plt.ylim([2, 8])
    plt.axes().set_aspect('equal')
    plt.savefig('ExampledataSet1.pdf')
    
    #PCA
    
    #normalize features
    X_norm, mu, sigma = featureNormalize(X)
    
    #run pca
    U, s = pca(X_norm)
    #print(np.shape(s)) 
    #print(np.shape(U)) # n by n, n is # of features
    
    #draw left singular vectors with lengths proportional to corresponding singular values
    #centered at mean of data to show the directions of maximum variations in the dataset
    plt.figure()
    plt.scatter(X[:,0], X[:,1], facecolors='none', edgecolors='b', s=20, linewidths=1)
    plt.plot([mu[0], mu[0] + 1.5*s[0]*U[0,0]],[mu[1], mu[1] + 1.5*s[0]*U[1,0]], color='k', linestyle='-', linewidth=2)
    plt.plot([mu[0], mu[0] + 1.5*s[1]*U[0,1]],[mu[1], mu[1] + 1.5*s[1]*U[1,1]], color='k', linestyle='-', linewidth=2)
    plt.xlim([0.5, 6.5])
    plt.ylim([2, 8])
    plt.axes().set_aspect('equal')
    plt.savefig('PCAdataSet1.pdf')
    print("left singular vector corresponding to largest singular value: ")
    print(U[:,0])
    print("should be about [-0.707 -0.707]")  # gives [-0.70710678 -0.70710678]
    
    #dimensionality reduction
    
    #project data into 1D
    K = 1
    Z = projectData(X_norm, U, K)
    print(Z[0], " should be about 1.481")  # gives [ 1.48127391] 
    
    #approximately recover original 2D data
    X_norm_rec  = recoverData(Z, U, K)
    print("recovered X_norm ", X_norm_rec[0], "should be about [-1.047 -1.047]")  # gives [-1.04741883 -1.04741883]
    print("original X_norm ", X_norm[0])
    print("original X ", X[0])
    
    #plot original data and projection
    plt.figure()
    plt.scatter(X_norm[:,0], X_norm[:,1], facecolors='none', edgecolors='b', s=20, linewidths=1)
    plt.scatter(X_norm_rec[:,0], X_norm_rec[:,1], facecolors='none', edgecolors='r', s=20, linewidths=1)
    plt.plot([X_norm[:,0], X_norm_rec[:,0]],[X_norm[:,1], X_norm_rec[:,1]], color='k', linestyle='--', linewidth=1)
    plt.xlim([-4, 3])
    plt.ylim([-4, 3])
    plt.axes().set_aspect('equal')
    plt.savefig('ProjectiondataSet1.pdf')
    plt.show()
    
    #faces dataset
    
    #load data
    mat = io.loadmat("ex7faces.mat")
    X = mat['X']  
    
    #display first 100 faces
    out = displayData(X[:100]) 
    fig = plt.figure()
    ax = fig.gca()
    ax.imshow(out,cmap="Greys_r")
    ax.set_axis_off()
    ax.set_title("Faces dataset")
    plt.savefig("100Faces.pdf")
    
    #normalize features
    X_norm, mu, sigma = featureNormalize(X)
    
    #display first 100 normalized faces
    out = displayData(X_norm[:100]) 
    fig = plt.figure()
    ax = fig.gca()
    ax.imshow(out,cmap="Greys_r")
    ax.set_axis_off()
    ax.set_title("Normalized faces dataset")
    plt.savefig("100FacesNorm.pdf")
    #look not as contrast as original
    
    #run pca on faces
    U, s = pca(X_norm)
    
    #display 36 principal "eigenfaces"
    out = displayData(U[:,:36].T)
    fig = plt.figure()
    ax = fig.gca()
    ax.imshow(out,cmap="Greys_r")
    ax.set_axis_off()
    ax.set_title("Principal components of faces dataset")
    plt.savefig("36Eigenfaces.pdf")
    
    #reduce dimension to 100
    K = 100
    Z = projectData(X_norm, U, K)
    
    #approximately recover data
    X_norm_rec  = recoverData(Z, U, K)
    #display first 100 normalized original and recovered faces
    fig = plt.figure(figsize=(10, 8), dpi=200)  # make large figure so that individual faces and quality difference can be recognized
    out = displayData(X_norm[:100])
    out_rec = displayData(X_norm_rec[:100])
    ax = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    ax.imshow(out,cmap="Greys_r")
    ax.set_title("Original faces")
    ax.set_axis_off()
    ax2.imshow(out_rec,cmap="Greys_r")
    ax2.set_title("Recovered faces")
    ax2.set_axis_off()
    plt.savefig("OriginalVsRecoveredFaces.pdf")
    plt.show()
    
    #PCA for visualizations
    
    #load the image
    A = misc.imread("bird_small.png")
    #divide by 255 so that all values are in the range [0,1]
    A = A/255.0
    #reshape into 128*128 (# of pixels) by 3 (RGB intensities) matrix
    X = A.reshape(np.shape(A)[0]*np.shape(A)[1], 3)
    
    K_c = 16  # 16 clusters
    max_iters = 10 
    
    #randomly initialize centroids
    initial_centroids = kMeansInitCentroids(X, K_c)
    #run K-means
    centroids, idx = runkMeans(X, initial_centroids, max_iters)
    
    #visualize centroid assignment (for the entire dataset)
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.scatter(X[:,0], X[:,1], X[:,2], c = idx, marker='.')
    plt.savefig("3DVisualization.pdf")
    
    #normalize features
    X_norm, mu, sigma = featureNormalize(X)
    #to get the plot in the same bounds as in Fig 11 of assignment text 
    #need only subtract mean during normalization (no division by std)
    #mu = np.mean(X,axis=0)
    #X_norm = X-mu
    
    #run pca
    U, s = pca(X_norm)
    
    #reduce dimension to 2D
    K = 2
    Z = projectData(X_norm, U, K)
    
    #print(X[:10], np.max(X),np.min(X),np.mean(X),np.std(X))
    #print(X_norm[:10], np.max(X_norm),np.min(X_norm),np.mean(X),np.std(X_norm))
    #print(Z[:10], np.max(Z),np.min(Z),np.mean(X),np.std(Z))
    
    #visualize centroid assignment for 2D data
    fig = plt.figure()
    ax = fig.gca()
    ax.scatter(Z[:,0], Z[:,1], c = idx, marker='.')
    #the plot is flipped wrt both axes compared to the one provided in the assignment text,
    #evidentely we got negative singular vectors compared to assignment text 
    #(which is OK since U and -U are equally valid)
    #so using -U, and hence -Z gives the same plot as in the text
    #ax.scatter(-Z[:,0], -Z[:,1], c = idx, marker='.')
    plt.savefig("2DVisualization.pdf")
    plt.show()
    