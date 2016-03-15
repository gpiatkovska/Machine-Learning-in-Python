# -*- coding: utf-8 -*-
"""
Created on Mon Nov 30 16:44:28 2015

@author: Hanna
"""
import numpy as np
import scipy.io as io
import matplotlib.pyplot as plt
import scipy.misc as misc

def findClosestCentroids(X, centroids):
    m = np.shape(X)[0]
    idx = np.zeros(m)
    K = np.shape(centroids)[0]

    for i in range(0,m):
        min_distance  = np.float("inf")
        for j in range(0,K):
            distance_j = np.dot(X[i]-centroids[j],(X[i]-centroids[j]).T)
            if distance_j < min_distance:
                min_distance = distance_j
                idx[i] = j+1
              
    return idx
    
            
def computeCentroids(X, idx, K): 
    n = np.shape(X)[1]
    centroids = np.zeros((K, n))
          
    for i in range(0,K):
        centroids[i] = np.mean(X[np.where(idx == i+1)],axis=0)
                
    return centroids
    
    
def runkMeans(X, initial_centroids, max_iters, plot = False):
    K = np.shape(initial_centroids)[0]
    centroids = np.copy(initial_centroids)
    previous_centroids = np.copy(centroids)
    
    if plot == True:
        plt.ion()
        plt.figure()
        
    for i in range(0, max_iters):
        idx = findClosestCentroids(X,centroids)
        
        if plot == True:
            plt.scatter(X[:,0], X[:,1], c = idx, facecolors='none')
            plt.scatter(centroids[:,0], centroids[:,1], c = 'k', marker = 'x', s=40, linewidths=2)
            for j in range(0,K):
                plt.plot([centroids[j,0], previous_centroids[j,0]],[centroids[j,1], previous_centroids[j,1]], color='k', linestyle='-', linewidth=2)
            plt.title("Iteration Number " + str(i+1))
            input("Press Enter to continue...") 
            previous_centroids = np.copy(centroids)
 
        centroids = computeCentroids(X,idx,K)
        
    if plot == True:
        plt.savefig("centroids_data2.pdf")
        
    return centroids, idx
    
    
def kMeansInitCentroids(X, K):
 
	return np.random.permutation(X)[0:K]
 

if __name__ == '__main__':
    
    #load the example dataset
    mat = io.loadmat("ex7data2.mat")
    X = mat['X']  # m by n matrix: m -- # of examples, n -- dimention of data 
    
    K = 3  # # of centroids (classes)

    initial_centroids = np.array([[3, 3], [6, 2], [8, 5]])  # K by n matrix
    #print(np.shape(initial_centroids))

    #find centroids assignment using initial centroids
    idx = findClosestCentroids(X, initial_centroids)
    print("Centroid assignment: ", idx[0:3], "should be [1 3 2]")  # get [ 1.  3.  2.]
    
    #find new centroids based on assignment
    centroids = computeCentroids(X, idx, K)
    print("New centroids:")
    print(centroids)
    print("should be [[ 2.428301 3.157924 ] [ 5.813503 2.633656 ] [ 7.119387 3.616684 ]")
    # get [[ 2.42830111  3.15792418] [ 5.81350331  2.63365645] [ 7.11938687  3.6166844 ]]
    
    #run K-means with 10 iterations
    max_iters = 10
    runkMeans(X, initial_centroids, max_iters, plot = True)
    
    #test random initialization
    print("test random initialization: ", kMeansInitCentroids(X, K))
    
    #image compression
    
    #load the image
    A = misc.imread("bird_small.png")
    #divide by 255 so that all values are in the range [0,1]
    A = A/255.0
    #reshape into 128*128 (# of pixels) by 3 (RGB intensities) matrix
    X = A.reshape(np.shape(A)[0]*np.shape(A)[1], 3)
    
    K = 16 # 3
    max_iters = 10 # 20 # 50 #the quality of compressed image increases with max_iters
    
    #randomly initialize centroids
    initial_centroids = kMeansInitCentroids(X, K)
    #run K-means
    centroids, idx = runkMeans(X, initial_centroids, max_iters)
    
    #assign each pixel the color of the centroid to which it belongs
    X_compressed = np.zeros(np.shape(X))
    for i in range(0,K):
        X_compressed[np.where(idx == i+1)] = centroids[i]
    
    #plot the original and compressed images
    A_compressed = X_compressed.reshape(np.shape(A))
    fig = plt.figure()
    ax = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    ax.imshow(A)
    ax.set_title('Original')
    ax2.imshow(A_compressed)
    ax2.set_title('Compressed with ' + str(K) + ' colors')
    plt.savefig('bird_OriginalVsCompressed' + str(K) + 'Colors.png')
    input("Press Enter to continue...")
    
    #load another image
    A = misc.imread("mulberry_tree.png")
    #divide by 255 so that all values are in the range [0,1]
    A = A/255
    #reshape into 128*128 (# of pixels) by 3 (RGB intensities) matrix
    X = A.reshape(np.shape(A)[0]*np.shape(A)[1], 3)
    
    K = 16 # 3
    max_iters = 10 # 20 # 50 #the quality of compressed image increases with max_iters
    
    #randomly initialize centroids
    initial_centroids = kMeansInitCentroids(X, K)
    #run K-means
    centroids, idx = runkMeans(X, initial_centroids, max_iters)
    
    #assign each pixel the color of the centroid to which it belongs
    X_compressed = np.zeros(np.shape(X))
    for i in range(0,K):
        X_compressed[np.where(idx == i+1)] = centroids[i]
    
    #plot the original and compressed images
    A_compressed = X_compressed.reshape(np.shape(A))
    fig = plt.figure()
    ax = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    ax.imshow(A)
    ax.set_title('Original')
    ax2.imshow(A_compressed)
    ax2.set_title('Compressed with ' + str(K) + ' colors')
    plt.savefig('mulberry_OriginalVsCompressed' + str(K) + 'Colors.png')
    input("Press Enter to close the program...")
        