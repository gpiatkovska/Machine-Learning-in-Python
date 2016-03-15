# -*- coding: utf-8 -*-
"""
Created on Wed Dec  2 10:33:39 2015

@author: Hanna
"""

import numpy as np
import scipy.io as io
import matplotlib.pyplot as plt
import re
import scipy.optimize as op

def cofiCostFunc(params, Y, R, num_users, num_movies, num_features, lambbda):
    #unroll parameters:
    X = params[:num_movies*num_features].reshape(num_movies,num_features)
    Theta = params[num_movies*num_features:].reshape(num_users,num_features)
    #cost:
    cost = np.sum((R*(np.dot(X,Theta.T)-Y))**2)/2 + lambbda*(np.sum(Theta**2)+np.sum(X**2))/2
    #gradients:
    X_grad = np.dot(R*(np.dot(X,Theta.T)-Y),Theta) + lambbda*X
    Theta_grad = np.dot((R*(np.dot(X,Theta.T)-Y)).T,X) + lambbda*Theta
    
    return (cost, np.vstack((X_grad,Theta_grad)).flatten())
    
    
def computeNumericalGradient(J,theta):
    numgrad = np.zeros(np.shape(theta))
    perturb = np.zeros(np.shape(theta))
    eps = 0.0001

    for p in range(0, len(theta)):
        perturb[p] = eps
        loss1 = J(theta-perturb)
        loss2 = J(theta+perturb)
        numgrad[p] = (loss2-loss1)/(2.0*eps)
        perturb[p] = 0

    return numgrad
    
    
def loadMovieList():
    movieList = {}
    f = open("movie_ids.txt", 'r', encoding = "ISO-8859-1")
    lines = f.readlines()
    for line in lines:
        x = re.split('\W+',line,1)
        movieList[int(x[0])] = x[1]
    f.close()
        
    return  movieList
    
    
def normalizeRatings(Y, R):
    m = len(Y)
    Ymean = np.zeros((m, 1))
    Ynorm = np.zeros(np.shape(Y))
    for i in range(0,m):
        idx = np.nonzero(R[i])
        Ymean[i] = np.mean(Y[i,idx])
        Ynorm[i,idx] = Y[i,idx] - Ymean[i]

    return Ynorm, Ymean
    
	
if __name__ == '__main__':
    
    #load and visualize the data
    mat = io.loadmat("ex8_movies.mat")
    Y, R = mat['Y'], mat['R']
    #Y is a 1682x943 matrix, containing ratings (1-5) of 1682 movies on 943 users, Y(i,j) = 0 if user j didn't rate movie i
    #R is a 1682x943 matrix, where R(i,j) = 1 if and only if user j gave a rating to movie i, and 0 otherwise
    print("Average rating for movie 1 (Toy Story): ", np.mean(Y[0,np.nonzero(R[0])]))
    fig = plt.figure()
    ax = fig.gca()
    cax = ax.imshow(Y)
    fig.colorbar(cax,ticks=[0,1,2,3,4,5])
    plt.xlabel("Users")
    plt.ylabel("Movies")
    plt.savefig('MoviesDataSet.pdf')
    plt.show()
    
    #load pre-trained weights (X, Theta, num_users, num_movies, num_features)
    mat = io.loadmat("ex8_movieParams.mat")
    X, Theta = mat['X'], mat['Theta']
    num_users, num_movies, num_features = mat['num_users'], mat['num_movies'], mat['num_features']
    #print(num_users, num_movies, num_features)
    #print(np.shape(X))  # num_movies x num_features
    #print(np.shape(Theta)) # num_users x num_features
    
    #reduce the dataset size for testing
    num_users = 4
    num_movies = 5
    num_features = 3
    X = X[:num_movies, :num_features]
    Theta = Theta[:num_users, :num_features]
    Y = Y[:num_movies, :num_users]
    R = R[:num_movies, :num_users]
    
    #test cost function
    cost = cofiCostFunc(np.vstack((X,Theta)).flatten(), Y, R, num_users, num_movies, num_features, 0)[0]
    print("Cost at loaded parameters: ", cost, " , should be about 22.22")  # get 22.2246037257
    
    #test gradients
    gradient = cofiCostFunc(np.vstack((X,Theta)).flatten(), Y, R, num_users, num_movies, num_features, 0)[1]
    J = lambda params: cofiCostFunc(params, Y, R, num_users, num_movies, num_features, 0)[0]
    numgrad = computeNumericalGradient(J,np.vstack((X,Theta)).flatten())
    diff = np.linalg.norm(numgrad-gradient)/np.linalg.norm(numgrad+gradient)
    print("This should be very small if gradients are correct: ", diff)  # get 8.94703685936e-13
    
    #test regularized cost function
    cost = cofiCostFunc(np.vstack((X,Theta)).flatten(), Y, R, num_users, num_movies, num_features, 1.5)[0]
    print("Cost at loaded parameters with lambda=1.5 : ", cost, " , should be about 31.34")  # get 31.3440562443
    
    #test regularized gradients
    gradient = cofiCostFunc(np.vstack((X,Theta)).flatten(), Y, R, num_users, num_movies, num_features, 1.5)[1]
    J = lambda params: cofiCostFunc(params, Y, R, num_users, num_movies, num_features, 1.5)[0]
    numgrad = computeNumericalGradient(J,np.vstack((X,Theta)).flatten())
    diff = np.linalg.norm(numgrad-gradient)/np.linalg.norm(numgrad+gradient)
    print("This should be very small if regularized gradients are correct: ", diff)  # get 1.37368375273e-12
    
    #provide ratings
    movies = loadMovieList()
    
    my_ratings = np.zeros((len(movies), 1))
    
    my_ratings[0] = 4
    my_ratings[97] = 2
    my_ratings[6] = 3
    my_ratings[11] = 5
    my_ratings[53] = 4
    my_ratings[63] = 5
    my_ratings[65] = 3
    my_ratings[68] = 5
    my_ratings[182] = 4
    my_ratings[225] = 5
    my_ratings[354] = 5
    '''
    my_ratings[21] = 5
    my_ratings[50] = 4
    my_ratings[55] = 5
    my_ratings[59] = 5
    my_ratings[63] = 5
    my_ratings[68] = 5
    my_ratings[88] = 4
    my_ratings[194] = 2
    my_ratings[201] = 5
    my_ratings[356] = 5
    my_ratings[377] = 3
    my_ratings[780] = 5
    '''
    print("Original ratings provided:")
    for i in range(0, len(my_ratings)):
        if my_ratings[i] > 0:
            print("Rated", int(my_ratings[i]), "for", movies[i+1])
            
    #add provided ratings to dataset
    mat = io.loadmat("ex8_movies.mat")
    Y, R = mat['Y'], mat['R']
    Y = np.hstack((my_ratings,Y))
    R = np.hstack(((my_ratings>0),R))
    
    #normalize ratings
    Ynorm, Ymean = normalizeRatings(Y, R)
    #print(Ymean[0])
    
    num_users = np.shape(Y)[1]
    num_movies = np.shape(Y)[0]
    num_features = 10
    
    #randomly initialize parameters
    X = np.random.randn(num_movies, num_features)
    Theta 	= np.random.randn(num_users, num_features)
    
    #train the recommender model
    lambbda = 10
    
    #function and gradient to pass to the optimization routine
    cost = lambda params: cofiCostFunc(params, Y, R, num_users, num_movies, num_features, lambbda)[0]
    grad = lambda params: cofiCostFunc(params, Y, R, num_users, num_movies, num_features, lambbda)[1]
    
    #minimize using nonlinear conjugate gradient analogous to matlab fmincg
    result = op.minimize(cost, np.vstack((X,Theta)).flatten(), method='CG', jac=grad, options={'disp': True})
    params = result.x
    
    #unroll learned parameters:
    X = params[:num_movies*num_features].reshape(num_movies,num_features)
    Theta = params[num_movies*num_features:].reshape(num_users,num_features)
    
    #make predictions
    p = np.dot(X,Theta.T)
    my_predictions = p[:,0] + Ymean.flatten()
    
    #sort in descending order
    ix = my_predictions.argsort()[::-1]
    print("Top recommendations:")
    for i in range(0, 10):
        j = ix[i]
        print("Predicting rating", my_predictions[j], "for movie", movies[j+1])
        