# -*- coding: utf-8 -*-
"""
Created on Mon Nov 23 09:32:46 2015

@author: Hanna
"""
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import scipy.io as io
import scipy.optimize as op

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
    
    
def displayHidden(X):
    width=20  # images are 20x20 pixels
    rows, cols = 5, 5  # 25 hidden units shown on a square 5 by 5
    out = np.zeros((width*rows,width*cols))
    indices = range(0, 25)
    for y in range(0, rows):
        for x in range(0, cols):
            start_x = x*width
            start_y = y*width
            out[start_x:start_x+width, start_y:start_y+width] = X[indices[rows*y+x]].reshape(width, width).T    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.imshow(out,cmap="Greys_r")
    ax.set_axis_off()
    plt.savefig("hiddenLayerViz.pdf")
    plt.show()
    
    
def sigmoid(z):
    #works elementwise for any array z
    
    return sp.special.expit(z)  # expit(x) = 1/(1+exp(-x)) elementwise for array x
    
    
def sigmoidGradient(z):
    #works elementwise for any array z
    
    return sigmoid(z)*(1.0 - sigmoid(z))
    
    
def randInitializeWeights(L_in,L_out):
    #eps = np.sqrt(6.0)/np.sqrt(L_in+L_out)
    eps = 0.12
    W = np.random.random((L_out, L_in + 1)) * 2 * eps - eps
 
    return W
    
    
def nnCostFunction1(nn_params,input_layer_size,hidden_layer_size,num_labels,X,y,lambbda):  # works for any two layer NN
    #put parameters back to matrix form
    Theta1 = nn_params[:(input_layer_size+1)*hidden_layer_size].reshape(hidden_layer_size,input_layer_size+1)
    Theta2 = nn_params[(input_layer_size+1)*hidden_layer_size:].reshape(num_labels,hidden_layer_size+1)
    
    #forward propagation:
    
    #input activation
    m = np.shape(X)[0]  # # of examples
    a1 = X.T
    a1 = np.vstack((np.ones((1,m)),a1))
    
    #hidden layer activation
    z2 = np.dot(Theta1,a1)
    a2 = sigmoid(z2)
    a2 = np.vstack((np.ones((1,m)),a2))
    
    #output layer activation
    z3 = np.dot(Theta2,a2)
    a3 = sigmoid(z3)
    
    #compute cost using direct summation over examples
    cost = 0.0
    h = a3
    for i in range(0,m):
        y_i = np.zeros((num_labels,1))
        y_i[y[i,0]-1,0] = 1
        h_i = h[:,i].reshape(num_labels,1)
        cost = cost - np.dot(y_i.T,sp.log(h_i)) - np.dot((1.0-y_i).T,sp.log(1.0-h_i))
    cost = cost/m
    
    return cost
    
    
def nnCostFunction(nn_params,input_layer_size,hidden_layer_size,num_labels,X,y,lambbda):  # works for any two layer NN
    #put parameters back to matrix form
    Theta1 = nn_params[:(input_layer_size+1)*hidden_layer_size].reshape(hidden_layer_size,input_layer_size+1)
    Theta2 = nn_params[(input_layer_size+1)*hidden_layer_size:].reshape(num_labels,hidden_layer_size+1)
    
    #forward propagation:
    
    #input activation
    m = np.shape(X)[0]  # # of examples
    a1 = X.T
    a1 = np.vstack((np.ones((1,m)),a1))
    
    #hidden layer activation
    z2 = np.dot(Theta1,a1)
    a2 = sigmoid(z2)
    a2 = np.vstack((np.ones((1,m)),a2))
    
    #output layer activation
    z3 = np.dot(Theta2,a2)
    a3 = sigmoid(z3)
    
    #compute cost
    
    h = a3
    #get labels in the same form as hypothesis (matrix)
    y_mat = np.zeros((num_labels,m))
    for i in range(0,m):
        y_mat[y[i,0]-1,i] = 1
    cost = -y_mat*sp.log(h) - (1.0-y_mat)*sp.log(1.0-h)
    cost = np.sum(cost)/m
    #regularization
    cost_reg = (np.sum(Theta1[:,1:]*Theta1[:,1:]) + np.sum(Theta2[:,1:]*Theta2[:,1:]))*lambbda/2.0/m
    cost = cost + cost_reg
    
    #backpropagation: compute gradients
    
    delta3 = a3 - y_mat
    delta2 = np.dot(Theta2.T,delta3)*sigmoidGradient(np.vstack((np.ones((1,m)),z2)))
    #actually better to use
    #delta2 = np.dot(Theta2.T,delta3)*a2*(1.0-a2)
    delta2 = delta2[1:,:]

    Delta1 = np.dot(delta2,a1.T)/m
    Delta2 = np.dot(delta3,a2.T)/m
    #regularization
    Delta1[:,1:] = Delta1[:,1:] + lambbda*Theta1[:,1:]/m
    Delta2[:,1:] = Delta2[:,1:] + lambbda*Theta2[:,1:]/m

    gradient = np.vstack((Delta1.reshape((input_layer_size+1)*hidden_layer_size,1),Delta2.reshape((hidden_layer_size+1)*num_labels,1))).flatten()

    return (cost, gradient)
    
    
def debugInitializeWeights(fan_out,fan_in):
    W = np.zeros(fan_out*(fan_in+1)) 
    for i in range(0,fan_out*(fan_in+1)):
        W[i] = np.sin(i+1)/10
    W = W.reshape(fan_out,1+fan_in)
    
    return W


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

def checkNNGradients(lambbda=0.0):
    input_layer_size = 3
    hidden_layer_size = 5
    num_labels = 3
    m = 5

    Theta1 = debugInitializeWeights(hidden_layer_size,input_layer_size)
    Theta2 = debugInitializeWeights(num_labels,hidden_layer_size)
    X = debugInitializeWeights(m,input_layer_size-1)
    y = 1 + np.mod(range(1,m+1),num_labels)
    y = y.reshape(len(y),1)

    nn_params = np.vstack((Theta1.reshape((input_layer_size+1)*hidden_layer_size,1),Theta2.reshape((hidden_layer_size+1)*num_labels,1))).flatten()
    gradient = nnCostFunction(nn_params,input_layer_size,hidden_layer_size,num_labels,X,y,lambbda)[1]
    J = lambda theta: nnCostFunction(theta,input_layer_size,hidden_layer_size,num_labels,X,y,lambbda)[0]
    numgrad = computeNumericalGradient(J,nn_params)
    diff = np.linalg.norm(numgrad-gradient)/np.linalg.norm(numgrad+gradient)
    
    return diff
    
    
def predict(nn_params,input_layer_size,hidden_layer_size,num_labels,X,lambbda):  # works for any two layer NN
    #put parameters back to matrix form
    Theta1 = nn_params[:(input_layer_size+1)*hidden_layer_size].reshape(hidden_layer_size,input_layer_size+1)
    Theta2 = nn_params[(input_layer_size+1)*hidden_layer_size:].reshape(num_labels,hidden_layer_size+1)
    
    #forward propagation:
    
    #input activation
    m = np.shape(X)[0]  # # of examples
    a1 = X.T
    a1 = np.vstack((np.ones((1,m)),a1))
    
    #hidden layer activation
    z2 = np.dot(Theta1,a1)
    a2 = sigmoid(z2)
    a2 = np.vstack((np.ones((1,m)),a2))
    
    #output layer activation
    z3 = np.dot(Theta2,a2)
    a3 = sigmoid(z3)
    
    pred = np.argmax(a3,axis=0) + 1  # we do not have class 0 but have class 10 
    
    return pred.reshape(m,1)  # same form as labels

        
    
if __name__ == '__main__':
    
    #load data
    mat = io.loadmat("ex4data1.mat")
    X, y = mat['X'], mat['y']
    #print(np.shape(y))
    
    #display 100 random examples
    displayData(X)
    
    #load already trained weights
    mat = io.loadmat("ex4weights.mat")
    Theta1, Theta2 = mat['Theta1'], mat['Theta2']
    #Theta1 and Theta2 correspond to a network with:
    #400 (+1 bias) input units (= # of feature -- 20x20 image)
    #one hidden layer with 25 (+1 bias) units
    #10 output units corresponding to 10 classes
    print(np.shape(Theta1))  # Theta1 shape is (25,401)
    print(np.shape(Theta2))  # Theta2 shape is (10,26)
    
    #the code should work for any two layer neural network
    #i.e. with any input layer size, hidden layer size, output layer size (# of labels)
    #in this case (exluding bias units):
    input_layer_size = 400
    hidden_layer_size 	= 25
    num_labels = 10
    
    nn_params = np.vstack((Theta1.reshape((input_layer_size+1)*hidden_layer_size,1),Theta2.reshape((hidden_layer_size+1)*num_labels,1))).flatten()
    
    lambbda = 0
    #check the cost for debugging parameters. should be 0.287629
    print(nnCostFunction(nn_params,input_layer_size,hidden_layer_size,num_labels,X,y,lambbda)[0])  # gives 0.287629165161
    
    lambbda = 1.0
    #check the regularized cost for debugging parameters. should be 0.383770 for lambbda = 1.0
    print(nnCostFunction(nn_params,input_layer_size,hidden_layer_size,num_labels,X,y,lambbda)[0])  # gives 0.383769859091
    
    #test sigmoidGradient. should be 0, 0.25, 0
    print(sigmoidGradient((-100,0,100)))  # gives [  3.72007598e-44   2.50000000e-01   0.00000000e+00]
    
    #random initialize weights
    Theta1 = randInitializeWeights(input_layer_size,hidden_layer_size)
    Theta2 = randInitializeWeights(hidden_layer_size,num_labels)
    initial_nn_params = np.vstack((Theta1.reshape((input_layer_size+1)*hidden_layer_size,1),Theta2.reshape((hidden_layer_size+1)*num_labels,1))).flatten()
    
    #check gradients
    print(checkNNGradients()) # should be less than 1e-9, gives 2.01227058242e-11
    
    #check gradients with regularization
    lambbda = 3.0
    print(checkNNGradients(lambbda))  # should be less than 1e-9, gives 1.92733169803e-11
    #check the regularized cost for debugging parameters. should be about 0.576051 for lambbda = 3.0
    print(nnCostFunction(nn_params,input_layer_size,hidden_layer_size,num_labels,X,y,lambbda)[0])  # gives 0.57605124695
    
    #train NN
    lambbda = 1.0
    cost = lambda theta: nnCostFunction(theta,input_layer_size,hidden_layer_size,num_labels,X,y,lambbda)[0]
    grad = lambda theta: nnCostFunction(theta,input_layer_size,hidden_layer_size,num_labels,X,y,lambbda)[1]
    result = op.minimize(cost, initial_nn_params, method='CG', jac=grad, options={'maxiter': 50,'disp': True})
    trained_nn_params = result.x
    
    #make a prediction and determine training set prediction accuracy
    pred = predict(trained_nn_params,input_layer_size,hidden_layer_size,num_labels,X,lambbda)
    training_accuracy = np.mean(pred == y) * 100.0
    print("training set prediction accuracy = ", training_accuracy,"%")
    print("supposed to be 95.3 +- 1% for 50 iterations")  # get 96.18 %
    #visually compare
    #print(pred[:15])
    #print(y[:15])
    #print(pred[3456:3468])
    #print(y[3456:3468])
    
    #train NN with more iterations
    result = op.minimize(cost, initial_nn_params, method='CG', jac=grad, options={'maxiter': 400,'disp': True})
    trained_nn_params = result.x
    pred = predict(trained_nn_params,input_layer_size,hidden_layer_size,num_labels,X,lambbda)
    training_accuracy = np.mean(pred == y) * 100.0
    print("training set prediction accuracy for 400 iterations is ", training_accuracy,"%")  # get 99.5 %
    
    #train NN with more iterations and smaller regularization parameter
    lambbda = 0.1
    cost = lambda theta: nnCostFunction(theta,input_layer_size,hidden_layer_size,num_labels,X,y,lambbda)[0]
    grad = lambda theta: nnCostFunction(theta,input_layer_size,hidden_layer_size,num_labels,X,y,lambbda)[1]
    result = op.minimize(cost, initial_nn_params, method='CG', jac=grad, options={'maxiter': 400,'disp': True})
    trained_nn_params = result.x
    pred = predict(trained_nn_params,input_layer_size,hidden_layer_size,num_labels,X,lambbda)
    training_accuracy = np.mean(pred == y) * 100.0
    print("training set prediction accuracy for 400 iterations and lambbda = 0.1 is ", training_accuracy,"%")  # get 100.0 %
    
    #visualize the hidden layer
    Theta1 = trained_nn_params[:(input_layer_size+1)*hidden_layer_size].reshape(hidden_layer_size,input_layer_size+1)
    displayHidden(Theta1[:, 1:])
  