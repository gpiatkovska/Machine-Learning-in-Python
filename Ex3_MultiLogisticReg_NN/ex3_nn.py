# -*- coding: utf-8 -*-
"""
Created on Thu Nov 19 14:25:42 2015

@author: Hanna
"""
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import scipy.io as io

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
    
  
def sigmoid(z):
    #works elementwise for any array z
    
    return sp.special.expit(z)  # expit(x) = 1/(1+exp(-x)) elementwise for array x
    
    
def forward(Theta1,Theta2,X):  # works for any # of examples
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
    
    return a3
    

def predictOneVsAllNN(h):
    all_probability = h.T
    prediction = np.argmax(all_probability,axis=1) + 1  # we do not have class 0 but have class 10    
    
    return prediction.reshape(np.shape(h)[1],1)
       
       
if __name__ == '__main__':
    
    #load data
    mat = io.loadmat("ex3data1.mat")
    X, y = mat['X'], mat['y']
    
    #display 100 random examples
    displayData(X)
    
    #load already trained weights
    mat = io.loadmat("ex3weights.mat")
    Theta1, Theta2 = mat['Theta1'], mat['Theta2']
    #Theta1 and Theta2 correspond to a network with:
    #400 (+1 bias) input units (= # of feature -- 20x20 image)
    #one hidden layer with 25 (+1 bias) units
    #10 output units corresponding to 10 classes
    print(np.shape(Theta1))  # Theta1 shape is (25,401)
    print(np.shape(Theta2))  # Theta2 shape is (10,26)
    
    #NN prediction
    h = forward(Theta1,Theta2,X)
    prediction = predictOneVsAllNN(h)
    training_accuracy = np.mean(prediction == y) * 100.0
    print("NN training set prediction accuracy = ", training_accuracy,"%")  # get 97.52 %
    print("supposed to be 97.5")
    
    #show images and print corresponding predictions one by one
    m = np.shape(X)[0]  # # of examples
    sequence = np.random.permutation(m)
    print("Note that 0 is labeled by 10")
    plt.ion()
    for i in sequence:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.imshow(X[i,:].reshape(20, 20).T, cmap="Greys_r")
        ax.set_axis_off()
        print(prediction[i,0])
        input("Press Enter to continue...")
        plt.close(fig)
    