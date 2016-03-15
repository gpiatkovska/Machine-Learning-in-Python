# -*- coding: utf-8 -*-
"""
Created on Sun Nov 29 09:28:21 2015

@author: Hanna
"""
import re
import nltk
import numpy as np
import scipy.io as io
from sklearn import svm
#from sklearn.metrics import accuracy_score

def getVocabList():
    vocab_list = {}
    f = open("vocab.txt", 'r')
    lines = f.readlines()
    for line in lines:
        x = re.split('\W+',line)
        vocab_list[x[1]] = int(x[0])
    f.close()
        
    return  vocab_list
    
    
def emailFeatures(word_indices):
    n = len(getVocabList())
    x = np.zeros((n, 1))
    for index in word_indices:
        x[index] = 1
        
    return x


def processEmail(email_contents):
    #prepocess email vcontents:    
    
    #lower case
    email_contents = email_contents.lower()
    
    #strip HTML
    email_contents = re.sub('<[^<>]+>', ' ', email_contents)
    
    #normalize numbers
    email_contents = re.sub('[0-9]+', 'number', email_contents)
    
    #normalize URLs
    email_contents = re.sub('(http|https)://[^\s]*', 'httpaddr', email_contents)
    
    #normalize email addresses
    email_contents = re.sub('[^\s]+@[^\s]+', 'emailaddr', email_contents)
    
    #normalize dollar signs
    email_contents = re.sub('[$]+', 'dollar', email_contents)
    
    #remove single apostrophes
    #with this uncommented get e.g. your from you're, othervise get you re
    #email_contents = email_contents.replace("'", "")
    
    #tokenize into words removing any non alphanumeric characters
    tokens = re.split('\W+',email_contents)
    
    #word stemming
    stemmer = nltk.stem.porter.PorterStemmer()
    tokens=[stemmer.stem(token) for token in tokens if len(token) > 0]
    #print("Preprocessed email:")
    #print(tokens)
    
    #look up the word in the dictionary and add to word_indices if found:
    
    vocab_list = getVocabList()
    #print("Vocabulary list:")
    #print(vocab_list)
    word_indices = []
    for token in tokens:
        if token in vocab_list:
            word_indices.append(vocab_list[token])

    return word_indices


if __name__ == '__main__':
    #read the sample email
    f = open("emailSample1.txt", 'r')
    file_contents = f.read()
    #print the email
    print("Original email:")
    print(file_contents)
    
    #get word indices for the email
    word_indices = processEmail(file_contents)
    print("Word indices for the email:")
    print(word_indices)  # same as on fig 11 in assignment
    
    #extract the feature vector
    x = emailFeatures(word_indices)
    print("The feature vector for the sample email has length", len(x))
    print("and", np.sum(x>0),"non-zero entries")
    
    #close the sample email
    f.close()
    
    #train linear svm for spam classification
    
    #load the training data
    mat = io.loadmat("spamTrain.mat")
    X, y = mat['X'], mat['y']
    
    #train SVM with linear kernel and C = 0.1
    svc = svm.SVC(kernel='linear',C=0.1)
    svc.fit(X,y.ravel())
    #alternatively, use LinearSVC
    svl = svm.LinearSVC(C=0.1)
    svl.fit(X,y.ravel())
    
    #trainig set accuracy
    accuracy_train = svc.score(X,y.ravel())  # or accuracy_score(svc.predict(X), y.ravel())
    accuracy_train_l = svl.score(X,y.ravel())
    
    #load the test data
    mat = io.loadmat("spamTest.mat")
    Xtest, ytest = mat['Xtest'], mat['ytest']
    
    #test set accuracy 
    accuracy_test = svc.score(Xtest,ytest.ravel())
    accuracy_test_l = svl.score(Xtest,ytest.ravel())
    
    #report accuracy
    print("Training accuracy is ", accuracy_train*100, "%, should be 99.8%")  #get 99.825%
    print("Test accuracy is ", accuracy_test*100, "%, should be 98.5%")  #get 98.9 %, a bit higher maybe because sklearn uses better svm implementation than provided for octave
    print("Training accuracy using LinearSVC is ", accuracy_train_l*100, "%, should be 99.8%")#get 99.975 %
    print("Test accuracy using LinearSVC is ", accuracy_test_l*100, "%, should be 98.5%")#get 99.2 %    
    
    #top predictors of spam
    w = svc.coef_[0]
    #sort weights in descending order: we are interested in 15 largest positive weights to get top spam predictors
    sorted_ind = w.argsort()[::-1]
    #sorted_ind = w.argsort()  #to get top non-spam predictors
    #sorted_ind = (w**2).argsort()[::-1]  #to get top overall predictors
    vocabList = getVocabList()
    print(vocabList.items())
    print('Top 15 predictors of spam:')
    for i in range(0,15):
        for word, number in vocabList.items():
            if number == sorted_ind[i]:
                print(word)
    #gives different words than provided on Fig 12 in assignment text
    #top predictors of spam using LinearSVC
    w = svl.coef_[0]
    #sort weights in descending order: we are interested in 15 largest positive weights to get top spam predictors
    sorted_ind = w.argsort()[::-1]
    #sorted_ind = w.argsort()#to get top non-spam predictors
    #sorted_ind = (w**2).argsort()[::-1]#to get top overall predictors
    vocabList = getVocabList()
    print('Top 15 predictors of spam from LinearSVC:')
    for i in range(0,15):
        for word, number in vocabList.items():
            if number == sorted_ind[i]:
                print(word)
    #gives different words than provided on Fig 12 in assignment text
                
    #classify example spam and non-spam emails:
                
    #classify spam sample 1 email
    f = open("spamSample1.txt", 'r')
    file_contents = f.read()
    #print the email
    print("Spam email 1:")
    print(file_contents)
    #get word indices for the email
    word_indices = processEmail(file_contents)
    #extract the feature vector
    x = emailFeatures(word_indices)
    #make a prediction
    pred = svc.predict(x.ravel())
    if pred == 0:
        print("Not Spam")
    else: 
        print("Spam!")
    #gives not spam, misclassifies
    #use LinearSVC
    pred = svl.predict(x.ravel())
    if pred == 0:
        print("LinearSVC: Not Spam")
    else: 
        print("LinearSVC: Spam!")
    #gives not spam, misclassifies
        
    #close the sample email
    f.close()
    
    #classify spam sample 2 email
    f = open("spamSample2.txt", 'r')
    file_contents = f.read()
    #print the email
    print("Spam email 2:")
    print(file_contents)
    #get word indices for the email
    word_indices = processEmail(file_contents)
    #extract the feature vector
    x = emailFeatures(word_indices)
    #make a prediction
    pred = svc.predict(x.ravel())
    if pred == 0:
        print("Not Spam")
    else:
        print("Spam!")
    #gives spam, correctly
    #use LinearSVC
    pred = svl.predict(x.ravel())
    if pred == 0:
        print("LinearSVC: Not Spam")
    else: 
        print("LinearSVC: Spam!")
    #gives spam, correctly
    
    #close the sample email
    f.close()
    
    #classify non-spam email sample 1 email
    f = open("emailSample1.txt", 'r')
    file_contents = f.read()
    #print the email
    print("Non-spam email 1:")
    print(file_contents)
    #get word indices for the email
    word_indices = processEmail(file_contents)
    #extract the feature vector
    x = emailFeatures(word_indices)
    #make a prediction
    pred = svc.predict(x.ravel())
    if pred == 0:
        print("Not Spam")
    else:
        print("Spam!")
    #gives not spam, correctly
    #use LinearSVC
    pred = svl.predict(x.ravel())
    if pred == 0:
        print("LinearSVC: Not Spam")
    else: 
        print("LinearSVC: Spam!")
    #gives not spam, correctly
    
    #close the sample email
    f.close()
    
    #classify non-spam email sample 2 email
    f = open("emailSample2.txt", 'r')
    file_contents = f.read()
    #print the email
    print("Non-spam email 2:")
    print(file_contents)
    #get word indices for the email
    word_indices = processEmail(file_contents)
    #extract the feature vector
    x = emailFeatures(word_indices)
    #make a prediction
    pred = svc.predict(x.ravel())
    if pred == 0:
        print("Not Spam")
    else:
        print("Spam!")
    #gives not spam, correctly
    #use LinearSVC
    pred = svl.predict(x.ravel())
    if pred == 0:
        print("LinearSVC: Not Spam")
    else: 
        print("LinearSVC: Spam!")
    #gives not spam, correctly
    
    #close the sample email
    f.close()
    
    #classify my spam email
    f = open("spamSampleMy3.txt", 'r')
    file_contents = f.read()
    #print the email
    print("Spam email 3:")
    print(file_contents)
    #get word indices for the email
    word_indices = processEmail(file_contents)
    #extract the feature vector
    x = emailFeatures(word_indices)
    #make a prediction
    pred = svc.predict(x.ravel())
    if pred == 0:
        print("Not Spam")
    else:
        print("Spam!")
    #gives spam, correctly
    #use LinearSVC
    pred = svl.predict(x.ravel())
    if pred == 0:
        print("LinearSVC: Not Spam")
    else: 
        print("LinearSVC: Spam!")
    #gives spam, correctly
    
    #close the sample email
    f.close()
    
    #better make a function for classification taking email file name as an argument -- not clutter the code 
