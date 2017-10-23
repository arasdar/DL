#-------------------------------------------------------------------------------
# Name:        CSC492 - Coding Assignment #1
# Purpose:
#
# Author:      Marion
#
# Created:     13/10/2017
# Copyright:   (c) Marion 2017
# Licence:     <your licence>
#-------------------------------------------------------------------------------

import csv
import numpy as np
from numpy.random import randn

#seed the random numbers to help debbugging

np.random.seed(1)

#define hyperparameters

LEARNING_RATE = 0.01
NB_FEATURES = 26
NB_TRAININGEX = 13000
NB_CLASSES = 13
NB_HIDDEN_NEURONS = 16
NB_TEST = 10400

#getting the test data from the csv file
testDatatemp = np.loadtxt(open("test.x.csv","rb"), dtype =np.float16,delimiter = ',',skiprows=1, usecols=range(1,27))

#invert testData to get a 26 * 13000 matrix
testDatatemp = testDatatemp.T

#getting the input data from the csv file
input = np.loadtxt(open("train.x.csv","rb"), dtype =np.float16,delimiter = ',',skiprows=1, usecols=range(1,27))

#invert input to get a 26 * 10400 matrix
inputFinal = input.T

testData = testDatatemp

#getting the output data from the csv file
outputtemp = np.genfromtxt(open("train.y.csv","rb"), dtype = 'str', delimiter=',',skip_header=1, usecols=(1))
output = np.zeros((NB_CLASSES,NB_TRAININGEX))

#initializing all the weigths randomly
syn1 = np.random.random((NB_FEATURES,NB_HIDDEN_NEURONS)) 
syn2 = np.random.random((NB_HIDDEN_NEURONS, NB_CLASSES)) 

#initializing the output matrix for the training data, we map the classes, we get a 13 * 13000 matrix, 1 for the good class, 0 for the others
j = 0
while j < NB_TRAININGEX:
	str = outputtemp[j]
	if str == 'International':
		output[0,j] = 1
	if str == 'Vocal':
		output[1,j] = 1
	if str == 'Latin':
		output[2,j] = 1
	if str == 'Blues':
		output[3,j] = 1
	if str == 'Country':
		output[4,j] = 1
	if str == 'Electronic':
		output[5,j] = 1
	if str == 'Folk':
		output[6,j] = 1
	if str == 'Jazz':
		output[7,j] = 1
	if str == 'New_Age':
		output[8,j] = 1
	if str == 'Pop_Rock':
		output[9,j] = 1
	if str == 'Rap':
		output[10,j] = 1
	if str == 'Reggae':
		output[11,j] = 1
	if str == 'RnB':
		output[12,j] = 1
	j = j+1

#values for reference
#international = 0
#vocal = 1
#latin = 2
#blues = 3
#country = 4
#electronic = 5
#folk = 6
#jazz= 7
#new-age=8
#pop_rock = 9
#rap = 10
#reggae = 11
#rnb = 12

def sigmoid(x):
	return 1/(1 + np.exp(-x))

def sigmoidDeriv(x):
	return x *(1 - x)

#1 hidden layer = 2 synapses = 2 

def forwardPass (inputLayer, weights1, weigths2):

	hiddenLayer = weights1.T.dot(inputLayer)

	# apply sigmoid on all activations
	i = 0
	while i < NB_HIDDEN_NEURONS:
		hiddenLayer[i] = sigmoid(hiddenLayer[i])
		i = i + 1

	result = weigths2.T.dot(hiddenLayer)
	return result

#result2 = forwardPass(inputFinal,syn1,syn2)

#calculate the error

#testing

testing = forwardPass(testData,syn1,syn2)
testing = testing.T

#add index column
finalOutput = np.zeros((NB_TEST,NB_CLASSES+1))
i = 0
while i < NB_TEST:
	finalOutput[i] = np.hstack((i+1,testing[i]))
	i = i + 1

finalOutput.astype(np.int32)

#test data&
with open('submission.csv','a') as f_handle:
	np.savetxt(f_handle, finalOutput, fmt='%i,%1.4f,%1.4f,%1.4f,%1.4f,%1.4f,%1.4f,%1.4f,%1.4f,%1.4f,%1.4f,%1.4f,%1.4f,%1.4f',delimiter=",")