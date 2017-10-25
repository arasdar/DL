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

# seed the random numbers to help debbugging
np.random.seed(1)

# define hyperparameters
LEARNING_RATE = 0.01
REGULARIZATION_WEIGTH = 1e-3
NB_FEATURES = 26
NB_TRAINING = 13000
NB_CLASSES = 13
NB_HIDDEN_NEURONS = 16
NB_TEST = 10400
epsilon = 1e-10

#for debbuging
np.set_printoptions(suppress=True)

# getting the test data from the csv file
testDatatemp = np.loadtxt(open("test.x.csv","rb"), dtype =np.float16,delimiter = ',',skiprows=1, usecols=range(1,27))

# invert testData to get a 26 * 13000 matrix
testDatatemp = testDatatemp.T

#getting the input data from the csv file
input = np.loadtxt(open("train.x.csv","rb"), dtype =np.float16,delimiter = ',',skiprows=1, usecols=range(1,27))

#normalize the inputs
#sum = np.sum(input, axis = 1)
#for i in range(NB_TRAINING):
#	for j in range(NB_FEATURES):
#		input[i,j] = input[i,j]/sum[i]

# getting the output data from the csv file
outputtemp = np.genfromtxt(open("train.y.csv","rb"), dtype = 'str', delimiter=',',skip_header=1, usecols=(1))
desiredOutput = np.zeros(NB_TRAINING, dtype= int)

# initializing all the weigths randomly
w = 0.01 * np.random.randn(NB_FEATURES,NB_CLASSES)
bias = np.zeros((1,NB_CLASSES))

# initializing the output matrix for the training data
j = 0
while j < NB_TRAINING:
	str = outputtemp[j]
	if str == 'International':
		desiredOutput[j] = 4
	if str == 'Vocal':
		desiredOutput[j] = 1
	if str == 'Latin':
		desiredOutput[j] = 6
	if str == 'Blues':
		desiredOutput[j] = 0
	if str == 'Country':
		desiredOutput[j] = 1
	if str == 'Electronic':
		desiredOutput[j] = 2
	if str == 'Folk':
		desiredOutput[j] = 3
	if str == 'Jazz':
		desiredOutput[j] = 5
	if str == 'New_Age':
		desiredOutput[j] = 7
	if str == 'Pop_Rock':
		desiredOutput[j] = 8
	if str == 'Rap':
		desiredOutput[j] = 9
	if str == 'Reggae':
		desiredOutput[j] = 11
	if str == 'RnB':
		desiredOutput[j] = 10
	j = j+1

desiredOutput.astype(int)

def forwardPass (inputLayer, weights,bias):
	nb_exemples = inputLayer.shape[0]

	output = np.dot(inputLayer,weights) + bias
	# compute probabilities from output

	# take the exponentiel of each output
	exp_output = np.exp(output)

	# normalize the data
	norm_output = exp_output / np.sum(exp_output,axis=1,keepdims=True)

	# compute the log prob
	#log_output = np.zeros(inputLayer.shape[0])
	#for i in range(inputLayer.shape[0]):
	#	log_output[i] = -np.log(norm_output[i,desiredOutput[i]])

	log_output = -np.log(norm_output[range(nb_exemples),desiredOutput[:nb_exemples]])

	data_loss = np.sum(log_output) / nb_exemples
	regulation_loss = 0.5*REGULARIZATION_WEIGTH*np.sum(weights*weights)
	loss = data_loss + regulation_loss
	print 'the loss is',loss

	# compute the gradient on the outputs
  	d_output = norm_output
  	d_output[range(nb_exemples),desiredOutput[:inputLayer.shape[0]]] -= 1
  	d_output /= nb_exemples
  	#print d_output.shape
  	# backpropagation of the gradients to get the gradient on the weights
  	dW = np.dot(inputLayer.T, d_output)
  	dW += REGULARIZATION_WEIGTH * weights # regularization 	
  	db = np.sum(d_output, axis=0, keepdims=True)

  	# update the weights
 	weights += -LEARNING_RATE * dW
 	bias += -LEARNING_RATE * db

for epoch in range(20):
  	index = np.random.randint(0,12999, size = 100)
	forwardPass(input[index,:],w,bias)