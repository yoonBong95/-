# -*- coding: utf-8 -*-
"""
Created on Fri Aug 21 22:24:55 2020

@author: capta
"""

import numpy as np
import math

# Sigmoid Function
def act(x):
    return 1/(1+np.exp(-x))


# datasets
x = np.array([[0, 0, 1], [0, 1, 1], [1, 0, 1], [1, 1, 1]])     # x1 and x2 in the same aisle is a set input node
y = np.array([[0, 1, 1, 0]]).T     # xor


# parameters
input_size = x.shape[1] # original input + bias
hidden_size = 4
output_size = 1
alpha = 0.1    # learning rate
falpha = 5


# weights
w1 = np.random.randn(input_size, hidden_size)
w2 = np.random.randn(hidden_size, output_size)

i=0


while True:
	# forward
	z1 = np.dot(x, w1)
	a1 = act(z1)
	z2 = np.dot(a1, w2)
	Y = act(z2)
	
	# back propagation
	delta2 = (Y-y) * (Y * (1-Y))
	delta1 = np.dot(delta2, w2.T) * (a1 * (1-a1))
	w2 -= alpha * np.dot(a1.T, delta2)
	w1 -= alpha * np.dot(x.T, delta1)

	i+=1
	if math.sqrt(sum((Y-y)**2)/4) < 0.05:
		break


# test
z1 = np.dot(x, w1)
a1 = act(z1)
z2 = np.dot(a1, w2)
Y = act(z2)
print 
print ("Output after training...")
print (Y)
print 
print ("Total Iteration is "), i