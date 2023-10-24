#!/usr/bin/env python
# coding: utf-8

# In[1]:


# %load MLP_Skeleton.py
"""
SANAD SAHA
933 620 612
"""
from __future__ import division
from __future__ import print_function

import sys
try:
   import _pickle as pickle
except:
   import pickle

import numpy as np


# In[2]:


# This is a class for a LinearTransform layer which takes an input 
# weight matrix W and computes W x as the forward step
class LinearTransform(object):

    def __init__(self, W, b):
    # DEFINE __init function
        self.W = W
        self.b = b
        
    def forward(self, x):
    # DEFINE forward function
        Z = np.dot(self.W , x) + self.b
        return Z
    
        
        
    def backward(
        self, 
        grad_output, 
        learning_rate=0.0, 
        momentum=0.0, 
        l2_penalty=0.0,
    ):
    # DEFINE backward function
        print("Backward not completed")
# ADD other operations in LinearTransform if needed


# In[3]:


# This is a class for a ReLU layer max(x,0)
class ReLU(object):

    def forward(self, x):
        #x[x < 0.0] = 0.0
        self.x = np.maximum(x, 0)
        return self.x

    def backward(
        self, 
        grad_output, 
        learning_rate=0.0, 
        momentum=0.0, 
        l2_penalty=0.0,
    ):
    # DEFINE backward function
        print("relu backprop not done")
# ADD other operations in ReLU if needed


# In[4]:


# This is a class for a sigmoid layer followed by a cross entropy layer, the reason 
# this is put into a single layer is because it has a simple gradient form
class SigmoidCrossEntropy(object):
    def forward(self, x):
        self.x = 1 / (1 + np.exp(-x))
        return self.x
    
    def backward(self, grad_output, learning_rate=0.0, momentum=0.0, l2_penalty=0.0):
        print("Nothing")
        # DEFINE backward function
    
# ADD other operations and data entries in SigmoidCrossEntropy if needed
    def compute_cost(self, y, target_y):
        m = target_y.shape[1]
        logprobs = np.multiply(np.log(y),target_y)/m +  np.multiply(np.log(1 - y), 1- target_y)/m
        cost = - np.sum(logprobs) 
        cost = np.squeeze(cost)
        return cost 


# In[7]:


# This is a class for the Multilayer perceptron
class MLP(object):

    def __init__(self, input_dims, hidden_units):
    # INSERT CODE for initializing the network
        self.hidden_units = hidden_units
        self.input_dims = input_dims
        #initialize W1, w2, b1, b2
        
        self.w1 = np.random.randn(self.hidden_units, self.input_dims) * 0.01
        self.b1 = np.zeros((self.hidden_units, 1))
        self.w2 = np.random.randn(1, self.hidden_units) * 0.01
        self.b2 = np.zeros((1, 1))

        assert (self.w1.shape == (self.hidden_units, self.input_dims))
        assert (self.b1.shape == (self.hidden_units, 1))
        assert (self.w2.shape == (1, self.hidden_units))
        assert (self.b2.shape == (1, 1))
        
    def train(self, x_batch, y_batch, learning_rate = 0.0, momentum = 0.0, l2_penalty = 0.0):
    # INSERT CODE for training the network
        
        m = x_batch.shape[1]
        lt1 = LinearTransform(self.w1, self.b1)
        z1 = lt1.forward(x_batch)
        r = ReLU()
        a1 = r.forward(z1)
        #print(x_batch[1:10, 1:10])
        print(self.w1[3, 1:3])
        
        
        lt2 = LinearTransform(self.w2, self.b2)
        z2 = lt2.forward(a1)
        print(z2[:, 1:3])
        sigmoid_cross_entropy = SigmoidCrossEntropy()
        a2 = sigmoid_cross_entropy.forward(z2)
        
        #loss = sigmoid_cross_entropy.compute_cost(a2, y_batch)
        #print("loss ")
        #print(loss)
        #BACKPROP
        assert(a2.shape == y_batch.shape)
        dz2 = a2 - y_batch
        dw2 = np.dot(dz2, a1.T)
        db2 = np.sum(dz2, axis = 1, keepdims = True)
        #print(db2)
        
        assert (self.w2.shape == dw2.shape) 
        assert (self.b2.shape == db2.shape)
        
        grad_relu = (z1 > 0.0)
        
        dw1 = np.dot(self.w2.T, dz2)
        dz1 = np.multiply(dw1, grad_relu)
        dw1 = np.dot(dz1, x_batch.T)
        db1 = np.sum(dz1, axis = 1, keepdims = True)
        
        assert(dw1.shape == self.w1.shape)
        assert (self.b1.shape == db1.shape)
       
        print(self.evaluate(a2, y_batch))
        
        #print("Testing")
        #print(self.w1[10, 1:5])
        self.w1 = self.w1 - learning_rate * dw1
        self.b1 = self.b1 - learning_rate * db1
        self.w2 = self.w2 - learning_rate * dw2
        self.b2 = self.b2 - learning_rate * db2
        
        #print(dw1[10, 1:5])
        #print(dw2)
        #print("Match testing ")
        #print(self.w1[10, 1:5])
            
    def evaluate(self, x, y):
        predictions = (x > 0.5)
        accuracy = float((np.dot(y,predictions.T) + np.dot(1-y,1-predictions.T))/float(y.size)*100)
        return accuracy
    
    # INSERT CODE for testing the network
# ADD other operations and data entries in MLP if needed
        


if __name__ == '__main__':
    if sys.version_info[0] < 3:
        data = pickle.load(open('cifar_2class_py2.p', 'rb'))
    else:
        data = pickle.load(open('cifar_2class_py2.p', 'rb'), encoding='bytes')
    
    train_x = data[b'train_data']
    train_y = data[b'train_labels']
    test_x = data[b'test_data']
    test_y = data[b'test_labels']

    num_examples, input_dims = train_x.shape

    #normalizing the input
    #x_norm = np.linalg.norm(train_x, ord = 2, axis = 1, keepdims = True)
    
    # Divide x by its norm.
    #train_x = train_x/x_norm
    train_x = train_x - train_x.mean()
    train_x = train_x / train_x.var()
    
    print(train_x.shape)
    print(train_y.shape)
#     print (train_x)
    
    # INSERT YOUR CODE HERE
    # YOU CAN CHANGE num_epochs AND num_batches TO YOUR DESIRED VALUES
    num_epochs = 10
    num_batches = 1000
    hidden_units = 50
    
    mlp = MLP(input_dims, hidden_units)
    
    
      
#     frm = epoch * num_batches
#     to = (epoch+1) * num_batches 
#     mlp.train(train_x[frm:to, :].T, train_y[frm:to, :].T, 1.5)
    
    for epoch in range(num_epochs):
        mlp.train(train_x, train_y, 0.01)
    
    # INSERT YOUR CODE FOR EACH EPOCH HERE
        
    
        #train_accuracy = evaluate()
#         for b in range(num_batches):
#             total_loss = 0.0
#             # INSERT YOUR CODE FOR EACH MINI_BATCH HERE
#             # MAKE SURE TO UPDATE total_loss
#             print(
#                 '\r[Epoch {}, mb {}]    Avg.Loss = {:.3f}'.format(
#                     epoch + 1,
#                     b + 1,
#                     total_loss,
#                 ),
#                 end='',
#             )
#             sys.stdout.flush()
        # INSERT YOUR CODE AFTER ALL MINI_BATCHES HERE
        # MAKE SURE TO COMPUTE train_loss, train_accuracy, test_loss, test_accuracy
#         print()
#         print('    Train Loss: {:.3f}    Train Acc.: {:.2f}%'.format(
#             train_loss,
#             100. * train_accuracy,
#         ))
#         print('    Test Loss:  {:.3f}    Test Acc.:  {:.2f}%'.format(
#             test_loss,
#             100. * test_accuracy,
#         ))

