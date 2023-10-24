
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
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


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
        Z = np.dot(x, self.W) + self.b
        return Z
    
        
        
    def backward(self, grad_output, learning_rate=0.0, momentum=0.0, l2_penalty=0.0,):
    # DEFINE backward function
       print("nothing")
# ADD other operations in LinearTransform if needed


# In[3]:

# This is a class for a ReLU layer max(x,0)
class ReLU(object):

    def forward(self, x):
        #x[x < 0.0] = 0.0
        self.z1 = x
        self.a1 = np.maximum(self.z1, 0.0)
        return self.a1

    def backward(self, grad_output, learning_rate=0.0, momentum=0.0, l2_penalty=0.0,):
    	dz1_relu = (grad_output > 0.0)
    	return dz1_relu

    # DEFINE backward function
        
# ADD other operations in ReLU if needed


# In[4]:


# This is a class for a sigmoid layer followed by a cross entropy layer, the reason 
# this is put into a single layer is because it has a simple gradient form
class SigmoidCrossEntropy(object):
    def forward(self, x):
        self.x = 1 / (1 + np.exp(-x))
        return self.x
    
    def backward(self, grad_output, learning_rate=0.0, momentum=0.0, l2_penalty=0.0):
        
        print("not used")
        # DEFINE backward function
    
# ADD other operations and data entries in SigmoidCrossEntropy if needed
    def compute_cost(self, yhat, y):
    	
        m = y.shape[0]
        epsilon = 1e-10
        loss = y * np.log(yhat.T + epsilon) + (1-y) * np.log(1 - yhat.T + epsilon)
        total_loss = - np.sum(loss) / m
        return total_loss 


# In[5]:


# This is a class for the Multilayer perceptron
class MLP(object):

    def __init__(self, input_dims, hidden_units):

    # INSERT CODE for initializing the network

        self.hidden_units = hidden_units
        self.input_dims = input_dims

        #initialize W1, w2, b1, b2
        
        self.w1 = np.random.randn(self.input_dims, self.hidden_units) * np.sqrt(2/(self.input_dims - 1))
        self.b1 = np.zeros((1, self.hidden_units))
        self.w2 = np.random.randn(self.hidden_units, 1) * np.sqrt(2/(self.hidden_units - 1))
        self.b2 = np.zeros((1, 1))

        assert (self.w1.shape == (self.input_dims, self.hidden_units))
        assert (self.b1.shape == (1, self.hidden_units))
        assert (self.w2.shape == (self.hidden_units, 1))
        assert (self.b2.shape == (1, 1))
        
    def train(self, x_batch, y_batch, learning_rate = 0.0, momentum = 0.0, l2_penalty = 0.0):
        # INSERT CODE for training the network

        # Forward Pass
        parameters = self.forward_pass(x_batch, y_batch)
        z1 = parameters['z1']
        a1 = parameters['a1']
        z2 = parameters['z2']
        a2 = parameters['a2']
        cost = parameters['cost'] 

        #BACKPROP
        dw1, db1, dw2, db2 = self.get_gradients(z1, a1, z2, a2, x_batch, y_batch)

        vdw1 = vdw2 = vdb1 = vdb2 = 0
        vdw1 = momentum * vdw1 + (1 - momentum) * dw1
        vdb1 = momentum * vdb1 + (1 - momentum) * db1
        vdw2 = momentum * vdw2 + (1 - momentum) * dw2
        vdb2 = momentum * vdb2 + (1 - momentum) * db2

        jedi = (1 - learning_rate * l2_penalty)

        self.w1 = jedi * self.w1 - learning_rate * vdw1
        self.b1 = jedi * self.b1 - learning_rate * vdb1
        self.w2 = jedi * self.w2 - learning_rate * vdw2
        self.b2 = jedi * self.b2 - learning_rate * vdb2

        return cost
                       
    def evaluate(self, x, y):
        
        parameters = self.forward_pass(x, y)
        a2 = parameters['a2']
        predictions = (a2 > (0.5-(1e-10)))
        pp = np.count_nonzero(predictions==y)
        accuracy = (float(pp)/float(y.shape[0]))
		# return accuracy			
        cost = parameters['cost']
        return cost, accuracy 

    # INSERT CODE for testing the network
	# ADD other operations and data entries in MLP if needed
    def forward_pass(self, x_batch, y_batch):

        lt1 = LinearTransform(self.w1, self.b1)
        z1 = lt1.forward(x_batch)
        relu = ReLU()
        a1 = relu.forward(z1)

        lt2 = LinearTransform(self.w2, self.b2)
        z2 = lt2.forward(a1)
        sigmoid_cross_entropy = SigmoidCrossEntropy()
        a2 = sigmoid_cross_entropy.forward(z2)
        cost = sigmoid_cross_entropy.compute_cost(a2, y_batch)

        parameters = {"z1": z1, "a1": a1, "z2": z2, "a2": a2, "cost": cost} 
        return parameters
		

    def get_gradients(self, z1, a1, z2, a2, x_batch, y_batch):

        m = y_batch.shape[0]
        assert(a2.shape == y_batch.shape)

        dz2 = a2 - y_batch
        dw2 = np.dot(a1.T, dz2)
        db2 = np.sum(dz2, axis = 0, keepdims = True)
        dw2 /= m
        db2 /= m
        assert (self.w2.shape == dw2.shape) 
        assert (self.b2.shape == db2.shape)

        relu = ReLU()
        grad_relu = relu.backward(z1)

        temp = np.dot(dz2, self.w2.T)
        dz1 = np.multiply(temp, grad_relu)
        dw1 = np.dot(x_batch.T, dz1)
        db1 = np.sum(dz1, axis = 0, keepdims = True)
        dw1 /= m
        db1 /= m
        assert(dw1.shape == self.w1.shape)
        assert (self.b1.shape == db1.shape)

        return dw1, db1, dw2, db2

    def plot(self, x_axis, train_accuracy, test_accuracy, x_label, xstep = 1.0, file = None):
        
        plt.xticks(np.arange(min(x_axis), max(x_axis)+1, xstep))

        plt.plot(x_axis, train_accuracy, '-r', label='Training')
        plt.plot(x_axis, test_accuracy, '-b', label='Test')
        
        plt.ylabel('Accuracy')
        plt.xlabel(x_label)
        plt.legend(loc='best')
        if file is not None:
            plt.savefig('figures/%s' % file, bbox_inches='tight')
        plt.gcf().clear()


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
 	
    train_x = (train_x - np.mean(train_x, axis = 0))/ np.std(train_x, axis = 0)
    test_x = (test_x - np.mean(test_x, axis = 0)) / np.std(test_x, axis = 0)
   
    # INSERT YOUR CODE HERE
    # YOU CAN CHANGE num_epochs AND num_batches TO YOUR DESIRED VALUES
    # num_epochs = [50, 60, 70, 80, 90, 100, 110]
    # num_batches = [400, 500, 1000, 2000]  
    hidden_units = [50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150]
    
    num_epochs = [50]
    num_batches = [1000]  
    #hidden_units = [50]
    
    training_acc = []
    test_acc = []
    x_axis = [] 
    
    for ne in range(len(num_epochs)):
        for hu in range(len(hidden_units)):
            for nb in range(len(num_batches)):

                mlp = MLP(input_dims, hidden_units[hu])

                # training_acc = []
                # test_acc = []
                # x_axis = []

                for epoch in range(num_epochs[ne]):
              
                #INSERT YOUR CODE FOR EACH EPOCH HERE
                            
                    for b in range(num_batches[nb]):
                        total_loss = 0.0
                        # INSERT YOUR CODE FOR EACH MINI_BATCH HERE
                        total_loss += mlp.train(train_x[b * 10: (b+1) * 10, :], train_y[b * 10: (b+1) * 10, :], 1e-2, 0.8, 0.001)
                        #MAKE SURE TO UPDATE total_loss

                        print(
                            '\r[Epoch {}, mb {}]    Avg.Loss = {:.3f}'.format(
                                epoch + 1,
                                b + 1,
                                total_loss,
                            ),
                            end='',
                        )
                        sys.stdout.flush()
                    
                    #train_loss, train_accuracy = mlp.evaluate(train_x, train_y)
                    #test_loss, test_accuracy = mlp.evaluate(test_x, test_y)
            
                    #training_acc.append(train_accuracy)
                    #test_acc.append(test_accuracy)
                    #x_axis.append(epoch+1)
                    
                    #INSERT YOUR CODE AFTER ALL MINI_BATCHES HERE
                    #MAKE SURE TO COMPUTE train_loss, train_accuracy, test_loss, test_accuracy
                    
                    # print()
                    # print('    Train Loss: {:.3f}    Train Acc.: {:.2f}%'.format(
                    #     train_loss,
                    #     100. * train_accuracy,
                    # ))
                    # print('    Test Loss:  {:.3f}    Test Acc.:  {:.2f}%'.format(
                    #     test_loss,
                    #     100. * test_accuracy,
                    # ))

            train_loss, train_accuracy = mlp.evaluate(train_x, train_y)
            test_loss, test_accuracy = mlp.evaluate(test_x, test_y)
            training_acc.append(train_accuracy)
            test_acc.append(test_accuracy)
            x_axis.append(hidden_units[hu])                        
                
                #mlp.plot(x_axis, training_acc, test_acc, 'epoch', 4.0, 'figure1a.png')
    print(x_axis)
    print(training_acc)
    print(test_acc)
    mlp.plot(x_axis, training_acc, test_acc, 'epoch', 10.0, 'figure1c.png')