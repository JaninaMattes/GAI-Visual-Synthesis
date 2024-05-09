import numpy as np

import os
import torch
from tqdm import tqdm #status bar

from sklearn.datasets import make_moons

import matplotlib.pyplot as plt

RESULTS_PATH = "./results"

#define single layers
class Linear:
    def __init__(self, in_channels, out_channels):
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # initialize weights with standard normal distribution and bias with zeros
        self.weight = torch.randn(self.in_channels, self.out_channels)
        self.bias = torch.zeros(self.out_channels)
        
        self.last_input = None
        self.grad_weight = None
        self.grad_bias = None
        
    def forward(self, x, remember=False):
        if remember:
            self.last_input = x
        newx = torch.matmul(x, self.weight) + self.bias
        return newx
    
    def backward(self, gradient):
        # calculate gradients
        self.grad_weight = torch.matmul(self.last_input.t(), gradient)
        self.grad_bias = torch.sum(gradient, dim=0)

        # calculate gradient for previous layer
        newgrad = torch.matmul(gradient, self.weight.t())
        
        return newgrad
    
    def update(self, learning_rate):
        # update weights and bias
        self.weight -= learning_rate * self.grad_weight
        self.bias -= learning_rate * self.grad_bias
        
        
class ReLU:
    def __init__(self):
        self.last_input = None
    
    def forward(self, x, remember=False):
        if remember:
            self.last_input = x
        # ReLU activation
        newx = torch.max(x, torch.zeros_like(x))
        return newx
    
    def backward(self, gradient):
        # ReLU gradient
        newgrad = torch.where(self.last_input>0, gradient, 0.0)
        return newgrad
    
    def update(self, learning_rate):
        #we don't have any parameters here
        pass
    
############################################# no need to change anything below this line #############################################    
class Softmax:
    def __init__(self, dim=-1):
        self.last_output = None
        self.dim = dim
        
    def forward(self, x, remember=False):
        x = torch.exp(x-torch.amax(x, dim=-1, keepdims=True)) #numerical stable version -> normalize by max(x)
        x = x/(torch.sum(x, dim=self.dim, keepdim=True)+1e-12)
        if remember:
            self.last_output = x
        return x
    
    def backward(self, gradient):
        jacobian = -self.last_output[:,:,None]*self.last_output[:,None,:] #BxLxL
        #correct diagonal entries
        jacobian += torch.eye(self.last_output.size(-1)).unsqueeze(0)*self.last_output.unsqueeze(-1).repeat(1,1,self.last_output.size(-1))
        return torch.einsum("bj,bji->bi", gradient, jacobian)
    
    def update(self, learning_rate):
        #we don't have any parameters here
        pass
    
class CrossEntropyLoss:
    def __init__(self, dim=-1):
        self.last_input = None
        self.last_ground_truth = None
        self.dim = dim
    
    def forward(self, p, y):
        #convert y to one hot
        one_hot = torch.eye(p.size(-1))[y]
        self.last_input = p
        self.last_ground_truth = one_hot
        
        losses = -torch.sum(one_hot*torch.log(p), dim=-1)
        total_loss = torch.mean(losses)
        
        return total_loss
    
    def backward(self):
        return torch.where(self.last_ground_truth==1,-1.0/self.last_input, 0.0)
          
              
class MLP:
    def __init__(self, in_channels=2, hidden_channels=[], out_channels=2):
        self.in_channels = in_channels
        
        self.layers = []
        if len(hidden_channels)==0:
            self.layers.append(Linear(in_channels, out_channels))
        else:
            self.layers.append(Linear(in_channels, hidden_channels[0]))
            self.layers.append(ReLU())
            for i in range(len(hidden_channels)-1):
                self.layers.append(Linear(hidden_channels[i], hidden_channels[i+1]))
                self.layers.append(ReLU())
            self.layers.append(Linear(hidden_channels[-1], out_channels))
        self.layers.append(Softmax(dim=-1))
        
        self.criterion = CrossEntropyLoss(dim=-1)
        
    def forward(self, x, remember=False):
        for layer in self.layers:
            x = layer.forward(x, remember=remember)
        return x
    
    def backward(self): #calculate gradients
        grad = self.criterion.backward()
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
    
    def update(self, learning_rate): #update each layer via gradient descent
        for layer in self.layers:
            layer.update(learning_rate)
    
    def training_step(self, x, y, learning_rate):
        probabilities = self.forward(x, remember=True) #store inputs for backward pass!
        loss = self.criterion.forward(probabilities, y)
        self.backward() #calculate gradients
        self.update(learning_rate) #update using gradient descent
        
        return loss
            
############################################# no need to change anything above this line #############################################  


#training

#create datasets
Ntrain = 8000
Ntest = 2000
Xtrain, ytrain = make_moons(n_samples=Ntrain, noise=0.08, random_state=42)
Xtest, ytest = make_moons(n_samples=Ntest, noise=0.08, random_state=42)

#rescale data to [-1,1]
amin = np.amin(Xtrain, axis=0, keepdims=True)
amax = np.amax(Xtrain, axis=0, keepdims=True)

Xtrain = ((Xtrain-amin)/(amax-amin)-0.5)/0.5
Xtest = ((Xtest-amin)/(amax-amin)-0.5)/0.5


batch_size = 32
num_batches_train = int(np.ceil(Ntrain/batch_size))
num_batches_test = int(np.ceil(Ntest/batch_size))

num_epochs = 10

mlp = MLP(2, [30,30], 2)
learning_rate = 5e-2


#train network
losses_train = []
losses_test = []

for epoch in range(num_epochs):
    #reshuffle training data
    ind = np.random.permutation(len(Xtrain))
    Xtrain = Xtrain[ind]
    ytrain = ytrain[ind]
    #training pass
    for it in tqdm(range(num_batches_train)):
        start = it*batch_size
        end = min((it+1)*batch_size, len(Xtrain))
        X = torch.FloatTensor(Xtrain[start:end])
        y = torch.LongTensor(ytrain[start:end])
        
        # compute loss and update weights
        loss = mlp.training_step(X, y, learning_rate)
        losses_train.append(loss.item())

        # update weights
        mlp.update(learning_rate)

    
    #testing pass
    for it in range(num_batches_test):
        start = it*batch_size
        end = min((it+1)*batch_size, len(Xtest))
        X = torch.FloatTensor(Xtest[start:end])
        y = torch.LongTensor(ytest[start:end])
        
        # compute loss
        probabilities = mlp.forward(X)
        loss = mlp.criterion.forward(probabilities, y)

        losses_test.append(loss.item())

#plot losses
plt.figure()
plt.plot(losses_train, label="train")
plt.plot(losses_test, label="test")
plt.legend()
plt.show()
# store plot in results folder
# check if folder exists, if not create it
if not os.path.exists(RESULTS_PATH):
    os.makedirs(RESULTS_PATH)

plt.savefig(f"{RESULTS_PATH}/losses.png")


