import numpy as np

import os
import torch
from tqdm import tqdm #status bar

from sklearn.datasets import make_moons

import matplotlib.pyplot as plt

RESULTS_PATH = "./results"

# utility functions
def standard_normalization(x):
    return (x - x.mean()) / x.std()

def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w

def normaldist_init(input_size, output_size):
    # Normal distribution initialization
    return torch.randn(1, output_size, input_size)/(input_size + 1)
    
def xavier_init(input_size, output_size):
    # Xavier initialization
    return torch.randn(input_size, output_size) * np.sqrt(2.0/(input_size + output_size))


#define single layers
class Linear:
    def __init__(self, in_channels, out_channels, init="normal"):
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        if init == "normal":
            # initialize weights with standard normal distribution and bias with zeros
            self.weight = normaldist_init(self.in_channels, self.out_channels)
        elif init == "xavier":
            # Xavier initialization
            self.weight = xavier_init(self.in_channels, self.out_channels)  
        else:
            raise ValueError("Unknown initialization type")
        
        self.bias = torch.zeros(1, out_channels)
        
        # store last input for backpropagation
        self.last_input = None
        self.grad_weight = None
        self.grad_bias = None
        
    def forward(self, x, remember=False):
        if remember:
            self.last_input = x

        # reshape input to 2D tensor
        if len(x.shape) > 2:
            x = x.view(x.size(0), -1)

        # calculate linear transformation
        x = torch.matmul(x, self.weight) + self.bias
        return x
    
    def backward(self, gradient):
        # calculate gradients
        self.grad_weight = torch.matmul(self.last_input.t(), gradient)
        self.grad_bias = torch.sum(gradient, dim=0)
        
        # reshape gradient to original input shape
        if len(self.last_input.shape) > 2:
            gradient = gradient.view(*self.last_input.shape)

        # calculate gradient for previous layer
        # return gradient to next layer
        gradient = torch.matmul(gradient, self.weight.t())
        return gradient

    def update(self, learning_rate):
        # gradient descent (update weights + biases)
        self.weight = self.weight - learning_rate * self.grad_weight
        self.bias = self.bias - learning_rate * self.grad_bias
        
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
        gradient = torch.where(self.last_input > 0, gradient, 0.0)
        return gradient
    
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


#training function

def train_network(mlp, Ntrain, Ntest, Xtrain, ytrain, Xtest, ytest, num_epochs=10, batch_size=32, learning_rate=5e-3):
    num_batches_train = int(np.ceil(Ntrain/batch_size))
    num_batches_test = int(np.ceil(Ntest/batch_size))

    #train network
    losses_train = []
    losses_test = []

    for epoch in range(num_epochs):
        #reshuffle training data
        ind = np.random.permutation(len(Xtrain))
        Xtrain = Xtrain[ind]
        ytrain = ytrain[ind]

        epoch_train_loss = 0.0
        epoch_test_loss = 0.0

        #training pass
        for it in tqdm(range(num_batches_train)):
            start = it*batch_size
            end = min((it+1)*batch_size, len(Xtrain))
            X = torch.FloatTensor(Xtrain[start:end])
            y = torch.LongTensor(ytrain[start:end])

            # compute loss and update weights
            loss = mlp.training_step(X, y, learning_rate)
            epoch_train_loss += loss.item()

            # update weights
            mlp.update(learning_rate)

            if it%10==0:
                print(f"Epoch {epoch+1}/{num_epochs}, Iteration {it+1}/{num_batches_train}, Train Loss: {loss.item()}")

        #testing pass
        for it in range(num_batches_test):
            start = it*batch_size
            end = min((it+1)*batch_size, len(Xtest))
            X = torch.FloatTensor(Xtest[start:end])
            y = torch.LongTensor(ytest[start:end])

            # compute loss
            probabilities = mlp.forward(X)
            loss = mlp.criterion.forward(probabilities, y)
            epoch_test_loss += loss.item()

            if it%10==0:
                print(f"Epoch {epoch+1}/{num_epochs}, Iteration {it+1}/{num_batches_test}, Test Loss: {loss.item()}")

        # append average loss for this epoch
        losses_train.append(epoch_train_loss / num_batches_train)
        losses_test.append(epoch_test_loss / num_batches_test)

    return losses_train, losses_test

def task1(batch_size=32, num_epochs=10, learning_rate=3e-2, hidden_channels=[30,30]):

    # generate data
    Ntrain = 8000
    Ntest = 2000
    Xtrain, ytrain = make_moons(n_samples=Ntrain, noise=0.08, random_state=42)
    Xtest, ytest = make_moons(n_samples=Ntest, noise=0.08, random_state=42)

    # rescale data to [-1,1]
    amin = np.amin(Xtrain, axis=0, keepdims=True)
    amax = np.amax(Xtrain, axis=0, keepdims=True)

    Xtrain = ((Xtrain-amin)/(amax-amin)-0.5)/0.5
    Xtest = ((Xtest-amin)/(amax-amin)-0.5)/0.5

    mlp = MLP(2, hidden_channels, 2)

    losses_train = []
    losses_test = []

    # train network
    losses_train, losses_test = train_network(mlp, Ntrain, Ntest, Xtrain, ytrain, Xtest, ytest, num_epochs=num_epochs, batch_size=batch_size, learning_rate=learning_rate)

    # plot loss curves
    plt.figure()
    plt.plot(moving_average(losses_train, 10), label="Train Loss")
    plt.plot(moving_average(losses_test, 10), label="Test Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(RESULTS_PATH, "loss_curve.png"))
    plt.show()

    print("Task completed")

# STARTING POINT
if __name__ == "__main__":
    # create results folder if it does not exist
    if not os.path.exists(RESULTS_PATH):
        os.makedirs(RESULTS_PATH)

    batch_size = 32
    num_epochs = 10
    learning_rate = 3e-2

    # create MLP
    hidden_channels = [30,30]

    task1(batch_size=batch_size, num_epochs=num_epochs, learning_rate=learning_rate, hidden_channels=hidden_channels)