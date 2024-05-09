from torchvision import datasets
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

# The first thing to do is specify our model. The multilayer perceptron does a matrix
# multiplication of its internal weights with the inputs and adds a bias in
# each layer. After that it activates the resulting vector.
# This can be done using the `Linear` layer. 
# Our model itself will be implemented in an object-oriented manner. You can
# find a skeleton implementation below. Please fill in the blanks marked with
# `# Your code here`. As we want to use mnist vectors as input data, make sure
# to pass the correct dimensions to the `Linear` module. 

# As our multilayer perceptron is a pytorch module, it has to inherit from the base
# `nn.Module`.  All pytorch modules expect at least a `forward` method, which
# defines what happens, when you call the instance of such a module on some
# data.


class MultilayerPerceptron(nn.Module):
    
    def __init__(self, size_hidden=100, size_out=10):
        super().__init__()
        
        self.fc1 = # Your Code here
        self.fc2 = # Your Code here
        self.fc3 = # Your Code here
        self.fc4 = # Your Code here
        self.out_layer = # Your Code here
        
        self.relu =# Your Code here
        
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)

        # Your Code here: The rest of the layers

        return out


# Pytorch modules keep track of all model parameters internally. Those will be
# e.g. the matrix and bias of the `Linear` operation we just implemented.

# To be able to feed the mnist vectors to out MultilayerPerceptron we first have to
# convert them to `torch.Tensor`s. To not have to do this everytime we want to
# do an operation on those vectors you can find a `torch.Dataset` version of
# the mnist vectors below. All it does is a simple casting operation.


class MnistVectors(torch.utils.data.Dataset):
    '''A Pytorch Dataset, which does the same data preparation as was done in
    the PCA exercise.'''

    def __init__(self, split='train'):
        super().__init__()
        
        mnist = datasets.MNIST('../data',
                               train=split=='train',
                               download=True)

        ########################
        #### Your Code here ####
        # self.mnist_vectors, self.labels = your conversion function from task 2
        ########################

            
    def __getitem__(self, idx):
        '''Implements the ``[idx]`` method. Here we convert the numpy data to
        torch tensors.
        '''

        mvec = torch.tensor(self.mnist_vectors[idx]).float()
        label = torch.tensor(self.labels[idx]).long()

        return mvec, label
    
    def __len__(self):
        return len(self.labels)


# The following two functions are needed to track the progress of the training.
# One transforms the output of the MultilayerPerceptron into a scalar class label, the 
# other uses that label to calculate the batch accuracy.

def batch_accuracy(prediction, label):
    return # Your Code here

def class_label(prediction):
    return # Your code here

def train(use_gpu=False): # if torch.cuda.is_available(), use gpu to speed up training
    
    # Here we instantiate our model. The weights of the model are automatically
    # initialized by pytorch
    P = MultilayerPerceptron()
    
    TrainData = MnistVectors()
    TestData = MnistVectors('test')
    # Dataloaders allow us to load the data in batches. This allows us a better
    # estimate of the parameter updates when doing backprop.
    # We need two Dataloaders so that we can train on the train data split
    # and evaluate on the test datasplit.
    Dl = DataLoader(TrainData, batch_size=16, shuffle=True)
    testDl = DataLoader(TestData, batch_size=16, shuffle=False)
    
    # Use the Adam optimizer with learning rate 1e-4 and otherwise default
    # values
    optimizer =  # Your code here

    # Use the Cross Entropy loss from pytorch. Make sure your MultilayerPerceptron does
    # not use any activation function on the output layer! (Do you know why?)
    criterion =  # Your code here
    
    if use_gpu:
        P.cuda()
        criterion.cuda()
    
    for epoch in tqdm(range(5), desc='Epoch'):
        for step, [example, label] in enumerate(tqdm(Dl, desc='Batch')):
            if use_gpu:
                example = example.cuda()
                label = label.cuda()
            
            # The optimizer knows about all model parameters. These in turn
            # store their own gradients. When calling loss.backward() the newly
            # computed gradients are added on top of the existing ones. Thus
            # at before calculating new gradients we need to clear the old ones
            # using ther zero_grad() method.
            optimizer.zero_grad()
            
            prediction = P(example)
            
            loss = criterion(prediction, label)
            
            # Here pytorch applies backpropagation for us completely
            # automatically!!! That is quite awesome!
            loss.backward()

            # The step method now adds the gradients onto the model parameters
            # as specified by the optimizer and the learning rate.
            optimizer.step()
            
            # To keep track of what is happening print some outputs from time
            # to time.
            if (step % 375) == 0:
                # Your code here
                acc = batch_accuracy(class_label(prediction), label)
                tqdm.write('Batch Accuracy: {}%, Loss: {}'.format(acc, loss))
                
        # Now validate on the whole test set
        accuracies = []
        for idx, [test_ex, test_l] in enumerate(tqdm(testDl, desc='Test')):
            if use_gpu:
                test_ex = test_ex.cuda()
                test_l = test_l.cuda()

            #########################
            #### Your Code here  ####
            #########################

            # Using your batch_accuracy function, also print the mean accuracy
            # over the whole test split of the data.

        print('Validation Accuracy: {}%'.format(np.mean(accuracies)))

        # Now let's write out a checkpoint of the model, so that we can
        # reuse it:
        torch.save(model.state_dict(), 'perceptron_{}.ckpt'.format(step))

        # If you need to load the checkpoint instanciate your model and the
        # load the state dict from a checkpoint:
        # P = MultilayerPerceptron()
        # P.load_state_dict(torch.load(perceptron_3750.ckpt))
        # Make sure to use the latest checkpoint by entering the right number.

        ######################################
        ###### Add code for task 4 here ######
        ######################################

        


if __name__ == '__main__':
    train(use_gpu=True if torch.cuda.is_available() else False)
