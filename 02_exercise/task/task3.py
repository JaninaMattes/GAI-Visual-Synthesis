import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt

# load dataset
data = load_digits()
x, y = (data.images / 16.0).reshape(-1, 8 * 8), data.target
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.25, shuffle=True, random_state=0)

def task3(x, lr=0.01, num_iterations=1000):

    # initialize parameters
    weights = np.ones((1,10,x.shape[1])) #np.random.normal(0,1,size=(1,10,x.shape[1]))
    bias = np.zeros((1,10))

    # Iterate through number of iterations
    for i in tqdm(range(num_iterations), desc="Training"):

        # Calculate logits and probabilities
        logits = weights @ xtrain.T + bias
        probs = stable_softmax(logits)

        # Calculate the gradient with respect to weights and bias
        grad_weights = xtrain @ (probs - ytrain).T
        grad_bias = np.sum(probs - ytrain, axis=1, keepdims=True)

        # Update the parameters using gradient descent
        weights -= lr * grad_weights
        bias -= lr * grad_bias

    # Evaluate the model on the test set
    test_logits = weights @ xtest.T + bias
    test_probs = stable_softmax(test_logits)
    test_predictions = np.argmax(test_probs, axis=1)
    test_accuracy = np.mean(test_predictions == ytest)

    print("Test accuracy:", test_accuracy)

def stable_softmax(logits):
    """ Compute the softmax of vector x in a numerically stable way.
        Prevent overflow and underflow by subtracting the maximum value from the logits.
    """
    logits_max = np.max(logits, axis=1, keepdims=True)
    exp_logits = np.exp(logits - logits_max)
    sum_exp_logits = np.sum(exp_logits, axis=1, keepdims=True)
    return exp_logits / sum_exp_logits

if __name__ == "__main__":
    task3(xtrain, lr=0.01, num_iterations=1000)
    print("Task completed")
