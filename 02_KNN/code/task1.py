from typing import Counter
import numpy as np
import os
import matplotlib.pyplot as plt

from sklearn import neighbors
from sklearn.datasets import make_moons
from sklearn.metrics import classification_report
from scipy.stats import mode

from utils import euclidean_distance, calc_accuracy, calc_rss, generate_colors, scatter_2D, scatter_2D_linspace


class KNN(object):
    """ Does not require any learning as the model stores the entire dataset
        and classifies data points based on the points that are similar to it.
        It makes predictions based on the training data only.
    """
    def __init__(self, n_neighbors):
        self.n_neighbors = n_neighbors

    def fit(self, x, y):
        self.x = x
        self.y = y

    def kneighbors(self, xquery):
        """ Returns the indices and distances of the k nearest neighbor points 
            based on distances to them. Uses the euclidean distance as distance metric.
        """
        point_dist = []
        for row_j in self.x:
            # calculate distances
            point_dist.append(euclidean_distance(row_j, xquery))
        point_dist = np.array(point_dist)
        # get k neighbors by their indices
        nearest_neighbor_ids = np.argsort(point_dist)[:self.n_neighbors]
        return nearest_neighbor_ids

    def predict(self, xquery):
        """ Returns predicted label for a given query point.
        """
        predictions = []
        for i in range(len(xquery)):
            indices = self.kneighbors(xquery[i])
            pred_labels = self.y[indices.astype(int)]
            # calc mode to get most occuring values
            pred_mode_label = int(mode(pred_labels)[0])
            # np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=0, arr=pred_labels)
            predictions.append(pred_mode_label)
        return predictions

    def predict_proba(self, xquery):
        """Returns class probabilities for a given query point."""
        xquery = np.atleast_2d(xquery)

        # Get indices of k-nearest neighbors for all query points
        indices = self.kneighbors(xquery)

        # Get labels of k-nearest neighbors
        neighbor_labels = self.y[indices]

        # Calculate class frequencies for each query point
        class_freqs = [Counter(neighbor_label) for neighbor_label in neighbor_labels.reshape(-1, self.n_neighbors)]

        # Convert frequencies to probabilities
        class_probas = np.array([[freq / self.n_neighbors for freq in freqs.values()] for freqs in class_freqs])

        return class_probas
        
def create_data(N=1000, noise=0.2, random_state=0):
    N_train = int(N*0.9) #use 90% for training
    N_test = N - N_train #rest for testing

    # 2D dataset consisting of points x and binary labels y
    x, y = make_moons(n_samples=N, noise=noise,random_state=random_state)
    
    #split into train and test set
    xtrain, ytrain = x[:N_train,...], y[:N_train,...]
    xtest, ytest = x[N_train:,...], y[N_train:,...]

    return xtrain, ytrain, xtest, ytest
       
def task1():
    # get data
    N = 1000
    N_train = int(N*0.9) #use 90% for training
    N_test = N - N_train #rest for testing

    # 2D dataset consisting of points x and binary labels y
    x, y = make_moons(n_samples=N, noise=0.2,random_state=0)
    
    #split into train and test set
    xtrain, ytrain = x[:N_train,...], y[:N_train,...]
    xtest, ytest = x[N_train:,...], y[N_train:,...]

    print("Task 2.1: Visualize data via scatterplot")
    # randomly generate colour dict
    labels = np.unique(ytrain)
    colors = generate_colors(len(labels))
    cdict_ = dict(zip(labels, colors))

    scatter_2D(xtrain[0:,0], xtrain[0:,1], ytrain, cdict=cdict_, title='Task 2: KNN Train Dataset Scatterplot')
    scatter_2D(xtest[0:,0], xtest[0:,1], ytest, cdict=cdict_, title='Task 2: KNN Test Dataset Scatterplot')

    print("Task 2.4 For k=5 fit your KNN implementation and that from sklearn to the training data.")
    
    k = 5
    sknn = neighbors.KNeighborsClassifier(n_neighbors=k)
    knn = KNN(n_neighbors=k)

    sknn.fit(xtrain, ytrain)
    knn.fit(xtrain, ytrain)

    skit_ypred = sknn.predict(xtest)
    print(f"1. SKNN Predicted y values {skit_ypred}")

    knn_ypred = knn.predict(xtest)
    print(knn_ypred)
    print(f"2. KNN Predicted y values {skit_ypred}")

    scatter_2D(xtest[0:,0], xtest[0:,1], knn_ypred, cdict=cdict_, title='Task 2: KNN Classifier Predicted Y Labels Scatterplot')
    rss = calc_rss(ytest, knn_ypred)
    acc = calc_accuracy(ytest, knn_ypred)
    print("The RSS error for KNN is: " + str(rss))
    print("The accuracy score for KNN is: " + str(acc))

    print(classification_report(ytest, knn_ypred, labels=np.unique(ytest)))
    
    # analyze different values of k
    ks = [2**i for i in range(10)]
    accuracies = []

    for k in ks:
        print("KNN with k: " + str(k))
        # fit and evaluate accuracy on test data
        knn = KNN(n_neighbors=k)
        knn.fit(xtrain, ytrain)
        knn_ypred = knn.predict(xtest)

        knn_proba = knn.predict_proba(xtest)
        print(f"KNN Predicted y values {knn_ypred}")

        rss = calc_rss(ytest, knn_ypred)
        acc = calc_accuracy(ytest, knn_ypred)
        print("The minimum RSS error for KNN is: " + str(rss))
        print("The minimum accuracy score for KNN is: " + str(acc))

        accuracies.append((k, acc))
        # plot decision boundary
        N = 100
        linx = np.linspace(-1.5, 2.5, N)
        liny = np.linspace(-1.0, 1.5, N)
        print("Plot decision boundary")
        scatter_2D_linspace(xtest[0:,0], xtest[0:,1], knn_ypred, cdict=cdict_, title=f'Task 2 KNN with k=[{k}] Predicted Y Labels Scatterplot', line_x=linx, line_y=liny)

        print("Task 2.5: Evaluate and plot accuracy of KNN classifier.")
        print(accuracies)

    fig = plt.figure(figsize=(5, 3))
    acc_x = [acc[0] for acc in accuracies] # k-value
    acc_y = [acc[1] for acc in accuracies] # accuracy
        
    plt.title('Accuracy per k-value')
    plt.xlabel('k-value')
    plt.ylabel('accuracy')
    plt.plot(acc_x, acc_y)
    plt.show()

    img_path = './result/Accuracy_per_kVal_KNNClassifier'
    # if path does not exist, create it
    if not os.path.exists('./result/'):
        os.makedirs('./result/')

    fig.savefig(img_path, dpi=250)

    print("DONE")
        
if __name__ == "__main__":
    task1()
    print("Task completed")

