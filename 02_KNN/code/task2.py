import numpy as np

from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# utility functions
from utils import generate_colors, calc_accuracy, calc_rss, scatter_2D, scatter_2D_linspace

def make_data(noise=0.2, outlier=1):
    prng = np.random.RandomState(0)
    n = 500

    x0 = np.array([0, 0])[None, :] + noise * prng.randn(n, 2)
    y0 = np.ones(n)
    x1 = np.array([1, 1])[None, :] + noise * prng.randn(n, 2)
    y1 = -1 * np.ones(n)

    x = np.concatenate([x0, x1])
    y = np.concatenate([y0, y1]).astype(np.int32)

    xtrain, xtest, ytrain, ytest = train_test_split(
        x, y, test_size=0.1, shuffle=True, random_state=0
    )
    xplot, yplot = xtrain, ytrain

    outlier = outlier * np.array([1, 1.75])[None, :]
    youtlier = np.array([-1])
    xtrain = np.concatenate([xtrain, outlier])
    ytrain = np.concatenate([ytrain, youtlier])
    return xtrain, xtest, ytrain, ytest, xplot, yplot
    
    
class LinearLeastSquares(object):
    """ Implementation of Linear Regression using Least Squares.
        Is a linear approach to modelling the relationship between a dependent
        variable and one or more independent variables.
    """
    def fit(self, x, y):
        """ Normal equation to find a minimizer 
            for the least square objective.
            >>> x = independent variable
            >>> y = dependent variable
        """
        # vectors that shall be combined
        # prepend ones to the input to use the bias trick
        m = len(x)
        X_w = np.array([np.ones(m), x[:, 0], x[:, 1]]).T

        # matrix inversion and calculate solution
        self.w = np.linalg.inv(X_w.T.dot(X_w)).dot(X_w.T.dot(y))
 
        return self.w
    
    def predict(self, xquery):
        # Concatenate numpy array of ones to predicted y_pred values
        # apply dot product with w to predict y values
        m = len(xquery)
        X_pred = np.array([np.ones(m), xquery[:, 0], xquery[:, 1]]).T
        
        y_pred = X_pred.dot(self.w)
        y_pred = np.rint(y_pred).astype('int8')
        
        return y_pred
        
        
        
def task2():
    """The training data includes an outlier,
       the parameter outlier controlls its magnitude.
    """
    # get data
    for outlier in [1, 2, 4, 8, 16]:
        # get data. xplot, yplot is same as xtrain, ytrain but without outlier
        xtrain, xtest, ytrain, ytest, xplot, yplot = make_data(outlier=outlier)

        # randomly generate colour dict
        labels = np.unique(ytrain)
        colors = generate_colors(len(labels))
        cdict_ = dict(zip(labels, colors))

        title = f'Task 1 Scatterplot for Dataset with Outlier scale={outlier}'
        print("Task 1: Visualize training data.")
        # visualize xtrain via scatterplot
        scatter_2D(xtrain[0:,0], xtrain[0:,1], ytrain, cdict=cdict_, title=title)
    
        lls = LinearLeastSquares()
        beta = lls.fit(xtrain, ytrain)

        # evaluate accuracy and decision boundary of LLS
        ypred = lls.predict(xtest)
        # randomly generate colour dict
        labels = np.unique(ypred)
        colors = generate_colors(len(labels))
        cdict_ = dict(zip(labels, colors))

        rss = calc_rss(ytest, ypred)
        acc = calc_accuracy(ytest, ypred)
        print("The RSS error is: " + str(rss))
        print("The accuracy score is: " + str(acc))

        print("Task 1: Visualize predicted y values.")

        # Plot the resulting regression line
        N = 100
        linx = np.linspace(-1.5, 2.5, N)
        liny = -beta[0] / beta[2] - (beta[1] / beta[2]) * linx
        title = f'Task 1 Scatterplot for Prediction LLS with Outlier [{outlier}]'
        scatter_2D_linspace(xtest[0:,0], xtest[0:,1], ypred, cdict=cdict_, title=title, line_x=linx, line_y=liny)

        print("DONE")


if __name__ == "__main__":
    task2()
    print("Task completed")
